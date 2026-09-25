"""Toy benchmark: PPO vs naive replay + PPO vs MPOPI + PPO on a point mass.

Runs every arm for several seeds on the pure-torch ``PointMassVecEnv`` (CPU,
no MuJoCo) through the real ``MjlabOnPolicyRunner`` / mode switch, evaluates the
deterministic policy after every update, and reports sample-efficiency (area
under the eval curve), final return and Welch t-tests between arms.

Example::

  uv run --extra cpu python scripts/benchmarks/mpopi_toy_benchmark.py \\
    --seeds 10 --iterations 60 --out-dir /tmp/mpopi_toy
"""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import torch
import tyro
from scipy import stats

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.rl.mpopi import MpopiCfg
from mjlab.rl.mpopi.toy_env import PointMassVecEnv
from mjlab.rl.runner import MjlabOnPolicyRunner

ARMS: dict[str, MpopiCfg] = {
  "A_ppo": MpopiCfg(mode="ppo"),
  "B_naive_replay": MpopiCfg(mode="naive_replay_ppo"),
  "C_mpopi": MpopiCfg(mode="mpopi_ppo"),
  "C_mpopi_noclip": MpopiCfg(mode="mpopi_ppo", importance_weight_clip_max=None),
}

LOGGED_KEYS = (
  "kl",
  "clip_fraction",
  "mpopi/accepted",
  "mpopi/ess",
  "mpopi/weight_mean",
  "mpopi/weight_std",
  "mpopi/weight_max",
  "mpopi/raw_ratio_max",
  "mpopi/clipped_frac",
  "mpopi/behavior_kl",
  "mpopi/policy_age_mean",
)


@dataclass
class BenchmarkCfg:
  seeds: int = 10
  seed_offset: int = 0
  """First seed; use fresh seeds for confirmation runs."""
  iterations: int = 60
  num_envs: int = 32
  num_steps_per_env: int = 16
  episode_length: int = 50
  eval_episodes: int = 16
  replay_buffer_size: int = 4
  replay_ratio: float = 1.0
  arms: tuple[str, ...] = tuple(ARMS)
  out_dir: Path = Path("logs/mpopi_toy")
  ppo: RslRlPpoAlgorithmCfg = field(
    default_factory=lambda: RslRlPpoAlgorithmCfg(
      num_learning_epochs=5, num_mini_batches=4, learning_rate=1e-3
    )
  )


def evaluate(runner: MjlabOnPolicyRunner, cfg: BenchmarkCfg) -> float:
  """Mean undiscounted return of the deterministic policy from fixed starts."""
  env = PointMassVecEnv(num_envs=cfg.eval_episodes, device="cpu")
  env.pos = torch.linspace(-2.0, 2.0, cfg.eval_episodes).view(-1, 1)
  policy = runner.alg.get_policy()
  total = torch.zeros(cfg.eval_episodes)
  alive = torch.ones(cfg.eval_episodes, dtype=torch.bool)
  with torch.inference_mode():
    for _ in range(cfg.episode_length):
      obs = env.get_observations()
      action = policy(obs)  # Deterministic mean; consumes no RNG.
      _, reward, dones, _ = env.step(action)
      total += reward * alive
      alive &= dones == 0
  return float(total.mean())


def run_one(arm: str, seed: int, cfg: BenchmarkCfg) -> list[dict]:
  mpopi = replace(
    ARMS[arm],
    replay_buffer_size=cfg.replay_buffer_size,
    replay_ratio=cfg.replay_ratio,
  )
  agent = RslRlOnPolicyRunnerCfg(
    seed=seed,
    num_steps_per_env=cfg.num_steps_per_env,
    actor=RslRlModelCfg(
      hidden_dims=(32, 32),
      distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0},
    ),
    critic=RslRlModelCfg(hidden_dims=(32, 32)),
    algorithm=replace(cfg.ppo, mpopi=mpopi),
    logger="tensorboard",
  )
  torch.manual_seed(seed)
  env = PointMassVecEnv(
    num_envs=cfg.num_envs, max_episode_length=cfg.episode_length, seed=seed
  )
  runner = MjlabOnPolicyRunner(env, asdict(agent), log_dir=None, device="cpu")
  rows: list[dict] = []
  env_steps_per_it = cfg.num_envs * cfg.num_steps_per_env

  def on_log(**kw) -> None:
    it = kw["it"]
    loss = kw["loss_dict"]
    row = {
      "arm": arm,
      "seed": seed,
      "iteration": it,
      "env_steps": (it + 1) * env_steps_per_it,
      "eval_return": evaluate(runner, cfg),
      "gradient_samples": loss.get("mpopi/gradient_samples", env_steps_per_it),
    }
    row.update({k: loss.get(k, math.nan) for k in LOGGED_KEYS})
    rows.append(row)

  runner.logger.log = on_log  # type: ignore[method-assign]
  runner.learn(num_learning_iterations=cfg.iterations)
  return rows


def _ci95(x: list[float]) -> tuple[float, float]:
  n = len(x)
  mean = sum(x) / n
  if n < 2:
    return mean, math.nan
  sd = math.sqrt(sum((v - mean) ** 2 for v in x) / (n - 1))
  return mean, float(stats.t.ppf(0.975, n - 1)) * sd / math.sqrt(n)


def summarize(rows: list[dict], cfg: BenchmarkCfg) -> dict:
  per_run: dict[str, dict[int, list[dict]]] = {}
  for r in rows:
    per_run.setdefault(r["arm"], {}).setdefault(r["seed"], []).append(r)

  def run_stats(runs: list[dict]) -> dict:
    ret = [r["eval_return"] for r in runs]
    tail = max(1, len(ret) // 10)
    return {
      "auc": sum(ret) / len(ret),  # Mean eval return over training.
      "final": sum(ret[-tail:]) / tail,  # Mean of the last 10% of iterations.
    }

  summary: dict = {"config": {k: str(v) for k, v in asdict(cfg).items()}}
  metric_values: dict[str, dict[str, list[float]]] = {}
  for arm, seeds in per_run.items():
    s = [run_stats(v) for v in seeds.values()]
    metric_values[arm] = {m: [x[m] for x in s] for m in ("auc", "final")}
    diag = {}
    for key in LOGGED_KEYS:
      vals = [r[key] for v in seeds.values() for r in v if not math.isnan(r[key])]
      diag[key] = sum(vals) / len(vals) if vals else math.nan
    summary[arm] = {
      m: dict(zip(("mean", "ci95"), _ci95(v), strict=True))
      for m, v in metric_values[arm].items()
    } | {"diagnostics_mean": diag}

  tests = {}
  for a, b in (
    ("C_mpopi", "A_ppo"),
    ("C_mpopi", "B_naive_replay"),
    ("B_naive_replay", "A_ppo"),
    ("C_mpopi_noclip", "C_mpopi"),
  ):
    if a not in metric_values or b not in metric_values:
      continue
    for m in ("auc", "final"):
      t = stats.ttest_ind(metric_values[a][m], metric_values[b][m], equal_var=False)
      tests[f"{a} vs {b} [{m}]"] = {
        "diff": sum(metric_values[a][m]) / len(metric_values[a][m])
        - sum(metric_values[b][m]) / len(metric_values[b][m]),
        "t": float(t.statistic),  # type: ignore[attr-defined]
        "p": float(t.pvalue),  # type: ignore[attr-defined]
      }
  summary["welch_tests"] = tests
  return summary


def main(cfg: BenchmarkCfg) -> None:
  cfg.out_dir.mkdir(parents=True, exist_ok=True)
  rows: list[dict] = []
  for arm in cfg.arms:
    for seed in range(cfg.seed_offset, cfg.seed_offset + cfg.seeds):
      start = time.time()
      run_rows = run_one(arm, seed, cfg)
      rows += run_rows
      print(
        f"{arm:>16} seed {seed}: final eval {run_rows[-1]['eval_return']:8.3f}"
        f"  ({time.time() - start:.1f}s)",
        flush=True,
      )

  with open(cfg.out_dir / "curves.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
  summary = summarize(rows, cfg)
  (cfg.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

  print("\nEval return (higher is better), mean ± 95% CI over seeds")
  print(f"{'arm':>16} {'AUC':>18} {'final':>18}")
  for arm in cfg.arms:
    s = summary[arm]
    print(
      f"{arm:>16} {s['auc']['mean']:9.3f} ± {s['auc']['ci95']:6.3f}"
      f" {s['final']['mean']:9.3f} ± {s['final']['ci95']:6.3f}"
    )
  print("\nWelch t-tests")
  for name, t in summary["welch_tests"].items():
    print(f"  {name:<40} diff {t['diff']:+8.3f}  t {t['t']:+6.2f}  p {t['p']:.3g}")
  print(f"\nWrote {cfg.out_dir / 'curves.csv'} and {cfg.out_dir / 'summary.json'}")


if __name__ == "__main__":
  main(tyro.cli(BenchmarkCfg))

"""PPO with an MPOPI replay-correction stage in front of the update.

``MpopiPpo`` is selected through RSL-RL's ``algorithm.class_name`` and is built
by the unmodified ``PPO.construct_algorithm``. It keeps PPO as the only
optimizer. MPOPI only prepares data: at the start of ``update()`` it turns the
replay buffer into extra weighted samples, which are mixed with the fresh
rollout and optimized with PPO's clipped objective.

``update()`` mirrors ``rsl_rl.algorithms.PPO.update`` from rsl-rl-lib 5.5.1
because the upstream loss is inline and has no hook for per-sample weights.
When no replay sample is available it takes exactly the upstream code path, so
the resulting parameters are identical to plain PPO.
"""

from dataclasses import asdict
from typing import Any, Generator, cast

import torch
import torch.distributed as dist
import torch.nn as nn
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from mjlab.rl.mpopi.config import MpopiCfg
from mjlab.rl.mpopi.mpopi import Mpopi, MpopiBatch
from mjlab.rl.mpopi.replay_buffer import ReplayBuffer

_Pool = dict[str, Any]


class MpopiPpo(PPO):
  """PPO whose update batch is fresh data plus MPOPI-corrected replay data."""

  def __init__(
    self,
    actor: MLPModel,
    critic: MLPModel,
    storage: RolloutStorage,
    mpopi: dict[str, Any] | MpopiCfg | None = None,
    **kwargs,
  ) -> None:
    super().__init__(actor, critic, storage, **kwargs)
    if isinstance(mpopi, dict):
      mpopi = MpopiCfg(**cast(dict[str, Any], mpopi))
    cfg = mpopi if mpopi is not None else MpopiCfg(mode="mpopi_ppo")
    if actor.is_recurrent or critic.is_recurrent:
      raise ValueError("MPOPI does not support recurrent actors or critics.")
    if self.rnd is not None or self.symmetry is not None:
      raise ValueError("MPOPI does not support RND or symmetry augmentation.")
    self.mpopi_cfg = cfg
    self.mpopi = Mpopi(cfg, gamma=self.gamma, lam=self.lam)
    self.replay = ReplayBuffer(cfg.replay_buffer_size, device=self.device)
    self.policy_version = 0
    shape = (storage.num_transitions_per_env, storage.num_envs, 1)
    self._raw_rewards = torch.zeros(shape, device=self.device)
    self._time_outs = torch.zeros(shape, dtype=torch.bool, device=self.device)
    self._bootstrap_obs: TensorDict | None = None

  # Rollout hooks.

  def process_env_step(
    self,
    obs: TensorDict,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    extras: dict[str, torch.Tensor],
  ) -> None:
    # Record raw rewards and time-outs before PPO folds the stale time-out
    # bootstrap gamma * V_old(s_t) into the stored rewards.
    step = self.storage.step
    self._raw_rewards[step].copy_(rewards.view(-1, 1))
    if "time_outs" in extras:
      self._time_outs[step].copy_(extras["time_outs"].view(-1, 1))
    else:
      self._time_outs[step].zero_()
    super().process_env_step(obs, rewards, dones, extras)

  def compute_returns(self, obs: TensorDict) -> None:
    super().compute_returns(obs)
    self._bootstrap_obs = obs.clone()

  # Update.

  def update(self) -> dict[str, float]:
    st = self.storage
    num_fresh = st.num_envs * st.num_transitions_per_env
    replay, mpopi_metrics = self.mpopi.process(
      self.replay, self.actor, self.critic, self.policy_version, num_fresh
    )
    if replay is not None and not bool(replay.mask.any()):
      replay = None  # Everything rejected: fall back to plain PPO.
    pool = self._build_pool(replay)
    weighted = replay is not None

    mean_value_loss = 0.0
    mean_surrogate_loss = 0.0
    mean_entropy = 0.0
    mean_kl = 0.0
    mean_clip_fraction = 0.0

    for batch, weights, mask in self._mini_batch_generator(pool):
      if self.normalize_advantage_per_mini_batch:
        with torch.no_grad():
          batch.advantages = _normalize(batch.advantages, mask)  # type: ignore[arg-type]

      with torch.amp.autocast(  # pyright: ignore[reportPrivateImportUsage]
        device_type=torch.device(self.device).type,
        enabled=self.use_mixed_precision,
        dtype=torch.bfloat16,
      ):
        self.actor(batch.observations, stochastic_output=True)
        actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore[arg-type]
        values = self.critic(batch.observations)
        distribution_params = self.actor.output_distribution_params
        entropy = self.actor.output_entropy

        with torch.inference_mode():
          kl = self.actor.get_kl_divergence(
            batch.old_distribution_params,  # type: ignore[arg-type]
            distribution_params,
          )
          kl_mean = _mean(kl, mask)
          if self.desired_kl is not None and self.schedule == "adaptive":
            self._adapt_learning_rate(kl_mean)

        assert batch.old_actions_log_prob is not None
        assert batch.advantages is not None
        assert batch.values is not None and batch.returns is not None
        surrogate_loss, ratio = weighted_clipped_surrogate(
          actions_log_prob,
          torch.squeeze(batch.old_actions_log_prob),
          torch.squeeze(batch.advantages),
          self.clip_param,
          weights=torch.squeeze(weights, -1) if weighted else None,  # type: ignore[arg-type]
          mask=mask,
        )

        if self.use_clipped_value_loss:
          value_clipped = batch.values + (values - batch.values).clamp(
            -self.clip_param, self.clip_param
          )
          value_losses = (values - batch.returns).pow(2)
          value_losses_clipped = (value_clipped - batch.returns).pow(2)
          value_loss = _mean(torch.max(value_losses, value_losses_clipped), mask)
        else:
          value_loss = _mean((batch.returns - values).pow(2), mask)

        entropy_mean = _mean(entropy, mask)
        loss = (
          surrogate_loss
          + self.value_loss_coef * value_loss
          - self.entropy_coef * entropy_mean
        )

      self.optimizer.zero_grad()
      loss.backward()
      if self.is_multi_gpu:
        self.reduce_parameters()
      nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
      nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
      self.optimizer.step()

      with torch.no_grad():
        clipped = ((ratio - 1.0).abs() > self.clip_param).float()
        mean_clip_fraction += _mean(clipped, mask).item()
      mean_value_loss += value_loss.item()
      mean_surrogate_loss += surrogate_loss.item()
      mean_entropy += entropy_mean.item()
      mean_kl += kl_mean.item()

    # Normalizers are updated on fresh on-policy observations only (upstream).
    obs = cast(TensorDict, st.observations.flatten(0, 1))
    self.actor.update_normalization(obs)
    self.critic.update_normalization(obs)

    num_updates = self.num_learning_epochs * self.num_mini_batches
    loss_dict = {
      "value": mean_value_loss / num_updates,
      "surrogate": mean_surrogate_loss / num_updates,
      "entropy": mean_entropy / num_updates,
      "kl": mean_kl / num_updates,
      "clip_fraction": mean_clip_fraction / num_updates,
    }
    num_accepted = int(replay.mask.sum()) if replay is not None else 0
    mpopi_metrics["gradient_samples"] = float(num_fresh + num_accepted)
    loss_dict.update({f"mpopi/{k}": v for k, v in mpopi_metrics.items()})

    self._store_fresh_segment()
    self.policy_version += 1
    st.clear()
    return loss_dict

  def save(self) -> dict:
    saved = super().save()
    saved["mpopi_cfg"] = asdict(self.mpopi_cfg)
    return saved

  # Private helpers.

  def _adapt_learning_rate(self, kl_mean: torch.Tensor) -> None:
    """Upstream adaptive KL learning-rate rule (``ppo.py:246-266``)."""
    if self.is_multi_gpu:
      dist.all_reduce(kl_mean, op=dist.ReduceOp.SUM)  # ty: ignore[possibly-missing-attribute]
      kl_mean /= self.gpu_world_size
    if self.gpu_global_rank == 0:
      if kl_mean > self.desired_kl * 2.0:
        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
      elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
    if self.is_multi_gpu:
      lr_tensor = torch.tensor(self.learning_rate, device=self.device)
      dist.broadcast(lr_tensor, src=0)  # ty: ignore[possibly-missing-attribute]
      self.learning_rate = lr_tensor.item()
    for param_group in self.optimizer.param_groups:
      param_group["lr"] = self.learning_rate

  def _build_pool(self, replay: MpopiBatch | None) -> _Pool:
    """Flatten the fresh rollout and append the replay batch, if any."""
    st = self.storage
    assert st.distribution_params is not None
    fresh: _Pool = {
      "observations": st.observations.flatten(0, 1),
      "actions": st.actions.flatten(0, 1),
      "values": st.values.flatten(0, 1),
      "returns": st.returns.flatten(0, 1),
      "advantages": st.advantages.flatten(0, 1),
      "old_actions_log_prob": st.actions_log_prob.flatten(0, 1),
      "old_distribution_params": tuple(p.flatten(0, 1) for p in st.distribution_params),
      "weights": None,
      "mask": None,
    }
    if replay is None:
      return fresh

    # Upstream already normalized the fresh advantages on their own; recover
    # the raw ones so fresh and replay are normalized together.
    fresh_adv = (st.returns - st.values).flatten(0, 1)
    advantages = torch.cat([fresh_adv, replay.advantages])
    num_fresh = fresh_adv.shape[0]
    ones = torch.ones(num_fresh, 1, device=self.device)
    mask = torch.cat([ones.bool(), replay.mask])
    advantages = torch.where(mask, advantages, torch.zeros_like(advantages))
    if not self.normalize_advantage_per_mini_batch:
      advantages = _normalize(advantages, mask)
    return {
      "observations": TensorDict.cat([fresh["observations"], replay.observations]),
      "actions": torch.cat([fresh["actions"], replay.actions]),
      "values": torch.cat([fresh["values"], replay.values]),
      "returns": torch.cat([fresh["returns"], replay.returns]),
      "advantages": advantages,
      "old_actions_log_prob": torch.cat(
        [fresh["old_actions_log_prob"], replay.old_actions_log_prob]
      ),
      "old_distribution_params": tuple(
        torch.cat([f, r])
        for f, r in zip(
          fresh["old_distribution_params"],
          replay.old_distribution_params,
          strict=True,
        )
      ),
      "weights": torch.cat([ones, replay.weights]),
      "mask": mask,
    }

  def _mini_batch_generator(
    self, pool: _Pool
  ) -> Generator[
    tuple[RolloutStorage.Batch, torch.Tensor | None, torch.Tensor | None],
    None,
    None,
  ]:
    """Same shuffling as ``RolloutStorage.mini_batch_generator``, over the pool."""
    batch_size = pool["actions"].shape[0]
    mini_batch_size = batch_size // self.num_mini_batches
    indices = torch.randperm(
      self.num_mini_batches * mini_batch_size,
      requires_grad=False,
      device=self.device,
    )
    for _ in range(self.num_learning_epochs):
      for i in range(self.num_mini_batches):
        idx = indices[i * mini_batch_size : (i + 1) * mini_batch_size]
        batch = RolloutStorage.Batch(
          observations=pool["observations"][idx],
          actions=pool["actions"][idx],
          values=pool["values"][idx],
          advantages=pool["advantages"][idx],
          returns=pool["returns"][idx],
          old_actions_log_prob=pool["old_actions_log_prob"][idx],
          old_distribution_params=tuple(
            p[idx] for p in pool["old_distribution_params"]
          ),
        )
        weights = pool["weights"][idx] if pool["weights"] is not None else None
        mask = pool["mask"][idx] if pool["mask"] is not None else None
        yield batch, weights, mask

  def _store_fresh_segment(self) -> None:
    st = self.storage
    assert self._bootstrap_obs is not None, "compute_returns() must run first."
    assert st.distribution_params is not None
    self.replay.insert(
      observations=st.observations,
      actions=st.actions,
      rewards=self._raw_rewards,
      dones=st.dones,
      time_outs=self._time_outs,
      behavior_log_prob=st.actions_log_prob,
      behavior_distribution_params=st.distribution_params,
      bootstrap_observations=self._bootstrap_obs,
      policy_version=self.policy_version,
    )


def weighted_clipped_surrogate(
  log_prob: torch.Tensor,
  old_log_prob: torch.Tensor,
  advantages: torch.Tensor,
  clip_param: float,
  weights: torch.Tensor | None = None,
  mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
  """MPOPI's decoupled clipped surrogate loss (to minimize).

  ``mean_i[ -w_i * min(r_i A_i, clip(r_i, 1 - eps, 1 + eps) A_i) ]`` over
  accepted samples, with ``r = pi_theta / pi_old`` and constant weights
  ``w = clip(pi_old / mu)``. With ``weights`` and ``mask`` both None this is
  exactly RSL-RL's PPO surrogate.

  Returns:
    The loss and the ratio ``r``.
  """
  ratio = torch.exp(log_prob - old_log_prob)
  surrogate = -advantages * ratio
  surrogate_clipped = -advantages * torch.clamp(
    ratio, 1.0 - clip_param, 1.0 + clip_param
  )
  per_sample = torch.max(surrogate, surrogate_clipped)
  if weights is not None:
    per_sample = per_sample * weights
  return _mean(per_sample, mask), ratio


def _mean(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
  """Mean over accepted samples; plain ``mean()`` when there is no mask."""
  if mask is None:
    return x.mean()
  m = mask.reshape(x.shape).to(x.dtype)
  return (x * m).sum() / m.sum().clamp_min(1.0)


def _normalize(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
  """Upstream advantage normalization, restricted to accepted samples."""
  if mask is None:
    return (x - x.mean()) / (x.std() + 1e-8)
  accepted = x[mask]
  out = (x - accepted.mean()) / (accepted.std() + 1e-8)
  return torch.where(mask, out, torch.zeros_like(out))

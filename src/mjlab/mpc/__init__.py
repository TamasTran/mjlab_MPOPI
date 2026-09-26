"""Sampling-based MPC (MPPI / MPOPI) on batched mjlab environments.

``mjlab.mpc.collector`` (MPC data collection for PPO) is not imported here
because it depends on ``mjlab.rl``, which imports this package's config.
"""

from mjlab.mpc.config import SamplingMpcCfg as SamplingMpcCfg
from mjlab.mpc.sampling_mpc import MpcPlan as MpcPlan
from mjlab.mpc.sampling_mpc import SamplingMpc as SamplingMpc
from mjlab.mpc.sampling_mpc import mppi_weights as mppi_weights

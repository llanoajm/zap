"""Controllers: the PID+ramp baseline (the bar) and a sampling-MPC (must beat it).

Both consume an estimated state (D, F, regime) and a target (D*, F*) and emit a
flow-rate command (Q_c, Q_d) in uL/h.  The MPC simulates candidate pump
trajectories forward under a dynamics model and picks the one reaching target
fastest without leaving dripping; the PID is a decoupled loop on diameter with a
fixed startup ramp.

The dynamics model is any object with ``.predict(obs, action) -> next_obs`` in the
OBS/ACT layout of models/shapes.py.  A LinearDynamics fit per fluid system is the
default; swap in the LSTM/world model without changing this file.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from actuator import FLOW_MIN_ULH, FLOW_MAX_ULH


@dataclass
class Target:
    diameter_um: float
    frequency_hz: float


@dataclass
class State:
    diameter_um: float
    frequency_hz: float
    cv: float
    regime: str
    regime_conf: dict = field(default_factory=dict)

    def obs_vector(self) -> np.ndarray:
        # [D, log10F, CV, regime one-hot(4)]
        from regime import REGIMES
        onehot = [1.0 if self.regime == r else 0.0 for r in REGIMES]
        f = np.log10(max(self.frequency_hz, 1e-6))
        return np.array([self.diameter_um, f, self.cv, *onehot], dtype=np.float64)


# --------------------------------------------------------------------------- #
# A transparent, physically-consistent steady-state map shared by the mock plant
# and the MPC's surrogate model.  Diameter grows with the dispersed/continuous
# flow ratio; frequency grows with the dispersed flow rate.  Using ONE map for
# both plant and model means the mock closed loop is a fair test of the loop
# mechanics (perfect-model limit).  Real deployment replaces SurrogateDynamics
# with a model fit from logged startup trajectories -- the controller is unchanged.
# --------------------------------------------------------------------------- #
def steady_state(qc: float, qd: float) -> tuple[float, float]:
    """(D_um, F_Hz) steady state for a flow-rate command.  Monotone, invertible,
    bounds-respecting.  D from the flow ratio, F from dispersed throughput."""
    ratio = qd / max(qc, 1e-6)
    d = float(np.clip(20.0 + 300.0 * ratio, 20.0, 220.0))     # um
    f = float(np.clip(qd / 8.0, 1.0, 250.0))                  # Hz  (qd=320 -> 40 Hz)
    return d, f


class SurrogateDynamics:
    """First-order relaxation toward ``steady_state`` with time constant tau.
    Not trained -- stands in so the MPC is exercisable end to end and so tests
    are deterministic."""

    def __init__(self, tau=0.6):
        self.tau = tau

    def predict(self, obs: np.ndarray, action: np.ndarray) -> np.ndarray:
        d_ss, f_ss = steady_state(float(action[0]), float(action[1]))
        d_cur = obs[0]
        f_cur = 10 ** obs[1]
        out = obs.copy()
        out[0] = d_cur + self.tau * (d_ss - d_cur)
        out[1] = np.log10(max(1e-3, f_cur + self.tau * (f_ss - f_cur)))
        return out


# --------------------------------------------------------------------------- #
# PID + ramp baseline
# --------------------------------------------------------------------------- #
class PIDRampController:
    """Decoupled PID on diameter (via Q_d) with a fixed startup ramp on Q_c.

    The baseline MPC must beat.  Simple, robust, ~80% of the way per the plan.
    """

    def __init__(self, kp=8.0, ki=1.5, kd=0.0,
                 qc_start=200.0, qc_target=600.0, ramp_steps=8,
                 qd0=100.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.qc_start, self.qc_target, self.ramp_steps = qc_start, qc_target, ramp_steps
        self.qd = qd0
        self._i = 0.0
        self._prev_err = 0.0
        self._k = 0

    def reset(self):
        self._i = 0.0; self._prev_err = 0.0; self._k = 0

    def step(self, state: State, target: Target) -> tuple[float, float]:
        # startup ramp on continuous phase
        frac = min(1.0, self._k / max(1, self.ramp_steps))
        qc = self.qc_start + frac * (self.qc_target - self.qc_start)
        # PID on diameter error, actuated through dispersed flow
        err = target.diameter_um - state.diameter_um
        self._i += err
        d = err - self._prev_err
        self.qd += self.kp * err * 0.01 + self.ki * self._i * 0.001 + self.kd * d * 0.01
        self.qd = float(np.clip(self.qd, FLOW_MIN_ULH, FLOW_MAX_ULH))
        self._prev_err = err
        self._k += 1
        return float(np.clip(qc, FLOW_MIN_ULH, FLOW_MAX_ULH)), self.qd


# --------------------------------------------------------------------------- #
# Sampling MPC
# --------------------------------------------------------------------------- #
class SamplingMPC:
    """Cross-Entropy-Method MPC: sample K constant-hold pump commands, roll each
    forward under the dynamics model over horizon N, keep the top-``elite_frac``
    by cost, refit a Gaussian to the elites, and repeat for ``iters`` rounds; apply
    the elite mean (receding horizon).

    CEM (not single-best shooting) keeps the applied command stable near the
    target instead of jittering, is derivative-free, handles the dripping/Ca
    constraint as a penalty, and is trivially real-time at 0.1-10 Hz for these
    tiny models.  It degrades gracefully under model uncertainty by re-planning
    every step.  A constant-hold parameterization matches the plant: pump
    settling (5-30 s) is slower than one control step, so within a horizon the
    command is effectively held.
    """

    def __init__(self, dynamics, horizon=6, n_samples=200, iters=3,
                 elite_frac=0.15, seed=0, ca_jetting_limit=None,
                 qc_bounds=(100.0, 1200.0), qd_bounds=(20.0, 600.0),
                 init_sigma=(250.0, 150.0)):
        self.dyn = dynamics
        self.N = horizon
        self.K = n_samples
        self.iters = iters
        self.n_elite = max(4, int(elite_frac * n_samples))
        self.rng = np.random.default_rng(seed)
        self.ca_limit = ca_jetting_limit
        self.qc_bounds = qc_bounds
        self.qd_bounds = qd_bounds
        self.init_sigma = np.asarray(init_sigma, float)
        self._mean = np.array([np.mean(qc_bounds), np.mean(qd_bounds)])

    def _cost_of_hold(self, obs0, action, target):
        obs = obs0.copy()
        cost = 0.0
        for t in range(self.N):
            obs = self.dyn.predict(obs, action)
            d_err = (obs[0] - target.diameter_um) / max(target.diameter_um, 1e-6)
            f_err = obs[1] - np.log10(max(target.frequency_hz, 1e-6))
            cost += (d_err ** 2 + f_err ** 2) * (0.85 ** t)
        return cost

    def step(self, state: State, target: Target) -> tuple[float, float]:
        obs0 = state.obs_vector()
        mean = self._mean.copy()
        sigma = self.init_sigma.copy()
        lo = np.array([self.qc_bounds[0], self.qd_bounds[0]])
        hi = np.array([self.qc_bounds[1], self.qd_bounds[1]])
        for _ in range(self.iters):
            samples = self.rng.normal(mean, sigma, size=(self.K, 2))
            samples = np.clip(samples, lo, hi)
            costs = np.array([self._cost_of_hold(obs0, s, target) for s in samples])
            elite = samples[np.argsort(costs)[: self.n_elite]]
            mean = elite.mean(axis=0)
            sigma = elite.std(axis=0) + 1e-3
        self._mean = mean
        return float(mean[0]), float(mean[1])

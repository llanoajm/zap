"""Watchdog and convergence detection for the control loop.

The watchdog is the last line between an ML controller and a flooded stage: if
frames stop arriving, or the measured state goes non-physical, or the regime
leaves dripping, it trips the actuator to its safe state and latches until reset.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass


@dataclass
class WatchdogConfig:
    max_frame_gap_s: float = 2.0        # no fresh frames -> trip
    max_diameter_um: float = 1000.0     # non-physical measurement -> trip
    min_droplets_for_valid: int = 1
    trip_on_jetting: bool = True


class Watchdog:
    def __init__(self, cfg: WatchdogConfig | None = None):
        self.cfg = cfg or WatchdogConfig()
        self._last_frame_t = time.monotonic()
        self.tripped = False
        self.reason = ""

    def feed_frames(self, now: float | None = None):
        self._last_frame_t = time.monotonic() if now is None else now

    def check(self, measurement, regime: str, now: float | None = None) -> bool:
        """Return True if safe, False (and latch) if a trip condition is met."""
        now = time.monotonic() if now is None else now
        if self.tripped:
            return False
        if now - self._last_frame_t > self.cfg.max_frame_gap_s:
            return self._trip("frame timeout")
        if measurement.diameter_um > self.cfg.max_diameter_um:
            return self._trip(f"diameter {measurement.diameter_um:.0f}um non-physical")
        if self.cfg.trip_on_jetting and regime == "jetting":
            return self._trip("regime left dripping (jetting)")
        return True

    def _trip(self, reason: str) -> bool:
        self.tripped = True
        self.reason = reason
        return False

    def reset(self):
        self.tripped = False
        self.reason = ""
        self._last_frame_t = time.monotonic()


class ConvergenceDetector:
    """Declares convergence when diameter and frequency are within tolerance of
    target and stable (low variance) over a sliding window of control steps."""

    def __init__(self, d_tol_um=6.0, f_tol_frac=0.15, window=5, stable_cv=0.07):
        self.d_tol = d_tol_um
        self.f_tol = f_tol_frac
        self.window = window
        self.stable_cv = stable_cv
        self._d = deque(maxlen=window)
        self._f = deque(maxlen=window)

    def update(self, state, target) -> bool:
        import numpy as np
        self._d.append(state.diameter_um)
        self._f.append(state.frequency_hz)
        if len(self._d) < self.window:
            return False
        d_ok = abs(state.diameter_um - target.diameter_um) <= self.d_tol
        f_ok = (abs(state.frequency_hz - target.frequency_hz)
                <= self.f_tol * max(target.frequency_hz, 1e-6))
        d_stable = np.std(self._d) / max(np.mean(self._d), 1e-6) <= self.stable_cv
        return bool(d_ok and f_ok and d_stable)

"""The control loop.  Synchronous path first (default), asynchronous path behind
a flag, exactly as specified in docs/architecture.md.

Synchronous (default): capture N frames -> extract (D, F, regime) -> run
controller -> write pump command -> wait out the remainder of the control
interval.  One thread, deterministic, inspectable.

Asynchronous (--async): Thread A captures + measures into a FIFO continuously at
camera rate; Thread B reads the recent window at control rate, runs the
controller, writes the command.  queue.Queue suffices (OpenCV and PyTorch release
the GIL).  multiprocessing.Queue + shared-memory frames is the next step if
contention proves real -- noted, not implemented, because it is premature here.

Everything runs against the mock camera/actuator with no hardware attached.
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field

from measurement import Calibration, measure_batch
from controller import State, Target
from safety import Watchdog, ConvergenceDetector


@dataclass
class LoopConfig:
    control_interval_s: float = 5.0     # 0.1-10 Hz; 5 s is the settling-limited default
    frames_per_step: int = 250          # 50 fps * 5 s -> 250 frames per control step
    fps: float = 250.0
    max_steps: int = 40
    stop_on_convergence: bool = True


@dataclass
class StepLog:
    t: float
    state: State
    command: tuple
    converged: bool
    tripped: bool
    reason: str = ""


class SynchronousLoop:
    def __init__(self, camera, actuator, controller, calib: Calibration,
                 regime_estimator, cfg: LoopConfig | None = None,
                 watchdog: Watchdog | None = None):
        self.cam = camera
        self.act = actuator
        self.ctrl = controller
        self.calib = calib
        self.regime = regime_estimator
        self.cfg = cfg or LoopConfig()
        self.wd = watchdog or Watchdog()
        self.conv = ConvergenceDetector()
        self.log: list[StepLog] = []

    def run(self, target: Target, now_fn=time.monotonic) -> list[StepLog]:
        cfg = self.cfg
        for k in range(cfg.max_steps):
            t0 = now_fn()
            frames = self.cam.grab(cfg.frames_per_step)
            self.wd.feed_frames(now=t0)
            meas = measure_batch(frames, cfg.fps, self.calib)
            label, conf = self.regime.classify(frames)
            state = State(meas.diameter_um, meas.frequency_hz, meas.diameter_cv,
                          label, conf)
            # safety gate BEFORE actuation
            safe = self.wd.check(meas, label, now=t0)
            if not safe:
                self.act.safe()
                self.log.append(StepLog(t0, state, self.act._last, False, True,
                                        self.wd.reason))
                break
            qc, qd = self.ctrl.step(state, target)
            st = self.act.command((qc, qd), now=t0)
            converged = self.conv.update(state, target)
            self.log.append(StepLog(t0, state, st.setpoints, converged, False))
            # let the mock chip "respond" so closed-loop tests can converge
            self._maybe_actuate_mock(qc, qd)
            if converged and cfg.stop_on_convergence:
                break
            self._wait(t0, now_fn)
        return self.log

    def _maybe_actuate_mock(self, qc, qd):
        # if the camera is a MockCamera, let its emulated chip respond to the
        # command (via the SAME steady_state map the MPC models) so the loop is a
        # genuine closed loop in simulation, with first-order settling.
        if hasattr(self.cam, "set_state"):
            from controller import steady_state
            d_ss, f_ss = steady_state(qc, qd)
            # first-order approach in the plant, in micron/Hz
            cur_d_um = self.cam.diameter_px * self.calib.microns_per_pixel
            new_d_um = cur_d_um + 0.6 * (d_ss - cur_d_um)
            new_f = self.cam.drop_freq_hz + 0.6 * (f_ss - self.cam.drop_freq_hz)
            self.cam.set_state(drop_freq_hz=max(1.0, new_f),
                               diameter_px=max(4.0, new_d_um
                                               / self.calib.microns_per_pixel))

    def _wait(self, t0, now_fn):
        # in real deployment: sleep to the interval; in sim/test: skip real sleep
        remaining = self.cfg.control_interval_s - (now_fn() - t0)
        if remaining > 0 and now_fn is time.monotonic:
            time.sleep(remaining)


class AsynchronousLoop:
    """Thread A: capture+measure into a FIFO at camera rate.  Thread B: read the
    most recent measurement window at control rate and actuate.

    queue.Queue suffices because measure_batch (OpenCV) and any torch inference
    release the GIL.  For heavier inference or a second camera, move to
    multiprocessing.Queue + shared_memory frame arrays -- see module docstring.
    """

    def __init__(self, camera, actuator, controller, calib, regime_estimator,
                 cfg: LoopConfig | None = None, watchdog: Watchdog | None = None):
        self.cam = camera
        self.act = actuator
        self.ctrl = controller
        self.calib = calib
        self.regime = regime_estimator
        self.cfg = cfg or LoopConfig()
        self.wd = watchdog or Watchdog()
        self.conv = ConvergenceDetector()
        self._q: queue.Queue = queue.Queue(maxsize=8)
        self._stop = threading.Event()
        self.log: list[StepLog] = []

    def _capture_thread(self):
        batch = max(8, int(self.cfg.fps * 0.1))  # 100 ms of frames per measurement
        while not self._stop.is_set():
            frames = self.cam.grab(batch)
            meas = measure_batch(frames, self.cfg.fps, self.calib)
            label, _ = self.regime.classify(frames)
            try:
                self._q.put((time.monotonic(), meas, label), timeout=0.1)
            except queue.Full:
                try:
                    self._q.get_nowait()  # drop oldest, keep fresh
                except queue.Empty:
                    pass

    def run(self, target: Target, wall_clock_s: float = 2.0) -> list[StepLog]:
        cap = threading.Thread(target=self._capture_thread, daemon=True)
        cap.start()
        t_end = time.monotonic() + wall_clock_s
        try:
            while time.monotonic() < t_end and not self._stop.is_set():
                t0 = time.monotonic()
                try:
                    ts, meas, label = self._q.get(timeout=1.0)
                except queue.Empty:
                    self.act.safe()
                    break
                self.wd.feed_frames(now=ts)
                state = State(meas.diameter_um, meas.frequency_hz,
                              meas.diameter_cv, label, {})
                if not self.wd.check(meas, label, now=t0):
                    self.act.safe()
                    self.log.append(StepLog(t0, state, self.act._last, False,
                                            True, self.wd.reason))
                    break
                qc, qd = self.ctrl.step(state, target)
                st = self.act.command((qc, qd), now=t0)
                converged = self.conv.update(state, target)
                self.log.append(StepLog(t0, state, st.setpoints, converged, False))
                if hasattr(self.cam, "set_state"):
                    from controller import steady_state
                    d_ss, f_ss = steady_state(qc, qd)
                    cur_d_um = self.cam.diameter_px * self.calib.microns_per_pixel
                    new_d_um = cur_d_um + 0.5 * (d_ss - cur_d_um)
                    new_f = self.cam.drop_freq_hz + 0.5 * (f_ss - self.cam.drop_freq_hz)
                    self.cam.set_state(drop_freq_hz=max(1.0, new_f),
                                       diameter_px=max(4.0, new_d_um
                                                       / self.calib.microns_per_pixel))
                time.sleep(max(0.0, self.cfg.control_interval_s
                               - (time.monotonic() - t0)))
        finally:
            self._stop.set()
            cap.join(timeout=1.0)
        return self.log

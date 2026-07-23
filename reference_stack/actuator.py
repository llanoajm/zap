"""Actuator abstraction: syringe pumps now, pressure controllers later, WITHOUT
touching controller code.

The controller only ever sees ``Actuator.command(setpoints)`` where setpoints are
flow rates (uL/h).  A pressure controller subclass maps a commanded flow rate to
a pressure through a device-specific fluidic resistance, so the swap the open
question asks for is a subclass, not a controller rewrite.

SAFETY.  The bounds below are HARD CONSTANTS and are intentionally not
constructor arguments and not overridable.  An actuator under model control with
no bounds checking destroys devices and floods the stage.  Every command is
clamped to [MIN, MAX], slew-rate-limited, and gated by a watchdog that trips the
actuator to a safe state if it is not fed within WATCHDOG_TIMEOUT_S.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# HARD LIMITS -- do not make these configurable.
# --------------------------------------------------------------------------- #
FLOW_MIN_ULH = 0.0            # uL/h; pumps do not run in reverse here
FLOW_MAX_ULH = 5000.0         # uL/h; above this, dead-volume transit + Ca push
                              # every device in the matrix past jetting
FLOW_SLEW_MAX_ULH_PER_S = 2000.0   # max change per second (protects the interface)
PRESSURE_MIN_MBAR = 0.0
PRESSURE_MAX_MBAR = 2000.0    # mbar; bonded single-layer PDMS delamination margin
PRESSURE_SLEW_MAX_MBAR_PER_S = 800.0
WATCHDOG_TIMEOUT_S = 2.0      # if not commanded within this, go to SAFE_STATE
SAFE_STATE_ULH = (0.0, 0.0)   # (Q_c, Q_d) the watchdog drives to


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


@dataclass
class ActuatorState:
    setpoints: tuple[float, ...]
    clamped: bool
    slew_limited: bool
    t: float


class Actuator:
    """Base class.  Subclasses implement ``_write(setpoints)`` for real hardware.

    The public API (``command``, ``safe``, ``fed_within``) is what the controller
    uses; it never sees pressures vs flow rates.
    """

    n_channels = 2
    _min = FLOW_MIN_ULH
    _max = FLOW_MAX_ULH
    _slew = FLOW_SLEW_MAX_ULH_PER_S
    units = "uL/h"

    def __init__(self):
        self._last = tuple(0.0 for _ in range(self.n_channels))
        self._last_t = time.monotonic()

    # -- safety-enforcing entry point -------------------------------------- #
    def command(self, setpoints, now: float | None = None) -> ActuatorState:
        now = time.monotonic() if now is None else now
        if len(setpoints) != self.n_channels:
            raise ValueError(f"expected {self.n_channels} setpoints")
        dt = max(1e-3, now - self._last_t)
        out, clamped, slewed = [], False, False
        for prev, want in zip(self._last, setpoints):
            c = _clamp(float(want), self._min, self._max)
            if c != want:
                clamped = True
            max_step = self._slew * dt
            if abs(c - prev) > max_step:
                c = prev + max_step * (1 if c > prev else -1)
                slewed = True
            out.append(c)
        out = tuple(out)
        self._write(out)
        self._last, self._last_t = out, now
        return ActuatorState(out, clamped, slewed, now)

    def safe(self):
        """Force the actuator to its safe state immediately (bypasses slew)."""
        state = tuple(SAFE_STATE_ULH[:self.n_channels])
        self._write(state)
        self._last, self._last_t = state, time.monotonic()
        return state

    def fed_within(self, timeout: float = WATCHDOG_TIMEOUT_S,
                   now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        return (now - self._last_t) <= timeout

    def _write(self, setpoints):  # pragma: no cover - overridden
        raise NotImplementedError

    def close(self):
        pass


class MockActuator(Actuator):
    """In-memory pump; records every write.  Lets the whole stack run with no
    hardware attached and lets tests assert on the command history."""

    def __init__(self):
        super().__init__()
        self.history: list[tuple[float, tuple[float, ...]]] = []

    def _write(self, setpoints):
        self.history.append((self._last_t, setpoints))


class HarvardSyringePump(Actuator):
    """Harvard Apparatus syringe pump(s) over USB-serial, simple ASCII protocol.

    One serial channel per pump.  Flow rate command is translated to the pump's
    ASCII 'irate'/'wrate' set + 'run'.  Requires pyserial; import is deferred so
    the module loads on machines without it (e.g. CI, the dev box).
    """

    def __init__(self, ports, syringe_diameter_mm=4.61, baud=9600):
        super().__init__()
        import serial  # deferred
        self.n_channels = len(ports)
        self._ser = [serial.Serial(p, baud, timeout=0.2) for p in ports]
        for s in self._ser:
            s.write(f"diameter {syringe_diameter_mm}\r".encode())

    def _write(self, setpoints):
        # Harvard ASCII: set rate in uL/hr then run.  One command per pump.
        for s, rate in zip(self._ser, setpoints):
            s.write(f"irate {rate:.4f} ul/hr\r".encode())
            s.write(b"irun\r")

    def close(self):
        for s in self._ser:
            try:
                s.write(b"stop\r")
                s.close()
            except Exception:
                pass


class PressureController(Actuator):
    """Drop-in pressure actuator.  The controller still commands FLOW RATES; this
    class converts them to pressures via a device fluidic resistance R
    (mbar per uL/h), so no controller code changes when the actuator swaps.

    P = R * Q.  Bounds and slew are enforced in PRESSURE units here; the flow-rate
    setpoint from the controller is mapped, clamped in pressure, then (optionally)
    reported back.  This demonstrates the abstraction; a real device needs a
    calibrated R and a pressure sensor for closed-loop flow.
    """

    _min = PRESSURE_MIN_MBAR
    _max = PRESSURE_MAX_MBAR
    _slew = PRESSURE_SLEW_MAX_MBAR_PER_S
    units = "mbar"

    def __init__(self, resistance_mbar_per_ulh=(0.5, 0.5)):
        self.n_channels = len(resistance_mbar_per_ulh)
        self._R = resistance_mbar_per_ulh
        super().__init__()
        self.history: list[tuple[float, tuple[float, ...]]] = []

    def command(self, setpoints_flow, now=None):
        # map flow -> pressure, then run the base safety machinery in pressure units
        press = tuple(r * q for r, q in zip(self._R, setpoints_flow))
        return super().command(press, now)

    def _write(self, pressures):
        self.history.append((self._last_t, pressures))

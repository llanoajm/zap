"""Tests for the reference stack -- all green with no camera and no pump.

Run:  cd reference_stack && python -m pytest -q
  or: python -m pytest reference_stack -q
"""
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

import physics as ph
from measurement import (Calibration, measure_batch, synthetic_dripping,
                         frequency_from_batch)
from actuator import (MockActuator, PressureController, FLOW_MAX_ULH,
                      FLOW_SLEW_MAX_ULH_PER_S, SAFE_STATE_ULH)
from controller import (Target, State, SamplingMPC, SurrogateDynamics,
                        PIDRampController, steady_state)
from safety import Watchdog, WatchdogConfig, ConvergenceDetector
from regime import HeuristicRegime
from camera import MockCamera
from loop import SynchronousLoop, LoopConfig


# --------------------------------------------------------------------------- #
# Physics
# --------------------------------------------------------------------------- #
def test_w_over_ell_is_oh_minus_2():
    for fs in ph.FLUID_SYSTEMS.values():
        for w in (15e-6, 60e-6, 200e-6):
            assert math.isclose(ph.w_over_ellstar(fs, w),
                                ph.ohnesorge_number(fs, w) ** -2, rel_tol=1e-12)


def test_we_equals_ca_times_re():
    fs = ph.FLUID_SYSTEMS["hfe_lowsurf"]
    u = ph.u_continuous(2e-10, 40e-6, 45e-6)
    assert math.isclose(ph.weber_number(fs, u, 40e-6),
                        ph.capillary_number(fs, u) * ph.reynolds_number(fs, u, 40e-6),
                        rel_tol=1e-12)


def test_ell_star_spans_orders_of_magnitude():
    ells = [fs.ell_star_um() for fs in ph.FLUID_SYSTEMS.values()]
    assert max(ells) / min(ells) > 100  # discriminating power for normalization


def test_hfe_reference_corrected_not_14um():
    # the source-doc 14 um is wrong given HFE density; corrected value ~5.3 um
    assert 4.5 < ph.FLUID_SYSTEMS["hfe_highsurf"].ell_star_um() < 6.0


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fhz", [20.0, 40.0, 60.0])
def test_frequency_recovery(fhz):
    frames = synthetic_dripping(200, drop_freq_hz=fhz, fps=250.0, diameter_px=24)
    est = frequency_from_batch(frames, 250.0)
    assert abs(est - fhz) <= 2.0


def test_diameter_and_cv_reasonable():
    cal = Calibration.from_reference(100.0, 50.0)  # 2 um/px
    frames = synthetic_dripping(200, drop_freq_hz=40.0, diameter_px=24.0)
    m = measure_batch(frames, 250.0, cal)
    assert 40 < m.diameter_um < 52     # ~48 um nominal, minus morph-open bias
    assert 0 <= m.diameter_cv < 0.2
    assert m.n_droplets > 0


def test_calibration_rejects_bad_reference():
    with pytest.raises(ValueError):
        Calibration.from_reference(100.0, 0.0)


# --------------------------------------------------------------------------- #
# Actuator safety -- the limits that protect devices
# --------------------------------------------------------------------------- #
def test_flow_is_clamped_to_max():
    a = MockActuator()
    st = a.command((99999.0, 99999.0), now=100.0)
    assert st.clamped
    assert all(v <= FLOW_MAX_ULH for v in st.setpoints)


def test_slew_rate_limited():
    a = MockActuator()
    a.command((0.0, 0.0), now=0.0)
    st = a.command((FLOW_MAX_ULH, FLOW_MAX_ULH), now=0.1)  # 0.1 s later
    assert st.slew_limited
    # max change in 0.1 s
    assert st.setpoints[0] <= FLOW_SLEW_MAX_ULH_PER_S * 0.1 + 1e-6


def test_safe_state():
    a = MockActuator()
    a.command((500.0, 300.0), now=0.0)
    s = a.safe()
    assert s == tuple(SAFE_STATE_ULH[:a.n_channels])


def test_limits_not_overridable_via_constructor():
    # HARD constants: no constructor path sets a higher max
    a = MockActuator()
    assert a._max == FLOW_MAX_ULH


def test_pressure_controller_is_drop_in():
    # controller commands FLOW; pressure actuator maps + clamps in pressure units
    a = PressureController(resistance_mbar_per_ulh=(0.5, 0.5))
    st = a.command((1000.0, 1000.0), now=0.0)
    assert a.units == "mbar"
    assert len(st.setpoints) == 2


# --------------------------------------------------------------------------- #
# Watchdog + convergence
# --------------------------------------------------------------------------- #
def test_watchdog_trips_on_frame_timeout():
    wd = Watchdog(WatchdogConfig(max_frame_gap_s=1.0))
    wd.feed_frames(now=0.0)

    class M:  # minimal measurement stub
        diameter_um = 50.0
    assert wd.check(M(), "dripping", now=0.5) is True
    assert wd.check(M(), "dripping", now=2.0) is False
    assert wd.tripped and "timeout" in wd.reason


def test_watchdog_trips_on_jetting():
    wd = Watchdog()
    wd.feed_frames(now=0.0)

    class M:
        diameter_um = 50.0
    assert wd.check(M(), "jetting", now=0.1) is False


def test_convergence_requires_stability():
    cd = ConvergenceDetector(d_tol_um=5, f_tol_frac=0.15, window=3)
    tgt = Target(100.0, 40.0)
    for _ in range(3):
        conv = cd.update(State(100.0, 40.0, 0.03, "dripping"), tgt)
    assert conv is True


# --------------------------------------------------------------------------- #
# Controllers + closed loop
# --------------------------------------------------------------------------- #
def test_steady_state_monotone():
    d1, _ = steady_state(1000.0, 100.0)
    d2, _ = steady_state(1000.0, 300.0)   # more dispersed -> bigger drops
    assert d2 > d1
    _, f1 = steady_state(1000.0, 100.0)
    _, f2 = steady_state(1000.0, 300.0)   # more dispersed -> higher freq
    assert f2 > f1


def test_mpc_respects_bounds():
    mpc = SamplingMPC(SurrogateDynamics(), qc_bounds=(100, 1200),
                      qd_bounds=(20, 600))
    qc, qd = mpc.step(State(20.0, 5.0, 0.0, "no_droplets"), Target(100.0, 40.0))
    assert 100 <= qc <= 1200 and 20 <= qd <= 600


def test_mpc_closed_loop_converges():
    cam = MockCamera(hw=(256, 256), fps=250.0, drop_freq_hz=15.0, diameter_px=10.0)
    act = MockActuator()
    cal = Calibration.from_reference(100.0, 50.0)
    ctrl = SamplingMPC(SurrogateDynamics(), horizon=6, n_samples=160)
    cfg = LoopConfig(control_interval_s=0.0, frames_per_step=200, fps=250.0,
                     max_steps=25)
    loop = SynchronousLoop(cam, act, ctrl, cal, HeuristicRegime(), cfg)
    log = loop.run(Target(diameter_um=100.0, frequency_hz=40.0))
    assert log[-1].converged, f"did not converge: final={log[-1].state}"
    assert not log[-1].tripped


def test_pid_reaches_diameter_but_misses_frequency():
    # documents the baseline's weakness the MPC must beat: decoupled PID controls
    # diameter but has no authority over frequency at a joint target.
    cam = MockCamera(fps=250.0, drop_freq_hz=15.0, diameter_px=10.0)
    act = MockActuator()
    cal = Calibration.from_reference(100.0, 50.0)
    ctrl = PIDRampController(qc_target=600.0)
    cfg = LoopConfig(control_interval_s=0.0, frames_per_step=200, max_steps=25)
    loop = SynchronousLoop(cam, act, ctrl, cal, HeuristicRegime(), cfg)
    log = loop.run(Target(100.0, 40.0))
    # frequency error remains large -> not converged on the joint target
    assert abs(log[-1].state.frequency_hz - 40.0) > 4.0

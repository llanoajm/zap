"""Run the full closed loop against mock hardware -- no camera, no pump attached.

    python reference_stack/run_mock.py                 # synchronous, MPC
    python reference_stack/run_mock.py --controller pid
    python reference_stack/run_mock.py --async         # asynchronous path

Demonstrates: capture -> measure -> estimate state -> control -> actuate -> wait,
with safety bounds, watchdog, and convergence detection all live.
"""
from __future__ import annotations

import argparse

from camera import MockCamera
from actuator import MockActuator
from measurement import Calibration
from regime import HeuristicRegime
from controller import (Target, PIDRampController, SamplingMPC, SurrogateDynamics)
from loop import SynchronousLoop, AsynchronousLoop, LoopConfig


def build(controller_name: str):
    cam = MockCamera(hw=(256, 256), fps=250.0, drop_freq_hz=15.0, diameter_px=10.0)
    act = MockActuator()
    calib = Calibration.from_reference(known_um=100.0, measured_px=50.0)  # 2 um/px
    if controller_name == "pid":
        ctrl = PIDRampController()
    else:
        ctrl = SamplingMPC(SurrogateDynamics(), horizon=6, n_samples=128)
    return cam, act, calib, ctrl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", choices=["mpc", "pid"], default="mpc")
    ap.add_argument("--async", dest="async_", action="store_true")
    ap.add_argument("--target-d", type=float, default=100.0)
    ap.add_argument("--target-f", type=float, default=40.0)
    args = ap.parse_args()

    cam, act, calib, ctrl = build(args.controller)
    target = Target(diameter_um=args.target_d, frequency_hz=args.target_f)
    cfg = LoopConfig(control_interval_s=0.0, frames_per_step=200, fps=250.0,
                     max_steps=25)

    if args.async_:
        loop = AsynchronousLoop(cam, act, ctrl, calib, HeuristicRegime(), cfg)
        log = loop.run(target, wall_clock_s=1.5)
    else:
        loop = SynchronousLoop(cam, act, ctrl, calib, HeuristicRegime(), cfg)
        log = loop.run(target)

    print(f"controller={args.controller}  target D={target.diameter_um}um "
          f"F={target.frequency_hz}Hz  steps={len(log)}")
    print(f"{'step':>4}{'D_um':>8}{'F_Hz':>8}{'regime':>11}"
          f"{'Qc':>8}{'Qd':>8}{'conv':>6}")
    for i, s in enumerate(log):
        print(f"{i:>4}{s.state.diameter_um:>8.1f}{s.state.frequency_hz:>8.1f}"
              f"{s.state.regime:>11}{s.command[0]:>8.0f}{s.command[1]:>8.0f}"
              f"{'Y' if s.converged else '':>6}")
    final = log[-1]
    print(f"\nfinal: D={final.state.diameter_um:.1f}um "
          f"F={final.state.frequency_hz:.1f}Hz "
          f"converged={final.converged} tripped={final.tripped} {final.reason}")


if __name__ == "__main__":
    main()

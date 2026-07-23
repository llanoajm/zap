# Architecture

One architecture, committed. Primary and exactly one fallback, with the trigger
that switches between them. All parameter counts, FLOPs, and memory footprints
are computed (`python models/profiling.py`) and latencies measured
(`python bench/benchmark.py`), not estimated.

## The commitment

**Primary:** Raspberry Pi 5 (8 GB) + one ZWO ASI174MM (USB3) + two Harvard
Apparatus syringe pumps (USB-serial). Classical-CV measurement, a lightweight CNN
(or frame-diff heuristic) regime classifier, a per-fluid **linear** dynamics
model, and a CEM-MPC controller. Synchronous loop first. This runs the whole
single-emulsion program at 0.1–1 Hz control.

**Fallback (one):** mini-PC/NUC + GPU, adopted **only** if the measurement path
must become a *learned high-resolution detector* — i.e. classical CV fails to
extract diameter/frequency reliably (low index contrast, dense/overlapping
droplets). **Trigger:** measured CV *or* detector p95 latency > 40 % of the
control interval on the Pi 5 at the required ROI/fps, or a need for sustained
> 250 fps full-frame detection. The same PyTorch/ONNX code runs on both; only the
device string changes. Jetson Orin Nano is the embedded flavour of this fallback
for a deployed unit. **FPGA is not on the path** (see `docs/decisions.md`).

The DE (double-emulsion) future — two cameras + Pico trigger + three pumps —
forces the NUC for a *bandwidth* reason (two ASI174MM saturate one Pi 5 USB3
controller), independent of the inference trigger above.

## Build order (8 weeks) — endorsed, with one change

The proposed order (classifier → dynamics → controller) is right: it goes
easiest-and-most-data-efficient first, and each stage de-risks the next. One
change: **start logging startup trajectories from week 1**, before the classifier
is good, because every run is training data for the dynamics model and transients
are the signal (a 60 s run at 50 fps is hundreds of (state, action, next-state)
tuples). Don't serialize data collection behind model readiness.

- **Weeks 1–2 — measurement + logging.** Stand up the synchronous loop against the
  reference stack (`reference_stack/`): pixel-to-micron calibration, contour
  diameter, FFT frequency, CV polydispersity, safety bounds, watchdog. Log every
  run. Deliverable: the loop closes on a real chip under manual setpoints, data
  accumulating.
- **Weeks 2–4 — regime classifier.** Frame-diff heuristic first (zero training),
  CNN if it isn't discriminative. Synthetic data (`synthetic/`) for the rare
  regimes. Deliverable: 4-way regime label gating control actions.
- **Weeks 4–6 — dynamics model.** Fit per-fluid linear A, B from the logged
  trajectories; quantify steady-state and transient error. Deliverable: a model
  the MPC can roll forward, with an honest error bar.
- **Weeks 6–8 — controller.** CEM-MPC on top of the linear model; PID+ramp
  baseline for comparison. Deliverable: cold-start-to-target convergence beating
  the baseline on time-to-target and overshoot, staying in dripping.

## Components, sized

State layout everywhere (`models/shapes.py`): `OBS(7) = [D_um, log10 F, CV,
regime one-hot ×4]`, `ACT(2) = [Q_c, Q_d]`, window 16 steps. Full table:

| model | role | in → out | params | MFLOPs/call | param MB (fp32) | peak act MB | ONNX-CPU p50 |
|---|---|---|---|---|---|---|---|
| regime_classifier | per-frame | (B,1,128,128)→(B,4) | 98,148 | 29.49 | 0.393 | 0.262 | 0.167 ms/frame |
| tiny_detector | per-frame (learned CV alt.) | (B,1,128,128)→(B,5,16,16) | 60,549 | 39.09 | 0.242 | 0.262 | 0.142 ms/frame |
| lstm_dynamics | per-step | (B,16,9)→(B,7) | 19,655 | 0.599 | 0.079 | 0.004 | 0.048 ms/step |
| linear_dynamics | per-step (**chosen**) | (B,7)+(B,2)→(B,7) | 70 | ~0.0003 | 0.0003 | ~0 | 0.008 ms/step |
| neural_ode_dynamics | per-step | (B,7)+(B,2)→(B,7) | 5,255 | 0.164 | 0.021 | ~0 | 0.073 ms/step |
| latent_world_model | per-step | (B,16,9)+(B,2)→(B,7)+(B,16) | 5,095 | 0.046 | 0.020 | 0.001 | 0.031 ms/step |

Runnable, randomly-initialized versions of all six are in `models/`, exported to
ONNX (verified load+run+match-eager, 6/6) and CoreML (converted + spec-validated;
prediction pending macOS), so Andrea benchmarks throughput immediately:
`python models/export.py && python bench/benchmark.py`.

### Regime classifier
Lightweight CNN, 4 strided conv blocks → global average pool → linear, **98 k
params, 29.5 MFLOPs/frame, 0.39 MB weights**. At 0.167 ms/frame (ONNX-CPU) it has
a compute-bound ceiling of ~1,700 fps on a Pi 5 CPU — but it never runs per-frame:
it classifies once per control step on a representative/median ROI. The
frame-difference heuristic (`measurement.frame_diff_regime_hint`) is the
zero-training default and is likely sufficient for dripping/jetting/no-droplets;
the CNN exists for satellite discrimination, which is where synthetic data pays
off. **Decision: ship the heuristic; train the CNN only for the rare regimes.**

### Measurement path — classical CV, not a learned detector (for now)
Classical CV (Otsu threshold + circularity-gated contours for diameter; FFT of a
downstream intensity column for frequency; CV over N frames for polydispersity)
measures at **0.155–4.16 ms/frame** depending on ROI (`bench/results.json`).
Because CV throughput is the step that must finish inside the control interval,
this is the real budget: 250 frames at a 5 s interval cost 39–115 ms at
small/medium ROI — orders of magnitude of headroom, on a Pi 5 too. **Decision:
classical CV is the primary measurement path.** `tiny_detector` (a learned
alternative, 60 k params, benchmarked alongside CV) is held in reserve for the
fallback trigger above — low-contrast fluid pairs (water/HFE Δn = 0.043) or dense
droplets where thresholding fails.

### Dynamics model — per-fluid linear
Chosen from the candidates against realistic data volume (`docs/decisions.md`,
`docs/proposals/world_models.md`): **a per-fluid-system linear dynamical system
x_{t+1} = A x_t + B u_t, 70 parameters.** Near steady state it is likely
sufficient, it identifies from a handful of startup runs, and it makes the MPC a
convex QP (`docs/proposals/convex_relaxation.md`). The LSTM, neural ODE, GP, and
latent world model are all implemented and benchmarked; each is the *upgrade* if
the linear model's transient error exceeds the control spec on real data — the
LSTM for window nonlinearity, the GP where the controller needs calibrated
uncertainty, the world model for multi-fluid transfer.

**Hidden parameters** (contact angle θ, dynamic γ(t), PDMS compliance, settling
time τ) are *not* modeled explicitly at this stage: they are absorbed as
per-fluid, per-device residual structure into A, B (fit separately per system) and
into the process-noise term the MPC re-plans against every step. This is the
defensible marginalization at current data volume — a learned latent that
*encodes* them is the world model, deferred until the data justifies it. τ in
particular is learnable directly from the first-order settling visible in logged
step responses.

### Controller — CEM-MPC over the PID+ramp baseline
`SamplingMPC` (`reference_stack/controller.py`): cross-entropy-method sampling MPC
— sample K constant-hold pump commands, roll each forward under the dynamics
model over horizon N, keep the elite fraction by cost, refit, apply the elite
mean (receding horizon). Derivative-free, handles the dripping/Ca constraint as a
penalty, real-time by orders of magnitude at 0.1–10 Hz. Constant-hold matches the
plant: pump settling (5–30 s) is slower than a control step, so within a horizon
the command is effectively held.

**The bar it must beat** is `PIDRampController`: a fixed startup ramp on Q_c and a
decoupled PID on diameter via Q_d. In the mock closed loop the MPC converges to
(D = 100 µm, F = 40 Hz) in 7 steps and latches; the decoupled PID reaches the
diameter but **misses the joint frequency target by design** — it has no authority
over F (`test_pid_reaches_diameter_but_misses_frequency`). That gap is precisely
what a joint (D, F) controller buys, and it is the comparison to report on real
hardware.

## Per-control-step compute budget (primary path, 5 s interval)

Summing the primary path at a representative 256×256 ROI, 250 frames:

- CV measurement: 250 × 0.573 ms = **143 ms** (x86; ~1 s on a Pi 5 at ~7× — still
  20 % of interval).
- regime classify (1×): **~0.2 ms**.
- MPC (linear model, K=200, 3 CEM iters, horizon 6): dynamics is 70 params /
  ~0.0003 MFLOPs per predict; the whole plan is < **5 ms**.
- serial write + wait: the remainder of the 5 s.

Compute is not the constraint anywhere on this path; pump settling is. That single
fact is what makes the Pi 5 sufficient and the FPGA pointless.

## Synchronous first, async behind a flag
`reference_stack/loop.py` implements both. Synchronous
(capture→measure→estimate→control→actuate→wait) is the default: one thread,
deterministic, inspectable, and it maps onto batched tensors later for free.
Async (Thread A capture+measure→FIFO at camera rate; Thread B control-rate
inference) is behind `--async`, using `queue.Queue` (OpenCV and PyTorch release
the GIL). Move to `multiprocessing.Queue` + shared-memory frame arrays only when
profiling shows CV approaching the interval budget or a second camera contends —
premature before then.

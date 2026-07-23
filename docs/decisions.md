# Decisions — the open questions, resolved

These are the live disagreements between the plan and what the team is currently
doing. Each is decided here on the evidence and propagates into everything
downstream. Numbers are computed in this repo (`reference_stack/physics.py`,
`models/profiling.py`, `bench/benchmark.py`, `normalization/study.py`); commands
to reproduce or extend them are given inline.

---

## 1. Raspberry Pi 5 vs Jetson vs FPGA

**Decision: Raspberry Pi 5 (8 GB) is the primary target. Jetson Orin Nano is the
single fallback, live only under one specific trigger. FPGA is dead — save the
money.**

The argument turns on one fact the FPGA/Jetson discussion has been conflating:
**capture-and-measure throughput is decoupled from control rate.** The control
loop is settling-limited to 0.1–10 Hz (pump ramp 0.1–0.5 s + tubing compliance
2–20 s + dead-volume transit; total 5–30 s). Frame-by-frame real-time inference
is therefore unnecessary — frames are accumulated between control decisions and
processed as a batch. What each compute path actually has to do:

- **Model inference** at 0.1–10 Hz. Every candidate is tiny (measured, ONNX-CPU,
  batch-1 p50): regime classifier **0.167 ms/frame**, all dynamics models
  **< 0.1 ms/step** (`bench/results.json`). Even the largest, the regime CNN, is
  29.5 MFLOPs/frame → a compute-bound ceiling of **~1,700 fps on a Pi 5 CPU**
  (~50 GFLOP/s fp32) before any accelerator. Inference is not the constraint on
  any device.
- **CV measurement** — the step that must finish inside the control interval.
  Measured single-thread on x86 (`bench/benchmark.py`): 128×128 **0.155 ms/frame**
  (6,464 fps), 256×256 **0.573 ms/frame** (1,744 fps), 800×600 **4.16 ms/frame**
  (241 fps). A 5 s interval with 250 frames costs **39–115 ms** at small/medium
  ROI — three to four orders of magnitude of headroom. On a Pi 5 (~5–10× slower
  CPU) that is still well inside the budget; only *sustained full-frame 800×600
  classical CV* approaches it.
- **USB3 bandwidth.** ASI174MM mono 8-bit: 800×600@250 fps = 120 MB/s,
  1936×1216@120 fps = 282 MB/s — under the ~500 MB/s practical USB3 ceiling. A
  single camera never saturates the Pi 5's USB3 controller; the limit is sensor
  readout/exposure. (Two cameras at high fps *do* saturate one controller — that
  is the DE-only reason to move to a NUC, see architecture.md.)

**Where horsepower would actually be needed:** not model inference, and not
classical CV at sane ROI — but a *learned high-resolution detector* replacing
classical CV. Andrea's YOLOv8-on-MPS ceiling (~170 fps, unbatchable, sequential
per-frame overhead) is a symptom of that path, and it is the only path where a
Jetson earns its price. So:

- **Jetson Orin Nano — fallback, live only if:** classical CV cannot extract
  diameter/frequency reliably (low index contrast, dense/overlapping droplets)
  and we must switch the measurement path to a learned detector at high
  resolution. **Concrete trigger:** CV or detector p95 latency exceeds 40 % of
  the control interval on the Pi 5 at the required ROI/fps, *or* a use case needs
  sustained > 250 fps full-frame detection. Until that trigger fires, the Jetson
  is idle money.
- **FPGA — dead.** Its advantage is deterministic low-latency streaming/pipelined
  throughput. At a 0.1–10 Hz *control* rate with batched frame processing and
  sub-millisecond models, there is nothing to pipeline that isn't already
  real-time on a $80 board. The development cost (HDL/HLS, toolchain, iteration
  time on a 3-person team) is unjustified. Revisit only if a future product needs
  µs-scale per-frame closed-loop actuation — which this system, by its physics,
  does not.

Reproduce on his hardware: `python bench/benchmark.py` (auto-detects CoreML/
CUDA/CPU, degrades gracefully). See `docs/compute_recommendation.md` for the
named machine and price.

---

## 2. Syringe pumps vs pressure control

**Decision: assume Harvard Apparatus syringe pumps; Q_c and Q_d are the control
variables. Write the actuator abstraction so a pressure controller drops in
without touching controller code.** Implemented in `reference_stack/actuator.py`.

The two actuators differ in settling behaviour and compliance coupling, and the
taxonomy already treats flow rates as the control inputs, so syringe pumps are
the default and the entire dynamics/controller stack is written against flow-rate
commands. The abstraction is a base `Actuator` whose public API is
`command(setpoints)` in flow-rate units; `HarvardSyringePump` writes the ASCII
`irate/irun` protocol over `pyserial`, and `PressureController` is a subclass that
maps a commanded flow rate to a pressure via a device fluidic resistance
(`P = R·Q`) and enforces its bounds/slew in pressure units — **the controller
never learns which actuator it is driving.** Swapping is a one-line change at
construction; `test_pressure_controller_is_drop_in` asserts it.

Caveat recorded for later: pressure control changes the *plant*, not just the
actuator — settling and compliance coupling differ, so a dynamics model fit under
syringe pumps will not transfer unchanged. The abstraction makes the code swap
free; re-fitting the dynamics model is not, and is flagged in architecture.md.

---

## 3. World model vs the four named candidates

**Decision: a per-fluid-system linear dynamical system + MPC is the baseline and
we open with it. The latent world model is not justified at current data volume.**
Full argument in `docs/proposals/world_models.md`; adjudication summary here.

Data volume is real but so is identifiability. A per-fluid linear system is **70
parameters** (`models/profiling.py`); the latent world model is **5,095**. Both
run in < 0.1 ms/step, so compute does not decide this — sample complexity does. A
60 s startup run yields hundreds of (state, action, next-state) tuples, plenty to
fit A, B near steady state; far short of robustly identifying an encoder +
transition + decoder across regimes and fluids. The linear system + CEM-MPC
(`reference_stack/controller.py`) is the boring correct answer and is the bar the
world model — and the LSTM, neural ODE, and GP — must clear on real data before
they earn their complexity. The world model's *adopt* condition (strong transient
nonlinearity or multi-fluid transfer that the LDS provably misses, with ≥ 3 fluid
systems logged) and *abandon* condition (linear + MPC hits the control spec) are
stated in the proposal. This overrules the standing request to build a world
model, as the modeling lead explicitly invited.

---

## 4. Can the DAFD data test the normalization claim?

**Decision: no. The DAFD 3.0 release cannot test the cross-fluid W/ℓ\* claim.
Run the device matrix instead.** Full write-up in `normalization/RESULTS.md`;
verdict here.

Audited in `normalization/study.py` against `Comprehensive_normalized.xlsx` (868
rows). Three independent reasons it can't discriminate: (1) the release carries
**no interfacial-tension, no density, no absolute-viscosity** columns — ℓ\* is
simply not computable from it (Ca and λ leave one continuous-phase property
unidentified); (2) **54.6 % of rows are a single viscosity-ratio system**
(λ = 57.2), the other ten clusters thin and imbalanced, collapsing to ~5 distinct
*imputed* ℓ\* values — too few degrees of freedom; (3) within one fluid system
W/ℓ\* and W/W_or differ only by a per-system constant, so a grouped split can
attribute a change to ℓ\* only with many distinct **measured**-ℓ\* systems. The
numbers bear it out: identical models score R² ≈ 0.96 under a random split for
*both* normalizations, and under leave-one-fluid-out both collapse
(diameter R² 0.18 ± 0.67 for W_or vs −10.6 ± 19.8 for ℓ\*; the difference is
smaller than the noise). The remedy is `data/device_matrix.csv`: ≥ 3 fluid
systems with **measured** η_c, ρ_c, γ spanning ℓ\* over 0.13–149 µm, geometry
held fixed, grouped by fluid system.

One correction that must reach the paper's authors regardless (physics, not data):
the source document's identity "W/ℓ\* is equivalent to Oh²" is inverted — it is
**Oh⁻²** (asserted in `test_w_over_ell_is_oh_minus_2`), and the fluorocarbon
reference ℓ\* ≈ 14 µm is not self-consistent with HFE-7500's density; the correct
value is ≈ 5.3 µm (see `normalization/RESULTS.md` and `ASSUMPTIONS.md`).

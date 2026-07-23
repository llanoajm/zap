# Stack review checklist — Andrea's acquisition/inference implementation

Notes for whoever reviews Andrea's code once he shares it. This isn't a compliance gate — Andrea built his version fast and it works, and that's exactly what we wanted. This is "here's what I'd look at, roughly in order of how much it matters." Most of it is about the handful of places where a wrong call is expensive (destroyed chips, a flooded stage, a week lost to premature async) versus the many places where his choice is as good as mine.

The reference stack in `reference_stack/` is one concrete way to satisfy each of these. Cite it where it helps, but don't hold Andrea to the same structure — hold him to the same *properties*.

Ordered highest-impact first.

---

## 1. Safety — bounds, slew, watchdog, kill path

This is #1 by a wide margin. An actuator under model control with no bounds checking is the single highest-consequence failure in the whole system: the model asks for a number, the pump obeys, and the chip delaminates or the stage floods. Everything else on this list is about quality; this one is about not destroying hardware.

- [ ] **Hard flow/pressure bounds exist as non-overridable constants.** Not constructor args, not config, not "usually we set it low." In the reference stack these live in `actuator.py`: `FLOW_MAX_ULH = 5000` (above this, dead-volume transit + Ca pushes every device in the matrix past jetting), `PRESSURE_MAX_MBAR = 2000` (PDMS delamination margin). *Why it matters:* if a bound is overridable, it will get overridden at 2am during a debugging session and stay that way. *What good looks like:* the clamp is applied inside the actuator's command path, so it holds no matter what the controller emits.
- [ ] **Setpoint slew-rate limiting.** A step command from the model shouldn't hit the interface as a step. Reference: `FLOW_SLEW_MAX_ULH_PER_S = 2000`, enforced per-command against wall-clock dt (`actuator.py` `command()`). *Why:* protects the interface and the tubing from pressure transients the model never reasoned about. *What good looks like:* slew is enforced in the same place as the clamp, and the returned state flags when it engaged so it's visible in logs.
- [ ] **A watchdog that trips to a safe state.** Three trip conditions at minimum: frames stopped arriving (`max_frame_gap_s`), diameter went non-physical (`max_diameter_um = 1000`), regime left dripping (jetting). Reference: `safety.py` `Watchdog.check()`, which latches once tripped and requires an explicit `reset()`. *Why:* the model can't be trusted to notice it's driving blind or into a bad regime — the watchdog is the thing that notices for it. *What good looks like:* the watchdog is checked *before* actuation each step, it latches (doesn't silently un-trip), and the trip reason is recorded.
- [ ] **A kill path that bypasses slew.** `safe()` drives straight to `(0, 0)` immediately — you do not slew-limit your way out of an emergency. Reference: `actuator.py` `safe()`. *Why:* the whole point of a safe state is that it's reachable instantly.
- [ ] **Safe state is `(0, 0)` and the loop actually calls it on a trip.** Check the loop wiring, not just that the method exists — in `loop.py` the trip path calls `self.act.safe()` and breaks. A watchdog nobody consults is decoration.

If Andrea's version is missing any one of these, that's the first conversation to have, ahead of everything below.

---

## 2. Actuator abstraction — flow vs pressure behind one interface

- [ ] **The controller commands flow rates and never sees "pressure vs flow."** A pressure controller should drop in as a subclass without touching controller code. Reference: `actuator.py` — `PressureController` maps flow→pressure via a fluidic resistance `R` (P = R·Q), enforces bounds/slew *in pressure units*, and the controller is unchanged. *Why it matters:* we haven't committed to syringe pumps forever (decisions.md keeps pressure as a live option), and the cost of that swap should be one class, not a controller rewrite. *What good looks like:* `command(setpoints)` is the only entry point the controller uses; units, bounds, and hardware protocol all live below that line.
- [ ] **Mock and real actuators share the base class** so the safety machinery is inherited, not reimplemented per backend. In the reference `MockActuator`, `HarvardSyringePump`, and `PressureController` all subclass `Actuator` and only override `_write`.

---

## 3. Measurement correctness

The CV path is where a plausible-looking-but-wrong number does the most quiet damage, because a wrong measurement feeds a confident-but-wrong control action.

- [ ] **Pixel-to-micron calibration from a known reference distance.** Not a guessed scale. Reference: `measurement.py` `Calibration.from_reference(known_um, measured_px)` — image a stage micrometer or a channel of known width once. *Why:* every diameter downstream is only as trustworthy as this one number.
- [ ] **Diameter via threshold + contour with a circularity gate.** Otsu threshold → external contours → equivalent-circle diameter, *and* a circularity floor (reference uses `< 0.6` rejected) to drop channel walls and merged blobs. *Why:* without the circularity gate you measure the channel edge and call it a giant droplet. *What good looks like:* a minimum-area filter plus the circularity gate, so only round-ish drops count.
- [ ] **Frequency via a robust method, not per-droplet tracking.** Reference: FFT of the mean intensity in a 1-px downstream column (`frequency_from_batch`) — a droplet passing modulates brightness periodically. *Why:* per-droplet tracking is fragile (missed/merged detections wreck the count); an FFT of a downstream signal degrades gracefully. If Andrea tracks individual droplets, ask how it behaves when detection drops a few.
- [ ] **Polydispersity as CV over many droplets/frames.** std/mean of per-droplet diameters. Reference computes it across the batch (`measure_batch`, `diameter_stride`). *Why:* CV is a state variable the controller and convergence detector both use — it needs to be a real spread, not a single-frame artifact.
- [ ] **No disk I/O in the measurement path.** Frames stay as in-memory numpy; nothing writes PNGs or shells out to ffmpeg. *Why:* this is the real-time-critical step (see §6) — a file write in the inner loop silently blows the interval budget. The reference module docstring is explicit that it discarded uDROP's ffmpeg/GUI architecture for exactly this.

---

## 4. Loop architecture — synchronous first, async only when profiled

- [ ] **The default path is synchronous and reads top-to-bottom:** capture N frames → measure (D, F, regime) → controller → actuate → wait out the interval. One thread, deterministic, inspectable. Reference: `loop.py` `SynchronousLoop.run()`. *Why it matters:* at 0.1–10 Hz control you have seconds of budget; a synchronous loop is trivially fast enough and you can actually reason about what happened on step 12.
- [ ] **Async is behind a flag, not the default, and justified by profiling — not by taste.** The bar is: async only when CV time approaches the interval budget. Reference keeps `AsynchronousLoop` present but the docstring is explicit that `multiprocessing.Queue` + shared memory is "noted, not implemented, because it is premature here." *Why:* premature async buys nondeterminism and race conditions in exchange for headroom you already had. If Andrea went async by default, the question isn't "is it wrong" — it's "did the profile say you needed it." (Almost certainly not yet.)
- [ ] **If async is used, `queue.Queue` is the right first tool.** cv2 and torch release the GIL, so threads genuinely overlap. `multiprocessing.Queue` + shared-memory frames is the *next* step, only if queue contention is measured to be real (it becomes real with a second camera — the DE setup — not with one). Don't let a reach for multiprocessing go unquestioned when a thread queue would do.

---

## 5. Control rate sanity

- [ ] **Control interval sits in 0.1–10 Hz and is understood as settling-limited.** The plant settles on the order of pump ramp + tubing compliance (2–20 s) + dead volume. Reference default: `control_interval_s = 5.0`. *Why it matters:* commanding faster than ~1 Hz is pointless — you're issuing new commands before the chip has responded to the last one, so you're controlling noise. If Andrea is running the loop at 20 Hz, that's not "more responsive," it's wasted actuation.
- [ ] **Frames are batched between control decisions.** 250 frames at 50 fps = 5 s of observation per control step (reference `frames_per_step = 250`). *Why:* one control decision should rest on a stable measurement over the interval, not a single frame. This also gives the FFT enough samples to resolve frequency.

---

## 6. Compute headroom — profile the CV path, not the model

- [ ] **The thing profiled against the interval budget is the CV path, not model inference.** Per the benchmark (FACTS.md): every dynamics/classifier model is sub-millisecond per call (regime classifier 0.167 ms/frame, tiny detector 0.142, LSTM 0.048, world model 0.031). The models are trivially fast. The real cost is classical CV, and it's resolution-driven: 128×128 is 0.155 ms/frame, 256×256 is 0.573 ms/frame, but 800×600 full-frame is 4.156 ms/frame. *Why it matters:* if Andrea is sizing a Jetson/FPGA purchase off model FLOPs, he's optimizing the cheap part. The buy should be justified against the CV path at his chosen ROI and fps.
- [ ] **CV budget is checked at the actual ROI/fps, on the target machine, before hardware is bought.** At a 5 s interval, 250 frames cost 39–115 ms at small/medium ROI — a rounding error against 5000 ms, even on a Pi5 (~5–10× slower CPU). Only sustained high-res (800×600) full-frame CV approaches the budget on a Pi5. *What good looks like:* Andrea runs `bench/benchmark.py` on the Pi5 at his real ROI and confirms CV p95 is well under the interval. The Jetson only becomes live if CV p95 > 40% of the interval on the Pi5, or a learned high-res detector replaces classical CV (decisions.md trigger). FPGA is dead — no batching benefit at this control rate.

---

## 7. Regime handling gates control

- [ ] **A regime estimator supplies qualitative state the dynamics model can't.** dripping / jetting / no_droplets / satellite. A frame-diff heuristic is a legitimate first backend; a CNN is the upgrade if the heuristic isn't discriminative. Reference: `regime.py` — `HeuristicRegime` (frame-differencing + periodicity, zero training) and `CNNRegime` behind one `classify(frames) -> (label, conf)` interface. *Why:* the dynamics model predicts continuous state; it can't tell you "we've fallen out of dripping." That's categorical information only the classifier has.
- [ ] **Regime actually gates control actions.** Not just logged — consulted. In the reference, jetting is a watchdog trip condition (§1) and the MPC carries a jetting/Ca constraint as a cost penalty (`controller.py` `SamplingMPC`, `ca_jetting_limit`). *Why:* a regime signal that doesn't change what the controller does is a dashboard, not a safety input. Check that leaving dripping has a consequence.

---

## 8. Testability without hardware

- [ ] **The whole stack runs green with a mock camera + mock pump.** Reference: `MockCamera` serves synthetic dripping frames and `set_state()` lets the emulated chip respond to commands, so a closed-loop test genuinely converges; `MockActuator` records every write so tests can assert on command history. FACTS notes 21/21 tests green and the mock loop converging D=100µm/F=40Hz in 7 steps. *Why it matters:* if it needs a chip on the stage to run, nobody runs it, and the safety logic in §1 goes untested until the day it's supposed to save a device. *What good looks like:* `pytest` passes on a laptop with no camera, no pump, and hardware imports deferred (reference defers `zwoasi`, `pyserial`, `torch` so the modules load anywhere).

---

## 9. Reproducibility / data logging

- [ ] **Every startup run logs (state, action, next-state) tuples.** Reference: `loop.py` `StepLog` captures `t, state, command, converged, tripped, reason` per step. *Why it matters:* this log *is* the training set for the dynamics model — the per-fluid linear model (and anything richer later) is fit from exactly these transitions. Cold-start runs are the data. *What good looks like:* the schema matches the OBS/ACT layout the models expect (OBS(7)=[D, log10F, CV, regime×4], ACT(2)=[Q_c, Q_d]) so logs are training-ready without a translation step.
- [ ] **Transients are logged, not discarded.** The settling behavior between steps is the most informative part of the trajectory for a dynamics model — it's where the time constants live. If Andrea only logs converged endpoints, he's throwing away the signal. *Why:* a model fit only on steady states can't predict the approach, and the MPC rolls forward through exactly that approach.

---

### Quick triage if you're short on time

Read §1 first and in full — that's the one where a gap is expensive and urgent. Then §3 (a wrong measurement quietly corrupts everything downstream) and §6 (so the hardware purchase is aimed at the real bottleneck). The rest are quality-of-life and can be raised as suggestions rather than blockers.

# Latent world model vs. the four baselines: when (and only when) it earns its place

For: Bernardo (fab), Andrea (acquisition/inference), Antonio (modeling).
Scope: the dynamics model that the MPC rolls forward to plan pump commands during
cold-start. Not the regime classifier, not the CV measurement path.

## The verdict, stated first

At the data volume we actually have, the world model loses to the per-fluid linear
dynamical system + CEM-MPC. That is the recommendation. Antonio said it plainly —
"do not build a world model because I asked for it" — and this document takes that
literally. The linear system is 70 parameters (a 7×7 `A`, a 7×2 `B` plus bias) fit
per fluid system; near steady state, where the loop spends almost all of its time,
that is very likely enough to plan good pump moves. It is the boring, correct
baseline, and it must be beaten on real startup data before anything with an encoder
earns a place in the loop. Everything below is the argument for *what would have to
be true* for the world model to win, and the explicit test that decides it — not a
case for building it now.

Compute is not the tie-breaker and should be taken off the table immediately. On the
x86 CPU bench this session (`bench/results.json`, batch-1 p50): linear 0.008 ms/step,
world model 0.031 ms/step, neural ODE 0.073 ms/step, LSTM 0.048 ms/step. Against a
control interval of 0.1–10 Hz these are all rounding error, and the CEM-MPC samples a
few hundred constant-hold trajectories over a horizon of ~6, which even at the world
model's per-step cost is well under a millisecond of planning per control step. The
Pi5 is ~5–10× slower and it still does not matter. So the decision is about **data and
identifiability**, not FLOPs or latency. Anyone arguing the world model on "it's still
fast enough" is arguing the wrong axis. (Pi5/CoreML numbers pending: `python
bench/benchmark.py` on the target.)

## Why additive-noise latent dynamics are the *right shape* for this plant, if any learned model is

The physics here is unusually friendly to a simple transition. Droplet formation at a
flow-focusing junction, once in dripping, evolves under approximately additive process
noise: the diameter and generation frequency drift and relax rather than jump, and the
disturbances (small pressure ripples, thermal drift, surfactant transients) enter as
roughly additive perturbations on a slowly-moving operating point. Pump settling is
slow and smooth — 5–30 s to settle a setpoint change, which is *longer than one
control step* — so within an MPC horizon a commanded flow rate is effectively held and
the state relaxes toward a new steady value. This is exactly the regime a first-order,
additive-update model is built for.

That is why the candidate is an *additive-noise* latent model and not a general RNN.
Look at the transition in `models/architectures.py`: `z' = z + transition(z, u)`. The
latent is nudged, not recomputed, each step. This matches the plant's own relaxation
behavior and keeps the learned dynamics close to the identity — the correct inductive
bias when the true system is a slow settling process. The `SurrogateDynamics` we use
for the mock loop encodes the same intuition explicitly (first-order relaxation toward
`steady_state` with time constant τ). A world model that is worth having is one that
learns that relaxation plus the corrections a fixed linear `A,B` cannot represent —
not one that throws the structure away.

Contrast the other three learned candidates on shape alone. The **LSTM** (19,655
params) is a general recurrent function with no additive/settling prior; it is the
most flexible and the least identifiable from our data, and it buys flexibility we have
no evidence we need. The **neural ODE** (5,255 params, RK4 4-step) assumes smooth
continuous-time dynamics, which is fine, but it models the *observable* state directly
and gives us no latent to absorb the hidden fluid parameters — it is a fancier way to
fit the same 7-dim map the linear system fits, at 75× the parameters. The **GP over
trajectories** gives calibrated uncertainty, which is genuinely attractive for a
safety-gated loop, but it scales poorly with trajectory count, has no natural notion of
a persistent per-fluid latent, and is awkward to roll forward inside a sampling MPC
that wants a cheap deterministic `predict(obs, action)`. If we are going to pay for a
learned model at all, the additive-noise latent is the one whose structure matches the
plant.

## What the latent is *for*: absorbing the parameters we cannot measure in situ

The normalization taxonomy already told us which quantities govern droplet behavior and
cannot be read off a monochrome brightfield frame in the loop: static contact angle θ,
the *dynamic* interfacial tension γ(t) (surfactant is still migrating to the fresh
interface during startup — this is time-varying, not a constant), PDMS channel
compliance (the chip deforms under pressure and relaxes), and the fluid settling time τ.
None of these are in the observation. The OBS vector is only
`[D_um, log10_F, CV, regime_onehot×4]` (7 dims) and the action is `[Q_c, Q_d]` (2 dims).

That gap is the entire case for a latent. The per-fluid linear system handles the gap
by *refitting* — you get a new `A,B` per fluid system, and the hidden parameters are
baked into those 70 numbers as constants. That works precisely as long as they behave
like constants over the window that matters. The world model's job is different: encode
a 16-dim latent `z` from the recent window (16 steps of obs+action) that captures the
current value of these hidden parameters, including their *drift* during startup — γ(t)
falling as surfactant loads the interface, compliance relaxing as pressure settles —
and carry that state forward. When those effects are genuinely dynamic within a run,
constants cannot represent them and a latent can. When they are effectively constant
over a run, the latent is 5,095 parameters spent to relearn what 70 already captured.

## How actions enter and how it feeds the MPC

Cleanly, and identically to the other candidates, which is the point of keeping the
controller model-agnostic (`SamplingMPC` in `reference_stack/controller.py` calls
`dyn.predict(obs, action)` and does not care what `dyn` is). Pump setpoints `(Q_c, Q_d)`
are the action; they enter the transition directly — `transition(cat[z, u])` — so a
candidate flow command perturbs the latent's evolution. The MPC does the rest: CEM
samples ~200 constant-hold `(Q_c, Q_d)` commands, rolls each forward over the horizon
under the model, scores diameter + log-frequency tracking with a discount, keeps the
elite fraction, refits a Gaussian, repeats a few iterations, and applies the elite mean
(receding horizon). Constant-hold is not a simplification we're apologizing for — it
matches the plant, because pump settling is slower than a step.

The specific reason a world model is *architecturally* nice here: it can roll the
latent forward under a candidate action sequence **without decoding to observation
space every step**. `forward` returns `(next_obs, z')`; the MPC only needs to decode at
the horizon steps it scores, and can advance `z' = z + g(z,u)` cheaply in between. That
is a real efficiency for long horizons or dense sampling. It does not change the
verdict — we established compute is not the binding constraint — but it is the honest
"nice-to-have" and it belongs in the record.

## The crux: data volume vs. identifiability

Here is where the world model actually loses today. Every cold-start run yields
hundreds of `(state, action, next-state)` tuples — good, but that is hundreds *per run*,
and they are heavily autocorrelated and concentrated near the trajectory the controller
happened to drive. To identify a per-fluid linear `A,B` you need to excite 70
parameters near steady state; a handful of startup runs with any setpoint variety does
that, and the fit is well-posed because the model is small and near-identity. To
identify an encoder + transition + decoder — 5,095 parameters, with a 16-dim latent
that is only weakly pinned by a 7-dim reconstruction target — you need far more, and in
particular you need data that *varies the hidden parameters*, i.e. multiple fluid
systems and multiple runs per system, or the latent has nothing to encode and collapses
to a constant. Fitting 5,095 parameters to absorb θ/γ(t)/compliance/τ from the logs of
one or two fluid systems is not identification, it is overfitting with extra steps, and
it will generalize worse than the 70-parameter linear fit it is trying to beat. The
parameter counts (70 vs. 5,095) are not a compute statement — they are a *sample-
complexity* statement. That ratio is the whole argument.

## When the world model actually wins

Three conditions, and it needs at least one to hold on real data — not on intuition:

1. **Strong transient nonlinearity across regimes.** If startup routinely crosses
   regime boundaries (squeezing → dripping, dripping-near-jetting) and the transient is
   sharply nonlinear, a single per-fluid `A,B` linearized near steady state will
   mispredict the transient and the MPC will overshoot or take a bad path. A learned
   nonlinear transition can capture that. Test is empirical: does linear-MPC show
   transient overshoot on real startups that a nonlinear model removes.

2. **Multi-fluid transfer.** If we log enough fluid systems (the device matrix already
   points at ≥3 with measured η_c, ρ_c, γ, and we have 8 fabricable devices), a *shared*
   latent world model can amortize across fluid systems — encode the fluid from its
   recent window instead of refitting 70 numbers from scratch per fluid. That is the one
   thing the per-fluid LDS structurally cannot do: it has no mechanism to transfer, by
   construction there is a separate `A,B` per fluid. This is the strongest long-term
   argument and it is exactly what the double-emulsion / many-fluid future would want.

3. **Genuine partial observability.** If θ, γ(t), or compliance move enough *within* a
   run that treating them as per-run constants demonstrably hurts prediction, the latent
   earns its keep by tracking them. γ(t) during surfactant loading is the most likely
   real instance.

None of these is established. Each is a measurement, and the measurement is the device
matrix plus logged startup trajectories, not a whiteboard argument.

## Adopt / abandon

Decide against the linear baseline's actual numbers on real hardware, not the mock loop
(the mock converges D=100 µm / F=40 Hz in 7 steps with 21/21 tests green, but that is a
perfect-model result — plant and surrogate share one `steady_state` map, so it tests
loop mechanics, not model fidelity).

**ADOPT the world model only if BOTH hold:**
- On logged real startup data, per-fluid linear + CEM-MPC exceeds the control spec —
  either steady-state tracking error or transient overshoot past the agreed bar (bar
  pending: set it with Antonio against the spec, e.g. steady-state |ΔD|/D and |Δlog F|,
  plus a peak-overshoot ceiling) — AND that failure is attributable to model error, not
  actuator/measurement error; **and**
- We have logged ≥3 fluid systems (measured η_c, ρ_c, γ) with multiple runs each, so a
  shared latent is actually identifiable and transfer can be tested.
  (Data: fabricate from `data/device_matrix.csv`; log startups; then
  `python bench/benchmark.py` for the on-target latency check and a held-out
  leave-one-fluid-out prediction comparison of linear vs. world model.)

**ABANDON the world model if:**
- Per-fluid linear + CEM-MPC meets the control spec on real startup data. If the boring
  baseline hits target, there is nothing to beat, and the world model is 5,095
  parameters solving a problem we do not have. Ship the linear system, revisit only if
  the multi-fluid transfer case (condition 2 above) becomes a live requirement.

Everything is exportable and swappable either way — `controller.py` takes any object
with `predict(obs, action)`, so this is a data question we can defer without a code
rewrite, not a fork we have to commit to now.

## Decision

Build and ship per-fluid linear + CEM-MPC as the baseline. Hold the world model until
it beats that baseline on logged multi-fluid startup data by the stated bar; if the
linear baseline meets spec, drop it.

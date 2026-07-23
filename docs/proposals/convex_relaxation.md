# Where (and whether) convex relaxation earns its place

**Scope.** Antonio, for Bernardo and Andrea. This adjudicates whether convex
relaxation is a tool we should be reaching for at the current stage of the
control stack, and if not, what would have to change for it to belong. It is a
scoping memo, not a result. No empirical numbers here are new; the control-stack
facts are from `reference_stack/controller.py`, `reference_stack/actuator.py`,
and `docs/decisions.md`.

## Verdict up front

At the current stage — a per-fluid **linear** dynamics model, **box** constraints
on `(Q_c, Q_d)`, and a CEM/QP MPC running at 0.1–10 Hz on models of a few
hundred to a few thousand parameters — **convex relaxation does not earn its
place.** There is nothing to relax. The finite-horizon control problem is
*either already a convex quadratic program, or* small enough that the
derivative-free CEM sampler in `SamplingMPC` solves it comfortably inside a
control interval. Reaching for an SDP or a moment relaxation today would be
solving a problem we do not have.

That is a statement about *now*, and it is contingent. Below I show precisely why
the current problem is convex-or-trivial, then name the three specific ways the
problem could change so that a convex relaxation would genuinely buy us
something, and which convex tool fits each.

## Why the current MPC is already convex (or trivially solvable)

Take the default configuration: the `LinearDynamics` candidate (70 params, an
affine map) fit per fluid system,

$$x_{t+1} = A x_t + B u_t + c, \qquad u_t = (Q_c, Q_d)_t,$$

with state `x` in the OBS layout `[D, log10 F, CV, regime one-hot×4]` and control
`u` the two flow rates. The actuator (`actuator.py`) imposes **hard box
constraints** `u_lo ≤ u_t ≤ u_hi` (`FLOW_MIN…FLOW_MAX`), plus a linear slew bound
`|u_t − u_{t−1}| ≤ Δ_max`. The tracking objective is quadratic in the deviation
from target.

Roll the linear dynamics forward over horizon `N` and stack the controls into
one vector `U = [u_0; …; u_{N-1}]`. Because the dynamics are affine, the state
trajectory is affine in `U`:

$$X = \mathbf{F}\,x_0 + \mathbf{G}\,U + \mathbf{c}.$$

Substituting into the quadratic cost gives, up to constants,

$$\min_{U}\ \tfrac12 U^\top H U + g^\top U
\quad\text{s.t.}\quad u_{\text{lo}} \le u_t \le u_{\text{hi}},\ \ |u_t-u_{t-1}|\le\Delta_{\max},$$

with $H = \mathbf{G}^\top \bar{Q}\,\mathbf{G} + \bar{R} \succeq 0$. `H` is positive
semidefinite by construction; the constraints are a box and a difference bound,
both polyhedral. **This is a convex QP.** It is *convex optimization*, not convex
relaxation — there is no nonconvex problem sitting underneath that we are
approximating. An off-the-shelf QP solver returns the global optimum; nothing is
being relaxed.

And we do not even need the QP: `SamplingMPC` is a cross-entropy sampler over
two scalars with a short horizon. Two decision variables, box-bounded, cost
evaluated by a handful of matrix-vector products — CEM finds the basin in a few
iterations and, per `decisions.md`, the loop is settling-limited (pump settling
5–30 s) not compute-limited. Inference is sub-millisecond. There is no real-time
pressure that a convex solver would relieve.

So both readings of the current problem — "solve the QP exactly" and "sample it
with CEM" — leave convex relaxation with no job.

## What would have to be true for relaxation to belong

Relaxation earns its place only when a **nonconvex** feasible set or a
**nonconvex** map enters the problem. Three concrete ways that happens here.

### (a) The dripping→jetting regime constraint becomes hard *and* its nonconvexity bites

Today the regime constraint is a **soft penalty** inside the CEM cost, backed by
the hard watchdog that trips to `(0,0)` on detected jetting. That is fine as long
as we tolerate the occasional excursion and let the watchdog catch it. The moment
we want the optimizer to *guarantee* the planned trajectory stays in dripping —
i.e. enforce `Ca < Ca_crit` as a hard constraint rather than a penalty — the
geometry of that constraint in the actuated variables matters.

The subtlety, stated honestly: in its **simplest** form the constraint is convex.
The continuous-phase capillary number is `Ca = η_c U_c / γ` with
`U_c = Q_c / A_c`, so `Ca ∝ Q_c`, and `Ca_crit ~ Oh^{-1}` is a **design-time
constant** per device (`Oh = η_c/√(ρ_c γ W_or)` does not depend on flow). Read
this way, `Ca < Ca_crit` is a single linear half-space `Q_c < κ` — convex — and
the constrained MPC is *still* a convex QP with one extra linear inequality.
Relaxation still would not be needed.

The nonconvexity appears only when the transition boundary genuinely **curves in
the `(Q_c, Q_d)` plane** — which the regime maps do (Utada 2007; DAFD's regime
classifier draws a curved dripping/jetting line, not a vertical one). Physically
that is because the transition depends on both phases: a dispersed-phase Weber
criterion `We_d ∝ Q_d^2` and the continuous-phase `Ca ∝ Q_c` together, and the
critical value depends on the flow **ratio** `φ = Q_d/Q_c`. A boundary of the
natural monomial form

$$Q_c^{\alpha}\,Q_d^{\beta} \le \kappa, \qquad \alpha<0<\beta$$

(mixed-sign exponents = ratio dependence) is **nonconvex** in the native
`(Q_c, Q_d)` coordinates, and the *dripping band* — bounded below by squeezing and
above by jetting — is a nonconvex set even when each boundary is smooth.

This is exactly where a convex tool earns its keep, and it is a well-behaved
case: a single such monomial/signomial constraint is convex under the log change
of variables `ξ = log Q` (geometric programming), and the quadratic
`We_d ∝ Q_d^2 ≤ b` is a **second-order cone** constraint directly. **Tool: SOCP**
(or a GP reformulation) for the Ca/Ohnesorge regime constraint, layered onto the
QP. This is the first place I would actually reach for convex machinery — but
only *after* we decide the penalty-plus-watchdog scheme is insufficient and the
fitted boundary is measurably curved in the actuated coordinates.

### (b) Nonlinear dynamics, or inverse geometry design over a nonconvex model

Two flavors, same root cause — a nonconvex map.

*Nonlinear dynamics.* If we swap `LinearDynamics` for the LSTM, neural-ODE, or
latent world model, the rollout equality `x_{t+1} = f(x_t, u_t)` is no longer
affine, and the MPC ceases to be a QP. `decisions.md` already parks these models
("world model not justified at current data volume"), so this is hypothetical —
but if it happens, the exact QP is gone and one either stays derivative-free
(CEM, what we do) or relaxes the nonconvex program. Given two control variables
and a short horizon, CEM remains the pragmatic answer; a relaxation would only be
worth it if we needed a certified global optimum, which routes back to (c).

*Inverse geometry design.* The more defensible future use. Given a target
`(D*, F*)`, find geometry `g` (orifice width, height, …) **and** flow rates `u`
such that a surrogate `M(g, u) = (D*, F*)`. This is the DAFD inverse problem, and
`M` is a nonconvex regressor (gradient-boosted trees / NN). Inverting it by
relaxation is legitimate: **McCormick envelopes** for the bilinear/product terms
that show up when geometry and flow multiply, or a **moment/SOS (Lasserre)
relaxation** if we fit a low-degree polynomial surrogate, giving a certified
global inverse or a certificate of infeasibility. Worth noting the honest
counterweight: our near-term inverse problem is over the **8-device matrix**
(`data/device_matrix.csv`) — a tiny discrete set. Brute-force enumeration over 8
geometries with a local flow optimization inside each dominates any relaxation on
effort and risk. Relaxation for inverse design earns its place when the geometry
space becomes **continuous and high-dimensional** (co-designing a chip rather than
picking from a fabricated set), not before.

### (c) Formal stability / constraint-satisfaction certificates

If we ever need to *prove* — for a regulatory story, or to protect an expensive
chip — that the linear-model MPC keeps the state inside the dripping polytope
under **bounded model error**, that is a certification problem, not an
optimization problem. For the linear model `x_{t+1} = A x_t + B u_t + w`,
`‖w‖ ≤ ρ`, the standard route is an **LMI/SDP**: find a Lyapunov matrix `P ≻ 0`
and an invariant ellipsoid such that the dripping constraints hold for all
admissible `w` under the control law. **Tool: SDP/LMI** for robust invariance and
constraint-satisfaction certificates.

At a 3-person research stage with a hard watchdog already tripping to a safe
state, we do not need a formal certificate — the watchdog *is* our safety
argument. This becomes relevant only if "the watchdog occasionally fires" stops
being an acceptable answer (device-protection SLA, or a publication that claims
guaranteed constraint satisfaction).

## Summary

| Framing | Convex tool | Belongs now? | Trigger to revisit |
|---|---|---|---|
| Linear-model box-constrained MPC | (convex QP — no relaxation) | Already convex / CEM suffices | — |
| Hard `Ca < Ca_crit` regime constraint | SOCP / GP | No | Penalty+watchdog proves insufficient **and** fitted boundary curves in `(Q_c,Q_d)` |
| Nonlinear dynamics MPC | relaxation vs. CEM | No | We adopt LSTM/world model **and** need a global optimum |
| Inverse geometry design | McCormick / moment (SOS) | No | Continuous high-dim geometry co-design (beyond the 8-device set) |
| Stability / safety certificate | SDP / LMI | No | Formal guarantee required (regulatory / device-protection) |

The one-line takeaway: **the control problem we have is convex or trivial;
relaxation is a tool for problems we have chosen not to have yet.** The first of
those to actually arrive will most likely be the hard, curved regime constraint —
so if we want to pre-invest, an SOCP formulation of the Ca/Ohnesorge boundary is
the thing to prototype, and only once the fitted dripping boundary is shown to
curve in the actuated coordinates.

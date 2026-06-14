# JEPA World Models as AC/DC-OPF Surrogates and Latent-Space Planners
## A Mathematical Design Document

**Author:** Ralph deep-research loop + walkthrough
**Repo:** `llanoajm/zap`, branch `claude/leworld-lejpa-acpf-surrogate-aj3ew0`
**Date:** 2026-06-14
**Status of empirical work:** *No model has been trained on any data.* See [§8](#8-honest-status-what-is-real-vs-designed). The only code executed is a small **diagnostic** experiment that calls the real `zap` solver on a hand-built 6-bus network.

---

## 0. What this document is

This collects, in one place and with the math spelled out, everything produced in this session:

1. The **Ralph research loop** scaffolding (`ralph/PROMPT.md`, `ralph/ralph.sh`) and what it produced (`REPORT.md`, `NOTES.md`, `SOURCES.md`, `TODO.md`).
2. The **mathematical objects** the designs rest on: `zap`'s DC-OPF as a linear program, its KKT system, the dual variables (LMPs), and the implicit-differentiation backward pass that makes `zap` a differentiable layer.
3. The **multi-value / multi-period expansion planning** problem (the Degleris et al. formulation `zap` implements).
4. The **JEPA / LeJEPA** mathematics (SIGReg, the Epps–Pulley statistic, the Cramér–Wold argument).
5. The **three hybrid designs**, with Design A worked out in full, mapped onto concrete `zap` code interfaces.
6. The **LP piecewise-constant-gradient theorem** and the diagnostic experiment that empirically confirms it (with the actual numbers reproduced today).
7. An explicit **honesty section** on what exists vs. what is a proposal.

Notation is standard: vectors lowercase bold-free (`x`), matrices uppercase (`A`), `⊙` Hadamard product, `diag(·)` diagonal matrix, `[·]_+` projection onto nonnegatives.

---

## 1. The problem we are trying to solve

You want one artifact with two properties that today live in different systems:

- **Foundation-model generality & speed** — like Microsoft's **GridSFM** (May 2026): one model that ingests *many* grid topologies, loads, and contingencies and returns an OPF answer fast. GridSFM reports a **2.23 % median cost gap** vs. IPOPT and a **1.66× warm-start speedup**, but it is a *surrogate*: it does not emit trustworthy dual variables.
- **Planning capability & exact economics** — like the **`zap`** stack (Degleris, El Gamal, Rajagopal): GPU-accelerated, **differentiable** DC-OPF that yields **exact dual variables** (locational marginal prices) and supports **gradient-based multi-value expansion planning**. But `zap` is specialized — each network is an explicitly built model, not a learned, transferable representation.

The thesis under investigation: **put a JEPA encoder on top of `zap`'s differentiable optimization layer** so the encoder supplies generality and the layer supplies exactness. The rest of this document makes "on top of" precise.

---

## 2. `zap`'s DC-OPF as a linear program

### 2.1 Variables and data

Consider a network with `n` buses (nodes), a set of devices `D` (generators, loads, AC lines, batteries, grounds), and a time horizon `T`. For the single-period DC case the decision variables are:

- `p` — per-device terminal **power** injections,
- `θ ∈ ℝ^{n}` — bus voltage **angles** (global angle variable in `zap`),
- local device variables (e.g. battery state of charge `s`).

Generators have linear cost `c` and capacity limits; lines have susceptance `b` and thermal ratings `f̄`; loads are fixed demand `ℓ`.

### 2.2 The optimization

DC-OPF is the **linear program**

$$
\begin{aligned}
\min_{p,\;\theta}\quad & c^\top p \\
\text{s.t.}\quad
& A p = 0 && \text{(nodal power balance at every bus)} &&\leftarrow \nu \\
& p_g \le \bar p_g,\;\; p_g \ge 0 && \text{(generator capacity)} &&\leftarrow \lambda^{\text{gen}} \\
& -\bar f \le B_\theta\, \theta \le \bar f && \text{(line thermal limits, DC flow } f = B_\theta\theta) &&\leftarrow \mu \\
& \theta_{\text{ref}} = 0 && \text{(slack/ground reference)} &&\leftarrow
\end{aligned}
$$

where `A` is the network incidence operator mapping device terminals to buses, and `B_θ` encodes the DC power-flow relation `flow = susceptance × angle difference`. The Greek letters on the right are the **dual variables** attached to each constraint — these are the economically meaningful outputs.

> **Important code fact.** Despite the name `ACLine`, `zap`'s line device implements **DC** power flow (`flow = B·Δθ`). So everything below is DC-OPF; AC is discussed only as a future extension (§7.4). This matches `zap/devices/transporter/ac_line.py`.

### 2.3 The dual variables = prices

The Lagrangian is

$$
\mathcal{L}(p,\theta,\nu,\lambda,\mu) = c^\top p + \nu^\top (A p) + (\lambda^{\text{gen}})^\top(p_g-\bar p_g) + \mu^\top(B_\theta\theta - \bar f) + \cdots
$$

- **`ν` (power-balance dual) is the Locational Marginal Price (LMP)** at each bus — the marginal cost of serving one more unit of load there. In `zap` this is `DispatchOutcome.prices` (`zap/network.py`, field `prices`, computed by `_kkt_power_balance`).
- **`μ` (line-limit dual) is the congestion price** — nonzero exactly when a line is at its thermal limit. In `zap` this surfaces as `phase_duals` and the line inequality duals.
- **`λ` (capacity duals)** are scarcity rents on generation.

`zap`'s `DispatchOutcome` carries all of them explicitly:

```python
@dataclass
class DispatchOutcome(Sequence):
    global_angle, power, angle, local_variables,
    prices,               # ν  — nodal LMPs        (n × T)
    phase_duals,          # μ  — line/phase duals
    local_equality_duals, # device equality duals
    local_inequality_duals
```

These duals are **exact** because they come from the KKT conditions of the actual LP solve, not from a regression.

### 2.4 KKT optimality conditions

At the optimum, the primal-dual point `z = (p,θ,s,ν,λ,μ)` satisfies the KKT system `K(z; ϑ) = 0` where `ϑ` are the **problem parameters** (costs, capacities, susceptances). Schematically:

$$
K(z;\vartheta)=
\begin{cases}
\nabla_p \mathcal{L} = c + A^\top\nu + \cdots = 0 & \text{(stationarity)}\\
\nabla_\theta \mathcal{L} = B_\theta^\top \mu + \cdots = 0 & \text{(stationarity)}\\
A p = 0 & \text{(primal feasibility, equalities)}\\
\lambda \ge 0,\; \mu \ge 0 & \text{(dual feasibility)}\\
\lambda \odot (p_g - \bar p_g)=0,\;\; \mu \odot (B_\theta\theta-\bar f)=0 & \text{(complementary slackness)}
\end{cases}
$$

The complementary-slackness lines are what make OPF combinatorial under the hood: each inequality is either **binding** (`active`, dual > 0) or **slack** (dual = 0). The set of binding constraints is the **active set** `𝒜`. This single fact drives the entire gradient analysis in §6.

---

## 3. `zap` as a differentiable layer (the implicit-function backward pass)

This is the mathematical heart of `zap` and the reason Design A can train end-to-end.

### 3.1 Forward

`DispatchLayer.forward(**ϑ)` (in `zap/layer.py`) solves the LP and returns `z⋆(ϑ)`, the optimal primal-dual point as a function of the parameters `ϑ`. The parameters are injected through a name map:

```python
# zap/layer.py  (verified)
parameter_names = {
    "gen_capacity":  (0, "nominal_capacity"),   # → Generator.nominal_capacity
    "gen_cost":      (0, "linear_cost"),         # → Generator.linear_cost
    "line_capacity": (2, "nominal_capacity"),    # → ACLine.nominal_capacity
}
def setup_parameters(self, **kwargs):
    assert kwargs.keys() == self.parameter_names.keys()
    parameters = [{} for _ in self.devices]
    for k, (i, name) in self.parameter_names.items():
        parameters[i][name] = kwargs[k]      # routes tensor onto device attribute
    return parameters
```

So **any upstream neural network that produces tensors named `gen_capacity`, `gen_cost`, `line_capacity` is differentiably wired into the OPF.** This is the socket Design A plugs into.

### 3.2 Backward via the implicit function theorem

Because `z⋆(ϑ)` is *defined implicitly* by `K(z⋆(ϑ); ϑ) = 0`, we differentiate that identity:

$$
\frac{\partial K}{\partial z}\,\frac{\partial z^\star}{\partial \vartheta} + \frac{\partial K}{\partial \vartheta} = 0
\quad\Longrightarrow\quad
\frac{\partial z^\star}{\partial \vartheta} = -\Big(\underbrace{\tfrac{\partial K}{\partial z}}_{J_z}\Big)^{-1}\,\underbrace{\tfrac{\partial K}{\partial \vartheta}}_{J_\vartheta}.
$$

We never form that Jacobian. For backprop we only need the **vector–Jacobian product** (VJP): given an upstream gradient `dz` on the outputs, the gradient on parameters is

$$
d\vartheta \;=\; -\,J_\vartheta^\top\,\big(J_z^{-\top}\,dz\big).
$$

This is **exactly** what `zap` implements:

```python
# zap/layer.py  (verified)
def backward(self, z, dz, regularize=1e-8, **kwargs):
    # dtheta = -JK_theta.T @ inv(JK_z.T) @ dz
    dz_bar = self.network.kkt_vjp_variables(dz, self.devices, z, ...)  # solve J_z^{-T} dz
    # ... then contract with J_theta^T to get parameter gradients
```

The comment `# dtheta = -JK_theta.T @ inv(JK_z.T) @ dz` is the implicit-function formula above, verbatim. `kkt_vjp_variables` solves the linear system `J_zᵀ x = dz`. This is **OptNet / `cvxpylayers`-style differentiable optimization** (Amos & Kolter 2017; Agrawal et al. 2019), specialized and GPU-vectorized for power networks.

**Consequence:** gradients of *any* downstream scalar (planning cost, emissions, a training loss) flow back through the exact optimizer to the parameters `ϑ` — and therefore, by the chain rule, into whatever neural net produced `ϑ`.

---

## 4. Multi-value, multi-period expansion planning (the Degleris formulation)

`zap`'s planning layer (`zap/planning/`) wraps the dispatch layer in an **outer investment problem**. Let `η` be **investment decisions** (line/generator capacity to build), with capital cost `I(η)`.

### 4.1 Bilevel structure

$$
\min_{\eta \ge 0}\;\; J(\eta) \;=\; I(\eta) \;+\; \sum_{\omega\in\Omega} \pi_\omega\, \underbrace{f_{\text{op}}\big(z^\star(\vartheta(\eta);\,\omega)\big)}_{\text{operating cost in scenario }\omega}
$$

where:
- inner problem: for each scenario `ω` (load/renewable realization, contingency), `z⋆` is the DC-OPF optimum from §2, with parameters `ϑ(η)` shifted by the investment;
- outer problem: choose `η` to minimize capital + expected operating cost;
- `π_ω` are scenario weights.

This is in `zap` as `PlanningProblemCVX(op_obj, inv_obj, layer, ...)` with `InvestmentObjective` (`I`) and `DispatchCostObjective`/operation objectives (`f_op`), exactly the objects the experiment in §6 instantiates.

### 4.2 "Multi-value" and the role of duals

**Multi-value** = the operating objective is a weighted combination of several societal values (cost, emissions, curtailment, reliability):

$$
f_{\text{op}} = \alpha_{\text{cost}}\, c^\top p + \alpha_{\text{CO}_2}\, e^\top p + \alpha_{\text{curtail}}\, (\text{curtailment}) + \cdots
$$

**Gradient method.** The outer gradient is obtained by backprop through the inner solve (§3):

$$
\nabla_\eta J = \nabla_\eta I + \sum_\omega \pi_\omega\,\Big(\tfrac{\partial \vartheta}{\partial \eta}\Big)^{\!\top}\Big(\tfrac{\partial z^\star}{\partial \vartheta}\Big)^{\!\top}\nabla_{z}\,f_{\text{op}}.
$$

The middle factor is precisely the implicit-function VJP of §3.2. And the **dual variables are the economic content of this gradient**: the marginal value of building a line is the congestion rent `μ` it relieves, integrated over scenarios. This is why "duals are non-optional" (the WARP result) — they *are* the investment signal, not a by-product.

---

## 5. The JEPA / LeJEPA mathematics

### 5.1 JEPA in one equation

A Joint-Embedding Predictive Architecture learns an encoder `f_θ` by predicting the **latent** representation of a target view `y` from a context view `x`, rather than reconstructing pixels/inputs:

$$
\min_{\theta,\phi}\;\; \big\| \,\mathrm{pred}_\phi\big(f_\theta(x)\big) - \mathrm{sg}\big(f_{\bar\theta}(y)\big)\,\big\|^2
$$

where `pred_φ` is a small predictor and `sg(·)` historically was a stop-gradient onto a target encoder `f_θ̄` (an EMA "teacher"). Predicting in latent space lets the model ignore unpredictable detail — the appeal for grids is that it can learn *what matters for dispatch* (congestion structure) without reconstructing every nodal time series.

### 5.2 The collapse problem and LeJEPA's fix

The trivial minimizer of the loss above is **collapse**: `f_θ ≡ const`. Prior JEPAs prevent it with heuristics (stop-gradient, EMA teacher, whitening, asymmetric augmentations). **LeJEPA** (Balestriero & LeCun, Nov 2025, arXiv:2511.08544) replaces all of them with a single principled regularizer.

**Theorem (informal, LeJEPA).** Among all embedding distributions, the **isotropic Gaussian** minimizes worst-case downstream prediction risk. So we should *regularize embeddings to be isotropic Gaussian.*

**SIGReg (Sketched Isotropic Gaussian Regularization).** Testing isotropic-Gaussianity in high dimension `d` directly is hard. LeJEPA uses the **Cramér–Wold theorem**: a distribution is isotropic Gaussian **iff every 1-D projection is standard Gaussian**. So:

1. Draw `M` random unit directions `{u_m}`.
2. Project embeddings: `h^{(m)} = Z u_m ∈ ℝ^N` (batch of `N`).
3. Test each `h^{(m)}` for standard-normality with a **differentiable** statistic `T`.

$$
\mathrm{SIGReg}(Z) \;=\; \frac{1}{M}\sum_{m=1}^{M} T\big(h^{(m)}\big).
$$

`T` is the **Epps–Pulley** statistic — an L²-distance between the empirical characteristic function `φ̂(t)` and the standard-normal one `e^{-t²/2}`:

$$
T \;=\; \int \big|\hat\varphi(t) - e^{-t^2/2}\big|^2\, w(t)\,dt,
\qquad \hat\varphi(t)=\frac1N\sum_{j} e^{\,i t\,h_j}.
$$

Unlike ECDF/Kolmogorov–Smirnov statistics, this is smooth in the samples, so it backprops. Total LeJEPA loss:

$$
\mathcal{L}_{\text{LeJEPA}} = \underbrace{\|\mathrm{pred}(f(x)) - f(y)\|^2}_{\text{prediction}} \;+\; \lambda\,\mathrm{SIGReg}(Z),
$$

**one** hyperparameter `λ`, no stop-gradient, no EMA. Cost is `O(N·M·k)`.

### 5.3 Why this matters for grids (and the open risk)

- **Benefit:** a clean, collapse-free objective that trains stably across architectures — including, in principle, a **graph** encoder for grids (Graph-JEPA, TS-JEPA exist).
- **Open risk (flagged, unresolved):** SIGReg assumes the `N` embeddings in a batch are **i.i.d.** draws to test Gaussianity. **Node embeddings within one grid are not i.i.d.** — they are correlated by the topology. Whether SIGReg's guarantee survives graph-structured, non-exchangeable embeddings is an unanswered theoretical question. This is the single biggest soundness gap for porting LeJEPA to power systems.

### 5.4 LeWorldModel — what it is and is not

**LeWorldModel** (Maes, LeCun, Balestriero et al., Mar 2026, arXiv:2603.19312) is the first JEPA world model trained **end-to-end from pixels** (loss = prediction + `λ`·SIGReg), 15M params, and it plans **48× faster than DINO-WM**. A follow-up (GC-IDM, May 2026) amortizes the planner for another 100–130×. **But:** it is *visual*, on continuous robotic actions, and **no JEPA planning paper handles hard constraints on the action space** — which is the defining feature of power dispatch (line limits, balance, SOC bounds). So LeWorldModel is a *proof that the JEPA-as-world-model recipe works*, not a drop-in grid planner. The grid adaptation must swap the pixel encoder for a graph encoder and solve the constraint problem — which is exactly what Designs A and C do by keeping a real solver in the loop.

---

## 6. The LP gradient theorem and the diagnostic experiment

This is the one place where code was actually run. It does **not** train anything; it stress-tests *whether a learned surrogate's gradients can be trusted* for planning.

### 6.1 The theorem

DC-OPF is an LP, so by parametric LP theory the optimal value `J(η)` is **piecewise linear and convex** in the parameters, and the **optimal dual/primal solution is piecewise constant** in `η`:

> **Within a fixed active set `𝒜`, the planning gradient `∇_η J` is exactly constant.** As `η` crosses an active-set boundary, the gradient **jumps discontinuously.**

Two corollaries that decide the whole architecture:

1. **LMP mean-squared error is the wrong accuracy metric.** A surrogate can have 5 % LMP MSE but the *correct* active set → **zero** gradient bias. Conversely <1 % LMP error with the *wrong* active set → wrong-direction gradient.
2. **The right metric is active-set prediction accuracy.** Smooth NN surrogates (ReLU/sigmoid) *interpolate across* the LP's discontinuities, so they are guaranteed to be wrong near every congestion boundary — exactly where investment decisions are made.

### 6.2 The experiment (`ralph/experiments/lmp_gradient_sensitivity.py`)

A **synthetic 6-bus network** is built with the *real* `zap` API (`PowerNetwork`, `Generator`, `Load`, `ACLine`, `Ground`, `PlanningProblemCVX`), with one tight "bottleneck" line whose `nominal_capacity` is the planning variable `η`. We sweep `η` and compare:

- the **exact** `zap` gradient `∇_η J` (via the §3 KKT backward pass), against
- a **finite-difference** gradient of a *smooth* approximation,

reporting cosine similarity and relative bias at each detected active-set boundary.

### 6.3 Results — reproduced today (2026-06-14)

Running `python ralph/experiments/lmp_gradient_sensitivity.py`:

**Experiment 1 — gradient discontinuity map** (60 points, all 60 solves valid):

| capacity scale | binding lines (active set) |
|---:|:---|
| 0.30 | `[1, 2, 5, 6]` |
| 0.55 | `[1, 2, 6]` |
| 0.58 | `[1, 2]` |
| 0.88 | `[1]` |
| 1.24 | `[]` (uncongested) |

- Active-set jump events (adjacent cosine < 0.99): **26**
- Distinct active sets: **5**
- LMPs are piecewise constant — only **5 distinct LMP vectors** across the whole sweep, e.g. `[15, 27.5, 90, 500, 40, 500]` when congested vs. `[40,40,40,40,40,40]` (uniform price) when uncongested.

**Experiment 2 — smooth-surrogate gradient bias.** The decisive boundary is the congestion→uncongested transition at **scale ≈ 1.22**:

$$
\text{cosine}(\nabla^{\text{surrogate}},\,\nabla^{\text{exact}}) = -0.9982,\qquad \text{relative bias} \approx 1.5\times10^{10}.
$$

A cosine of **−0.998 is a near-perfect sign flip**: the smooth surrogate points the optimizer in the *opposite* direction right where one constraint stops binding. (Caveat for honesty: many of the other "boundaries" the script lists at scale > 1.24 report cosine 0.0000 because both gradients are *zero* in the flat uncongested region — those are degenerate 0/0 detections, not real disagreements. The load-bearing result is the single −0.998 sign flip at the true congestion boundary.)

### 6.4 What the experiment proves for the three designs

- **Design B (pure latent surrogate planner):** vulnerable — its gradients will sign-flip at congestion boundaries unless it predicts active sets, not just values. Needs RAMBO-style boundary sampling (the second script, `rambo_boundary_sampling.py`) to even see those regimes in training data.
- **Designs A and C:** *immune by construction* — the **exact solver determines the active set**, so the encoder's prediction error only affects speed (extra ADMM iterations / a warmer start), never the correctness of the dispatch, duals, or planning gradient.

This is the quantitative argument for preferring Design A over Design B.

---

## 7. The three designs

All target **DC-OPF** (per §2.2). Each is "JEPA + `zap`" with a different division of labor.

### 7.1 Design C — JEPA warm-start for `zap`'s ADMM (most tractable)

The encoder predicts a **good initial point** `(p₀, ν₀, μ₀)` for `zap`'s ADMM/solver. The problem solved is unchanged; only convergence speed improves.

$$
z^\star = \texttt{zap.solve}\big(\vartheta;\ \text{init}=g_\psi(\text{grid})\big),\qquad
\text{train } g_\psi \text{ to minimize solver iterations.}
$$

- **Duals:** exact (the solver still runs to convergence).
- **Precedent:** WARP (arXiv:2605.05728) — full **primal+dual** warm start cut IPOPT iterations 76 %; **primal-only warm starts can diverge** the solver. So `g_ψ` must predict duals too.
- **Risk:** lowest. **Value:** real but bounded (a speed multiplier on an existing capability).

### 7.2 Design A — JEPA encoder + differentiable OPF head ("LatentOPF") (strongest)

The encoder produces a topology-general latent that a decoder turns into **effective OPF parameters** `ϑ` consumed by `zap`'s `DispatchLayer`. See §7.3 for the full math. This is the design that delivers generality *and* exact duals *and* end-to-end trainability.

### 7.3 Design A in full

**Pipeline.**

$$
\underbrace{\text{grid graph }G}_{\text{topology, loads, renewables}}
\;\xrightarrow{\;f_\theta\ (\text{HGNN, JEPA-pretrained})\;}\;
\underbrace{Z\in\mathbb{R}^{n\times d}}_{\text{per-node latents}}
\;\xrightarrow{\;g_\phi\ (\text{decoder})\;}\;
\underbrace{\vartheta}_{\text{OPF params}}
\;\xrightarrow{\;\texttt{zap DispatchLayer}\;}\;
\big(z^\star,\ \nu^\star,\ \mu^\star\big)
$$

**Stage 1–2: encoder.** `f_θ` is a **heterogeneous GNN** (node types: bus/gen/line), the same family as GridSFM and HH-MPNN (the latter already hits <1 % optimality gap across 14–2000 bus grids). It is **pretrained with the JEPA objective** of §5: mask an electrical zone / a post-contingency state and predict its latent from context, regularized by SIGReg. Variable graph size ⇒ this is where **compositional generalization** lives (new topology = new graph, latents adapt).

**Stage 3: decoder → `zap` socket.** `g_φ` emits exactly the tensors `DispatchLayer` expects (§3.1):

$$
g_\phi(Z) = \big(\underbrace{\bar p}_{\texttt{gen\_capacity}},\ \underbrace{c}_{\texttt{gen\_cost}},\ \underbrace{\bar f}_{\texttt{line\_capacity}}\big),
\qquad \text{each head } = \mathrm{softplus}(\text{MLP}(Z)).
$$

```python
class OPFParamDecoder(nn.Module):
    def forward(self, Z):
        return {
            "gen_capacity":  F.softplus(self.gen_cap_head(Z_gen)),
            "gen_cost":      F.softplus(self.gen_cost_head(Z_gen)),
            "line_capacity": F.softplus(self.line_cap_head(Z_line)),
        }
outcome = dispatch_layer(**OPFParamDecoder(encoder(G)))
lmps    = outcome.prices          # exact KKT duals  (n × T)
```

Loads `ℓ` stay **exogenous** (from the scenario), never decoded.

**Stage 4: the head returns exact duals.** `outcome.prices` (= `ν`) and `outcome.phase_duals` (= `μ`) come from the LP's KKT system (§2.4), so they are **exact regardless of encoder error**.

**End-to-end training objective.** Two phases:

$$
\underbrace{\min_{\theta}\ \mathcal{L}_{\text{LeJEPA}}}_{\text{pretrain encoder}}
\qquad\text{then}\qquad
\underbrace{\min_{\theta,\phi}\ \mathbb{E}_\omega\big[f_{\text{op}}(z^\star(\vartheta;\omega))\big]}_{\text{fine-tune through the solver (§3 backward)}}.
$$

The second phase backprops the **planning objective** through `zap`'s implicit-function backward pass into `g_φ` and `f_θ`. This is **decision-focused learning** (Donti–Kolter), *not* imitation of solver outputs.

**The "is this just a warm start?" question — answered precisely.** No. A warm start (Design C) only sets the solver's initial point; the problem and its solution are unchanged. In Design A the network **chooses the problem's parameters `ϑ`** and is trained **through** the solver against the downstream objective — so it changes *what is solved*, not merely *how fast*. Its non-trivial value must come from one of three places (else it would be pointless to predict known parameters):

1. **Effective vs. true parameters** — `g_φ` emits parameters of a DC problem whose *solution* matches messier reality (absorbing AC losses, reserves, weather-dependent ratings). This is the "deeper physics" channel.
2. **Amortized problem reduction** — `g_φ` parameterizes a *screened/aggregated* OPF (fewer constraints) whose duals are still self-consistent → cheaper inner solves inside the §4 planning loop.
3. **Cross-topology amortization** — one pretrained `f_θ` serves many grids/contingencies, which vanilla `zap` cannot.

The falsifiable failure mode: if, after training, `g_φ` just copies the known parameters through, Design A *was* a warm start with extra steps. An experiment must check exactly that.

**Verification add-on.** A safety layer (α-CROWN bound propagation, arXiv:2510.23196; or IBP to 8,316 buses, arXiv:2511.15624) can certify worst-case constraint violation of the encoder+decoder during fine-tuning (≥50 % violation reduction reported).

### 7.4 Design B — pure latent planner with dual decoder (highest risk/reward)

Drop the solver at inference; plan **in latent space** (à la LeWorldModel/GC-IDM) and **decode duals** with a learned head. Blocked on two unsolved problems: (i) **hard constraints** in latent planning, (ii) the **active-set / sign-flip** pathology of §6 (a smooth decoder cannot represent the LP's piecewise-constant duals). Recommended only as a long-horizon research bet.

### 7.5 Recommended sequence

**C → A → (maybe) B.** Build C to (a) prove value and (b) train the JEPA grid encoder; reuse that encoder in A by adding the KKT head and topology generalization; treat B as research.

---

## 8. Honest status: what is real vs. designed

**Nothing has been trained on any data.** To be explicit:

| Artifact | Status |
|---|---|
| `ralph/PROMPT.md`, `ralph/ralph.sh` | **Real.** The research loop scaffolding; ran 6 iterations, self-terminated. |
| `ralph/REPORT.md`, `NOTES.md`, `SOURCES.md`, `TODO.md` | **Real text.** A literature/architecture study. Most cited papers post-date the model's training cutoff and were gathered by the loop's web research; speculative claims are flagged in-place. |
| `ralph/experiments/lmp_gradient_sensitivity.py` | **Real, runs against real `zap`.** Builds a synthetic 6-bus network, solves DC-OPF, confirms the piecewise-constant-gradient theorem and the −0.998 sign flip (reproduced today). **No ML training.** |
| `ralph/experiments/rambo_boundary_sampling.py` | **Real, runs.** Demonstrates boundary-focused scenario sampling (100 % boundary hit rate vs. 90 % uniform). **No ML training.** |
| GridSFM / LeJEPA / LeWorldModel benchmark numbers | **Cited, not independently reproduced.** From the loop's web research; treat as literature claims. |
| Designs A, B, C | **Proposals.** No encoder, decoder, or hybrid model has been built or trained. The `OPFParamDecoder` snippets are illustrative, not implemented. |
| GridSFM as a pretrained backbone | **Not attempted.** The loop noted the package isn't installable in this environment (no PyG, CPU-only); the architecture mapping is paper-based. |

In short: the **mathematics and the `zap` interfaces are verified against the actual codebase**, the **diagnostic experiment is real and reproducible**, and **everything labeled "Design" is an untrained proposal.** The natural next step, if you want to cross from design into evidence, is the smallest end-to-end Design C prototype on a PGLib case — predict a primal+dual warm start, measure iteration reduction — since that both delivers value and produces the encoder that Design A needs.

---

## 9. Pointers into the repo

- DC-OPF LP, KKT, duals: `zap/network.py` (`DispatchOutcome`, `_kkt_power_balance`, `kkt_vjp_variables`)
- Differentiable layer (forward/backward): `zap/layer.py` (`DispatchLayer`, `parameter_names`, `backward`)
- Planning (expansion, multi-value): `zap/planning/` (`PlanningProblemCVX`, `InvestmentObjective`, operation objectives)
- DC line device (why it's DC not AC): `zap/devices/transporter/ac_line.py`
- Diagnostic experiments: `ralph/experiments/`
- Full literature study & bibliography: `ralph/REPORT.md`, `ralph/SOURCES.md`

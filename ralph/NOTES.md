# Ralph Research Loop — Raw Notes

## Zap Codebase Analysis (verified from source)

**Repository**: https://github.com/degleris1/zap
**Papers**:
- Degleris, El Gamal, Rajagopal, "GPU Accelerated Security Constrained Optimal Power Flow," arXiv:2410.17203, Oct 2024
- Degleris, El Gamal, Rajagopal, "Gradient Methods for Scalable Multi-value Electricity Network Expansion Planning," arXiv:2404.01255, 2024
- Sreekumar, Degleris, Rajagopal, "Large-Scale Network Utility Maximization via GPU-Accelerated Proximal Message Passing," arXiv:2509.10722, 2025

**Core architecture** (`zap/layer.py`, `zap/planning/`):
- `DispatchLayer`: maps device parameters → `DispatchOutcome` via CVXPY or ADMM
- `DispatchOutcome` contains: `power`, `angle`, `prices` (nodal LMPs), `phase_duals`, `local_equality_duals`, `local_inequality_duals`
- Prices computed via `_kkt_power_balance`: sum of net powers = KKT condition for power balance constraint → nodal LMPs
- `PlanningProblemCVX.backward()`: implicit differentiation through KKT conditions via `layer.backward()` + torch autograd for direct dependencies
- `PlanningProblemADMM.backward()`: PyTorch autograd through unrolled ADMM iterations

**Dual variables** (`zap/dual.py`, `zap/devices/dual/`):
- `DualInjector`, `DualGround`, `DualDCLine`, `DualACLine`, `DualBattery` classes
- `dualize()` function converts primal devices to dual versions for sensitivity analysis

**Multi-value objectives** (`zap/planning/`):
- `InvestmentObjective`: linear investment cost
- `MultiObjective`: weighted combination (cost, emissions via carbon_tax, curtailment)
- Experiment configs use `emissions_weight: 200.0 $/ton CO2`, `cost_weight: 1.0`
- Parameters expanded over: generator capacity, dc_line, ac_line, battery capacity

**GPU acceleration**:
- `ADMMLayer`: GPU-accelerated ADMM solver for dispatch
- `WeightedADMMSolver`: variant with adaptive step sizes
- Warm start support: reuses previous ADMM state
- Contingency support: N-k via `num_contingencies`, `contingency_device`, `contingency_mask`

**Data scale** (from experiment configs):
- `load_medium` PyPSA case: ~100 nodes
- 24-hour time horizons
- Stochastic scenarios with multiple snapshots
- Parameters: power_unit=1000 MW (GW scale), cost_unit=100 $/MWh

**Key insight for surrogate design**: zap already has:
1. Exact dual variables (LMPs) from KKT
2. Differentiable forward/backward through dispatch
3. Multi-period, multi-value planning with gradient descent on investments
4. GPU-parallel batch dispatch

A JEPA surrogate would need to interface with these existing capabilities, not replace them wholesale.

---

## JEPA Family — General Knowledge (pre-agent verification)

**I-JEPA** (Assran et al., 2023): Image JEPA — trains image encoder by predicting representations of masked patches. Uses context encoder + predictor in latent space. No pixel-level reconstruction. First paper showing JEPA principle at scale.

**V-JEPA** (Bardes et al., 2024): Video JEPA — extends to video by predicting spatiotemporal patches. 

**Key JEPA design principles** (LeCun's world model agenda):
- Encode context x → z_x, predict target representation ẑ_y, compare to encoded target z_y
- All in latent space — no pixel reconstruction → avoids modeling irrelevant details
- Collapse prevention: VICReg-style variance/covariance regularization in prior JEPAs

**LeJEPA** (Balestriero & LeCun, 2025 — to be verified):
- Alleged mechanism: SIGReg = Sketched Isotropic Gaussian Regularization
- Alleged fix: replaces heuristic variance/covariance terms with principled Gaussian regularization
- Need to verify exact paper and arXiv URL

**JEPA for planning / control**:
- LeCun's world model agenda: plan by gradient descent through frozen latent predictor
- Continuous action optimization: minimize cost(predict(encode(x), a)) over action a
- Challenge for power grids: actions are continuous dispatch vectors with hard constraints

---

## AC-OPF Learning — General Knowledge (pre-agent verification)

**DC3** (Donti et al., 2021, ICLR):
- "DC3: A learning method for optimization with hard constraints"
- Two-phase: (1) predict most variables, (2) complete remainder by solving equality constraints
- Feasibility repair via gradient correction
- Works on DC-OPF and AC-OPF; achieves near-zero constraint violations with repair

**DeepOPF** (Huang et al., 2021):
- Predicts AC-OPF solution with neural network, applies DC approximation
- DC-OPF: better accuracy; AC-OPF: harder due to non-convexity

**Primal-Dual Learning (Fioretto et al.)**:
- Uses Lagrangian relaxation in training loss to penalize constraint violations
- Goal: achieve both accuracy and feasibility in one shot
- Papers: "Predicting AC Optimal Power Flows: Combining Deep Learning and Lagrangian Dual Methods" (2020)
- **Dual variable recovery**: these methods do implicitly learn dual information, but accuracy is unclear

**E2ELR**: End-to-end learning with repair — predicts then corrects to feasibility

**LMP prediction literature**:
- LMPs are dual variables of the power balance constraints
- Small primal error (dispatch) can correspond to large dual error (prices) — especially at constraint boundaries (degenerate cases)
- Few papers specifically target dual accuracy; most focus on primal feasibility

---

## Differentiable Optimization Frameworks (general knowledge)

**OptNet** (Amos & Kolter, 2017): differentiable QP layer; backprop through KKT conditions
**cvxpylayers** (Agrawal et al., 2019): differentiable CVXPY layer; wraps conic solve with diffcp
**diffcp**: differentiable cone programs via implicit function theorem

**Relevance to zap**: zap already implements implicit differentiation through KKT conditions in `PlanningProblemCVX.backward()`. This is equivalent to what cvxpylayers does but custom-built for power networks.

---

## Grid Foundation Models — VERIFIED (Agent Results)

### Microsoft GridSFM

**Exact name**: GridSFM (Grid Small Foundation Model), under the GridFM initiative  
**Blog**: https://www.microsoft.com/en-us/research/blog/gridsfm-a-new-small-foundation-model-for-the-electric-grid/  
**Date**: May 13, 2026  
**GitHub**: https://github.com/microsoft/GridSFM  
**HuggingFace**: https://huggingface.co/datasets/microsoft/GridSFM_US_power_grid  
**Open-data pipeline paper**: arXiv:2605.04289 (May 2026) — "Building Power Grid Models from Open Data: A Complete Pipeline from OpenStreetMap to Optimal Power Flow"  
**Partnership**: MISO (Jan 6, 2026) — 15 US states, real-time operations

**Trained on**: 150+ base grid topologies, ~500K scenarios; varying loads, N-k outages, voltage bounds, cost coefficient variation; supervised by AC-OPF solver IPOPT via PowerModels.jl + physics constraints; companion dataset covers all 48 contiguous US states (Western Interconnection 5,076 buses, Eastern 21,697 buses) from public data (OpenStreetMap, EIA, Census)

**Model tiers**:
- GridSFM-Open: ~15M parameters, grids up to ~4,000 buses, MIT licensed
- GridSFM-Premier: ~100M parameters, grids up to ~80,000 buses

**Tasks**: AC-OPF approximation, AC system state prediction (voltages, angles, branch flows, dispatch), feasibility classification, warm-start initialization

**Reported results**:
- Median cost gap vs. IPOPT: **2.23%** (mean 3.41%)
- <5% cost gap on **83%** of test scenarios
- Feasibility detection: **95.5% accuracy**, AUC = **0.986**
- Warm-start speedup: **1.66× geometric mean** vs. cold start; **1.59×** vs. DC-OPF warm start

**Key authors**: Weiwei Yang, Andrea Britto Mattos Lima, Thiago Vallin Spina, Spencer Fowers, Baosen Zhang (UW)

**Important note**: The IBM papers cited "4-5 orders of magnitude speedup" were based on analogies to weather foundation models, NOT direct GridSFM benchmarks. Actual measured speedup is 1.66× for warm starting.

---

### IBM GridFM / OpenGridFM (LF Energy)

**IBM perspective paper**: arXiv:2407.09434 (July 2024) — "A Perspective on Foundation Models for the Electric Power Grid"  
**Journal version**: Joule (Cell Press) 2024  
**IBM GridFM 0.5**: IBM Research, Feb 11, 2025 (authors: Jonas Weiss, Alban Puech, Matteo Mazzonelli)  
**LF Energy project**: https://lfenergy.org/projects/gridfm/  
**Website**: https://gridfm.org/gridfm-ibm/

**Architecture**: GNN-based encoders; multi-modal encoder-decoder with self-supervised pre-training via masked reconstruction; transformer-based attention

**Status as of June 2026**: GridFM 0.5 = proof-of-concept; OpenGridFM-v0 = Sandbox lifecycle stage under LF Energy; contributors: IBM + Hydro-Québec + Imperial College

**Downstream tasks targeted**: Contingency analysis, outage prediction, load/renewable forecasting, expansion planning, stability analysis, congestion management

---

### OPFData (Google DeepMind)

**Paper**: arXiv:2406.07234 (June 2024) — "OPFData: Large-scale datasets for AC optimal power flow with topological perturbations"  
**Authors**: Sean Lovett et al. (all Google DeepMind)  
**Data location**: gs://gridopt-dataset/

**Scale**: ~3 million AC-OPF instances, 10 grid models (14 to 13,659 buses: IEEE 14/30/57/118, GOC 500/2000/10000, SDET 4661, RTE/PEGASE 6470/13659)  
**Variants**: FullTop (fixed topology, load scaled 0.8-1.2×) and N-1 (random generator outage + line removal)

---

### PowerGraph / GraphNeuralSolver

**GraphNeuralSolver**: Donon et al. (Inria), "Neural networks for power flow: Graph neural solver," EPSR Vol. 189, 2020, doi:10.1016/j.epsr.2020.106547 — GNN minimizing Kirchhoff violation; topology-robust  
**PowerGraph**: Varbella et al. (ETH Zürich), arXiv:2402.02827, NeurIPS 2024 D&B track — benchmark dataset for GNN evaluation on power grids (not a model)  
**Topology-aware GNN**: Liu, Wu, Zhu (UT Austin), arXiv:2205.10129, IEEE TPWRS 2022 — exploits LMP locality; efficient retraining for topology changes

---

## AC-OPF Surrogate Learning — VERIFIED (Agent Results)

### DC3
**Paper**: Donti, Rolnick, Kolter, "DC3: A learning method for optimization with hard constraints," ICLR 2021  
**arXiv**: 2104.12225  
**Code**: https://github.com/locuslab/DC3  
**Mechanism**: Deep Constraint Completion (equality) + Correction (unrolled gradient descent for inequalities)  
**Results**: 97/100 test cases near-zero violations; ~10× faster than PYPOWER; 0.22% optimality gap average

### DeepOPF Family
- DeepOPF (DC): arXiv:1910.14448 (2020) — up to 2 OOM speedup, <0.2% cost diff
- DeepOPF (AC): arXiv:2007.01002 (2022) — <0.1% cost diff on IEEE 30/118/300/2000-bus; inequalities NOT hard-guaranteed
- DeepOPF+: arXiv:2009.03147 (2020) — 100% feasible DC-OPF via limit calibration
- DeepOPF-V: arXiv:2103.11793 (2021) — up to 4 OOM speedup; 0.03% reactive violation

### E2ELR
**Paper**: Chen, Tanneau, Van Hentenryck, IEEE TPWRS 2023; arXiv:2304.11726  
**Mechanism**: Analytical differentiable repair layers for generator limits, power balance, reserves; self-supervised (no labeled solutions)  
**Results**: Order-of-magnitude better optimality gap on industry-scale (tens of thousands of buses)  
**Limitation**: Repair layers don't handle nonlinear AC thermal constraints

### Primal-Dual Learning (Fioretto et al.)
- Fioretto et al., AAAI 2020, arXiv:1909.10461: 0.2% prediction error; 2 OOM improvement vs. DC  
- Fioretto et al. Lagrangian Duality, ECML-PKDD 2020, arXiv:2001.09394: CCGA for constraint penalization  
- Park & Van Hentenryck, Self-Supervised PDL, AAAI 2023, arXiv:2208.09046: primal+dual networks jointly trained; negligible violations

### Dual Variable / LMP Recovery
- Pagnier, Chertkov et al. (IREP 2022, arXiv:2205.11641): Active set prediction → LMPs derived analytically from reduced KKT system
- Ferrando et al. (2023, arXiv:2304.00062): Same approach on 1,814-bus NYISO system with renewables
- Liu, Wu, Zhu (2021, arXiv:2106.10529): GNN for LMP prediction with spatial locality regularization
- Jami et al. (2023, arXiv:2306.10080): Direct LMP regression, 4-5 OOM faster, ~5-6% error

**Key insight (WARP paper, arXiv:2605.05728)**: Primal-only warm starts FAIL to reduce interior-point solver iterations. Full primal+dual warm start achieves 76% iteration reduction. Primal-without-dual can cause IPOPT divergence. This proves dual variable prediction is NOT optional for warm starts.

### Warm Starts
- Diehl 2019 (NeurIPS Climate Change AI): GNN warm-start for AC-OPF, mean 2.8× speedup on Texas grid
- WARP (arXiv:2605.05728, May 2026): Encode-process-decode GNN predicts full IP state; 76% IPOPT iteration reduction

---

## Compositional Generalization — VERIFIED (Agent Results)

### HH-MPNN (Best current result)
**Paper**: Arowolo & Cremer, arXiv:2510.06860 (Oct 2025)  
**Architecture**: Hybrid Heterogeneous MPNN = HGNN + Scalable Transformer + physics positional encodings + training data augmentation  
**Results**: <1% optimality gap on **14 to 2,000 bus** grids; zero-shot N-1 <3% gap; ~5,000× speedup vs. IP solver  
**Training data**: PGLearn + GridFM-DataKit

### OPF-HGNN
**Paper**: arXiv:2403.00892, IEEE ICDCS 2024  
**Code**: https://github.com/yamizi/OPF-HGNN  
**Results**: 100% constraint satisfaction on 9-bus and 14-bus; 98% on 30-bus; 3 OOM better than FCNN baselines; transfer from 9-bus to 30-bus: 98% constraint satisfaction

### Degleris et al. — Verified Details
**Paper**: arXiv:2404.01255 (April 2024)  
**MEP formulation**: min_η γᵀη + h(x*(η), λ*(η))  
**Objective variants**: cost, emissions (e·g*(η)), profit (LMPs × generation)  
**Gradient**: ∇J(η) = γ + ∂z*(η)ᵀ · ∇h(z*(η)) — sensitivity of congestion prices to investment  
**Results**: 5.3× lower objective, 16.5× faster vs. IP reformulation; 40% carbon reduction at $17.1/MWh additional cost  
**Follow-up**: arXiv:2410.13055 (Oct 2024) — >100M variables; 100× warm-start speedup for interactive scenario analysis

### Feasibility Restoration Layers
- Flow Matching (arXiv:2512.11127, Dec 2025): CFM refinement + iterative projection → 0.07% cost gap, 100% feasibility
- DC-to-AC hard-constrained NN (arXiv:2602.06255, Feb 2026): 40× speedup on PEGASE-9241, 10⁻⁴ violations
- Homeomorphic Projection (JMLR 25, 2024): INN maps constraint set ↔ unit ball; provable feasibility for non-convex AC-OPF
- Input-Convex NN for N-k screening (arXiv:2410.00796, Oct 2024): 10-20× speedup, zero false negatives

### JEPA Planning — Verified Details
See JEPA section above:
- V-JEPA-2-AC: 65-80% robot manipulation success, zero-shot
- DINO-WM: Maze 81.6%, Wall 64.1%, Push-T 66.0%
- LeWM: 48× faster than DINO-WM, continuous actions, 15M params
- PLDM: Best generalization to new layouts (NeurIPS 2025)
- **No paper handles hard constraints in latent-space planning** (confirmed by agent)

---

## Failure Modes — Architecture Analysis (from first principles)

**Critical issue: duals are derivative objects**
- LMP = dCost/d(net injection) — a derivative of the optimal value function
- Small primal error (||x* - x̂||) does NOT imply small dual error (||λ* - λ̂||)
- At constraint boundaries: dual variable can be discontinuous (degenerate LP)
- Example: if a line is exactly at its limit, the congestion rent (shadow price) is sensitive — a tiny primal shift changes which constraints are binding

**AC non-convexity challenge**:
- AC-OPF is non-convex; multiple local optima exist
- A surrogate trained on one set of solutions may not generalize to different local optima
- The "physics" of AC power flow (Kirchhoff's laws) must be satisfied

**Data coverage of binding-constraint patterns**:
- Most AC-OPF training data may not cover all binding constraint configurations
- N-k contingencies can create novel binding patterns
- Transfer across grid topologies is especially challenging

**Compositional generalization**:
- Fixed-size MLPs cannot generalize to different grid sizes
- GNNs can in principle, but permutation equivariance ≠ topology equivariance
- Adding/removing buses requires careful treatment of the graph structure

---

## Iteration 2 New Findings

### "Not All Warm Starts Help" (arXiv:2606.08984)

**Full title**: "Not All Warm Starts Help: Benchmarking Primal-Dual Initializations for ACOPF Algorithms"
**Authors**: Babak Taheri, Daniel K. Molzahn (Georgia Tech)
**Date**: June 2026

**Key results** (19 AC-OPF instances, 5–30,000 buses):
- Full primal+dual initialization: **47.6% median solve-time speedup**
- Partial combinations: 12 of 14 tested produce **negative speedups** (worse than cold start)
- Full bound-multiplier vectors only: 90.7% convergence, +26.8% median speedup
- DC seeding post-presolve: statistically insignificant effect (p = 0.4171)

**Implication**: Independently confirms WARP's finding. Providing partial dual information does not just fail to help — it actively degrades solver performance. The complete primal+dual state is the minimum necessary requirement for a neural warm start to be beneficial.

**Key distinction from WARP**: WARP (Suri et al.) uses a trained GNN to predict full IP state and measures 76% IPOPT iteration reduction. Taheri & Molzahn use hand-crafted partial initializations to identify the performance floor of each component. Together they bracket the design space: WARP shows the ceiling (~76-85%), Taheri/Molzahn show what happens without complete duals (often worse than cold start).

---

### Graph-JEPA (arXiv:2309.36014, TMLR)

**Title**: "Graph-level Representation Learning with Joint-Embedding Predictive Architectures"
**Authors**: Geri Skenderi, Hang Li, Jiliang Tang, Marco Cristani
**Venue**: Transactions on Machine Learning Research (TMLR)

**Architecture**:
- Context GNN encodes unmasked subgraphs; target GNN encodes masked subgraph
- Predictor bridges context embedding to predicted target embedding
- Hierarchical objective: predict coordinates on unit hyperbola (implicit graph hierarchy)
- No negative sampling, no generative pixel/node reconstruction

**Relevance to power grids**:
- Power grids are heterogeneous graphs with implicit hierarchy (transmission → distribution)
- Masking electrical zones/substations and predicting their representations forces encoder to learn congestion patterns
- Topology-change generalization: model operates on any subgraph structure, not fixed-size

---

### TS-JEPA (arXiv:2509.25449, NeurIPS 2024 Workshop)

**Title**: "Joint Embeddings Go Temporal"
**Authors**: Sofiane Ennadir, Siavash Golkar, Leopoldo Sarra
**Venue**: NeurIPS 2024 Workshop "Time Series in the Age of Large Models"

**Architecture**: JEPA applied to time series — predict future-timestep latent representations from past context. No pixel/value reconstruction.

**Claims**: Matches or surpasses SOTA on classification and forecasting across multiple datasets; robust to noise.

**For power grids**:
- 24-hour dispatch sequences: TS-JEPA can pre-train on historical hourly generation/price data
- Battery SOC trajectories: temporal prediction task directly addressable
- Renewable forecasting context: load + weather features → future grid state
- Limitation: no constraint handling in temporal planning

---

### FF-JEPA (arXiv:2606.09311, Jun 2026)

**Title**: "FF-JEPA: Long-Horizon Planning in World Models with Latent Planners"
**Authors**: Sergi Masip, Jonathan Swinnen, Yutong Hu, Renaud Detry, Tinne Tuytelaars

**Problem**: Standard JEPA + CEM (Cross-Entropy Method) is too expensive and ineffective for long horizons. "Flat world models" (single-step) exhibit long-horizon collapse.

**Solution**: Two models:
1. Action-free subgoal predictor: predicts intermediate states without requiring actions
2. Forward model: predicts next state from (current state, action)

**For multi-period OPF**:
- Subgoal predictor: predict 24-hour dispatch trajectory envelope (load envelopes, price corridors) without solving per-hour dispatch
- Forward model: given current grid state + investment decision, predict next-hour state
- Planning: find investment actions that drive the trajectory to low-cost subgoals

**Preliminary results only** — no quantitative benchmarks in the paper.

---

### Value-Guided JEPA (arXiv:2601.00844, Dec 2025)

**Title**: "Value-guided action planning with JEPA world models"
**Authors**: Matthieu Destrade, Oumayma Bounou, Quentin Le Lidec, Jean Ponce, Yann LeCun

**Key idea**: Train the JEPA encoder so that goal-conditioned value function v(z_current, z_goal) = -d(z_current, z_goal) (latent distance = negative value/cost-to-go).

**For power grids**:
- Train the encoder so that dispatch cost is monotonically encoded in latent distance
- Planning by gradient descent in latent space becomes equivalent to minimizing dispatch cost
- Addresses the Design B surrogate gradient bias: if latent distance = cost, then ∇z(cost) is the correct planning gradient

**Caveat**: Demonstrated only on simple continuous-action control tasks; constraint satisfaction not demonstrated.

---

### ADMMState Structure (verified from zap/admm/basic_solver.py)

```python
@dataclasses.dataclass
class ADMMState:
    num_terminals: int                    # total terminal count
    num_ac_terminals: int                 # AC terminal count (0 for DC-only)
    power: List[List[Tensor]]             # [device_idx][terminal] → (n_devices, T)
    phase: List[List[Tensor]]             # same structure; None entries for DC devices
    dual_power: Tensor                    # (num_nodes, T) — CRITICAL: prices = -rho_power * dual_power
    dual_phase: List[List[Tensor]]        # same structure as phase
    avg_power: Tensor                     # (num_nodes, T) — ADMM consensus variable
    avg_phase: Tensor                     # (num_nodes, T) — ADMM consensus variable
    resid_power: List[List[Tensor]]       # per-device residuals
    resid_phase: List[List[Tensor]]       # per-device residuals
    objective: float = None
    clone_power: List[List[Tensor]] = None   # z variables (clone)
    clone_phase: Tensor = None
    rho_power: float = None               # ADMM step size for power
    rho_angle: float = None               # ADMM step size for angle
    local_variables: List = None          # device-local quantities
```

**Key**: `prices = -rho_power * dual_power` (from `as_outcome()` in basic_solver.py). For a DC-only 100-node 24-hour problem: `dual_power` shape = (100, 24) = 2,400 scalar predictions.

**NeuralWarmStart must predict**: dual_power (LMPs), power (dispatch), avg_power (consensus). Residuals and clones can be initialized to zero and will be corrected in early ADMM iterations.

---

### DispatchLayer Parameter Interface (verified from zap/layer.py)

```python
# parameter_names = {"kwarg_name": (device_index, "attribute_name")}
parameter_names = {
    "gen_capacity": (0, "nominal_capacity"),    # shape [n_generators]
    "gen_cost":     (0, "linear_cost"),         # shape [n_generators, T]
    "line_capacity":(2, "nominal_capacity"),    # shape [n_lines]
}
outcome = dispatch_layer(**params)
# outcome.prices: Tensor (num_nodes, T) — exact KKT LMPs
# outcome.power: List[Tensor] — dispatch per device
```

**For Design A**: HGNN encoder per-node latents → node-type-specific decoders → per-device capacity/cost parameters → DispatchLayer. Load (`load[n_loads, T]`) is exogenous and passed separately, not decoded from the encoder.

---

### OPFData DC-OPF Compatibility

**Finding**: OPFData (DeepMind, arXiv:2406.07234) provides AC-OPF solutions from IPOPT/PowerModels.jl. Zap uses DC-OPF (linearized power flow). These are NOT directly compatible for supervised training.

**Path 1 (preferred)**: Re-solve DC-OPF on PGLib topologies using zap after PyPSA conversion. Produces zap-compatible DC-OPF ground truth.

**Path 2**: Use OPFData grid topologies + load scenarios; replace AC solutions with DC solutions from zap. Leverages OPFData's rich topology/scenario diversity.

**Path 3**: Use OPFData for unsupervised Graph-JEPA pre-training only (topology/load patterns are transferable; AC active power flows are approximately consistent with DC).

**OPFData grids available**: IEEE 14/30/57/118, GOC 500/2000/10000, SDET 4661, RTE/PEGASE 6470/13659 buses. Most are available in MATPOWER format, which can be converted to PyPSA via `pandapower.converter.from_mpc()` or similar.

---

### LMP Accuracy Threshold Analysis (from first principles; no paper found)

**No published threshold exists.** Analysis:

For MEP profit-objective: gradient sign flip when |LMP_error| > |congestion_rent_of_marginal_line|
- Typical congestion rents: $1–50/MWh
- At $40/MWh reference price: Jami et al.'s 5-6% error = ~$2-3/MWh absolute error
- For barely-congested lines ($1-2/MWh rent), 5-6% LMP error can flip the gradient sign

For MEP cost-objective (zap's default): gradient depends on dual through ∂z*/∂η; more robust to LMP errors asymptotically.

**Design B requires**: Either higher-accuracy LMP prediction than current SOTA, OR the value-guided JEPA approach that encodes cost geometry directly into the latent space.

---

## Iteration 3 New Findings

### Zap's "ACLine" Is DC Power Flow, Not True AC-OPF (verified from source)

**Critical clarification** (`zap/devices/transporter/ac_line.py`):

Zap's `ACLine` class implements the **linearized DC power flow approximation** — NOT full AC-OPF with reactive power and voltage magnitudes. The linearized flow equality constraint is:

```python
# From ACLine.equality_constraints():
eq_constraints += [power[1] - la.multiply(b, pnom_dtheta)]
# where: pnom_dtheta = nominal_capacity × (θ_source - θ_sink)
# b = susceptance (B in the B-θ matrix)
```

This is the standard DC approximation: `P = B × Pnom × Δθ`. It handles:
- ✅ Phase angles (θ per bus, per timestep)
- ✅ Active power flows (P on each line)
- ✅ Thermal limits (via inequality constraints)
- ✅ Phase consistency constraints (angle dual variables = `phase_duals`)
- ❌ Reactive power (Q) — absent
- ❌ Voltage magnitudes (|V|) — absent
- ❌ Nonlinear Kirchhoff equations — absent

**Implication for surrogate design**: Zap is a DC-OPF system with angle variables. "AC" in zap refers to the fact that phase angles are tracked (vs. pure DC power injection matching). This resolves a potential source of confusion in the report — all comparisons to GridSFM (AC-OPF) and IPOPT-solved AC-OPF must account for this difference.

**For Design A/C**: The JEPA encoder must predict DC-OPF duals (active power LMPs, angle-related phase duals), NOT AC-OPF quantities (voltage magnitudes, reactive dispatch). This simplifies the prediction task compared to full AC-OPF.

**For AC-OPF extension**: True AC-OPF would require:
1. Q variables and V magnitude variables per node
2. Nonlinear equality constraints: `P = |V_i|·|V_j|·(G_ij·cos(θ_ij) + B_ij·sin(θ_ij))`
3. New proximal operators for ADMM (nonlinear local problems)
4. Non-convex KKT conditions (necessary but not sufficient for optimality)
This is substantial new development. The better near-term path for AC extension is to use zap's DC layer + a DC-to-AC correction NN (arXiv:2602.06255).

---

### MATPOWER → PyPSA → Zap Conversion Path (verified from code)

**PyPSA 0.30.2** (zap's dependency) supports reading MATPOWER/pypower format:
- `pypsa.Network().import_from_pypower(pypower_net)` for pypower-format dicts
- Standard IEEE cases available via `from pypower.case14 import case14; net.import_from_pypower(case14())`
- For PGLib `.m` files: use `pandapower.converter.from_mpc(filename)` → PyPSA

**Complete pipeline** (verified from code):
```python
# Step 1: Load PGLib MATPOWER case (two options)
import pandapower.converter as pc
pp_net = pc.from_mpc('pglib_opf_case14_ieee.m')  # pandapower format

# OR for standard IEEE via pypower:
from pypower.case14 import case14
import pypsa
psa_net = pypsa.Network()
psa_net.import_from_pypower(case14())

# Step 2: Convert to zap
from zap.importers.pypsa import load_pypsa_network
network, devices = load_pypsa_network(psa_net, snapshots=snapshots, power_unit=1000.0)
```

`load_pypsa_network()` is the correct entry point, accepting any PyPSA network with generators, loads, lines, and buses. It handles:
- Parse buses → `PowerNetwork(num_nodes)`
- Parse generators → `Generator(nominal_capacity, linear_cost, ...)`
- Parse lines → `ACLine(susceptance, capacity, ...)`
- Parse loads → `Load(nominal_demand, ...)`
- Parse storage (if present) → `StorageUnit(charge_rate, ...)`

**Note**: No pandapower dependency in zap's pyproject.toml — pandapower would need to be installed separately for MATPOWER `.m` import. For PGLib standard IEEE cases (14/30/57/118/300), the direct `pypower.case*` route is cleaner.

---

### GridSFM Architecture (verified from github.com/microsoft/GridSFM code)

**Full name**: GridSFM — Grid Small Foundation Model  
**Code**: https://github.com/microsoft/GridSFM  
**Core class**: `GridTransformerBackbone` (`/model/gridsfm/model.py`)  
**Framework**: PyTorch + PyTorch Geometric (PyG); requires `torch>=2.6,<2.9` + `torch_geometric>=2.5,<3`  

#### Architecture: Trunk-and-Heads HGNN

**Backbone**: 8 stacked `GridBlock` modules, each with:
1. **Linear Self-Attention** (Performer-style, per node type): ELU activation; 4 attention heads; per-graph attention to handle variable-size grids
2. **Heterogeneous Message Passing**:
   - `SignedIncidenceConv` for topological edges (bus-branch, cycle-branch) with signed ±1 direction weights
   - `SAGEConv` (mean aggregation) for other edge types (generator-bus, load-bus, shunt-bus)
3. **Feed-Forward Network**: per-node-type 2-layer MLP, GELU, 4× expansion, residual connections

#### Node Types (7):
`bus` (4→16 features), `generator` (11 features), `load` (2 features), `shunt` (2 features), `branch_ac` (9 features), `branch_tr` (11 features), `cycle` (4 features)

#### Edge Types (16): AC lines, transformers, generator-bus links, load-bus links, shunt-bus links, cycle-branch (in_cycle), and their reverses

#### Positional Encoding:
- **Hodge PE**: 8-step diffusion over signed Hodge Laplacians → 8-dim node encoding (topology-aware)
- **ER Landmark Moments**: Electrical resistance distances to 64 landmark buses → 5 moment statistics
- **DC Prior Features**: Voltage angles, angle stress, power injection from DC power flow solution (pre-cached with SuperLU)

#### Output Heads (5):
| Head | Output | Task |
|------|--------|------|
| `head_theta` | per-bus scalar | Voltage angle (θ, rad) |
| `head_V` | per-bus scalar | Voltage magnitude (|V|, p.u.) |
| `head_Pg` | per-generator scalar | Active dispatch (MW) |
| `head_Qg` | per-generator scalar | Reactive dispatch (MVAr) |
| `head_feas` | single logit | Feasibility probability |

#### Compatible with Transfer Learning:
- Backbone can be loaded and used independently of output heads
- Intermediate embeddings accessible at all 8 block depths
- `finetune_opfdata()` supports full fine-tuning on OPFData format
- Pre-trained weights via HuggingFace: `microsoft/GridSFM_Open` (Open tier, MIT licensed)

---

### GridSFM × Zap Compatibility Analysis for Design A/C

**Bottom line**: GridSFM's pre-trained backbone is **compatible in principle but requires fine-tuning** to serve as a warm-start encoder for zap's DC-OPF.

**Similarities** (enabling transfer):
- Both use heterogeneous GNNs with type-specific message passing
- Both use PyTorch, enabling weight sharing and fine-tuning
- GridSFM's Hodge PE and ER landmark moments encode grid topology — transferable to DC-OPF
- GridSFM's cycle basis (fundamental loops) captures topological congestion patterns — relevant for DC-OPF

**Differences** (requiring adaptation):
1. **AC vs. DC**: GridSFM predicts |V|, Q, θ (AC-OPF quantities); zap needs θ, P (DC-OPF quantities). Output heads must be replaced.
2. **Node features**: GridSFM uses Qmin/Qmax, Vg per generator, reactive shunt features — zap has no reactive power. These AC features have no DC analog and must be handled carefully (zero-padded or dropped).
3. **Data format**: GridSFM requires OPFData format (.pyg.json); zap uses PyPSA → custom device arrays. Data loading pipelines differ entirely.
4. **Grid scale**: GridSFM-Open handles up to ~4,000-bus grids; zap experiments use ~100-node PyPSA-USA medium case. Both are within each other's range.

**Proposed fine-tuning protocol for Design C (NeuralWarmStart)**:
```python
# Step 1: Load pre-trained backbone (without output heads)
from gridsfm import load_from_hf
gridsfm_model = load_from_hf("microsoft/GridSFM_Open")
backbone = gridsfm_model.backbone  # GridTransformerBackbone without heads

# Step 2: Replace AC output heads with DC-OPF warm-start heads
class DCWarmStartHead(nn.Module):
    def __init__(self, hidden_dim, num_nodes, T):
        super().__init__()
        # Predict: dual_power (LMPs), power (dispatch), avg_power (consensus)
        self.dual_power_head = nn.Linear(hidden_dim, T)   # per-node LMP
        self.power_head = nn.Linear(hidden_dim, T)         # per-node active power

# Step 3: Fine-tune on zap-generated DC-OPF (scenario, converged ADMMState) pairs
# Training: backbone.parameters() (frozen or slowly unfrozen) + DCWarmStartHead
```

**Key feasibility issue**: GridSFM was trained on AC-OPF problems (PowerModels.jl + IPOPT). Its internal representations encode AC quantities (voltage magnitudes, reactive power patterns). Fine-tuning for DC-OPF will cause some representation drift — the Hodge PE and structural features should transfer, but the learned bus/generator embeddings may not. A phased fine-tuning (PE frozen, then gradually unfreeze backbone) is recommended.

**Alternative to GridSFM**: Train from scratch using the same HGNN architecture (Heterogeneous GNN with Hodge PE) but on zap-generated DC-OPF data. This avoids the AC/DC compatibility gap but foregoes the pre-trained topological representations. HH-MPNN (arXiv:2510.06860) provides the architecture blueprint without the AC-OPF training bias.

---

### Binding Constraint Patterns (Active Sets) in DC-OPF (Iteration 3)

**Key question**: How many distinct active sets (binding constraint patterns) appear in practice? This is essential for understanding Design B's data requirements.

#### Summary table

| Grid | Distinct Active Sets | Method | Source |
|------|----------------------|--------|--------|
| 3–39 bus (small) | 1–2 | Streaming discovery | Misra et al. (arXiv:1802.09639) |
| 57–162 bus (medium) | 2–17 | Streaming discovery | Misra et al. (arXiv:1802.09639) |
| IEEE 118-bus | **3** (uniform sampling) / **48–53** (RAMBO targeted) | 50,000 samples | Deka & Misra 2019; RAMBO 2304.10912 |
| IEEE 300-bus | **78** | 50,000 samples | Misra et al. (arXiv:1802.09639) |
| 1,888–1,951-bus | 3–11 | — | Misra et al. (arXiv:1802.09639) |
| case240_pserc (outlier) | **>2,993**, still growing | — | Misra et al. (arXiv:1802.09639) |

#### Key papers

**Misra, Roald & Ng (2022)**: "Learning for Constrained Optimization: Identifying Optimal Active Constraint Sets," INFORMS J. Computing 34(1):463–480; arXiv:1802.09639. Streaming algorithm discovers active sets; out-of-sample prediction probability >99% for low-complexity systems. Tested 15 networks from 3 to 1,951 buses.

**Deka & Misra (2019)**: "Learning for DC-OPF: Classifying active sets using neural nets," IEEE PowerTech; arXiv:1902.05607. Across PGLib-OPF, only 4 test cases had >3 active sets across 50,000 samples. Neural network classifiers (multi-class per active set, or binary per constraint) achieve high accuracy.

**Ventura Nadal & Chevalier — RAMBO (2023)**: "Scalable Bilevel Optimization for Generating Maximally Representative OPF Datasets," arXiv:2304.10912. Shows uniform sampling dramatically undercounts active set diversity:
- 118-bus minimum voltage constraint: uniform sampling finds **0** unique patterns; RAMBO finds **48**
- 118-bus generator constraints: uniform = 37, RAMBO = 53

**OPF-Learn (Joswig-Jones, Baker & Zamzam, IEEE ISGT 2022)**: arXiv:2111.01228. Explicitly designed to maximize active constraint set variety in training data. GitHub: NREL/OPFLearn.jl.

**Stratigakos, Pineda et al. (IEEE TPWRS 2024)**: "Interpretable ML for DC-OPF with Feasibility Guarantees." Prescriptive decision trees encoding binding constraints. Comparable to NN approaches with guaranteed feasibility via robust optimization.

#### Implications for Design B (latent planner with dual decoder)

- For a ~100-node PyPSA `load_medium` case with typical ±30% load variation: **expect 3–50 distinct active sets** (interpolating between 57-bus and 118-bus results, closer to 118-bus)
- With deliberate boundary sampling (RAMBO-style): potentially 50–200 distinct active sets for a 100-bus grid
- This is **tractable** for classification: a Design B surrogate can in principle learn to identify the active set, especially if trained on RAMBO-generated boundary scenarios
- Critical nuance: **active set identification is the hard part for Design B** — LMP discontinuities occur precisely at the boundaries between active sets. A latent planner that smoothly interpolates across these boundaries will produce wrong LMPs.
- Design A (JEPA encoder → exact KKT head) avoids this entirely: the KKT layer determines the active set automatically at inference

---

### DINO-WM vs. LeWorldModel for Non-Visual Domains (Iteration 3)

#### DINO-WM (arXiv:2411.04983, ICML 2025)

**Architecture**: Frozen DINOv2 ViT encoder (pretrained on ~124M images) + trained transition ViT (dynamics model). Planning in frozen DINOv2 latent space. All 6 tested environments are **visual** (maze, push manipulation, rope, granular materials).

**Non-visual applicability**: None — the frozen DINOv2 encoder produces meaningless representations for non-pixel inputs. Power grid state (bus voltages, line flows) has no spatial image structure; DINOv2 patch features are completely inapplicable.

**ICML 2025 poster**: Confirmed peer-reviewed.

#### LeWorldModel (arXiv:2603.19312, v3 Jun 2026)

**Architecture**: End-to-end JEPA from pixels, 2 losses (prediction + SIGReg), no stop-gradient, no frozen encoder, ~15M params. **All tested tasks are visual** (Push-T, OGBench-Cube, Two-Room, Reacher). GitHub: github.com/lucas-maes/le-wm.

**Non-visual applicability**: Also no in current form — "from pixels" in the title refers to literal image pixels. However, the architectural **principle** (end-to-end JEPA with Gaussian regularizer, no frozen encoder) is transferable: replace pixel CNN/ViT with GNN encoder.

**DINO-WM vs. LeWM comparison** (from LeWM paper):
- LeWM 48× faster planning (no large frozen foundation model)
- LeWM outperforms DINO-WM on Push-T, Reacher
- DINO-WM slightly better on OGBench-Cube (higher visual complexity may benefit from large pretrained features)
- LeWM: "frozen encoders limit expressivity to pretraining knowledge; end-to-end not bounded by pretraining"

#### Graph World Model (arXiv:2507.10539, July 2025)

**NEW RELEVANT FINDING**: "Graph World Model" (Feng, Wu, Lin, You) — first world model explicitly designed for **graph-structured state**. Uses message-passing GNNs to encode graph state + latent dynamics model. Tested on recommendation systems, multi-agent scenarios, graph prediction tasks.

**Not tested on power grids** — but this is the closest existing framework to what a "Grid World Model" would be. Architecture: GNN encoder → JEPA-style dynamics in latent space. This is precisely the backbone needed for Designs A/B/C.

#### Conclusions for Power Grid World Models

1. **DINO-WM is inapplicable** to power grids: frozen visual encoder has zero transferability to graph-structured grid data
2. **LeWM principle applies but code does not**: end-to-end JEPA with Gaussian regularizer (= SIGReg from LeJEPA) + GNN encoder is the correct adaptation; this is architecturally Design C/A
3. **Graph World Model (arXiv:2507.10539) is the closest prior work** for grid-like state, though untested on power grids
4. **LeWM's speed advantage (48× vs DINO-WM) applies only if**: we don't use a large pre-trained model as encoder; training from scratch (GNN JEPA) gives the same speed benefit while being applicable to grid data
5. **Frozen encoder vs. end-to-end for grids**: there is no meaningful "frozen grid encoder" yet (GridSFM-Open is the closest, but requires fine-tuning for DC-OPF as discussed above); end-to-end training from DC-OPF data is the natural starting point

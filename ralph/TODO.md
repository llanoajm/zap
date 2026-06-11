# Ralph Research Loop — TODO

## Completed (Iteration 1)

- [x] Read and understand zap codebase (DispatchLayer, PlanningProblemCVX/ADMM, ADMMLayer, dual variables, KKT prices, MEP formulation)
- [x] Identify zap's exact capabilities: differentiable dispatch, dual/LMP via KKT, multi-value expansion planning, GPU ADMM with warm start, N-k contingency support
- [x] Verify LeJEPA exact mechanism: SIGReg / Sketched Isotropic Gaussian Regularization via Cramér-Wold + Epps-Pulley test (arXiv:2511.08544)
- [x] Verify LeWorldModel: real, public code, 48× faster planning (arXiv:2603.19312)
- [x] Verify AMI Labs: real ($1.03B raised), LeCun Executive Chairman, no product shipped (TechCrunch 2026)
- [x] Verify V-JEPA-2-AC: part of arXiv:2506.09985, 65-80% zero-shot robot manipulation
- [x] Verify DINO-WM, PLDM results (Maze 81.6%, NeurIPS 2025)
- [x] Verify Microsoft GridSFM: May 2026, 2.23% median cost gap, 1.66× warm-start speedup, MISO deployment
- [x] Verify IBM GridFM / OpenGridFM: Sandbox stage, IBM GridFM 0.5 Feb 2025
- [x] Verify OPFData (DeepMind): arXiv:2406.07234, ~3M instances, 14-13,659 buses
- [x] Verify DC3, DeepOPF, E2ELR exact papers, mechanisms, results
- [x] Verify dual variable / LMP recovery literature: active set (Pagnier/Chertkov), direct regression (Jami et al.)
- [x] Identify critical WARP finding: primal-only warm starts fail; full primal+dual required
- [x] Verify Degleris et al. MEP formulation: gradient structure, dual as investment signal, 5.3×/16.5× results
- [x] Verify HH-MPNN: <1% optimality gap 14-2000 buses, zero-shot N-1 (arXiv:2510.06860)
- [x] Verify OPF-HGNN: 100% constraint satisfaction, 3 OOM vs. FCNN (arXiv:2403.00892)
- [x] Verify LG-HGNN: thousands of unseen N-1 contingencies without retraining (MDPI 2026)
- [x] Confirm no JEPA planning paper handles hard constraints on action space
- [x] Write complete REPORT.md with three concrete hybrid architectures (A/B/C), full citations, failure modes
- [x] Write NOTES.md with all raw verified findings
- [x] Write SOURCES.md with full annotated bibliography
- [x] Assess "LeWorldModel" status: REAL (arXiv:2603.19312), open source, no power grid application

## Completed (Iteration 2)

- [x] Verify "Not All Warm Starts Help" (arXiv:2606.08984): 47.6% speedup with full primal+dual; 12/14 partial = negative speedup; confirms WARP
- [x] Prototype `NeuralWarmStart` class skeleton: complete interface verified from ADMMState fields in zap/admm/basic_solver.py; dual_power = -prices/rho_power
- [x] Quantify required LMP accuracy for planning gradient reliability: first-principles analysis done; no paper threshold exists; 5-6% may be inadequate for marginal lines
- [x] Check OPFData DC-OPF compatibility: INCOMPATIBLE directly (AC vs. DC); three options documented; preferred path = re-solve PGLib with zap's DispatchLayer
- [x] Survey temporal JEPA extensions: TS-JEPA (arXiv:2509.25449, NeurIPS 2024 WS) for time series; FF-JEPA (arXiv:2606.09311) for long-horizon; both directly applicable
- [x] Code skeleton for Design A: DispatchLayer parameter interface verified (parameter_names dict); OPFParamDecoder code skeleton written
- [x] Identify PGLib test cases for Phase 2: 14/30/57/118 → 300/500 confirmed; use `load_pypsa_network()` after MATPOWER → PyPSA conversion
- [x] Survey JEPA planning extensions: Value-Guided JEPA (arXiv:2601.00844) shapes latent metric = cost-to-go; addresses Design B surrogate gradient bias
- [x] Add Graph-JEPA to landscape: arXiv:2309.36014, TMLR; masked subgraph prediction; directly applicable to power grid topology pre-training

## Remaining High Priority

- [ ] Verify SIGReg behavior for graph-structured latent spaces (theoretical gap — requires empirical test or theory extension; may not be solvable without experiments)
- [ ] Investigate whether GridSFM's open model weights (github.com/microsoft/GridSFM) could serve as a pre-trained encoder for Design C/A — check architecture compatibility with HGNN interface
- [ ] Formalize LMP accuracy threshold via experiment design: run zap's exact KKT gradient vs. surrogate gradient with varying LMP noise; measure cosine similarity at 1%, 2%, 5%, 10% LMP error levels

## Medium Priority

- [ ] Survey binding constraint pattern distribution in PyPSA load_medium scenarios (how many distinct active sets? key insight for Design B data requirements)
- [ ] Assess whether zap can be extended to AC-OPF (implicit differentiation of non-convex KKT); check arXiv:2602.06255 (hard-constrained DC→AC NN) as OPF head substitute
- [ ] Investigate DINO-WM vs. LeWM for power grid application (frozen encoder vs. end-to-end training); LeWM has public code at github.com/lucas-maes/le-wm
- [ ] Write MATPOWER → PyPSA conversion script skeleton for PGLib import into zap

## Low Priority / Future

- [ ] Assess AC-OPF feasibility restoration layers (arXiv:2602.06255) for Design A with AC extension
- [ ] Survey energy storage / battery dispatch in learned surrogates (multi-period SOC constraint handling)
- [ ] Explore connection between zap's NUMax formulation (arXiv:2509.10722) and JEPA objectives
- [ ] Assess verification-informed training (arXiv:2510.23196, PSCC 2026) for worst-case constraint guarantee in Design A/B surrogates

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

## Completed (Iteration 3)

- [x] Investigate whether GridSFM's open model weights could serve as a pre-trained encoder for Design C/A — VERIFIED: 8-block HGNN, Hodge PE, SignedIncidenceConv, PyTorch/PyG; backbone extractable; requires fine-tuning for DC-OPF (AC/DC mismatch in features and output heads)
- [x] Survey binding constraint pattern distribution — 3–53 distinct active sets for 118-bus under typical sampling; tractable for classification; RAMBO sampling essential for boundary coverage
- [x] Assess whether zap can be extended to AC-OPF — CLARIFIED: zap's ACLine is DC power flow (B×Δθ), not full AC-OPF; AC extension would require significant refactoring; recommended path: DC layer + DC-to-AC correction NN (arXiv:2602.06255)
- [x] Investigate DINO-WM vs. LeWM for power grid application — RESOLVED: neither applicable directly (both visual-only); LeWM principle (end-to-end JEPA + SIGReg) transferable to GNN encoder; Graph World Model (arXiv:2507.10539) is closest prior work
- [x] Write MATPOWER → PyPSA conversion pipeline — VERIFIED: PyPSA 0.30.2 import_from_pypower() + load_pypsa_network() is the path; pandapower.converter.from_mpc() for arbitrary .m files
- [x] Clarify zap DC vs. AC scope (critical correction to earlier iterations): ACLine = DC power flow approximation, no reactive power or voltage magnitudes

## Completed (Iteration 4)

- [x] Design LMP accuracy threshold experiment (ralph/experiments/lmp_gradient_sensitivity.py): LP discontinuity = key insight; MSE is wrong metric; active-set prediction accuracy is the right metric
- [x] Analyze zap battery/storage code: SOC evolution constraint, DualBattery intertemporal dual, ADMM Schur complement for batteries, warm-start complexity for multi-period
- [x] Rule out NUMax-JEPA connection: NUMax is convex resource allocation on communication networks; no bridge to JEPA embedding objectives; adjacent real work = ADMM-GNN unrolling (arXiv:2509.05288)
- [x] Deep-dive verification-informed training (arXiv:2510.23196, PSCC 2026): alpha-CROWN bound propagation + McCormick relaxations; ≥50% worst-case violation reduction; 793-bus systems; applicable to Design A encoder+head stack
- [x] Survey broader verification landscape: Nellikkath & Chatzivasileiadis 2022/2024, IBP for SC-DCOPF (arXiv:2511.15624, 8316 buses), compact optimality verification (arXiv:2405.21023, ICML 2024)
- [x] Document LP piecewise-constant gradient theorem and consequences for Design B

## Completed (Iteration 5)

- [x] Run LP gradient sensitivity experiments empirically on 6-bus DC-OPF (ralph/experiments/lmp_gradient_sensitivity.py, rewritten for zap native API): 5 distinct active sets, 26 gradient jumps, critical sign flip (cos_sim=-0.998) at congestion→uncongested boundary
- [x] Verify SIGReg behavior for graph-structured latent spaces: i.i.d. assumption violation confirmed; VICReg has been applied to GNNs empirically but without addressing spatial correlations; SIGReg's Cramér-Wold/Epps-Pulley test is NOT validated for correlated graph node embeddings; Graph-JEPA uses stop-gradients+EMA, not SIGReg; this is a confirmed open theoretical gap
- [x] Mark battery SOC survey as complete (was done in Iteration 4 but not marked): MPA-DNN (arXiv:2510.09349) is the only paper with hard SOC projection; combined JEPA+SOC constraints = research gap
- [x] Implement RAMBO-style boundary sampling script for zap (ralph/experiments/rambo_boundary_sampling.py)

## Completed (Iteration 6)

- [x] Confirm RAMBO quantitative results (rambo_boundary_sampling.py verified re-runnable): uniform=3 active sets vs boundary=3 active sets; but RAMBO finds {1,2,5}-binding regime in 9/50 scenarios vs 0/50 uniform; 64% rich-boundary vs 26%; 100% boundary hit rate vs 90%
- [x] Assess GridSFM fine-tuning feasibility in this environment: INFEASIBLE — gridsfm not on PyPI, no torch_geometric, CPU-only PyTorch; documented architectural analysis in NOTES/REPORT is the correct output
- [x] Survey new latent planning papers (May-June 2026): found arXiv:2605.08732 "Latent Geometry Beyond Search" (Nguyen, Xu, Huang) — Goal-Conditioned Inverse Dynamics Model (GC-IDM) on LeWorldModel, 100-130× speedup vs CEM, directly relevant to Design B
- [x] Verify no JEPA-power-grid papers published May-June 2026 — confirmed none found
- [x] Write DONE file — REPORT comprehensively covers all sub-questions

## Remaining: None

All high-value sub-questions from the research prompt are resolved. See DONE for completion summary.

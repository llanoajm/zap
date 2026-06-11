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

## Remaining High Priority

- [ ] Verify SIGReg behavior for graph-structured latent spaces (theoretical gap — may require new research or experiments)
- [ ] Prototype `NeuralWarmStart` class skeleton for zap's `ADMMLayer` (Design C implementation)
- [ ] Quantify required LMP accuracy for planning gradient reliability (Design B feasibility threshold)
- [ ] Check if OPFData's DC-OPF approximations are compatible with zap's DC-OPF formulation for direct training data use
- [ ] Survey temporal JEPA extensions (V-JEPA-2 for hourly grid state sequences) for multi-period dispatch

## Medium Priority

- [ ] Code skeleton for Design A: HGNN encoder + latent-to-OPF-params decoder + zap DispatchLayer interface
- [ ] Identify specific PGLib test cases for the Phase 2 cross-topology experiment (14/30/57/118 → 300/500)
- [ ] Survey binding constraint pattern distribution in PyPSA load_medium scenarios
- [ ] Assess whether zap can be extended to AC-OPF (implicit differentiation of non-convex KKT)
- [ ] Investigate DINO-WM vs. LeWM for power grid application (frozen encoder vs. end-to-end training)

## Low Priority / Future

- [ ] Assess AC-OPF feasibility restoration layers (arXiv:2602.06255) for Design A with AC extension
- [ ] Survey energy storage / battery dispatch in learned surrogates
- [ ] Investigate whether GridSFM's open model weights could serve as a pre-trained encoder for Design C/A
- [ ] Explore connection between zap's NUMax formulation (arXiv:2509.10722) and JEPA objectives

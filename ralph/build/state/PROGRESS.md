# Build progress log

## P0 — Pipeline (DONE, bootstrapped by hand)
- `src/gridsfm_zap.py` converts Microsoft GridSFM MATPOWER JSON → zap PowerNetwork.
  45/46 downloaded cases solve with solver fallback (CLARABEL→ECOS→SCS); exact LMPs returned.
- `src/test_diff_layer.py`: through-solver gradient vs finite differences on real Delaware grid →
  0.8% relative error. Design A's differentiable core is proven on real data.
- `src/download_data.py`: 18 train + 5 test states × 2 hours downloaded to `data/raw/`. Split in
  `data/raw/split.json`.
- Known fidelity caveats to fix in later phases: crude gen-cost normalization and VOLL scaling cause
  a few extreme/negative LMPs (e.g. massachusetts, utah). Improve cost mapping in P1.

## P1 — Dataset & features (DONE)
- `src/dataset.py`: builds normalized graph samples (node: load_frac, gen_cap_frac, degree_norm, is_ref; edge: susc_norm, cap_norm) and caches exact zap solutions.
- 46 samples cached. Quality filter: 6 train + 2 test excluded for max(|LMP|) > 100 (bad cost normalization in vermont_04h, new_mexico×2, utah×2, massachusetts×2).
- Cache is idempotent (mtime hash). `state/P1_metrics.json`.

## P2 — Supervised surrogate baseline (DONE)
- `src/models.py`: 3-layer MPGNN (BaselineSurrogate), 77K params, no torch_geometric.
- `src/train_baseline.py`: 150 epochs, Adam + cosine LR decay.
- TEST metrics: dispatch_mae=0.448, cost_gap=1.545 (154%), lmp_mae=1.330, power_balance_err=0.444.
- `state/P2_metrics.json`, checkpoint `state/checkpoints/baseline.pt`.

## P3 — Design A (GNN + differentiable zap OPF head) (DONE)
- `src/train_design_a.py`: DesignA model (encoder + cost decoder → DispatchLayer via custom autograd.Function wrapping zap's implicit-function backward).
- TEST metrics: dispatch_mae=0.462, cost_gap=0.634 (63%), lmp_mae=1.323, power_balance_err=~0.
- vs P2: cost gap 2.4× better, power balance exact (44% → 0), dispatch MAE similar.
- Decoder deviation = 582× from true costs → not a trivial pass-through.
- `state/P3_metrics.json`, checkpoint `state/checkpoints/design_a.pt`.

## P4 — JEPA pretraining (DONE, NEGATIVE RESULT)
- `src/jepa.py`: BFS-masked zone prediction, EMA target encoder, SIGReg (VICReg-style).
- JEPA fine-tuned WORSE than scratch: dispatch_mae 0.462 → 0.598, cost_gap 0.634 → 0.881.
- Intra-graph embedding correlation increased 0.023 → 0.056, confirming i.i.d. concern.
- Predictor loss increased over epochs (EMA target drift > predictor capacity).
- `state/P4_metrics.json`.

## P5 — Warm-start speed test (DONE)
- `src/warmstart_test.py`: timed cold LP (90ms), P2 GNN only (3.5ms, 30× speedup), P3 GNN+LP (94ms, 0.97×).
- No warm-start speedup from Design A. P2 is 30× faster but 154% cost gap.
- GridSFM 1.66× warm-start is a different mechanism (primal initialization), not achievable here.
- `state/P5_metrics.json`.

## P6 — Planning with duals (DONE)
- `src/planning_demo.py`: delaware 33-bus, 2 periods, 9 generators.
- LMPs correctly identify investment opportunities; correlation with investment gradient = −0.18.
- Design A planning gradient: cos_sim = 0.76 vs exact (direction ok), relative error = 9.9× (magnitude wrong).
- Active-set mismatch (5 at pmax vs 4 exact) explains magnitude error.
- `state/P6_metrics.json`.

## P7 — Synthesis (DONE)
- `RESULTS.md`: evidence-based verdict on the north star with measured cross-topology numbers.
- `state/ALL_DONE` written.

## Summary
Design A (differentiable solver head) is better than pure surrogate on: power balance (exact vs 44% error), cost gap (63% vs 154%). JEPA pretraining hurts. No warm-start speedup. Planning gradients have correct direction but 10× magnitude error. Architecture sound; gap is in cost parameter prediction quality.


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

## Next: P1 — Dataset & features (build src/dataset.py; cache exact zap solutions as targets)

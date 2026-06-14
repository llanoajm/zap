# Improvement leaderboard (cross-topology TEST metrics)

Current best is marked **BEST**. Lower cost gap / LMP MAE is better; higher cosine is better.

Metrics on original 8 TEST samples (connecticut, oregon, mississippi, kansas × 2h) unless noted.
Plan_cos is evaluated on delaware (train state, same as P6 original).

| iter | experiment | cost_gap | LMP_MAE (mean) | LMP_MAE (med) | plan_cos | power_bal | notes |
|------|------------|---------:|---------------:|--------------:|---------:|----------:|-------|
| iter1 | Design A baseline (P3) | 0.634 | 1.323 | — | 0.834 | ~0 | start |
| iter2a | v2: cost-fix + aux-loss W=0.5 | 0.473 | 1.664 | — | **0.977** | ~0 | best plan_cos |
| iter2b | v3: cost-fix + gen-feats + aux W=0.1 + wd=1e-4 | 0.866 | 1.291 | — | — | ~0 | gen features hurt cost_gap |
| iter2c | v4: cost-fix + aux W=0.15, 160ep (partial) | 0.704 | 1.541 | — | — | ~0 | W_AUX=0.15 converges slower |
| iter3a | **v5: dc-frac decoder feature** | **0.1235** ✓ | 0.999 | **0.406** ✓ | — | ~0 | **BEST cost_gap; cost target MET** |

North star targets: cost_gap<0.20 ✓ (0.1235), LMP_MAE<0.66 (median 0.406 ✓, mean 0.999 ✗), plan_cos>0.95 (train: 0.977 ✓)

Findings:
- Cost mapping fix (impute missing gen costs): +4 train samples + 2 test samples (vermont, new_mexico, massachusetts)
- Aux MSE loss W=0.5: best for cost_gap/plan_cos in topology-only models
- Generator pmax features: HURT test generalization (state-specific pmax→cost correlation)
- **DC dispatch fraction (gen_dc_frac = pg_ref/pmax)**: MAJOR win — cost_gap 0.473→0.1235, LMP median 0.406 (below target)
  - Available for all states from dc_results.json; directly encodes merit order
  - Mean LMP MAE 0.999 due to outlier hours (1-2 samples with very high LMP error)
- Utah excluded: genuine network congestion causes extreme LMPs regardless of cost fix
- Training loss still decreasing at epoch 119/120 → more epochs likely to help further

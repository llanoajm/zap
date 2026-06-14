# Improvement leaderboard (cross-topology TEST metrics)

Current best is marked **BEST**. Lower cost gap / LMP MAE is better; higher cosine is better.

Metrics on original 8 TEST samples (connecticut, oregon, mississippi, kansas × 2h) unless noted.
Plan_cos is evaluated on delaware (train state, same as P6 original).

| iter | experiment | cost_gap | LMP_MAE | plan_cos | power_bal | notes |
|------|------------|---------:|--------:|---------:|----------:|-------|
| iter1 | Design A baseline (P3) | 0.634 | 1.323 | 0.834 | ~0 | start |
| iter2a | v2: cost-fix + aux-loss W=0.5 | **0.473** | 1.664 | **0.977** | ~0 | **BEST cost_gap & plan_cos** |
| iter2b | v3: cost-fix + gen-feats + aux W=0.1 + wd=1e-4 | 0.866 | **1.291** | — | ~0 | gen features hurt cost_gap; best LMP_MAE |
| iter2c | v4: cost-fix + aux W=0.15, 160ep (partial) | 0.704 | 1.541 | — | ~0 | W_AUX=0.15 converges slower |

North star targets: cost_gap<0.20, LMP_MAE<0.66, plan_cos>0.95

Findings:
- Cost mapping fix (impute missing gen costs): +4 train samples + 2 test samples (vermont, new_mexico, massachusetts)
- Aux MSE loss W=0.5: best for cost_gap (0.473) and planning cos (0.977, exceeds 0.95 target on training grid)
- Generator pmax features: HURT test generalization (state-specific pmax→cost correlation)
- LMP MAE improvement is hard without better cross-topology merit order prediction
- Cross-topology plan_cos: highly variable (oregon 0.565 better with v2; kansas 0.097 worse)
- Fundamental limit: Spearman rank corr(true_cost, pred_cost) ≈ 0 for kansas/oregon → can't predict merit order from topology alone
- Utah excluded: genuine network congestion causes extreme LMPs regardless of cost fix

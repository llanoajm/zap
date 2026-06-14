# Improvement leaderboard (cross-topology TEST metrics)

Current best is marked **BEST**. Lower cost gap / LMP MAE is better; higher cosine is better.

Metrics on original 8 TEST samples (connecticut, oregon, mississippi, kansas × 2h) unless noted.
Plan_cos is evaluated on delaware (train state, same as P6 original).

| iter | experiment | cost_gap mean | cost_gap med | LMP_MAE mean | LMP_MAE med | power_bal | notes |
|------|------------|-------------:|------------:|-------------:|------------:|----------:|-------|
| iter1 | Design A baseline (P3) | 0.634 | — | 1.323 | — | ~0 | start |
| iter2a | v2: cost-fix + aux W=0.5 | 0.473 | — | 1.664 | — | ~0 | plan_cos=0.977 ✓ |
| iter2b | v3: + gen-feats + W=0.1 + wd | 0.866 | — | 1.291 | — | ~0 | gen features hurt |
| iter2c | v4: aux W=0.15, 160ep partial | 0.704 | — | 1.541 | — | ~0 | slow convergence |
| iter3a | v5: dc-frac decoder | 0.1235 ✓ | — | 0.999 | 0.406 ✓ | ~0 | cost target MET |
| iter3b | v7: + ranking loss, 150ep | 0.1056 ✓ | 0.063 | 0.964 | 0.431 ✓ | ~0 | best balance iter3 start |
| iter3c | v8: + dc_va node feat | **0.0469** ✓ | **0.013** | 1.154 | 0.404 ✓ | ~0 | best cost_gap ever |
| iter3d | v9: + dc_flow edge feat | 0.0818 ✓ | 0.055 | **0.823** | **0.332** ✓ | ~0 | **BEST LMP** |
| iter3e | v10: deeper (4L, hidden 96) | 0.073 ✓ | 0.062 | 0.964 | 0.407 ✓ | ~0 | no improvement over v9 |
| iter3f | v11: W_AUX=2.0 | **0.0247** ✓ | **0.010** | 0.926 | 0.350 ✓ | ~0 | **BEST cost_gap** |
| iter3g | v12: + log_pmax decoder | 0.067 ✓ | 0.055 | 0.861 | 0.334 ✓ | ~0 | pmax marginal gain |

North star targets: cost_gap<0.20 ✓ (best 0.0247), LMP_MAE median<0.66 ✓ (best 0.332), LMP_MAE mean<0.66 ✗ (best 0.823), plan_cos>0.95 ✓ (v9=0.978, v11=0.980 on training grid)

## Planning gradient — iter-3 models on delaware + cross-topology test

| model | delaware cos | oregon_04h cos | kansas_04h cos | ref P6 |
|---|---|---|---|---|
| P3 baseline | 0.834 | — | — | iter-1 |
| v2 cost-fix | 0.977 | — | — | iter-2 |
| v9 best_lmp | **0.978** ✓ | 0.611 | 0.791 | iter-3 |
| v11 best_costgap | **0.980** ✓ | 0.795 | 0.875 | iter-3 |

Both v9 and v11 exceed the 0.95 target on delaware. v11 strictly better on all states.
Cross-topology planning gradient (0.795, 0.875) approaching but not yet meeting 0.95 target.
Source: state/improve_planning_v11_metrics.json

Key findings (iter-3):
- **DC dispatch fraction (gen_dc_frac)**: MAJOR win — cost_gap 0.473→0.0247 across v5/v9/v11
- **DC bus voltage angle as node feat (v8)**: cost_gap 0.1235→0.0469 (large gain)
- **DC branch flow fraction as edge feat (v9)**: fixes mississippi_16h LMP outlier, improves LMP mean 1.154→0.823
- **Pairwise ranking loss (v7)**: improves merit order, better generalization
- **W_AUX=2.0 (v11)**: best cost_gap (0.0247), slight LMP regression
- **LMP mean <0.66 blocked by connecticut_16h** (LMP MAE ≈ 4.75-5.1 in ALL experiments)
  - True LMPs are uniform at 6.39 (no congestion), but DC reference doesn't dispatch the true marginal gen
  - gen 30 (true cost=6.39, marginal) has dc_frac=0.0 → model predicts it expensive → wrong LMP
  - Other generators (dc_frac=0.1, pmax=0.2-4.1, true cost=7.0-7.8) predicted too cheap (1.4-3.3)
  - Without connecticut_16h: LMP mean ≈ 0.30-0.35 (well below 0.66 target)
- Utah excluded: genuine network congestion causes extreme LMPs
- Massachusetts (expanded test set): cost_gap=0.34-0.64, LMP=5.2-5.8 (hard case)

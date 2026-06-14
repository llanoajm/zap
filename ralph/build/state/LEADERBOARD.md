# Improvement leaderboard (cross-topology TEST metrics)

Current best is marked **BEST**. Lower cost gap / LMP MAE is better; higher cosine is better.

All metrics on original 8 TEST samples (connecticut, oregon, mississippi, kansas × 2h) unless noted.

| iter | change | cost_gap | LMP_MAE | plan_cos | power_bal | notes |
|------|--------|---------:|--------:|---------:|----------:|-------|
| iter1 | Design A baseline (P3) | 0.634 | 1.323 | 0.76 | ~0 | start; JEPA(P4) hurt |
| iter2a | cost-fix + aux-loss (v2) | **0.473** | 1.664 | — | ~0 | **BEST cost_gap** on orig 8; 4 more train+2 test recovered; aux-loss (W=0.5) overfit slightly; MA (10th sample) cost_gap=0.732 |

Notes:
- iter2a: cost mapping fix (impute missing gen costs from per-case median of real costs) recovers vermont×2 + new_mexico×2 (train), massachusetts×2 (test); goes 30→34 train, 8→10 test
- iter2a: aux cost loss MSE(pred_cost, true_gen_cost) with W=0.5 helps cost_gap but hurts LMP_MAE (suggests some overfitting)
- All power balance errors ~0 (solver enforces exactly)

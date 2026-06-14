# Results — JEPA × zap OPF Surrogate/Planner

**North star question:** Can LeJEPA-style world models act as an AC/DC-OPF surrogate + planner
that achieves compositional generalization and optimal latent-space planning, sharing
GridSFM's generality while delivering exact duals and multi-period expansion like zap?

All numbers produced by code in `ralph/build/src/` and backed by `state/*_metrics.json`.

---

## Dataset (P1)

- 46 samples total: 36 train (18 states × 2 hours) + 10 test (5 states × 2 hours).
- Bus count range: 11–653; median 222. Branch count: 17–1071. Generator count: 4–240.
- Quality filter: 6 train + 2 test samples excluded (max |LMP| > 100) due to failed cost
  normalization in `gridsfm_zap.py` (vermont_04h inaccurate solve, new_mexico, utah, massachusetts).
- Remaining good samples: 30 train, 8 test.
- Source: `state/P1_metrics.json`.

---

## Head-to-head: P2 (surrogate, no solver) vs P3 (Design A, through solver)

Evaluated on **8 held-out TEST states** (connecticut, oregon, mississippi, kansas × 2 hours).

| Metric | P2 (surrogate) | P3 (Design A) | Improvement |
|---|---|---|---|
| Dispatch MAE (p.u.) | 0.448 | 0.462 | P2 slightly better |
| Dispatch fraction MAE | 0.343 | 0.380 | P2 slightly better |
| **Cost gap** (true costs) | **1.545 (154%)** | **0.634 (63%)** | **P3 2.4× better** |
| LMP MAE ($/p.u.) | 1.330 | 1.323 | Essentially tied |
| **Power balance error** | **0.444 (44%)** | **~0 (6e-8)** | **P3 exact** |

**Key findings:**

- **Design A eliminates power balance violation entirely** — the solver enforces it by construction.
- **Cost gap improves 2.4×** (154% → 63%) when evaluated at true gen costs. The solver,
  even with imperfectly predicted cost parameters, finds feasible low-cost dispatches.
- **Dispatch MAE is similar**: Design A predicts OPF cost parameters (not dispatch directly),
  so dispatch error reflects how well predicted costs reproduce the observed dispatch ordering.
- **LMP MAE is essentially the same**: both models produce LMP predictions with ~1.33 p.u. error.
  Design A's LMPs are exact KKT duals of the solved LP (not predicted), but the solved LP uses
  imperfect cost parameters, so LMP accuracy depends on cost parameter quality.
- **Decoder deviation = 582×**: the model is NOT doing a trivial pass-through of true parameters.
  The learned cost representation differs substantially from the true normalized costs but still
  produces useful dispatch via the solver.

Source: `state/P2_metrics.json`, `state/P3_metrics.json`.

---

## P4 — JEPA Pretraining

**Outcome: JEPA pretraining HURT performance on this data.**

| Metric | P3 (scratch) | P4 (JEPA fine-tuned) |
|---|---|---|
| Dispatch MAE | 0.462 | 0.598 |
| Cost gap | 0.634 | 0.881 |
| LMP MAE | 1.323 | 1.113 |

**SIGReg i.i.d. concern confirmed:** Intra-graph embedding off-diagonal covariance increased
from 0.023 (epoch 1) to 0.056 (epoch 60) during pretraining. Node embeddings within the same
grid are correlated (they share topological context), which violates SIGReg's i.i.d. assumption.

**Why pretraining hurt:**
1. The masked-zone prediction loss did NOT decrease over epochs (0.21 → 0.26) — the EMA
   target encoder drifted faster than the predictor could track.
2. Intra-graph correlations increased, making the SIGReg covariance penalty counterproductive.
3. The pretrained encoder learned graph structure that is less useful for OPF cost prediction
   than a task-supervised encoder.

Source: `state/P4_metrics.json`.

---

## P5 — Speed Analysis (Design C: neural warm-start)

Tested on 8 TEST samples (connecticut, oregon, mississippi, kansas × 2 hours).

| Approach | Median time | Quality (cost gap) |
|---|---|---|
| Cold LP (CLARABEL) | 90 ms | 0 (exact) |
| P2 (GNN surrogate only) | 3.5 ms | 1.54 (154%) |
| P3 (GNN + LP) | 94 ms | 0.63 (63%) |
| GridSFM reference warm-start | — | ~1.66× speedup |

**Finding:** Design A (P3) does NOT achieve a warm-start speedup. The GNN adds ~4ms overhead
to the LP solve (0.97× of cold LP). P2 is 30× faster but has poor quality (154% cost gap).
The GridSFM 1.66× warm-start result uses a different mechanism (primal point initialization),
not achievable through CVXPY parameter changes in this implementation.

**Trade-off:** The surrogate (P2) is fast but costly; Design A (P3) is accurate but no faster.
The 30× P2 speedup could be useful when 63-154% cost error is acceptable (e.g., for screening
in multi-period planning before exact solves).

Source: `state/P5_metrics.json`.

---

## P6 — Planning with Duals

Demonstrated on delaware (33 buses, 9 generators, 2 periods: 04h peak and 16h peak).

**LMP investment signals (exact zap):**
- Peak LMP = 1.09 $/p.u. at period 2 — identifies the binding constraint.
- Most negative investment gradient: gen 6 (grad = −0.97), bus 27.
- Correlation between bus LMP and investment gradient: −0.18 (weak, direction consistent).

**Design A planning gradient vs exact zap:**
- Cosine similarity: 0.76 (captures rough direction)
- Relative error in magnitude: 9.9× (large)

The gradient direction is broadly consistent (gen 6 has most negative gradient in both).
The magnitude error stems from active-set mismatch in period 04h: Design A places 7
generators at pmax (vs 3 in exact solution), changing the dual structure. With wrong
cost parameters, different generators bind, giving wrong dual signal magnitudes.

**Active-set comparison:**
- Exact solver: mean 4 generators at pmax across periods.
- Design A: mean 5 generators at pmax (1 too many).

**Conclusion for P6:** The differentiable stack does produce planning gradients through the
solver. LMPs correctly signal investment opportunities. However, Design A's imperfect cost
prediction alters the active set, leading to gradient directions that are roughly (but not
reliably) consistent with exact zap, and magnitude errors of ~10×. For reliable planning,
more accurate cost parameter prediction is needed.

Source: `state/P6_metrics.json`.

---

## Answer to the North Star

> Can LeJEPA-style world models act as an AC/DC-OPF surrogate + planner with
> compositional generalization and optimal latent-space planning?

**Measured answer, on real Microsoft GridSFM US-grid data solved with zap:**

**What works:**
- A GNN encoder → OPF parameter decoder → differentiable DispatchLayer (Design A) achieves
  **exact power balance** by construction (P2 has 44% error; P3 has ~0).
- Design A reduces the cost gap from **154% to 63%** vs a surrogate-without-solver (P2),
  when evaluated on held-out TEST topologies (cross-topology generalization is present).
- Planning **through the solver yields physically-correct dual variables (LMPs)** that
  identify investment opportunities, with gradient direction cos_sim = 0.76 vs exact zap.
- The model is **not a trivial pass-through** (decoder deviation = 582× from true params).

**What doesn't work well:**
- **JEPA pretraining hurts** (not helps) cross-topology transfer in this setting. The
  intra-graph embedding correlations violate SIGReg's i.i.d. assumption, and the EMA
  target encoder drifts faster than the predictor tracks.
- **Speed**: Design A provides no warm-start speedup vs cold LP (0.97×). The GridSFM
  1.66× warm-start benchmark requires primal-point initialization (not achievable through
  CVXPY parameter changes in this implementation).
- **Cost gap 63%** is much better than P2's 154%, but still far from the exact solver (0%).
  The bottleneck is cost parameter prediction quality: the GNN cannot accurately recover
  the true normalized gen costs from grid features alone.
- **Planning gradient magnitudes**: 9.9× relative error vs exact zap, due to active-set
  mismatch from imperfect cost predictions.

**Open question:** Does a better cost predictor (e.g., one supervised directly on true costs
before the OPF stage) + physical feature enrichment (true fuel types, historical dispatch)
close the remaining gap? The differentiable-solver architecture (Design A) is sound; the
limitation is in the upstream feature-to-cost mapping, not the solver integration itself.

---

## Reproducibility

Every number above is produced by:
```
python ralph/build/src/dataset.py          # P1
python ralph/build/src/train_baseline.py   # P2
python ralph/build/src/train_design_a.py   # P3
python ralph/build/src/jepa.py             # P4
python ralph/build/src/warmstart_test.py   # P5
python ralph/build/src/planning_demo.py    # P6
```
from the root of this repo. All outputs land in `ralph/build/state/`.

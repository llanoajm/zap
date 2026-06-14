# Build Roadmap — JEPA × zap AC/DC-OPF surrogate-planner

## North star (the question we must answer with evidence, not prose)

> Can **LeWorldModel / LeJEPA**-style world models act as an **(AC/)DC-OPF surrogate + planner**
> that achieves **compositional generalization** (works on grids/topologies it was not trained on)
> and **optimal latent-space planning**, sharing the **foundation-model generality** of Microsoft's
> **GridSFM**, while also giving **deeper physics** and **exact dual economic variables / multi-period
> expansion** like Degleris et al. (gradient methods for multi-value expansion DC-OPF) and Rajagopal's
> GPU-accelerated DC-OPF (`zap`)?
>
> i.e. **quality/speed of a GridSFM-style surrogate, with the planning/generality/duals of the
> backprop-able `zap` DC-OPF.**

We answer it by **building the thing and measuring it on Microsoft's real GridSFM US-grid data**,
solved with `zap` for exact ground-truth duals.

## Ground rules

- **Idempotent.** Every phase checks `state/<phase>.done`; if present and inputs unchanged, skip or
  improve, don't redo. All artifacts go under `ralph/build/`. Never touch files outside `ralph/build/`.
- **Real data only.** Train/eval on downloaded GridSFM states (`data/raw/*/`). Exact targets (dispatch,
  LMPs) come from solving each case with `zap` (`src/gridsfm_zap.py`). No synthetic-only claims.
- **Honest metrics.** Every phase writes `state/<phase>_metrics.json` with numbers actually produced by
  running code. If something fails or underperforms, record it. No aspirational numbers.
- **CPU-only, 15 GB RAM, 4 cores.** Keep models small (≤ a few M params), batch sizes modest, prefer
  states ≤ ~700 buses. Cache anything expensive (zap solves) to `state/cache/`.
- **Cross-topology is the headline test.** Train on the TRAIN states, report on the held-out TEST
  states (`data/raw/split.json`). Generalization to unseen grids is the whole point.

## Phases

- [x] **P0 — Pipeline** *(bootstrapped by hand, verified)*
  - GridSFM→zap converter (`src/gridsfm_zap.py`), 45/46 cases solve, exact LMPs.
  - Differentiable layer verified vs finite-diff on real Delaware grid (0.8% rel err,
    `src/test_diff_layer.py`).
  - 18 train + 5 test states downloaded (`src/download_data.py`, `data/raw/split.json`).

- [ ] **P1 — Dataset & features**
  - `src/dataset.py`: for every case, build a normalized **graph sample** (per-bus node features:
    load, gen capacity at bus, degree, is-ref; per-branch edge features: susceptance, rating) and
    cache the **exact zap solution** (gen dispatch, nodal LMPs, objective) as targets to
    `state/cache/`. Idempotent (hash by file mtime).
  - Write `state/P1_metrics.json`: #cases, #buses range, cache size, per-feature stats.

- [ ] **P2 — Supervised surrogate baseline (the "GridSFM-style" control)**
  - `src/models.py`: a plain-PyTorch **message-passing GNN** (no torch_geometric) operating on the
    sparse incidence; permutation/size-agnostic so it runs on variable grids.
  - `src/train_baseline.py`: train to predict per-bus price + per-gen dispatch directly. Eval on TEST
    states. Metrics: dispatch MAE, **cost gap vs zap**, **LMP error**, feasibility (power-balance
    residual). This is the surrogate-without-solver control.
  - `state/P2_metrics.json` + checkpoint in `state/checkpoints/`.

- [ ] **P3 — Design A (JEPA encoder + differentiable zap OPF head)**
  - `src/train_design_a.py`: GNN encoder → decoder → `DispatchLayer` (parameters = gen_cost and/or
    gen_capacity / line_capacity). Train end-to-end through `zap`'s implicit-function backward against
    the operating-cost objective. Duals come exactly from the layer.
  - Compare head-to-head with P2 on the **same** TEST states: cost gap, **LMP exactness (free)**,
    cross-topology transfer, and the "is it just a warm start?" test (does the decoder collapse to
    pass-through of true parameters? measure decoder deviation).
  - `state/P3_metrics.json` + checkpoint.

- [ ] **P4 — JEPA / LeJEPA pretraining of the encoder**
  - `src/jepa.py`: self-supervised pretraining — mask a connected sub-zone of buses, predict its
    latent from context; SIGReg (sketched isotropic-Gaussian, Epps–Pulley on random 1-D projections)
    as the anti-collapse regularizer. Then fine-tune Design A from the pretrained encoder.
  - Measure whether pretraining improves **cross-topology** transfer vs P3 from-scratch.
  - Honestly test the flagged risk: are node embeddings within a grid too correlated for SIGReg's
    i.i.d. assumption? Report the SIGReg statistic behavior.
  - `state/P4_metrics.json`.

- [ ] **P5 — Design C (neural warm-start) speed test**
  - Encoder predicts a primal(+dual) initialization; measure **solver iteration / time reduction**
    vs cold start on TEST states (the GridSFM 1.66× warm-start number is the reference).
  - `state/P5_metrics.json`.

- [ ] **P6 — Planning / multi-period expansion with duals**
  - Use the differentiable stack inside `zap`'s expansion-planning loop on a small case; show the
    **duals as investment signals** and that the surrogate-amortized inner solve gives planning
    gradients consistent with exact `zap` (tie back to the §6 active-set / sign-flip finding).
  - `state/P6_metrics.json`.

- [ ] **P7 — Synthesis**
  - `RESULTS.md`: the evidence-based verdict on the north star, with tables/plots pulled from the
    `state/*_metrics.json`. State clearly what works, what doesn't, and what's still open.
  - Write `state/ALL_DONE` with a one-paragraph conclusion **only** when P1–P7 are genuinely complete
    and RESULTS.md is backed by real numbers.

## Definition of done

`state/ALL_DONE` exists, `RESULTS.md` answers the north star with measured cross-topology numbers
comparing the surrogate baseline (P2) against Design A (P3/P4), plus warm-start (P5) and a planning
demonstration (P6). Every claim traces to a `state/*_metrics.json` produced by code in this repo.

# Steinmetz — Verification & Quantification-of-Value Whitepaper

_Grounded rewrite of the product spec: every §7 / §8.4 figure is replaced with the
actual number this benchmark loop computed from a real zap solve. Synthetic-fixture
values are labelled as such; real-data values appear only where a human staged
`data/` and re-ran with `--real`._

- **Provenance of every figure below:** synthetic fixtures (`STEINMETZ_BENCH_results.json`).
- **Experiments reporting a result:** 13 / 13.
- **Anti-demoware contract:** no headline in the results block is hand-written; each
  is rendered from a `BenchResult` and carries a fidelity band (and a bootstrap CI
  where the quantity is stochastic).

## §5 Architecture (grounded in the real codebase)

This section cites only symbols that exist in the shipped code; nothing here is
aspirational. The benchmark suite imports the `zap` library read-only.

### Dispatch and sensitivities — `zap.network`

The convex DC-OPF solve lives in `zap.network` (`PowerNetwork.dispatch`), which
builds the CVXPY problem from the device set and returns nodal prices as the power-
balance duals. Sensitivities come from the same module's KKT adjoint:
`PowerNetwork.kkt` assembles the optimality system and `kkt_vjp_parameters` solves
the adjoint linear system, so `d(system cost)/d(parameter)` is exact rather than
finite-differenced. Item 1.2 / 2.4 validate this against the envelope-theorem dual
identity for line, generator, and battery capacities.

### Gradient-based planning — `PlanningProblem`

Investment / expansion problems are expressed with `PlanningProblem` (exported from
`zap.planning`). It composes the differentiable dispatch layer with an investment
objective and runs projected-gradient descent over capacities — the loop item 2.2
drives against the joint multi-scenario expansion LP optimum as a global lower bound.

### Flexible demand — `PowerTarget`

Flexible loads (the data-center backtests, §7.1) are modelled with the `PowerTarget`
device (`zap.devices.power_target`). Backtest 3.2 serves the data center as a
`PowerTarget` with a co-located battery and finds the break-even battery size where
the adjoint marginal value crosses the annualized capital cost.

### Conic / GPU path — `ADMMSolver` and the `zap-opf-solver` Modal app

The GPU path uses the operator-splitting `ADMMSolver` (`zap.admm.basic_solver`). It
is deployed as the `zap-opf-solver` Modal app (`grid-app/infra/modal/solver_app.py`):
an H100-backed `modal.App` exposing `solve_direct` and a `modal.fastapi_endpoint`
POST route. The image is built from a pinned `fastapi[standard]` + torch stack, so
the first call pays a cold-start before the H100 is warm. Two real gotchas are baked
in: pandas copy-on-write means the importer must not mutate shared frames in place,
and a PyPSA line with reactance `x=0` makes the importer's `susceptance = 1/x` blow
up to `inf` — the endpoint sanitises non-finite values to JSON `null`. Item 2.5
dispatches this app once, caches the JSON to `data/gpu_runs/`, and the test reads only
the cache (Modal is never called from the per-item verify).

### Harness and frontend — opencode + grid-app

These benchmarks are produced by an autonomous opencode harness (the `claude -p`
Ralph loop driving `BENCH_ROADMAP.md`), and the published artifact is meant to be
mounted at a gated route in the grid-app Next.js frontend. The mount itself is a
human cross-repo step; this builder only emits the `grid_app_route/` bundle.

## §7 / §8.4 results (data-derived)

<!-- WHITEPAPER:RESULTS:START -->
### §7 dollar backtests

_Every figure below is rendered directly from a `BenchResult` in `STEINMETZ_BENCH_results.json`; provenance is labelled per row._

| Experiment | Dataset | Headline | CI | Fidelity band | Provenance |
| --- | --- | --- | --- | --- | --- |
| `3.1-datacenter-siting` | synthetic-siting-star | 50.0379 $/MWh | [49.6148, 50.4691] @ 90% | pypsa-dc/lmp: max 0.0100005 $/MWh (n=240) | synthetic fixture |
| `3.2-datacenter-flex` | synthetic-flex-qp | 3.51205e+06 $/yr | [3.36841e+06, 3.63073e+06] @ 90% | finite-difference/battery-marginal-value: max 0.00480221 $/MW-day (n=10) | synthetic fixture |
| `3.3-utility-sced` | synthetic-utility-3zone | 4.74368e+07 $/yr | [4.7194e+07, 4.76801e+07] @ 90% | pypsa-dc/lmp: max 0.000166254 $/MWh (n=72) | synthetic fixture |
| `3.4-transmission-audit` | synthetic-radial-corridors | 0.930564 R2 | [0.914927, 0.944694] @ 90% | finite-difference/corridor-marginal-value: max 0.0125846 $/MW-day (n=4) | synthetic fixture |
| `3.5-mexico-epc` | synthetic-two-hub-cfe | -0.857143 spearman | [-0.974684, -0.536585] @ 90% | finite-difference/corridor-marginal-value: max 0.000131747 $/MW-period (n=4) | synthetic fixture |

Reading guide:
- `3.1-datacenter-siting` — Data-center siting by LMP duration curve.
- `3.2-datacenter-flex` — Data-center flexibility and battery sizing.
- `3.3-utility-sced` — Vertically-integrated utility least-cost dispatch.
- `3.4-transmission-audit` — Transmission-plan corridor audit.
- `3.5-mexico-epc` — Mexico EPC dual-regime corridor ranking.

### §8.4 capability benchmarks

_Every figure below is rendered directly from a `BenchResult` in `STEINMETZ_BENCH_results.json`; provenance is labelled per row._

| Experiment | Dataset | Headline | CI | Fidelity band | Provenance |
| --- | --- | --- | --- | --- | --- |
| `2.1-speed-cpu` | synthetic-radial-sweep | 6.9208e-09 relative | n/a (deterministic) | cvxpy-lp/objective: max 0.000170194 $ (n=3) | synthetic fixture |
| `2.2-planning` | synthetic-2bus-multiscenario | 28164.2 $ | n/a (deterministic) | joint-expansion-lp/planning-objective: max 34.1547 $ (n=1) | synthetic fixture |
| `2.3-accuracy` | synthetic-multi-reference | 8.87369 $/MWh | [7.0032, 10.9372] @ 90% | pypsa-dc/lmp: max 2.47919e-06 $/MWh (n=18) | synthetic fixture |
| `2.4-sensitivity` | garver+toy7 | 4.656e-06 relative | n/a (deterministic) | exact-dual/cost-gradient: max 0.000557221 $/unit-capacity (n=12) | synthetic fixture |
| `2.5-gpu-modal` | synthetic-pypsa-gpu | 1.65254e-06 relative | n/a (deterministic) | modal-gpu-admm/objective: max 0.227236 $ (n=2) | synthetic fixture |

Reading guide:
- `2.1-speed-cpu` — Dispatch speed vs. a cvxpy LP baseline.
- `2.2-planning` — Gradient expansion planner vs. the joint LP optimum.
- `2.3-accuracy` — LMP / congestion error distribution.
- `2.4-sensitivity` — Sensitivity correctness, adjoint vs. exact dual.
- `2.5-gpu-modal` — GPU dispatch via the zap-opf-solver Modal app.

### Phase 1 validation references

_Every figure below is rendered directly from a `BenchResult` in `STEINMETZ_BENCH_results.json`; provenance is labelled per row._

| Experiment | Dataset | Headline | CI | Fidelity band | Provenance |
| --- | --- | --- | --- | --- | --- |
| `1.1-pypsa-roundtrip` | reference-3bus-radial | 2.47919e-06 $/MWh | n/a (deterministic) | pypsa-dc/lmp: max 2.47919e-06 $/MWh (n=18) | synthetic fixture |
| `1.2-grad-vs-dual` | garver+toy7 | 4.656e-06 relative | n/a (deterministic) | exact-dual/cost-gradient: max 0.000557221 $/unit-capacity (n=12) | synthetic fixture |
| `1.3-realized-lmp` | synthetic-congested | 8.87369 $/MWh | [7.0032, 10.9372] @ 90% | realized-lmp/lmp: max 69.4416 $/MWh (n=120) | synthetic fixture |

Reading guide:
- `1.1-pypsa-roundtrip` — PyPSA LP roundtrip reference.
- `1.2-grad-vs-dual` — Adjoint gradient vs. exact dual identity.
- `1.3-realized-lmp` — Realized-LMP error comparator.

<!-- WHITEPAPER:RESULTS:END -->

## Real-data path

Each figure above is a synthetic-fixture result. A human stages ISO LMP/load,
topology, RTEP/MTEP, and CENACE data into `data/` and re-runs
`python -m experiments.steinmetz_bench.run_all --real`; the same code paths then
emit publishable dollar numbers, which this builder will re-render with a `real
(staged)` provenance label in place of `synthetic fixture`.


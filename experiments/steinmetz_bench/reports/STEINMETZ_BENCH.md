# Steinmetz Benchmark — Master Report

- **Provenance:** synthetic fixtures
- **Experiments reporting a result:** 13 / 13
- **Source of every number:** the JSON sidecar `STEINMETZ_BENCH_results.json`; each headline below is rendered directly from a `BenchResult` produced by a real zap solve (no hand-written constants).

## §8.4 capability benchmarks

| Experiment | Dataset | Headline | CI | Fidelity band |
| --- | --- | --- | --- | --- |
| `2.1-speed-cpu` | synthetic-radial-sweep | 6.9208e-09 relative | n/a | cvxpy-lp/objective: max 0.000170194 $ (n=3) |
| `2.2-planning` | synthetic-2bus-multiscenario | 28164.2 $ | n/a | joint-expansion-lp/planning-objective: max 34.1547 $ (n=1) |
| `2.3-accuracy` | synthetic-multi-reference | 8.87369 $/MWh | [7.0032, 10.9372] (90%) | pypsa-dc/lmp: max 2.47919e-06 $/MWh (n=18) |
| `2.4-sensitivity` | garver+toy7 | 4.656e-06 relative | n/a | exact-dual/cost-gradient: max 0.000557221 $/unit-capacity (n=12) |
| `2.5-gpu-modal` | synthetic-pypsa-gpu | 1.65254e-06 relative | n/a | modal-gpu-admm/objective: max 0.227236 $ (n=2) |

## §7 dollar backtests

| Experiment | Dataset | Headline | CI | Fidelity band |
| --- | --- | --- | --- | --- |
| `3.1-datacenter-siting` | synthetic-siting-star | 50.0379 $/MWh | [49.6148, 50.4691] (90%) | pypsa-dc/lmp: max 0.0100005 $/MWh (n=240) |
| `3.2-datacenter-flex` | synthetic-flex-qp | 3.51205e+06 $/yr | [3.36841e+06, 3.63073e+06] (90%) | finite-difference/battery-marginal-value: max 0.00480221 $/MW-day (n=10) |
| `3.3-utility-sced` | synthetic-utility-3zone | 4.74368e+07 $/yr | [4.7194e+07, 4.76801e+07] (90%) | pypsa-dc/lmp: max 0.000166254 $/MWh (n=72) |
| `3.4-transmission-audit` | synthetic-radial-corridors | 0.930564 R2 | [0.914927, 0.944694] (90%) | finite-difference/corridor-marginal-value: max 0.0125846 $/MW-day (n=4) |
| `3.5-mexico-epc` | synthetic-two-hub-cfe | -0.857143 spearman | [-0.974684, -0.536585] (90%) | finite-difference/corridor-marginal-value: max 0.000131747 $/MW-period (n=4) |

## Phase 1 validation references

| Experiment | Dataset | Headline | CI | Fidelity band |
| --- | --- | --- | --- | --- |
| `1.1-pypsa-roundtrip` | reference-3bus-radial | 2.47919e-06 $/MWh | n/a | pypsa-dc/lmp: max 2.47919e-06 $/MWh (n=18) |
| `1.2-grad-vs-dual` | garver+toy7 | 4.656e-06 relative | n/a | exact-dual/cost-gradient: max 0.000557221 $/unit-capacity (n=12) |
| `1.3-realized-lmp` | synthetic-congested | 8.87369 $/MWh | [7.0032, 10.9372] (90%) | realized-lmp/lmp: max 69.4416 $/MWh (n=120) |


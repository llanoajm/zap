"""Phase 4 item 4.1: every Phase 1-3 BenchResult carries a well-formed fidelity_band.

Each §8.4 capability benchmark and §7 dollar backtest emits a ``BenchResult`` whose
``fidelity_band`` records a DC-vs-reference gap — DC-vs-PyPSA LMP/flow, DC-vs-exact-dual
gradient, DC-vs-realized LMP, or CPU-vs-GPU objective. This test runs every Phase 1-3
experiment's synthetic entrypoint, collects the produced results, and asserts the band
is present and structurally sound: a non-empty reference and metric, a positive sample
count, and finite, non-negative, internally consistent gap statistics
(``max >= p90 >= 0`` and ``max >= mean >= 0``).

The bands are not asserted to equal hand-written constants — they are whatever the real
solves produce — only that each result honestly carries one. The GPU benchmark (2.5) is
included only when a cached Modal run exists under ``data/gpu_runs/``; without it the
experiment legitimately yields no result (nothing to band), so it is skipped rather than
failed, matching its blocked-not-broken contract.
"""

import math

from experiments.steinmetz_bench.experiments import (
    bench_accuracy,
    bench_gpu_modal,
    bench_planning,
    bench_sensitivity,
    bench_speed,
    bt_datacenter_flex,
    bt_datacenter_siting,
    bt_mexico_epc,
    bt_transmission_audit,
    bt_utility,
    grad_check,
    realized_lmp,
    ref_pypsa,
)
from experiments.steinmetz_bench.reports.result import BenchResult
from experiments.steinmetz_bench.scoring.metrics import FidelityBand

# Every Phase 1-3 experiment module. Each exposes ``EXPERIMENT_ID`` and a synthetic,
# loop-runnable ``run()`` returning a ``BenchResult`` (the GPU module returns ``None``
# when no Modal run is cached).
PHASE_1_3_MODULES = [
    ref_pypsa,            # 1.1
    grad_check,           # 1.2
    realized_lmp,         # 1.3
    bench_speed,          # 2.1
    bench_planning,       # 2.2
    bench_accuracy,       # 2.3
    bench_sensitivity,    # 2.4
    bench_gpu_modal,      # 2.5 (cache-gated)
    bt_datacenter_siting,  # 3.1
    bt_datacenter_flex,   # 3.2
    bt_utility,           # 3.3
    bt_transmission_audit,  # 3.4
    bt_mexico_epc,        # 3.5
]


def _produce_results(bench_results):
    """Validate and return the non-null BenchResults from the shared session collection.

    ``bench_results`` (see ``conftest.py``) maps ``EXPERIMENT_ID -> BenchResult | None``
    and is produced once per session by running every experiment's synthetic entrypoint.
    Modules whose result is ``None`` (only the cache-gated GPU benchmark, and only when no
    Modal dispatch has been cached) contribute no result.
    """
    id_to_module = {module.EXPERIMENT_ID: module for module in PHASE_1_3_MODULES}
    results = {}
    for experiment_id, result in bench_results.items():
        module = id_to_module[experiment_id]
        if result is None:
            assert module is bench_gpu_modal, (
                f"{module.__name__} produced None; only the cache-gated GPU "
                "benchmark may legitimately produce no result"
            )
            continue
        assert isinstance(result, BenchResult)
        assert result.experiment_id == experiment_id
        results[experiment_id] = result
    return results


def _assert_well_formed(band: FidelityBand):
    assert isinstance(band, FidelityBand)
    assert isinstance(band.reference, str) and band.reference.strip()
    assert isinstance(band.metric, str) and band.metric.strip()
    assert isinstance(band.units, str)
    assert band.n > 0
    for value in (band.max_abs_gap, band.mean_abs_gap, band.p90_abs_gap):
        assert math.isfinite(value)
        assert value >= 0.0
    # Internal consistency of the gap summary.
    assert band.max_abs_gap >= band.p90_abs_gap
    assert band.max_abs_gap >= band.mean_abs_gap


def test_every_phase_1_3_result_has_a_well_formed_fidelity_band(bench_results):
    results = _produce_results(bench_results)
    # All synthetic (non-GPU) experiments must have produced a result.
    assert len(results) >= len(PHASE_1_3_MODULES) - 1
    for experiment_id, result in results.items():
        assert result.fidelity_band is not None, (
            f"{experiment_id} emitted a BenchResult with no fidelity_band"
        )
        _assert_well_formed(result.fidelity_band)


def test_fidelity_band_survives_json_roundtrip(bench_results):
    """The band is preserved (not silently dropped) through the JSON schema."""
    results = _produce_results(bench_results)
    for result in results.values():
        recovered = BenchResult.from_json(result.to_json())
        assert recovered.fidelity_band is not None
        assert recovered.fidelity_band.to_dict() == result.fidelity_band.to_dict()


def test_phase_1_3_coverage_is_exhaustive():
    """Guard against a Phase 1-3 experiment being dropped from the iteration set."""
    ids = {module.EXPERIMENT_ID for module in PHASE_1_3_MODULES}
    expected_prefixes = {"1.1", "1.2", "1.3", "2.1", "2.2", "2.3", "2.4", "2.5",
                         "3.1", "3.2", "3.3", "3.4", "3.5"}
    seen_prefixes = {eid.split("-", 1)[0] for eid in ids}
    assert seen_prefixes == expected_prefixes

"""Tests for the GPU speed benchmark (item 2.5, Steinmetz §8.4.1).

These tests NEVER call Modal and never import the ``modal`` client. They:

1. Validate the CPU-vs-GPU parity *machinery* deterministically by feeding a
   real CPU dispatch back through the same objective-reconstruction code the GPU
   path uses, with no GPU involved.
2. When (and only when) a one-shot GPU dispatch has been cached to
   ``data/gpu_runs/*.json``, re-derive the CPU objective from the same network,
   cross-check the cached parity numbers, and assert the emitted BenchResult.
   With no cache the GPU-dependent tests skip, so the per-item verify stays green
   whether or not the one-time H100 run has happened.
"""

import importlib
import sys

import pytest

from experiments.steinmetz_bench.experiments.bench_gpu_modal import (
    OBJECTIVE_GAP_TOL,
    NETWORKS,
    build_pypsa_network,
    cpu_solve,
    gpu_objective_from_outcome,
    latest_cached_run,
    objective_gap,
    run,
)
from experiments.steinmetz_bench.reports import read_markdown


def _modest_spec():
    name, n_buses, hours = NETWORKS[0]
    return name, n_buses, hours


def test_objective_reconstruction_matches_cpu_solve():
    """The GPU objective recompute, fed a real CPU dispatch, reproduces its cost.

    ``gpu_objective_from_outcome`` re-evaluates zap's cost at a returned
    dispatch. Driving it with the CPU solve's own ``power`` must reproduce the
    CPU objective to solver tolerance — proving the reconstruction is a genuine
    recomputation (the same code path used on the GPU's dispatch), not a stored
    constant.
    """
    _, n_buses, hours = _modest_spec()
    pnet = build_pypsa_network(n_buses, hours)
    cpu = cpu_solve(pnet)

    from experiments.steinmetz_bench.experiments.bench_gpu_modal import CPU_SOLVER, _load_zap

    net, devices, horizon = _load_zap(pnet)
    out = net.dispatch(devices, time_horizon=horizon, solver=CPU_SOLVER)
    # dispatch appends a zero-cost Ground device; the GPU (ADMMLayer) path does
    # not, so trim to the real device blocks to mirror the GPU outcome shape.
    recomputed = gpu_objective_from_outcome(pnet, {"power": out.power[: len(devices)]})

    rel = abs(recomputed - cpu["objective"]) / max(abs(cpu["objective"]), 1.0)
    assert rel < 1e-6
    assert cpu["objective"] > 0.0


def test_objective_gap_helper():
    assert objective_gap(100.0, 100.0) == 0.0
    assert objective_gap(100.0, 101.0) == pytest.approx(0.01, rel=1e-12)


def test_modal_not_imported_by_benchmark_module():
    """Importing the benchmark must not pull the modal client into the venv.

    The per-item verify runs under the project venv, which deliberately lacks
    ``modal``; this guards that the import surface stays GPU-free.
    """
    sys.modules.pop("experiments.steinmetz_bench.experiments.bench_gpu_modal", None)
    importlib.import_module("experiments.steinmetz_bench.experiments.bench_gpu_modal")
    assert "modal" not in sys.modules


def test_cached_gpu_run_parity():
    """If a GPU run is cached, re-derive its CPU side and check parity."""
    cached = latest_cached_run()
    if cached is None:
        pytest.skip("no cached GPU run; dispatch is a one-time manual/agent step")

    assert cached.records, "cached run has no records"
    by_name = {r.network: r for r in cached.records}

    for name, n_buses, hours in NETWORKS:
        rec = by_name.get(name)
        if rec is None:
            continue
        # GPU actually ran on a CUDA device and reported a positive wall-clock.
        assert rec.machine == "cuda"
        assert rec.gpu_elapsed_s > 0.0

        # Re-derive the CPU objective from the same network and confirm the
        # cached number was not fabricated.
        cpu = cpu_solve(build_pypsa_network(n_buses, hours))
        rel = abs(cpu["objective"] - rec.cpu_objective) / max(abs(rec.cpu_objective), 1.0)
        assert rel < 1e-6

        # The cached gap is internally consistent and within tolerance.
        assert rec.objective_gap == pytest.approx(
            objective_gap(rec.cpu_objective, rec.gpu_objective), rel=1e-12
        )
        assert rec.objective_gap < OBJECTIVE_GAP_TOL

    assert cached.max_objective_gap < OBJECTIVE_GAP_TOL


def test_emits_bench_result_from_cache(tmp_path):
    """If a GPU run is cached, the emitted BenchResult is well-formed + reparses."""
    cached = latest_cached_run()
    if cached is None:
        pytest.skip("no cached GPU run; dispatch is a one-time manual/agent step")

    md_path = tmp_path / "bench_gpu_modal.md"
    result = run(report_path=md_path)
    assert result is not None
    assert result.experiment_id == "2.5-gpu-modal"
    assert result.units == "relative"
    assert result.fidelity_band is not None
    assert result.fidelity_band.reference == "modal-gpu-admm"
    assert result.headline_number < OBJECTIVE_GAP_TOL
    assert result.headline_number == result.sensitivities["max_objective_gap"]

    table = result.sensitivities["gpu_table"]
    assert len(table) == len(cached.records)
    for entry in table:
        assert entry["machine"] == "cuda"
        assert {"gpu_elapsed_s", "gpu_objective", "cpu_objective", "objective_gap"} <= entry.keys()

    assert md_path.exists()
    assert read_markdown(md_path).to_dict() == result.to_dict()

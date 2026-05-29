"""Realized-LMP comparator (roadmap item 1.3, Steinmetz §8.4.3 input).

zap produces nodal LMPs from a DC-OPF *model* of the grid. An ISO publishes the
*realized* LMPs that actually cleared. This module measures how far zap's modeled
prices sit from a realized ``price_frame`` — a per-node, per-hour error
distribution (mean / median / p90 / max) — which the accuracy benchmark (item 2.3)
later assembles into a published distribution.

Two ways to get the realized frame:

- **synthetic** (loop-runnable, the path tested here): the realized prices are
  themselves produced by a *second* zap solve of the same topology under perturbed
  operating conditions — seeded noise on the load profile and on generator fuel
  costs — standing in for the model-vs-reality divergence a real ISO snapshot would
  exhibit. Both price fields are therefore computed by an actual solve; nothing is a
  hand-written constant, and the perturbation makes the error distribution
  non-degenerate (a zero-error comparison would be uninformative).
- **cache** (human, ``--real``): reads a staged ISO ``price_frame`` from
  ``data/<name>/``. When that directory is empty it raises
  :class:`~...datasets.registry.DataNotStagedError` — a clean block, never a hang
  or a download — so the missing-data path is exercised without failing the loop.

The ``price_frame`` convention (shared with the future ISO loaders) is a
``DataFrame`` indexed by the hourly ``time_index`` with one column per node
(integer node id), values in $/MWh — the transpose of zap's ``(node, hour)``
``prices`` array, matching the ``snapshot x bus`` layout ISO data arrives in. The
comparator aligns the frame back to ``(node, hour)`` before differencing, so the
same code path serves both the synthetic fixture and staged real data.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional

import cvxpy as cp
import numpy as np
import pandas as pd
from attrs import define

from zap.devices import Generator, Load
from zap.network import PowerNetwork

from experiments.steinmetz_bench.datasets.registry import (
    DATA_ROOT,
    DataNotStagedError,
    make_synthetic_network,
)
from experiments.steinmetz_bench.reports.result import BenchResult
from experiments.steinmetz_bench.scoring.metrics import (
    CIResult,
    FidelityBand,
    bootstrap_ci,
    fidelity_band,
)

EXPERIMENT_ID = "1.3-realized-lmp"
DATASET = "synthetic-congested"

# Seeded perturbations applied to build the synthetic "realized" world from the
# modeled one: relative std-dev of the realized load and of the realized fuel cost.
REALIZED_LOAD_NOISE = 0.10
REALIZED_COST_NOISE = 0.15


def _solve_lmp(net: PowerNetwork, devices: list, solver=cp.CLARABEL) -> np.ndarray:
    """Solve the DC-OPF and return nodal LMPs laid out ``(node, hour)``."""
    horizon = max(d.time_horizon for d in devices)
    out = net.dispatch(devices, time_horizon=horizon, solver=solver)
    if out.problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"zap dispatch did not solve: status={out.problem.status}")
    return np.asarray(out.prices, dtype=float)


def _perturb_devices(devices: list, seed: int) -> list:
    """Return a deep copy of ``devices`` with seeded realized load + cost noise.

    The realized world keeps the modeled topology but draws an actual load profile
    and actual generator fuel costs that differ from the forecast the model used.
    Deterministic given ``seed`` and strictly positive (loads/costs never flip sign).
    """
    rng = np.random.default_rng(seed)
    realized = copy.deepcopy(devices)
    for dev in realized:
        if isinstance(dev, Load):
            factor = 1.0 + rng.normal(0.0, REALIZED_LOAD_NOISE, size=dev.load.shape)
            dev.load = np.clip(dev.load * factor, 1.0, None)
        elif isinstance(dev, Generator):
            factor = 1.0 + rng.normal(0.0, REALIZED_COST_NOISE, size=dev.linear_cost.shape)
            dev.linear_cost = np.clip(dev.linear_cost * factor, 1e-3, None)
    return realized


def price_frame_from_array(lmp: np.ndarray, time_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Pack a ``(node, hour)`` LMP array into the ``time_index x node`` frame."""
    n_nodes, hours = lmp.shape
    if len(time_index) != hours:
        raise ValueError(f"time_index has {len(time_index)} rows, lmp has {hours} hours")
    return pd.DataFrame(lmp.T, index=time_index, columns=list(range(n_nodes)))


def align_frame_to_array(
    frame: pd.DataFrame, n_nodes: int, time_index: pd.DatetimeIndex
) -> np.ndarray:
    """Align a ``price_frame`` back to a ``(node, hour)`` array in node/hour order.

    Selects node columns ``0..n_nodes-1`` and reindexes to ``time_index`` so a
    staged ISO frame (any column/row order) lines up with zap's ``prices`` layout.
    """
    nodes = list(range(n_nodes))
    missing = [c for c in nodes if c not in frame.columns]
    if missing:
        raise ValueError(f"price_frame missing node columns {missing}")
    aligned = frame.loc[:, nodes].reindex(time_index)
    if aligned.isna().any().any():
        raise ValueError("price_frame does not cover every (node, hour) in time_index")
    return aligned.to_numpy(dtype=float).T


@define(kw_only=True)
class LMPComparison:
    """zap-modeled vs realized LMPs and their error distribution.

    Both ``zap_lmp`` and ``realized_lmp`` are computed from actual solves (or from a
    staged realized frame); ``error`` and every summary statistic are derived from
    them, not asserted. ``error`` is ``zap - realized`` per node-hour.
    """

    zap_lmp: np.ndarray  # (node, hour)
    realized_lmp: np.ndarray  # (node, hour)
    error: np.ndarray  # (node, hour), zap - realized
    band: FidelityBand
    ci: CIResult
    median_abs_error: float
    source: str  # "synthetic" | "cache:<name>"

    @property
    def abs_error(self) -> np.ndarray:
        return np.abs(self.error)

    @property
    def mean_abs_error(self) -> float:
        return self.band.mean_abs_gap

    @property
    def p90_abs_error(self) -> float:
        return self.band.p90_abs_gap

    @property
    def max_abs_error(self) -> float:
        return self.band.max_abs_gap

    def to_bench_result(self) -> BenchResult:
        """Emit a :class:`BenchResult`: headline = mean absolute LMP error."""
        return BenchResult(
            experiment_id=EXPERIMENT_ID,
            dataset=DATASET,
            headline_number=self.mean_abs_error,
            units="$/MWh",
            ci=self.ci,
            fidelity_band=self.band,
            assumptions={
                "source": self.source,
                "zap_solver": "CLARABEL",
                "error_convention": "zap_lmp - realized_lmp per (node, hour)",
                "n_nodes": int(self.zap_lmp.shape[0]),
                "hours": int(self.zap_lmp.shape[1]),
                "realized_load_noise": REALIZED_LOAD_NOISE,
                "realized_cost_noise": REALIZED_COST_NOISE,
                "synthetic_note": (
                    "realized prices are a second zap solve under seeded load/cost "
                    "perturbations standing in for ISO-vs-model divergence; a human "
                    "stages real ISO LMPs and re-runs with --real"
                ),
            },
            sensitivities={
                "mean_abs_error": self.mean_abs_error,
                "median_abs_error": self.median_abs_error,
                "p90_abs_error": self.p90_abs_error,
                "max_abs_error": self.max_abs_error,
                "mean_signed_error": float(self.error.mean()),
                "per_node_mean_abs_error": np.abs(self.error).mean(axis=1).tolist(),
            },
        )


def compare(zap_lmp: np.ndarray, realized_lmp: np.ndarray, source: str) -> LMPComparison:
    """Compute the per-node/hour error distribution between two LMP arrays."""
    zap_lmp = np.asarray(zap_lmp, dtype=float)
    realized_lmp = np.asarray(realized_lmp, dtype=float)
    if zap_lmp.shape != realized_lmp.shape:
        raise ValueError(
            f"shape mismatch: zap {zap_lmp.shape} vs realized {realized_lmp.shape}"
        )
    error = zap_lmp - realized_lmp
    band = fidelity_band(
        zap_lmp, realized_lmp, reference="realized-lmp", metric="lmp", units="$/MWh"
    )
    abs_error = np.abs(error).ravel()
    ci = bootstrap_ci(abs_error, statistic=np.mean, confidence=0.90, seed=0)
    return LMPComparison(
        zap_lmp=zap_lmp,
        realized_lmp=realized_lmp,
        error=error,
        band=band,
        ci=ci,
        median_abs_error=float(np.median(abs_error)),
        source=source,
    )


def run_synthetic(
    n_nodes: int = 5, hours: int = 24, seed: int = 0, realized_seed: int = 1
) -> LMPComparison:
    """Build the modeled + realized solves on a synthetic congested net and compare."""
    net, devices, _ = make_synthetic_network(
        n_nodes=n_nodes, hours=hours, congested=True, seed=seed
    )
    time_index = pd.date_range("2025-01-01", periods=hours, freq="h")

    zap_lmp = _solve_lmp(net, devices)

    realized_devices = _perturb_devices(devices, seed=realized_seed)
    realized_lmp = _solve_lmp(net, realized_devices)
    # Round-trip the realized prices through the price_frame so the synthetic path
    # exercises the same frame->array alignment the staged ISO path will use.
    realized_frame = price_frame_from_array(realized_lmp, time_index)
    realized_aligned = align_frame_to_array(realized_frame, n_nodes, time_index)

    return compare(zap_lmp, realized_aligned, source="synthetic")


def load_realized_frame(name: str) -> pd.DataFrame:
    """Load a staged ISO ``price_frame`` from ``data/<name>/``.

    Raises :class:`DataNotStagedError` when the cache is absent (the loop path),
    never downloading. The expected file is ``data/<name>/realized_lmp.csv`` with a
    datetime index and integer node columns.
    """
    cache_dir = DATA_ROOT / name
    if not cache_dir.is_dir() or not any(cache_dir.iterdir()):
        raise DataNotStagedError(
            f"No staged realized LMPs for {name!r}: expected a price_frame under "
            f"{cache_dir} (e.g. realized_lmp.csv). A human must stage real ISO data "
            f"there (see data/README.md); the benchmark loop never downloads."
        )
    csv_path = cache_dir / "realized_lmp.csv"
    if not csv_path.is_file():
        raise DataNotStagedError(
            f"{cache_dir} exists but has no realized_lmp.csv to read as a price_frame."
        )
    frame = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    frame.columns = [int(c) for c in frame.columns]
    return frame


def run_realized(name: str, zap_lmp: np.ndarray, time_index: pd.DatetimeIndex) -> LMPComparison:
    """Compare a modeled ``zap_lmp`` against a staged realized frame (human path)."""
    frame = load_realized_frame(name)
    realized = align_frame_to_array(frame, zap_lmp.shape[0], time_index)
    return compare(zap_lmp, realized, source=f"cache:{name}")


def run(report_path: Optional[Path] = None) -> BenchResult:
    """Run the synthetic comparator and emit (optionally write) a ``BenchResult``."""
    result = run_synthetic().to_bench_result()
    if report_path is not None:
        result.write_markdown(report_path)
    return result


if __name__ == "__main__":
    comp = run_synthetic()
    print(f"source        : {comp.source}")
    print(f"mean |error|  : {comp.mean_abs_error:.4f} $/MWh")
    print(f"median |error|: {comp.median_abs_error:.4f} $/MWh")
    print(f"p90 |error|   : {comp.p90_abs_error:.4f} $/MWh")
    print(f"max |error|   : {comp.max_abs_error:.4f} $/MWh")
    print(f"mean CI (90%) : [{comp.ci.lo:.4f}, {comp.ci.hi:.4f}]")

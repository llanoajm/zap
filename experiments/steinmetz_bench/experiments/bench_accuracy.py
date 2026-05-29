"""Accuracy benchmark (roadmap item 2.3, Steinmetz §8.4.3).

zap's accuracy claim has two faces: its DC-OPF prices reproduce an independent
DC solver (PyPSA) to numerical noise, and they track the *realized* prices a grid
actually clears. This benchmark assembles both as **error distributions, not point
estimates** — for two quantities each:

- **LMP error** — the per-node/hour gap between zap's nodal price and the reference.
- **Congestion-component error** — the gap in the *congestion* part of the LMP. An
  LMP decomposes (losslessly, in DC-OPF) into a system-wide energy price plus a
  nodal congestion component; here the congestion component at a node is its LMP
  minus the reference-node LMP in the same hour (``congestion_components``). Pulling
  the energy level out isolates the part of the price that congestion alone creates,
  which is the quantity siting/transmission decisions actually turn on — so its
  accuracy is reported separately from the raw LMP.

Two references are assembled (each its own distribution; the networks differ and
need not match):

- **vs PyPSA** (``run_reference`` from item 1.1) — the bundled 3-bus radial net
  solved in both zap (CLARABEL) and PyPSA (HiGHS). This is the DC-vs-DC *fidelity
  floor*: it should be ~solver noise, and it is the result's ``fidelity_band``.
- **vs realized** (``run_synthetic`` from item 1.3) — zap's prices on a synthetic
  congested net against a "realized" world (a second zap solve under seeded
  load/cost perturbations). This is the model-vs-reality accuracy and supplies the
  headline number.

Nothing here is a hand-written constant: every distribution is differenced from two
real solves and re-derivable from the arrays the report carries. The ``--real`` path
swaps the synthetic realized world for a staged ISO ``price_frame`` and blocks via
:class:`DataNotStagedError` when ``data/`` is empty, so the human path is wired but
never downloads in the loop.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from attrs import define

from experiments.steinmetz_bench.datasets.registry import make_synthetic_network
from experiments.steinmetz_bench.experiments.realized_lmp import (
    _solve_lmp,
    align_frame_to_array,
    compare,
    load_realized_frame,
    run_synthetic as run_realized_synthetic,
)
from experiments.steinmetz_bench.experiments.ref_pypsa import run_reference
from experiments.steinmetz_bench.reports.result import BenchResult
from experiments.steinmetz_bench.scoring.metrics import (
    CIResult,
    FidelityBand,
    bootstrap_ci,
    fidelity_band,
)

EXPERIMENT_ID = "2.3-accuracy"
DATASET = "synthetic-multi-reference"

# Reference node for the congestion-component decomposition: the congestion part of
# every node's LMP is measured relative to this node's price in the same hour.
REF_NODE = 0


def congestion_components(lmp, ref_node: int = REF_NODE) -> np.ndarray:
    """Congestion component of an LMP array: ``lmp - lmp[ref_node]`` per hour.

    ``lmp`` is laid out ``(node, hour)``. Subtracting the reference node's price
    removes the system-wide energy level, leaving the nodal congestion component
    (the reference node's own component is identically zero).
    """
    lmp = np.asarray(lmp, dtype=float)
    return lmp - lmp[ref_node : ref_node + 1, :]


@define(kw_only=True)
class ErrorDistribution:
    """One assembled error distribution (LMP or congestion, vs one reference).

    ``band`` carries the mean/p90/max absolute gap and ``ci`` the bootstrap interval
    on the mean absolute error. Every field is derived from ``dc`` minus ``ref``;
    the raw arrays are kept so a test can re-derive the whole distribution.
    """

    name: str
    reference: str  # "pypsa-dc" | "realized-lmp"
    metric: str  # "lmp" | "congestion"
    units: str
    dc: np.ndarray  # (node, hour)
    ref: np.ndarray  # (node, hour)
    band: FidelityBand
    ci: CIResult
    median_abs_error: float

    @classmethod
    def from_arrays(
        cls,
        name: str,
        dc,
        ref,
        reference: str,
        metric: str,
        units: str = "$/MWh",
        seed: int = 0,
    ) -> "ErrorDistribution":
        dc = np.asarray(dc, dtype=float)
        ref = np.asarray(ref, dtype=float)
        band = fidelity_band(dc, ref, reference=reference, metric=metric, units=units)
        abs_err = np.abs(dc - ref).ravel()
        ci = bootstrap_ci(abs_err, statistic=np.mean, confidence=0.90, seed=seed)
        return cls(
            name=name,
            reference=reference,
            metric=metric,
            units=units,
            dc=dc,
            ref=ref,
            band=band,
            ci=ci,
            median_abs_error=float(np.median(abs_err)),
        )

    @property
    def mean_abs_error(self) -> float:
        return self.band.mean_abs_gap

    @property
    def p90_abs_error(self) -> float:
        return self.band.p90_abs_gap

    @property
    def max_abs_error(self) -> float:
        return self.band.max_abs_gap

    def to_dict(self) -> dict:
        """Distribution summary for the result's ``sensitivities`` block."""
        return {
            "reference": self.reference,
            "metric": self.metric,
            "units": self.units,
            "mean_abs_error": self.mean_abs_error,
            "median_abs_error": self.median_abs_error,
            "p90_abs_error": self.p90_abs_error,
            "max_abs_error": self.max_abs_error,
            "ci_lo": self.ci.lo,
            "ci_mid": self.ci.mid,
            "ci_hi": self.ci.hi,
            "n": self.band.n,
        }


# The headline component: model-vs-reality LMP accuracy is the number §8.4.3 reports.
HEADLINE = "lmp_vs_realized"


@define(kw_only=True)
class AccuracyReport:
    """The four assembled error distributions plus their provenance.

    ``components`` is keyed ``{lmp,congestion}_vs_{pypsa,realized}``. The headline is
    the mean absolute LMP error vs realized; the DC-vs-PyPSA LMP band is the result's
    ``fidelity_band`` (the numerical fidelity floor).
    """

    components: dict[str, ErrorDistribution]
    source: str  # "synthetic" | "cache:<name>"

    @property
    def headline_component(self) -> ErrorDistribution:
        return self.components[HEADLINE]

    def to_bench_result(self) -> BenchResult:
        head = self.headline_component
        pypsa_lmp = self.components["lmp_vs_pypsa"]
        sensitivities = {name: c.to_dict() for name, c in self.components.items()}
        sensitivities["headline_component"] = HEADLINE
        return BenchResult(
            experiment_id=EXPERIMENT_ID,
            dataset=DATASET,
            headline_number=head.mean_abs_error,
            units="$/MWh",
            ci=head.ci,
            fidelity_band=pypsa_lmp.band,
            assumptions={
                "source": self.source,
                "zap_solver": "CLARABEL",
                "headline": "mean |LMP error| vs realized (model-vs-reality accuracy)",
                "fidelity_band": "DC-vs-PyPSA LMP gap (numerical fidelity floor)",
                "congestion_decomposition": (
                    f"congestion component = LMP - LMP[node {REF_NODE}] per hour; "
                    "isolates the congestion part of the price from the energy level"
                ),
                "references": {
                    "pypsa": "bundled 3-bus radial net, zap CLARABEL vs PyPSA HiGHS (item 1.1)",
                    "realized": (
                        "synthetic congested net, zap vs a seeded-perturbation second "
                        "zap solve standing in for ISO-vs-model divergence (item 1.3); "
                        "--real swaps in a staged ISO price_frame"
                    ),
                },
                "distribution_not_point": (
                    "every reference reports mean/median/p90/max |error| + a bootstrap "
                    "CI, not a single point"
                ),
            },
            sensitivities=sensitivities,
        )


def _assemble(ref_comp, lmp_comp, source: str) -> AccuracyReport:
    """Build the four error distributions from a PyPSA and a realized comparison."""
    components = {
        "lmp_vs_pypsa": ErrorDistribution.from_arrays(
            "lmp_vs_pypsa", ref_comp.zap_lmp, ref_comp.pypsa_lmp, "pypsa-dc", "lmp"
        ),
        "congestion_vs_pypsa": ErrorDistribution.from_arrays(
            "congestion_vs_pypsa",
            congestion_components(ref_comp.zap_lmp),
            congestion_components(ref_comp.pypsa_lmp),
            "pypsa-dc",
            "congestion",
        ),
        "lmp_vs_realized": ErrorDistribution.from_arrays(
            "lmp_vs_realized", lmp_comp.zap_lmp, lmp_comp.realized_lmp, "realized-lmp", "lmp"
        ),
        "congestion_vs_realized": ErrorDistribution.from_arrays(
            "congestion_vs_realized",
            congestion_components(lmp_comp.zap_lmp),
            congestion_components(lmp_comp.realized_lmp),
            "realized-lmp",
            "congestion",
        ),
    }
    return AccuracyReport(components=components, source=source)


def run_synthetic(
    n_nodes: int = 5, hours: int = 24, seed: int = 0, realized_seed: int = 1
) -> AccuracyReport:
    """Assemble the accuracy distributions on bundled + synthetic fixtures (loop path)."""
    ref_comp = run_reference()
    lmp_comp = run_realized_synthetic(
        n_nodes=n_nodes, hours=hours, seed=seed, realized_seed=realized_seed
    )
    return _assemble(ref_comp, lmp_comp, source="synthetic")


def run_real(name: str, n_nodes: int = 5, hours: int = 24, seed: int = 0) -> AccuracyReport:
    """Assemble accuracy vs a staged ISO ``price_frame`` (human ``--real`` path).

    Loads the staged realized frame first, so a missing cache blocks via
    :class:`DataNotStagedError` *before* any solve. A human stages both the real
    network and its realized prices; the PyPSA fidelity floor stays the bundled net.
    """
    frame = load_realized_frame(name)  # DataNotStagedError if data/<name>/ is empty
    ref_comp = run_reference()
    net, devices, _ = make_synthetic_network(
        n_nodes=n_nodes, hours=hours, congested=True, seed=seed
    )
    time_index = pd.date_range("2025-01-01", periods=hours, freq="h")
    zap_lmp = _solve_lmp(net, devices)
    realized = align_frame_to_array(frame, zap_lmp.shape[0], time_index)
    lmp_comp = compare(zap_lmp, realized, source=f"cache:{name}")
    return _assemble(ref_comp, lmp_comp, source=f"cache:{name}")


def run(report_path=None) -> BenchResult:
    """Run the synthetic accuracy benchmark and emit (optionally write) a result."""
    result = run_synthetic().to_bench_result()
    if report_path is not None:
        result.write_markdown(report_path)
    return result


def _print_report(report: AccuracyReport) -> None:
    print(f"source: {report.source}")
    print(f"{'component':<24}{'mean':>10}{'median':>10}{'p90':>10}{'max':>10}")
    for name, comp in report.components.items():
        print(
            f"{name:<24}{comp.mean_abs_error:>10.4f}{comp.median_abs_error:>10.4f}"
            f"{comp.p90_abs_error:>10.4f}{comp.max_abs_error:>10.4f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Steinmetz accuracy benchmark (§8.4.3)")
    parser.add_argument(
        "--synthetic", action="store_true", default=True,
        help="run on bundled + synthetic fixtures (default, loop-runnable)",
    )
    parser.add_argument(
        "--real", metavar="NAME", default=None,
        help="assemble against a staged ISO price_frame in data/NAME/ (human path)",
    )
    args = parser.parse_args()
    rep = run_real(args.real) if args.real else run_synthetic()
    _print_report(rep)

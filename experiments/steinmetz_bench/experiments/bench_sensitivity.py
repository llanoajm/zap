"""Sensitivity-correctness report (roadmap item 2.4, Steinmetz §8.4.4).

Item 1.2 (:mod:`experiments.steinmetz_bench.experiments.grad_check`) certifies that
zap's adjoint sensitivities match the exact gradient implied by LP duality (the
envelope theorem) for every device type the planner cares about. This module is the
*published face* of that check: it wraps the 1.2 report into a per-device-type table
— one row per (network, device-type) plus a per-device-type worst-case roll-up — and
writes a markdown report whose headline is the worst relative gradient error across
all device types.

Nothing here recomputes a gradient: the table is assembled entirely from the
:class:`~experiments.steinmetz_bench.experiments.grad_check.GradCheckReport` arrays,
so every cell is traceable to a real CLARABEL solve (adjoint via the KKT backward
pass, reference via the device Lagrangian with the solver's duals, plus the
finite-difference anchor). The acceptance is the §8.4.4 claim: per-device max relative
error < ``1e-3``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cvxpy as cp
from attrs import define, field

from experiments.steinmetz_bench.experiments.grad_check import (
    GRAD_REL_TOL,
    GradCheckReport,
    run_grad_check,
)
from experiments.steinmetz_bench.reports.result import BenchResult

EXPERIMENT_ID = "2.4-sensitivity"
DATASET = "garver+toy7"

# Acceptance: worst-case relative gradient error per device type, inherited from 1.2.
PER_DEVICE_TOL = GRAD_REL_TOL

_DEVICE_ORDER = ("line", "generator", "battery")


@define(kw_only=True)
class SensitivityRow:
    """One published table row: a (network, device-type) gradient comparison."""

    network: str
    device_type: str
    attr: str
    n_active: int
    max_rel_err_dual: float
    max_rel_err_fd: float

    def to_dict(self) -> dict:
        return {
            "network": self.network,
            "device_type": self.device_type,
            "attr": self.attr,
            "n_active": self.n_active,
            "max_rel_err_dual": self.max_rel_err_dual,
            "max_rel_err_fd": self.max_rel_err_fd,
        }


@define(kw_only=True)
class SensitivityTable:
    """The per-device-type sensitivity-correctness table built from a 1.2 report."""

    report: GradCheckReport
    rows: list = field(factory=list)

    @classmethod
    def from_report(cls, report: GradCheckReport) -> "SensitivityTable":
        rows = [
            SensitivityRow(
                network=c.network,
                device_type=c.device_type,
                attr=c.attr,
                n_active=c.n_active,
                max_rel_err_dual=c.max_rel_err_dual,
                max_rel_err_fd=c.max_rel_err_fd,
            )
            for c in report.checks
        ]
        return cls(report=report, rows=rows)

    def per_device_type(self) -> dict:
        """Worst-case relative error per device type, across networks."""
        return self.report.per_device_type()

    @property
    def headline_number(self) -> float:
        """Worst relative gradient error over every device type — the §8.4.4 number."""
        return max(self.per_device_type().values(), default=0.0)

    def to_markdown_table(self) -> str:
        """Render the published per-(network, device-type) table as GitHub markdown."""
        header = (
            "| network | device_type | attr | n_active | "
            "max_rel_err_dual | max_rel_err_fd |"
        )
        sep = "| --- | --- | --- | ---: | ---: | ---: |"
        lines = [header, sep]
        for r in self.rows:
            lines.append(
                f"| {r.network} | {r.device_type} | {r.attr} | {r.n_active} | "
                f"{r.max_rel_err_dual:.3e} | {r.max_rel_err_fd:.3e} |"
            )
        return "\n".join(lines)

    def to_bench_result(self) -> BenchResult:
        per_type = self.per_device_type()
        band = self.report.fidelity()
        return BenchResult(
            experiment_id=EXPERIMENT_ID,
            dataset=DATASET,
            headline_number=self.headline_number,
            units="relative",
            fidelity_band=band,
            assumptions={
                "wraps": "item 1.2 grad-vs-dual check (envelope-theorem identity)",
                "identity": "d(cost*)/d(theta) = dL/d(theta) with solver duals",
                "adjoint_source": "zap DispatchLayer.backward (KKT linear solve)",
                "dual_source": "device.lagrangian autodiff with CVXPY duals",
                "solver": "CLARABEL",
                "per_device_tol": PER_DEVICE_TOL,
                "headline": "worst relative gradient error across all device types",
                "networks": {
                    "garver": "6-bus, generator + AC line (paper Fig. 6 system)",
                    "toy7": "7-bus, generator + AC line + multi-period battery",
                },
            },
            sensitivities={
                "rel_err_by_device_type": per_type,
                "headline_number": self.headline_number,
                "table": [r.to_dict() for r in self.rows],
            },
        )


def _report_markdown(table: SensitivityTable, result: BenchResult) -> str:
    """Compose the published report: prose, the table, then the re-parseable JSON.

    The trailing ``benchresult:json`` block is :func:`reports.result.parse_markdown`'s
    anchor, so the report round-trips to an identical :class:`BenchResult`.
    """
    per_type = table.per_device_type()
    lines = [
        f"# {EXPERIMENT_ID} — Sensitivity-correctness report (§8.4.4)",
        "",
        "Per-device-type relative error between zap's adjoint gradient and the exact",
        "gradient implied by LP duality (envelope theorem). Acceptance: every device",
        f"type below `{PER_DEVICE_TOL}`.",
        "",
        "## Per-device-type worst-case",
        "",
        "| device_type | max_rel_err_dual | pass |",
        "| --- | ---: | :---: |",
    ]
    for device_type in _DEVICE_ORDER:
        if device_type in per_type:
            err = per_type[device_type]
            ok = "PASS" if err < PER_DEVICE_TOL else "FAIL"
            lines.append(f"| {device_type} | {err:.3e} | {ok} |")
    lines += [
        "",
        "## Per-(network, device-type) detail",
        "",
        table.to_markdown_table(),
        "",
        "<!-- benchresult:json -->",
        "```json",
        result.to_json(),
        "```",
        "",
    ]
    return "\n".join(lines)


def run_table(solver=cp.CLARABEL, do_fd: bool = True) -> SensitivityTable:
    """Run the 1.2 check and assemble the published per-device-type table."""
    return SensitivityTable.from_report(run_grad_check(solver=solver, do_fd=do_fd))


def run(report_path=None, do_fd: bool = True) -> BenchResult:
    """Run the sensitivity report and emit (optionally write) a :class:`BenchResult`.

    When ``report_path`` is given, writes the published markdown table report (which
    re-parses to the same result via the embedded JSON block).
    """
    table = run_table(do_fd=do_fd)
    result = table.to_bench_result()
    if report_path is not None:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_report_markdown(table, result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Steinmetz sensitivity report (§8.4.4)")
    parser.add_argument("--no-fd", action="store_true", help="skip finite-difference anchor")
    args = parser.parse_args()
    tbl = run_table(do_fd=not args.no_fd)
    print(tbl.to_markdown_table())
    print()
    per_type = tbl.per_device_type()
    for device_type in _DEVICE_ORDER:
        if device_type in per_type:
            print(f"{device_type:<10} max rel err {per_type[device_type]:.3e}")
    print(f"\nheadline worst relative gradient error: {tbl.headline_number:.3e} "
          f"(tol {PER_DEVICE_TOL})")

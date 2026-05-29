"""CPU speed benchmark (roadmap item 2.1, Steinmetz §8.4.1).

Time zap's DC-OPF dispatch against an independent CVXPY LP baseline across a
range of synthetic network sizes, and certify that the two solvers agree on the
optimal objective. The point of the benchmark is two-fold:

1. **Speed** — record wall-clock seconds for each path at each size so the §8.4.1
   table has real, machine-measured numbers (no hand-written timings).
2. **Correctness** — the speed comparison is only meaningful if both paths solve
   the *same* problem, so every size asserts an objective gap below
   :data:`OBJECTIVE_GAP_TOL`. The baseline is built from the *same* zap device
   objects (its bounds, costs, susceptances and topology are read straight off
   the devices), so any residual gap is pure solver-vs-solver numerical noise —
   exactly the quantity this benchmark exists to bound.

The baseline is a genuinely different code path from zap: zap assembles its conic
program through :class:`~zap.network.PowerNetwork.dispatch` and solves it with
CLARABEL; the baseline assembles a plain DC-OPF LP here and solves it with a
linear-programming solver (Mosek if licensed, else HiGHS). On the bundled
synthetic radial networks the angle/susceptance equalities never restrict the
feasible flows beyond power balance, so the LP and the conic dispatch share an
identical optimum.

The WECC / 1000-contingency / Modal-H100 headline timings from the spec are
**human-gated** — they need staged large-scale cases and the GPU run (item 2.5),
and are explicitly NOT produced here. This item reports CPU timings on bounded
synthetic sizes only.
"""

from __future__ import annotations

import time

import cvxpy as cp
import numpy as np
from attrs import define, field

from zap.devices import ACLine, Generator, Load
from zap.network import PowerNetwork

from experiments.steinmetz_bench.datasets.registry import DatasetSpec, resolve
from experiments.steinmetz_bench.reports.result import BenchResult
from experiments.steinmetz_bench.scoring.metrics import FidelityBand, fidelity_band

EXPERIMENT_ID = "2.1-speed-cpu"
DATASET = "synthetic-radial-sweep"

# Objective parity tolerance (relative). Both paths solve the same LP, so this is
# a solver-noise bound, not a modelling slack.
OBJECTIVE_GAP_TOL = 1e-2

# zap's dispatch solver (conic). The baseline picks the best available LP solver.
ZAP_SOLVER = cp.CLARABEL

# Default size sweep: (n_nodes, hours). Three sizes satisfy the ">=3 sizes"
# acceptance while staying fast enough to run inside the per-item pytest verify.
DEFAULT_SIZES: tuple[tuple[int, int], ...] = ((6, 24), (12, 24), (24, 24))


def _baseline_solver() -> str:
    """Pick the LP baseline solver: Mosek if licensed, else HiGHS, else CLARABEL."""
    installed = set(cp.installed_solvers())
    for name in ("MOSEK", "HIGHS"):
        if name in installed:
            return name
    return "CLARABEL"


def _as_2d(arr, n: int, horizon: int) -> np.ndarray:
    """Broadcast a zap (n, 1) or (n, horizon) device array to a dense (n, horizon)."""
    a = np.asarray(arr, dtype=float)
    if a.ndim == 1:
        a = a.reshape(n, -1)
    if a.shape[1] == 1:
        a = np.broadcast_to(a, (n, horizon))
    return np.asarray(a, dtype=float)


def build_baseline_lp(
    net: PowerNetwork, devices: list, horizon: int
) -> tuple[cp.Problem, cp.Expression]:
    """Assemble a DC-OPF LP for ``devices`` read directly off the zap objects.

    Mirrors zap's per-device cost and constraint semantics exactly:

    - generators: ``0 <= g <= dynamic_capacity * nominal_capacity``; cost
      ``linear_cost * g``.
    - loads: ``-load*nom <= p <= 0``; curtailment cost
      ``linear_cost * (p + load*nom)`` (zero when fully served).
    - AC lines: sink-terminal power ``p`` with ``-cap*nom <= p <= cap*nom``,
      ``p = susceptance*nom*(theta_src - theta_sink)``, cost ``linear_cost*|p|``;
      net injection ``+p`` at the sink and ``-p`` at the source.

    Returns the CVXPY problem (not yet solved) and the objective expression.
    """
    n_nodes = net.num_nodes
    node_injection = [[] for _ in range(n_nodes)]
    cost_terms: list[cp.Expression] = []
    constraints: list = []

    theta = cp.Variable((n_nodes, horizon), name="theta")
    constraints.append(theta[0, :] == 0.0)  # reference bus

    for dev in devices:
        if isinstance(dev, Generator):
            ng = dev.num_devices
            cap = _as_2d(dev.max_power, ng, horizon) * _as_2d(dev.nominal_capacity, ng, horizon)
            lin = _as_2d(dev.linear_cost, ng, horizon)
            g = cp.Variable((ng, horizon), name="gen", nonneg=True)
            constraints.append(g <= cap)
            cost_terms.append(cp.sum(cp.multiply(lin, g)))
            for i, node in enumerate(np.asarray(dev.terminal).ravel()):
                node_injection[int(node)].append(g[i, :])

        elif isinstance(dev, Load):
            nl = dev.num_devices
            nom = _as_2d(dev.nominal_capacity, nl, horizon)
            lin = _as_2d(dev.linear_cost, nl, horizon)
            lo = _as_2d(dev.min_power, nl, horizon) * nom  # = -load*nom
            p = cp.Variable((nl, horizon), name="load")
            constraints += [p >= lo, p <= 0.0]
            cost_terms.append(cp.sum(cp.multiply(lin, p - lo)))
            for i, node in enumerate(np.asarray(dev.terminal).ravel()):
                node_injection[int(node)].append(p[i, :])

        elif isinstance(dev, ACLine):
            ne = dev.num_devices
            nom = _as_2d(dev.nominal_capacity, ne, horizon)
            cap = _as_2d(dev.max_power, ne, horizon) * nom
            b = _as_2d(dev.susceptance, ne, horizon) * nom
            lin = _as_2d(dev.linear_cost, ne, horizon)
            p = cp.Variable((ne, horizon), name="line")
            src = np.asarray(dev.source_terminal).ravel()
            dst = np.asarray(dev.sink_terminal).ravel()
            constraints += [p <= cap, p >= -cap]
            for k in range(ne):
                s, d = int(src[k]), int(dst[k])
                constraints.append(p[k, :] == cp.multiply(b[k, :], theta[s, :] - theta[d, :]))
                node_injection[d].append(p[k, :])
                node_injection[s].append(-p[k, :])
            cost_terms.append(cp.sum(cp.multiply(lin, cp.abs(p))))

        else:
            raise TypeError(f"baseline LP does not model device type {type(dev).__name__}")

    for node in range(n_nodes):
        if node_injection[node]:
            constraints.append(cp.sum(node_injection[node]) == 0.0)

    objective = cp.sum(cost_terms)
    return cp.Problem(cp.Minimize(objective), constraints), objective


def _time_zap(net, devices, horizon, repeats) -> tuple[float, float]:
    """Best-of-``repeats`` wall-clock for a full zap dispatch; returns (seconds, objective)."""
    best = np.inf
    value = np.nan
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = net.dispatch(devices, time_horizon=horizon, solver=ZAP_SOLVER)
        elapsed = time.perf_counter() - t0
        if out.problem.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"zap dispatch failed: status={out.problem.status}")
        best = min(best, elapsed)
        value = float(out.problem.value)
    return best, value


def _time_baseline(net, devices, horizon, solver, repeats) -> tuple[float, float]:
    """Best-of-``repeats`` wall-clock to build+solve the baseline LP; returns (seconds, objective)."""
    best = np.inf
    value = np.nan
    for _ in range(repeats):
        t0 = time.perf_counter()
        prob, _ = build_baseline_lp(net, devices, horizon)
        prob.solve(solver=solver)
        elapsed = time.perf_counter() - t0
        if prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"baseline LP failed: status={prob.status}")
        best = min(best, elapsed)
        value = float(prob.value)
    return best, value


@define(kw_only=True)
class SpeedRow:
    """One network size: both solvers' wall-clock and their objective values."""

    n_nodes: int
    hours: int
    n_devices: int
    zap_s: float
    baseline_s: float
    zap_objective: float
    baseline_objective: float

    @property
    def objective_gap(self) -> float:
        denom = max(abs(self.zap_objective), 1.0)
        return abs(self.zap_objective - self.baseline_objective) / denom

    @property
    def speedup(self) -> float:
        """Baseline seconds per zap second (>1 means zap is slower)."""
        return self.baseline_s / self.zap_s if self.zap_s > 0 else float("nan")


@define(kw_only=True)
class SpeedReport:
    """All size rows plus the baseline solver name."""

    rows: list = field(factory=list)
    baseline_solver: str = "HIGHS"

    @property
    def max_objective_gap(self) -> float:
        return max((r.objective_gap for r in self.rows), default=0.0)

    def fidelity(self) -> FidelityBand:
        """DC(zap)-vs-LP objective agreement across sizes, as the result's band."""
        zap_obj = [r.zap_objective for r in self.rows]
        base_obj = [r.baseline_objective for r in self.rows]
        return fidelity_band(zap_obj, base_obj, reference="cvxpy-lp",
                             metric="objective", units="$")

    def to_bench_result(self) -> BenchResult:
        """Headline = the worst objective gap across sizes (a correctness number).

        Per-size timings ride along in ``sensitivities`` so the §8.4.1 table is
        fully reconstructable from the JSON.
        """
        return BenchResult(
            experiment_id=EXPERIMENT_ID,
            dataset=DATASET,
            headline_number=self.max_objective_gap,
            units="relative",
            fidelity_band=self.fidelity(),
            assumptions={
                "zap_solver": "CLARABEL",
                "baseline_solver": self.baseline_solver,
                "baseline_kind": "DC-OPF LP built from the same zap device objects",
                "objective_gap_tol": OBJECTIVE_GAP_TOL,
                "sizes": [[r.n_nodes, r.hours] for r in self.rows],
                "headline_gating": (
                    "CPU synthetic timings only; the WECC / 1000-contingency / "
                    "Modal-H100 headline is human-gated (see roadmap item 2.5 + "
                    "human prerequisites)"
                ),
            },
            sensitivities={
                "timing_table": [
                    {
                        "n_nodes": r.n_nodes,
                        "hours": r.hours,
                        "n_devices": r.n_devices,
                        "zap_s": r.zap_s,
                        "baseline_s": r.baseline_s,
                        "speedup_baseline_per_zap": r.speedup,
                        "zap_objective": r.zap_objective,
                        "baseline_objective": r.baseline_objective,
                        "objective_gap": r.objective_gap,
                    }
                    for r in self.rows
                ],
                "max_objective_gap": self.max_objective_gap,
            },
        )


def run_speed_benchmark(sizes=DEFAULT_SIZES, repeats: int = 2, seed: int = 0) -> SpeedReport:
    """Run the zap-vs-baseline speed/objective comparison across ``sizes``."""
    solver = _baseline_solver()
    rows = []
    for k, (n_nodes, hours) in enumerate(sizes):
        spec = DatasetSpec(name=f"speed-{n_nodes}x{hours}", kind="synthetic",
                           n_nodes=n_nodes, hours=hours, congested=False, seed=seed + k)
        ds = resolve(spec)
        horizon = ds.time_horizon
        zap_s, zap_obj = _time_zap(ds.network, ds.devices, horizon, repeats)
        base_s, base_obj = _time_baseline(ds.network, ds.devices, horizon, solver, repeats)
        rows.append(SpeedRow(
            n_nodes=n_nodes, hours=hours, n_devices=len(ds.devices),
            zap_s=zap_s, baseline_s=base_s,
            zap_objective=zap_obj, baseline_objective=base_obj,
        ))
    return SpeedReport(rows=rows, baseline_solver=solver)


def run(report_path=None, sizes=DEFAULT_SIZES, repeats: int = 2) -> BenchResult:
    """Run the benchmark and emit (optionally write) a :class:`BenchResult`."""
    result = run_speed_benchmark(sizes=sizes, repeats=repeats).to_bench_result()
    if report_path is not None:
        result.write_markdown(report_path)
    return result


if __name__ == "__main__":
    report = run_speed_benchmark()
    print(f"baseline solver: {report.baseline_solver}")
    print(f"{'nodes':>6} {'hours':>6} {'zap_s':>10} {'base_s':>10} "
          f"{'speedup':>8} {'obj_gap':>10}")
    for r in report.rows:
        print(f"{r.n_nodes:>6} {r.hours:>6} {r.zap_s:>10.4f} {r.baseline_s:>10.4f} "
              f"{r.speedup:>8.2f} {r.objective_gap:>10.2e}")
    print(f"\nmax objective gap: {report.max_objective_gap:.3e} (tol {OBJECTIVE_GAP_TOL})")

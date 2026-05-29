"""GPU speed benchmark via the deployed ``zap-opf-solver`` Modal app (item 2.5, §8.4.1).

Dispatches zap's ADMM solver on an H100 (the EXISTING deployed app at
``grid-app/infra/modal/solver_app.py``, function ``solve_direct``) for a modest
and a slightly larger synthetic network, records the GPU wall-clock, and
certifies CPU-vs-GPU parity by **re-evaluating zap's exact cost function at the
GPU's returned dispatch** and comparing it to an independent CPU LP solve of the
same network. The objective gap is therefore computed, never asserted — exactly
the anti-demoware bar this suite holds itself to.

Cost guard
----------
Modal is dispatched **exactly once**, by an agent/human running
:func:`dispatch` (``python -m ...bench_gpu_modal --dispatch``). Its result is
cached to ``data/gpu_runs/<timestamp>.json``. Every test and the per-item verify
read ONLY that cache; nothing under the project venv ever imports ``modal`` or
calls the GPU. The actual Modal RPC lives in the sibling :mod:`_modal_call`
module, which is run as a subprocess under the system interpreter (the venv has
no ``modal`` module on purpose). If the cache is absent the test skips, so the
loop's verify stays green whether or not the one-time GPU run has happened.

The full WECC / 1000-contingency headline stays a human-gated run (see the
roadmap's human prerequisites); this item dispatches bounded synthetic sizes.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cvxpy as cp
import numpy as np
from attrs import asdict, define, field

from experiments.steinmetz_bench.datasets.registry import DATA_ROOT
from experiments.steinmetz_bench.reports.result import BenchResult
from experiments.steinmetz_bench.scoring.metrics import FidelityBand, fidelity_band

EXPERIMENT_ID = "2.5-gpu-modal"
DATASET = "synthetic-pypsa-gpu"

GPU_RUNS_DIR = DATA_ROOT / "gpu_runs"

# CPU-vs-GPU objective parity tolerance (relative). ADMM is an iterative solver;
# with enough iterations on these well-conditioned uncongested networks it lands
# within this bound of the exact CPU LP optimum.
OBJECTIVE_GAP_TOL = 1e-2

# zap's exact CPU dispatch solver (conic LP). Used as the parity reference.
CPU_SOLVER = cp.CLARABEL

# ADMM args sent to the GPU. float64 + many iterations + tight tolerances buy
# objective parity (see grid-app/infra/modal/PARITY_REPORT.md for why the
# endpoint defaults of 1000 iters / float32 are too loose for price parity).
GPU_ADMM_ARGS: dict = {
    "num_iterations": 5000,
    "rho_power": 1.0,
    "rho_angle": 1.0,
    "atol": 1e-8,
    "rtol": 1e-8,
    "dtype": "float64",
}

# Bounded network sizes (name, n_buses, hours). Kept small + uncongested so the
# single H100 dispatch is cheap and ADMM converges to the LP optimum.
NETWORKS: tuple[tuple[str, int, int], ...] = (
    ("modest", 6, 8),
    ("larger", 14, 8),
)

# System interpreter that has the `modal` client (the venv intentionally does not).
SYSTEM_PYTHON = "/usr/bin/python3"
_MODAL_CALL = Path(__file__).resolve().parent / "_modal_call.py"

_PANDAS_COW_PATCHED = False


def _patch_pandas_cow() -> None:
    """Make ``DataFrame/Series.values`` writable again under pandas CoW.

    ``zap.importers.pypsa`` mutates ``.values`` in place (``/=``), which pandas
    Copy-on-Write turns into read-only arrays. Mirrors ``zap/tests/conftest.py``
    and ``solver_app._patch_pandas_cow_in_container`` so the CPU parity solve
    loads the same network the GPU container does. Idempotent + inert on
    pre-CoW pandas.
    """
    global _PANDAS_COW_PATCHED
    if _PANDAS_COW_PATCHED:
        return
    import pandas as pd

    orig_df = pd.DataFrame.values.fget
    orig_series = pd.Series.values.fget

    def _writable(arr):
        if isinstance(arr, np.ndarray) and not arr.flags.writeable:
            return arr.copy()
        return arr

    pd.DataFrame.values = property(lambda self: _writable(orig_df(self)))
    pd.Series.values = property(lambda self: _writable(orig_series(self)))
    _PANDAS_COW_PATCHED = True


# --------------------------------------------------------------------------- #
# Network construction (PyPSA, so the GPU app's pypsa importer and the CPU side
# both build identical zap devices from the same netCDF bytes).
# --------------------------------------------------------------------------- #
def build_pypsa_network(n_buses: int, hours: int, seed: int = 0):
    """A radial PyPSA network: one cheap slack gen, per-bus load, x>0 lines.

    Uncongested by construction (generous ``s_nom``) and free of zero-reactance
    lines, so the zap importer's ``1/x`` susceptance is finite on every line and
    ADMM converges cleanly — both prerequisites for tight CPU-vs-GPU parity.
    """
    import pandas as pd
    import pypsa

    _patch_pandas_cow()
    rng = np.random.default_rng(seed)
    pnet = pypsa.Network()
    pnet.set_snapshots(pd.date_range("2025-01-01", periods=hours, freq="h"))

    buses = [f"b{i}" for i in range(n_buses)]
    for b in buses:  # per-bus add: PyPSA 0.30 treats a list arg as one literal name.
        pnet.add("Bus", b)

    # A single zero-emission carrier so the zap importer's carrier lookup resolves.
    pnet.add("Carrier", "elec", co2_emissions=0.0)

    # Cheap base generator at bus 0, a pricier backstop at the last bus.
    pnet.add("Generator", "g_base", bus="b0", p_nom=500.0, marginal_cost=15.0, carrier="elec")
    pnet.add("Generator", "g_peak", bus=buses[-1], p_nom=500.0, marginal_cost=60.0, carrier="elec")

    # Per-bus load with a mild daily shape + seeded noise (no curtailment needed).
    t = np.arange(hours)
    shape = 1.0 + 0.2 * np.sin(2 * np.pi * (t - 6) / 24.0)
    for i, b in enumerate(buses):
        base = 20.0 + 5.0 * i
        profile = np.clip(base * shape + rng.normal(0.0, 1.0, size=hours), 1.0, None)
        pnet.add("Load", f"l_{b}", bus=b, p_set=profile)

    # Radial AC path b0-b1-...-b(n-1); wide s_nom keeps every line uncongested.
    for i in range(n_buses - 1):
        pnet.add(
            "Line", f"L{i}", bus0=buses[i], bus1=buses[i + 1],
            x=0.1, r=0.0, s_nom=10_000.0,
        )
    return pnet


def network_to_nc_bytes(pnet) -> bytes:
    """Serialise a PyPSA network to netCDF bytes (path-only PyPSA API)."""
    with tempfile.NamedTemporaryFile(suffix=".nc") as tf:
        pnet.export_to_netcdf(tf.name)
        return Path(tf.name).read_bytes()


def _load_zap(pnet):
    """Load a PyPSA network into zap devices exactly as the GPU app does.

    Mirrors ``solver_app._run_solve``: ``load_pypsa_network(pnet)`` with the
    default import args (the GPU side passes ``import_args={}``), so the device
    set, ordering, and susceptance normalisation are identical on both sides.
    """
    from zap.importers.pypsa import load_pypsa_network

    _patch_pandas_cow()
    net, devices = load_pypsa_network(pnet)
    horizon = max(d.time_horizon for d in devices)
    return net, devices, horizon


def cpu_solve(pnet) -> dict:
    """Exact CPU dispatch of ``pnet`` via zap + CLARABEL; the parity reference."""
    net, devices, horizon = _load_zap(pnet)
    t0 = time.perf_counter()
    out = net.dispatch(devices, time_horizon=horizon, solver=CPU_SOLVER)
    elapsed = time.perf_counter() - t0
    if out.problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"CPU dispatch failed: status={out.problem.status}")
    return {
        "objective": float(out.problem.value),
        "elapsed_s": elapsed,
        "num_buses": int(net.num_nodes),
        "horizon": int(horizon),
    }


def _nested_to_numpy(power):
    """Convert the GPU outcome's nested power lists to numpy, preserving None.

    ``outcome.power`` is a list over devices; each device is a list over its
    terminals (or ``None`` for a terminal zap left unmodelled). The JSON encoder
    in solver_app already turned tensors into nested Python lists.
    """
    out = []
    for dev_power in power:
        if dev_power is None:
            out.append(None)
            continue
        out.append([None if term is None else np.asarray(term, dtype=float) for term in dev_power])
    return out


def gpu_objective_from_outcome(pnet, outcome: dict) -> float:
    """Re-evaluate zap's exact cost at the GPU's returned dispatch.

    Loads the same network into CPU zap devices and calls
    ``net.operation_cost`` at the GPU power. For this suite's device set
    (Generator / Load / ACLine) ``operation_cost`` reads only the device powers
    and ignores angle / local variables, so the GPU's ``power`` tensor alone
    fully determines the cost — making this a genuine recomputation, not an
    echo of a number the GPU sent back.
    """
    net, devices, _ = _load_zap(pnet)
    power = _nested_to_numpy(outcome["power"])
    if len(power) != len(devices):
        raise ValueError(
            f"GPU returned {len(power)} device-power blocks but the CPU load "
            f"produced {len(devices)} devices; alignment broken"
        )
    angle = [None] * len(devices)
    local_vars = [None] * len(devices)
    cost = net.operation_cost(devices, power, angle, local_vars, la=np)
    return float(cost)


def objective_gap(cpu_objective: float, gpu_objective: float) -> float:
    """Relative objective gap, floored denominator so tiny costs stay sane."""
    return abs(cpu_objective - gpu_objective) / max(abs(cpu_objective), 1.0)


# --------------------------------------------------------------------------- #
# Cached-run record + (de)serialisation.
# --------------------------------------------------------------------------- #
@define(kw_only=True)
class GpuRunRecord:
    """One network's GPU dispatch + its CPU-vs-GPU parity, all code-computed."""

    network: str
    n_buses: int
    hours: int
    machine: str
    gpu_tier: Optional[str]
    gpu_elapsed_s: float
    cpu_elapsed_s: float
    gpu_objective: float
    cpu_objective: float
    objective_gap: float
    solver_args: dict = field(factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "GpuRunRecord":
        return cls(
            network=str(d["network"]),
            n_buses=int(d["n_buses"]),
            hours=int(d["hours"]),
            machine=str(d["machine"]),
            gpu_tier=d.get("gpu_tier"),
            gpu_elapsed_s=float(d["gpu_elapsed_s"]),
            cpu_elapsed_s=float(d["cpu_elapsed_s"]),
            gpu_objective=float(d["gpu_objective"]),
            cpu_objective=float(d["cpu_objective"]),
            objective_gap=float(d["objective_gap"]),
            solver_args=dict(d.get("solver_args") or {}),
        )


@define(kw_only=True)
class GpuRun:
    """A full one-shot dispatch: one record per network + dispatch metadata."""

    timestamp: str
    records: list = field(factory=list)

    @property
    def max_objective_gap(self) -> float:
        return max((r.objective_gap for r in self.records), default=0.0)

    @property
    def machines(self) -> set:
        return {r.machine for r in self.records}

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "records": [asdict(r) for r in self.records],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GpuRun":
        return cls(
            timestamp=str(d["timestamp"]),
            records=[GpuRunRecord.from_dict(r) for r in d["records"]],
        )

    def write_json(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        return path

    def fidelity(self) -> FidelityBand:
        cpu = [r.cpu_objective for r in self.records]
        gpu = [r.gpu_objective for r in self.records]
        return fidelity_band(cpu, gpu, reference="modal-gpu-admm",
                             metric="objective", units="$")

    def to_bench_result(self) -> BenchResult:
        """Headline = worst CPU-vs-GPU objective gap (a correctness number).

        Per-network GPU timings + objectives ride along in ``sensitivities`` so
        the §8.4.1 GPU table is fully reconstructable from the JSON.
        """
        return BenchResult(
            experiment_id=EXPERIMENT_ID,
            dataset=DATASET,
            headline_number=self.max_objective_gap,
            units="relative",
            fidelity_band=self.fidelity(),
            assumptions={
                "cpu_solver": "CLARABEL",
                "gpu_solver": "ADMMSolver (zap-opf-solver Modal app, H100)",
                "gpu_admm_args": GPU_ADMM_ARGS,
                "objective_gap_tol": OBJECTIVE_GAP_TOL,
                "parity_method": (
                    "re-evaluate zap operation_cost at the GPU's returned dispatch "
                    "and compare to the exact CPU LP optimum"
                ),
                "headline_gating": (
                    "bounded synthetic sizes only; the full WECC / 1000-contingency "
                    "H100 headline stays a human-gated run (see roadmap prerequisites)"
                ),
                "dispatch_timestamp": self.timestamp,
            },
            sensitivities={
                "gpu_table": [
                    {
                        "network": r.network,
                        "n_buses": r.n_buses,
                        "hours": r.hours,
                        "machine": r.machine,
                        "gpu_tier": r.gpu_tier,
                        "gpu_elapsed_s": r.gpu_elapsed_s,
                        "cpu_elapsed_s": r.cpu_elapsed_s,
                        "gpu_objective": r.gpu_objective,
                        "cpu_objective": r.cpu_objective,
                        "objective_gap": r.objective_gap,
                    }
                    for r in self.records
                ],
                "max_objective_gap": self.max_objective_gap,
            },
        )


def latest_cached_run() -> Optional[GpuRun]:
    """Most recent cached GPU run, or None if no dispatch has happened yet."""
    if not GPU_RUNS_DIR.is_dir():
        return None
    files = sorted(GPU_RUNS_DIR.glob("*.json"))
    if not files:
        return None
    return GpuRun.from_dict(json.loads(files[-1].read_text()))


def modal_available() -> bool:
    """True iff the ``modal`` CLI is on PATH and ``~/.modal.toml`` exists.

    Mirrors the acceptance condition; does NOT import modal or hit the network.
    """
    return shutil.which("modal") is not None and (Path.home() / ".modal.toml").is_file()


# --------------------------------------------------------------------------- #
# One-shot dispatch (run manually; NEVER from the per-item verify).
# --------------------------------------------------------------------------- #
def _call_modal(nc_bytes: bytes, args: dict) -> dict:
    """Dispatch one GPU solve via the system-python ``_modal_call`` subprocess."""
    with tempfile.TemporaryDirectory() as tmp:
        nc_path = Path(tmp) / "network.nc"
        args_path = Path(tmp) / "args.json"
        nc_path.write_bytes(nc_bytes)
        args_path.write_text(json.dumps(args))
        proc = subprocess.run(
            [SYSTEM_PYTHON, str(_MODAL_CALL), str(nc_path), str(args_path)],
            capture_output=True, text=True, timeout=900,
        )
    if proc.returncode != 0:
        raise RuntimeError(
            f"modal dispatch failed (rc={proc.returncode}): {proc.stderr.strip()}"
        )
    return json.loads(proc.stdout)


def dispatch(networks=NETWORKS, write: bool = True) -> GpuRun:
    """Build the bounded nets, solve each on CPU + once on the H100, cache it.

    This is the single cost-incurring entrypoint. It runs Modal once per network
    in a single session (the warm container is reused inside the scaledown
    window), recomputes the GPU objective from the returned dispatch, and writes
    one timestamped JSON to ``data/gpu_runs/``.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    records = []
    for name, n_buses, hours in networks:
        pnet = build_pypsa_network(n_buses, hours)
        cpu = cpu_solve(pnet)
        nc_bytes = network_to_nc_bytes(pnet)
        gpu = _call_modal(nc_bytes, GPU_ADMM_ARGS)
        gpu_obj = gpu_objective_from_outcome(pnet, gpu["outcome"])
        records.append(GpuRunRecord(
            network=name,
            n_buses=n_buses,
            hours=hours,
            machine=str(gpu["machine"]),
            gpu_tier=gpu.get("gpu"),
            gpu_elapsed_s=float(gpu["elapsed_s"]),
            cpu_elapsed_s=float(cpu["elapsed_s"]),
            gpu_objective=gpu_obj,
            cpu_objective=cpu["objective"],
            objective_gap=objective_gap(cpu["objective"], gpu_obj),
            solver_args=gpu.get("solver_args", GPU_ADMM_ARGS),
        ))
    run = GpuRun(timestamp=timestamp, records=records)
    if write:
        run.write_json(GPU_RUNS_DIR / f"{timestamp}.json")
    return run


def run(report_path=None) -> Optional[BenchResult]:
    """Emit a :class:`BenchResult` from the cached GPU run, if one exists."""
    cached = latest_cached_run()
    if cached is None:
        return None
    result = cached.to_bench_result()
    if report_path is not None:
        result.write_markdown(report_path)
    return result


if __name__ == "__main__":
    if "--dispatch" in sys.argv[1:]:
        if not modal_available():
            print("modal CLI / ~/.modal.toml not available; cannot dispatch.")
            raise SystemExit(1)
        gpu_run = dispatch()
        print(f"cached GPU run {gpu_run.timestamp}")
        for rec in gpu_run.records:
            print(f"  {rec.network:>7} buses={rec.n_buses:>3} "
                  f"machine={rec.machine} gpu_s={rec.gpu_elapsed_s:.3f} "
                  f"cpu_s={rec.cpu_elapsed_s:.3f} obj_gap={rec.objective_gap:.3e}")
        print(f"max objective gap: {gpu_run.max_objective_gap:.3e} (tol {OBJECTIVE_GAP_TOL})")
    else:
        bench = run()
        if bench is None:
            print("no cached GPU run; run with --dispatch to create one.")
        else:
            print(bench.to_json())

"""Plot results from multi-year stochastic planning experiments.

Reads JSON result files produced by runner.py and generates:
  1. Convergence plots (loss, suboptimality, gradient norm, timing)
  2. Final capacity bar chart aggregated by generator type (carrier)
  3. Capacity trajectory over iterations (if param history is available)

Usage:
    python experiments/multi_year/plot_results.py outputs/results.json
    python experiments/multi_year/plot_results.py outputs/results.json --output-dir plots/
    python experiments/multi_year/plot_results.py outputs/results.json --network data/network.nc
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pypsa

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["font.size"] = 10

# ---------------------------------------------------------------------------
# Carrier colours (consistent with gradient_ra and experiments/plan/plotter)
# ---------------------------------------------------------------------------

CARRIER_COLORS: Dict[str, str] = {
    "solar": "#f0ad4e",
    "onwind": "#5cb85c",
    "offwind": "#2e8b57",
    "offwind_floating": "#2e8b57",
    "hydro": "#3498db",
    "CCGT": "#e67e22",
    "OCGT": "#d35400",
    "coal": "#7f8c8d",
    "nuclear": "#9b59b6",
    "oil": "#2c3e50",
    "biomass": "#27ae60",
    "geothermal": "#8b4513",
    "hydrogen_ct": "#1abc9c",
    "CCGT-95CCS": "#e74c3c",
    "battery": "#3498db",
    "H2": "#16a085",
}

_TAB20 = matplotlib.colormaps["tab20"]

# Map fine-grained carriers to aggregate type groups
CARRIER_GROUPS: Dict[str, str] = {
    "nuclear": "Nuclear",
    "coal": "Coal",
    "CCGT": "Gas",
    "OCGT": "Gas",
    "CCGT-95CCS": "Gas CCS",
    "hydro": "Hydro",
    "geothermal": "Geo",
    "onwind": "Wind",
    "offwind": "Wind",
    "offwind_floating": "Wind",
    "solar": "Solar",
    "biomass": "Biomass",
    "oil": "Oil",
    "hydrogen_ct": "H2 CT",
    "battery": "Battery",
    "H2": "H2 Storage",
}

GROUP_COLORS: Dict[str, str] = {
    "Solar": "#f0ad4e",
    "Wind": "#5cb85c",
    "Gas": "#e67e22",
    "Gas CCS": "#e74c3c",
    "Coal": "#7f8c8d",
    "Nuclear": "#9b59b6",
    "Hydro": "#3498db",
    "Geo": "#8b4513",
    "Biomass": "#27ae60",
    "Oil": "#2c3e50",
    "H2 CT": "#1abc9c",
    "Battery": "#3498db",
    "H2 Storage": "#16a085",
}


def _color(carrier: str, idx: int = 0) -> str:
    if carrier in GROUP_COLORS:
        return GROUP_COLORS[carrier]
    if carrier in CARRIER_COLORS:
        return CARRIER_COLORS[carrier]
    return _TAB20(idx % 20)


def _group_carrier(carrier: str) -> str:
    """Map a fine-grained carrier name to its aggregate group."""
    return CARRIER_GROUPS.get(carrier, carrier)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def load_results(json_path: Path) -> dict:
    """Load a results JSON file."""
    with open(json_path) as f:
        return json.load(f)


def resolve_network_path(results: dict, explicit_path: Optional[Path]) -> Optional[Path]:
    """Find a PyPSA network .nc file for carrier mapping.

    Checks (in order): explicit --network arg, exported network, first input
    network file.
    """
    if explicit_path is not None and explicit_path.exists():
        return explicit_path

    # Try the exported optimized network
    export_path = results.get("export_path")
    if export_path and Path(export_path).exists():
        return Path(export_path)

    # Try the first input network
    config = results.get("config", {})
    network_files = config.get("network_files", [])
    if network_files:
        # Resolve relative to the experiment data directory
        script_dir = Path(__file__).resolve().parent
        data_dir = script_dir / "data"
        candidate = data_dir / network_files[0]
        if candidate.exists():
            return candidate

    return None


def load_carrier_map(network_path: Path) -> Dict[str, List[str]]:
    """Load carrier labels for generators and storage units from a PyPSA network.

    Returns a dict with keys ``"generators"`` and ``"storage_units"``, each
    mapping to a list of carrier strings in component order.
    """
    net = pypsa.Network(str(network_path))
    result = {
        "generators": list(net.generators.carrier.values),
        "storage_units": (
            list(net.storage_units.carrier.values)
            if len(net.storage_units) > 0
            else []
        ),
    }
    return result


def _flatten_param_values(values) -> np.ndarray:
    """Convert nested list-of-lists [[v1], [v2], ...] to flat numpy array."""
    return np.array(values).flatten()


def _get_carriers(
    results: dict,
    param_key: str,
    carrier_map: Optional[Dict[str, List[str]]],
    carrier_map_key: str,
    n_expected: int,
) -> List[str]:
    """Resolve carrier labels for a parameter, with grouped names.

    Priority:
      1. ``results["carrier_labels"][param_key]`` (embedded by runner.py)
      2. ``carrier_map[carrier_map_key]`` (loaded from network file)
      3. Generic fallback labels
    """
    # Try embedded labels first
    embedded = results.get("carrier_labels", {}).get(param_key)
    if embedded is not None and len(embedded) == n_expected:
        return [_group_carrier(c) for c in embedded]

    # Try network-derived carrier map
    if carrier_map is not None:
        labels = carrier_map.get(carrier_map_key, [])
        if len(labels) == n_expected:
            return [_group_carrier(c) for c in labels]

    return [f"{param_key}_{i}" for i in range(n_expected)]


# ---------------------------------------------------------------------------
# Plot 1: Convergence
# ---------------------------------------------------------------------------


def plot_convergence(results: dict, output_dir: Path) -> None:
    """2x2 convergence figure: loss, suboptimality, gradient norm, timing."""
    history = results.get("history", {})
    if not history:
        print("  No history data — skipping convergence plot.", flush=True)
        return

    n_iters = len(history.get("loss", []))
    iters = np.arange(n_iters)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # --- Top-left: Loss ---
    ax = axes[0, 0]
    loss_vals = [float(x) for x in history.get("loss", [])]
    if loss_vals:
        ax.plot(iters, loss_vals, label="Loss", alpha=0.4, linewidth=0.8)
    rolling = history.get("rolling_loss", [])
    if rolling:
        ax.plot(iters, rolling, label="Rolling loss", linewidth=1.5)
    lower = results.get("lower_bound")
    if lower is not None:
        ax.axhline(lower, color="red", linestyle="--", linewidth=1, label="Lower bound")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost")
    ax.set_title("Objective Value")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # --- Top-right: Suboptimality ---
    ax = axes[0, 1]
    subopt = history.get("suboptimality", [])
    if subopt:
        ax.plot(iters, subopt, linewidth=1.5, color="tab:red")
        ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Suboptimality (J/LB - 1)")
    ax.set_title("Suboptimality Gap")
    ax.grid(alpha=0.3)

    # --- Bottom-left: Gradient norms ---
    ax = axes[1, 0]
    gn = history.get("grad_norm", [])
    pgn = history.get("proj_grad_norm", [])
    if gn:
        ax.plot(iters, gn, label="Grad norm", linewidth=1, alpha=0.6)
    if pgn:
        ax.plot(iters, pgn, label="Proj grad norm", linewidth=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Norm")
    ax.set_title("Gradient Norms")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # --- Bottom-right: Loss vs wall-clock time ---
    ax = axes[1, 1]
    times = history.get("time", [])
    if times and loss_vals:
        ax.plot(times, loss_vals, alpha=0.4, linewidth=0.8, label="Loss")
        if rolling:
            ax.plot(times, rolling, linewidth=1.5, label="Rolling loss")
        if lower is not None:
            ax.axhline(lower, color="red", linestyle="--", linewidth=1, label="Lower bound")
        ax.legend(fontsize=8)
    ax.set_xlabel("Wall-Clock Time (s)")
    ax.set_ylabel("Cost")
    ax.set_title("Convergence vs Time")
    ax.grid(alpha=0.3)

    experiment_name = results.get("config", {}).get("name", "experiment")
    fig.suptitle(f"Convergence: {experiment_name}", fontsize=13, y=1.01)
    fig.tight_layout()
    out_path = output_dir / "convergence.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Plot 2: Final capacity by carrier
# ---------------------------------------------------------------------------


def plot_final_capacities(
    results: dict,
    carrier_map: Optional[Dict[str, List[str]]],
    output_dir: Path,
) -> None:
    """Bar chart of final optimized capacity aggregated by carrier type."""
    optimal = results.get("optimal_parameters", {})
    power_unit = results.get("config", {}).get("pypsa_args", {}).get("power_unit", 1.0)

    fig, axes = plt.subplots(1, 3, figsize=(16, 6), gridspec_kw={"width_ratios": [3, 1, 1]})

    # --- Generators ---
    ax = axes[0]
    gen_caps = optimal.get("generator_capacity")
    if gen_caps is not None:
        caps_mw = _flatten_param_values(gen_caps) * power_unit
        carriers = _get_carriers(results, "generator_capacity", carrier_map, "generators", len(caps_mw))

        # Aggregate by carrier group
        carrier_totals: Dict[str, float] = {}
        for cap, carrier in zip(caps_mw, carriers):
            carrier_totals[carrier] = carrier_totals.get(carrier, 0.0) + cap
        # Convert to GW
        carrier_totals = {k: v / 1e3 for k, v in carrier_totals.items()}

        sorted_carriers = sorted(carrier_totals.keys(), key=lambda c: -carrier_totals[c])
        vals = [carrier_totals[c] for c in sorted_carriers]
        colors = [_color(c, i) for i, c in enumerate(sorted_carriers)]

        bars = ax.bar(sorted_carriers, vals, color=colors, edgecolor="black", linewidth=0.5)
        for bar in bars:
            h = bar.get_height()
            if h > 0.01:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h,
                    f"{h:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_xlabel("Carrier")
    ax.set_ylabel("Capacity (GW)")
    ax.set_title("Generator Capacity by Type")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)

    # --- Storage ---
    ax = axes[1]
    su_caps = optimal.get("storageunit_power")
    if su_caps is not None:
        su_mw = _flatten_param_values(su_caps) * power_unit
        su_carriers = _get_carriers(results, "storageunit_power", carrier_map, "storage_units", len(su_mw))

        su_totals: Dict[str, float] = {}
        for cap, carrier in zip(su_mw, su_carriers):
            su_totals[carrier] = su_totals.get(carrier, 0.0) + cap
        su_totals = {k: v / 1e3 for k, v in su_totals.items()}

        sorted_su = sorted(su_totals.keys(), key=lambda c: -su_totals[c])
        vals = [su_totals[c] for c in sorted_su]
        colors = [_color(c, i) for i, c in enumerate(sorted_su)]
        ax.bar(sorted_su, vals, color=colors, edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Carrier")
    ax.set_ylabel("Capacity (GW)")
    ax.set_title("Storage Power")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)

    # --- Transmission ---
    ax = axes[2]
    tx_types = []
    tx_vals = []
    for key, label in [("dcline_capacity", "DC Lines"), ("acline_capacity", "AC Lines")]:
        caps = optimal.get(key)
        if caps is not None:
            total_gw = _flatten_param_values(caps).sum() * power_unit / 1e3
            tx_types.append(label)
            tx_vals.append(total_gw)
    if tx_types:
        ax.bar(tx_types, tx_vals, color=["tab:blue", "tab:orange"][: len(tx_types)],
               edgecolor="black", linewidth=0.5)
        for i, v in enumerate(tx_vals):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Capacity (GW)")
    ax.set_title("Transmission")
    ax.grid(axis="y", alpha=0.3)

    experiment_name = results.get("config", {}).get("name", "experiment")
    fig.suptitle(f"Final Optimized Capacities: {experiment_name}", fontsize=13, y=1.01)
    fig.tight_layout()
    out_path = output_dir / "final_capacities.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Plot 3: Capacity trajectory over iterations (requires param history)
# ---------------------------------------------------------------------------


def plot_capacity_trajectory(
    results: dict,
    carrier_map: Optional[Dict[str, List[str]]],
    output_dir: Path,
) -> None:
    """Line plots of total capacity per carrier over iterations.

    Requires the ``"param"`` key in the results history (enabled by setting
    ``save_param_history: true`` in the optimizer config).
    """
    history = results.get("history", {})
    param_history = history.get("param")
    if not param_history:
        print("  No param history — skipping capacity trajectory plot.", flush=True)
        print("  (Set save_param_history: true in optimizer config for future runs.)",
              flush=True)
        return

    power_unit = results.get("config", {}).get("pypsa_args", {}).get("power_unit", 1.0)
    n_iters = len(param_history)
    iters = np.arange(n_iters)

    # ---- Generator trajectories ----
    has_gen_trajectory = "generator_capacity" in param_history[0]

    if has_gen_trajectory:
        # Extract generator capacities at each iteration
        all_gen_caps = []
        for snapshot in param_history:
            caps = _flatten_param_values(snapshot["generator_capacity"]) * power_unit / 1e3  # GW
            all_gen_caps.append(caps)
        all_gen_caps = np.array(all_gen_caps)  # (n_iters, n_generators)

        n_gens = all_gen_caps.shape[1]
        carriers = _get_carriers(results, "generator_capacity", carrier_map, "generators", n_gens)

        # Aggregate by carrier group at each iteration
        unique_carriers = sorted(set(carriers))
        carrier_trajectories: Dict[str, np.ndarray] = {}
        for c in unique_carriers:
            mask = np.array([car == c for car in carriers])
            carrier_trajectories[c] = all_gen_caps[:, mask].sum(axis=1)

        # Sort by final capacity (descending)
        sorted_carriers = sorted(
            unique_carriers, key=lambda c: -carrier_trajectories[c][-1]
        )

        fig, ax = plt.subplots(figsize=(12, 6))
        for idx, c in enumerate(sorted_carriers):
            traj = carrier_trajectories[c]
            if traj[-1] < 0.001:  # skip negligible carriers
                continue
            ax.plot(iters, traj, label=c, color=_color(c, idx), linewidth=1.5)

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Total Capacity (GW)")
        ax.set_title("Generator Capacity by Type Over Iterations")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out_path = output_dir / "capacity_trajectory_generators.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}", flush=True)

    # ---- Storage trajectory ----
    has_su_trajectory = "storageunit_power" in param_history[0]

    if has_su_trajectory:
        all_su_caps = []
        for snapshot in param_history:
            caps = _flatten_param_values(snapshot["storageunit_power"]) * power_unit / 1e3
            all_su_caps.append(caps)
        all_su_caps = np.array(all_su_caps)

        n_su = all_su_caps.shape[1]
        su_labels = _get_carriers(results, "storageunit_power", carrier_map, "storage_units", n_su)

        unique_su = sorted(set(su_labels))
        su_trajectories: Dict[str, np.ndarray] = {}
        for c in unique_su:
            mask = np.array([lab == c for lab in su_labels])
            su_trajectories[c] = all_su_caps[:, mask].sum(axis=1)

        fig, ax = plt.subplots(figsize=(10, 5))
        for idx, c in enumerate(unique_su):
            ax.plot(iters, su_trajectories[c], label=c, color=_color(c, idx), linewidth=1.5)

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Power Capacity (GW)")
        ax.set_title("Storage Capacity by Type Over Iterations")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out_path = output_dir / "capacity_trajectory_storage.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}", flush=True)

    # ---- Transmission trajectory ----
    tx_keys = [
        ("dcline_capacity", "DC Lines"),
        ("acline_capacity", "AC Lines"),
    ]
    tx_found = [k for k, _ in tx_keys if k in param_history[0]]
    if tx_found:
        fig, ax = plt.subplots(figsize=(10, 5))
        for key, label in tx_keys:
            if key not in param_history[0]:
                continue
            traj = []
            for snapshot in param_history:
                total = _flatten_param_values(snapshot[key]).sum() * power_unit / 1e3
                traj.append(total)
            ax.plot(iters, traj, label=label, linewidth=1.5)

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Total Capacity (GW)")
        ax.set_title("Transmission Capacity Over Iterations")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out_path = output_dir / "capacity_trajectory_transmission.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Plot 4: Initial vs Final capacity comparison
# ---------------------------------------------------------------------------


def _load_initial_from_network(
    network_path: Path,
    carrier_map: Optional[Dict[str, List[str]]],
    power_unit: float,
) -> Optional[dict]:
    """Fallback: derive initial parameters from a PyPSA network file.

    Returns a dict shaped like ``results["initial_parameters"]`` keyed by
    the same names used in runner.py's ``setup_parameter_names``.
    """
    try:
        net = pypsa.Network(str(network_path))
    except Exception:
        return None

    initial: dict = {}

    # Generator nominal capacities (p_nom)
    if len(net.generators) > 0:
        caps = net.generators.p_nom.values / power_unit
        initial["generator_capacity"] = caps.tolist()

    # Storage-unit power capacities (p_nom)
    if len(net.storage_units) > 0:
        caps = net.storage_units.p_nom.values / power_unit
        initial["storageunit_power"] = caps.tolist()

    # DC lines (s_nom)
    if len(net.links) > 0:
        caps = net.links.p_nom.values / power_unit
        initial["dcline_capacity"] = caps.tolist()

    # AC lines (s_nom)
    if len(net.lines) > 0:
        caps = net.lines.s_nom.values / power_unit
        initial["acline_capacity"] = caps.tolist()

    return initial


def plot_initial_vs_final_capacities(
    results: dict,
    carrier_map: Optional[Dict[str, List[str]]],
    output_dir: Path,
    network_path: Optional[Path] = None,
) -> None:
    """Grouped bar chart comparing initial and optimized capacities.

    Falls back to loading initial capacities from a PyPSA network file if
    ``initial_parameters`` is not present in *results* (backward compat).
    """
    optimal = results.get("optimal_parameters", {})
    initial = results.get("initial_parameters")
    power_unit = results.get("config", {}).get("pypsa_args", {}).get("power_unit", 1.0)

    # Fallback for older result files
    if initial is None:
        if network_path is not None:
            print("  initial_parameters missing — loading from network file.", flush=True)
            initial = _load_initial_from_network(network_path, carrier_map, power_unit)
        if initial is None:
            print("  No initial_parameters and no network file — skipping.", flush=True)
            return

    fig, axes = plt.subplots(1, 3, figsize=(16, 6), gridspec_kw={"width_ratios": [3, 1, 1]})
    bar_width = 0.35

    # --- Generators by carrier group ---
    ax = axes[0]
    gen_init = initial.get("generator_capacity")
    gen_opt = optimal.get("generator_capacity")
    if gen_init is not None and gen_opt is not None:
        init_mw = _flatten_param_values(gen_init) * power_unit
        opt_mw = _flatten_param_values(gen_opt) * power_unit
        carriers = _get_carriers(results, "generator_capacity", carrier_map, "generators", len(opt_mw))

        # Aggregate by carrier group
        init_totals: Dict[str, float] = {}
        opt_totals: Dict[str, float] = {}
        for ic, oc, carrier in zip(init_mw, opt_mw, carriers):
            init_totals[carrier] = init_totals.get(carrier, 0.0) + ic
            opt_totals[carrier] = opt_totals.get(carrier, 0.0) + oc

        # Convert to GW
        init_totals = {k: v / 1e3 for k, v in init_totals.items()}
        opt_totals = {k: v / 1e3 for k, v in opt_totals.items()}

        sorted_carriers = sorted(opt_totals.keys(), key=lambda c: -opt_totals[c])
        x = np.arange(len(sorted_carriers))

        init_vals = [init_totals.get(c, 0.0) for c in sorted_carriers]
        opt_vals = [opt_totals[c] for c in sorted_carriers]
        colors = [_color(c, i) for i, c in enumerate(sorted_carriers)]

        ax.bar(x - bar_width / 2, init_vals, bar_width,
               color="0.75", edgecolor="black", linewidth=0.5, label="Initial")
        bars = ax.bar(x + bar_width / 2, opt_vals, bar_width,
                      color=colors, edgecolor="black", linewidth=0.5, label="Optimized")
        for bar in bars:
            h = bar.get_height()
            if h > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.1f}",
                        ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(sorted_carriers)

    ax.set_xlabel("Carrier")
    ax.set_ylabel("Capacity (GW)")
    ax.set_title("Generator Capacity by Type")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # --- Storage ---
    ax = axes[1]
    su_init = initial.get("storageunit_power")
    su_opt = optimal.get("storageunit_power")
    if su_init is not None and su_opt is not None:
        init_mw = _flatten_param_values(su_init) * power_unit
        opt_mw = _flatten_param_values(su_opt) * power_unit
        su_carriers = _get_carriers(results, "storageunit_power", carrier_map, "storage_units", len(opt_mw))

        su_init_totals: Dict[str, float] = {}
        su_opt_totals: Dict[str, float] = {}
        for ic, oc, carrier in zip(init_mw, opt_mw, su_carriers):
            su_init_totals[carrier] = su_init_totals.get(carrier, 0.0) + ic
            su_opt_totals[carrier] = su_opt_totals.get(carrier, 0.0) + oc
        su_init_totals = {k: v / 1e3 for k, v in su_init_totals.items()}
        su_opt_totals = {k: v / 1e3 for k, v in su_opt_totals.items()}

        sorted_su = sorted(su_opt_totals.keys(), key=lambda c: -su_opt_totals[c])
        x = np.arange(len(sorted_su))
        init_vals = [su_init_totals.get(c, 0.0) for c in sorted_su]
        opt_vals = [su_opt_totals[c] for c in sorted_su]
        colors = [_color(c, i) for i, c in enumerate(sorted_su)]

        ax.bar(x - bar_width / 2, init_vals, bar_width,
               color="0.75", edgecolor="black", linewidth=0.5, label="Initial")
        ax.bar(x + bar_width / 2, opt_vals, bar_width,
               color=colors, edgecolor="black", linewidth=0.5, label="Optimized")
        ax.set_xticks(x)
        ax.set_xticklabels(sorted_su)

    ax.set_xlabel("Carrier")
    ax.set_ylabel("Capacity (GW)")
    ax.set_title("Storage Power")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)

    # --- Transmission ---
    ax = axes[2]
    tx_labels = []
    tx_init_vals = []
    tx_opt_vals = []
    for key, label in [("dcline_capacity", "DC Lines"), ("acline_capacity", "AC Lines")]:
        init_caps = initial.get(key)
        opt_caps = optimal.get(key)
        if init_caps is not None and opt_caps is not None:
            tx_labels.append(label)
            tx_init_vals.append(_flatten_param_values(init_caps).sum() * power_unit / 1e3)
            tx_opt_vals.append(_flatten_param_values(opt_caps).sum() * power_unit / 1e3)
    if tx_labels:
        x = np.arange(len(tx_labels))
        ax.bar(x - bar_width / 2, tx_init_vals, bar_width,
               color="0.75", edgecolor="black", linewidth=0.5, label="Initial")
        ax.bar(x + bar_width / 2, tx_opt_vals, bar_width,
               color=["tab:blue", "tab:orange"][:len(tx_labels)],
               edgecolor="black", linewidth=0.5, label="Optimized")
        for i, v in enumerate(tx_opt_vals):
            ax.text(i + bar_width / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(tx_labels)
    ax.set_ylabel("Capacity (GW)")
    ax.set_title("Transmission")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    experiment_name = results.get("config", {}).get("name", "experiment")
    fig.suptitle(f"Initial vs Optimized Capacities: {experiment_name}", fontsize=13, y=1.01)
    fig.tight_layout()
    out_path = output_dir / "initial_vs_final_capacities.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot results from multi-year stochastic planning experiments."
    )
    parser.add_argument(
        "results_json",
        type=Path,
        help="Path to the results JSON file produced by runner.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for output plots. Default: same directory as the JSON file.",
    )
    parser.add_argument(
        "--network",
        type=Path,
        default=None,
        help=(
            "Path to a PyPSA .nc network file for carrier mapping. "
            "If not provided, auto-detected from the results config."
        ),
    )
    args = parser.parse_args()

    if not args.results_json.exists():
        print(f"ERROR: File not found: {args.results_json}", flush=True)
        sys.exit(1)

    output_dir = args.output_dir or args.results_json.parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60, flush=True)
    print("Multi-Year Experiment Results Plotter", flush=True)
    print("=" * 60, flush=True)

    # Load results
    print(f"Loading results from {args.results_json} ...", flush=True)
    results = load_results(args.results_json)
    experiment_name = results.get("config", {}).get("name", "unknown")
    n_iters = len(results.get("history", {}).get("loss", []))
    print(f"  Experiment: {experiment_name}", flush=True)
    print(f"  Iterations: {n_iters}", flush=True)
    print(f"  Final cost: {results.get('final_cost', 'N/A')}", flush=True)
    lower = results.get("lower_bound")
    if lower is not None:
        gap = (results["final_cost"] / lower - 1) * 100
        print(f"  Lower bound: {lower:.2f}  (gap: {gap:.2f}%)", flush=True)

    has_param_history = "param" in results.get("history", {})
    print(f"  Param history available: {has_param_history}", flush=True)

    # Load carrier mapping
    network_path = resolve_network_path(results, args.network)
    carrier_map = None
    if network_path:
        print(f"Loading carrier map from {network_path} ...", flush=True)
        carrier_map = load_carrier_map(network_path)
        print(
            f"  {len(carrier_map['generators'])} generators, "
            f"{len(carrier_map['storage_units'])} storage units",
            flush=True,
        )
    else:
        print("  No network file found — using generic labels.", flush=True)

    # Generate plots
    print("\nGenerating plots...", flush=True)

    print("[1/4] Convergence...", flush=True)
    plot_convergence(results, output_dir)

    print("[2/4] Final capacities...", flush=True)
    plot_final_capacities(results, carrier_map, output_dir)

    print("[3/4] Capacity trajectories...", flush=True)
    plot_capacity_trajectory(results, carrier_map, output_dir)

    print("[4/4] Initial vs final capacities...", flush=True)
    plot_initial_vs_final_capacities(results, carrier_map, output_dir, network_path)

    print(f"\nAll plots saved to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()

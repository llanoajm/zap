"""
Multi-year stochastic planning experiment runner.

This module provides functionality for running stochastic planning experiments
with multiple weather years, using block sampling for computational tractability.

Example usage:
    python experiments/multi_year/runner.py config.yaml
"""

import numpy as np
import pandas as pd
import cvxpy as cp
import pypsa
import yaml
import json
import logging
import time

from pathlib import Path
from typing import Union, List
from copy import deepcopy

import zap
import zap.planning.trackers as tr
from zap.importers.multi_year import MultiYearBlockSampler
from zap.planning import (
    InvestmentObjective,
    RelaxedPlanningProblem,
    GradientDescent,
)
from zap.planning.operation_objectives import DispatchCostObjective

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Path Configuration
# ============================================================================

ZAP_PATH = Path(zap.__file__).parent.parent
DATA_PATH = ZAP_PATH / "data"
EXPERIMENT_PATH = ZAP_PATH / "experiments" / "multi_year"
EXPERIMENT_DATA_PATH = EXPERIMENT_PATH / "data"
OUTPUT_PATH = EXPERIMENT_PATH / "outputs"


# ============================================================================
# Default Configuration
# ============================================================================

DEFAULT_PYPSA_ARGS = {
    "power_unit": 1.0e3,
    "cost_unit": 10.0,
}

DEFAULT_SOLVER_ARGS = {
    # Note: Don't pass solver here - it causes deepcopy issues
    # Pass solver to RelaxedPlanningProblem instead
}


# ============================================================================
# Config Loading
# ============================================================================


def load_config(config_path: Union[str, Path]) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def datadir(*args) -> Path:
    """Get path relative to output directory."""
    path = OUTPUT_PATH.joinpath(*args)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# ============================================================================
# Network Loading
# ============================================================================


def list_available_networks(data_path: Path = None) -> List[str]:
    """
    List available PyPSA network files in the data directory.

    Parameters
    ----------
    data_path : Path, optional
        Path to data directory. Defaults to EXPERIMENT_DATA_PATH.

    Returns
    -------
    List[str]
        List of network filenames (sorted by name)
    """
    if data_path is None:
        data_path = EXPERIMENT_DATA_PATH

    if not data_path.exists():
        logger.warning(f"Data directory does not exist: {data_path}")
        return []

    networks = sorted([f.name for f in data_path.glob("*.nc")])
    return networks


def load_multi_year_networks(
    network_files: List[str] = None,
    data_path: Path = None,
    snapshots: pd.DatetimeIndex = None,
    hours_per_year: int = None,
) -> tuple[List[pypsa.Network], List[pd.DatetimeIndex]]:
    """
    Load multiple PyPSA networks from the data directory.

    Parameters
    ----------
    network_files : List[str], optional
        List of network filenames to load. If None, loads all .nc files.
    data_path : Path, optional
        Path to data directory. Defaults to EXPERIMENT_DATA_PATH.
    snapshots : pd.DatetimeIndex, optional
        Specific snapshots to use. If None, uses all snapshots from each network.
    hours_per_year : int, optional
        Limit number of hours per year. If None, uses all hours.

    Returns
    -------
    tuple[List[pypsa.Network], List[pd.DatetimeIndex]]
        Lists of loaded networks and their snapshots
    """
    if data_path is None:
        data_path = EXPERIMENT_DATA_PATH

    if network_files is None:
        network_files = list_available_networks(data_path)

    if len(network_files) == 0:
        raise ValueError(f"No network files found in {data_path}")

    networks = []
    snapshots_list = []

    for filename in network_files:
        filepath = data_path / filename
        logger.info(f"Loading network: {filepath}")

        net = pypsa.Network(str(filepath))
        networks.append(net)

        # Determine snapshots for this network
        if snapshots is not None:
            net_snapshots = snapshots
        elif hours_per_year is not None:
            net_snapshots = net.snapshots[:hours_per_year]
        else:
            net_snapshots = net.snapshots

        snapshots_list.append(net_snapshots)
        logger.info(f"  Loaded {len(net_snapshots)} snapshots from {filename}")

    logger.info(f"Loaded {len(networks)} networks")
    return networks, snapshots_list


# ============================================================================
# Experiment Setup
# ============================================================================


def setup_parameter_names(devices: list) -> dict[str, tuple[int, str]]:
    """
    Create parameter_names dict for planning from device list.

    Returns mapping from parameter name to (device_index, attribute_name).
    Uses device type and index for simple, consistent naming.
    """
    parameter_names = {}

    for i, dev in enumerate(devices):
        dev_type = type(dev).__name__.lower()

        # Check for expandable capacity
        if hasattr(dev, "nominal_capacity"):
            cap = dev.nominal_capacity
            if cap is not None and hasattr(dev, "capital_cost") and dev.capital_cost is not None:
                parameter_names[f"{dev_type}_capacity"] = (i, "nominal_capacity")

        if hasattr(dev, "power_capacity"):
            cap = dev.power_capacity
            if cap is not None and hasattr(dev, "capital_cost") and dev.capital_cost is not None:
                parameter_names[f"{dev_type}_power"] = (i, "power_capacity")

    return parameter_names


def run_experiment(config: dict) -> dict:
    """
    Run a multi-year stochastic planning experiment.

    Parameters
    ----------
    config : dict
        Experiment configuration with keys:
        - network_files: List of network filenames (optional, loads all if not specified)
        - hours_per_year: Hours per year (optional, uses all hours if not specified)
        - block_size: Hours per block (default 168 for weekly)
        - sampling_strategy: 'all', 'random', 'stratified', 'uniform'
        - num_blocks: Number of blocks (for random/stratified)
        - num_workers: Parallel workers for StochasticPlanningProblem
        - pypsa_args: Arguments for load_pypsa_network (optional)

    Returns
    -------
    dict
        Experiment results including optimal capacities and costs
    """
    logger.info("=" * 60)
    logger.info("Starting multi-year stochastic planning experiment")
    logger.info("=" * 60)

    # Extract config
    network_files = config.get("network_files", None)
    hours_per_year = config.get("hours_per_year", None)
    block_size = config.get("block_size", 168)
    sampling_strategy = config.get("sampling_strategy", "all")
    num_blocks = config.get("num_blocks", None)
    num_workers = config.get("num_workers", 1)
    seed = config.get("seed", 42)
    pypsa_args = config.get("pypsa_args", DEFAULT_PYPSA_ARGS)

    # -------------------------------------------------------------------------
    # Step 1: Load multi-year networks
    # -------------------------------------------------------------------------
    logger.info("Loading multi-year networks...")
    start_time = time.time()

    networks, snapshots_list = load_multi_year_networks(
        network_files=network_files,
        hours_per_year=hours_per_year,
    )

    load_time = time.time() - start_time
    logger.info(f"Network loading took {load_time:.2f}s")

    # -------------------------------------------------------------------------
    # Step 2: Create MultiYearBlockSampler
    # -------------------------------------------------------------------------
    logger.info("Creating MultiYearBlockSampler...")
    start_time = time.time()

    sampler = MultiYearBlockSampler(
        networks,
        snapshots_list,
        **pypsa_args,
    )

    logger.info(f"Sampler summary: {sampler.summary()}")
    sampler_time = time.time() - start_time
    logger.info(f"Sampler creation took {sampler_time:.2f}s")

    # -------------------------------------------------------------------------
    # Step 3: Sample blocks
    # -------------------------------------------------------------------------
    logger.info(f"Sampling blocks with strategy='{sampling_strategy}'...")

    blocks = sampler.sample_blocks(
        block_size=block_size,
        num_blocks=num_blocks,
        strategy=sampling_strategy,
        seed=seed,
    )

    logger.info(f"Sampled {len(blocks)} blocks of {block_size} hours each")
    logger.info(f"Total hours covered: {len(blocks) * block_size}")

    # -------------------------------------------------------------------------
    # Step 4: Setup parameter names for planning
    # -------------------------------------------------------------------------
    parameter_names = setup_parameter_names(sampler.base_devices)
    logger.info(f"Planning parameters: {list(parameter_names.keys())}")

    # -------------------------------------------------------------------------
    # Step 5: Create StochasticPlanningProblem
    # -------------------------------------------------------------------------
    logger.info("Creating StochasticPlanningProblem...")
    start_time = time.time()

    def op_objective_fn(devices):
        return DispatchCostObjective(sampler.base_network, devices)

    def inv_objective_fn(devices, layer):
        return InvestmentObjective(devices, layer)

    problem = sampler.create_stochastic_problem(
        blocks=blocks,
        parameter_names=parameter_names,
        operation_objective_fn=op_objective_fn,
        investment_objective_fn=inv_objective_fn,
        **DEFAULT_SOLVER_ARGS,
    )

    problem_time = time.time() - start_time
    logger.info(f"Problem creation took {problem_time:.2f}s")
    logger.info(f"Created problem with {problem.num_subproblems} subproblems")

    # -------------------------------------------------------------------------
    # Step 6: Initialize parallel workers
    # -------------------------------------------------------------------------
    if num_workers > 1:
        logger.info(f"Initializing {num_workers} parallel workers...")
        problem.initialize_workers(num_workers)

    # -------------------------------------------------------------------------
    # Step 7: Solve relaxed problem (for lower bound and initialization)
    # -------------------------------------------------------------------------
    relaxation_config = config.get("relaxation", {})
    should_solve_relaxation = relaxation_config.get("should_solve", True)

    relaxation_result = None
    if should_solve_relaxation:
        logger.info("Solving relaxed problem...")
        start_time = time.time()

        relaxation = RelaxedPlanningProblem(
            problem,
            max_price=relaxation_config.get("price_bound", 100.0),
            solver=cp.MOSEK,
            solver_kwargs={"verbose": False},
        )
        relaxed_params, relax_solve_data = relaxation.solve()

        relax_time = time.time() - start_time
        logger.info(f"Relaxation solve took {relax_time:.2f}s")

        lower_bound = relax_solve_data["problem"].value
        logger.info(f"Lower bound from relaxation: {lower_bound:.2f}")

        relaxation_result = {
            "relaxed_parameters": relaxed_params,
            "lower_bound": lower_bound,
            "solve_time": relax_time,
        }
    else:
        logger.info("Skipping relaxation...")
        relax_time = 0.0

    # -------------------------------------------------------------------------
    # Step 8: Solve the actual problem with gradient descent
    # -------------------------------------------------------------------------
    optimizer_config = config.get("optimizer", {})
    num_iterations = optimizer_config.get("num_iterations", 100)
    step_size = optimizer_config.get("step_size", 1e-3)
    clip = optimizer_config.get("clip", 1e3)
    initial_state_source = optimizer_config.get("initial_state", "relaxation")
    batch_size = optimizer_config.get("batch_size", 0)
    batch_strategy = optimizer_config.get("batch_strategy", "sequential")

    logger.info("Solving with gradient descent...")
    logger.info(f"  num_iterations: {num_iterations}")
    logger.info(f"  step_size: {step_size}")
    logger.info(f"  clip: {clip}")
    logger.info(f"  batch_size: {batch_size}")
    logger.info(f"  batch_strategy: {batch_strategy}")

    start_time = time.time()

    # Initialize from relaxation or from scratch
    if initial_state_source == "relaxation" and relaxation_result is not None:
        logger.info("Initializing with relaxation solution.")
        initial_state = deepcopy(relaxation_result["relaxed_parameters"])
    else:
        logger.info("Initializing with default parameters (no investment).")
        initial_state = None

    # Create algorithm
    algorithm = GradientDescent(step_size=step_size, clip=clip)

    # Solve
    optimal_params, history = problem.solve(
        num_iterations=num_iterations,
        algorithm=algorithm,
        trackers=tr.DEFAULT_TRACKERS,
        initial_state=initial_state,
        lower_bound=relaxation_result["lower_bound"] if relaxation_result else None,
        batch_size=batch_size,
        batch_strategy=batch_strategy,
        verbosity=1,
    )

    solve_time = time.time() - start_time
    logger.info(f"Gradient descent took {solve_time:.2f}s")

    # -------------------------------------------------------------------------
    # Step 9: Extract results
    # -------------------------------------------------------------------------
    # Evaluate final cost
    final_cost = problem.forward(**optimal_params)

    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info(f"Final cost: {final_cost:.2f}")
    if relaxation_result:
        logger.info(f"Lower bound: {relaxation_result['lower_bound']:.2f}")
        logger.info(
            f"Optimality gap: {(final_cost - relaxation_result['lower_bound']) / relaxation_result['lower_bound'] * 100:.2f}%"
        )
    logger.info("Optimal capacities:")
    for name, value in optimal_params.items():
        logger.info(f"  {name}: {value}")

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------
    if num_workers > 1:
        problem.shutdown_workers()

    return {
        "config": config,
        "num_subproblems": problem.num_subproblems,
        "final_cost": float(final_cost),
        "lower_bound": float(relaxation_result["lower_bound"]) if relaxation_result else None,
        "optimal_parameters": {
            k: v.tolist() if hasattr(v, "tolist") else v for k, v in optimal_params.items()
        },
        "history": {
            k: [float(x) if isinstance(x, (int, float, np.floating)) else x for x in v]
            for k, v in history.items()
        },
        "timing": {
            "network_loading": load_time,
            "sampler_creation": sampler_time,
            "problem_creation": problem_time,
            "relaxation_solve": relax_time,
            "gradient_descent": solve_time,
        },
        "sampler_summary": sampler.summary(),
    }


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    """Main entry point for the experiment."""
    import sys

    if len(sys.argv) > 1:
        # Load config from file
        config_path = sys.argv[1]
        config = load_config(config_path)
    else:
        # Default config using available data
        available = list_available_networks()
        logger.info(f"Available networks: {available}")

        config = {
            "name": "multi_year_experiment",
            "network_files": available,  # Use all available networks
            "hours_per_year": 168 * 4,  # 4 weeks per year for testing
            "block_size": 168,  # Weekly blocks
            "sampling_strategy": "all",
            "num_workers": 1,
            "seed": 42,
            "relaxation": {
                "should_solve": True,
                "price_bound": 100.0,
            },
            "optimizer": {
                "num_iterations": 100,
                "step_size": 1e-3,
                "clip": 1e3,  # Gradient clipping threshold
                "initial_state": "relaxation",
                "batch_size": 0,  # 0 = use all subproblems
                "batch_strategy": "sequential",
            },
        }

    # Run experiment
    results = run_experiment(config)

    # Save results
    output_path = datadir(f"{config.get('name', 'experiment')}_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved to {output_path}")

    return results


if __name__ == "__main__":
    main()

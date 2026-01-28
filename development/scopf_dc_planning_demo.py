import marimo

__generated_with = "0.11.21"
app = marimo.App(width="medium", app_title="SCOPF DC Planning Demo")


@app.cell
def _():
    import os
    from copy import deepcopy
    import cvxpy as cp
    import geopandas as gpd
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import pypsa
    import seaborn as sns
    import torch

    import zap
    from zap.admm import ADMMLayer, ADMMSolver
    from zap.devices import ACLine, DataCenterLoad
    from zap.importers.pypsa import load_pypsa_network, parse_buses

    sns.set_theme()
    return (
        ACLine,
        ADMMLayer,
        ADMMSolver,
        DataCenterLoad,
        cp,
        deepcopy,
        gpd,
        load_pypsa_network,
        mo,
        np,
        os,
        parse_buses,
        pd,
        plt,
        pypsa,
        sns,
        torch,
        zap,
    )


@app.cell
def _(mo):
    mo.md(
        """
        # SCOPF Data Center Planning Demo

        This notebook demonstrates Security-Constrained Optimal Power Flow (SCOPF)
        for network expansion planning with distributed data centers.

        **Key Features:**
        - N-1 contingency analysis (each line outage scenario)
        - Distributed data center placement optimization
        - Comparison: baseline vs SCOPF planning
        """
    )
    return


@app.cell
def _():
    LOAD_SCALING_FACTOR = 1.27
    GEN_SCALING_FACTOR = 1.24
    LINE_SCALING_FACTOR = 0.7

    # Candidate nodes for data center placement (sorted by land cost)
    INVESTMENT_NODE_CANDS = [32, 82, 50, 18, 15, 22, 43, 14, 23, 20, 94, 65, 78]

    UPSAMPLE_FACTOR = 4
    TIME_HORIZON = 24 * UPSAMPLE_FACTOR

    # Contingencies
    NUM_CONTINGENCIES = 10
    # Optional: set to an explicit list/array of line indices (e.g. cong_lines) to use as contingencies.
    # Example: CRITICAL_LINES = np.array([88, 168, 176, 170, 149, 49, 22, 80, 180, 73])
    CRITICAL_LINES = None
    return (
        CRITICAL_LINES,
        GEN_SCALING_FACTOR,
        INVESTMENT_NODE_CANDS,
        LINE_SCALING_FACTOR,
        LOAD_SCALING_FACTOR,
        NUM_CONTINGENCIES,
        TIME_HORIZON,
        UPSAMPLE_FACTOR,
    )


@app.cell
def _(mo):
    mo.md("""## Load PyPSA Network""")
    return


@app.cell
def upsample_zap_devices():
    def upsample_zap_devices(devices, factor=4, original_timesteps=24):
        """Upsample time-varying device attributes by repeating each timestep."""
        upsampled_devices = []
        for dev in devices:
            upsampled_devices.append(
                dev.sample_time(original_timesteps * factor, original_timesteps)
            )
        return upsampled_devices
    return (upsample_zap_devices,)


@app.cell
def _(INVESTMENT_NODE_CANDS, gpd, np, os, parse_buses, pd, pypsa):
    HOME_PATH = os.environ.get("HOME")
    PYPSA_NETWORK_PATH = HOME_PATH + "/zap_data/pypsa-networks/western_small/network_2023.nc"
    pn = pypsa.Network(PYPSA_NETWORK_PATH)
    snapshots = pn.generators_t.p_max_pu.index

    # Use 24 hours from peak hybrid day (load + renewables)
    snapshot_data = snapshots[5448:5472]  # 8/16/21

    buses, buses_to_index = parse_buses(pn)
    index_to_bus = {idx: name for name, idx in buses_to_index.items()}
    pypsa_bus_names = [index_to_bus[i] for i in INVESTMENT_NODE_CANDS]

    b = pn.buses.copy()
    gdf = gpd.GeoDataFrame(b, geometry=gpd.points_from_xy(b["x"], b["y"]), crs="EPSG:4326")
    county_url = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_county_500k.zip"
    counties = gpd.read_file(county_url)[
        ["STATEFP", "COUNTYFP", "GEOID", "NAME", "STATE_NAME", "geometry"]
    ]

    j = gpd.sjoin(gdf, counties.to_crs("EPSG:4326"), how="left", predicate="within")

    pn.buses["county_fips"] = j["GEOID"]
    pn.buses["county_name"] = j["NAME"]
    pn.buses["state_fips"] = j["STATEFP"]
    pn.buses["state_name"] = j["STATE_NAME"]

    selected_node_fips = pn.buses.loc[pypsa_bus_names, "county_fips"]
    county_land_lut_df = pd.read_csv(HOME_PATH + "/zap/development/county_land_lut.csv")

    sel = selected_node_fips.rename("county_fips").to_frame()
    bus_to_terminal = {bus: term for term, bus in index_to_bus.items()}
    sel["terminal"] = sel.index.map(bus_to_terminal)
    sel["county_fips"] = sel["county_fips"].astype(str).str.zfill(5)
    county_land_lut_df["county_fips"] = county_land_lut_df["county_fips"].astype(str).str.zfill(5)

    sel = sel.merge(
        county_land_lut_df,
        left_on="county_fips",
        right_on="county_fips",
        how="left",
    )

    terminal_cost = (
        sel.groupby("terminal")["land_usd2017_per_acre"].first().sort_index()
    )
    CAPITAL_COSTS = np.array(sel.land_usd2017_per_acre)

    WORKLOAD_PROFILE_PATH = (
        HOME_PATH + "/zap/development/load_profiles/example_inference_azure_conv.csv"
    )
    return (
        CAPITAL_COSTS,
        HOME_PATH,
        PYPSA_NETWORK_PATH,
        WORKLOAD_PROFILE_PATH,
        b,
        bus_to_terminal,
        buses,
        buses_to_index,
        counties,
        county_land_lut_df,
        county_url,
        gdf,
        index_to_bus,
        j,
        pn,
        pypsa_bus_names,
        sel,
        selected_node_fips,
        snapshot_data,
        snapshots,
        terminal_cost,
    )


@app.cell
def _(
    UPSAMPLE_FACTOR,
    load_pypsa_network,
    pn,
    snapshot_data,
    upsample_zap_devices,
):
    # Convert PyPSA to Zap
    pypsa_net, pypsa_devices = load_pypsa_network(
        pn, snapshot_data, power_unit=1.0e3, cost_unit=100.0
    )
    pypsa_devices = upsample_zap_devices(pypsa_devices, factor=UPSAMPLE_FACTOR, original_timesteps=24)
    return pypsa_devices, pypsa_net


@app.cell
def _(mo):
    mo.md("""## Helper Functions""")
    return


@app.cell
def _(deepcopy, np, zap):
    def create_planning_devices(pypsa_devices, planning_devices_params_dict):
        num_nodes = planning_devices_params_dict["num_nodes"]
        investment_node_cands = planning_devices_params_dict["investment_node_cands"]
        gen_scaling_factor = planning_devices_params_dict["gen_scaling_factor"]
        load_scaling_factor = planning_devices_params_dict["load_scaling_factor"]
        line_scaling_factor = planning_devices_params_dict["line_scaling_factor"]
        dc_nominal_capacity = planning_devices_params_dict["dc_nominal_capacity"]
        capital_costs = planning_devices_params_dict["capital_costs"]
        workload_profile = planning_devices_params_dict["workload_profile"]
        time_horizon = planning_devices_params_dict["time_horizon"]
        pypsa_net = planning_devices_params_dict["pypsa_net"]
        pypsa_devices = planning_devices_params_dict["pypsa_devices"]

        pypsa_devices_dc = deepcopy(pypsa_devices)

        # Scale load, gen, and line capacities
        pypsa_devices_dc[1].load *= load_scaling_factor
        pypsa_devices_dc[0].dynamic_capacity *= gen_scaling_factor
        # pypsa_devices_dc[3].nominal_capacity *= line_scaling_factor
        # pypsa_devices_dc[3].nominal_capacity[168] = 0.5
        # pypsa_devices_dc[3].nominal_capacity[176] = 0.5
        # pypsa_devices_dc[3].nominal_capacity[49] = 0.3

        # Select which nodes to build at
        dc_terminals = np.array(investment_node_cands[:num_nodes])
        n_dc = len(dc_terminals)
        dc_capital_costs = capital_costs[:n_dc]

        # Build nominal capacities for DC loads
        if np.isscalar(dc_nominal_capacity):
            nominal_capacity = np.full(n_dc, dc_nominal_capacity)
        else:
            nominal_capacity = dc_nominal_capacity

        dcloads = zap.DataCenterLoad(
            num_nodes=pypsa_net.num_nodes,
            terminal=dc_terminals,
            profiles=n_dc * [workload_profile],
            nominal_capacity=nominal_capacity,
            linear_cost=np.zeros(n_dc),
            settime_horizon=time_horizon,
            capital_cost=dc_capital_costs,
        )

        pypsa_devices_dc.append(dcloads)
        return pypsa_devices_dc
    return (create_planning_devices,)


@app.cell
def device_index():
    def device_index(devices, device_type):
        """Find index of device type in device list."""
        for i, d in enumerate(devices):
            if isinstance(d, device_type):
                return i
        return None
    return (device_index,)


@app.cell
def _(mo):
    mo.md("""## Baseline Planning (No Contingencies)""")
    return


@app.cell
def _(
    CAPITAL_COSTS,
    DataCenterLoad,
    GEN_SCALING_FACTOR,
    INVESTMENT_NODE_CANDS,
    LINE_SCALING_FACTOR,
    LOAD_SCALING_FACTOR,
    TIME_HORIZON,
    WORKLOAD_PROFILE_PATH,
    cp,
    create_planning_devices,
    device_index,
    np,
    pypsa_devices,
    pypsa_net,
    zap,
):
    # Configuration for baseline
    num_dc_nodes = 3
    total_dc_budget = 2.5  # GW

    planning_devices_params_dict_v2 = {
        "num_nodes": num_dc_nodes,
        "investment_node_cands": INVESTMENT_NODE_CANDS,
        "gen_scaling_factor": GEN_SCALING_FACTOR,
        "load_scaling_factor": LOAD_SCALING_FACTOR,
        "line_scaling_factor": LINE_SCALING_FACTOR,
        "dc_nominal_capacity": 1.0,
        "capital_costs": np.zeros_like(CAPITAL_COSTS),
        "workload_profile": WORKLOAD_PROFILE_PATH,
        "time_horizon": TIME_HORIZON,
        "pypsa_net": pypsa_net,
        "pypsa_devices": pypsa_devices,
    }
    baseline_devices = create_planning_devices(pypsa_devices, planning_devices_params_dict_v2)
    dc_device_idx_v2 = device_index(baseline_devices, DataCenterLoad)

    # Create dispatch layer (CVX-based for baseline)
    baseline_layer = zap.DispatchLayer(
        pypsa_net,
        baseline_devices,
        parameter_names={"dc_capacity": (dc_device_idx_v2, "nominal_capacity")},
        time_horizon=TIME_HORIZON,
        solver=cp.CLARABEL,
    )

    # Setup objectives
    # Multi-objective: dispatch cost + tail LMP pressure at DC sites (encourages natural spreading).
    baseline_cost_obj = zap.planning.DispatchCostObjective(pypsa_net, baseline_devices)
    baseline_price_obj = zap.planning.DCTailPriceObjective(
        baseline_devices,
        dc_device_idx=dc_device_idx_v2,
        lmp_metric="cvar",
        cvar_alpha=0.95,
        weight_by_capacity=True,
        aggregation_across_dcs="sum",
    )
    baseline_op_obj = baseline_cost_obj + 0.05 * baseline_price_obj

    base_inv_v2 = zap.planning.InvestmentObjective(baseline_devices, baseline_layer)
    baseline_inv_obj = zap.planning.RegularizedInvestmentObjective(
        base_inv_v2,
        [zap.planning.CapacityL2Regularizer("dc_capacity", weight=1e-3)],
    )

    # Setup bounds
    uniform_cap = total_dc_budget / num_dc_nodes
    baseline_lower_bounds = {"dc_capacity": np.full(num_dc_nodes, 0.05)}
    baseline_upper_bounds = {"dc_capacity": np.full(num_dc_nodes, min(2.5 * uniform_cap, 1.0))}

    # Create planning problem
    baseline_problem = zap.planning.PlanningProblem(
        operation_objective=baseline_op_obj,
        investment_objective=baseline_inv_obj,
        layer=baseline_layer,
        lower_bounds=baseline_lower_bounds,
        upper_bounds=baseline_upper_bounds,
    )

    # Add budget constraint
    baseline_problem.extra_projections = {
        "dc_capacity": zap.planning.SimplexBudgetProjection(budget=total_dc_budget, strict=True)
    }
    return (
        base_inv_v2,
        baseline_cost_obj,
        baseline_devices,
        baseline_inv_obj,
        baseline_layer,
        baseline_lower_bounds,
        baseline_op_obj,
        baseline_price_obj,
        baseline_problem,
        baseline_upper_bounds,
        dc_device_idx_v2,
        num_dc_nodes,
        planning_devices_params_dict_v2,
        total_dc_budget,
        uniform_cap,
    )


@app.cell
def _(baseline_problem, np, num_dc_nodes, total_dc_budget):
    print("Solving baseline planning (no contingencies)...")
    baseline_init = {"dc_capacity": np.full(num_dc_nodes, total_dc_budget / num_dc_nodes)}
    baseline_state, baseline_history = baseline_problem.solve(
        num_iterations=5,
        initial_state=baseline_init,
    )

    baseline_capacities = baseline_state["dc_capacity"]
    print(f"Baseline capacities: {baseline_capacities}")
    print(f"Total allocation: {baseline_capacities.sum():.3f} GW")
    return baseline_capacities, baseline_history, baseline_init, baseline_state


@app.cell
def _(
    CAPITAL_COSTS,
    GEN_SCALING_FACTOR,
    HOME_PATH,
    INVESTMENT_NODE_CANDS,
    LINE_SCALING_FACTOR,
    LOAD_SCALING_FACTOR,
    TIME_HORIZON,
    cp,
    create_planning_devices,
    pypsa_devices,
    pypsa_net,
):
    _lin_dev_params = {
                "num_nodes": 3,
                "investment_node_cands": INVESTMENT_NODE_CANDS,
                "gen_scaling_factor": GEN_SCALING_FACTOR,
                "load_scaling_factor": LOAD_SCALING_FACTOR,
                "line_scaling_factor": LINE_SCALING_FACTOR,
                "dc_nominal_capacity": 2.5,
                "capital_costs": 0 * CAPITAL_COSTS,
                "workload_profile": HOME_PATH
                + "/zap/development/load_profiles/example_inference_azure_conv.csv",
                "pypsa_net": pypsa_net,
                "pypsa_devices": pypsa_devices,
        "time_horizon": TIME_HORIZON
    }
    _lin_devices_eval = create_planning_devices(pypsa_devices, _lin_dev_params)
    lin_outcome_eval = pypsa_net.dispatch(
        devices=_lin_devices_eval[:-1],
        time_horizon=96,
        solver=cp.CLARABEL,
        add_ground=False,
    )
    return (lin_outcome_eval,)


@app.cell
def _(lin_outcome_eval, np):
    mu_lo = lin_outcome_eval.local_inequality_duals[3][0]
    mu_hi = lin_outcome_eval.local_inequality_duals[3][1]
    mu_sum = mu_lo + mu_hi
    mu_sum_avg = mu_sum.mean(axis=1) # [251,]
    k_lines = 10  # choose how many spikes you want
    cong_lines = np.argsort(mu_sum_avg)[-k_lines:][::-1]   # indices of k largest
    cong_lines
    return cong_lines, k_lines, mu_hi, mu_lo, mu_sum, mu_sum_avg


@app.cell
def _(mo):
    mo.md("""## SCOPF Planning (With N-1 Contingencies)""")
    return


@app.cell
def _(
    ADMMLayer,
    ADMMSolver,
    CAPITAL_COSTS,
    CRITICAL_LINES,
    DataCenterLoad,
    GEN_SCALING_FACTOR,
    INVESTMENT_NODE_CANDS,
    LINE_SCALING_FACTOR,
    LOAD_SCALING_FACTOR,
    NUM_CONTINGENCIES,
    TIME_HORIZON,
    WORKLOAD_PROFILE_PATH,
    cong_lines,
    create_planning_devices,
    device_index,
    np,
    num_dc_nodes,
    pypsa_devices,
    pypsa_net,
    torch,
    zap,
):
    # Create devices for SCOPF (torchified for ADMM)
    planning_devices_params_dict = {
        "num_nodes": 3,
        "investment_node_cands": INVESTMENT_NODE_CANDS,
        "gen_scaling_factor": GEN_SCALING_FACTOR,
        "load_scaling_factor": LOAD_SCALING_FACTOR,
        "line_scaling_factor": LINE_SCALING_FACTOR,
        "dc_nominal_capacity": 2.5,
        "capital_costs": np.zeros_like(CAPITAL_COSTS),
        "workload_profile": WORKLOAD_PROFILE_PATH,
        "time_horizon": TIME_HORIZON,
        "pypsa_net": pypsa_net,
        "pypsa_devices": pypsa_devices,
    }
    scopf_devices_np = create_planning_devices(pypsa_devices, planning_devices_params_dict)

    # Torchify for ADMM
    scopf_devices = [d.torchify(machine="cpu", dtype=torch.float32) for d in scopf_devices_np]

    # Setup contingency parameters
    line_device_idx_scopf = device_index(scopf_devices, zap.ACLine)
    dc_device_idx_scopf = device_index(scopf_devices, DataCenterLoad)
    num_lines = scopf_devices[line_device_idx_scopf].num_devices

    # Pick contingency lines:
    #   1) explicit override via CRITICAL_LINES
    #   2) cong_lines if provided (e.g. top-k congested lines from a baseline run)
    #   3) fallback: first NUM_CONTINGENCIES lines
    if CRITICAL_LINES is not None:
        critical_lines = [int(i) for i in np.asarray(CRITICAL_LINES).ravel().tolist()]
    elif cong_lines is not None:
        critical_lines = [int(i) for i in np.asarray(cong_lines).ravel().tolist()]
    else:
        critical_lines = list(range(min(int(NUM_CONTINGENCIES), int(num_lines))))

    num_contingencies = len(critical_lines)

    print(f"Total lines: {num_lines}")
    print(f"Contingencies: {num_contingencies}")

    # Create contingency mask for chosen lines
    contingency_mask = zap.planning.create_critical_line_contingency_mask(
        critical_lines,
        num_lines,
        device="cpu",
        dtype=torch.float32
    )

    # Create ADMM solver
    scopf_solver = ADMMSolver(
        dtype=torch.float32,
        num_iterations=1000,
        minimum_iterations=250,
        atol=3.0e-3,
        adaptive_rho=True,
        rho_power=1.0,
        rho_angle=1.0,
        resid_norm=2,
    )

    # Create SCOPF layer
    scopf_layer = ADMMLayer(
        network=pypsa_net,
        devices=scopf_devices,
        parameter_names={"dc_capacity": (dc_device_idx_scopf, "nominal_capacity")},
        time_horizon=TIME_HORIZON,
        solver=scopf_solver,
        num_contingencies=num_contingencies,
        contingency_device=line_device_idx_scopf,
        contingency_mask=contingency_mask,
    )

    # Setup SCOPF objectives
    # Multi-objective: expected dispatch cost + tail contingency overload + tail LMP pressure at DC sites.
    scopf_cost_obj = zap.planning.SCOPFDispatchCostObjective(
        pypsa_net,
        scopf_devices,
        contingency_device_idx=line_device_idx_scopf,
        aggregation="cvar",
        cvar_alpha=0.90,
    )

    # Penalize worst-case contingencies (tail over scenarios) rather than the mean.
    scopf_overload_obj = 0.1 * zap.planning.SCOPFLineOverloadObjective(
        scopf_devices,
        line_device_idx=line_device_idx_scopf,
        thr=0.95,
        aggregation='cvar',
        cvar_alpha=0.90,
    )

    scopf_price_obj = 0.05 * zap.planning.DCTailPriceObjective(
        scopf_devices,
        dc_device_idx=dc_device_idx_scopf,
        lmp_metric="cvar",
        cvar_alpha=0.95,
        weight_by_capacity=True,
        aggregation_across_dcs="sum",
    )

    scopf_op_obj = scopf_cost_obj + scopf_overload_obj + scopf_price_obj

    base_inv = zap.planning.InvestmentObjective(scopf_devices, scopf_layer)
    scopf_inv_obj = zap.planning.RegularizedInvestmentObjective(
        base_inv,
        [zap.planning.CapacityL2Regularizer("dc_capacity", weight=1e-3)],
    )

    # Setup bounds (same as baseline)
    scopf_lower_bounds = {"dc_capacity": np.full(num_dc_nodes, 0.0)}
    scopf_upper_bounds = {"dc_capacity": np.full(num_dc_nodes, 2.5)}

    # Create SCOPF planning problem
    scopf_problem = zap.planning.PlanningProblem(
        operation_objective=scopf_op_obj,
        investment_objective=scopf_inv_obj,
        layer=scopf_layer,
        lower_bounds=scopf_lower_bounds,
        upper_bounds=scopf_upper_bounds,
    )

    # Add budget constraint
    scopf_problem.extra_projections = {
        "dc_capacity": zap.planning.SimplexBudgetProjection(budget=2.5, strict=True)
    }
    return (
        base_inv,
        contingency_mask,
        critical_lines,
        dc_device_idx_scopf,
        line_device_idx_scopf,
        num_contingencies,
        num_lines,
        planning_devices_params_dict,
        scopf_cost_obj,
        scopf_devices,
        scopf_devices_np,
        scopf_inv_obj,
        scopf_layer,
        scopf_lower_bounds,
        scopf_op_obj,
        scopf_overload_obj,
        scopf_price_obj,
        scopf_problem,
        scopf_solver,
        scopf_upper_bounds,
    )


@app.cell
def _(np, num_dc_nodes, scopf_problem, total_dc_budget):
    # Solve SCOPF planning
    print("Solving SCOPF planning (with N-1 contingencies)...")
    scopf_init = {"dc_capacity": np.full(num_dc_nodes, total_dc_budget / num_dc_nodes)}

    scopf_state, scopf_history = scopf_problem.solve(
        num_iterations=75,
        initial_state=scopf_init
    )

    scopf_capacities = scopf_state["dc_capacity"].detach().cpu().numpy()
    print(f"SCOPF capacities: {scopf_capacities}")
    print(f"Total allocation: {scopf_capacities.sum():.3f} GW")
    return scopf_capacities, scopf_history, scopf_init, scopf_state


@app.cell
def _(mo):
    mo.md("""## Compare Results""")
    return


@app.cell
def _(
    CAPITAL_COSTS,
    GEN_SCALING_FACTOR,
    HOME_PATH,
    INVESTMENT_NODE_CANDS,
    LINE_SCALING_FACTOR,
    LOAD_SCALING_FACTOR,
    TIME_HORIZON,
    cp,
    create_planning_devices,
    pypsa_devices,
    pypsa_net,
    scopf_capacities,
):
    lin_dev_params_scopf = {
                "num_nodes": 3,
                "investment_node_cands": INVESTMENT_NODE_CANDS,
                "gen_scaling_factor": GEN_SCALING_FACTOR,
                "load_scaling_factor": LOAD_SCALING_FACTOR,
                "line_scaling_factor": LINE_SCALING_FACTOR,
                "dc_nominal_capacity": scopf_capacities,
                "capital_costs": 0 * CAPITAL_COSTS,
                "workload_profile": HOME_PATH
                + "/zap/development/load_profiles/example_inference_azure_conv.csv",
                "pypsa_net": pypsa_net,
                "pypsa_devices": pypsa_devices,
        "time_horizon": TIME_HORIZON
    }
    lin_devices_scopf = create_planning_devices(pypsa_devices, lin_dev_params_scopf)
    lin_outcome_scopf = pypsa_net.dispatch(
        devices=lin_devices_scopf,
        time_horizon=96,
        solver=cp.CLARABEL,
        add_ground=False,
    )
    return lin_dev_params_scopf, lin_devices_scopf, lin_outcome_scopf


@app.cell
def _(deepcopy, lin_outcome_eval, lin_outcome_scopf, np, pd):
    def cvar_upper_tail(values, alpha: float):
        """
        Upper-tail CVaR for "higher is worse" scalars.
        values: 1D array-like
        alpha: in (0, 1)
        """
        x = np.asarray(values, dtype=float).ravel()
        x = x[np.isfinite(x)]
        if x.size == 0:
            return np.nan
        x = np.sort(x)  # ascending
        k0 = int(np.floor(float(alpha) * x.size))
        k0 = min(max(k0, 0), x.size - 1)
        return float(x[k0:].mean())

    def node_price_summaries(prices, topk=5, q=(0.95, 0.99)):
        """
        prices: [N,T]
        Returns dict of [N,] summaries over time for each node.
        """
        prices = np.asarray(prices)

        out = {}
        for qq in q:
            out[f"p{int(qq * 100)}"] = np.quantile(prices, qq, axis=1)
        out["mean"] = prices.mean(axis=1)
        k = int(topk)
        out[f"mean_top{k}"] = np.sort(prices, axis=1)[:, -k:].mean(axis=1)
        out["max"] = prices.max(axis=1)
        return out

    def get_congestion_metric(dispatch_outcome, *, line_device_idx: int = 3):
        mu_lo = dispatch_outcome.local_inequality_duals[line_device_idx][0]  # (L,T)
        mu_hi = dispatch_outcome.local_inequality_duals[line_device_idx][1]  # (L,T)
        mu = np.max(mu_lo + mu_hi, axis=1)  # max over time -> (L,)
        return float(np.mean(mu))

    def safe_dispatch(pypsa_net, devices, *, time_horizon, solver, add_ground: bool):
        try:
            out = pypsa_net.dispatch(
                devices=devices,
                time_horizon=time_horizon,
                solver=solver,
                add_ground=add_ground,
            )
            return {"ok": True, "outcome": out, "status": str(out.problem.status)}
        except Exception as e:
            # PowerNetwork.dispatch asserts OPTIMAL / OPTIMAL_INACCURATE. With add_ground=False,
            # infeasibility is expected for some N-1 outages. We treat that as a failed scenario.
            return {"ok": False, "outcome": None, "status": f"{type(e).__name__}: {e}"}

    def _zero_out_line(devs, line_device_idx: int, line_idx: int):
        dev = devs[line_device_idx]
        cap = np.asarray(dev.nominal_capacity)
        cap = cap.copy()
        if cap.ndim == 1:
            cap[line_idx] = 0.0
        else:
            cap[line_idx, ...] = 0.0
        dev.nominal_capacity = cap
        return devs

    def evaluate_n1_cvx(
        *,
        pypsa_net,
        base_devices,
        critical_lines,
        dc_terminals,
        time_horizon: int,
        solver,
        add_ground: bool,
        cvar_alpha: float = 0.90,
        method: str = "method",
        line_device_idx: int = 3,
    ):
        rows = []

        def _row(scenario_name: str, out):
            if not out["ok"]:
                return {
                    "method": method,
                    "scenario": scenario_name,
                    "feasible": False,
                    "status": out["status"],
                    "dispatch_cost": np.inf,
                    "congestion_metric": np.inf,
                    "dc_mean_max_lmp": np.inf,
                }
            outcome = out["outcome"]
            dc_prices = outcome.prices[np.asarray(dc_terminals, dtype=int), :]
            dc_stats = node_price_summaries(dc_prices, topk=5)
            return {
                "method": method,
                "scenario": scenario_name,
                "feasible": True,
                "status": out["status"],
                "dispatch_cost": float(outcome.problem.value) * 100.0,
                "congestion_metric": get_congestion_metric(outcome, line_device_idx=line_device_idx),
                "dc_mean_max_lmp": float(np.mean(dc_stats["max"])) * 100.0 * 4.0,
            }

        base_out = safe_dispatch(
            pypsa_net,
            base_devices,
            time_horizon=time_horizon,
            solver=solver,
            add_ground=add_ground,
        )
        rows.append(_row("base", base_out))

        for line_idx in list(critical_lines):
            devs = deepcopy(base_devices)
            devs = _zero_out_line(devs, line_device_idx=line_device_idx, line_idx=int(line_idx))
            out = safe_dispatch(
                pypsa_net,
                devs,
                time_horizon=time_horizon,
                solver=solver,
                add_ground=add_ground,
            )
            rows.append(_row(f"outage_line_{int(line_idx)}", out))

        df = pd.DataFrame(rows)

        base = df[df["scenario"] == "base"].iloc[0].to_dict()
        cont = df[df["scenario"] != "base"].copy()
        infeas = int((~cont["feasible"]).sum())
        if infeas > 0:
            agg = {
                "method": method,
                "contingency_count": int(cont.shape[0]),
                "infeasible_count": infeas,
                "cvar_alpha": float(cvar_alpha),
                "base_dispatch_cost": float(base["dispatch_cost"]),
                "base_congestion_metric": float(base["congestion_metric"]),
                "base_dc_mean_max_lmp": float(base["dc_mean_max_lmp"]),
                "mean_dispatch_cost": np.inf,
                "mean_congestion_metric": np.inf,
                "mean_dc_mean_max_lmp": np.inf,
                "max_dispatch_cost": np.inf,
                "max_congestion_metric": np.inf,
                "max_dc_mean_max_lmp": np.inf,
                "cvar_dispatch_cost": np.inf,
                "cvar_congestion_metric": np.inf,
                "cvar_dc_mean_max_lmp": np.inf,
            }
        else:
            agg = {
                "method": method,
                "contingency_count": int(cont.shape[0]),
                "infeasible_count": 0,
                "cvar_alpha": float(cvar_alpha),
                "base_dispatch_cost": float(base["dispatch_cost"]),
                "base_congestion_metric": float(base["congestion_metric"]),
                "base_dc_mean_max_lmp": float(base["dc_mean_max_lmp"]),
                "mean_dispatch_cost": float(cont["dispatch_cost"].mean()),
                "mean_congestion_metric": float(cont["congestion_metric"].mean()),
                "mean_dc_mean_max_lmp": float(cont["dc_mean_max_lmp"].mean()),
                "max_dispatch_cost": float(cont["dispatch_cost"].max()),
                "max_congestion_metric": float(cont["congestion_metric"].max()),
                "max_dc_mean_max_lmp": float(cont["dc_mean_max_lmp"].max()),
                "cvar_dispatch_cost": cvar_upper_tail(cont["dispatch_cost"], alpha=cvar_alpha),
                "cvar_congestion_metric": cvar_upper_tail(cont["congestion_metric"], alpha=cvar_alpha),
                "cvar_dc_mean_max_lmp": cvar_upper_tail(cont["dc_mean_max_lmp"], alpha=cvar_alpha),
            }

        return df, pd.DataFrame([agg])

    # Keep existing quick base-case metrics (for sanity)
        met_base = {
            "dispatch_cost": float(lin_outcome_eval.problem.value) * 100.0,
            "congestion_metric": get_congestion_metric(lin_outcome_eval, line_device_idx=3),
            "mean_max_lmp_system": float(np.mean(node_price_summaries(lin_outcome_eval.prices)["max"]))
            * 100.0
            * 4.0,
        }
        met_scopf = {
            "dispatch_cost": float(lin_outcome_scopf.problem.value) * 100.0,
            "congestion_metric": get_congestion_metric(lin_outcome_scopf, line_device_idx=3),
            "mean_max_lmp_system": float(np.mean(node_price_summaries(lin_outcome_scopf.prices)["max"]))
            * 100.0
            * 4.0,
        }
    return (
        cvar_upper_tail,
        evaluate_n1_cvx,
        get_congestion_metric,
        node_price_summaries,
        safe_dispatch,
    )


@app.cell
def _(
    CAPITAL_COSTS,
    GEN_SCALING_FACTOR,
    INVESTMENT_NODE_CANDS,
    LINE_SCALING_FACTOR,
    LOAD_SCALING_FACTOR,
    TIME_HORIZON,
    WORKLOAD_PROFILE_PATH,
    baseline_capacities,
    cp,
    create_planning_devices,
    critical_lines,
    device_index,
    evaluate_n1_cvx,
    np,
    pd,
    pypsa_devices,
    pypsa_net,
    scopf_capacities,
    zap,
):
    # N-1 evaluation settings
    CVAR_ALPHA_EVAL = 0.90
    ADD_GROUND_EVAL = False

    # Single-node baseline: inject the full budget at terminal 20 (matches pushing_capacity.py)
    single_node_terminals = [20]
    single_node_budget = float(np.sum(scopf_capacities))

    dev_params_single = {
        "num_nodes": 1,
        "investment_node_cands": single_node_terminals,
        "gen_scaling_factor": GEN_SCALING_FACTOR,
        "load_scaling_factor": LOAD_SCALING_FACTOR,
        "line_scaling_factor": LINE_SCALING_FACTOR,
        "dc_nominal_capacity": single_node_budget,
        "capital_costs": np.zeros_like(CAPITAL_COSTS),
        "workload_profile": WORKLOAD_PROFILE_PATH,
        "time_horizon": TIME_HORIZON,
        "pypsa_net": pypsa_net,
        "pypsa_devices": pypsa_devices,
    }
    devs_single = create_planning_devices(pypsa_devices, dev_params_single)

    # Baseline planned (no contingencies) capacities
    dev_params_baseline = {
        "num_nodes": len(baseline_capacities),
        "investment_node_cands": INVESTMENT_NODE_CANDS,
        "gen_scaling_factor": GEN_SCALING_FACTOR,
        "load_scaling_factor": LOAD_SCALING_FACTOR,
        "line_scaling_factor": LINE_SCALING_FACTOR,
        "dc_nominal_capacity": baseline_capacities,
        "capital_costs": np.zeros_like(CAPITAL_COSTS),
        "workload_profile": WORKLOAD_PROFILE_PATH,
        "time_horizon": TIME_HORIZON,
        "pypsa_net": pypsa_net,
        "pypsa_devices": pypsa_devices,
    }
    devs_baseline = create_planning_devices(pypsa_devices, dev_params_baseline)

    # SCOPF planned capacities
    dev_params_scopf = {
        "num_nodes": len(scopf_capacities),
        "investment_node_cands": INVESTMENT_NODE_CANDS,
        "gen_scaling_factor": GEN_SCALING_FACTOR,
        "load_scaling_factor": LOAD_SCALING_FACTOR,
        "line_scaling_factor": LINE_SCALING_FACTOR,
        "dc_nominal_capacity": scopf_capacities,
        "capital_costs": np.zeros_like(CAPITAL_COSTS),
        "workload_profile": WORKLOAD_PROFILE_PATH,
        "time_horizon": TIME_HORIZON,
        "pypsa_net": pypsa_net,
        "pypsa_devices": pypsa_devices,
    }
    devs_scopf = create_planning_devices(pypsa_devices, dev_params_scopf)

    line_device_idx = device_index(devs_scopf, zap.ACLine)
    if line_device_idx is None:
        raise ValueError("Could not find ACLine device index for N-1 evaluation.")

    df_single_scen, df_single_agg = evaluate_n1_cvx(
        pypsa_net=pypsa_net,
        base_devices=devs_single,
        critical_lines=critical_lines,
        dc_terminals=single_node_terminals,
        time_horizon=TIME_HORIZON,
        solver=cp.CLARABEL,
        add_ground=ADD_GROUND_EVAL,
        cvar_alpha=CVAR_ALPHA_EVAL,
        method="single_node",
        line_device_idx=line_device_idx,
    )

    df_base_scen, df_base_agg = evaluate_n1_cvx(
        pypsa_net=pypsa_net,
        base_devices=devs_baseline,
        critical_lines=critical_lines,
        dc_terminals=INVESTMENT_NODE_CANDS[: len(baseline_capacities)],
        time_horizon=TIME_HORIZON,
        solver=cp.CLARABEL,
        add_ground=ADD_GROUND_EVAL,
        cvar_alpha=CVAR_ALPHA_EVAL,
        method="baseline_planning",
        line_device_idx=line_device_idx,
    )

    df_scopf_scen, df_scopf_agg = evaluate_n1_cvx(
        pypsa_net=pypsa_net,
        base_devices=devs_scopf,
        critical_lines=critical_lines,
        dc_terminals=INVESTMENT_NODE_CANDS[: len(scopf_capacities)],
        time_horizon=TIME_HORIZON,
        solver=cp.CLARABEL,
        add_ground=ADD_GROUND_EVAL,
        cvar_alpha=CVAR_ALPHA_EVAL,
        method="scopf_planning",
        line_device_idx=line_device_idx,
    )

    df_agg = pd.concat([df_single_agg, df_base_agg, df_scopf_agg], ignore_index=True)
    return (
        ADD_GROUND_EVAL,
        CVAR_ALPHA_EVAL,
        dev_params_baseline,
        dev_params_scopf,
        dev_params_single,
        devs_baseline,
        devs_scopf,
        devs_single,
        df_agg,
        df_base_agg,
        df_base_scen,
        df_scopf_agg,
        df_scopf_scen,
        df_single_agg,
        df_single_scen,
        line_device_idx,
        single_node_budget,
        single_node_terminals,
    )


@app.cell
def _(baseline_capacities, pd, scopf_capacities):
    # Create comparison dataframe
    comparison_df = pd.DataFrame({
        "Node": range(len(baseline_capacities)),
        "Baseline (GW)": baseline_capacities,
        "SCOPF (GW)": scopf_capacities,
        "Difference (GW)": scopf_capacities - baseline_capacities,
        "Difference (%)": 100 * (scopf_capacities - baseline_capacities) / (baseline_capacities + 1e-6)
    })

    comparison_df
    return (comparison_df,)


@app.cell
def _(df_scopf_scen):
    print(df_scopf_scen.to_string())
    return


if __name__ == "__main__":
    app.run()

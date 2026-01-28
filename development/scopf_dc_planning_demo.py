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
    from zap.planning.operation_objectives import SCOPFLMPObjective
    import scipy.sparse as sp


    sns.set_theme()
    return (
        ACLine,
        ADMMLayer,
        ADMMSolver,
        DataCenterLoad,
        SCOPFLMPObjective,
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
        sp,
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
def _(os, pypsa):
    HOME_PATH = os.environ.get("HOME")
    WORKLOAD_PROFILE_PATH = '/Users/akshaysreekumar/Documents/Stanford/S3L/zap/development/load_profiles/example_inference_azure_conv.csv'
    PYPSA_NETW0RK_PATH = (
        HOME_PATH + "/zap_data/pypsa-networks/western_small/network_2023.nc"
    )
    pn = pypsa.Network(PYPSA_NETW0RK_PATH)
    snapshots = pn.generators_t.p_max_pu.index
    snapshot_data = snapshots[5616:5640]  # 8/23/21 # hourly
    return (
        HOME_PATH,
        PYPSA_NETW0RK_PATH,
        WORKLOAD_PROFILE_PATH,
        pn,
        snapshot_data,
        snapshots,
    )


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
def _(INVESTMENT_NODE_CANDS, gpd, parse_buses, pd, pn):
    buses, buses_to_index = parse_buses(pn) # buses_to_index is dict of "pyspa_bus_name": "zap_terminal"
    index_to_bus = {idx: name for name, idx in buses_to_index.items()}
    pypsa_bus_names = [index_to_bus[i] for i in INVESTMENT_NODE_CANDS]

    b = pn.buses.copy()
    gdf = gpd.GeoDataFrame(
        b, geometry=gpd.points_from_xy(b["x"], b["y"]), crs="EPSG:4326"
    )

    county_url = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_county_500k.zip"
    counties = gpd.read_file(county_url)[["STATEFP","COUNTYFP","GEOID","NAME","STATE_NAME","geometry"]]

    j = gpd.sjoin(gdf, counties.to_crs("EPSG:4326"), how="left", predicate="within")

    pn.buses["county_fips"] = j["GEOID"]  # 5-digit FIPS
    pn.buses["county_name"] = j["NAME"]
    pn.buses["state_fips"]  = j["STATEFP"]
    pn.buses["state_name"]  = j["STATE_NAME"]

    selected_node_fips = pn.buses.loc[pypsa_bus_names, "county_fips"]
    county_land_lut_df = pd.read_csv("development/county_land_lut.csv")
    return (
        b,
        buses,
        buses_to_index,
        counties,
        county_land_lut_df,
        county_url,
        gdf,
        index_to_bus,
        j,
        pypsa_bus_names,
        selected_node_fips,
    )


@app.cell
def _(county_land_lut_df, index_to_bus, selected_node_fips):
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
        sel.groupby("terminal")["land_usd2017_per_acre"]
          .first()   # or .mean(), depending on what you want
          .sort_index()
    )

    terminal_cost_sorted = terminal_cost.sort_values(ascending=True)
    return bus_to_terminal, sel, terminal_cost, terminal_cost_sorted


@app.cell
def _(np, sel):
    CAPITAL_COSTS = np.array(sel.land_usd2017_per_acre)
    return (CAPITAL_COSTS,)


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
def _(np):
    # mu_lo = lin_outcome_eval.local_inequality_duals[3][0]
    # mu_hi = lin_outcome_eval.local_inequality_duals[3][1]
    # mu_sum = mu_lo + mu_hi
    # mu_sum_avg = mu_sum.mean(axis=1) # [251,]
    # k_lines = 10  # choose how many spikes you want
    # cong_lines = np.argsort(mu_sum_avg)[-k_lines:][::-1]   # indices of k largest
    cong_lines = np.array([168, 176, 88, 170, 49, 180, 149, 80, 73, 67])
    return (cong_lines,)


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
    SCOPFLMPObjective,
    TIME_HORIZON,
    WORKLOAD_PROFILE_PATH,
    cong_lines,
    create_planning_devices,
    device_index,
    np,
    pypsa_devices,
    pypsa_net,
    sp,
    torch,
    zap,
):
    num_dc_nodes = 3
    total_dc_budget = 2.5


    # Cell 12: SCOPF Planning (With N-1 Contingencies)
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

    line_device_idx_scopf = device_index(scopf_devices, zap.ACLine)
    dc_device_idx_scopf = device_index(scopf_devices, DataCenterLoad)
    num_lines = scopf_devices[line_device_idx_scopf].num_devices

    # Pick contingency lines
    if CRITICAL_LINES is not None:
        critical_lines = [int(i) for i in np.asarray(CRITICAL_LINES).ravel().tolist()]
    elif cong_lines is not None:
        critical_lines = [int(i) for i in np.asarray(cong_lines).ravel().tolist()]
    else:
        critical_lines = list(range(min(int(NUM_CONTINGENCIES), int(num_lines))))

    num_contingencies = len(critical_lines)

    print(f"Total lines: {num_lines}")
    print(f"Contingencies: {num_contingencies}")

    # Create contingency mask
    contingency_mask = sp.lil_matrix(
        (num_contingencies, scopf_devices[line_device_idx_scopf].num_devices)
    )

    for idx, c in enumerate(cong_lines):
        contingency_mask[idx, c] = 1.0

    contingency_mask = contingency_mask.tocsr()
    torch_mask = torch.tensor(contingency_mask.todense(), device="cpu", dtype=torch.float32)
    torch_mask = torch.vstack(
        [
            torch.zeros(torch_mask.shape[1], device="cpu", dtype=torch.float32),
            torch_mask,
        ]
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
        contingency_mask=torch_mask,
    )

    scopf_lmp_obj = SCOPFLMPObjective(pypsa_net, scopf_devices)

    scopf_op_obj = scopf_lmp_obj

    base_inv = zap.planning.InvestmentObjective(scopf_devices, scopf_layer)

    scopf_lower_bounds = {"dc_capacity": np.full(num_dc_nodes, 0.0)}
    scopf_upper_bounds = {"dc_capacity": np.full(num_dc_nodes, 2.5)}

    scopf_problem = zap.planning.PlanningProblem(
        operation_objective=scopf_op_obj,
        investment_objective=base_inv,
        layer=scopf_layer,
        lower_bounds=scopf_lower_bounds,
        upper_bounds=scopf_upper_bounds,
    )

    scopf_problem.extra_projections = {
        "dc_capacity": zap.planning.SimplexBudgetProjection(budget=2.5, strict=True)
    }
    return (
        base_inv,
        c,
        contingency_mask,
        critical_lines,
        dc_device_idx_scopf,
        idx,
        line_device_idx_scopf,
        num_contingencies,
        num_dc_nodes,
        num_lines,
        planning_devices_params_dict,
        scopf_devices,
        scopf_devices_np,
        scopf_layer,
        scopf_lmp_obj,
        scopf_lower_bounds,
        scopf_op_obj,
        scopf_problem,
        scopf_solver,
        scopf_upper_bounds,
        torch_mask,
        total_dc_budget,
    )


@app.cell
def _(cp, pypsa_net, scopf_devices):
    base_outcome = pypsa_net.dispatch(
        devices=scopf_devices[:-1],
        time_horizon=96,
        solver=cp.CLARABEL,
        add_ground=False,
    )
    return (base_outcome,)


@app.cell
def _(num_contingencies, pypsa_net, scopf_devices, scopf_solver, torch_mask):
    solution_admm, history_admm = scopf_solver.solve(
        pypsa_net,
        scopf_devices,
        time_horizon=96,
        num_contingencies=num_contingencies,
        contingency_device=3,
        contingency_mask=torch_mask,
    )
    return history_admm, solution_admm


@app.cell
def _(mo):
    mo.md("""## SCOPF Planning (With N-1 Contingencies)""")
    return


@app.cell
def _(num_dc_nodes, scopf_problem, torch, total_dc_budget):
    # Solve SCOPF planning
    print("Solving SCOPF planning (with N-1 contingencies)...")
    scopf_init = {"dc_capacity": torch.full((num_dc_nodes,), total_dc_budget / num_dc_nodes)}

    scopf_state, scopf_history = scopf_problem.solve(
        num_iterations=75,
        initial_state=None
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
    INVESTMENT_NODE_CANDS,
    LINE_SCALING_FACTOR,
    LOAD_SCALING_FACTOR,
    TIME_HORIZON,
    WORKLOAD_PROFILE_PATH,
    create_planning_devices,
    num_contingencies,
    pypsa_devices,
    pypsa_net,
    scopf_capacities,
    scopf_devices,
    scopf_solver,
    torch_mask,
):
    ## Simulate Contingency Solve with Planned Caps

    planned_devices_params_dict = {
                "num_nodes": 3,
                "investment_node_cands": INVESTMENT_NODE_CANDS,
                "gen_scaling_factor": GEN_SCALING_FACTOR,
                "load_scaling_factor": LOAD_SCALING_FACTOR,
                "line_scaling_factor": LINE_SCALING_FACTOR,
                "dc_nominal_capacity": scopf_capacities,
                "capital_costs": 0 * CAPITAL_COSTS,
                "workload_profile": WORKLOAD_PROFILE_PATH,
                "pypsa_net": pypsa_net,
                "pypsa_devices": pypsa_devices,
        "time_horizon": TIME_HORIZON
    }
    planned_devices = create_planning_devices(pypsa_devices, planned_devices_params_dict)

    planned_solution_admm, planned_history_admm = scopf_solver.solve(
        pypsa_net,
        scopf_devices,
        time_horizon=96,
        num_contingencies=num_contingencies,
        contingency_device=3,
        contingency_mask=torch_mask,
    )

    planned_outcome = planned_solution_admm.as_outcome()
    return (
        planned_devices,
        planned_devices_params_dict,
        planned_history_admm,
        planned_outcome,
        planned_solution_admm,
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
    create_planning_devices,
    np,
    num_contingencies,
    pypsa_devices,
    pypsa_net,
    scopf_solver,
    torch,
    torch_mask,
):
    ## Run the single node injection across all the contingencies

    sn_devices_params_dict = {
                "num_nodes": 3,
                "investment_node_cands": INVESTMENT_NODE_CANDS,
                "gen_scaling_factor": GEN_SCALING_FACTOR,
                "load_scaling_factor": LOAD_SCALING_FACTOR,
                "line_scaling_factor": LINE_SCALING_FACTOR,
                "dc_nominal_capacity": np.array([0, 0, 2.5]),
                "capital_costs": 0 * CAPITAL_COSTS,
                "workload_profile": WORKLOAD_PROFILE_PATH,
                "pypsa_net": pypsa_net,
                "pypsa_devices": pypsa_devices,
        "time_horizon": TIME_HORIZON
    }
    sn_devices = create_planning_devices(pypsa_devices, sn_devices_params_dict)
    sn_devices_admm = [d.torchify(machine="cpu", dtype=torch.float32) for d in sn_devices]

    sn_solution_admm, sn_history_admm = scopf_solver.solve(
        pypsa_net,
        sn_devices_admm,
        time_horizon=96,
        num_contingencies=num_contingencies,
        contingency_device=3,
        contingency_mask=torch_mask,
    )

    sn_outcome = sn_solution_admm.as_outcome()

    return (
        sn_devices,
        sn_devices_admm,
        sn_devices_params_dict,
        sn_history_admm,
        sn_outcome,
        sn_solution_admm,
    )


@app.cell
def _(compute_metrics, np, sn_outcome):
    sn_metrics = compute_metrics(sn_outcome, np.array([0, 0, 2.5]))
    return (sn_metrics,)


@app.cell
def _(sn_metrics):
    sn_metrics
    return


@app.cell
def _(scopf_problem):
    scopf_problem.get_op_cost()
    return


@app.cell
def _(CAPITAL_COSTS, compute_metrics, planned_outcome, scopf_capacities):
    scopf_planned_metrics = compute_metrics(planned_outcome, scopf_capacities, capital_costs=CAPITAL_COSTS)
    return (scopf_planned_metrics,)


@app.cell
def _(scopf_planned_metrics):
    scopf_planned_metrics["mean_max_lmp"]
    return


@app.cell
def _(plt, scopf_planned_metrics, sn_metrics):
    plt.plot((sn_metrics["mean_max_lmp"] - scopf_planned_metrics["mean_max_lmp"])/sn_metrics["mean_max_lmp"] * 100)
    return


@app.cell
def _(CAPITAL_COSTS, np):

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


    def compute_metrics(distributed_outcome, planned_dc_capacities, capital_costs = CAPITAL_COSTS):

        metrics = {}

        # DC Land Cost
        if planned_dc_capacities is not None:
            num_nodes = len(planned_dc_capacities)
            metrics["dc_land_cost"] = np.dot(1000*planned_dc_capacities, capital_costs[:num_nodes])
        else:
            metrics["dc_land_cost"] = None

        # Mean Max LMP
        lmps = distributed_outcome.prices
        if len(lmps.shape) == 2:
            # Mean max LMP
            dist_stats = node_price_summaries(lmps, topk=5)
            mean_max_lmp = np.mean(dist_stats["max"])
            metrics["mean_max_lmp"] = mean_max_lmp * 100.0 * 4.0

            # Dispatch Cost
            metrics["dispatch"] = distributed_outcome.problem.value

            # Mean Max Line Dual
            mu_sum = distributed_outcome.local_inequality_duals[3][0] + distributed_outcome.local_inequality_duals[3][1]
            mean_max_line_dual = np.mean(mu_sum.max(axis=1))
            metrics["mean_max_line_dual"] = mean_max_line_dual
        elif len(lmps.shape) == 3:
            num_scenarios = lmps.shape[-1]
            mean_max_lmp_results = []
            mean_max_line_dual_results = []
            for i in range(num_scenarios):
                cur_scenario_lmps = lmps[:,:,i]

                # Mean Max LMP
                dist_stats = node_price_summaries(cur_scenario_lmps, topk=5)
                mean_max_lmp = np.mean(dist_stats["max"])
                mean_max_lmp_results.append(mean_max_lmp * 100.0 * 4.0)

                # # Mean Max Line Dual
                # mu_sum = distributed_outcome.local_inequality_duals[3][0][]

            
            metrics["mean_max_lmp"] = np.array(mean_max_lmp_results) 
        

        return metrics

    def get_congestion_metric(dispatch_outcome, *, line_device_idx: int = 3):
        mu_lo = dispatch_outcome.local_inequality_duals[line_device_idx][0]  # (L,T)
        mu_hi = dispatch_outcome.local_inequality_duals[line_device_idx][1]  # (L,T)
        mu = np.max(mu_lo + mu_hi, axis=1)  # max over time -> (L,)
        return float(np.mean(mu))


    return compute_metrics, get_congestion_metric, node_price_summaries


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

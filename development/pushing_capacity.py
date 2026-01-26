import marimo

__generated_with = "0.11.21"
app = marimo.App(width="medium")


@app.cell
def _():
    import cvxpy as cp
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    import numpy_financial as npf
    import zap
    from zap.importers.pypsa import load_pypsa_network, parse_buses
    import os
    from zap.planning.problem_abstract import AbstractPlanningProblem
    from zap.planning.operation_objectives import DispatchOutcome
    from typing import Optional, Any
    from pathlib import Path
    import pypsa
    from zap.devices import ACLine
    import pandas as pd
    import geopandas as gpd
    from copy import deepcopy
    return (
        ACLine,
        AbstractPlanningProblem,
        Any,
        DispatchOutcome,
        Optional,
        Path,
        cp,
        deepcopy,
        gpd,
        load_pypsa_network,
        mo,
        np,
        npf,
        os,
        parse_buses,
        pd,
        plt,
        pypsa,
        sns,
        zap,
    )


@app.cell
def _(gpd, load_pypsa_network, np, os, parse_buses, pd, pypsa):
    LOAD_SCALING_FACTOR = 1.27
    GEN_SCALING_FACTOR = 1.24
    LINE_SCALING_FACTOR = 0.7
    INVESTMENT_NODE_CANDS = [32, 82, 50, 18, 15, 22, 43, 14, 23, 20, 94, 65, 78] # This is already sorted by land cost

    def upsample_zap_devices(devices, factor=4, original_timesteps=24):
        """Upsample time-varying attributes of zap devices by repeating each timestep."""
        upsampled_zap_devices = []
        for dev in devices:
            upsampled_dev = dev.sample_time(original_timesteps*factor, original_timesteps)
            upsampled_zap_devices.append(upsampled_dev)

        return upsampled_zap_devices

    HOME_PATH = os.environ.get("HOME")
    PYPSA_NETW0RK_PATH = (
        HOME_PATH + "/zap_data/pypsa-networks/western_small/network_2023.nc"
    )
    pn = pypsa.Network(PYPSA_NETW0RK_PATH)
    snapshots = pn.generators_t.p_max_pu.index
    # snapshot_data = snapshots[5616:5640]  # 8/23/21 # hourly
    snapshot_data = snapshots[5448:5472]  # 8/16/21 # hourly

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
        sel.groupby("terminal")["land_usd2017_per_acre"]
          .first()   # or .mean(), depending on what you want
          .sort_index()
    )

    CAPITAL_COSTS = np.array(sel.land_usd2017_per_acre)

    pypsa_kwargs = {}
    pypsa_net, pypsa_devices = load_pypsa_network(
        pn, snapshot_data, power_unit=1.0e3, cost_unit=100.0, **pypsa_kwargs
    )
    print("Before upsample:", pypsa_devices[3].capital_cost[0])

    pypsa_devices = upsample_zap_devices(pypsa_devices, factor=4, original_timesteps=24)
    print("After upsample:", pypsa_devices[3].capital_cost[0])
    return (
        CAPITAL_COSTS,
        GEN_SCALING_FACTOR,
        HOME_PATH,
        INVESTMENT_NODE_CANDS,
        LINE_SCALING_FACTOR,
        LOAD_SCALING_FACTOR,
        PYPSA_NETW0RK_PATH,
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
        pypsa_devices,
        pypsa_kwargs,
        pypsa_net,
        sel,
        selected_node_fips,
        snapshot_data,
        snapshots,
        terminal_cost,
        upsample_zap_devices,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Universal Device Creator""")
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
        pypsa_net = planning_devices_params_dict["pypsa_net"]
        pypsa_devices = planning_devices_params_dict["pypsa_devices"]

        pypsa_devices_dc = deepcopy(pypsa_devices)

        # Scale load, gen, and line capacities
        pypsa_devices_dc[1].load *= load_scaling_factor
        pypsa_devices_dc[0].dynamic_capacity *= gen_scaling_factor
        pypsa_devices_dc[3].nominal_capacity *= line_scaling_factor
        pypsa_devices_dc[3].nominal_capacity[168] = 0.5
        pypsa_devices_dc[3].nominal_capacity[176] = 0.5
        pypsa_devices_dc[3].nominal_capacity[49] = 0.3

        # Select which nodes to build at
        dc_terminals = np.array(investment_node_cands[:num_nodes])
        n_dc = len(dc_terminals)
        dc_capital_costs = capital_costs[:n_dc]

        # Build nominal capacities for DC loads
        if np.isscalar(dc_nominal_capacity):
            nominal_capacity = np.full(n_dc, dc_nominal_capacity)
        else:
            nominal_capacity = dc_nominal_capacity

        # Build DCLoad object
        dcloads = zap.DataCenterLoad(
            num_nodes=pypsa_net.num_nodes,
            terminal=dc_terminals,
            profiles=n_dc*[workload_profile],
            nominal_capacity=nominal_capacity,
            linear_cost=np.ones(n_dc) * 0,
            settime_horizon=96,
            capital_cost=dc_capital_costs
        )

        pypsa_devices_dc.append(dcloads)
        return pypsa_devices_dc
    return (create_planning_devices,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Transmission Expansion Planning Routine""")
    return


@app.cell
def _(cp, zap):
    def run_transmission_expansion_planning_experiment(pypsa_net, pypsa_devices_dc, transmission_planning_exp_params_dict):
        trans_expansion_factor = transmission_planning_exp_params_dict["trans_expansion_factor"]
        op_obj_selector = transmission_planning_exp_params_dict["op_obj_selector"]
        lmp_metric = transmission_planning_exp_params_dict.get("lmp_metric", "meanmax")
        lmp_beta = transmission_planning_exp_params_dict.get("lmp_beta", 1.0)
        num_iters = transmission_planning_exp_params_dict["num_iters"]

        # Create dispatch layer
        xstar = zap.DispatchLayer(
            pypsa_net,
            pypsa_devices_dc,
            parameter_names={"line_capacity": (3, "nominal_capacity")},
            time_horizon=96,
            solver=cp.CLARABEL,
        )

        lower_bounds = {"line_capacity": pypsa_devices_dc[3].nominal_capacity}
        upper_bounds = {"line_capacity": trans_expansion_factor * pypsa_devices_dc[3].nominal_capacity}

        eta = {"line_capacity": pypsa_devices_dc[3].nominal_capacity}

        # Create objectives
        inv_obj = zap.planning.InvestmentObjective(pypsa_devices_dc, xstar)
        if op_obj_selector == "dispatch":
            op_obj = zap.planning.DispatchCostObjective(pypsa_net, pypsa_devices_dc)
        elif op_obj_selector == "lmp":
            op_obj = zap.planning.LMPObjective(pypsa_net, pypsa_devices_dc, lmp_metric=lmp_metric, lmp_beta = lmp_beta)

        P = zap.planning.PlanningProblem(
            operation_objective=op_obj,
            investment_objective=inv_obj,
            layer=xstar,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )

        P.extra_projections = {}

        cost = P(**eta, requires_grad=True)
        grad = P.backward()

        state = P.solve(num_iterations=num_iters, initial_state=eta)


        return state, P
    return (run_transmission_expansion_planning_experiment,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Generation Expansion Planning Routine""")
    return


@app.cell
def _(cp, planning_exp_params_dict, zap):
    def run_generation_expansion_planning_experiment(pypsa_net, pypsa_devices_dc, generation_planning_exp_params_dict):
        generation_expansion_factor = planning_exp_params_dict["generation_expansion_factor"]
        op_obj_selector = planning_exp_params_dict["op_obj_selector"]
        lmp_metric = planning_exp_params_dict.get("lmp_metric", "meanmax")
        lmp_beta = planning_exp_params_dict.get("lmp_beta", 1.0)
        num_iters = planning_exp_params_dict["num_iters"]

        # Create dispatch layer
        xstar = zap.DispatchLayer(
            pypsa_net,
            pypsa_devices_dc,
            parameter_names={"generator_capacity": (0, "nominal_capacity")},
            time_horizon=96,
            solver=cp.CLARABEL,
        )

        lower_bounds = {"generator_capacity": pypsa_devices_dc[0].nominal_capacity}
        upper_bounds = {"generator_capacity": generation_expansion_factor * pypsa_devices_dc[0].nominal_capacity}

        eta = {"generator_capacity": pypsa_devices_dc[0].nominal_capacity}

        # Create objectives
        inv_obj = zap.planning.InvestmentObjective(pypsa_devices_dc, xstar)
        if op_obj_selector == "dispatch":
            op_obj = zap.planning.DispatchCostObjective(pypsa_net, pypsa_devices_dc)
        elif op_obj_selector == "lmp":
            op_obj = zap.planning.LMPObjective(pypsa_net, pypsa_devices_dc, lmp_metric=lmp_metric, lmp_beta = lmp_beta)

        P = zap.planning.PlanningProblem(
            operation_objective=op_obj,
            investment_objective=inv_obj,
            layer=xstar,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )

        P.extra_projections = {}

        cost = P(**eta, requires_grad=True)
        grad = P.backward()

        state = P.solve(num_iterations=num_iters, initial_state=eta)


        return state, P
    return (run_generation_expansion_planning_experiment,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Distributed DC Planning Routine""")
    return


@app.cell
def _(cp, np, zap):
    def run_distributed_dc_planning_experiment(pypsa_net, pypsa_devices_dc, distributed_dc_planning_exp_params_dict):
        total_dc_budget = distributed_dc_planning_exp_params_dict["total_dc_budget"]
        dc_lower_bound = 0.0
        dc_upper_bound = total_dc_budget
        op_obj_selector = distributed_dc_planning_exp_params_dict["op_obj_selector"]
        lmp_metric = distributed_dc_planning_exp_params_dict.get("lmp_metric", "meanmax")
        lmp_beta = distributed_dc_planning_exp_params_dict.get("lmp_beta", 1.0)
        crit_idx = distributed_dc_planning_exp_params_dict.get("crit_idx", None)
        base_line_util = distributed_dc_planning_exp_params_dict.get("base_line_util", None)
        num_iters = distributed_dc_planning_exp_params_dict["num_iters"]

        n_dc = len(pypsa_devices_dc[-1].terminals)

        xstar = zap.DispatchLayer(
            pypsa_net,
            pypsa_devices_dc,
            parameter_names={"dc_capacity": (5, "nominal_capacity")},
            time_horizon=96,
            solver=cp.CLARABEL,
        )

        lower_bounds = {"dc_capacity": np.full(n_dc, dc_lower_bound)}
        upper_bounds = {"dc_capacity": np.full(n_dc, dc_upper_bound)}

        init_eta = np.full(n_dc, total_dc_budget / n_dc)
        print(init_eta)
        eta = {"dc_capacity": init_eta}

        # Create objectives
        inv_obj = zap.planning.InvestmentObjective(pypsa_devices_dc, xstar)
        if op_obj_selector == "dispatch":
            op_obj = zap.planning.DispatchCostObjective(pypsa_net, pypsa_devices_dc)
        elif op_obj_selector == "lmp":
            op_obj = zap.planning.LMPObjective(pypsa_net, pypsa_devices_dc, lmp_metric=lmp_metric, lmp_beta = lmp_beta)

        P = zap.planning.PlanningProblem(
            operation_objective=op_obj,
            investment_objective=inv_obj,
            layer=xstar,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )

        P.extra_projections = {
            "dc_capacity": zap.planning.SimplexBudgetProjection(
                budget=total_dc_budget, strict=True
            )
        }

        cost = P(**eta, requires_grad=True)
        grad = P.backward()

        state = P.solve(num_iterations=num_iters, initial_state=eta)


        return state, P
    return (run_distributed_dc_planning_experiment,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Metric Scorecard""")
    return


@app.cell
def _(DispatchOutcome, np):
    def get_congestion_metric(dispatch_outcome: DispatchOutcome):
        mu_lo = dispatch_outcome.local_inequality_duals[3][0]  # (L,T)
        mu_hi = dispatch_outcome.local_inequality_duals[3][1]  # (L,T)
        mu = np.max(mu_lo + mu_hi, axis=1)   # (L,T)
        C_avg = np.mean(mu) 
        return C_avg
    return (get_congestion_metric,)


@app.cell
def _(np):
    def node_price_summaries(prices, topk=5, q=(0.95, 0.99)):
        """
        prices: [N,T]
        Returns dict of [N,] summaries over time for each node.
        """
        prices = np.asarray(prices)

        out = {}
        # percentiles over time (axis=1)
        for qq in q:
            out[f"p{int(qq*100)}"] = np.quantile(prices, qq, axis=1)

        # mean over time
        out["mean"] = prices.mean(axis=1)

        # mean of top-k hours (per node)
        k = int(topk)
        out[f"mean_top{k}"] = np.sort(prices, axis=1)[:, -k:].mean(axis=1)

        # max (what you're already doing)
        out["max"] = prices.max(axis=1)

        return out

    return (node_price_summaries,)


@app.cell
def _(
    AbstractPlanningProblem,
    Any,
    CAPITAL_COSTS,
    DispatchOutcome,
    Optional,
    get_congestion_metric,
    node_price_summaries,
    np,
    npf,
):
    def get_score_entry(capacity_added: float, 
                        dispatch_outcome: DispatchOutcome,
                        pypsa_devices: list,
                        planning_state: Optional[tuple[dict | Any, dict]] = None,
                        dc_nominal_capacities: Optional[np.ndarray] = None,
                        P: Optional[AbstractPlanningProblem] = None, 
                        expansion_descriptor: Optional[str] = "single"):
        dispatch_cost = dispatch_outcome.problem.value
        congestion_metric = get_congestion_metric(dispatch_outcome=dispatch_outcome)
        transmission_cost = 0.0
        generation_cost = 0.0
        dc_investment_cost = 0.0
        generation_added = 0.0
        transmission_added = 0.0
        if "single" in expansion_descriptor:
            dc_investment_cost = CAPITAL_COSTS[0] * capacity_added
        elif "distributed" in expansion_descriptor and dc_nominal_capacities is not None:
            dc_investment_cost = np.dot(CAPITAL_COSTS[:len(dc_nominal_capacities)], dc_nominal_capacities)
        else:
            raise NotImplementedError("Bad Expansion Descriptor: cannot compute dc invesment cost")

        if "transmission" in expansion_descriptor and P is not None:
            transmission_cost = P.get_inv_cost()
            transmission_added = np.sum(planning_state[0]["line_capacity"].squeeze(1) - pypsa_devices[3].nominal_capacity.squeeze(1))

        if "generation" in expansion_descriptor and P is not None:
            generation_cost = P.get_inv_cost()
            generation_added = np.sum(planning_state[0]["generator_capacity"].squeeze(1) - pypsa_devices[0].nominal_capacity.squeeze(1))

        lmp_max = np.mean(node_price_summaries(prices=dispatch_outcome.prices, topk=5)["max"]) * 100 * 4
    

        annualized_dc_inv_cost = -npf.pmt(0.07, 20, dc_investment_cost)
        dispatch_cost_adjusted = dispatch_cost * 100.0
        generation_cost_adjusted = generation_cost * (8760.0 / 96.0) 
        transmission_cost_adjusted = transmission_cost * (8760.0 / 96.0)
    
        entry = {"Provisioned Capacity (GW)": capacity_added,
                 "Expansion Descriptor": expansion_descriptor,
                 "Dispatch Cost ($/day)": dispatch_cost_adjusted,
                 "Congestion Metric": congestion_metric,
                 "DC Investment Cost ($/yr)": annualized_dc_inv_cost,
                 "Transmission Cost ($/yr)": transmission_cost_adjusted,
                 "Transmission Added (MW)": transmission_added,
                 "Generation Cost ($/yr)": generation_cost_adjusted,
                 "Generation Added (MW)": generation_added,
                 "LMP Max ($)": lmp_max}
        return entry
    return (get_score_entry,)


@app.cell
def _(mo):
    mo.md(r"""## Build Single Node Table of LMP Arrs""")
    return


@app.cell
def _(
    CAPITAL_COSTS,
    GEN_SCALING_FACTOR,
    HOME_PATH,
    INVESTMENT_NODE_CANDS,
    LINE_SCALING_FACTOR,
    LOAD_SCALING_FACTOR,
    cp,
    create_planning_devices,
    get_score_entry,
    node_price_summaries,
    np,
    pypsa_devices,
    pypsa_net,
):
    DC_CAP_MAX = 5.0 # GW
    DC_CAP_MIN = 0.5 # GW
    DC_CAP_STEP = 0.1

    results = []
    lmp_dict = {}

    for dc_cap in np.arange(DC_CAP_MIN, DC_CAP_MAX, DC_CAP_STEP):
        planning_devices_params_dict = {
            "num_nodes": 1,
            "investment_node_cands": INVESTMENT_NODE_CANDS,
            "gen_scaling_factor": GEN_SCALING_FACTOR,
            "load_scaling_factor": LOAD_SCALING_FACTOR,
            "line_scaling_factor": LINE_SCALING_FACTOR,
            "dc_nominal_capacity": dc_cap, # GW
            "capital_costs": 0*CAPITAL_COSTS,
            "workload_profile": HOME_PATH + "/zap/development/load_profiles/example_inference_azure_conv.csv",
            "pypsa_net": pypsa_net,
            "pypsa_devices": pypsa_devices,
        }
        planning_devices = create_planning_devices(pypsa_devices, planning_devices_params_dict)
        single_node_outcome = pypsa_net.dispatch(
            planning_devices, time_horizon=96, solver=cp.CLARABEL, add_ground=False
        )
        
        lmp_arr = node_price_summaries(prices=single_node_outcome.prices, topk=5)
        entry = get_score_entry(capacity_added=dc_cap,
                        dispatch_outcome=single_node_outcome,
                        pypsa_devices=planning_devices)
        print(entry)
        lmp_dict[dc_cap] = lmp_arr
        results.append(entry)
    return (
        DC_CAP_MAX,
        DC_CAP_MIN,
        DC_CAP_STEP,
        dc_cap,
        entry,
        lmp_arr,
        lmp_dict,
        planning_devices,
        planning_devices_params_dict,
        results,
        single_node_outcome,
    )


@app.cell
def _(lmp_dict, np, pd, results):
    single_node_lmp_dict = {}
    for cap, arr in lmp_dict.items():
        single_node_lmp_dict[np.round(float(cap), 2)] = np.mean(arr["max"])
    df_single_node_lmp = (
        pd.DataFrame(
            single_node_lmp_dict.items(),
            columns=["Capacity (GW)", "LMP"]
        )
        .sort_values("Capacity (GW)")
    )

    df_single_node_lmp.to_csv("single_node_lmp_results.csv", index=False)


    df_single_node_results = pd.DataFrame(results)
    df_single_node_results.to_csv("single_node_full_results.csv")
    return (
        arr,
        cap,
        df_single_node_lmp,
        df_single_node_results,
        single_node_lmp_dict,
    )


@app.cell
def _(df_single_node_lmp):
    df_single_node_lmp
    return


@app.cell
def _(mo):
    mo.md(r"""## Distributed DC Experiment""")
    return


@app.cell
def _(
    CAPITAL_COSTS,
    DC_CAP_STEP,
    GEN_SCALING_FACTOR,
    HOME_PATH,
    INVESTMENT_NODE_CANDS,
    LINE_SCALING_FACTOR,
    LOAD_SCALING_FACTOR,
    cp,
    create_planning_devices,
    node_price_summaries,
    np,
    pd,
    pypsa_devices,
    pypsa_net,
    run_distributed_dc_planning_experiment,
):
    def cap_key(x: float) -> float:
        return float(np.round(x, 2))

    def scalar_lmp_mean_of_node_time_max(outcome) -> float:
        # outcome.prices assumed shape [N, T] in your helper
        arr = node_price_summaries(prices=outcome.prices, topk=5)
        return float(np.mean(arr["max"]))

    current_dc_cap = 1.0
    distributed_dc_only_results = []
    distributed_dc_only_lmp_dict = {}
    distributed_nominal_capacities = {}

    df = pd.read_csv("single_node_lmp_results.csv")
    single_node_lmp_dict_csv = {cap_key(c): float(v) for c, v in zip(df["Capacity (GW)"], df["LMP"])}

    single_node_dc_cap_reference = 1.0

    while cap_key(single_node_dc_cap_reference) in single_node_lmp_dict_csv:
        ref_key = cap_key(single_node_dc_cap_reference)
        ref_lmp = single_node_lmp_dict_csv[ref_key]

        best_feasible_cap = None

        while True:
            cur_key = cap_key(current_dc_cap)

            uniform_amnt = current_dc_cap / 10
            dc_lower_bound = max(0.5 * uniform_amnt, 0.05)
            dc_upper_bound = 2.5 * uniform_amnt

            planning_exp_params_dict_distributed_dc_only = {
                "total_dc_budget": current_dc_cap,
                "dc_lower_bound": dc_lower_bound,
                "dc_upper_bound": dc_upper_bound,
                "op_obj_selector": "lmp",
                "lmp_metric": "sumsmoothmax",
                "lmp_beta": 1000.0,
                "num_iters": 10,
            }

            planning_devices_params_dict_dist_only = {
                "num_nodes": 10,
                "investment_node_cands": INVESTMENT_NODE_CANDS,
                "gen_scaling_factor": GEN_SCALING_FACTOR,
                "load_scaling_factor": LOAD_SCALING_FACTOR,
                "line_scaling_factor": LINE_SCALING_FACTOR,
                "dc_nominal_capacity": current_dc_cap,   # just used to construct DataCenterLoad, planning will overwrite
                "capital_costs": 0 * CAPITAL_COSTS,
                "workload_profile": HOME_PATH + "/zap/development/load_profiles/example_inference_azure_conv.csv",
                "pypsa_net": pypsa_net,
                "pypsa_devices": pypsa_devices,
            }

            pypsa_devices_dc_dist_only = create_planning_devices(pypsa_devices, planning_devices_params_dict_dist_only)

            # NOTE: if your function actually returns only `state`, do NOT unpack into two vars
            planning_state_distributed_dc_only, P_distributed_dc_only = run_distributed_dc_planning_experiment(
                pypsa_net=pypsa_net,
                pypsa_devices_dc=pypsa_devices_dc_dist_only,
                distributed_dc_planning_exp_params_dict=planning_exp_params_dict_distributed_dc_only,
            )
            planned_dc_capacities = planning_state_distributed_dc_only[0]["dc_capacity"]
            distributed_nominal_capacities[cur_key] = planned_dc_capacities

            planned_devices_params_dict = {
                **planning_devices_params_dict_dist_only,
                "dc_nominal_capacity": planned_dc_capacities,
                "capital_costs": CAPITAL_COSTS,
            }
            planned_devices = create_planning_devices(pypsa_devices, planned_devices_params_dict)

            dist_outcome = pypsa_net.dispatch(
                devices=planned_devices,
                time_horizon=96,
                solver=cp.CLARABEL,
                add_ground=False,
            )

            dist_lmp = scalar_lmp_mean_of_node_time_max(dist_outcome)
            distributed_dc_only_lmp_dict[cur_key] = dist_lmp

            print("cur cap:", cur_key, "dist_lmp:", dist_lmp, "ref cap:", ref_key, "ref_lmp:", ref_lmp)

            # Feasible if distributed LMP stays under the single-node reference
            if dist_lmp <= ref_lmp:
                best_feasible_cap = cur_key
                current_dc_cap = current_dc_cap + DC_CAP_STEP
                continue

            break

        single_node_dc_cap_reference = single_node_dc_cap_reference + DC_CAP_STEP
        if best_feasible_cap is not None:
            current_dc_cap = max(current_dc_cap, best_feasible_cap)

    return (
        P_distributed_dc_only,
        best_feasible_cap,
        cap_key,
        cur_key,
        current_dc_cap,
        dc_lower_bound,
        dc_upper_bound,
        df,
        dist_lmp,
        dist_outcome,
        distributed_dc_only_lmp_dict,
        distributed_dc_only_results,
        distributed_nominal_capacities,
        planned_dc_capacities,
        planned_devices,
        planned_devices_params_dict,
        planning_devices_params_dict_dist_only,
        planning_exp_params_dict_distributed_dc_only,
        planning_state_distributed_dc_only,
        pypsa_devices_dc_dist_only,
        ref_key,
        ref_lmp,
        scalar_lmp_mean_of_node_time_max,
        single_node_dc_cap_reference,
        single_node_lmp_dict_csv,
        uniform_amnt,
    )


@app.cell
def _(
    HOME_PATH,
    Path,
    cap_key,
    curcap_to_refcap,
    distributed_dc_only_lmp_dict,
    distributed_nominal_capacities,
    np,
    pd,
    single_node_lmp_dict_csv,
):
    import json 
    outdir = Path(HOME_PATH) / "zap" / "development" / "results" / "lmp_distributed"
    outdir.mkdir(parents=True, exist_ok=True)

    # --- 3A) Save scalar dict (distributed_dc_only_lmp_dict) as CSV ---
    rows = []
    for cap_x, lmp in distributed_dc_only_lmp_dict.items():
        cap_k = cap_key(cap_x)
        row = {
            "distributed_total_cap_gw": cap_k,
            "dist_lmp_mean_of_node_time_max": float(lmp),
        }

        # Optional: if you added curcap_to_refcap during the run:
        if "curcap_to_refcap" in globals():
            ref_cap = curcap_to_refcap.get(cap_k, None)
            row["reference_cap_gw"] = ref_cap
            row["reference_lmp"] = (single_node_lmp_dict_csv.get(ref_cap) if ref_cap is not None else None)

        # Optional: quick stats on the planned capacity vector
        vec = distributed_nominal_capacities.get(cap_k, None)
        if vec is not None:
            v = np.asarray(vec).astype(float).ravel()
            row["dc_vec_min_gw"] = float(np.min(v))
            row["dc_vec_max_gw"] = float(np.max(v))
            row["dc_vec_sum_gw"] = float(np.sum(v))
        rows.append(row)

    df_dist = pd.DataFrame(rows).sort_values("distributed_total_cap_gw")
    csv_path = outdir / "distributed_lmp_frontier.csv"
    df_dist.to_csv(csv_path, index=False)

    # --- 3B) Save planned vectors losslessly ---
    # Store as NPZ with stable arrays: capacities list + padded matrix of vectors
    caps_sorted = [cap_key(c) for c in sorted(distributed_nominal_capacities.keys())]
    vecs = [np.asarray(distributed_nominal_capacities[c]).astype(float).ravel() for c in caps_sorted]

    # sanity: all vectors should have same length (n_dc)
    lengths = {v.shape[0] for v in vecs}
    if len(lengths) != 1:
        raise ValueError(f"Not all dc capacity vectors have same length: {sorted(lengths)}")
    n_dc = vecs[0].shape[0]

    vec_mat = np.vstack(vecs)  # shape: [num_caps, n_dc]
    npz_path = outdir / "distributed_nominal_capacities.npz"
    np.savez(
        npz_path,
        capacities_gw=np.array(caps_sorted, dtype=float),
        dc_capacity_vectors_gw=vec_mat,
    )

    # --- 3C) Save dicts as JSON for readability (optional) ---
    # NOTE: JSON won't preserve numpy types; convert carefully
    json_path = outdir / "distributed_lmp_dict.json"
    with open(json_path, "w") as f:
        json.dump({str(cap_key(k)): float(v) for k, v in distributed_dc_only_lmp_dict.items()}, f, indent=2)

    # If you want the single-node baseline too:
    baseline_path = outdir / "single_node_lmp_dict.json"
    with open(baseline_path, "w") as f:
        json.dump({str(cap_key(k)): float(v) for k, v in single_node_lmp_dict_csv.items()}, f, indent=2)

    print("Saved:")
    print(" -", csv_path)
    print(" -", npz_path)
    print(" -", json_path)
    print(" -", baseline_path)

    return (
        baseline_path,
        cap_k,
        cap_x,
        caps_sorted,
        csv_path,
        df_dist,
        f,
        json,
        json_path,
        lengths,
        lmp,
        n_dc,
        npz_path,
        outdir,
        ref_cap,
        row,
        rows,
        v,
        vec,
        vec_mat,
        vecs,
    )


@app.cell
def _(
    cap_key,
    distributed_dc_only_lmp_dict,
    np,
    plt,
    single_node_lmp_dict_csv,
):
    # --- 1) Sort distributed results into arrays ---
    dist_caps = np.array(sorted(cap_key(k) for k in distributed_dc_only_lmp_dict.keys()), dtype=float)
    dist_lmps = np.array([float(distributed_dc_only_lmp_dict[cap_key(k)]) for k in dist_caps], dtype=float)

    # --- 2) For each single-node reference cap, find max feasible distributed cap ---
    ref_caps = np.array(sorted(cap_key(k) for k in single_node_lmp_dict_csv.keys()), dtype=float)
    ref_lmps = np.array([float(single_node_lmp_dict_csv[cap_key(k)]) for k in ref_caps], dtype=float)

    max_dist_under_ref = np.full_like(ref_caps, np.nan, dtype=float)

    for i, thr in enumerate(ref_lmps):
        feasible = dist_caps[dist_lmps <= thr]
        if feasible.size > 0:
            max_dist_under_ref[i] = feasible.max()

    # (optional) only keep refs where we found a feasible distributed point
    mask = ~np.isnan(max_dist_under_ref)
    ref_caps_plot = ref_caps[mask]
    max_dist_plot = max_dist_under_ref[mask]

    # --- 3) Plot: y=x baseline + distributed frontier points ---
    plt.figure(figsize=(7, 6))

    xmin = min(ref_caps_plot.min(), max_dist_plot.min())
    xmax = max(ref_caps_plot.max(), max_dist_plot.max())
    grid = np.linspace(xmin, xmax, 200)
    plt.plot(grid, grid, linewidth=2, label="Single-node")
    plt.scatter(ref_caps_plot, max_dist_plot, s=40, label="Distributed")

    plt.xlabel("Single-node added capacity (GW)")
    plt.ylabel("Achievable Capacity (GW)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 5)
    plt.ylim(0, 5)
    plt.tight_layout()
    plt.show()

    return (
        dist_caps,
        dist_lmps,
        feasible,
        grid,
        i,
        mask,
        max_dist_plot,
        max_dist_under_ref,
        ref_caps,
        ref_caps_plot,
        ref_lmps,
        thr,
        xmax,
        xmin,
    )


@app.cell
def _(
    CAPITAL_COSTS,
    GEN_SCALING_FACTOR,
    HOME_PATH,
    INVESTMENT_NODE_CANDS,
    LINE_SCALING_FACTOR,
    LOAD_SCALING_FACTOR,
    cp,
    create_planning_devices,
    distributed_dc_only_lmp_dict,
    distributed_nominal_capacities,
    node_price_summaries,
    np,
    pypsa_devices,
    pypsa_net,
    run_distributed_dc_planning_experiment,
):
    # --- TEMP linear sweep extension to 7 GW (no name collisions) ---

    def _cap_key_lin(x: float) -> float:
        return float(np.round(float(x), 2))

    def _unwrap_state_lin(z):
        # Iterative search through nested tuples/lists to find a dict with "dc_capacity"
        stack = [z]
        while stack:
            cur = stack.pop()
            if isinstance(cur, dict):
                if "dc_capacity" in cur:
                    return cur
                # sometimes state dict is nested under keys; keep searching values
                stack.extend(list(cur.values()))
            elif isinstance(cur, (tuple, list)):
                stack.extend(list(cur))
        raise TypeError(f"Could not unwrap state dict from type={type(z)}")


    def _dist_lmp_scalar_lin(outcome) -> float:
        arr = node_price_summaries(prices=outcome.prices, topk=5)
        return float(np.mean(arr["max"]))

    # NEW dicts (do not reuse old names)
    distributed_nominal_capacities_lin = dict(distributed_nominal_capacities)   # copy existing
    distributed_lmp_lin = dict(distributed_dc_only_lmp_dict)                    # copy existing

    # Determine start point from what you've already computed
    _lin_start_gw = max(distributed_nominal_capacities_lin.keys()) if len(distributed_nominal_capacities_lin) > 0 else 1.0
    _lin_start_gw = _cap_key_lin(_lin_start_gw)

    _lin_target_gw = 7.0
    _lin_step_gw = 0.1  # or DC_CAP_STEP if you want, but keep name unique

    print(f"Extending distributed sweep from {_lin_start_gw:.2f} -> {_lin_target_gw:.2f} GW (step={_lin_step_gw})")

    _lin_budget_list = np.arange(_lin_start_gw + _lin_step_gw, _lin_target_gw + 1e-9, _lin_step_gw)

    for _lin_total_gw in _lin_budget_list:
        _lin_budget_key = _cap_key_lin(_lin_total_gw)

        if _lin_budget_key in distributed_nominal_capacities_lin:
            continue

        # bounds consistent with your regime (10 DC sites)
        _lin_uniform = _lin_total_gw / 10.0
        _lin_lb = max(0.5 * _lin_uniform, 0.05)
        _lin_ub = 2.5 * _lin_uniform

        _lin_plan_params = {
            "total_dc_budget": float(_lin_total_gw),
            "dc_lower_bound": float(_lin_lb),
            "dc_upper_bound": float(_lin_ub),
            "op_obj_selector": "lmp",
            "lmp_metric": "sumsmoothmax",
            "lmp_beta": 1000.0,
            "num_iters": 10,
        }

        _lin_dev_params = {
            "num_nodes": 10,
            "investment_node_cands": INVESTMENT_NODE_CANDS,
            "gen_scaling_factor": GEN_SCALING_FACTOR,
            "load_scaling_factor": LOAD_SCALING_FACTOR,
            "line_scaling_factor": LINE_SCALING_FACTOR,
            "dc_nominal_capacity": float(_lin_total_gw),  # placeholder; planning overwrites
            "capital_costs": 0 * CAPITAL_COSTS,
            "workload_profile": HOME_PATH + "/zap/development/load_profiles/example_inference_azure_conv.csv",
            "pypsa_net": pypsa_net,
            "pypsa_devices": pypsa_devices,
        }

        _lin_devices_for_plan = create_planning_devices(pypsa_devices, _lin_dev_params)

        _lin_solve_out = run_distributed_dc_planning_experiment(
            pypsa_net=pypsa_net,
            pypsa_devices_dc=_lin_devices_for_plan,
            distributed_dc_planning_exp_params_dict=_lin_plan_params,
        )

        _lin_state_dict = _unwrap_state_lin(_lin_solve_out)
        _lin_planned_vec = np.asarray(_lin_state_dict["dc_capacity"], dtype=float).ravel()

        # Evaluate with dispatch for consistent LMP metric
        _lin_dev_params_eval = {
            **_lin_dev_params,
            "dc_nominal_capacity": _lin_planned_vec,
            "capital_costs": CAPITAL_COSTS,
        }
        _lin_devices_eval = create_planning_devices(pypsa_devices, _lin_dev_params_eval)

        _lin_outcome_eval = pypsa_net.dispatch(
            devices=_lin_devices_eval,
            time_horizon=96,
            solver=cp.CLARABEL,
            add_ground=False,
        )

        distributed_nominal_capacities_lin[_lin_budget_key] = _lin_planned_vec
        distributed_lmp_lin[_lin_budget_key] = _dist_lmp_scalar_lin(_lin_outcome_eval)

        print(
            f"budget={_lin_budget_key:.2f}  "
            f"sum={_lin_planned_vec.sum():.4f}  "
            f"lmp={distributed_lmp_lin[_lin_budget_key]:.4f}"
        )

    # At end of cell, use these NEW dicts downstream:
    #   distributed_nominal_capacities_lin
    #   distributed_lmp_lin

    return distributed_lmp_lin, distributed_nominal_capacities_lin


@app.cell
def _(
    HOME_PATH,
    Path,
    curcap_to_refcap,
    distributed_dc_only_lmp_dict,
    distributed_lmp_lin,
    distributed_nominal_capacities,
    distributed_nominal_capacities_lin,
    json,
    np,
    pd,
    single_node_lmp_dict_csv,
):
    def _cap_key_merge_local(x: float) -> float:
        return float(np.round(float(x), 2))

    # ----------------------------
    # 0) Output directory (same place as before)  [RENAMED]
    # ----------------------------
    _merge_outdir = Path(HOME_PATH) / "zap" / "development" / "results" / "lmp_distributed"
    _merge_outdir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # 1) Merge dicts (lin overrides base on conflicts)  [RENAMED]
    # ----------------------------
    distributed_nominal_capacities_merged_local = dict(distributed_nominal_capacities)
    distributed_nominal_capacities_merged_local.update(distributed_nominal_capacities_lin)

    distributed_lmp_merged_local = dict(distributed_dc_only_lmp_dict)
    distributed_lmp_merged_local.update(distributed_lmp_lin)

    # ----------------------------
    # 2) Build CSV-friendly rows  [RENAMED]
    # ----------------------------
    _merge_rows = []

    for _k in sorted(distributed_lmp_merged_local.keys()):
        _kk = _cap_key_merge_local(_k)

        _row = {
            "distributed_total_cap_gw": _kk,
            "dist_lmp_mean_of_node_time_max": float(distributed_lmp_merged_local[_k]),
        }

        # Optional: reference mapping (only if present)
        if "curcap_to_refcap" in globals():
            _ref_cap = curcap_to_refcap.get(_kk, None)
            _row["reference_cap_gw"] = _ref_cap
            _row["reference_lmp"] = (single_node_lmp_dict_csv.get(_ref_cap) if _ref_cap is not None else None)

        # Capacity vector (try both float keys)
        _vec = distributed_nominal_capacities_merged_local.get(_k, None)
        if _vec is None:
            _vec = distributed_nominal_capacities_merged_local.get(_kk, None)

        if _vec is not None:
            _v = np.asarray(_vec, dtype=float).ravel()
            _row["dc_vec_sum_gw"] = float(_v.sum())
            _row["dc_vec_min_gw"] = float(_v.min())
            _row["dc_vec_max_gw"] = float(_v.max())
            _row["dc_capacity_vector_gw_json"] = json.dumps([float(x) for x in _v.tolist()])
        else:
            _row["dc_vec_sum_gw"] = None
            _row["dc_vec_min_gw"] = None
            _row["dc_vec_max_gw"] = None
            _row["dc_capacity_vector_gw_json"] = None

        _merge_rows.append(_row)

    df_distributed_merged_local = pd.DataFrame(_merge_rows).sort_values("distributed_total_cap_gw")

    # ----------------------------
    # 3) Write merged CSV (same filename as before)  [RENAMED]
    # ----------------------------
    _merge_csv_path = _merge_outdir / "distributed_lmp_frontier.csv"
    df_distributed_merged_local.to_csv(_merge_csv_path, index=False)

    print("Wrote merged CSV:", _merge_csv_path)
    print("Rows:", len(df_distributed_merged_local))

    # ----------------------------
    # 4) Save vectors losslessly as NPZ (same filename as before)  [RENAMED]
    # ----------------------------
    _merge_caps_sorted = df_distributed_merged_local["distributed_total_cap_gw"].to_numpy(dtype=float)

    _merge_vecs = []
    _merge_missing = []
    for _c in _merge_caps_sorted:
        _c_k = _cap_key_merge_local(_c)
        _vec = distributed_nominal_capacities_merged_local.get(_c, None)
        if _vec is None:
            _vec = distributed_nominal_capacities_merged_local.get(_c_k, None)
        if _vec is None:
            _merge_missing.append(_c_k)
            continue
        _merge_vecs.append(np.asarray(_vec, dtype=float).ravel())

    if len(_merge_missing) > 0:
        print("WARNING: missing vectors for caps:", _merge_missing)
        print("NPZ will NOT be written (to avoid misaligned matrices).")
    else:
        _merge_lengths = {v.shape[0] for v in _merge_vecs}
        if len(_merge_lengths) != 1:
            raise ValueError(f"Not all dc capacity vectors have same length: {sorted(_merge_lengths)}")

        _merge_vec_mat = np.vstack(_merge_vecs)  # [num_caps, n_dc]
        _merge_npz_path = _merge_outdir / "distributed_nominal_capacities.npz"
        np.savez(
            _merge_npz_path,
            capacities_gw=_merge_caps_sorted,
            dc_capacity_vectors_gw=_merge_vec_mat,
        )
        print("Wrote merged NPZ:", _merge_npz_path)

    # ----------------------------
    # 5) Save merged dict JSONs (same filenames as before)  [RENAMED]
    # ----------------------------
    _merge_json_path = _merge_outdir / "distributed_lmp_dict.json"
    with open(_merge_json_path, "w") as _fh:
        json.dump({str(_cap_key_merge_local(k)): float(v) for k, v in distributed_lmp_merged_local.items()}, _fh, indent=2)

    _merge_baseline_path = _merge_outdir / "single_node_lmp_dict.json"
    with open(_merge_baseline_path, "w") as _fh:
        json.dump({str(_cap_key_merge_local(k)): float(v) for k, v in single_node_lmp_dict_csv.items()}, _fh, indent=2)

    print("Wrote JSONs:")
    print(" -", _merge_json_path)
    print(" -", _merge_baseline_path)

    # export the things you likely want downstream (and these names are new)
    distributed_nominal_capacities_merged = distributed_nominal_capacities_merged_local
    distributed_lmp_merged = distributed_lmp_merged_local
    df_distributed_merged = df_distributed_merged_local

    return (
        df_distributed_merged,
        df_distributed_merged_local,
        distributed_lmp_merged,
        distributed_lmp_merged_local,
        distributed_nominal_capacities_merged,
        distributed_nominal_capacities_merged_local,
    )


@app.cell
def _(cp, node_price_summaries, np, zap):
    def cap_key_upgrade_v2(x: float) -> float:
        return float(np.round(float(x), 2))

    def extract_state_dict_upgrade_v2(z, want_key: str):
        stack = [z]
        while stack:
            cur = stack.pop()
            if isinstance(cur, dict):
                if want_key in cur:
                    return cur
                stack.extend(list(cur.values()))
            elif isinstance(cur, (tuple, list)):
                stack.extend(list(cur))
        raise TypeError(f"Could not find key={want_key} in solve output type={type(z)}")

    def lmp_scalar_upgrade_v2(outcome) -> float:
        arr = node_price_summaries(prices=outcome.prices, topk=5)
        return float(np.mean(arr["max"]))

    def run_generation_expansion_planning_experiment_v2(pypsa_net, pypsa_devices_dc, generation_planning_exp_params_dict):
        generation_expansion_factor = generation_planning_exp_params_dict["generation_expansion_factor"]
        op_obj_selector = generation_planning_exp_params_dict["op_obj_selector"]
        lmp_metric = generation_planning_exp_params_dict.get("lmp_metric", "meanmax")
        lmp_beta = generation_planning_exp_params_dict.get("lmp_beta", 1.0)
        num_iters = generation_planning_exp_params_dict["num_iters"]

        xstar = zap.DispatchLayer(
            pypsa_net,
            pypsa_devices_dc,
            parameter_names={"generator_capacity": (0, "nominal_capacity")},
            time_horizon=96,
            solver=cp.CLARABEL,
        )

        lower_bounds = {"generator_capacity": pypsa_devices_dc[0].nominal_capacity}
        upper_bounds = {"generator_capacity": generation_expansion_factor * pypsa_devices_dc[0].nominal_capacity}
        eta = {"generator_capacity": pypsa_devices_dc[0].nominal_capacity}

        inv_obj = zap.planning.InvestmentObjective(pypsa_devices_dc, xstar)
        if op_obj_selector == "dispatch":
            op_obj = zap.planning.DispatchCostObjective(pypsa_net, pypsa_devices_dc)
        elif op_obj_selector == "lmp":
            op_obj = zap.planning.LMPObjective(pypsa_net, pypsa_devices_dc, lmp_metric=lmp_metric, lmp_beta=lmp_beta)
        else:
            raise ValueError(f"bad op_obj_selector: {op_obj_selector}")

        P = zap.planning.PlanningProblem(
            operation_objective=op_obj,
            investment_objective=inv_obj,
            layer=xstar,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )
        P.extra_projections = {}

        _ = P(**eta, requires_grad=True)
        _ = P.backward()
        state = P.solve(num_iterations=num_iters, initial_state=eta)

        return state, P

    return (
        cap_key_upgrade_v2,
        extract_state_dict_upgrade_v2,
        lmp_scalar_upgrade_v2,
        run_generation_expansion_planning_experiment_v2,
    )


@app.cell
def _(
    CAPITAL_COSTS,
    GEN_SCALING_FACTOR,
    HOME_PATH,
    INVESTMENT_NODE_CANDS,
    LINE_SCALING_FACTOR,
    LOAD_SCALING_FACTOR,
    cap_key_upgrade_v2,
    cp,
    create_planning_devices,
    distributed_nominal_capacities_merged,
    extract_state_dict_upgrade_v2,
    lmp_scalar_upgrade_v2,
    np,
    pd,
    pypsa_devices,
    pypsa_net,
    run_transmission_expansion_planning_experiment,
):
    tx_params_v2 = {
        "trans_expansion_factor": 1.05,
        "op_obj_selector": "lmp",
        "lmp_metric": "sumsmoothmax",
        "lmp_beta": 1000.0,
        "num_iters": 10,
    }

    tx_upgrade_results_by_budget_v2 = {}

    tx_budget_list_v2 = sorted(cap_key_upgrade_v2(k) for k in distributed_nominal_capacities_merged.keys())

    for B in tx_budget_list_v2:
        dc_vec_B = np.asarray(distributed_nominal_capacities_merged[B], dtype=float).ravel()

        dev_params_tx = {
            "num_nodes": 10,
            "investment_node_cands": INVESTMENT_NODE_CANDS,
            "gen_scaling_factor": GEN_SCALING_FACTOR,
            "load_scaling_factor": LOAD_SCALING_FACTOR,
            "line_scaling_factor": LINE_SCALING_FACTOR,
            "dc_nominal_capacity": dc_vec_B,
            "capital_costs": CAPITAL_COSTS,
            "workload_profile": HOME_PATH + "/zap/development/load_profiles/example_inference_azure_conv.csv",
            "pypsa_net": pypsa_net,
            "pypsa_devices": pypsa_devices,
        }

        devs_tx_base = create_planning_devices(pypsa_devices, dev_params_tx)

        tx_state_out, tx_P_out = run_transmission_expansion_planning_experiment(
            pypsa_net=pypsa_net,
            pypsa_devices_dc=devs_tx_base,
            transmission_planning_exp_params_dict=tx_params_v2,
        )

        tx_state_dict = extract_state_dict_upgrade_v2(tx_state_out, "line_capacity")

        # Apply upgraded lines for evaluation
        devs_tx_eval = create_planning_devices(pypsa_devices, dev_params_tx)
        upgraded_lines = np.asarray(tx_state_dict["line_capacity"], dtype=float).squeeze()
        base_lines = np.asarray(devs_tx_eval[3].nominal_capacity, dtype=float).squeeze()
        devs_tx_eval[3].nominal_capacity = upgraded_lines

        out_tx = pypsa_net.dispatch(
            devices=devs_tx_eval,
            time_horizon=96,
            solver=cp.CLARABEL,
            add_ground=False,
        )

        tx_upgrade_results_by_budget_v2[B] = {
            "budget_gw": float(B),
            "post_tx_lmp": lmp_scalar_upgrade_v2(out_tx),
            "post_tx_dispatch_cost": float(out_tx.problem.value),
            "tx_inv_cost_raw": float(tx_P_out.get_inv_cost()),
            "tx_added_sum": float(np.sum(upgraded_lines - base_lines)),
        }

        print("TX budget", B, "post_lmp", tx_upgrade_results_by_budget_v2[B]["post_tx_lmp"])

    df_tx_upgrade_v2 = pd.DataFrame(list(tx_upgrade_results_by_budget_v2.values())).sort_values("budget_gw")
    df_tx_upgrade_v2

    return (
        B,
        base_lines,
        dc_vec_B,
        dev_params_tx,
        devs_tx_base,
        devs_tx_eval,
        df_tx_upgrade_v2,
        out_tx,
        tx_P_out,
        tx_budget_list_v2,
        tx_params_v2,
        tx_state_dict,
        tx_state_out,
        tx_upgrade_results_by_budget_v2,
        upgraded_lines,
    )


@app.cell
def _(
    CAPITAL_COSTS,
    GEN_SCALING_FACTOR,
    HOME_PATH,
    INVESTMENT_NODE_CANDS,
    LINE_SCALING_FACTOR,
    LOAD_SCALING_FACTOR,
    cap_key_upgrade_v2,
    cp,
    create_planning_devices,
    distributed_nominal_capacities_merged,
    extract_state_dict_upgrade_v2,
    lmp_scalar_upgrade_v2,
    np,
    pd,
    pypsa_devices,
    pypsa_net,
    run_generation_expansion_planning_experiment_v2,
):
    gen_params_v2 = {
        "generation_expansion_factor": 1.025,  # <= adjust
        "op_obj_selector": "lmp",
        "lmp_metric": "sumsmoothmax",
        "lmp_beta": 1000.0,
        "num_iters": 10,
    }

    gen_upgrade_results_by_budget_v2 = {}

    gen_budget_list_v2 = sorted(cap_key_upgrade_v2(k) for k in distributed_nominal_capacities_merged.keys())

    for B in gen_budget_list_v2:
        dc_vec_B = np.asarray(distributed_nominal_capacities_merged[B], dtype=float).ravel()

        dev_params_gen = {
            "num_nodes": 10,
            "investment_node_cands": INVESTMENT_NODE_CANDS,
            "gen_scaling_factor": GEN_SCALING_FACTOR,
            "load_scaling_factor": LOAD_SCALING_FACTOR,
            "line_scaling_factor": LINE_SCALING_FACTOR,
            "dc_nominal_capacity": dc_vec_B,
            "capital_costs": CAPITAL_COSTS,
            "workload_profile": HOME_PATH + "/zap/development/load_profiles/example_inference_azure_conv.csv",
            "pypsa_net": pypsa_net,
            "pypsa_devices": pypsa_devices,
        }

        devs_gen_base = create_planning_devices(pypsa_devices, dev_params_gen)

        gen_state_out, gen_P_out = run_generation_expansion_planning_experiment_v2(
            pypsa_net=pypsa_net,
            pypsa_devices_dc=devs_gen_base,
            generation_planning_exp_params_dict=gen_params_v2,
        )

        gen_state_dict = extract_state_dict_upgrade_v2(gen_state_out, "generator_capacity")

        # Apply upgraded generator capacities for evaluation
        devs_gen_eval = create_planning_devices(pypsa_devices, dev_params_gen)
        upgraded_gens = np.asarray(gen_state_dict["generator_capacity"], dtype=float).squeeze()
        base_gens = np.asarray(devs_gen_eval[0].nominal_capacity, dtype=float).squeeze()
        devs_gen_eval[0].nominal_capacity = upgraded_gens

        out_gen = pypsa_net.dispatch(
            devices=devs_gen_eval,
            time_horizon=96,
            solver=cp.CLARABEL,
            add_ground=False,
        )

        gen_upgrade_results_by_budget_v2[B] = {
            "budget_gw": float(B),
            "post_gen_lmp": lmp_scalar_upgrade_v2(out_gen),
            "post_gen_dispatch_cost": float(out_gen.problem.value),
            "gen_inv_cost_raw": float(gen_P_out.get_inv_cost()),
            "gen_added_sum": float(np.sum(upgraded_gens - base_gens)),
        }

        print("GEN budget", B, "post_lmp", gen_upgrade_results_by_budget_v2[B]["post_gen_lmp"])

    df_gen_upgrade_v2 = pd.DataFrame(list(gen_upgrade_results_by_budget_v2.values())).sort_values("budget_gw")
    df_gen_upgrade_v2

    return (
        B,
        base_gens,
        dc_vec_B,
        dev_params_gen,
        devs_gen_base,
        devs_gen_eval,
        df_gen_upgrade_v2,
        gen_P_out,
        gen_budget_list_v2,
        gen_params_v2,
        gen_state_dict,
        gen_state_out,
        gen_upgrade_results_by_budget_v2,
        out_gen,
        upgraded_gens,
    )


@app.cell
def _(HOME_PATH, Path, df_tx_upgrade_v2, np):
    outdir_upgrades = Path(HOME_PATH) / "zap" / "development" / "results" / "lmp_distributed"
    outdir_upgrades.mkdir(parents=True, exist_ok=True)

    # --- CSVs ---
    tx_csv = outdir_upgrades / "tx_upgrade_results.csv"
    # gen_csv = outdir_upgrades / "gen_upgrade_results.csv"

    df_tx_upgrade_v2.to_csv(tx_csv, index=False)
    # df_gen_upgrade_v2.to_csv(gen_csv, index=False)

    # --- NPZ (lossless) ---
    tx_npz = outdir_upgrades / "tx_upgrade_results.npz"
    # gen_npz = outdir_upgrades / "gen_upgrade_results.npz"

    np.savez(
        tx_npz,
        budget_gw=df_tx_upgrade_v2["budget_gw"].to_numpy(dtype=float),
        post_tx_lmp=df_tx_upgrade_v2["post_tx_lmp"].to_numpy(dtype=float),
        post_tx_dispatch_cost=df_tx_upgrade_v2["post_tx_dispatch_cost"].to_numpy(dtype=float),
        tx_inv_cost_raw=df_tx_upgrade_v2["tx_inv_cost_raw"].to_numpy(dtype=float),
        tx_added_sum=df_tx_upgrade_v2["tx_added_sum"].to_numpy(dtype=float),
    )

    # np.savez(
    #     gen_npz,
    #     budget_gw=df_gen_upgrade_v2["budget_gw"].to_numpy(dtype=float),
    #     post_gen_lmp=df_gen_upgrade_v2["post_gen_lmp"].to_numpy(dtype=float),
    #     post_gen_dispatch_cost=df_gen_upgrade_v2["post_gen_dispatch_cost"].to_numpy(dtype=float),
    #     gen_inv_cost_raw=df_gen_upgrade_v2["gen_inv_cost_raw"].to_numpy(dtype=float),
    #     gen_added_sum=df_gen_upgrade_v2["gen_added_sum"].to_numpy(dtype=float),
    # )

    print("Saved:")
    print(" -", tx_csv)
    # print(" -", gen_csv)
    print(" -", tx_npz)
    # print(" -", gen_npz)

    return outdir_upgrades, tx_csv, tx_npz


@app.cell
def _(
    df_tx_upgrade_v2,
    distributed_dc_only_lmp_dict,
    distributed_lmp_lin,
    distributed_lmp_merged,
    np,
    plt,
    single_node_lmp_dict_csv,
):
    def _ck_tx(x):
        return float(np.round(float(x), 2))

    # ----------------------------
    # 0) TX-upgraded curve from df_tx_upgrade_v2
    # ----------------------------
    _tx_caps = np.array(sorted(_ck_tx(c) for c in df_tx_upgrade_v2["budget_gw"].values), dtype=float)

    _tx_post_lmps = np.array(
        [float(df_tx_upgrade_v2.loc[df_tx_upgrade_v2["budget_gw"] == c, "post_tx_lmp"].iloc[0]) for c in _tx_caps],
        dtype=float
    )

    # ----------------------------
    # 1) Distributed-only curve (use merged if available, else fall back)
    # ----------------------------
    if "distributed_lmp_merged" in globals():
        _dist_lmp_dict = distributed_lmp_merged
    elif "distributed_lmp_lin" in globals():
        _dist_lmp_dict = distributed_lmp_lin
    else:
        _dist_lmp_dict = distributed_dc_only_lmp_dict

    _dist_caps = np.array(sorted(_ck_tx(k) for k in _dist_lmp_dict.keys()), dtype=float)
    _dist_lmps = np.array([float(_dist_lmp_dict[_ck_tx(k)]) for k in _dist_caps], dtype=float)

    # ----------------------------
    # 2) Single-node reference thresholds
    # ----------------------------
    _ref_caps_tx = np.array(sorted(_ck_tx(k) for k in single_node_lmp_dict_csv.keys()), dtype=float)
    _ref_lmps_tx = np.array([float(single_node_lmp_dict_csv[_ck_tx(k)]) for k in _ref_caps_tx], dtype=float)

    # ----------------------------
    # 3) For each ref LMP threshold, max feasible capacity for:
    #    (a) distributed-only
    #    (b) distributed + TX upgrade
    # ----------------------------
    _max_dist_under_ref = np.full_like(_ref_caps_tx, np.nan, dtype=float)
    _max_tx_under_ref = np.full_like(_ref_caps_tx, np.nan, dtype=float)

    for _idx_tx, _thr_tx in enumerate(_ref_lmps_tx):
        _feasible_dist = _dist_caps[_dist_lmps <= _thr_tx]
        if _feasible_dist.size > 0:
            _max_dist_under_ref[_idx_tx] = _feasible_dist.max()

        _feasible_tx = _tx_caps[_tx_post_lmps <= _thr_tx]
        if _feasible_tx.size > 0:
            _max_tx_under_ref[_idx_tx] = _feasible_tx.max()

    # keep only defined points (separate masks so one curve can extend farther than the other)
    _mask_dist = ~np.isnan(_max_dist_under_ref)
    _ref_caps_plot_dist = _ref_caps_tx[_mask_dist]
    _max_dist_plot = _max_dist_under_ref[_mask_dist]

    _mask_tx = ~np.isnan(_max_tx_under_ref)
    _ref_caps_plot_tx = _ref_caps_tx[_mask_tx]
    _max_tx_plot = _max_tx_under_ref[_mask_tx]

    # ----------------------------
    # 4) Plot
    # ----------------------------
    plt.figure(figsize=(7, 6))

    _xmax_tx = 0.0
    if _ref_caps_plot_dist.size > 0:
        _xmax_tx = max(_xmax_tx, float(_ref_caps_plot_dist.max()), float(_max_dist_plot.max()))
    if _ref_caps_plot_tx.size > 0:
        _xmax_tx = max(_xmax_tx, float(_ref_caps_plot_tx.max()), float(_max_tx_plot.max()))
    _xmax_tx = max(_xmax_tx, float(_ref_caps_tx.max()))

    _grid_tx = np.linspace(0.0, _xmax_tx, 200)

    plt.plot(_grid_tx, _grid_tx, linewidth=2, label="Single-node (y=x)")
    plt.scatter(_ref_caps_plot_dist, _max_dist_plot, s=40, label="Distributed")
    plt.scatter(_ref_caps_plot_tx, _max_tx_plot, s=40, label="Distributed + TX upgrade")

    plt.xlabel("Single-node added capacity (GW)")
    plt.ylabel("Achievable distributed capacity (GW)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return


if __name__ == "__main__":
    app.run()

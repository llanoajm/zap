import marimo

__generated_with = "0.11.21"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    from copy import deepcopy
    from pathlib import Path
    from typing import Any, Optional

    import cvxpy as cp
    import geopandas as gpd
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import numpy_financial as npf
    import pandas as pd
    import pypsa
    import seaborn as sns

    import zap
    from zap.devices import ACLine
    from zap.importers.pypsa import load_pypsa_network, parse_buses
    from zap.planning.operation_objectives import DispatchOutcome
    from zap.planning.problem_abstract import AbstractPlanningProblem
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
    INVESTMENT_NODE_CANDS = [
        32,
        82,
        50,
        18,
        15,
        22,
        43,
        14,
        23,
        20,
        94,
        65,
        78,
    ]  # This is already sorted by land cost

    def upsample_zap_devices(devices, factor=4, original_timesteps=24):
        """Upsample time-varying attributes of zap devices by repeating each timestep."""
        upsampled_zap_devices = []
        for dev in devices:
            upsampled_dev = dev.sample_time(original_timesteps * factor, original_timesteps)
            upsampled_zap_devices.append(upsampled_dev)

        return upsampled_zap_devices

    HOME_PATH = os.environ.get("HOME")
    PYPSA_NETW0RK_PATH = HOME_PATH + "/zap_data/pypsa-networks/western_small/network_2023.nc"
    pn = pypsa.Network(PYPSA_NETW0RK_PATH)
    snapshots = pn.generators_t.p_max_pu.index
    # snapshot_data = snapshots[5616:5640]  # 8/23/21 # hourly
    snapshot_data = snapshots[5448:5472]  # 8/16/21 # hourly

    buses, buses_to_index = parse_buses(
        pn
    )  # buses_to_index is dict of "pyspa_bus_name": "zap_terminal"
    index_to_bus = {idx: name for name, idx in buses_to_index.items()}
    pypsa_bus_names = [index_to_bus[i] for i in INVESTMENT_NODE_CANDS]

    b = pn.buses.copy()
    gdf = gpd.GeoDataFrame(b, geometry=gpd.points_from_xy(b["x"], b["y"]), crs="EPSG:4326")

    county_url = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_county_500k.zip"
    counties = gpd.read_file(county_url)[
        ["STATEFP", "COUNTYFP", "GEOID", "NAME", "STATE_NAME", "geometry"]
    ]

    j = gpd.sjoin(gdf, counties.to_crs("EPSG:4326"), how="left", predicate="within")

    pn.buses["county_fips"] = j["GEOID"]  # 5-digit FIPS
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
        sel.groupby("terminal")["land_usd2017_per_acre"]
        .first()  # or .mean(), depending on what you want
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
            profiles=n_dc * [workload_profile],
            nominal_capacity=nominal_capacity,
            linear_cost=np.ones(n_dc) * 0,
            settime_horizon=96,
            capital_cost=dc_capital_costs,
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
    def run_transmission_expansion_planning_experiment(
        pypsa_net, pypsa_devices_dc, transmission_planning_exp_params_dict
    ):
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
        upper_bounds = {
            "line_capacity": trans_expansion_factor * pypsa_devices_dc[3].nominal_capacity
        }

        eta = {"line_capacity": pypsa_devices_dc[3].nominal_capacity}

        # Create objectives
        inv_obj = zap.planning.InvestmentObjective(pypsa_devices_dc, xstar)
        if op_obj_selector == "dispatch":
            op_obj = zap.planning.DispatchCostObjective(pypsa_net, pypsa_devices_dc)
        elif op_obj_selector == "lmp":
            op_obj = zap.planning.LMPObjective(
                pypsa_net, pypsa_devices_dc, lmp_metric=lmp_metric, lmp_beta=lmp_beta
            )

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
    def run_generation_expansion_planning_experiment(
        pypsa_net, pypsa_devices_dc, generation_planning_exp_params_dict
    ):
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
        upper_bounds = {
            "generator_capacity": generation_expansion_factor * pypsa_devices_dc[0].nominal_capacity
        }

        eta = {"generator_capacity": pypsa_devices_dc[0].nominal_capacity}

        # Create objectives
        inv_obj = zap.planning.InvestmentObjective(pypsa_devices_dc, xstar)
        if op_obj_selector == "dispatch":
            op_obj = zap.planning.DispatchCostObjective(pypsa_net, pypsa_devices_dc)
        elif op_obj_selector == "lmp":
            op_obj = zap.planning.LMPObjective(
                pypsa_net, pypsa_devices_dc, lmp_metric=lmp_metric, lmp_beta=lmp_beta
            )

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
    def run_distributed_dc_planning_experiment(
        pypsa_net, pypsa_devices_dc, distributed_dc_planning_exp_params_dict
    ):
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
            op_obj = zap.planning.LMPObjective(
                pypsa_net, pypsa_devices_dc, lmp_metric=lmp_metric, lmp_beta=lmp_beta
            )

        P = zap.planning.PlanningProblem(
            operation_objective=op_obj,
            investment_objective=inv_obj,
            layer=xstar,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )

        P.extra_projections = {
            "dc_capacity": zap.planning.BoxBudgetProjection(
                budget=total_dc_budget,
                lower_bounds=np.full(n_dc, dc_lower_bound),
                upper_bounds=np.full(n_dc, dc_upper_bound),
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
        mu = np.max(mu_lo + mu_hi, axis=1)  # (L,T)
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
            out[f"p{int(qq * 100)}"] = np.quantile(prices, qq, axis=1)

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
    def get_score_entry(
        capacity_added: float,
        dispatch_outcome: DispatchOutcome,
        pypsa_devices: list,
        planning_state: Optional[tuple[dict | Any, dict]] = None,
        dc_nominal_capacities: Optional[np.ndarray] = None,
        P: Optional[AbstractPlanningProblem] = None,
        expansion_descriptor: Optional[str] = "single",
    ):
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
            dc_investment_cost = np.dot(
                CAPITAL_COSTS[: len(dc_nominal_capacities)], dc_nominal_capacities
            )
        else:
            raise NotImplementedError("Bad Expansion Descriptor: cannot compute dc invesment cost")

        if "transmission" in expansion_descriptor and P is not None:
            transmission_cost = P.get_inv_cost()
            transmission_added = np.sum(
                planning_state[0]["line_capacity"].squeeze(1)
                - pypsa_devices[3].nominal_capacity.squeeze(1)
            )

        if "generation" in expansion_descriptor and P is not None:
            generation_cost = P.get_inv_cost()
            generation_added = np.sum(
                planning_state[0]["generator_capacity"].squeeze(1)
                - pypsa_devices[0].nominal_capacity.squeeze(1)
            )

        lmp_max = (
            np.mean(node_price_summaries(prices=dispatch_outcome.prices, topk=5)["max"]) * 100 * 4
        )

        annualized_dc_inv_cost = -npf.pmt(0.07, 20, dc_investment_cost)
        dispatch_cost_adjusted = dispatch_cost * 100.0
        generation_cost_adjusted = generation_cost * (8760.0 / 96.0)
        transmission_cost_adjusted = transmission_cost * (8760.0 / 96.0)

        entry = {
            "Provisioned Capacity (GW)": capacity_added,
            "Expansion Descriptor": expansion_descriptor,
            "Dispatch Cost ($/day)": dispatch_cost_adjusted,
            "Congestion Metric": congestion_metric,
            "DC Investment Cost ($/yr)": annualized_dc_inv_cost,
            "Transmission Cost ($/yr)": transmission_cost_adjusted,
            "Transmission Added (MW)": transmission_added,
            "Generation Cost ($/yr)": generation_cost_adjusted,
            "Generation Added (MW)": generation_added,
            "LMP Max ($)": lmp_max,
        }
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
    DC_CAP_MAX = 5.15  # GW
    DC_CAP_MIN = 2.25  # GW
    DC_CAP_STEP = 0.1

    results = []
    lmp_dict = {}

    for dc_cap in np.arange(DC_CAP_MIN, DC_CAP_MAX, DC_CAP_STEP):
        planning_devices_params_dict = {
            "num_nodes": 1,
            "investment_node_cands": [20],
            "gen_scaling_factor": GEN_SCALING_FACTOR,
            "load_scaling_factor": LOAD_SCALING_FACTOR,
            "line_scaling_factor": LINE_SCALING_FACTOR,
            "dc_nominal_capacity": dc_cap,  # GW
            "capital_costs": 0 * CAPITAL_COSTS,
            "workload_profile": HOME_PATH
            + "/zap/development/load_profiles/example_inference_azure_conv.csv",
            "pypsa_net": pypsa_net,
            "pypsa_devices": pypsa_devices,
        }
        planning_devices = create_planning_devices(pypsa_devices, planning_devices_params_dict)
        single_node_outcome = pypsa_net.dispatch(
            planning_devices, time_horizon=96, solver=cp.CLARABEL, add_ground=False
        )

        lmp_arr = node_price_summaries(prices=single_node_outcome.prices, topk=5)
        entry = get_score_entry(
            capacity_added=dc_cap,
            dispatch_outcome=single_node_outcome,
            pypsa_devices=planning_devices,
        )
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
    df_single_node_lmp = pd.DataFrame(
        single_node_lmp_dict.items(), columns=["Capacity (GW)", "LMP"]
    ).sort_values("Capacity (GW)")

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
    run_distributed_dc_planning_experiment,
):
    # --- TEMP linear sweep extension to 7 GW (no name collisions) ---

    def _bin_dc_nominal_capacities_to_step(dc_vec_gw, step_gw: float = 0.05):
        v = np.asarray(dc_vec_gw, dtype=float).ravel()
        step = float(step_gw)
        scaled = v / step
        binned = np.round(scaled) * step
        binned = np.round(binned / step) * step
        return binned.astype(float)

    def _cap_key_lin(x: float) -> float:
        return float(np.round(float(x), 2))

    def _unwrap_state_lin(z):
        stack = [z]
        while stack:
            cur = stack.pop()
            if isinstance(cur, dict):
                if "dc_capacity" in cur:
                    return cur
                stack.extend(list(cur.values()))
            elif isinstance(cur, (tuple, list)):
                stack.extend(list(cur))
        raise TypeError(f"Could not unwrap state dict from type={type(z)}")

    def _dist_lmp_scalar_lin(outcome) -> float:
        arr = node_price_summaries(prices=outcome.prices, topk=5)
        return float(np.mean(arr["max"]))

    _lin_start_gw = 2.4
    _lin_start_gw = _cap_key_lin(_lin_start_gw)
    _lin_target_gw = 7.0
    _lin_step_gw = 0.1

    distributed_nominal_capacities_lin = {}
    distributed_lmp_lin = {}
    distributed_scores_lin = []


    _lin_budget_list = np.arange(_lin_start_gw + _lin_step_gw, _lin_target_gw + 1e-9, _lin_step_gw)

    for _lin_total_gw in _lin_budget_list:
        _lin_budget_key = _cap_key_lin(_lin_total_gw)

        _lin_uniform = _lin_total_gw / 10.0

        _lin_plan_params = {
            "total_dc_budget": float(_lin_total_gw),
            "dc_lower_bound": 0.0,
            "dc_upper_bound": _lin_total_gw,
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
            "dc_nominal_capacity": float(_lin_total_gw),
            "capital_costs": 0 * CAPITAL_COSTS,
            "workload_profile": HOME_PATH
            + "/zap/development/load_profiles/example_inference_azure_conv.csv",
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
        _lin_planned_vec = _bin_dc_nominal_capacities_to_step(_lin_planned_vec)

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
        distributed_scores_lin.append(get_score_entry(capacity_added=_lin_total_gw, dispatch_outcome=_lin_outcome_eval, pypsa_devices=_lin_devices_eval, planning_state=_lin_solve_out[0], dc_nominal_capacities=_lin_planned_vec, P=_lin_solve_out[1], expansion_descriptor="distributed"))

        print(
            f"budget={_lin_budget_key:.2f}  "
            f"sum={_lin_planned_vec.sum():.4f}  "
            f"lmp={distributed_lmp_lin[_lin_budget_key]:.4f}"
        )
    return (
        distributed_lmp_lin,
        distributed_nominal_capacities_lin,
        distributed_scores_lin,
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

    def run_generation_expansion_planning_experiment_v2(
        pypsa_net, pypsa_devices_dc, generation_planning_exp_params_dict
    ):
        generation_expansion_factor = generation_planning_exp_params_dict[
            "generation_expansion_factor"
        ]
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
        upper_bounds = {
            "generator_capacity": generation_expansion_factor * pypsa_devices_dc[0].nominal_capacity
        }
        eta = {"generator_capacity": pypsa_devices_dc[0].nominal_capacity}

        inv_obj = zap.planning.InvestmentObjective(pypsa_devices_dc, xstar)
        if op_obj_selector == "dispatch":
            op_obj = zap.planning.DispatchCostObjective(pypsa_net, pypsa_devices_dc)
        elif op_obj_selector == "lmp":
            op_obj = zap.planning.LMPObjective(
                pypsa_net, pypsa_devices_dc, lmp_metric=lmp_metric, lmp_beta=lmp_beta
            )
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

    tx_budget_list_v2 = sorted(
        cap_key_upgrade_v2(k) for k in distributed_nominal_capacities_merged.keys()
    )

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
            "workload_profile": HOME_PATH
            + "/zap/development/load_profiles/example_inference_azure_conv.csv",
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

    df_tx_upgrade_v2 = pd.DataFrame(list(tx_upgrade_results_by_budget_v2.values())).sort_values(
        "budget_gw"
    )
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

    gen_budget_list_v2 = sorted(
        cap_key_upgrade_v2(k) for k in distributed_nominal_capacities_merged.keys()
    )

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
            "workload_profile": HOME_PATH
            + "/zap/development/load_profiles/example_inference_azure_conv.csv",
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

    df_gen_upgrade_v2 = pd.DataFrame(list(gen_upgrade_results_by_budget_v2.values())).sort_values(
        "budget_gw"
    )
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


if __name__ == "__main__":
    app.run()

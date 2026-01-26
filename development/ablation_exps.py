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

    import zap
    from zap.importers.pypsa import load_pypsa_network, parse_buses
    import os
    from pathlib import Path
    import pypsa
    from zap.devices import ACLine
    import pandas as pd
    import geopandas as gpd
    from copy import deepcopy
    return (
        ACLine,
        Path,
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
        zap,
    )


@app.cell
def _():
    LOAD_SCALING_FACTOR = 1.27
    GEN_SCALING_FACTOR = 1.24
    # GEN_SCALING_FACTOR = 1
    LINE_SCALING_FACTOR = 0.7
    INVESTMENT_NODE_CANDS = [32, 82, 50, 18, 15, 22, 43, 14, 23, 20, 94, 65, 78] # This is already sorted by land cost
    # INVESTMENT_NODE_CANDS = [0, 82, 50, 18, 15, 22, 43, 14, 23, 20, 45, 65, 78] # This is already sorted by land cost
    return (
        GEN_SCALING_FACTOR,
        INVESTMENT_NODE_CANDS,
        LINE_SCALING_FACTOR,
        LOAD_SCALING_FACTOR,
    )


@app.cell
def upsample_zap_devices():
    def upsample_zap_devices(devices, factor=4, original_timesteps=24):
        """Upsample time-varying attributes of zap devices by repeating each timestep."""
        upsampled_zap_devices = []
        for dev in devices:
            upsampled_dev = dev.sample_time(original_timesteps*factor, original_timesteps)
            upsampled_zap_devices.append(upsampled_dev)

        return upsampled_zap_devices
    return (upsample_zap_devices,)


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
def _(os, pypsa):
    HOME_PATH = os.environ.get("HOME")
    PYPSA_NETW0RK_PATH = (
        HOME_PATH + "/zap_data/pypsa-networks/western_small/network_2023.nc"
    )
    pn = pypsa.Network(PYPSA_NETW0RK_PATH)
    snapshots = pn.generators_t.p_max_pu.index
    # snapshot_data = snapshots[5616:5640]  # 8/23/21 # hourly
    snapshot_data = snapshots[5448:5472]  # 8/16/21 # hourly
    return HOME_PATH, PYPSA_NETW0RK_PATH, pn, snapshot_data, snapshots


@app.cell
def _(mo):
    mo.md(r"""## Get terminal/bus cost information for possible investment candidate nodes""")
    return


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
    return bus_to_terminal, sel, terminal_cost


@app.cell
def _(np, sel):
    CAPITAL_COSTS = np.array(sel.land_usd2017_per_acre)
    return (CAPITAL_COSTS,)


@app.cell
def _(mo):
    mo.md(r"""## Convert PyPSA network to Zap""")
    return


@app.cell
def _(load_pypsa_network, pn, snapshot_data, upsample_zap_devices):
    pypsa_kwargs = {}
    pypsa_net, pypsa_devices = load_pypsa_network(
        pn, snapshot_data, power_unit=1.0e3, cost_unit=100.0, **pypsa_kwargs
    )

    pypsa_devices = upsample_zap_devices(pypsa_devices, factor=4, original_timesteps=24)
    return pypsa_devices, pypsa_kwargs, pypsa_net


@app.cell
def _(mo):
    mo.md(r"""## Helper to create devices for planning problem""")
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


@app.cell
def _(cp, np, zap):
    def run_planning_experiment(pypsa_net, pypsa_devices_dc, planning_exp_params_dict):
        total_dc_budget = planning_exp_params_dict["total_dc_budget"]
        dc_lower_bound = planning_exp_params_dict["dc_lower_bound"]
        dc_upper_bound = planning_exp_params_dict["dc_upper_bound"] 
        op_obj_selector = planning_exp_params_dict["op_obj_selector"]
        lmp_metric = planning_exp_params_dict.get("lmp_metric", "meanmax")
        lmp_beta = planning_exp_params_dict.get("lmp_beta", 1.0)
        crit_idx = planning_exp_params_dict.get("crit_idx", None)
        base_line_util = planning_exp_params_dict.get("base_line_util", None)
        num_iters = planning_exp_params_dict["num_iters"]

        n_dc = len(pypsa_devices_dc[-1].terminals)

        # Create dispatch layer
        xstar = zap.DispatchLayer(
            pypsa_net,
            pypsa_devices_dc,
            parameter_names={"dc_capacity": (5, "nominal_capacity")},
            time_horizon=96,
            solver=cp.CLARABEL,
        )

        lower_bounds = {"dc_capacity": np.full(n_dc, dc_lower_bound)}
        upper_bounds = {"dc_capacity": np.full(n_dc, dc_upper_bound)}

        # Initialize capacities eta randomly within bounds
        # init_eta = np.random.rand(n_dc).clip(dc_lower_bound, dc_upper_bound)
        init_eta = np.full(n_dc, total_dc_budget / n_dc)
        # init_eta = np.zeros(n_dc)
        # init_eta[0] = total_dc_budget
        print(init_eta)
        eta = {"dc_capacity": init_eta}

        # Create objectives
        inv_obj = zap.planning.InvestmentObjective(pypsa_devices_dc, xstar)
        if op_obj_selector == "dispatch":
            op_obj = zap.planning.DispatchCostObjective(pypsa_net, pypsa_devices_dc)
        elif op_obj_selector == "lmp":
            op_obj = zap.planning.LMPObjective(pypsa_net, pypsa_devices_dc, lmp_metric=lmp_metric, lmp_beta = lmp_beta)
        elif op_obj_selector == "line_util":
            line_device_idx = 3  # ACLine index in devices
            line_idx = crit_idx      # all lines
            thr = 0.90
            use_mean_over_time = False
            op_obj = zap.planning.LineOverloadObjective(
                devices=pypsa_devices_dc,
                line_device_idx=line_device_idx,
                line_idx=line_idx,
                thr=thr,
                use_mean_over_time=use_mean_over_time
            )
        elif op_obj_selector == "delta_line_util":
            line_device_idx = 3  # ACLine index in devices
            line_idx = crit_idx      # critical lines from single-node case
            base_line_util = base_line_util
            thr = 0.90
            use_mean_over_time = False
            op_obj = zap.planning.LineDeltaOverloadObjective(
                devices=pypsa_devices_dc,
                line_device_idx=line_device_idx,
                line_idx=line_idx,
                base_line_util=base_line_util,
                thr=thr,
                use_mean_over_time=use_mean_over_time
            )

        P = zap.planning.PlanningProblem(
            operation_objective=op_obj,
            investment_objective=inv_obj,
            layer=xstar,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )

        # P.extra_projections = {
        #     "dc_capacity": zap.planning.SimplexBudgetProjection(
        #         budget=total_dc_budget, strict=True
        #     )
        # }
        P.extra_projections = {
            "dc_capacity": zap.planning.BoxBudgetProjection(
                budget=total_dc_budget, lower_bounds=np.full(n_dc, dc_lower_bound), 
                upper_bounds=np.full(n_dc, dc_upper_bound)
            )
        }

        cost = P(**eta, requires_grad=True)
        grad = P.backward()

        state = P.solve(num_iterations=num_iters, initial_state=eta)


        return state, P
    return (run_planning_experiment,)


@app.cell
def _(CAPITAL_COSTS, node_price_summaries, np):
    def compute_metrics(distributed_outcome, planned_dc_capacities, capital_costs = CAPITAL_COSTS):

        metrics = {}
        num_nodes = len(planned_dc_capacities)

        # DC Land Cost
        metrics["dc_land_cost"] = np.dot(1000*planned_dc_capacities, capital_costs[:num_nodes])

        # Mean Max LMP
        dist_stats = node_price_summaries(distributed_outcome.prices, topk=5)
        mean_max_lmp = np.mean(dist_stats["max"])
        metrics["mean_max_lmp"] = mean_max_lmp * 100.0 * 4.0

        # Dispatch Cost
        metrics["dispatch"] = distributed_outcome.problem.value * 100.0

        # Mean Max Line Dual
        mu_sum = distributed_outcome.local_inequality_duals[3][0] + distributed_outcome.local_inequality_duals[3][1]
        mean_max_line_dual = np.mean(mu_sum.max(axis=1))
        metrics["mean_max_line_dual"] = mean_max_line_dual

        return metrics
    return (compute_metrics,)


@app.cell
def _(mo):
    mo.md(r"""## Site Ablation""")
    return


@app.cell
def _(
    CAPITAL_COSTS,
    GEN_SCALING_FACTOR,
    INVESTMENT_NODE_CANDS,
    LINE_SCALING_FACTOR,
    LOAD_SCALING_FACTOR,
    compute_metrics,
    cp,
    create_planning_devices,
    np,
    pd,
    run_planning_experiment,
):
    ablation_exp_params_dict = {
        "num_nodes": np.arange(1,14),
        "total_dc_budgets": np.array([1.0]),
        "num_iters": 30
    }

    def run_ablation_exp(pypsa_net,
                         pypsa_devices,
                         ablation_exp_params_dict,
                         dc_lower_bound_scalar=0.5,
                         dc_upper_bound_scalar=2.5):
        num_nodes_list = ablation_exp_params_dict["num_nodes"]
        total_dc_budgets = ablation_exp_params_dict["total_dc_budgets"]
        num_iters = ablation_exp_params_dict["num_iters"]

        results = []

        for num_node in num_nodes_list:
            for total_dc_budget in total_dc_budgets:
                print(f"Running: num_nodes={num_node}, budget={total_dc_budget} GW")

                # Compute bounds
                uniform_cap_level = total_dc_budget / num_node
                # dc_lower_bound = max(dc_lower_bound_scalar * uniform_cap_level, 0.05)
                dc_lower_bound = 0.05
                # dc_upper_bound = dc_upper_bound_scalar * uniform_cap_level
                dc_upper_bound = min(dc_upper_bound_scalar * uniform_cap_level, 1.0)

                # Create devices for planning (zero capital cost during optimization)
                planning_devices_params_dict = {
                    "num_nodes": num_node,
                    "investment_node_cands": INVESTMENT_NODE_CANDS,
                    "gen_scaling_factor": GEN_SCALING_FACTOR,
                    "load_scaling_factor": LOAD_SCALING_FACTOR,
                    "line_scaling_factor": LINE_SCALING_FACTOR,
                    "dc_nominal_capacity": 1,
                    "capital_costs": 0 * CAPITAL_COSTS,
                    "workload_profile": "development/load_profiles/example_inference_azure_conv.csv",
                    "pypsa_net": pypsa_net,
                    "pypsa_devices": pypsa_devices,
                }
                pypsa_devices_dc = create_planning_devices(pypsa_devices, planning_devices_params_dict)

                # Run planning
                planning_exp_params_dict = {
                    "total_dc_budget": total_dc_budget,
                    "dc_lower_bound": dc_lower_bound,
                    "dc_upper_bound": dc_upper_bound,
                    "op_obj_selector": "lmp",
                    "lmp_metric": "sumsmoothmax",
                    "lmp_beta": 1000.0,
                    "num_iters": num_iters
                }
                planning_state, P = run_planning_experiment(pypsa_net, pypsa_devices_dc, planning_exp_params_dict)

                # Simulate with optimized capacities
                planned_dc_capacities = planning_state[0]["dc_capacity"]

                # Do bin and round here

            
                sim_params = {
                    "num_nodes": num_node,
                    "investment_node_cands": INVESTMENT_NODE_CANDS,
                    "gen_scaling_factor": GEN_SCALING_FACTOR,
                    "load_scaling_factor": LOAD_SCALING_FACTOR,
                    "line_scaling_factor": LINE_SCALING_FACTOR,
                    "dc_nominal_capacity": planned_dc_capacities,
                    "capital_costs": CAPITAL_COSTS,
                    "workload_profile": "development/load_profiles/example_inference_azure_conv.csv",
                    "pypsa_net": pypsa_net,
                    "pypsa_devices": pypsa_devices,
                }
                planned_devices = create_planning_devices(pypsa_devices, sim_params)
                outcome = pypsa_net.dispatch(planned_devices, time_horizon=96, solver=cp.CLARABEL, add_ground=False)

                # Compute and store results
                metrics = compute_metrics(outcome, planned_dc_capacities)
                row = {
                    "num_nodes": num_node,
                    "total_dc_budget_gw": total_dc_budget,
                    "dc_lower_bound": dc_lower_bound,
                    "dc_upper_bound": dc_upper_bound,
                    "planned_dc_capacities": ",".join(f"{x:.4f}" for x in planned_dc_capacities),
                    "total_allocation": np.sum(planned_dc_capacities),
                    **metrics
                }
                results.append(row)

        return pd.DataFrame(results)
    return ablation_exp_params_dict, run_ablation_exp


@app.cell
def _(np, pypsa_devices, pypsa_net, run_ablation_exp):
    # Site Ablation Experiment

    site_ablation_exp_params = {
        "num_nodes": np.arange(1, 3),
        "total_dc_budgets": np.array([1.0]),
        "num_iters": 10
    }
    site_results = run_ablation_exp(pypsa_net, pypsa_devices, site_ablation_exp_params)
    return site_ablation_exp_params, site_results


@app.cell
def _(site_results):
    site_results
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

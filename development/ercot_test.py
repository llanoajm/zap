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
def _(os, pypsa):
    HOME_PATH = os.environ.get("HOME")
    PYPSA_NETW0RK_PATH = (
        HOME_PATH + "/zap_data/pypsa-networks/ercot_small/network_2023.nc"
    )
    pn = pypsa.Network(PYPSA_NETW0RK_PATH)
    snapshots = pn.generators_t.p_max_pu.index
    snapshot_data = snapshots[5616:5640]  # 8/23/21 # hourly
    return HOME_PATH, PYPSA_NETW0RK_PATH, pn, snapshot_data, snapshots


@app.cell
def _(load_pypsa_network, pn, snapshot_data, upsample_zap_devices):
    pypsa_kwargs = {}
    pypsa_net, pypsa_devices = load_pypsa_network(
        pn, snapshot_data, power_unit=1.0e3, cost_unit=100.0, scale_line_capacity_factor=0.5, **pypsa_kwargs
    )
    # Drop empty DC lines
    pypsa_devices_new = pypsa_devices[0:2] + pypsa_devices[3:]
    pypsa_devices = upsample_zap_devices(pypsa_devices_new, factor=4, original_timesteps=24)
    return pypsa_devices, pypsa_devices_new, pypsa_kwargs, pypsa_net


@app.cell
def _(cp, deepcopy, pypsa_devices, pypsa_net):
    pypsa_devices_base = deepcopy(pypsa_devices)
    outcome_base = pypsa_net.dispatch(pypsa_devices_base, time_horizon=96, solver=cp.CLARABEL, add_ground=False)
    return outcome_base, pypsa_devices_base


@app.cell
def _(outcome_base):
    outcome_base.problem.value
    return


@app.cell
def _():
    SEED = 42
    N_CANDIDATES = 5
    TOTAL_DC_GW = 2.0
    TIME_HORIZON = 96
    NUM_ITERS = 100
    return NUM_ITERS, N_CANDIDATES, SEED, TIME_HORIZON, TOTAL_DC_GW


@app.cell
def _(np):
    def node_price_summaries(prices, topk=5, q=(0.95, 0.99)):
        sorted_prices = np.sort(prices, axis=1)[:, ::-1]
        return {
            "p95": np.quantile(prices, q[0], axis=1),
            "p99": np.quantile(prices, q[1], axis=1),
            "mean": np.mean(prices, axis=1),
            "mean_top5": np.mean(sorted_prices[:, :topk], axis=1),
            "max": np.max(prices, axis=1),
        }

    def compute_metrics(outcome, dc_capacities, ac_line_idx):
        mu_sum = (
            outcome.local_inequality_duals[ac_line_idx][0]
            + outcome.local_inequality_duals[ac_line_idx][1]
        )
        price_stats = node_price_summaries(outcome.prices)
        return {
            "dispatch": outcome.problem.value * 100.0,
            "mean_max_lmp": float(np.mean(price_stats["max"]) * 100.0 * 4.0),
            "mean_max_line_dual": float(np.mean(mu_sum.max(axis=1))),
            "dc_capacities_mw": dc_capacities * 1000,
        }
    return compute_metrics, node_price_summaries


@app.cell
def _(N_CANDIDATES, SEED, np, pypsa_net):
    rng = np.random.default_rng(SEED)
    candidate_nodes = rng.choice(pypsa_net.num_nodes, size=N_CANDIDATES, replace=False)
    return candidate_nodes, rng


@app.cell
def _(
    ACLine,
    TIME_HORIZON,
    TOTAL_DC_GW,
    candidate_nodes,
    deepcopy,
    np,
    pypsa_devices,
    pypsa_net,
    zap,
):
    n_dc = len(candidate_nodes)
    workload = np.ones(TIME_HORIZON)
    VOLL = 1e6

    pypsa_devices_dc = deepcopy(pypsa_devices)
    dcloads = zap.DataCenterLoad(
        num_nodes=pypsa_net.num_nodes,
        terminal=candidate_nodes,
        profiles=[workload] * n_dc,
        nominal_capacity=np.full(n_dc, TOTAL_DC_GW / n_dc),
        linear_cost=np.full(n_dc, VOLL),
        settime_horizon=float(TIME_HORIZON),
        capital_cost=np.zeros(n_dc),
    )
    pypsa_devices_dc.append(dcloads)
    DC_IDX = len(pypsa_devices_dc) - 1
    AC_LINE_IDX = next(i for i, d in enumerate(pypsa_devices_dc) if isinstance(d, ACLine))
    return AC_LINE_IDX, DC_IDX, VOLL, dcloads, n_dc, pypsa_devices_dc, workload


@app.cell
def _(cp, np, zap):
    def run_planning_experiment(pypsa_net, pypsa_devices_dc, dc_idx, total_dc_gw, num_iters, time_horizon):
        n_dc = len(pypsa_devices_dc[dc_idx].terminals)

        xstar = zap.DispatchLayer(
            pypsa_net,
            pypsa_devices_dc,
            parameter_names={"dc_capacity": (dc_idx, "nominal_capacity")},
            time_horizon=time_horizon,
            solver=cp.CLARABEL,
        )
        lower_bounds = {"dc_capacity": np.zeros(n_dc)}
        upper_bounds = {"dc_capacity": np.full(n_dc, total_dc_gw)}
        init_eta = {"dc_capacity": np.full(n_dc, total_dc_gw / n_dc)}

        op_obj = zap.planning.DispatchCostObjective(pypsa_net, pypsa_devices_dc)
        inv_obj = zap.planning.InvestmentObjective(pypsa_devices_dc, xstar)

        P = zap.planning.PlanningProblem(
            operation_objective=op_obj,
            investment_objective=inv_obj,
            layer=xstar,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )
        P.extra_projections = {
            "dc_capacity": zap.planning.BoxBudgetProjection(
                budget=total_dc_gw,
                lower_bounds=np.zeros(n_dc),
                upper_bounds=np.full(n_dc, total_dc_gw),
            )
        }
        state = P.solve(num_iterations=num_iters, initial_state=init_eta)
        return state, xstar
    return (run_planning_experiment,)


@app.cell
def _(
    DC_IDX,
    NUM_ITERS,
    TIME_HORIZON,
    TOTAL_DC_GW,
    pypsa_devices_dc,
    pypsa_net,
    run_planning_experiment,
):
    plan_state, xstar = run_planning_experiment(
        pypsa_net, pypsa_devices_dc, DC_IDX, TOTAL_DC_GW, NUM_ITERS, TIME_HORIZON
    )
    dc_capacity_opt = plan_state[0]["dc_capacity"]
    return dc_capacity_opt, plan_state, xstar


@app.cell
def _(AC_LINE_IDX, compute_metrics, dc_capacity_opt, xstar):
    outcome_dist = xstar(dc_capacity=dc_capacity_opt)
    metrics_dist = compute_metrics(outcome_dist, dc_capacity_opt, AC_LINE_IDX)
    return metrics_dist, outcome_dist


@app.cell
def _(DC_IDX, TIME_HORIZON, cp, pypsa_devices_dc, pypsa_net, zap):
    xstar2 = zap.DispatchLayer(
            pypsa_net,
            pypsa_devices_dc,
            parameter_names={"dc_capacity": (DC_IDX, "nominal_capacity")},
            time_horizon=TIME_HORIZON,
            solver=cp.CLARABEL,
        )
    return (xstar2,)


@app.cell
def _(AC_LINE_IDX, TOTAL_DC_GW, compute_metrics, n_dc, np, xstar2):
    single_outcomes, single_metrics = [], []
    for _i in range(n_dc):
        _u = np.zeros(n_dc)
        _u[_i] = TOTAL_DC_GW
        try:
                _outcome_i = xstar2(dc_capacity=_u)
                single_outcomes.append(_outcome_i)
                single_metrics.append(compute_metrics(_outcome_i, _u, AC_LINE_IDX))
        except AssertionError:
            print("Infeasible!")
            dummy_dict = {"dispatch": np.inf,
                          "mean_max_lmp": np.inf,
                          "mean_max_line_dual": np.inf,
                          "dc_capacities_mw": _u*1000.0,  
                        }
            single_metrics.append(dummy_dict)

    # Now run the distirubted case
    _u = np.full(n_dc, TOTAL_DC_GW/n_dc)
    try:
        _outcome_i = xstar2(dc_capacity=_u)
        single_outcomes.append(_outcome_i)
        single_metrics.append(compute_metrics(_outcome_i, _u, AC_LINE_IDX))
    except AssertionError:
        print("Infeasible!")
        dummy_dict = {"dispatch": np.inf,
                      "mean_max_lmp": np.inf,
                      "mean_max_line_dual": np.inf,
                      "dc_capacities_mw": _u*1000.0,  
                    }
        single_metrics.append(dummy_dict)


    best_idx = int(np.argmin([m["dispatch"] for m in single_metrics]))
    return best_idx, dummy_dict, single_metrics, single_outcomes


@app.cell
def _(single_metrics):
    single_metrics
    return


@app.cell
def _():
    # _fig, _axes = plt.subplots(1, 4, figsize=(14, 4))

    # # 1. Optimized allocation
    # _ax = _axes[0]
    # _ax.bar(range(n_dc), metrics_dist["dc_capacities_mw"])
    # _ax.axhline(1000 / n_dc, color="gray", linestyle="--", label="Uniform")
    # _ax.set_xticks(range(n_dc))
    # _ax.set_xticklabels([f"Node\n{n}" for n in candidate_nodes], fontsize=8)
    # _ax.set_ylabel("MW")
    # _ax.set_title("Optimized Allocation")
    # _ax.legend()

    # # 2. Dispatch cost comparison
    # _ax = _axes[1]
    # baseline = metrics_dist["dispatch"]
    # _dispatch_rel = [m["dispatch"]/baseline for m in single_metrics] + [metrics_dist["dispatch"]/baseline]
    # _labels = [f"Node {candidate_nodes[i]}" for i in range(n_dc)] + ["Uniform"] + ["Distributed"]
    # _dispatch = [m["dispatch"] for m in single_metrics] + [metrics_dist["dispatch"]]
    # _colors = ["steelblue"] * n_dc + ["darkorange"] + ["purple"]
    # _colors[best_idx] = "green"
    # _ax.set_ylim(0,1e6)
    # _ax.bar(_labels, _dispatch, color=_colors)
    # _ax.set_ylabel("Dispatch Cost ($)")
    # _ax.set_title("Single-node vs Distributed")
    # _ax.tick_params(axis="x", rotation=30)


    # # 3. Mean max LMP comparison
    # _ax = _axes[2]
    # _lmps = [m["mean_max_lmp"] for m in single_metrics] + [metrics_dist["mean_max_lmp"]]
    # _ax.set_ylim(0,1e4)
    # _ax.bar(_labels, _lmps, color=_colors)
    # _ax.set_ylabel("Mean Max LMP ($/MWh)")
    # _ax.set_title("Price Pressure")
    # _ax.tick_params(axis="x", rotation=30)

    # # 3. Mean max Line Dual comparison
    # _ax = _axes[3]
    # _lmps = [m["mean_max_line_dual"] for m in single_metrics] + [metrics_dist["mean_max_line_dual"]]
    # _ax.set_ylim(0,1e4)
    # _ax.bar(_labels, _lmps, color=_colors)
    # _ax.set_ylabel("Mean Max Line Dual")
    # _ax.set_title("Congestion Metric")
    # _ax.tick_params(axis="x", rotation=30)


    # _fig.tight_layout()
    # _fig
    return


@app.cell
def _(best_idx, candidate_nodes, metrics_dist, single_metrics):
    print("=== Optimized DC Allocation ===")
    for _i, (_node, _cap) in enumerate(zip(candidate_nodes, metrics_dist["dc_capacities_mw"])):
        print(f"  Node {_node}: {_cap:.1f} MW")

    _best = single_metrics[best_idx]
    print(f"\nBest single node (Node {candidate_nodes[best_idx]}):")
    print(f"  Dispatch: {_best['dispatch']:.2f},  Mean Max LMP: {_best['mean_max_lmp']:.2f}")
    print(f"Distributed:")
    print(f"  Dispatch: {metrics_dist['dispatch']:.2f},  Mean Max LMP: {metrics_dist['mean_max_lmp']:.2f}")
    _improvement = (_best["dispatch"] - metrics_dist["dispatch"]) / abs(_best["dispatch"]) * 100
    print(f"\nDispatch improvement from distributing: {_improvement:.2f}%")
    return


@app.cell
def _(
    DC_IDX,
    N_CANDIDATES,
    TIME_HORIZON,
    TOTAL_DC_GW,
    cp,
    np,
    pypsa_devices_dc,
    pypsa_net,
    zap,
):
    xstar_b = zap.DispatchLayer(
        pypsa_net,
        pypsa_devices_dc,
        parameter_names={"dc_capacity": (DC_IDX, "nominal_capacity")},
        time_horizon=TIME_HORIZON,
        solver=cp.CLARABEL,
    )
    benders = zap.planning.BendersSolver(
        layer=xstar_b,
        capital_cost=np.zeros(N_CANDIDATES),
        budget=TOTAL_DC_GW,
        lower_bounds={"dc_capacity": np.zeros(N_CANDIDATES)},
        upper_bounds={"dc_capacity": np.full(N_CANDIDATES, TOTAL_DC_GW)},
        solver=cp.CLARABEL,
        dispatch_scalar=1.0,
    )
    benders_result = benders.solve(
        initial_u=np.full(N_CANDIDATES, TOTAL_DC_GW / N_CANDIDATES),
        max_iter=50,
        tol=1e-4,
        verbose=True,
    )
    dc_capacity_benders = benders_result["u"]
    return benders, benders_result, dc_capacity_benders, xstar_b


@app.cell
def _(dc_capacity_benders):
    dc_capacity_benders
    return


@app.cell
def _(AC_LINE_IDX, compute_metrics, dc_capacity_benders, xstar_b):
    outcome_benders = xstar_b(dc_capacity=dc_capacity_benders)
    metrics_benders = compute_metrics(outcome_benders, dc_capacity_benders, AC_LINE_IDX)
    return metrics_benders, outcome_benders


@app.cell
def _(outcome_benders):
    outcome_benders
    return


@app.cell
def _(best_idx, candidate_nodes, metrics_benders, n_dc, plt, single_metrics):
    _fig, _axes = plt.subplots(1, 4, figsize=(14, 4))

    # 1. Optimized allocation
    _ax = _axes[0]
    _ax.bar(range(n_dc), metrics_benders["dc_capacities_mw"])
    _ax.axhline(1000 / n_dc, color="gray", linestyle="--", label="Uniform")
    _ax.set_xticks(range(n_dc))
    _ax.set_xticklabels([f"Node\n{n}" for n in candidate_nodes], fontsize=8)
    _ax.set_ylabel("MW")
    _ax.set_title("Optimized Allocation")
    _ax.legend()

    # 2. Dispatch cost comparison
    _ax = _axes[1]
    baseline = metrics_benders["dispatch"]
    _dispatch_rel = [m["dispatch"]/baseline for m in single_metrics] + [metrics_benders["dispatch"]/baseline]
    _labels = [f"Node {candidate_nodes[i]}" for i in range(n_dc)] + ["Uniform"] + ["Distributed"]
    _dispatch = [m["dispatch"] for m in single_metrics] + [metrics_benders["dispatch"]]
    _colors = ["steelblue"] * n_dc + ["darkorange"] + ["purple"]
    _colors[best_idx] = "green"
    _ax.set_ylim(0,1e6)
    _ax.bar(_labels, _dispatch, color=_colors)
    _ax.set_ylabel("Dispatch Cost ($)")
    _ax.set_title("Single-node vs Distributed")
    _ax.tick_params(axis="x", rotation=90)


    # 3. Mean max LMP comparison
    _ax = _axes[2]
    _lmps = [m["mean_max_lmp"] for m in single_metrics] + [metrics_benders["mean_max_lmp"]]
    _ax.set_ylim(0,1e4)
    _ax.bar(_labels, _lmps, color=_colors)
    _ax.set_ylabel("Mean Max LMP ($/MWh)")
    _ax.set_title("Price Pressure")
    _ax.tick_params(axis="x", rotation=90)

    # 3. Mean max Line Dual comparison
    _ax = _axes[3]
    _lmps = [m["mean_max_line_dual"] for m in single_metrics] + [metrics_benders["mean_max_line_dual"]]
    _ax.set_ylim(0,1e4)
    _ax.bar(_labels, _lmps, color=_colors)
    _ax.set_ylabel("Mean Max Line Dual")
    _ax.set_title("Congestion Metric")
    _ax.tick_params(axis="x", rotation=90)


    _fig.tight_layout()
    _fig
    return (baseline,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

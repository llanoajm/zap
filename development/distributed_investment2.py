import marimo

__generated_with = "0.19.4"
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
    return cp, deepcopy, load_pypsa_network, np, os, pd, plt, pypsa, zap


@app.function
def upsample_zap_devices(devices, factor=4, original_timesteps=24):
    """Upsample time-varying attributes of zap devices by repeating each timestep."""
    upsampled_zap_devices = []
    for dev in devices:
        upsampled_dev = dev.sample_time(original_timesteps*factor, original_timesteps)
        upsampled_zap_devices.append(upsampled_dev)

    return upsampled_zap_devices
        # for attr in ['dynamic_capacity', 'load', 'linear_cost']:
        #     if hasattr(dev, attr):
        #         val = getattr(dev, attr)
        #         if val is not None and val.ndim == 2 and val.shape[1] == original_timesteps:
        #             setattr(dev, attr, np.repeat(val, factor, axis=1))


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
    return pn, snapshot_data


@app.cell
def _(load_pypsa_network, pn, snapshot_data):
    pypsa_kwargs = {}
    pypsa_net, pypsa_devices = load_pypsa_network(
        pn, snapshot_data, power_unit=1.0e3, cost_unit=100.0, **pypsa_kwargs
    )

    pypsa_devices = upsample_zap_devices(pypsa_devices, factor=4, original_timesteps=24)
    return pypsa_devices, pypsa_net


@app.cell
def _(plt, pypsa_devices):
    plt.plot(pypsa_devices[1].load[:, 0])
    plt.plot(pypsa_devices[1].load[:, 90])
    return


@app.cell
def _(pypsa_devices):
    pypsa_devices[1].load.sum(axis=0).max() * 1.2
    return


@app.cell
def _(pypsa_devices):
    pypsa_devices[1].load.shape
    return


@app.cell
def _(pypsa_devices):
    pypsa_devices[1].load[0].sum()
    return


@app.cell
def _(num_terminals):
    num_terminals
    return


@app.cell
def _(cp, deepcopy, np, pypsa_devices, pypsa_net, zap):
    # Experiment to check knees of all terminals in WECC
    # We will only check 0GW, 1GW, 2 GW, 5GW
    # consider the metric max. Each terminal should have a [1x4 array]

    # dc_caps = [0, 1, 2]
    dc_caps = np.random.random(10) * 5.0
    # dc_caps = [30]
    LOAD_SCALING_FACTOR = 1.27
    GEN_SCALING_FACTOR = 1.24
    # GEN_SCALING_FACTOR = 1
    LINE_SCALING_FACTOR = 0.7

    max_results = np.zeros((pypsa_net.num_nodes, len(dc_caps))) # [num_nodes, 4 DC caps]
    mean_results = np.zeros((pypsa_net.num_nodes, len(dc_caps)))
    dispatch_cost_results = np.zeros((pypsa_net.num_nodes, len(dc_caps)))
    n_dc = 1

    pypsa_devices_base = deepcopy(pypsa_devices)

    # Scale load, gen, and line capacities
    pypsa_devices_base[1].load *= LOAD_SCALING_FACTOR
    pypsa_devices_base[0].dynamic_capacity *= GEN_SCALING_FACTOR
    pypsa_devices_base[3].nominal_capacity *= LINE_SCALING_FACTOR
    pypsa_devices_base[3].nominal_capacity[168] = 0.5
    pypsa_devices_base[3].nominal_capacity[176] = 0.5
    pypsa_devices_base[3].nominal_capacity[49] = 0.3
    outcome_base = pypsa_net.dispatch(pypsa_devices_base, time_horizon=96, solver=cp.CLARABEL, add_ground=False)

    dispatch_cost_base = outcome_base.problem.value
    print(dispatch_cost_base)
    prices = outcome_base.prices  # (n_nodes, T)
    prices_base = outcome_base.prices
    dispatch_cost = outcome_base.problem.value

    dispatch_diff = dispatch_cost - dispatch_cost_base

    # system summaries
    sys_mean = float(np.mean(prices))
    sys_max  = float(np.max(prices))
    sys_spread_mean = float(np.mean(np.max(prices, axis=0) - np.min(prices, axis=0)))
    sys_p95_nodes_mean_t = float(np.mean(np.quantile(prices, 0.95, axis=0)))

    print(sys_max)
    num_terminals = pypsa_net.num_nodes

    results = []


    for terminal in range(num_terminals):
        dc_terminals = np.array([terminal])
        print(f'solving for terminal {terminal}')
        dc_caps = np.random.random(10) * 5.0
        for cap_idx, dc_cap in enumerate(dc_caps):
            dc_cap = np.round(dc_cap, 2)
            pypsa_devices_dc = deepcopy(pypsa_devices_base)
            print(f'adding {dc_cap}GW')
            entry = {"terminal": terminal, "capacity (GW)": dc_cap}
            # if dc_cap == 0: # no need to repeatedly solve base case
            #     node_price_t = outcome_base.prices[terminal, :]
            #     max_node_price = np.max(node_price_t)
            #     mean_node_price = np.mean(node_price_t)
            #     dispatch_cost = outcome_base.problem.value
            #     print(f'max node price over time: {max_node_price}')
            #     print(f'mean node price over time: {mean_node_price}')
            #     print(f'dispatch cost: {dispatch_cost}')
            #     max_results[terminal, cap_idx] = max_node_price
            #     mean_results[terminal, cap_idx] = mean_node_price
            #     dispatch_cost_results[terminal, cap_idx] = dispatch_cost
            #     continue
            # Make data center
            dcloads = zap.DataCenterLoad(
                num_nodes=pypsa_net.num_nodes,
                terminal=dc_terminals,
                profiles=n_dc*["development/load_profiles/dummy_peak_provision.csv"],
                nominal_capacity=dc_cap * np.ones((n_dc)),
                linear_cost=np.ones(n_dc) * 0,
                settime_horizon=96,
                capital_cost=0*np.ones(n_dc),
            )
            pypsa_devices_dc.append(dcloads)

            # Solve dispatch
            try:
                outcome_test = pypsa_net.dispatch(
                    pypsa_devices_dc, time_horizon=96, solver=cp.CLARABEL, add_ground=False
                )
                prices = outcome_test.prices  # (n_nodes, T)
                prices_base = outcome_base.prices
                dispatch_cost = outcome_test.problem.value

                dispatch_diff = dispatch_cost - dispatch_cost_base
            
                # local
                inj = terminal
                inj_max = float(np.max(prices[inj, :]))
                inj_mean = float(np.mean(prices[inj, :]))
            
                # system summaries
                sys_mean = float(np.mean(prices))
                sys_max  = float(np.max(prices))
                sys_spread_mean = float(np.mean(np.max(prices, axis=0) - np.min(prices, axis=0)))
                sys_p95_nodes_mean_t = float(np.mean(np.quantile(prices, 0.95, axis=0)))
            
                # deltas vs base
                d_sys_mean = sys_mean - float(np.mean(prices_base))
                d_sys_p95  = sys_p95_nodes_mean_t - float(np.mean(np.quantile(prices_base, 0.95, axis=0)))
                d_sys_max  = sys_max - float(np.max(prices_base))
                d_inj_max  = inj_max - float(np.max(prices_base[inj, :]))
                d_inj_mean = inj_mean - float(np.mean(prices_base[inj, :]))
            
                entry.update({
                    "inj_max_lmp": inj_max,
                    "inj_mean_lmp": inj_mean,
                    "sys_mean_lmp": sys_mean,
                    "sys_p95_nodes_mean_t": sys_p95_nodes_mean_t,
                    "sys_max_lmp": sys_max,
                    "sys_spread_mean": sys_spread_mean,
                    "d_sys_mean": d_sys_mean,
                    "d_sys_p95": d_sys_p95,
                    "d_sys_max": d_sys_max,
                    "d_inj_max": d_inj_max,
                    "d_inj_mean": d_inj_mean,
                    "dispatch": dispatch_cost,
                    "dispatch_diff": dispatch_diff
                })
            except Exception as e:
                print('infeasible!')
                entry.update({
                    "inj_max_lmp": np.nan,
                    "inj_mean_lmp": np.nan,
                    "sys_mean_lmp": np.nan,
                    "sys_p95_nodes_mean_t": np.nan,
                    "sys_max_lmp": np.nan,
                    "sys_spread_mean": np.nan,
                    "d_sys_mean": np.nan,
                    "d_sys_p95": np.nan,
                    "d_sys_max": np.nan,
                    "d_inj_max": np.nan,
                    "d_inj_mean": np.nan,
                    "dispatch": np.nan,
                    "dispatch_diff": np.nan
                })
            print(entry)
            results.append(entry)
    return num_terminals, outcome_test, results


@app.cell
def _(outcome_test):
    outcome_test.problem.value
    return


@app.cell
def _(outcome_test):
    outcome_test.local_inequality_duals[5][0].shape
    return


@app.cell
def _(outcome_test, plt):
    plt.plot(outcome_test.power[5][0][0])
    return


@app.cell
def _(outcome_test):
    outcome_test.power[5][0]
    return


@app.cell
def _(pd, results):
    df = pd.DataFrame(results)
    df.to_csv("cloud_congestion.csv")
    return


if __name__ == "__main__":
    app.run()

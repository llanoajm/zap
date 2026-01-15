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
            # for attr in ['dynamic_capacity', 'load', 'linear_cost']:
            #     if hasattr(dev, attr):
            #         val = getattr(dev, attr)
            #         if val is not None and val.ndim == 2 and val.shape[1] == original_timesteps:
            #             setattr(dev, attr, np.repeat(val, factor, axis=1))
    return (upsample_zap_devices,)


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
def _(load_pypsa_network, pn, snapshot_data, upsample_zap_devices):
    pypsa_kwargs = {}
    pypsa_net, pypsa_devices = load_pypsa_network(
        pn, snapshot_data, power_unit=1.0e3, cost_unit=100.0, **pypsa_kwargs
    )

    pypsa_devices = upsample_zap_devices(pypsa_devices, factor=4, original_timesteps=24)
    return pypsa_devices, pypsa_kwargs, pypsa_net


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
    dc_caps = np.random.random(10) * 2.0
    load_scaling = 1.05
    # dc_caps = [30]

    max_results = np.zeros((pypsa_net.num_nodes, len(dc_caps))) # [num_nodes, 4 DC caps]
    mean_results = np.zeros((pypsa_net.num_nodes, len(dc_caps)))
    dispatch_cost_results = np.zeros((pypsa_net.num_nodes, len(dc_caps)))
    n_dc = 1

    pypsa_devices_base = deepcopy(pypsa_devices)
    pypsa_devices_base[1].load *= load_scaling
    outcome_base = pypsa_net.dispatch(pypsa_devices_base, time_horizon=96, solver=cp.CLARABEL, add_ground=False)

    num_terminals = pypsa_net.num_nodes

    results = []


    for terminal in range(num_terminals):
        dc_terminals = np.array([terminal])
        print(f'solving for terminal {terminal}')
        dc_caps = np.random.random(10) * 2.0
        for cap_idx, dc_cap in enumerate(dc_caps):
            dc_cap = np.round(dc_cap, 2)
            pypsa_devices_dc = deepcopy(pypsa_devices)
            pypsa_devices_dc[1].load *= load_scaling
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
                node_price_t = outcome_test.prices[terminal, :]
                max_node_price = np.max(node_price_t)
                mean_node_price = np.mean(node_price_t)
                min_node_price = np.min(node_price_t)
                dispatch_cost = outcome_test.problem.value
                print(f'max node price over time: {max_node_price}')
                print(f'mean node price over time: {mean_node_price}')
                print(f'dispatch cost: {dispatch_cost}')
                entry.update({"Min": min_node_price,
                              "Mean": mean_node_price,
                              "Max": max_node_price,
                              "P10": np.quantile(node_price_t, 0.1), 
                              "P50": np.quantile(node_price_t, 0.5),
                              "P90": np.quantile(node_price_t, 0.9),
                              "P99": np.quantile(node_price_t, 0.99),
                              "Variance": np.var(node_price_t),
                              "Dispatch": dispatch_cost
                             })
                # max_results[terminal, cap_idx] = max_node_price
                # mean_results[terminal, cap_idx] = mean_node_price
                # dispatch_cost_results[terminal, cap_idx] = dispatch_cost
            except Exception as e:
                print('infeasible!')
                max_results[terminal, cap_idx] = np.nan
                mean_results[terminal, cap_idx] = np.nan
                dispatch_cost_results[terminal, cap_idx] = np.nan
                entry.update({"Min": np.nan,
                              "Mean": np.nan,
                              "Max": np.nan,
                              "P10": np.nan, 
                              "P50": np.nan,
                              "P90": np.nan,
                              "P99": np.nan,
                              "Variance": np.nan,
                              "Dispatch": np.nan
                             })
            results.append(entry)
            print(results)
    return (
        cap_idx,
        dc_cap,
        dc_caps,
        dc_terminals,
        dcloads,
        dispatch_cost,
        dispatch_cost_results,
        entry,
        load_scaling,
        max_node_price,
        max_results,
        mean_node_price,
        mean_results,
        min_node_price,
        n_dc,
        node_price_t,
        num_terminals,
        outcome_base,
        outcome_test,
        pypsa_devices_base,
        pypsa_devices_dc,
        results,
        terminal,
    )


@app.cell
def _(outcome_test):
    outcome_test.problem.value
    return


@app.cell
def _(outcome_test):
    outcome_test.local_inequality_duals[5][0].shape
    return


@app.cell
def _(node_price_t, plt):
    plt.plot(node_price_t)
    return


@app.cell
def _(node_price_t, plt):
    plt.plot(node_price_t[50:])
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
    return (df,)


if __name__ == "__main__":
    app.run()

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
        """Upsample time-varying attributes of zap devices by repepypsa_kwargs = {}
    pypsa_net, pypsa_devices = load_pypsa_network(
        pn, snapshot_data, power_unit=1.0e3, cost_unit=100.0, **pypsa_kwargs
    )

    pypsa_devices = upsample_zap_devices(pypsa_devices, factor=4, original_timesteps=24)ting each timestep."""
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
        HOME_PATH + "/zap_data/pypsa-networks/western_small/network_2023.nc"
    )
    pn = pypsa.Network(PYPSA_NETW0RK_PATH)
    snapshots = pn.generators_t.p_max_pu.index
    # snapshot_data = snapshots[5616:5640]  # 8/23/23 # hourly
    snapshot_data = snapshots[5448:5472]  # 8/16/23 # hourly
    # snapshot_data = snapshots[5448:5640] # 8/16/23-8/23/23
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
def _(deepcopy, np, pypsa_devices, pypsa_net, zap):
    load_scaling = 1.05
    pypsa_devices_dc = deepcopy(pypsa_devices)
    pypsa_devices_dc[1].load *= load_scaling
    dc_terminals = np.array([15, 25, 33, 12, 34, 58, 66, 14, 35, 45])
    n_dc = len(dc_terminals)
    capital_cost = np.ones(n_dc) * 0

    dcloads = zap.DataCenterLoad(
        num_nodes=pypsa_net.num_nodes,
        terminal=dc_terminals,
        profiles=n_dc*["development/load_profiles/example_inference_azure_conv.csv"],
        # profiles=n_dc*["development/load_profiles/dummy_peak_provision.csv"],
        # profiles=n_dc*["development/load_profiles/Hawk_power_15_min.csv"],
        nominal_capacity=1e-3 * np.ones((n_dc)),
        linear_cost=np.ones(n_dc) * 0,
        settime_horizon=768,
        capital_cost=capital_cost,
    )
    pypsa_devices_dc.append(dcloads)
    return (
        capital_cost,
        dc_terminals,
        dcloads,
        load_scaling,
        n_dc,
        pypsa_devices_dc,
    )


@app.cell
def _(dcloads):
    dcloads.profile.shape
    return


@app.cell
def _(pypsa_devices_dc):
    pypsa_devices_dc[0].min_power.shape
    return


@app.cell
def _(capital_cost, cp, n_dc, np, pypsa_devices_dc, pypsa_net, zap):
    TOTAL_DC_BUDGET = 1 # GW

    # initial_capacities = np.array([1, 0, 0])
    initial_capacities = None

    xstar = zap.DispatchLayer(
        pypsa_net,
        pypsa_devices_dc,
        parameter_names={"dc_capacity": (5, "nominal_capacity")},
        time_horizon=96,
        solver=cp.CLARABEL, 
    )

    B = zap.planning.BendersSolver(
        layer=xstar,
        capital_cost=capital_cost,
        budget=TOTAL_DC_BUDGET,
        lower_bounds={"dc_capacity": np.full(n_dc, 0.05)},
        upper_bounds={"dc_capacity": np.full(n_dc, 0.25)},
        dispatch_scalar=1.0,
    )

    result = B.solve(initial_u=initial_capacities, max_iter=100, tol=1e-6)
    return B, TOTAL_DC_BUDGET, initial_capacities, result, xstar


@app.cell
def _(plt, result):
    plt.bar(range(len(result['u'])),result['u'])
    return


@app.cell
def _(np, plt, result):
    plt.bar(range(len(result['u'])), np.array([0.05000003, 0.06190926, 0.05000248, 0.05000005, 0.05000248,
           0.25000001, 0.25000001, 0.05000005, 0.08004652, 0.10803913]))
    return


@app.cell
def _(
    capital_cost,
    dc_terminals,
    deepcopy,
    load_scaling,
    n_dc,
    np,
    pypsa_devices,
    pypsa_net,
    zap,
):
    pypsa_devices_dc_test = deepcopy(pypsa_devices)
    pypsa_devices_dc_test[1].load *= load_scaling
    # nominal_capacity = np.array([5, 0, 5])
    # nominal_capacity = np.array([4.99999873, 0.50953675, 4.49046451])
    nominal_capacity = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    nominal_capacity = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])


    dcloads_test = zap.DataCenterLoad(
        num_nodes=pypsa_net.num_nodes,
        terminal=dc_terminals,
        profiles=n_dc*["development/load_profiles/example_inference_azure_conv.csv"],
        nominal_capacity=nominal_capacity,
        linear_cost=np.ones(n_dc) * 0,
        settime_horizon=96,
        capital_cost=capital_cost,
    )
    pypsa_devices_dc_test.append(dcloads_test)
    return dcloads_test, nominal_capacity, pypsa_devices_dc_test


@app.cell
def _(cp, pypsa_devices_dc_test, pypsa_net):
    outcome_test = pypsa_net.dispatch(
        pypsa_devices_dc_test, time_horizon=96, solver=cp.CLARABEL, add_ground=False
    )
    print(f'Dispatch Cost: {outcome_test.problem.value}')
    return (outcome_test,)


@app.cell
def _():
    1616, 1943, 2336
    return


if __name__ == "__main__":
    app.run()

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
    from zap.devices.injector import ARCHETYPES
    from zap.importers.pypsa import load_pypsa_network, parse_buses
    import os
    from pathlib import Path
    import pypsa
    from zap.devices import ACLine
    import pandas as pd
    import geopandas as gpd
    return (
        ACLine,
        ARCHETYPES,
        Path,
        cp,
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
def _(np):
    def upsample_zap_devices(devices, factor=4, original_timesteps=24):
        """Upsample time-varying attributes of zap devices by repeating each timestep."""
        for dev in devices:
            for attr in ['dynamic_capacity', 'load', 'linear_cost']:
                if hasattr(dev, attr):
                    val = getattr(dev, attr)
                    if val is not None and val.ndim == 2 and val.shape[1] == original_timesteps:
                        setattr(dev, attr, np.repeat(val, factor, axis=1))
    return (upsample_zap_devices,)


@app.cell
def _(os, pypsa):
    HOME_PATH = os.environ.get("HOME")
    PYPSA_NETW0RK_PATH = (
        HOME_PATH + "/zap_data/pypsa-networks/western_small/network_2023.nc"
    )
    pn = pypsa.Network(PYPSA_NETW0RK_PATH)
    snapshots = pn.generators_t.p_max_pu.index
    snapshot_data = snapshots[5616:5640]  # 8/23/21 # hourly
    return HOME_PATH, PYPSA_NETW0RK_PATH, pn, snapshot_data, snapshots


@app.cell
def _(gpd, pn):
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
    return b, counties, county_url, gdf, j


@app.cell
def _(pn):
    pn.buses
    return


@app.cell
def _(load_pypsa_network, pn, snapshot_data, upsample_zap_devices):
    pypsa_kwargs = {}
    pypsa_net, pypsa_devices = load_pypsa_network(
        pn, snapshot_data, power_unit=1.0e3, cost_unit=100.0, **pypsa_kwargs
    )

    upsample_zap_devices(pypsa_devices, factor=4, original_timesteps=24)
    return pypsa_devices, pypsa_kwargs, pypsa_net


@app.cell
def _(np, pypsa_devices):
    props = vars(pypsa_devices[1])

    for k,v in props.items():
        if isinstance(v, np.ndarray):
            print(f"{k} has dimension: {v.shape}" )
    return k, props, v


@app.cell
def _(props):
    props
    return


@app.cell
def _(ACLine, np, zap):
    def calculate_ptdf(
            network: zap.PowerNetwork,
            devices: list[zap.devices.AbstractDevice],
            slack_bus: int = 0,
        ) -> np.ndarray:
            """Calculate the power transfer distribution factors for the given network and devices.

            :param network: Zap network.
            :param devices: Zap devices.

            return: PTDF matrix shape (num_lines, num_buses)
            """
            ac_lines = [d for d in devices if isinstance(d, ACLine)][0]

            Y_bus = compute_Y_bus(ac_lines)
            B_inv = np.linalg.pinv(Y_bus)

            A = ac_lines.incidence_matrix[1] - ac_lines.incidence_matrix[0]
            sus = np.diag(ac_lines.susceptance[:, 0] * ac_lines.nominal_capacity[:, 0])

            ptdf = (sus @ A.T) @ B_inv

            return ptdf

    def compute_Y_bus(ac_lines: ACLine) -> np.ndarray:
        """Compute the Y-bus matrix for the given AC lines.

        :param ac_lines: Zap AC line objects
        """
        sus = ac_lines.susceptance[:, 0] * ac_lines.nominal_capacity[:, 0]
        A = ac_lines.incidence_matrix[1] - ac_lines.incidence_matrix[0]

        Y_bus = A @ np.diag(sus) @ A.T
        return Y_bus
    return calculate_ptdf, compute_Y_bus


@app.cell
def _(calculate_ptdf, pypsa_devices, pypsa_net):
    ptdf = calculate_ptdf(pypsa_net, pypsa_devices)
    return (ptdf,)


@app.cell
def _(ptdf):
    ptdf.shape
    return


@app.cell
def _(pypsa_devices):
    pypsa_devices[2]
    return


@app.cell
def _(pypsa_devices):
    pypsa_devices[0].linear_cost.shape
    return


@app.cell
def _(cp, pypsa_devices, pypsa_net):
    outcome_test = pypsa_net.dispatch(
        pypsa_devices, time_horizon=96, solver=cp.CLARABEL, add_ground=False
    )
    return (outcome_test,)


@app.cell
def _(ACLine, np, outcome_test, pypsa_devices):
    ac_lines = [d for d in pypsa_devices if isinstance(d, ACLine)][0]
    nom = np.asarray(ac_lines.nominal_capacity)[:, 0] # [L,]
    cap = np.asarray(ac_lines.capacity) # All 1's anyway
    Fmax = nom[:, None] * cap # [L, 1]
    f = np.asarray(outcome_test.power[3][1]) # [L, T] of line flows
    return Fmax, ac_lines, cap, f, nom


@app.cell
def _(Fmax, f, np, parse_buses, pn, ptdf):
    eps = 1e-9
    PTDF = np.asarray(ptdf) # (L, N)
    PTDF_abs = np.abs(PTDF) + eps # (L, N)
    Fmax_T = np.repeat(Fmax, repeats=f.shape[1], axis=1)  # (L, T)
    # line headroom margin: (L, T)
    margin = np.clip(Fmax_T - np.abs(f), 0.0, None)
    ratio = margin.T[:, :, None] / PTDF_abs[None, :, :] # (T, L, N)
    h_tj = ratio.min(axis=1) # (T, N)

    # Conservative aggregation across time
    h_j = np.quantile(h_tj, 0.05, axis=0)                  # (N,)

    top10_bus_idx = np.argsort(-h_j)[:10]
    print("Top-10 bus indices by PTDF headroom:", top10_bus_idx)
    print("Top-10 scores:", h_j[top10_bus_idx])

    buses, buses_to_index = parse_buses(pn) # buses_to_index is dict of "pyspa_bus_name": "zap_terminal"
    index_to_bus = {idx: name for name, idx in buses_to_index.items()}
    pypsa_bus_names = [index_to_bus[i] for i in top10_bus_idx]
    print(pypsa_bus_names)
    return (
        Fmax_T,
        PTDF,
        PTDF_abs,
        buses,
        buses_to_index,
        eps,
        h_j,
        h_tj,
        index_to_bus,
        margin,
        pypsa_bus_names,
        ratio,
        top10_bus_idx,
    )


@app.cell
def _(pn, pypsa_bus_names):
    county_fips = pn.buses.loc[pypsa_bus_names, "county_fips"]
    county_fips
    return (county_fips,)


@app.cell
def _(buses_to_index):
    buses_to_index
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

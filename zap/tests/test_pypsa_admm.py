"""Regression tests for ADMMSolver on devices produced by load_pypsa_network.

The ADMM device loop in zap.admm.basic_solver unpacks `(p, v, lv)` from each
device's `admm_prox_update`. Several device classes returned only two values,
so any PyPSA-imported network that included one of them blew up with
`not enough values to unpack (expected 3, got 2)`. These tests pin the
contract: every device class produced by `load_pypsa_network` must return
the three-tuple expected by the solver.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pypsa
import pytest
import torch

from zap.admm import ADMMSolver
from zap.importers.pypsa import load_pypsa_network


def _build_battery_network() -> tuple[pypsa.Network, pd.DatetimeIndex]:
    """Tiny 2-bus PyPSA network with a generator, a load, an AC line, and a
    storage unit. The storage unit is the device that previously broke ADMM."""

    net = pypsa.Network()
    snapshots = pd.date_range("2024-01-01", periods=4, freq="h")
    net.set_snapshots(snapshots)

    net.add("Bus", "bus0")
    net.add("Bus", "bus1")
    net.add("Carrier", "gas", co2_emissions=0.0)
    net.add(
        "Generator",
        "g0",
        bus="bus0",
        p_nom=100.0,
        marginal_cost=10.0,
        carrier="gas",
    )
    net.add(
        "Load",
        "l0",
        bus="bus1",
        p_set=40.0 * np.ones(len(snapshots)),
    )
    net.add(
        "Line",
        "line0",
        bus0="bus0",
        bus1="bus1",
        s_nom=200.0,
        x=0.1,
    )
    net.add(
        "StorageUnit",
        "battery0",
        bus="bus1",
        p_nom=20.0,
        max_hours=4.0,
        efficiency_dispatch=1.0,
        efficiency_store=1.0,
    )
    return net, snapshots


def test_admm_runs_on_pypsa_storage_network():
    """ADMMSolver.solve must not raise on a network that includes a
    PyPSA-imported StorageUnit. Currently fails with
    `not enough values to unpack (expected 3, got 2)` on main."""

    pnet, snapshots = _build_battery_network()
    net, devices = load_pypsa_network(pnet, snapshots)

    assert any(d.__class__.__name__ == "StorageUnit" for d in devices), (
        "Regression network must include a StorageUnit to exercise the fix"
    )

    admm_devices = [
        d.torchify(machine="cpu", dtype=torch.float32) for d in devices
    ]
    solver = ADMMSolver(
        machine="cpu",
        dtype=torch.float32,
        num_iterations=200,
        atol=1e-4,
        rtol=1e-4,
    )

    state, _history = solver.solve(net, admm_devices, time_horizon=len(snapshots))

    assert state is not None
    assert len(state.power) == len(devices)
    for p in state.power:
        for tensor in p:
            assert torch.isfinite(tensor).all(), "ADMM produced non-finite power"


@pytest.mark.parametrize(
    "csv_dir",
    [
        pytest.param(
            "/home/agent/grid-app/data/networks/pypsa-eur-slice",
            id="pypsa-eur-slice",
            marks=pytest.mark.skipif(
                not __import__("os").path.isdir(
                    "/home/agent/grid-app/data/networks/pypsa-eur-slice"
                ),
                reason="pypsa-eur-slice fixture not present in this checkout",
            ),
        ),
    ],
)
def test_admm_runs_on_seeded_pypsa_eur_slice(csv_dir):
    """Smoke-level coverage on the real PyPSA-Eur slice that originally
    surfaced the bug. Skipped automatically when the dataset is absent."""

    pnet = pypsa.Network()
    pnet.import_from_csv_folder(csv_dir)
    snapshots = pnet.snapshots[:4]
    net, devices = load_pypsa_network(pnet, snapshots)

    admm_devices = [
        d.torchify(machine="cpu", dtype=torch.float32) for d in devices
    ]
    solver = ADMMSolver(
        machine="cpu",
        dtype=torch.float32,
        num_iterations=50,
        atol=1e-3,
        rtol=1e-3,
    )

    state, _history = solver.solve(net, admm_devices, time_horizon=len(snapshots))
    assert state is not None

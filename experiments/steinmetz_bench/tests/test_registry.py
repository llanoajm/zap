"""Tests for the dataset registry: synthetic + builtin resolve, cache blocks."""

import cvxpy as cp
import numpy as np
import pandas as pd
import pytest

from experiments.steinmetz_bench.datasets import (
    DataNotStagedError,
    DatasetSpec,
    resolve,
)
from zap.devices import ACLine, Generator, Load


def _by_type(devices, cls):
    return next(d for d in devices if isinstance(d, cls))


def test_synthetic_five_node_shapes():
    spec = DatasetSpec(name="syn5", kind="synthetic", n_nodes=5, hours=24, seed=0)
    ds = resolve(spec)

    assert ds.network.num_nodes == 5
    assert ds.time_horizon == 24
    assert isinstance(ds.time_index, pd.DatetimeIndex)
    assert len(ds.time_index) == 24

    gen = _by_type(ds.devices, Generator)
    load = _by_type(ds.devices, Load)
    line = _by_type(ds.devices, ACLine)

    # Two generators (cheap @ node 0, expensive @ load node), one load.
    assert gen.num_devices == 2
    assert load.num_devices == 1
    # A radial path on 5 nodes has 4 lines.
    assert line.num_devices == 4
    # Load profile spans the full horizon and is positive.
    assert load.load.shape == (1, 24)
    assert np.all(load.load > 0)
    # Line endpoints form the path 0-1-2-3-4.
    assert list(line.source_terminal) == [0, 1, 2, 3]
    assert list(line.sink_terminal) == [1, 2, 3, 4]


def test_synthetic_is_deterministic_given_seed():
    a = resolve(DatasetSpec(name="a", kind="synthetic", n_nodes=4, hours=12, seed=7))
    b = resolve(DatasetSpec(name="b", kind="synthetic", n_nodes=4, hours=12, seed=7))
    la = _by_type(a.devices, Load).load
    lb = _by_type(b.devices, Load).load
    np.testing.assert_array_equal(la, lb)


def test_synthetic_network_dispatches_and_congestion_separates_prices():
    free = resolve(
        DatasetSpec(name="free", kind="synthetic", n_nodes=4, hours=6, congested=False)
    )
    tight = resolve(
        DatasetSpec(name="tight", kind="synthetic", n_nodes=4, hours=6, congested=True)
    )

    out_free = free.network.dispatch(free.devices, solver=cp.CLARABEL)
    out_tight = tight.network.dispatch(tight.devices, solver=cp.CLARABEL)

    # Both solve to a finite cost, and the cheap generator actually dispatches
    # (a degenerate all-curtailed solve would make the price spread meaningless).
    assert np.isfinite(out_free.problem.value)
    assert np.isfinite(out_tight.problem.value)
    gen_free = np.asarray(out_free.power[0])
    gen_tight = np.asarray(out_tight.power[0])
    assert gen_free.sum() > 0
    assert gen_tight.sum() > 0
    # Congesting the radial path forces the expensive backstop on, raising cost.
    assert out_tight.problem.value > out_free.problem.value + 1.0

    # LMPs = power-balance duals, shape (num_nodes, hours).
    lmp_free = np.asarray(out_free.prices)
    lmp_tight = np.asarray(out_tight.prices)
    spread_free = lmp_free.max(axis=0) - lmp_free.min(axis=0)
    spread_tight = lmp_tight.max(axis=0) - lmp_tight.min(axis=0)

    # Tightening the radial lines must congest, raising the nodal price spread.
    assert spread_tight.max() > spread_free.max() + 1e-3


def test_builtin_garver():
    ds = resolve(DatasetSpec(name="garver", kind="builtin"))
    assert ds.network.num_nodes == 6

    gen = _by_type(ds.devices, Generator)
    load = _by_type(ds.devices, Load)
    line = _by_type(ds.devices, ACLine)

    # Garver: 3 generators, 5 loads, 15 candidate corridors.
    assert gen.num_devices == 3
    assert load.num_devices == 5
    assert line.num_devices == 15


def test_cache_missing_raises_data_not_staged():
    spec = DatasetSpec(name="definitely_not_staged_xyz", kind="cache")
    with pytest.raises(DataNotStagedError) as exc:
        resolve(spec)
    assert "definitely_not_staged_xyz" in str(exc.value)
    assert "stage" in str(exc.value).lower()


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        resolve(DatasetSpec(name="x", kind="nonsense"))

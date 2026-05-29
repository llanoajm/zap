"""Dataset registry + loaders for the Steinmetz benchmark suite.

A :class:`DatasetSpec` is resolved to a :class:`ResolvedDataset` bundling a zap
``PowerNetwork``, its device list, a pandas time index, and (optionally) price
and load frames. Three source kinds are supported:

- ``"synthetic"`` — a procedural DC-OPF network parameterized by ``n_nodes``,
  ``hours``, ``congested`` and ``seed``. Deterministic given the seed.
- ``"builtin"`` — wraps zap's existing toy importers (Garver, the 7-bus test
  net, the single-bus battery net) so benchmarks can reuse known fixtures.
- ``"cache"`` — reads real staged data from ``data/<name>/``. When the cache is
  absent it raises :class:`DataNotStagedError` (a clean block, never a hang or a
  download). Live market-data calls are never made here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from attrs import define, field

from zap.devices import AbstractDevice, ACLine, Generator, Load
from zap.importers.toy import (
    load_battery_network,
    load_garver_network,
    load_test_network,
)
from zap.network import PowerNetwork

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"

_BUILTIN_LOADERS = {
    "garver": load_garver_network,
    "toy7": load_test_network,
    "battery": load_battery_network,
}


class DataNotStagedError(FileNotFoundError):
    """Raised when a ``cache`` dataset is requested but ``data/<name>/`` is empty.

    The benchmark loop never downloads or retries; a human stages real data and
    re-runs. The message names the directory a human must populate.
    """


@define(kw_only=True)
class DatasetSpec:
    """Declarative description of a benchmark dataset.

    ``kind`` selects the source. ``n_nodes``/``hours``/``congested``/``seed``
    only apply to the synthetic generator; ``name`` selects the builtin loader
    or the ``data/<name>/`` cache directory.
    """

    name: str
    kind: str = "synthetic"  # one of: synthetic | builtin | cache
    n_nodes: int = 5
    hours: int = 24
    congested: bool = False
    seed: int = 0


@define(kw_only=True)
class ResolvedDataset:
    """A resolved dataset: a network + devices + time index (+ optional frames)."""

    spec: DatasetSpec
    network: PowerNetwork
    devices: list[AbstractDevice]
    time_index: pd.DatetimeIndex
    price_frame: Optional[pd.DataFrame] = field(default=None)
    load_frame: Optional[pd.DataFrame] = field(default=None)

    @property
    def time_horizon(self) -> int:
        return len(self.time_index)


def _hourly_index(hours: int) -> pd.DatetimeIndex:
    return pd.date_range("2025-01-01", periods=hours, freq="h")


def make_synthetic_network(
    n_nodes: int = 5,
    hours: int = 24,
    congested: bool = False,
    seed: int = 0,
) -> tuple[PowerNetwork, list[AbstractDevice], pd.DataFrame]:
    """Build a deterministic synthetic DC-OPF network on a path topology.

    Node 0 hosts a cheap base generator; the last node hosts an expensive local
    generator and the system load. A radial AC path 0-1-...-(n-1) carries cheap
    power outward. When ``congested`` is True the line capacities are tightened
    so the cheap generator cannot fully serve the remote load, forcing a price
    separation (nonzero LMP spread) the benchmarks can detect.
    """
    if n_nodes < 2:
        raise ValueError("synthetic network needs at least 2 nodes")

    rng = np.random.default_rng(seed)
    net = PowerNetwork(n_nodes)

    # Load profile at the last node: a daily-shaped curve with seeded noise.
    t = np.arange(hours)
    base = 100.0 + 40.0 * np.sin(2 * np.pi * (t - 6) / 24.0)
    load_profile = np.clip(base + rng.normal(0.0, 5.0, size=hours), 10.0, None)

    loads = Load(
        name="load",
        num_nodes=n_nodes,
        terminal=np.array([n_nodes - 1]),
        load=load_profile.reshape(1, hours),
        linear_cost=np.array([1000.0]),  # value of lost load / curtailment penalty
    )

    # Cheap generator at node 0, expensive backstop at the load node.
    cheap_cap = np.full((1, hours), 250.0)
    local_cap = np.full((1, hours), 250.0)
    generators = Generator(
        name="generator",
        num_nodes=n_nodes,
        terminal=np.array([0, n_nodes - 1]),
        dynamic_capacity=np.vstack([cheap_cap, local_cap]),
        linear_cost=np.array([10.0, 80.0]),
        nominal_capacity=np.array([1.0, 1.0]),
        capital_cost=np.array([20.0, 5.0]),
        emission_rates=np.array([0.4, 0.9]),
    )

    # Radial AC path. Capacity is tight under `congested`, generous otherwise.
    n_lines = n_nodes - 1
    line_cap = 60.0 if congested else 400.0
    lines = ACLine(
        name="line",
        num_nodes=n_nodes,
        source_terminal=np.arange(n_lines),
        sink_terminal=np.arange(1, n_nodes),
        susceptance=np.full(n_lines, 10.0),
        capacity=np.ones(n_lines),
        nominal_capacity=np.full(n_lines, line_cap),
        linear_cost=0.01 * np.ones(n_lines),
        capital_cost=np.ones(n_lines),
    )

    devices = [generators, loads, lines]
    load_frame = pd.DataFrame(
        {"node": n_nodes - 1, "load_mw": load_profile}, index=_hourly_index(hours)
    )
    return net, devices, load_frame


def _resolve_synthetic(spec: DatasetSpec) -> ResolvedDataset:
    net, devices, load_frame = make_synthetic_network(
        n_nodes=spec.n_nodes,
        hours=spec.hours,
        congested=spec.congested,
        seed=spec.seed,
    )
    return ResolvedDataset(
        spec=spec,
        network=net,
        devices=devices,
        time_index=_hourly_index(spec.hours),
        load_frame=load_frame,
    )


def _resolve_builtin(spec: DatasetSpec) -> ResolvedDataset:
    if spec.name not in _BUILTIN_LOADERS:
        raise KeyError(
            f"unknown builtin dataset {spec.name!r}; "
            f"choose one of {sorted(_BUILTIN_LOADERS)}"
        )
    net, devices = _BUILTIN_LOADERS[spec.name]()
    horizon = max(d.time_horizon for d in devices)
    horizon = max(horizon, 1)
    return ResolvedDataset(
        spec=spec,
        network=net,
        devices=devices,
        time_index=_hourly_index(horizon),
    )


def _resolve_cache(spec: DatasetSpec) -> ResolvedDataset:
    cache_dir = DATA_ROOT / spec.name
    if not cache_dir.is_dir() or not any(cache_dir.iterdir()):
        raise DataNotStagedError(
            f"No staged data for dataset {spec.name!r}: expected files under "
            f"{cache_dir}. A human must stage real data there (see data/README.md); "
            f"the benchmark loop never downloads."
        )
    raise NotImplementedError(
        "cache loading of staged real data is wired up by a later roadmap item; "
        "the synthetic path is the loop-runnable one."
    )


_RESOLVERS = {
    "synthetic": _resolve_synthetic,
    "builtin": _resolve_builtin,
    "cache": _resolve_cache,
}


def resolve(spec: DatasetSpec) -> ResolvedDataset:
    """Resolve a :class:`DatasetSpec` into a :class:`ResolvedDataset`."""
    if spec.kind not in _RESOLVERS:
        raise ValueError(
            f"unknown dataset kind {spec.kind!r}; choose one of {sorted(_RESOLVERS)}"
        )
    return _RESOLVERS[spec.kind](spec)

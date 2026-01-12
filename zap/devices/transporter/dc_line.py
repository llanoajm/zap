import numpy as np

from zap.devices.abstract import AbstractDevice, make_dynamic
from zap.util import replace_none
from .transporter import Transporter


class PowerLine(Transporter):
    """A simple symmetric transporter."""

    def __init__(
        self,
        *,
        num_nodes,
        name,
        source_terminal,
        sink_terminal,
        capacity,  # todo: should this be named dynamic_capacity to match generator and other devices?
        linear_cost=None,
        quadratic_cost=None,
        nominal_capacity=None,
        capital_cost=None,
        slack=None,
        min_nominal_capacity=None,
        max_nominal_capacity=None,
        reconductoring_cost=None,
        reconductoring_threshold=None,
    ):
        if linear_cost is None:
            linear_cost = np.zeros(capacity.shape)

        self.num_nodes = num_nodes
        self.name = name
        self.source_terminal = source_terminal
        self.sink_terminal = sink_terminal
        self.capacity = make_dynamic(capacity)
        self.linear_cost = make_dynamic(linear_cost)
        self.quadratic_cost = make_dynamic(quadratic_cost)
        self.nominal_capacity = make_dynamic(
            replace_none(nominal_capacity, np.ones(self.num_devices))
        )
        self.capital_cost = make_dynamic(capital_cost)
        self.slack = 0.0 if slack is None else make_dynamic(slack)
        self.min_nominal_capacity = make_dynamic(min_nominal_capacity)
        self.max_nominal_capacity = make_dynamic(max_nominal_capacity)
        self.reconductoring_cost = make_dynamic(reconductoring_cost)
        self.reconductoring_threshold = make_dynamic(reconductoring_threshold)

    @property
    def min_power(self):
        return -self.capacity

    @property
    def max_power(self):
        return self.capacity

    def sample_time(self, time_periods, original_time_horizon):
        # Skip Transporter.sample_time (which tries to set min_power/max_power properties)
        # and call AbstractDevice.sample_time directly for capital cost scaling
        dev = AbstractDevice.sample_time(self, time_periods, original_time_horizon)

        # Subsample time-varying attributes
        # Use modulo indexing if array is shorter than indices (e.g., single-year data
        # being sampled across multi-year horizon)
        if dev.capacity.shape[1] > 1:
            cap_indices = np.array(time_periods) % dev.capacity.shape[1]
            dev.capacity = dev.capacity[:, cap_indices]
        if dev.linear_cost.shape[1] > 1:
            cost_indices = np.array(time_periods) % dev.linear_cost.shape[1]
            dev.linear_cost = dev.linear_cost[:, cost_indices]

        return dev


class DCLine(PowerLine):
    """A simple symmetric transporter."""

    pass

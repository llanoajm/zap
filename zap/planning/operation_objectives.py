import torch
import numpy as np
import functools

from zap.network import DispatchOutcome, PowerNetwork
from zap.devices.abstract import AbstractDevice
from zap.devices.injector import Load


class AbstractOperationObjective:
    """Abstract implementation of operation objectives."""

    def __call__(self, y: DispatchOutcome, parameters=None, la=None):
        return self.forward(y, parameters=parameters, la=la)

    def forward(self, y: DispatchOutcome, parameters=None, la=None):
        raise NotImplementedError

    @property
    def is_convex(self):
        return False

    @property
    def is_linear(self):
        return False

    def __add__(self, other_objective):
        return MultiObjective([self, other_objective], [1.0, 1.0])

    def __mul__(self, weight):
        return MultiObjective([self], [weight])

    def __rmul__(self, weight):
        return self.__mul__(weight)


class MultiObjective(AbstractOperationObjective):
    """Weighted combination of multiple objectives."""

    def __init__(self, objectives: list[AbstractOperationObjective], weights: list[float]):
        self.objectives = objectives
        self.weights = weights

        # Merge multi-objectives
        new_objectives = []
        new_weights = []
        for obj, w in zip(objectives, weights):
            if isinstance(obj, MultiObjective):
                new_objectives.extend(obj.objectives)
                new_weights.extend([w * w_ for w_ in obj.weights])
            else:
                new_objectives.append(obj)
                new_weights.append(w)

        self.objectives = new_objectives
        self.weights = new_weights

        # Drop zero-weight objectives
        self.objectives = [obj for obj, w in zip(self.objectives, self.weights) if w > 0]
        self.weights = [w for w in self.weights if w > 0]

        # Check weights
        assert all([w >= 0 for w in weights])
        assert len(self.objectives) == len(self.weights)

    def forward(self, y: DispatchOutcome, parameters=None, la=np):
        return sum(
            w * obj(y, parameters=parameters, la=la)
            for w, obj in zip(self.weights, self.objectives)
        )

    @functools.cached_property
    def is_convex(self):
        return all(obj.is_convex for obj in self.objectives)

    @functools.cached_property
    def is_linear(self):
        return all(obj.is_linear for obj in self.objectives)


class DispatchCostObjective(AbstractOperationObjective):
    """Cost of the dispatch outcome."""

    def __init__(self, net: PowerNetwork, devices: list[AbstractDevice]):
        self.net = net
        self.devices = devices

        if getattr(devices[0], "torched", False):
            self.torch_devices = devices
            self.torched = True
        else:
            self.torch_devices = [d.torchify(machine="cpu") for d in devices]
            self.torched = False

    def forward(self, y: DispatchOutcome, parameters=None, la=None):
        if la is None:
            la = torch if self.torched else np

        devices = self.torch_devices if la == torch else self.devices
        return self.net.operation_cost(
            devices, y.power, y.angle, y.local_variables, parameters=parameters, la=la
        )

    @property
    def is_convex(self):
        return True

    @property
    def is_linear(self):
        return False


class EmissionsObjective(AbstractOperationObjective):
    """Total emissions of the dispatch outcome."""

    def __init__(self, devices: list[AbstractDevice]):
        self.devices = devices

        if getattr(devices[0], "torched", False):
            self.torch_devices = devices
            self.torched = True
        else:
            self.torch_devices = [d.torchify(machine="cpu") for d in devices]
            self.torched = False

    def forward(self, y: DispatchOutcome, parameters=None, la=None):
        if la is None:
            la = torch if self.torched else np

        devices = self.torch_devices if la == torch else self.devices
        emissions = [
            d.get_emissions(p, **param, la=la) for p, d, param in zip(y.power, devices, parameters)
        ]

        return sum(emissions)

    @property
    def is_convex(self):
        return True

    @property
    def is_linear(self):
        return True


class UnservedEnergyObjective(AbstractOperationObjective):
    """Total unserved energy across all Load devices.

    Computes the difference between nominal load demand and actual power served.
    For each Load device, unserved energy = load + power[0], where power[0] is
    negative (consumption) and ranges from -load (full demand met) to 0 (no demand met).
    """

    def __init__(self, devices: list[AbstractDevice]):
        self.devices = devices
        self.load_indices = [i for i, d in enumerate(devices) if isinstance(d, Load)]

        if getattr(devices[0], "torched", False):
            self.torch_devices = devices
            self.torched = True
        else:
            self.torch_devices = [d.torchify(machine="cpu") for d in devices]
            self.torched = False

    def forward(self, y: DispatchOutcome, parameters=None, la=None):
        if la is None:
            la = torch if self.torched else np

        devices = self.torch_devices if la == torch else self.devices

        total_unserved = 0.0
        for idx in self.load_indices:
            load_device = devices[idx]
            power = y.power[idx]
            # Unserved = nominal load - actual served = load + power[0]
            # (power[0] is negative, so this gives the shortfall)
            unserved = load_device.load + power[0]
            total_unserved = total_unserved + la.sum(unserved)

        return total_unserved

    @property
    def is_convex(self):
        return True

    @property
    def is_linear(self):
        return True

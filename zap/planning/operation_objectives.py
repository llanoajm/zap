import math
import torch
import numpy as np
import functools

from zap.network import DispatchOutcome, PowerNetwork
from zap.devices.abstract import AbstractDevice


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


class LMPObjective(AbstractOperationObjective):
    """Metric of dispatch LMPs."""

    def __init__(self, net: PowerNetwork, devices: list[AbstractDevice], lmp_metric: str = "max"):
        self.net = net
        self.devices = devices
        self.lmp_metric = lmp_metric

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
        lmps = y.prices
        if self.lmp_metric == "l2":
            return la.mean(lmps**2)
        elif self.lmp_metric == "l1":
            return la.sum(la.abs(lmps))
        elif self.lmp_metric == "max":
            return la.max(lmps)
        elif self.lmp_metric == "cvar":
            alpha = 0.95
            if la is torch:
                sorted_x, _ = torch.sort(lmps)  # ascending
                n = sorted_x.numel()
            else:
                sorted_x = np.sort(lmps)
                n = sorted_x.size

            # CVaR_alpha = mean of worst (1-alpha) tail
            k = int(math.floor(alpha * n))
            k = min(max(k, 0), n - 1)  # clamp to valid
            return sorted_x[k:].mean()

    @property
    def is_convex(self):
        return True

    @property
    def is_linear(self):
        return False

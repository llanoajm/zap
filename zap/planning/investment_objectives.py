import torch
import numpy as np

from zap.devices.abstract import AbstractDevice
from zap.layer import DispatchLayer


class AbstractInvestmentObjective:
    """Abstract implementation of investment objectives."""

    def __call__(self, la=np, **kwargs):
        return self.forward(la=la, **kwargs)

    def forward(self, la=np, **kwargs):
        raise NotImplementedError

    @property
    def is_convex(self):
        return False

    @property
    def is_linear(self):
        return False


class InvestmentObjective(AbstractInvestmentObjective):
    """Simple linear investment objective."""

    def __init__(self, devices: list[AbstractDevice], layer: DispatchLayer):
        self.devices = devices
        self.layer = layer

        if getattr(devices[0], "torched", False):
            self.torch_devices = devices
        else:
            self.torch_devices = [d.torchify(machine="cpu") for d in devices]

    def forward(self, la=np, **kwargs):
        parameters = self.layer.setup_parameters(**kwargs)
        devices = self.torch_devices if la == torch else self.devices

        costs = [d.get_investment_cost(la=la, **param) for d, param in zip(devices, parameters)]

        return sum(costs)

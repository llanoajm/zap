import functools
import math

import numpy as np
import torch

from zap.devices.abstract import AbstractDevice
from zap.network import DispatchOutcome, PowerNetwork


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

    def __init__(
        self,
        net: PowerNetwork,
        devices: list[AbstractDevice],
        lmp_metric: str = "meanmax",
        lmp_beta: float = 1.0,
    ):
        self.net = net
        self.devices = devices
        self.lmp_metric = lmp_metric
        self.lmp_beta = lmp_beta

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
        elif self.lmp_metric == "meanmax":
            if la == torch:
                return lmps.max(dim=1).values.mean()
            else:
                return np.mean(np.max(lmps, axis=1))
        elif self.lmp_metric == "summax":
            if la == torch:
                return lmps.max(dim=1).values.sum()
            else:
                return np.sum(np.max(lmps, axis=1))
        elif self.lmp_metric == "meantopk":
            k = int(getattr(self, "topk", 5))
            if la == torch:
                return torch.topk(lmps, k, dim=1).values.mean()
            else:
                # sort along axis=1 and take last k
                return np.sort(lmps, axis=1)[:, -k:].mean()
        elif self.lmp_metric == "sumtopk":
            k = int(getattr(self, "topk", 5))
            if la == torch:
                return torch.topk(lmps, k, dim=1).values.sum()
            else:
                return np.sort(lmps, axis=1)[:, -k:].sum()
        elif self.lmp_metric == "meanpctl":
            q = float(getattr(self, "pctl", 0.95))
            if la == torch:
                # torch.quantile exists in recent versions; fallback: topk approximation below if needed
                return torch.quantile(lmps, q, dim=1).mean()
            else:
                return np.quantile(lmps, q, axis=1).mean()

        elif self.lmp_metric == "sumpctl":
            q = float(getattr(self, "pctl", 0.95))
            if la == torch:
                return torch.quantile(lmps, q, dim=1).sum()
            else:
                return np.quantile(lmps, q, axis=1).sum()

        elif self.lmp_metric == "meansmoothmax":
            alpha = float(getattr(self, "smooth_alpha", 20.0))
            if la == torch:
                sm = torch.logsumexp(alpha * lmps, dim=1) / alpha  # [N]
                return self.lmp_beta * sm.mean()
            else:
                x = alpha * lmps
                m = np.max(x, axis=1, keepdims=True)
                sm = (np.log(np.sum(np.exp(x - m), axis=1)) + m.squeeze(1)) / alpha  # [N]
                return self.lmp_beta * sm.mean()

        elif self.lmp_metric == "sumsmoothmax":
            alpha = float(getattr(self, "smooth_alpha", 20.0))
            if la == torch:
                sm = torch.logsumexp(alpha * lmps, dim=1) / alpha
                return self.lmp_beta * sm.sum()
            else:
                x = alpha * lmps
                m = np.max(x, axis=1, keepdims=True)
                sm = (np.log(np.sum(np.exp(x - m), axis=1)) + m.squeeze(1)) / alpha
                return self.lmp_beta * sm.sum()

    @property
    def is_convex(self):
        return True

    @property
    def is_linear(self):
        return False


class SCOPFLMPObjective(AbstractOperationObjective):
    """
    Security-Constrained OPF LMP objective.

    LMP objective that is contingency-aware. Supports both 2D prices
    (nodes, time) and 3D prices (nodes, time, scenarios), aggregating
    across scenarios when present.

    Args:
        net: PowerNetwork instance
        devices: List of devices
        lmp_metric: Metric to use ('meanmax', 'l2', 'meansmoothmax', etc.)
        lmp_beta: Scaling factor for objective
        aggregation: How to aggregate if scenario-specific prices available
    """

    def __init__(
        self,
        net: PowerNetwork,
        devices: list[AbstractDevice],
        lmp_metric: str = "sumsmoothmax",
        lmp_beta: float = 1.0,
        aggregation: str = "mean",
        node_idx: np.ndarray | list[int] | None = None,
    ):
        self.net = net
        self.devices = devices
        self.lmp_metric = lmp_metric
        self.lmp_beta = lmp_beta
        self.aggregation = aggregation
        self.node_idx = None if node_idx is None else np.asarray(node_idx, dtype=int)

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
        if self.node_idx is not None:
            if la == torch:
                idx = torch.as_tensor(self.node_idx, device=lmps.device)
                lmps = lmps.index_select(0, idx)
            else:
                lmps = lmps[self.node_idx, :]

        def metric(lmp_2d):
            if self.lmp_metric == "l2":
                return la.mean(lmp_2d**2)
            if self.lmp_metric == "l1":
                return la.sum(la.abs(lmp_2d))
            if self.lmp_metric == "meanmax":
                if la == torch:
                    return lmp_2d.max(dim=1).values.mean()
                return np.mean(np.max(lmp_2d, axis=1))
            if self.lmp_metric == "meantopk":
                k = int(getattr(self, "topk", 5))
                if la == torch:
                    return torch.topk(lmp_2d, k, dim=1).values.mean()
                return np.sort(lmp_2d, axis=1)[:, -k:].mean()
            if self.lmp_metric == "meanpctl":
                q = float(getattr(self, "pctl", 0.95))
                if la == torch:
                    return torch.quantile(lmp_2d, q, dim=1).mean()
                return np.quantile(lmp_2d, q, axis=1).mean()
            if self.lmp_metric == "cvar":
                alpha = float(getattr(self, "cvar_alpha", 0.95))
                if la == torch:
                    sorted_x, _ = torch.sort(lmp_2d, dim=1)  # ascending
                    T = sorted_x.shape[1]
                    k0 = int(math.floor(alpha * T))
                    k0 = min(max(k0, 0), T - 1)
                    return sorted_x[:, k0:].mean()
                else:
                    sorted_x = np.sort(lmp_2d, axis=1)
                    T = sorted_x.shape[1]
                    k0 = int(math.floor(alpha * T))
                    k0 = min(max(k0, 0), T - 1)
                    return sorted_x[:, k0:].mean()
            if self.lmp_metric == "meansmoothmax":
                alpha = float(getattr(self, "smooth_alpha", 20.0))
                if la == torch:
                    sm = torch.logsumexp(alpha * lmp_2d, dim=1) / alpha
                    return self.lmp_beta * sm.mean()
                x = alpha * lmp_2d
                m = np.max(x, axis=1, keepdims=True)
                sm = (np.log(np.sum(np.exp(x - m), axis=1)) + m.squeeze(1)) / alpha
                return self.lmp_beta * sm.mean()
            if self.lmp_metric == "sumsmoothmax":
                alpha = float(getattr(self, "smooth_alpha", 20.0))
                if la == torch:
                    sm = torch.logsumexp(alpha * lmp_2d, dim=1) / alpha
                    return self.lmp_beta * sm.sum()
                else:
                    x = alpha * lmp_2d
                    m = np.max(x, axis=1, keepdims=True)
                    sm = (np.log(np.sum(np.exp(x - m), axis=1)) + m.squeeze(1)) / alpha
                    return self.lmp_beta * sm.sum()

        if lmps.ndim == 2:
            return metric(lmps)

        if lmps.ndim != 3:
            raise ValueError(f"Unexpected LMP shape: {lmps.shape}")

        num_scenarios = lmps.shape[2]
        per_scenario = [metric(lmps[:, :, s]) for s in range(num_scenarios)]
        if la == torch:
            per_scenario = torch.stack(per_scenario)
        else:
            per_scenario = np.array(per_scenario)

        if self.aggregation == "mean":
            return per_scenario.mean()
        if self.aggregation == "sum":
            return per_scenario.sum()
        if self.aggregation == "max":
            return per_scenario.max() if la == torch else np.max(per_scenario)
        raise ValueError(f"Unknown aggregation: {self.aggregation}")

    @property
    def is_convex(self):
        return True

    @property
    def is_linear(self):
        return False


# class DCTailPriceObjective(AbstractOperationObjective):
#     """
#     Penalize high LMPs *at the DC terminals* using a tail metric over time.

#     This makes "spreading" arise naturally (no caps) by making concentrated
#     capacity at a high-tail-price location expensive.

#     If weight_by_capacity:
#         sum_i dc_cap[i] * tail_t(price[terminal_i, :])
#     Else:
#         sum_i tail_t(price[terminal_i, :])

#     tail_t is controlled by lmp_metric (e.g. 'cvar', 'meantopk', 'meansmoothmax', 'meanmax').
#     """

#     def __init__(
#         self,
#         devices: list[AbstractDevice],
#         dc_device_idx: int,
#         lmp_metric: str = "cvar",
#         weight_by_capacity: bool = True,
#         aggregation_across_dcs: str = "sum",  # 'sum' | 'max' | 'mean'
#         cvar_alpha: float = 0.95,
#         topk: int = 5,
#         smooth_alpha: float = 20.0,
#     ):
#         self.devices = devices
#         self.dc_device_idx = int(dc_device_idx)
#         self.lmp_metric = lmp_metric
#         self.weight_by_capacity = bool(weight_by_capacity)
#         self.aggregation_across_dcs = aggregation_across_dcs
#         self.cvar_alpha = float(cvar_alpha)
#         self.topk = int(topk)
#         self.smooth_alpha = float(smooth_alpha)

#         if getattr(devices[0], "torched", False):
#             self.torch_devices = devices
#             self.torched = True
#         else:
#             self.torch_devices = [d.torchify(machine="cpu") for d in devices]
#             self.torched = False

#     def forward(self, y: DispatchOutcome, parameters=None, la=None):
#         if la is None:
#             la = torch if self.torched else np

#         devices = self.torch_devices if la == torch else self.devices
#         lmps = y.prices

#         dc_dev = devices[self.dc_device_idx]
#         terminals = dc_dev.terminals
#         if la == torch:
#             if not torch.is_tensor(terminals):
#                 terminals = torch.as_tensor(terminals, device=lmps.device)
#             else:
#                 terminals = terminals.to(device=lmps.device)
#         else:
#             terminals = np.asarray(terminals, dtype=int)

#         def tail_over_time(pr_2d):
#             # pr_2d: (n_dc, T)
#             if self.lmp_metric == "meanmax":
#                 if la == torch:
#                     return pr_2d.max(dim=1).values
#                 return np.max(pr_2d, axis=1)

#             if self.lmp_metric == "meantopk":
#                 k = max(1, int(self.topk))
#                 if la == torch:
#                     return torch.topk(pr_2d, k, dim=1).values.mean(dim=1)
#                 return np.sort(pr_2d, axis=1)[:, -k:].mean(axis=1)

#             if self.lmp_metric == "cvar":
#                 alpha = float(self.cvar_alpha)
#                 if la == torch:
#                     sorted_x, _ = torch.sort(pr_2d, dim=1)  # ascending
#                     T = sorted_x.shape[1]
#                     k0 = int(math.floor(alpha * T))
#                     k0 = min(max(k0, 0), T - 1)
#                     return sorted_x[:, k0:].mean(dim=1)
#                 sorted_x = np.sort(pr_2d, axis=1)
#                 T = sorted_x.shape[1]
#                 k0 = int(math.floor(alpha * T))
#                 k0 = min(max(k0, 0), T - 1)
#                 return sorted_x[:, k0:].mean(axis=1)

#             if self.lmp_metric == "meansmoothmax":
#                 alpha = float(self.smooth_alpha)
#                 if la == torch:
#                     return torch.logsumexp(alpha * pr_2d, dim=1) / alpha
#                 x = alpha * pr_2d
#                 m = np.max(x, axis=1, keepdims=True)
#                 return (np.log(np.sum(np.exp(x - m), axis=1)) + m.squeeze(1)) / alpha

#             raise ValueError(f"Unsupported lmp_metric for DCTailPriceObjective: {self.lmp_metric}")

#         def combine(dc_tail):
#             if self.aggregation_across_dcs == "sum":
#                 return dc_tail.sum() if la == torch else np.sum(dc_tail)
#             if self.aggregation_across_dcs == "mean":
#                 return dc_tail.mean() if la == torch else np.mean(dc_tail)
#             if self.aggregation_across_dcs == "max":
#                 return dc_tail.max() if la == torch else np.max(dc_tail)
#             raise ValueError(f"Unknown aggregation_across_dcs: {self.aggregation_across_dcs}")

#         if lmps.ndim == 2:
#             pr = lmps.index_select(0, terminals) if la == torch else lmps[terminals, :]
#             dc_tail = tail_over_time(pr)
#         elif lmps.ndim == 3:
#             num_scenarios = lmps.shape[2]
#             per_s = []
#             for s in range(num_scenarios):
#                 pr_s = lmps[:, :, s]
#                 pr_s = pr_s.index_select(0, terminals) if la == torch else pr_s[terminals, :]
#                 per_s.append(tail_over_time(pr_s))
#             if la == torch:
#                 dc_tail = torch.stack(per_s, dim=0).mean(dim=0)
#             else:
#                 dc_tail = np.mean(np.stack(per_s, axis=0), axis=0)
#         else:
#             raise ValueError(f"Unexpected prices shape: {lmps.shape}")

#         if self.weight_by_capacity:
#             dc_cap = None
#             if parameters is not None:
#                 dc_cap = parameters[self.dc_device_idx].get("nominal_capacity", None)
#             if dc_cap is None:
#                 dc_cap = getattr(dc_dev, "nominal_capacity", None)

#             if la == torch:
#                 if not torch.is_tensor(dc_cap):
#                     dc_cap = torch.as_tensor(dc_cap, device=dc_tail.device, dtype=dc_tail.dtype)
#                 dc_cap = dc_cap.reshape(-1)
#             else:
#                 dc_cap = np.asarray(dc_cap).reshape(-1)

#             return combine(dc_cap * dc_tail)

#         return combine(dc_tail)

import math
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from attrs import Factory, define, field
from numpy.typing import NDArray

from .abstract import AbstractDevice, get_time_horizon, make_dynamic


@define(kw_only=True, slots=False)
class AbstractInjector(AbstractDevice):
    """A single-node device that may deposit or withdraw power from the network. Abstract type that
    should not be instantiated but contains shared behavior among all subclasses."""

    num_nodes: int
    terminal: NDArray
    nominal_capacity: NDArray = field(
        default=Factory(lambda self: np.ones(self.num_devices), takes_self=True),
        converter=make_dynamic,
    )

    # These properties should be implemented by subclasses
    min_power: NDArray = field(init=False)
    max_power: NDArray = field(init=False)
    linear_cost: NDArray = field(init=False)
    quadratic_cost: Optional[NDArray] = field(init=False)
    capital_cost: Optional[NDArray] = field(init=False)
    emission_rates: Optional[NDArray] = field(init=False)

    @property
    def terminals(self):
        return self.terminal

    @property
    def time_horizon(self):
        return get_time_horizon(self.min_power)

    def scale_costs(self, scale):
        self.linear_cost /= scale
        if self.quadratic_cost is not None:
            self.quadratic_cost /= scale
        if self.capital_cost is not None:
            self.capital_cost /= scale

        # Emissions are in units of kg/MWh
        # but we scale them with costs so that prices can be stated in $/MWh
        if self.emission_rates is not None:
            self.emission_rates /= scale

    def scale_power(self, scale):
        self.nominal_capacity /= scale

        # Invert scaling because term is quadratic
        if self.quadratic_cost is not None:
            self.quadratic_cost *= scale

    # ====
    # CORE MODELING FUNCTIONS
    # ====

    def equality_constraints(
        self,
        power,
        angle,
        _,
        nominal_capacity=None,
        max_power=None,
        min_power=None,
        linear_cost=None,
        la=np,
        envelope=None,
    ):
        return []

    def inequality_constraints(
        self,
        power,
        angle,
        _,
        nominal_capacity=None,
        max_power=None,
        min_power=None,
        linear_cost=None,
        la=np,
        envelope=None,
    ):
        nominal_capacity = self.parameterize(nominal_capacity=nominal_capacity, la=la)
        max_power = self.parameterize(max_power=max_power, la=la)
        min_power = self.parameterize(min_power=min_power, la=la)
        power = power[0]

        return [
            la.multiply(min_power, nominal_capacity) - power,
            power - la.multiply(max_power, nominal_capacity),
        ]

    def operation_cost(
        self,
        power,
        angle,
        _,
        nominal_capacity=None,
        max_power=None,
        min_power=None,
        linear_cost=None,
        la=np,
        envelope=None,
    ):
        nominal_capacity = self.parameterize(nominal_capacity=nominal_capacity, la=la)
        min_power = self.parameterize(min_power=min_power, la=la)
        linear_cost = self.parameterize(linear_cost=linear_cost, la=la)
        power = power[0] - la.multiply(min_power, nominal_capacity)

        cost = la.sum(la.multiply(linear_cost, power))
        if self.quadratic_cost is not None:
            cost += la.sum(la.multiply(self.quadratic_cost, la.square(power)))

        return cost

    # ====
    # PLANNING FUNCTIONS
    # ====

    def sample_time(self, time_periods, original_time_horizon):
        dev = super().sample_time(time_periods, original_time_horizon)

        factor = time_periods // original_time_horizon
        if dev.linear_cost.shape[1] > 1:
            if factor > 1:
                # Upsample
                dev.linear_cost = np.ascontiguousarray(np.repeat(dev.linear_cost, factor, axis=1))
            else:
                # Subsample linear cost
                dev.linear_cost = dev.linear_cost[:, time_periods]

        return dev

    # ====
    # DIFFERENTIATION
    # ====

    def _equality_matrices(
        self,
        equalities,
        nominal_capacity=None,
        max_power=None,
        min_power=None,
        linear_cost=None,
        la=np,
    ):
        return equalities

    def _inequality_matrices(
        self,
        inequalities,
        nominal_capacity=None,
        max_power=None,
        min_power=None,
        linear_cost=None,
        la=np,
    ):
        size = inequalities[0].power[0].shape[1]
        inequalities[0].power[0] += -sp.eye(size)
        inequalities[1].power[0] += sp.eye(size)
        return inequalities

    def _hessian_power(
        self,
        hessians,
        power,
        angle,
        _,
        nominal_capacity=None,
        max_power=None,
        min_power=None,
        linear_cost=None,
        la=np,
    ):
        if self.quadratic_cost is None:
            return hessians

        hessians[0] += 2 * sp.diags((self.quadratic_cost * np.ones_like(power[0])).ravel())
        return hessians

    # ====
    # ADMM FUNCTIONS
    # ====

    def admm_prox_update(
        self,
        rho_power,
        rho_angle,
        power,
        angle,
        nominal_capacity=None,
        max_power=None,
        min_power=None,
        linear_cost=None,
        power_weights=None,
        angle_weights=None,
    ):
        nominal_capacity = self.parameterize(nominal_capacity=nominal_capacity)
        max_power = self.parameterize(max_power=max_power)
        min_power = self.parameterize(min_power=min_power)
        linear_cost = self.parameterize(linear_cost=linear_cost)

        # machine, dtype = power[0].device, power[0].dtype
        assert angle is None

        if self.has_changed:
            quadratic_cost = (
                0.0 * linear_cost if self.quadratic_cost is None else self.quadratic_cost
            )
            pmax = torch.multiply(max_power, nominal_capacity)
            pmin = torch.multiply(min_power, nominal_capacity)
            self.admm_data = (quadratic_cost, pmax, pmin)
            self.has_changed = False

        quadratic_cost, pmax, pmin = self.admm_data

        return _admm_prox_update(power, rho_power, linear_cost, quadratic_cost, pmin, pmax)

    def get_admm_power_weights(
        self,
        power,
        strategy: str,
        nominal_capacity=None,
        max_power=None,
        min_power=None,
        linear_cost=None,
    ):
        nominal_capacity = self.parameterize(nominal_capacity=nominal_capacity)
        linear_cost = self.parameterize(linear_cost=linear_cost)

        if strategy == "smart_cost":
            avg_cost = np.mean(linear_cost, axis=1).reshape((-1, 1))
            return [np.maximum(np.sqrt(1 / (avg_cost + 0.01)), 1.0)]

        if strategy == "smart_bounds":
            return [np.minimum(np.sqrt(1 / (nominal_capacity + 1.0)), 1.0)]

        else:
            return [np.ones_like(pi) for pi in power]


@define(kw_only=True, slots=False)
class Injector(AbstractInjector):
    """A single-node device that may deposit or withdraw power from the network."""

    min_power: NDArray = field(converter=make_dynamic)
    max_power: NDArray = field(converter=make_dynamic)
    linear_cost: NDArray = field(converter=make_dynamic)
    quadratic_cost: Optional[NDArray] = field(default=None, converter=make_dynamic)
    capital_cost: Optional[NDArray] = field(default=None, converter=make_dynamic)
    emission_rates: Optional[NDArray] = field(default=None, converter=make_dynamic)

    # TODO - Add dimension checks


@define(kw_only=True, slots=False)
class Generator(AbstractInjector):
    """An Injector that can only deposit power."""

    dynamic_capacity: NDArray = field(converter=make_dynamic)
    linear_cost: NDArray = field(converter=make_dynamic)
    quadratic_cost: Optional[NDArray] = field(default=None, converter=make_dynamic)
    capital_cost: Optional[NDArray] = field(default=None, converter=make_dynamic)
    emission_rates: Optional[NDArray] = field(default=None, converter=make_dynamic)
    min_nominal_capacity: Optional[NDArray] = field(default=None, converter=make_dynamic)
    max_nominal_capacity: Optional[NDArray] = field(default=None, converter=make_dynamic)

    # TODO - Add dimension checks

    @property
    def min_power(self):
        return 0.0 * self.dynamic_capacity

    @property
    def max_power(self):
        return self.dynamic_capacity

    def scale_power(self, scale):
        if self.min_nominal_capacity is not None:
            self.min_nominal_capacity /= scale
        if self.max_nominal_capacity is not None:
            self.max_nominal_capacity /= scale
        return super().scale_power(scale)

    # ====
    # PLANNING FUNCTIONS
    # ====

    def get_investment_cost(self, nominal_capacity=None, la=np):
        if self.capital_cost is None or nominal_capacity is None:
            return 0.0

        pnom_min = self.nominal_capacity
        capital_cost = self.capital_cost

        return la.sum(la.multiply(capital_cost, (nominal_capacity - pnom_min)))

    def get_emissions(self, power, nominal_capacity=None, la=np):
        if self.emission_rates is None:
            return 0.0
        else:
            return la.sum(la.multiply(self.emission_rates, power[0]))

    def sample_time(self, time_periods, original_time_horizon):
        dev = super().sample_time(time_periods, original_time_horizon)

        factor = time_periods // original_time_horizon
        if dev.dynamic_capacity.shape[1] > 1:
            if factor > 1:
                # Upsample
                dev.dynamic_capacity = np.ascontiguousarray(
                    np.repeat(dev.dynamic_capacity, factor, axis=1)
                )
            else:
                # Subsample dynamic capacity
                dev.dynamic_capacity = dev.dynamic_capacity[:, time_periods]

        return dev


@define(kw_only=True, slots=False)
class Load(AbstractInjector):
    """An Injector that can only withdraw power."""

    load: NDArray = field(converter=make_dynamic)
    linear_cost: NDArray = field(converter=make_dynamic)
    quadratic_cost: Optional[NDArray] = field(default=None, converter=make_dynamic)

    @property
    def min_power(self):
        return -self.load

    @property
    def max_power(self):
        return 0.0 * self.load

    @property
    def capital_cost(self):
        return None

    @property
    def emission_rates(self):
        return None

    def sample_time(self, time_periods, original_time_horizon):
        dev = super().sample_time(time_periods, original_time_horizon)

        factor = time_periods // original_time_horizon
        if dev.load.shape[1] > 1:
            if factor > 1:
                # Upsample
                dev.load = np.ascontiguousarray(np.repeat(dev.load, factor, axis=1))
            else:
                # Subsample load
                dev.load = dev.load[:, time_periods]

        return dev


@define(kw_only=True, slots=False)
class DataCenterLoad(AbstractInjector):
    """
    Data center device with per-unit load profiles driven by nominal_capacity.
    Profiles can be provided as arrays or loaded from CSV files.
    """

    # --- Profile controls
    time_resolution_hours: float = field(default=0.25)
    settime_horizon: float = field(default=24.0)

    # Standard device parameters
    profiles: Optional[Union[list[Union[NDArray, str, Path]], NDArray, str, Path]] = field(
        default=None
    )
    min_power: NDArray = field(init=False)
    max_power: NDArray = field(init=False)
    capital_cost: Optional[NDArray] = field(default=None)
    emission_rates: Optional[NDArray] = field(default=None)
    nominal_capacity: NDArray
    linear_cost: NDArray = field(converter=make_dynamic)
    quadratic_cost: Optional[NDArray] = field(default=None, converter=make_dynamic)
    min_nominal_capacity: Optional[NDArray] = field(default=None, converter=make_dynamic)
    max_nominal_capacity: Optional[NDArray] = field(default=None, converter=make_dynamic)

    def __attrs_post_init__(self):
        num_dcs = len(self.nominal_capacity)

        # Broadcast scalar profiles to list
        if self.profiles is not None and not isinstance(self.profiles, list):
            self.profiles = [self.profiles]

        # Length checks / broadcast
        if self.capital_cost is not None and len(self.capital_cost) != num_dcs:
            self.capital_cost = [self.capital_cost[0]] * num_dcs
        if self.emission_rates is not None and len(self.emission_rates) != num_dcs:
            self.emission_rates = [self.emission_rates[0]] * num_dcs
        if len(self.linear_cost) != num_dcs:
            self.linear_cost = [self.linear_cost[0]] * num_dcs
        if self.quadratic_cost is not None and len(self.quadratic_cost) != num_dcs:
            self.quadratic_cost = [self.quadratic_cost[0]] * num_dcs

        # Load profiles
        all_profiles = []
        for i in range(num_dcs):
            if self.profiles is not None and i < len(self.profiles):
                profile = DataCenterLoad._load_profile(self.profiles[i])
            else:
                raise ValueError(f"Profile not provided for data center {i}")
            all_profiles.append(profile)

        # Validate consistent time horizons
        if len(all_profiles) > 1:
            lengths = [len(p) for p in all_profiles]
            if len(set(lengths)) > 1:
                raise ValueError(
                    f"All profiles must have the same length. Got lengths: {lengths}. "
                    f"Check that all CSV files and arrays cover the same time horizon."
                )

        self.profile = np.vstack(all_profiles)  # shape (n_dc, T), per-unit
        self.min_power = -self.profile  # withdraw
        self.max_power = np.zeros_like(self.profile)

        super_class = super()
        if hasattr(super_class, "__attrs_post_init__"):
            super_class.__attrs_post_init__()

    @staticmethod
    def _load_profile(profile: Union[NDArray, str, Path]) -> NDArray:
        """Load profile from array or CSV file. Returns per-unit (0..1) load factors."""
        if isinstance(profile, (str, Path)):
            df = pd.read_csv(profile, index_col=0)
            # Try common column names for power values
            for col in ["watts", "power", "load", "p"]:
                if col in df.columns:
                    return df[col].to_numpy().reshape(-1)
            raise ValueError(f"CSV must contain a power column (watts/power/load/p): {profile}")
        # Handle array input
        if isinstance(profile, np.ndarray):
            if profile.ndim == 1:
                return profile.reshape(-1)
            if profile.shape[0] == 1:
                return profile[0]
        raise ValueError("Profile must be CSV path, 1D array, or shape (1, T) array.")

    def get_emissions(self, power, nominal_capacity=None, la=np):
        if self.emission_rates is None:
            return 0.0
        total_emissions = 0.0
        for i in range(len(self.emission_rates)):
            total_emissions += la.sum(la.multiply(self.emission_rates[i], power[0][i]))
        return total_emissions

    def get_investment_cost(self, nominal_capacity=None, la=np):
        if self.capital_cost is None or nominal_capacity is None:
            return 0.0
        return la.sum(la.multiply(self.capital_cost, (nominal_capacity - self.nominal_capacity)))

    def sample_time(self, time_periods, original_time_horizon):
        dev = super().sample_time(time_periods, original_time_horizon)
        # No additional sampling needed for synthetic profiles as they're generated on demand
        return dev


@torch.jit.script
def _admm_prox_update(power: list[torch.Tensor], rho: float, lin_cost, quad_cost, pmin, pmax):
    # Problem is
    #     min_p    a (p - pmin)^2 + b (p - pmin) + (rho / 2) || (p - power) ||_2^2 + {box constraints}
    # Objective derivative is
    #     2 a (p - pmin) + b +  rho (p - power) = 0
    # Which is solved by
    #     p = (rho power + 2 a pmin - b) / (2 a + rho )
    num = rho * power[0] + 2 * quad_cost * pmin - lin_cost
    denom = 2 * quad_cost + rho
    p = torch.divide(num, denom)

    # Finally, we project onto the box constraints
    p = torch.clip(p, pmin, pmax)

    return [p], None, None

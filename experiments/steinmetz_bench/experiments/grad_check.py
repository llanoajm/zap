"""Gradient-vs-exact-dual check (roadmap item 1.2, Steinmetz §8.4.4).

The headline claim behind every gradient-based planning result in zap is that its
*adjoint* (implicit-differentiation) sensitivities are exact, not approximate. This
module validates that claim against the one thing that can certify it: linear-program
duality. For a solved DC-OPF, the envelope theorem says the derivative of the optimal
system cost with respect to any device parameter ``theta`` equals the partial
derivative of the Lagrangian at the optimum,

    d(cost*)/d(theta) = dL/d(theta)|*  =  d(cost)/d(theta)
                                          + sum_j  nu_j * d(h_j)/d(theta)
                                          + sum_i  mu_i * d(g_i)/d(theta),

evaluated with the *solver's own* equality duals ``nu`` and inequality duals ``mu``.
We compute the left-hand side two independent ways and require them to agree:

1. **Adjoint** — zap's :class:`~zap.layer.DispatchLayer` backward pass, which solves
   the KKT linear system (``kkt_vjp_variables`` then ``kkt_vjp_parameters``). This is
   the gradient zap hands to its planner; it never sees the solver's duals.
2. **Exact dual** — the envelope right-hand side, obtained by torch-autodiff of each
   device's own ``lagrangian(...)`` with the CVXPY duals substituted in as constants.

These are genuinely different code paths (a sparse linear solve vs. direct dual
substitution), so their agreement certifies the adjoint. We additionally re-derive the
same gradient by central finite differences (re-solving the dispatch with perturbed
capacities) as a third, solver-independent anchor, recorded in the result.

The check runs for all three device types the planner cares about — line capacity,
generator capacity, and battery power — across the canonical Garver 6-bus network
(generator + AC line; the §8.4.4 / paper-Fig-6 system) and the 7-bus toy network
(which adds a multi-period battery). The headline number is the worst-case relative
gradient error over all device types.

A subtlety worth recording rather than hiding: zap models an AC line's flow constraint
as ``power = susceptance * nominal_capacity * angle_diff``, so ``nominal_capacity``
scales *both* the thermal limit and the line's electrical conductance. The clean
"``d(cost)/d(cap) = -mu``" identity (a pure thermal-limit reading) therefore holds for
generator capacity — whose only parameter dependence is the capacity inequality — but
for an AC line the exact gradient also carries the power-flow equality dual. We report
the capacity-inequality (``-mu``) contribution alongside the full gradient so the
decomposition is visible; the acceptance is on the full envelope identity, which holds
for every device type.
"""

from __future__ import annotations

import copy

import cvxpy as cp
import numpy as np
import torch
from attrs import define, field

from zap.devices import ACLine, Battery, Generator
from zap.importers.toy import load_garver_network, load_test_network
from zap.layer import DispatchLayer

from experiments.steinmetz_bench.reports.result import BenchResult
from experiments.steinmetz_bench.scoring.metrics import FidelityBand, fidelity_band

EXPERIMENT_ID = "1.2-grad-vs-dual"
DATASET = "garver+toy7"

# Acceptance: worst-case relative error between the adjoint and the exact dual.
GRAD_REL_TOL = 1e-3
# The finite-difference anchor is intrinsically looser (re-solve noise + the LP's
# piecewise-linear kinks), so it gets its own, wider tolerance.
FD_REL_TOL = 1e-2
# A gradient entry counts as "active" (its relative error is meaningful) only when it
# is at least this fraction of the device's largest gradient magnitude. Structurally
# zero entries (slack constraints) are checked for absolute smallness instead.
ACTIVE_FRAC = 1e-3
# Step for the central finite-difference re-solve (capacities are O(10-100) MW).
FD_EPS = 1e-2
_DTYPE = torch.float64


def _torch(x):
    """Recursively convert a (possibly nested, possibly ``None``) array to float64."""
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return [_torch(e) for e in x]
    return torch.tensor(np.asarray(x, dtype=float), dtype=_DTYPE)


def _cost_gradient_outcome(z, devices):
    """Build ``dy`` = gradient of total operation cost w.r.t. the primal variables.

    The dispatch objective is ``sum_d operation_cost_d(power, angle, local_vars)``; its
    gradient w.r.t. the dual/price entries is zero. We fill the primal slots per device
    using each device's own (torch-backed) ``operation_cost_gradients`` so the adjoint
    differentiates exactly the cost zap minimized. The appended ground device has zero
    cost and is left untouched.
    """
    dy = z.package(np.zeros_like(z.vectorize()))
    for i, dev in enumerate(devices):
        gp, ga, gu = dev.torchify(dtype=_DTYPE).operation_cost_gradients(
            z.power[i], z.angle[i], z.local_variables[i]
        )
        for t in range(len(dy.power[i])):
            dy.power[i][t] = dy.power[i][t] + np.asarray(gp[t])
        if dy.angle[i] is not None and ga is not None:
            for t in range(len(dy.angle[i])):
                if ga[t] is not None:
                    dy.angle[i][t] = dy.angle[i][t] + np.asarray(ga[t])
        if gu is not None and dy.local_variables[i] is not None:
            for t in range(len(dy.local_variables[i])):
                if gu[t] is not None:
                    dy.local_variables[i][t] = dy.local_variables[i][t] + np.asarray(gu[t])
    return dy


def _exact_dual_gradient(dev, z, i, attr):
    """Envelope gradient ``dL/d(theta)`` with the solver's duals held constant."""
    tdev = dev.torchify(dtype=_DTYPE)
    theta = torch.tensor(
        np.asarray(getattr(dev, attr), dtype=float), dtype=_DTYPE, requires_grad=True
    )
    lagrangian = tdev.lagrangian(
        _torch(z.power[i]),
        _torch(z.angle[i]),
        _torch(z.local_variables[i]),
        _torch(z.local_equality_duals[i]),
        _torch(z.local_inequality_duals[i]),
        la=torch,
        **{attr: theta},
    )
    (grad,) = torch.autograd.grad(lagrangian, theta)
    return grad.detach().numpy().ravel()


def _capacity_dual_gradient(dev, z, i, attr):
    """The capacity-inequality (``-mu``) contribution to the gradient alone.

    This is ``d/d(theta) sum_i mu_i g_i(theta)`` — the part of the envelope gradient
    that comes purely from the capacity/limit inequalities. For a generator this equals
    the full gradient (the clean ``-mu`` identity); for an AC line it omits the
    power-flow equality dual and so is only the thermal-limit piece.
    """
    tdev = dev.torchify(dtype=_DTYPE)
    theta = torch.tensor(
        np.asarray(getattr(dev, attr), dtype=float), dtype=_DTYPE, requires_grad=True
    )
    ineqs = tdev.inequality_constraints(
        _torch(z.power[i]), _torch(z.angle[i]), _torch(z.local_variables[i]),
        la=torch, **{attr: theta},
    )
    duals = _torch(z.local_inequality_duals[i])
    weighted = sum(torch.sum(g * d) for g, d in zip(ineqs, duals))
    (grad,) = torch.autograd.grad(weighted, theta)
    return grad.detach().numpy().ravel()


def _system_cost(net, devices, solver):
    horizon = max(d.time_horizon for d in devices)
    out = net.dispatch(devices, time_horizon=horizon, solver=solver)
    if out.problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"dispatch failed: status={out.problem.status}")
    return float(out.problem.value)


def _finite_difference_gradient(net, devices, i, attr, flat_indices, solver, eps=FD_EPS):
    """Central finite difference of system cost w.r.t. selected entries of ``theta``."""
    shape = np.asarray(getattr(devices[i], attr)).shape
    grad = {}
    for j in flat_indices:
        def perturbed(sign):
            dev = copy.deepcopy(devices[i])
            v = np.asarray(getattr(dev, attr), dtype=float).copy()
            v.ravel()[j] += sign * eps
            setattr(dev, attr, v.reshape(shape))
            patched = list(devices)
            patched[i] = dev
            return _system_cost(net, patched, solver)

        grad[j] = (perturbed(+1) - perturbed(-1)) / (2.0 * eps)
    return grad


@define(kw_only=True)
class DeviceGradCheck:
    """Per (network, device-type) gradient comparison, all arrays computed from solves."""

    network: str
    device_type: str
    attr: str
    adjoint: np.ndarray
    exact_dual: np.ndarray
    capacity_dual: np.ndarray
    finite_difference: np.ndarray  # NaN where not evaluated (inactive/unbuilt entries)
    active_mask: np.ndarray
    max_rel_err_dual: float
    max_rel_err_fd: float
    max_abs_gap_inactive: float

    @property
    def n_active(self) -> int:
        return int(self.active_mask.sum())


def _relative_errors(a, b, active):
    if not np.any(active):
        return np.array([]), 0.0
    rel = np.abs(a[active] - b[active]) / np.abs(b[active])
    return rel, float(rel.max())


def check_parameter(net, devices, i, attr, device_type, solver=cp.CLARABEL, do_fd=True):
    """Run the adjoint / exact-dual / finite-difference comparison for one parameter."""
    horizon = max(d.time_horizon for d in devices)
    name = f"{device_type}_param"
    layer = DispatchLayer(
        net, devices, parameter_names={name: (i, attr)},
        time_horizon=horizon, solver=solver, add_ground=True,
    )
    values = {name: getattr(devices[i], attr)}
    z = layer.forward(**values)

    dy = _cost_gradient_outcome(z, devices)
    adjoint = np.asarray(layer.backward(z, dy, regularize=1e-8, **values)[name]).ravel()
    exact_dual = _exact_dual_gradient(devices[i], z, i, attr)
    capacity_dual = _capacity_dual_gradient(devices[i], z, i, attr)

    base = np.asarray(getattr(devices[i], attr), dtype=float).ravel()
    built = base > 1e-9  # an unbuilt (zero-capacity) line has a degenerate sensitivity
    scale = max(float(np.abs(exact_dual[built]).max()) if built.any() else 0.0, 1.0)
    active = built & (np.abs(exact_dual) > ACTIVE_FRAC * scale)

    inactive_built = built & ~active
    max_abs_gap_inactive = (
        float(np.abs(adjoint[inactive_built] - exact_dual[inactive_built]).max())
        if inactive_built.any() else 0.0
    )

    _, max_rel_err_dual = _relative_errors(adjoint, exact_dual, active)

    fd = np.full(base.size, np.nan)
    max_rel_err_fd = 0.0
    if do_fd:
        idx = np.where(active)[0]
        fd_vals = _finite_difference_gradient(net, devices, i, attr, idx, solver)
        for j, val in fd_vals.items():
            fd[j] = val
        if idx.size:
            rel_fd = np.abs(adjoint[idx] - fd[idx]) / np.maximum(np.abs(fd[idx]), 1e-9)
            max_rel_err_fd = float(rel_fd.max())

    return DeviceGradCheck(
        network=net_label(net, devices),
        device_type=device_type,
        attr=attr,
        adjoint=adjoint,
        exact_dual=exact_dual,
        capacity_dual=capacity_dual,
        finite_difference=fd,
        active_mask=active,
        max_rel_err_dual=max_rel_err_dual,
        max_rel_err_fd=max_rel_err_fd,
        max_abs_gap_inactive=max_abs_gap_inactive,
    )


def net_label(net, devices) -> str:
    return f"{net.num_nodes}bus/{len(devices)}dev"


def _index_of(devices, cls) -> int:
    return next(i for i, d in enumerate(devices) if isinstance(d, cls))


@define(kw_only=True)
class GradCheckReport:
    """All per-device checks plus the pooled active gradients for the fidelity band."""

    checks: list = field(factory=list)

    @property
    def max_rel_err_dual(self) -> float:
        return max((c.max_rel_err_dual for c in self.checks), default=0.0)

    @property
    def max_rel_err_fd(self) -> float:
        return max((c.max_rel_err_fd for c in self.checks), default=0.0)

    def per_device_type(self) -> dict:
        out: dict[str, float] = {}
        for c in self.checks:
            out[c.device_type] = max(out.get(c.device_type, 0.0), c.max_rel_err_dual)
        return out

    def _pooled_active(self):
        adj, dual = [], []
        for c in self.checks:
            adj.append(c.adjoint[c.active_mask])
            dual.append(c.exact_dual[c.active_mask])
        return np.concatenate(adj), np.concatenate(dual)

    def fidelity(self) -> FidelityBand:
        adj, dual = self._pooled_active()
        return fidelity_band(adj, dual, reference="exact-dual", metric="cost-gradient",
                             units="$/unit-capacity")

    def to_bench_result(self) -> BenchResult:
        per_type = self.per_device_type()
        band = self.fidelity()
        return BenchResult(
            experiment_id=EXPERIMENT_ID,
            dataset=DATASET,
            headline_number=self.max_rel_err_dual,
            units="relative",
            fidelity_band=band,
            assumptions={
                "identity": "envelope theorem: d(cost*)/d(theta) = dL/d(theta) with solver duals",
                "adjoint_source": "zap DispatchLayer.backward (KKT linear solve)",
                "dual_source": "device.lagrangian autodiff with CVXPY duals",
                "solver": "CLARABEL",
                "networks": {
                    "garver": "6-bus, generator + AC line (paper Fig. 6 system)",
                    "toy7": "7-bus, generator + AC line + multi-period battery",
                },
                "grad_rel_tol": GRAD_REL_TOL,
                "fd_rel_tol": FD_REL_TOL,
                "active_frac": ACTIVE_FRAC,
                "ac_line_note": (
                    "nominal_capacity scales both the thermal limit and susceptance, so "
                    "an AC line's exact gradient includes the power-flow equality dual, "
                    "not only the thermal -mu term"
                ),
            },
            sensitivities={
                "max_rel_err_dual": self.max_rel_err_dual,
                "max_rel_err_fd": self.max_rel_err_fd,
                "rel_err_by_device_type": per_type,
                "checks": [
                    {
                        "network": c.network,
                        "device_type": c.device_type,
                        "n_active": c.n_active,
                        "max_rel_err_dual": c.max_rel_err_dual,
                        "max_rel_err_fd": c.max_rel_err_fd,
                        "max_abs_gap_inactive": c.max_abs_gap_inactive,
                        "adjoint_active": c.adjoint[c.active_mask].tolist(),
                        "exact_dual_active": c.exact_dual[c.active_mask].tolist(),
                        "capacity_dual_active": c.capacity_dual[c.active_mask].tolist(),
                    }
                    for c in self.checks
                ],
            },
        )


def run_grad_check(solver=cp.CLARABEL, do_fd=True) -> GradCheckReport:
    """Run the gradient-vs-dual check across all three device types and both networks."""
    checks = []

    garver_net, garver_devices = load_garver_network()
    checks.append(check_parameter(
        garver_net, garver_devices, _index_of(garver_devices, ACLine),
        "nominal_capacity", "line", solver=solver, do_fd=do_fd))
    checks.append(check_parameter(
        garver_net, garver_devices, _index_of(garver_devices, Generator),
        "nominal_capacity", "generator", solver=solver, do_fd=do_fd))

    toy_net, toy_devices = load_test_network()
    checks.append(check_parameter(
        toy_net, toy_devices, _index_of(toy_devices, ACLine),
        "nominal_capacity", "line", solver=solver, do_fd=do_fd))
    checks.append(check_parameter(
        toy_net, toy_devices, _index_of(toy_devices, Generator),
        "nominal_capacity", "generator", solver=solver, do_fd=do_fd))
    checks.append(check_parameter(
        toy_net, toy_devices, _index_of(toy_devices, Battery),
        "power_capacity", "battery", solver=solver, do_fd=do_fd))

    return GradCheckReport(checks=checks)


def run(report_path=None, do_fd=True) -> BenchResult:
    """Run the check and emit (optionally write) a :class:`BenchResult`."""
    result = run_grad_check(do_fd=do_fd).to_bench_result()
    if report_path is not None:
        result.write_markdown(report_path)
    return result


if __name__ == "__main__":
    report = run_grad_check()
    print(f"{'network':<12} {'device':<10} {'n_act':>5} "
          f"{'relerr(dual)':>14} {'relerr(FD)':>12}")
    for c in report.checks:
        print(f"{c.network:<12} {c.device_type:<10} {c.n_active:>5} "
              f"{c.max_rel_err_dual:>14.3e} {c.max_rel_err_fd:>12.3e}")
    print(f"\nheadline max relative gradient error (adjoint vs exact dual): "
          f"{report.max_rel_err_dual:.3e}  (tol {GRAD_REL_TOL})")

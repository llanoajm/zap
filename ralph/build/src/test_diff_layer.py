"""
Verify zap's differentiable DispatchLayer works on a real GridSFM grid:
analytic gradient (implicit-function backward) vs. finite differences.

This is the technical core of Design A: gradients of a downstream objective
flow through the exact DC-OPF solve to the OPF parameters (and hence, in the
full design, to an upstream neural net).
"""
import sys
from copy import deepcopy

import numpy as np
import cvxpy as cp

import zap
from zap import DispatchLayer
from gridsfm_zap import load_case


def dev_index(devices, cls):
    for i, d in enumerate(devices):
        if isinstance(d, cls):
            return i
    raise ValueError(cls)


def main(model_path):
    dc_path = model_path.replace("_model.json", "_dc_results.json")
    case = load_case(model_path, dc_path)
    gi = dev_index(case.devices, zap.Generator)

    # Parameter under test: generator nominal_capacity (scales pmax).
    parameter_names = {"gen_capacity": (gi, "nominal_capacity")}
    layer = DispatchLayer(
        case.net, case.devices, parameter_names=parameter_names,
        time_horizon=1, solver=cp.CLARABEL,
    )
    theta0 = {"gen_capacity": np.asarray(getattr(case.devices[gi], "nominal_capacity"), dtype=float)}

    y0 = layer(**theta0)

    # Downstream objective J(y) = total generation cost = <linear_cost, p_gen>.
    dJ = y0.package(np.zeros_like(y0.vectorize()))
    lin = np.asarray(case.devices[gi].linear_cost).ravel()
    dJ.power[gi][0] = dJ.power[gi][0] + lin.reshape(dJ.power[gi][0].shape)

    def J(y):
        p = np.asarray(y.power[gi][0]).ravel()
        return float(np.dot(lin, p))

    J0 = J(y0)
    grad = layer.backward(y0, dJ, **theta0, regularize=1e-6)["gen_capacity"]
    grad = np.asarray(grad).ravel()

    # Finite-difference check along a random direction (robust to per-coord noise).
    rng = np.random.default_rng(0)
    v = rng.standard_normal(grad.shape)
    v /= np.linalg.norm(v)
    delta = 1e-3
    th1 = deepcopy(theta0)
    th1["gen_capacity"] = theta0["gen_capacity"] + delta * v.reshape(theta0["gen_capacity"].shape)
    Jp = J(layer(**th1))
    th2 = deepcopy(theta0)
    th2["gen_capacity"] = theta0["gen_capacity"] - delta * v.reshape(theta0["gen_capacity"].shape)
    Jm = J(layer(**th2))
    fd_dir = (Jp - Jm) / (2 * delta)
    an_dir = float(np.dot(grad, v))

    print(f"Case: {case.name}  n_bus={case.n_bus}  n_gen={grad.size}")
    print(f"J0 (zap gen cost)         = {J0:.6f}")
    print(f"directional grad  analytic = {an_dir:.6e}")
    print(f"directional grad  finite-d = {fd_dir:.6e}")
    denom = max(1e-9, abs(fd_dir) + abs(an_dir))
    rel = abs(an_dir - fd_dir) / denom
    print(f"relative error            = {rel:.3e}")
    print("PASS" if rel < 0.05 else "CHECK (LP kink or scaling; inspect)")


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "ralph/build/data/raw/04h/delaware_model.json"
    main(p)

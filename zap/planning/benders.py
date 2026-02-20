"""Benders decomposition solver for capacity planning."""

from copy import deepcopy

import cvxpy as cp
import numpy as np

from zap.layer import DispatchLayer
from zap.network import DispatchOutcome


class BendersSolver:
    """
    Benders decomposition for datacenter capacity planning.

    Solves: min_u  c^T u + dispatch_scalar * Q(u)
            s.t.   u_min <= u <= u_max
                   sum(u) = budget

    where Q(u) is the optimal dispatch cost given datacenter capacities u.
    The dispatch_scalar allows balancing investment vs operation costs.

    Infeasible subproblems are handled by a penalized (curtailable) fallback
    layer. DC loads can be shed in the fallback at VoLL cost, so the subproblem
    is always feasible. The master naturally avoids infeasible allocations
    because VoLL >> any real dispatch cost.
    """

    def __init__(
        self,
        layer: DispatchLayer,
        capital_cost: np.ndarray,
        budget: float,
        lower_bounds: dict,
        upper_bounds: dict,
        solver=cp.CLARABEL,
        dispatch_scalar: float = 1.0,
    ):
        """
        Args:
            layer: DispatchLayer with parameter_names mapping to DC nominal_capacity
            capital_cost: Investment cost vector c (shape: num_dc,)
            budget: Budget constraint B for sum(u) = B
            lower_bounds: Dict mapping param name to lower bound array
            upper_bounds: Dict mapping param name to upper bound array
            solver: cvxpy solver
        """
        self.layer = layer
        self.capital_cost = capital_cost
        self.budget = budget
        self.solver = solver
        self.dispatch_scalar = dispatch_scalar

        # Extract parameter info from layer
        # Assumes single parameter "dc_capacity" -> (device_idx, "nominal_capacity")
        assert len(layer.parameter_names) == 1, "BendersSolver supports single DC parameter"
        self.param_name = list(layer.parameter_names.keys())[0]
        self.dc_device_idx, self.attr_name = layer.parameter_names[self.param_name]

        # Get bounds as arrays
        self.lower_bounds = lower_bounds[self.param_name]
        self.upper_bounds = upper_bounds[self.param_name]
        self.num_dc = len(self.lower_bounds)

        # Get datacenter device and profile for cut generation
        self.dc_device = layer.devices[self.dc_device_idx]
        self.profile = self.dc_device.profile  # shape (num_dc, T)

        # Cut storage: list of (alpha, pi) tuples
        self.cuts = []

        # Build penalized (curtailable) fallback layer for infeasible subproblems.
        # DC loads can be shed at VoLL cost, so this layer is always feasible.
        penalized_devices = [deepcopy(d) for d in layer.devices]
        dc_dev = penalized_devices[self.dc_device_idx]
        dc_dev.max_power = np.zeros_like(dc_dev.max_power)  # allow curtailment

        self.penalized_layer = DispatchLayer(
            network=layer.network,
            devices=penalized_devices,
            parameter_names=layer.parameter_names,
            time_horizon=layer.time_horizon,
            solver=solver,
        )

    def solve_master(self) -> tuple[np.ndarray, float]:
        """
        Solve master problem:
            min_{u, eta}  c^T u + eta
            s.t.        u_min <= u <= u_max
                        sum(u) = budget
                        eta >= alpha^j + (pi^j)^T u  for all cuts j
                        eta >= 0

        Returns: (u_optimal, eta_optimal)
        """
        u = cp.Variable(self.num_dc)
        eta = cp.Variable()

        constraints = [
            u >= self.lower_bounds,
            u <= self.upper_bounds,
            cp.sum(u) == self.budget,
            eta >= 0,
        ]

        # Add Benders cuts
        for alpha_j, pi_j in self.cuts:
            constraints.append(eta >= alpha_j + pi_j @ u)

        objective = cp.Minimize(self.capital_cost @ u + self.dispatch_scalar * eta)
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=self.solver)

        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            raise RuntimeError(f"Master problem failed: {problem.status}")

        return u.value, self.dispatch_scalar * eta.value

    def solve_subproblem(self, u: np.ndarray) -> tuple[float, DispatchOutcome, bool]:
        """
        Solve dispatch subproblem Q(u) for given datacenter capacities.

        Falls back to the penalized (curtailable) layer if the primary dispatch
        is infeasible. Returns a flag indicating which layer was used.

        Returns: (Q_value, DispatchOutcome with duals, is_penalized)
        """
        kwargs = {self.param_name: u}
        is_penalized = False

        try:
            outcome = self.layer.forward(**kwargs)
            active_layer = self.layer
        except AssertionError:
            is_penalized = True
            outcome = self.penalized_layer.forward(**kwargs)
            active_layer = self.penalized_layer

        parameters = active_layer.setup_parameters(**kwargs)
        Q = active_layer.network.operation_cost(
            active_layer.devices,
            outcome.power,
            outcome.angle,
            outcome.local_variables,
            parameters=parameters,
        )

        return Q, outcome, is_penalized

    def generate_cut(self, u: np.ndarray, Q: float, outcome: DispatchOutcome):
        """
        Generate Benders optimality cut from subproblem duals.

        Cut: η >= Q(u^k) + ∂Q/∂u|_{u^k} · (u - u^k)
             η >= α + π^T u

        where: π = ∂Q/∂u = Σ_t profile * (λ_1 - λ_0)
               α = Q - π^T u
        """
        # Extract duals for datacenter device inequality constraints
        # local_inequality_duals[device_idx][constraint_idx] -> shape (num_dc, T)
        lambda_0 = outcome.local_inequality_duals[self.dc_device_idx][0]  # g_0 dual
        lambda_1 = outcome.local_inequality_duals[self.dc_device_idx][1]  # g_1 dual

        # Gradient: ∂Q/∂u[i] = Σ_t profile[i,t] * (λ_1[i,t] - λ_0[i,t])
        pi = np.sum(self.profile * (lambda_1 - lambda_0), axis=1)

        # Intercept: α = Q - π^T u
        alpha = Q - pi @ u

        self.cuts.append((alpha, pi))
        return alpha, pi

    def solve(
        self,
        initial_u: np.ndarray = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        verbose: bool = True,
    ) -> dict:
        """
        Main Benders decomposition loop.

        Args:
            initial_u: Optional starting point for first iteration.
                       If None, master problem finds its own starting point.
            max_iter: Maximum Benders iterations
            tol: Convergence tolerance on relative gap
            verbose: Print progress

        Returns dict with:
            - 'u': optimal datacenter capacities
            - 'objective': optimal total cost (investment + dispatch)
            - 'num_iterations': iterations until convergence
            - 'history': dict with LB, UB, gap, u, Q, is_penalized per iteration
        """
        LB = -np.inf  # Lower bound from master
        UB = np.inf  # Upper bound from subproblem
        history = {"LB": [], "UB": [], "gap": [], "u": [], "Q": [], "is_penalized": []}

        for k in range(max_iter):
            # Step 1: Solve master problem (or use initial_u for first iteration)
            if k == 0 and initial_u is not None:
                u_k = initial_u
                eta_k = 0.0  # No cuts yet, so eta lower bound is 0
            else:
                u_k, eta_k = self.solve_master()

            inv_cost_k = self.capital_cost @ u_k
            LB = inv_cost_k + eta_k

            # Step 2: Solve subproblem (with penalized fallback for infeasible allocations)
            Q_k, outcome, is_penalized = self.solve_subproblem(u_k)

            # Only update UB on feasible (non-penalized) subproblems
            if not is_penalized:
                total_cost_k = inv_cost_k + self.dispatch_scalar * Q_k
                UB = min(UB, total_cost_k)

            # Step 3: Check convergence
            gap = (UB - LB) / max(abs(UB), 1e-10) if UB < np.inf else np.inf

            # Record history
            history["LB"].append(LB)
            history["UB"].append(UB)
            history["gap"].append(gap)
            history["u"].append(u_k.copy())
            history["Q"].append(Q_k)
            history["is_penalized"].append(is_penalized)

            if verbose:
                status = " [INFEASIBLE->penalized]" if is_penalized else ""
                print(
                    f"Iter {k}: LB={LB:.4f}, UB={UB:.4f}, gap={gap:.2%}, "
                    f"inv={inv_cost_k:.4f}, Q={Q_k:.4f}{status}"
                )

            if gap < tol:
                if verbose:
                    print(f"Converged in {k + 1} iterations!")
                break

            # Step 4: Generate Benders cut (penalized duals are valid for cut generation)
            self.generate_cut(u_k, Q_k, outcome)

        return {
            "u": u_k,
            "objective": UB,
            "investment_cost": self.capital_cost @ u_k,
            "dispatch_cost": Q_k,
            "scaled_dispatch_cost": self.dispatch_scalar * Q_k,
            "num_iterations": k + 1,
            "num_cuts": len(self.cuts),
            "final_gap": gap,
            "history": history,
        }

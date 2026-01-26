import cvxpy as cp
import numpy as np


class Projection:
    def __call__(self, x):
        raise NotImplementedError


class SimplexBudgetProjection(Projection):
    def __init__(self, budget, strict=True):
        self.budget = budget
        self.strict = strict

    def __call__(self, x):
        """
        Simplex projection algorithm from Duchi et al. (2008)
        https://ai.stanford.edu/~jduchi/projects/jd_ss_ys_l1.pdf
        """
        x = np.maximum(x, 0.0)
        s = x.sum()
        if (self.strict and abs(s - self.budget) < 1e-12) or (not self.strict and s <= self.budget):
            return x
        u = np.sort(x)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, len(u) + 1) > (cssv - self.budget))[0][-1]
        theta = (cssv[rho] - self.budget) / (rho + 1)
        return np.maximum(x - theta, 0.0)


class BoxBudgetProjection(Projection):
    """Project onto intersection of box constraints and budget equality."""

    def __init__(self, budget, lower_bounds, upper_bounds):
        self.budget = budget
        self.lower_bounds = np.asarray(lower_bounds).ravel()
        self.upper_bounds = np.asarray(upper_bounds).ravel()
        self.n = len(self.lower_bounds)

        # Check feasibility
        if np.sum(self.lower_bounds) > budget + 1e-9:
            raise ValueError(f"Budget {budget} too small for sum of lower bounds {np.sum(self.lower_bounds)}")
        if np.sum(self.upper_bounds) < budget - 1e-9:
            raise ValueError(f"Budget {budget} too large for sum of upper bounds {np.sum(self.upper_bounds)}")

    def __call__(self, y):
        y = np.asarray(y).ravel()
        x = cp.Variable(self.n)
        objective = cp.Minimize(cp.sum_squares(x - y))
        constraints = [
            x >= self.lower_bounds,
            x <= self.upper_bounds,
            cp.sum(x) == self.budget
        ]
        problem = cp.Problem(objective, constraints)
        problem.solve()
        return x.value

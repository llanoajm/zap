"""Scoring primitives for the Steinmetz benchmark suite.

Four pure-numpy building blocks the experiments and backtests share:

- :func:`counterfactual_delta` — the value created by a change: ``factual`` minus
  its ``counterfactual``, element-wise (e.g. baseline-node $/MWh minus chosen-node
  $/MWh, or "actual" dispatch cost minus least-cost dispatch cost).
- :func:`bootstrap_ci` — a percentile bootstrap confidence interval on any
  statistic of a sample, returned as a ``lo <= mid <= hi`` triple.
- :func:`duration_curve` — values sorted descending against their exceedance
  fraction (the classic load/price duration curve), monotone non-increasing.
- :func:`fidelity_band` — summary gap statistics between a DC (zap) vector and a
  reference vector (PyPSA-DC or realized), the band attached to every result.

Everything here is deterministic given a seed and free of zap/solver imports so it
can be unit-tested without a solve.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from attrs import define


@define
class CIResult:
    """A confidence interval on a statistic: ``lo <= mid <= hi`` by construction."""

    lo: float
    mid: float
    hi: float
    confidence: float

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.lo, self.mid, self.hi)

    def to_dict(self) -> dict:
        return {
            "lo": self.lo,
            "mid": self.mid,
            "hi": self.hi,
            "confidence": self.confidence,
        }


@define
class DurationCurve:
    """Values sorted high-to-low against the fraction of samples that exceed them."""

    value: np.ndarray  # descending
    exceedance: np.ndarray  # fraction of samples >= this value, in (0, 1]

    def percentile(self, p: float) -> float:
        """Value not exceeded ``p`` percent of the time (p in [0, 100])."""
        return float(np.percentile(self.value, p))

    def to_dict(self) -> dict:
        return {"value": self.value.tolist(), "exceedance": self.exceedance.tolist()}


@define
class FidelityBand:
    """DC-vs-reference gap summary attached to a :class:`BenchResult`."""

    reference: str  # e.g. "pypsa-dc", "realized-lmp"
    metric: str  # e.g. "lmp", "flow", "cost"
    units: str
    max_abs_gap: float
    mean_abs_gap: float
    p90_abs_gap: float
    n: int

    def to_dict(self) -> dict:
        return {
            "reference": self.reference,
            "metric": self.metric,
            "units": self.units,
            "max_abs_gap": self.max_abs_gap,
            "mean_abs_gap": self.mean_abs_gap,
            "p90_abs_gap": self.p90_abs_gap,
            "n": self.n,
        }


def counterfactual_delta(factual, counterfactual):
    """Return ``factual - counterfactual`` element-wise.

    Scalars return a float; array-likes return an ``ndarray``. This is the
    convention every backtest uses for "value created" (positive = the
    counterfactual change saved money / lowered cost).
    """
    fac = np.asarray(factual, dtype=float)
    cf = np.asarray(counterfactual, dtype=float)
    delta = fac - cf
    if delta.ndim == 0:
        return float(delta)
    return delta


def bootstrap_ci(
    samples,
    statistic: Callable[[np.ndarray], np.ndarray] = np.mean,
    n_boot: int = 2000,
    confidence: float = 0.90,
    seed: int = 0,
) -> CIResult:
    """Percentile-bootstrap CI on ``statistic`` of ``samples``.

    Resamples ``samples`` with replacement ``n_boot`` times, evaluates
    ``statistic`` per resample, and reads percentiles off the bootstrap
    distribution. ``mid`` is the bootstrap median, so ``lo <= mid <= hi`` always
    holds. ``statistic`` must accept an ``axis`` argument (``np.mean``,
    ``np.median``, ``np.sum`` … all do).
    """
    data = np.asarray(samples, dtype=float).ravel()
    n = data.size
    if n == 0:
        raise ValueError("bootstrap_ci needs at least one sample")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = np.asarray(statistic(data[idx], axis=1), dtype=float).ravel()

    alpha = (1.0 - confidence) / 2.0
    lo = float(np.percentile(boot, 100.0 * alpha))
    mid = float(np.percentile(boot, 50.0))
    hi = float(np.percentile(boot, 100.0 * (1.0 - alpha)))
    return CIResult(lo=lo, mid=mid, hi=hi, confidence=confidence)


def duration_curve(values) -> DurationCurve:
    """Build a duration curve: values sorted descending vs exceedance fraction.

    The returned ``value`` array is monotone non-increasing. ``exceedance[i]`` is
    the fraction of samples greater than or equal to ``value[i]`` — it runs from
    ``1/n`` up to ``1.0`` as the curve descends.
    """
    data = np.asarray(values, dtype=float).ravel()
    if data.size == 0:
        raise ValueError("duration_curve needs at least one value")
    ordered = np.sort(data)[::-1]
    n = ordered.size
    exceedance = np.arange(1, n + 1, dtype=float) / n
    return DurationCurve(value=ordered, exceedance=exceedance)


def fidelity_band(
    dc_values,
    reference_values,
    reference: str,
    metric: str,
    units: str = "",
) -> FidelityBand:
    """Summarize the absolute gap between a DC (zap) vector and a reference.

    Both inputs are flattened and must share a length. Records max, mean, and p90
    absolute gaps — the band the whitepaper attaches to each headline so a reader
    can see the DC-approximation error rather than trusting a point estimate.
    """
    dc = np.asarray(dc_values, dtype=float).ravel()
    ref = np.asarray(reference_values, dtype=float).ravel()
    if dc.shape != ref.shape:
        raise ValueError(
            f"dc_values and reference_values must match in length; "
            f"got {dc.shape} and {ref.shape}"
        )
    if dc.size == 0:
        raise ValueError("fidelity_band needs at least one value")

    gap = np.abs(dc - ref)
    return FidelityBand(
        reference=reference,
        metric=metric,
        units=units,
        max_abs_gap=float(gap.max()),
        mean_abs_gap=float(gap.mean()),
        p90_abs_gap=float(np.percentile(gap, 90.0)),
        n=int(dc.size),
    )

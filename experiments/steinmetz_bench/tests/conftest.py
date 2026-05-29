"""Shared pytest fixtures for the Steinmetz benchmark suite.

Running every Phase 1-3 experiment is the single most expensive thing the suite does
(~80s for a full pass — each module runs real zap solves). Several tests need the same
set of ``BenchResult`` objects: item 4.1 checks every result's fidelity band, item 4.2's
master report assembles them all. Without sharing, each consumer re-runs the whole pass
and the suite blows the loop's 10-minute verify budget.

The ``bench_results`` fixture runs that pass exactly once per session and hands the same
mapping to every consumer, so the suite stays well under budget while each experiment is
still solved for real (no caching of stale numbers).
"""

import pytest

from experiments.steinmetz_bench import run_all


@pytest.fixture(scope="session")
def bench_results():
    """Run every experiment's synthetic entrypoint once; share the results session-wide.

    Returns a mapping ``{EXPERIMENT_ID: BenchResult | None}`` (only the cache-gated GPU
    benchmark may be ``None``, and only when no Modal run has been dispatched).
    """
    return run_all.collect_results(synthetic=True)

"""Steinmetz benchmark suite.

Synthetic-first verification + quantification-of-value harness for zap's
differentiable DC-OPF (§7 backtests + §8.4 benchmarks). Every result number is
computed by code from an actual zap solve; nothing here makes a live market-data
call. Real data is staged by a human into ``data/`` and re-run with ``--real``.
"""

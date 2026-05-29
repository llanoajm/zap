"""Smoke test: the benchmark package and its subpackages import, and zap is reachable."""

import importlib


def test_package_imports():
    pkg = importlib.import_module("experiments.steinmetz_bench")
    assert pkg.__doc__ is not None


def test_subpackages_import():
    for sub in ("datasets", "scoring", "experiments", "reports"):
        importlib.import_module(f"experiments.steinmetz_bench.{sub}")


def test_zap_is_importable():
    zap = importlib.import_module("zap")
    assert hasattr(zap, "__version__")

"""Test-time shims for the zap suite.

Pandas 3.0 enabled Copy-on-Write by default, which makes
``DataFrame.values`` / ``Series.values`` return read-only ndarrays. Several
spots in ``zap.importers.pypsa`` and ``zap.devices.injector`` rely on
in-place ``+=`` / ``/=`` on those arrays, so importing any real PyPSA network
under pandas 3.0 explodes with ``ValueError: output array is read-only``.

The grid-app side ships an equivalent shim in
``grid-app/scripts/_pypsa_compat.py``; we mirror it here so the zap test
suite can also exercise the PyPSA importer path under pandas 3.0 without
having to patch every individual ``.values`` call in zap. The patch is
inert under older pandas (where ``.values`` is already writable).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _writable(arr):
    if isinstance(arr, np.ndarray) and not arr.flags.writeable:
        return arr.copy()
    return arr


_orig_df_values = pd.DataFrame.values.fget
_orig_series_values = pd.Series.values.fget


def _df_values(self):
    return _writable(_orig_df_values(self))


def _series_values(self):
    return _writable(_orig_series_values(self))


pd.DataFrame.values = property(_df_values)
pd.Series.values = property(_series_values)

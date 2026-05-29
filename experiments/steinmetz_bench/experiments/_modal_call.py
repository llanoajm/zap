"""System-python Modal dispatcher for item 2.5 — NOT importable by the test suite.

This module is the *only* place that talks to Modal. It is invoked as a
subprocess under the system interpreter (``/usr/bin/python3``, which has the
``modal`` client installed) by :func:`bench_gpu_modal.dispatch`; the project
venv deliberately has no ``modal`` module, so the per-item pytest verify can
never reach the GPU even by accident.

It calls the EXISTING deployed ``zap-opf-solver`` app (grid-app/infra/modal/
solver_app.py, function ``solve_direct``) via ``modal.Function.from_name`` so no
image rebuild / re-deploy happens here — we only dispatch one solve.

Usage::

    /usr/bin/python3 _modal_call.py <network.nc> <args.json>

Reads the netCDF bytes from ``<network.nc>`` and the ADMM args dict from
``<args.json>``, runs ``solve_direct.remote(nc_bytes, args, {})`` on the GPU,
and prints the JSON-serialisable result dict to stdout.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

APP_NAME = "zap-opf-solver"
FUNCTION_NAME = "solve_direct"


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: _modal_call.py <network.nc> <args.json>", file=sys.stderr)
        return 2
    nc_path, args_path = argv[1], argv[2]
    nc_bytes = Path(nc_path).read_bytes()
    args = json.loads(Path(args_path).read_text())

    fn = modal.Function.from_name(APP_NAME, FUNCTION_NAME)
    result = fn.remote(nc_bytes, args, {})

    sys.stdout.write(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

# `data/` — human-staged inputs (not committed)

The Steinmetz benchmark loop runs entirely on **synthetic generators + bundled
toy/Garver networks**. Nothing in the loop writes here or makes a live
market-data call. This directory is the drop point for the **real** inputs a
human stages by hand to turn the synthetic harness into publishable dollar
numbers (re-run the same scripts with `--real`).

Loaders read `data/<name>/`. When a requested cache is absent they raise
`DataNotStagedError` (a clean block) rather than hanging, retrying, or
downloading.

## What to stage here

- `data/iso_lmp/` — ISO LMP / load / congestion snapshots (e.g. ERCOT West, PJM)
  pulled via `gridstatus` for the backtest windows.
- `data/topology/` — PyPSA-USA topology/fleet, RTS-GMLC, relevant `pglib-opf` cases.
- `data/rtep/` — a PJM RTEP (or MISO MTEP) vintage + its approved-project list (§7.3).
- `data/cenace/` — CENACE PML history + a PRODESEN corridor list (§7.4).
- `data/gpu_runs/` — cached JSON from the one Modal/H100 dispatch (item 2.5).
  Written once by `bench_gpu_modal.py`; tests read only the cache, never Modal.

Everything under `data/` except this README is git-ignored.

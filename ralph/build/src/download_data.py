"""
Idempotent downloader for the GridSFM US power-grid dataset.
Skips files that already exist. Safe to re-run.

Curated split (small->medium grids, CPU-feasible):
  TRAIN: a spread of ~18 states across sizes/regions
  TEST : held-out states for cross-topology generalization (never seen in training)
"""
import os
import sys
import urllib.request

BASE = "https://huggingface.co/datasets/microsoft/GridSFM_US_power_grid/resolve/main"
HOURS = ["04h", "16h"]
FILE_TYPES = ["model", "dc_results", "ac_results"]

TRAIN = [
    "rhode_island", "delaware", "vermont", "maine", "new_hampshire",
    "new_mexico", "west_virginia", "wyoming", "idaho", "maryland",
    "utah", "south_dakota", "north_dakota", "new_jersey", "nevada",
    "montana", "nebraska", "alabama",
]
TEST = ["connecticut", "massachusetts", "oregon", "mississippi", "kansas"]


def fetch(region, hour, ftype, outdir):
    fname = f"{region}_{ftype}.json"
    url = f"{BASE}/{hour}/{fname}"
    dst_dir = os.path.join(outdir, hour)
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, fname)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return "skip"
    try:
        urllib.request.urlretrieve(url, dst)
        return "ok" if os.path.getsize(dst) > 0 else "empty"
    except Exception as e:
        return f"err:{e}"


def main(outdir="ralph/build/data/raw"):
    regions = sorted(set(TRAIN + TEST))
    n_ok = n_skip = n_err = 0
    for region in regions:
        for hour in HOURS:
            for ftype in FILE_TYPES:
                r = fetch(region, hour, ftype, outdir)
                if r == "ok":
                    n_ok += 1
                elif r == "skip":
                    n_skip += 1
                else:
                    n_err += 1
                    print(f"  {region}/{hour}/{ftype}: {r}")
    # Write the split manifest for downstream consumers.
    import json
    manifest = {"train": TRAIN, "test": TEST, "hours": HOURS, "outdir": outdir}
    with open(os.path.join(outdir, "split.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"download done: ok={n_ok} skip={n_skip} err={n_err}")
    print(f"train={len(TRAIN)} states, test={len(TEST)} states, hours={HOURS}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ralph/build/data/raw")

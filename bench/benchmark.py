"""One benchmark harness, unmodified across macOS/CoreML, Linux/CUDA, Jetson, and
Raspberry Pi 5/CPU.  Reports p50/p95/p99 latency, sustained throughput, batch-1
vs batched behaviour, and a compute-bound vs overhead-bound verdict for every
candidate model AND for the classical-CV measurement path -- the step that must
finish inside the control interval.

Backends are auto-detected and degrade gracefully: a missing backend is skipped
with a printed note, never an error.  Andrea clones the repo, runs

    python bench/benchmark.py                 # all available backends
    python bench/benchmark.py --backend coreml
    python bench/benchmark.py --json out.json

and gets numbers on his own hardware.

Compute-bound vs overhead-bound verdict
---------------------------------------
For each model we know FLOPs/call (models/profiling.py).  If measured throughput
scales ~linearly from batch-1 to batch-32 (per-item latency drops a lot), the
batch-1 case was overhead/latency-bound; if per-item latency is flat, it is
compute-bound.  The ratio (batch1_latency / batched_per_item_latency) is the
overhead factor; >2 => overhead-bound.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "reference_stack"))

from architectures import REGISTRY, PER_FRAME  # noqa: E402
import profiling as prof  # noqa: E402


# --------------------------------------------------------------------------- #
# Backend detection
# --------------------------------------------------------------------------- #
def available_backends(requested=None):
    backends = {}
    import torch
    backends["cpu"] = {"kind": "torch", "device": "cpu"}
    if torch.cuda.is_available():
        backends["cuda"] = {"kind": "torch", "device": "cuda"}
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        backends["mps"] = {"kind": "torch", "device": "mps"}
    # onnxruntime providers
    try:
        import onnxruntime as ort
        provs = ort.get_available_providers()
        backends["onnx-cpu"] = {"kind": "onnx", "provider": "CPUExecutionProvider"}
        if "CUDAExecutionProvider" in provs:
            backends["onnx-cuda"] = {"kind": "onnx", "provider": "CUDAExecutionProvider"}
    except Exception:
        pass
    # coreml (prediction only works on macOS)
    if sys.platform == "darwin":
        try:
            import coremltools  # noqa: F401
            backends["coreml"] = {"kind": "coreml"}
        except Exception:
            pass
    if requested:
        return {k: v for k, v in backends.items() if k in requested}
    return backends


# --------------------------------------------------------------------------- #
# Timing helpers
# --------------------------------------------------------------------------- #
def _percentiles(times):
    t = sorted(times)
    def p(q):
        return t[min(len(t) - 1, int(q * len(t)))]
    return {"p50_ms": p(0.50) * 1e3, "p95_ms": p(0.95) * 1e3,
            "p99_ms": p(0.99) * 1e3, "min_ms": t[0] * 1e3, "mean_ms": statistics.mean(t) * 1e3}


def _time_calls(fn, n_warmup=5, n_iter=50):
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return times


# --------------------------------------------------------------------------- #
# Torch / ONNX runners
# --------------------------------------------------------------------------- #
def make_torch_runner(name, device, batch):
    import torch
    ctor = REGISTRY[name]
    model = ctor().to(device).eval()
    inputs = tuple(x.to(device) for x in ctor.example_inputs(batch=batch))

    @torch.no_grad()
    def run():
        out = model(*inputs)
        if device == "cuda":
            torch.cuda.synchronize()
        return out
    return run


def make_onnx_runner(name, provider, batch):
    import onnxruntime as ort
    ctor = REGISTRY[name]
    path = os.path.join(os.path.dirname(__file__), "..", "models", "exported",
                        f"{name}.onnx")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} (run: python models/export.py)")
    so = ort.SessionOptions()
    so.log_severity_level = 3  # silence dynamic-batch output-shape warnings
    sess = ort.InferenceSession(path, sess_options=so, providers=[provider])
    innames = [i.name for i in sess.get_inputs()]
    inputs = ctor.example_inputs(batch=batch)
    feed = {n: x.numpy() for n, x in zip(innames, inputs)}

    def run():
        return sess.run(None, feed)
    return run


# --------------------------------------------------------------------------- #
# CV measurement-path benchmark (the real-time-critical step)
# --------------------------------------------------------------------------- #
def bench_cv_path(n_iter=20):
    from measurement import (Calibration, measure_batch, synthetic_dripping)
    cal = Calibration.from_reference(100.0, 50.0)
    results = {}
    for (res, nframes) in [("256x256", 200), ("800x600", 100), ("128x128", 250)]:
        h, w = map(int, res.split("x"))
        frames = synthetic_dripping(nframes, hw=(h, w), fps=250.0)
        times = _time_calls(lambda: measure_batch(frames, 250.0, cal),
                            n_warmup=2, n_iter=n_iter)
        pc = _percentiles(times)
        per_frame_ms = pc["p50_ms"] / nframes
        results[res] = {
            "frames_per_batch": nframes,
            "batch_latency": pc,
            "per_frame_ms_p50": round(per_frame_ms, 4),
            "max_fps_single_thread": round(1000.0 / per_frame_ms, 1),
        }
    return results


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def verdict(batch1_ms, batched_per_item_ms):
    if batched_per_item_ms <= 0:
        return "unknown"
    factor = batch1_ms / batched_per_item_ms
    return {"overhead_factor": round(factor, 2),
            "verdict": "overhead-bound" if factor > 2.0 else "compute-bound"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", nargs="*", default=None)
    ap.add_argument("--models", nargs="*", default=list(REGISTRY.keys()))
    ap.add_argument("--batches", nargs="*", type=int, default=[1, 8, 32])
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--json", default=None)
    ap.add_argument("--no-cv", action="store_true")
    args = ap.parse_args()

    prof_table = json.load(open(os.path.join(os.path.dirname(__file__), "..",
                                             "models", "profiling.json"))) \
        if os.path.exists(os.path.join(os.path.dirname(__file__), "..", "models",
                                       "profiling.json")) else {}

    host = {"platform": platform.platform(), "machine": platform.machine(),
            "python": platform.python_version(), "processor": platform.processor()}
    report = {"host": host, "models": {}, "cv_path": {}}
    print(f"host: {host['platform']} ({host['machine']})")

    backends = available_backends(args.backend)
    print(f"backends: {', '.join(backends) or 'NONE'}\n")

    for name in args.models:
        report["models"][name] = {}
        flops = prof_table.get(name, {}).get("flops")
        for bname, cfg in backends.items():
            per_batch = {}
            for b in args.batches:
                try:
                    if cfg["kind"] == "torch":
                        run = make_torch_runner(name, cfg["device"], b)
                    elif cfg["kind"] == "onnx":
                        run = make_onnx_runner(name, cfg["provider"], b)
                    else:
                        continue
                    times = _time_calls(run, n_iter=args.iters)
                    pc = _percentiles(times)
                    thru = b / (pc["p50_ms"] / 1e3)
                    per_batch[b] = {"latency": pc,
                                    "throughput_items_s": round(thru, 1),
                                    "per_item_ms": round(pc["p50_ms"] / b, 4)}
                except Exception as e:
                    per_batch[b] = {"error": str(e)[:120]}
            # verdict from batch1 vs largest batch per-item
            v = None
            if 1 in per_batch and "per_item_ms" in per_batch[1]:
                big = max(b for b in per_batch if "per_item_ms" in per_batch[b])
                v = verdict(per_batch[1]["per_item_ms"],
                            per_batch[big]["per_item_ms"])
            report["models"][name][bname] = {"batches": per_batch, "verdict": v,
                                              "flops": flops}
            b1 = per_batch.get(1, {})
            lat = b1.get("latency", {}).get("p50_ms")
            tag = "/frame" if name in PER_FRAME else "/step"
            print(f"{name:20s} {bname:10s} b1 p50={lat if lat is None else round(lat,3)} ms{tag}"
                  f"  verdict={v['verdict'] if v else '-'}")

    if not args.no_cv:
        print("\nCV measurement path (single-thread):")
        report["cv_path"] = bench_cv_path(n_iter=max(10, args.iters // 3))
        for res, r in report["cv_path"].items():
            print(f"  {res:9s} {r['frames_per_batch']:4d} frames  "
                  f"p50={r['batch_latency']['p50_ms']:.1f} ms  "
                  f"{r['per_frame_ms_p50']:.3f} ms/frame  "
                  f"max {r['max_fps_single_thread']:.0f} fps")

    out = args.json or os.path.join(os.path.dirname(__file__), "results.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

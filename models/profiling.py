"""Compute params, MACs/FLOPs, and memory footprint for every candidate model.

Self-contained: registers forward hooks on the leaf modules we use (Conv2d,
Linear, LSTM, GRU) and counts multiply-accumulates analytically, so it needs no
fvcore/thop and runs anywhere.  FLOPs = 2 x MACs (one multiply + one add).

Numbers feed docs/architecture.md.  Run:  python models/profiling.py
"""
from __future__ import annotations

import json
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from architectures import REGISTRY  # noqa: E402


def _conv_macs(m, inp, out):
    # out_elems * (in_ch/groups * kh * kw)
    out_elems = out.numel() // out.shape[0]  # per-sample
    kh, kw = m.kernel_size
    return out_elems * (m.in_channels // m.groups * kh * kw)


def _linear_macs(m, inp, out):
    return (out.numel() // out.shape[0]) * m.in_features


def _rnn_macs(m, inp, out):
    # LSTM: 4 gates; GRU: 3 gates. per timestep per layer: gates*hidden*(in+hidden)
    x = inp[0]
    if x.dim() == 3:
        b, T, _ = x.shape
    else:
        T = 1
    gates = 4 if isinstance(m, nn.LSTM) else 3
    macs = 0
    in_size = m.input_size
    for layer in range(m.num_layers):
        macs += gates * m.hidden_size * (in_size + m.hidden_size)
        in_size = m.hidden_size
    return macs * T  # per-sample over the sequence


HANDLERS = {nn.Conv2d: _conv_macs, nn.Linear: _linear_macs,
            nn.LSTM: _rnn_macs, nn.GRU: _rnn_macs}


def profile_model(model, inputs):
    macs = {"total": 0}
    acts = {"peak_bytes": 0}
    handles = []

    def make_hook(mod):
        def hook(m, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            fn = HANDLERS[type(m)]
            macs["total"] += fn(m, inp, o)
            acts["peak_bytes"] = max(acts["peak_bytes"],
                                     o.element_size() * o.numel())
        return hook

    for mod in model.modules():
        if type(mod) in HANDLERS:
            handles.append(mod.register_forward_hook(make_hook(mod)))

    model.eval()
    with torch.no_grad():
        model(*inputs)
    for h in handles:
        h.remove()

    params = sum(p.numel() for p in model.parameters())
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return {
        "params": params,
        "param_mem_MB_fp32": round(param_bytes / 1e6, 4),
        "param_mem_MB_int8": round(param_bytes / 4 / 1e6, 4),
        "macs": macs["total"],
        "flops": 2 * macs["total"],
        "gflops": round(2 * macs["total"] / 1e9, 5),
        "peak_activation_MB": round(acts["peak_bytes"] / 1e6, 4),
    }


def main():
    rows = {}
    print(f"{'model':22s}{'params':>10s}{'MFLOPs/call':>13s}"
          f"{'param MB':>10s}{'act MB':>9s}")
    for name, ctor in REGISTRY.items():
        model = ctor().eval()
        inputs = ctor.example_inputs(batch=1)
        p = profile_model(model, inputs)
        rows[name] = p
        print(f"{name:22s}{p['params']:>10,d}{p['flops']/1e6:>13.3f}"
              f"{p['param_mem_MB_fp32']:>10.3f}{p['peak_activation_MB']:>9.3f}")

    # derived: throughput budgets
    # per-frame models: FLOPs/frame; per-step models: FLOPs/control-step
    out = os.path.join(os.path.dirname(__file__), "profiling.json")
    with open(out, "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\nwrote {out}")

    # illustrative budget arithmetic (compute-bound ceiling, ignoring overhead)
    print("\ncompute-bound ceiling (fp32), assuming a device can sustain X GFLOP/s:")
    for gflops_s, dev in [(50, "RPi5 CPU ~50 GFLOP/s fp32"),
                          (1500, "Jetson Orin Nano ~1.5 TFLOP/s fp16")]:
        rc = rows["regime_classifier"]["flops"]
        max_fps = gflops_s * 1e9 / rc
        print(f"  {dev:32s}: regime_classifier ceiling ~{max_fps:,.0f} fps "
              f"(compute only; real fps is overhead-bound, see bench/)")


if __name__ == "__main__":
    main()

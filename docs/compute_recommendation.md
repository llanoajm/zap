# Compute recommendation

**Buy: a Raspberry Pi 5 (8 GB), ~$150 all-in.** One board, one command, numbers on
your own hardware: `python bench/benchmark.py`. This is where the
Pi-vs-Jetson-vs-FPGA question gets its answer; the reasoning is in
`docs/decisions.md`, the arithmetic is here.

| item | price (USD) |
|---|---|
| Raspberry Pi 5, 8 GB | 80 |
| Active cooler | 5 |
| 27 W USB-C PSU | 12 |
| NVMe HAT + 256 GB NVMe SSD (trajectory logs) | 40 |
| Case | 10 |
| **compute total** | **~147** |

The ZWO ASI174MM (~$450, already owned) and the Harvard pumps are separate and
unchanged by this choice. For comparison the Jetson Orin Nano dev kit is ~$250 and
an FPGA dev board + toolchain is $200–2,000 plus weeks of HDL/HLS engineering —
neither is bought now (below).

## USB3 bandwidth — does the bus saturate?

ASI174MM is monochrome 8-bit, so **data rate = W × H × 1 byte × fps**. USB3
SuperSpeed is 5 Gbit/s ≈ 625 MB/s theoretical, ~400–450 MB/s practical on a Pi 5.

| resolution | fps (measured by Andrea) | data rate | vs ~450 MB/s USB3 |
|---|---|---:|---|
| 1936×1216 (full sensor) | 120 | 282 MB/s | 63 % |
| 1920×1080 | 120 | 249 MB/s | 55 % |
| 800×600 | 250 | 120 MB/s | 27 % |
| 640×480 | 400 | 123 MB/s | 27 % |
| 256×256 (small ROI) | 1000 | 66 MB/s | 15 % |
| 128×128 (small ROI) | 1000 | 16 MB/s | 4 % |

**One ASI174MM never saturates the Pi 5's USB3 controller** at any resolution it
can run — the ceiling on frame rate is sensor readout/exposure, not the bus. The
binding constraint Andrea measured (resolution × framerate) is the camera's, not
the computer's.

The saturation point that *does* matter is **two cameras**: 2 × 800×600@250 fps =
240 MB/s plus protocol overhead on a single shared controller pushes past
comfortable margins, and the Pi 5 exposes effectively one USB3 controller for both
ports. **This — not compute — is the reason the double-emulsion build moves to a
NUC** (two controllers / PCIe). Single-emulsion stays on the Pi 5.

## Frames per control step — the headroom that makes this easy

Control runs at 0.1–10 Hz because settling is 5–30 s (pump ramp + tubing
compliance + dead volume), so frames accumulate between decisions:

`frames per control step = fps × control interval`

- 50 fps × 5 s = **250 frames** (the plan's baseline; ample for a stable diameter
  and a clean FFT frequency).
- 250 fps × 5 s = 1,250 frames — far more than needed; high fps buys temporal
  resolution for the dynamics model, not control necessity.

## Does the Pi 5 finish inside the interval?

The step that must complete within the control interval is the **CV measurement
path**, not model inference. Measured single-thread (`bench/results.json`, x86;
scale by ~5–7× for the Pi 5 CPU):

| ROI | ms/frame (x86) | 250 frames (x86) | 250 frames (Pi 5 est. ~7×) | vs 5 s interval |
|---|---:|---:|---:|---|
| 128×128 | 0.155 | 39 ms | ~270 ms | 5 % |
| 256×256 | 0.573 | 143 ms | ~1.0 s | 20 % |
| 800×600 | 4.16 | 1.04 s | ~7.3 s | **146 % — over** |

**Headroom under the chosen architecture:** at small/medium ROI the Pi 5 finishes
CV in 5–20 % of a 5 s interval, leaving the rest for the pump to settle. Model
inference adds < 5 ms (linear dynamics 70 params; regime CNN 0.17 ms/frame run
once per step). The one row that blows the budget is **sustained full-frame
800×600 classical CV** — which is exactly the case that triggers either (a)
cropping to a junction ROI (the right move — you only need the junction), or (b)
the learned-detector fallback on a NUC/GPU. Both are anticipated; neither is the
default.

The Pi 5 est. column is an **estimate** pending real measurement. Andrea produces
the actual numbers with one command on the actual board:

```
python bench/benchmark.py            # writes bench/results.json for this host
```

## Why not Jetson or FPGA now

- **Jetson Orin Nano ($250):** a fallback, not a purchase today. It earns its
  price only if the measurement path becomes a *learned high-resolution detector*
  (low-contrast fluids, dense droplets) — trigger: CV/detector p95 > 40 % of the
  control interval on the Pi 5, or sustained > 250 fps full-frame detection. Until
  then it is idle silicon. Andrea's YOLOv8-on-MPS ceiling (~170 fps, unbatchable)
  is the signature of that path; a Jetson helps there, but the whole point of the
  settling-limited loop is that we don't need it.
- **FPGA (dead):** its advantage is µs-scale pipelined per-frame streaming. At a
  0.1–10 Hz control rate with batched frames and sub-millisecond models there is
  nothing to pipeline that isn't already real-time on the $80 board. The
  engineering cost is unjustifiable for a 3-person team. Save the money.

## Bottom line for the purchase

Buy the Pi 5 now (~$150). Run `python bench/benchmark.py` on it and on the Mac to
get real p50/p95/p99 numbers. Only if the CV path on the Pi 5 exceeds 40 % of your
control interval at the ROI you actually need should you spend on the NUC+GPU
fallback — and the double-emulsion step will force the NUC anyway, for USB3
bandwidth, not compute.

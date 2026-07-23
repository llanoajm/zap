# Microfluidic droplet control

Closed-loop ML control of microfluidic flow-focusing droplet generation: from a
cold start to a user-specified (droplet diameter, generation frequency) target
with no human intervention beyond loading the device and entering the target. A
monochrome camera watches the junction, classical CV + a small model infer flow
state, and an MPC commands syringe pumps. This is the controller DAFD is not — it
has time, transients, and a fluid-aware geometry normalization.

> This project lives alongside the unrelated `zap` power-systems code in this repo;
> all microfluidics work is in `docs/`, `models/`, `bench/`, `normalization/`,
> `reference_stack/`, `synthetic/`, and `data/`. Start with `docs/decisions.md`.

## One command per collaborator

**Andrea (acquisition / inference — what to buy):**
```bash
pip install -r requirements-micro.txt
python models/export.py && python bench/benchmark.py
```
Exports every candidate model to ONNX/CoreML (verifying each by loading and
running it) and benchmarks model inference + the classical-CV measurement path on
your hardware — p50/p95/p99 latency, throughput, batch-1 vs batched, and a
compute-vs-overhead verdict. Degrades gracefully if a backend is absent. Then read
`docs/compute_recommendation.md`.

**Bernardo (fabrication — what to make):**
```bash
python data/build_device_matrix.py   # writes data/device_matrix.csv
```
Regenerates the 8-device fabricable matrix, each row checked against fab limits,
spanning W/ℓ\* by ~12,000×. Take `data/device_matrix.csv` to the cleanroom; read
`docs/device_spec.md` for the predicted envelope and the fabrication constraints.

**Antonio (modeling / normalization):**
```bash
python normalization/fetch_data.py && python normalization/study.py
```
Audits the DAFD dataset and runs the W/W_or-vs-W/ℓ\* generalization study; writes
`normalization/artifacts/`. Verdict in the first line of
`normalization/RESULTS.md`.

**Anyone (see the loop close, no hardware):**
```bash
python reference_stack/run_mock.py            # MPC, synchronous
python reference_stack/run_mock.py --controller pid   # the baseline it beats
python -m pytest reference_stack -q            # 21 tests, all green
```

## Layout

```
docs/decisions.md            the four open questions, resolved on the evidence
docs/architecture.md         one architecture + one fallback + the trigger; sized
docs/compute_recommendation.md   the machine, the price, the USB3/headroom math
docs/device_spec.md          predicted envelope + fabrication constraints
docs/stack_review_checklist.md   rubric for Andrea's implementation
docs/proposals/              world_models, convex_relaxation, synthetic_data
models/                      6 candidate models, profiler, ONNX/CoreML export
bench/benchmark.py           one cross-backend harness (CoreML/CUDA/Jetson/CPU)
normalization/               audit + study + verdict
reference_stack/             physics, camera, pumps, CV, safety, controller, loop
synthetic/render_droplet.py  headless Blender renderer (numpy fallback)
data/device_matrix.csv       the fabricable table
ASSUMPTIONS.md               every defensible choice, with reasoning
UPDATE.md                    plain-prose status to the team
```

## Setup

```bash
pip install -r requirements-micro.txt
```
CPU-only is fine (everything runs on the Pi 5 / a laptop). CUDA/CoreML are used
automatically if present. Blender is optional — `render_droplet.py` falls back to
a numpy renderer without it.

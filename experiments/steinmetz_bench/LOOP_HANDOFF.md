## Item just completed (experiments/steinmetz_bench/LOOP_QUEUE.md line 58)
- [x] 2.5 GPU speed via Modal (ROADMAP §8.4.1 headline)

## What landed
- `experiments/bench_gpu_modal.py` — builds a modest (6-bus) + larger (14-bus)
  uncongested radial PyPSA network, solves each exactly on CPU (zap+CLARABEL),
  dispatches the EXISTING deployed `zap-opf-solver` Modal app once on an H100,
  recomputes the GPU objective by re-evaluating zap's cost at the returned
  dispatch, and caches one timestamped JSON to `data/gpu_runs/`.
- `experiments/_modal_call.py` — the ONLY modal-touching code; run as a
  `/usr/bin/python3` subprocess (the venv has no `modal` client) so the verify
  command can never reach the GPU.
- `tests/test_bench_gpu_modal.py` — reads only the cache (skips when absent),
  cross-checks the CPU objective by re-deriving it, asserts machine=="cuda",
  positive `gpu_elapsed_s`, and objective gap < 1e-2; also validates the
  objective-reconstruction machinery deterministically with no GPU.

## Dispatch result (cached 20260529T135404Z, H100)
- modest (6 bus): gpu 4.65 s, cpu 0.059 s, objective gap 7.7e-7
- larger (14 bus): gpu 6.75 s, cpu 0.057 s, objective gap 1.7e-6
- Cache is intentionally gitignored (`data/*` = human-staged inputs + cached
  runs); it persists on this machine for verify and survives `git reset --hard`.
  On a fresh checkout the GPU tests skip, which is the acceptance's blocked path.

## Verify
`.venv/bin/python -m pytest experiments/steinmetz_bench/tests -q` → 74 passed.
`.venv/bin/ruff check experiments/steinmetz_bench` → clean.

STATUS: done
SUMMARY: bench_gpu_modal.py (2.5) dispatches the deployed zap-opf-solver H100 app once, caches data/gpu_runs/*.json, and certifies CPU-vs-GPU objective parity (gap ~1e-6 << 1e-2) with the verify command never touching Modal.
ACCEPTANCE: PASS — modal CLI + ~/.modal.toml present; cache JSON has machine=="cuda", positive elapsed_s, and CPU-vs-GPU objective gap < 1e-2 on both networks; emits a BenchResult; Modal is never called from the verify command (modal lives only in a system-python subprocess); tests skip cleanly if the cache is absent.
VERIFIED: yes

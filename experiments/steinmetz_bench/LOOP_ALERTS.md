[2026-05-29T11:22:53+00:00] DONE: - [ ] 0.1 Package skeleton (ROADMAP Phase 0) — create the benchmark package + smoke test.
[2026-05-29T11:31:42+00:00] DONE: - [ ] 0.2 Dataset registry + loaders (ROADMAP Phase 0) — resolve a DatasetSpec to a zap network + frames.
[2026-05-29T11:35:21+00:00] DONE: - [ ] 0.3 Scoring harness (ROADMAP Phase 0) — counterfactual delta, bootstrap CI, duration curve, fidelity band.
[2026-05-29T11:38:27+00:00] DONE: - [ ] 0.4 Result schema + report writer (ROADMAP Phase 0) — BenchResult dataclass + JSON + md stub.
[2026-05-29T11:47:28+00:00] DONE: - [ ] 1.1 PyPSA LP roundtrip reference (ROADMAP Phase 1) — zap-vs-PyPSA LMP/flow gap on a bundled net.
[2026-05-29T12:04:52+00:00] DONE: - [ ] 1.2 Gradient-vs-exact-dual check (ROADMAP Phase 1 / §8.4.4) — adjoint grad == dual identity.
[2026-05-29T12:10:10+00:00] DONE: - [ ] 1.3 Realized-LMP comparator (ROADMAP Phase 1) — zap-vs-realized LMP error distribution.
[2026-05-29T12:11:20+00:00] PARTIAL: - [ ] 2.1 Speed benchmark CPU (ROADMAP §8.4.1) — zap vs cvxpy baseline across sizes. (attempt 1)
[2026-05-29T12:11:53+00:00] THROTTLE: sleep 1800s (1/24) [fast-blank(28s)]
[2026-05-29T12:45:59+00:00] DONE: - [ ] 2.1 Speed benchmark CPU (ROADMAP §8.4.1) — zap vs cvxpy baseline across sizes.
[2026-05-29T13:16:26+00:00] DONE: - [ ] 2.2 Planning benchmark (ROADMAP §8.4.2) — gradient planner vs baseline on small expansion.
[2026-05-29T13:24:42+00:00] DONE: - [ ] 2.3 Accuracy benchmark (ROADMAP §8.4.3) — LMP/congestion error distributions vs PyPSA and realized.
[2026-05-29T13:30:40+00:00] THROTTLE: sleep 1800s (1/24) [fast-blank(58s)]
[2026-05-29T13:42:25+00:00] DONE: - [ ] 2.4 Sensitivity-correctness report (ROADMAP §8.4.4) — published per-device-type gradient-error table.
[2026-05-29T14:00:36+00:00] DONE: - [ ] 2.5 GPU speed via Modal (ROADMAP §8.4.1 headline) — dispatch zap-opf-solver on H100, cache result.
[2026-05-29T14:13:01+00:00] DONE: - [ ] 3.1 Data-center siting backtest (ROADMAP §7.1-A) — rank nodes by LMP duration curve + curtailment.

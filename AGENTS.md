# Agent guidance for the zap project

zap is a differentiable electricity-network optimization library: PowerNetwork + DispatchLayer + device library (Generator, Load, AC/DC lines, Storage, Store, Ground), with planning, dual, and ADMM modules. Underlying numerics: CVXPY for the convex problem layer, PyTorch for autograd, PyPSA for network ingest, scipy/numpy elsewhere.

This project is the engine room for a web client where end-users describe new objectives, constraints, or analyses in plain language and an agent ships them as code. Every feature you add should be runnable from a Python script, covered by at least a smoke test, and registered as a Skill so it persists across sessions and surfaces in the UI.

## Workspace layout

- `zap/` — the package itself
  - `network.py`, `layer.py` — top-level surfaces (`PowerNetwork`, `DispatchLayer`)
  - `devices/` — generators, loads, lines, storage, etc.
  - `planning/` — investment / expansion objectives and solvers
  - `dual/` — dual problem formulations
  - `admm/`, `conic/`, `resource_opt/` — alternative solvers and formulations
  - `importers/`, `exporters/` — PyPSA ingest and export
  - `tests/` — unit tests, including PyPSA dispatch + investment
- `.opencode/` — agent config for this project (see `.opencode/agent/`)
- `.venv/` — local Python environment (uv-managed). **All Python commands must use `./.venv/bin/python`** unless you explicitly activate the venv.

## Verifying changes (in order of speed)

1. **Import smoke:** `./.venv/bin/python -c "import zap; print(zap.__version__)"` — fastest check the package still loads after edits.
2. **Targeted pytest:** `./.venv/bin/python -m pytest zap/tests/<file>.py -k <name>` — for the module you touched.
3. **Full pytest:** `./.venv/bin/python -m pytest zap/tests` — note some tests require a Mosek license and may skip/fail without it; that's expected.
4. **End-to-end solve:** load a small PyPSA network via `zap.importers` and run `PowerNetwork.dispatch`. Use this when adding new objectives, constraints, or device types.

If you don't have Mosek, prefer `cvxpy.CLARABEL` or `cvxpy.SCS` as the solver in any new code you write.

## Shipping a feature

When the user asks for a new capability (e.g., "optimize for nitrogen emissions"), the lifecycle is:

1. **Plan.** Identify the right module: a new objective belongs in `zap/planning/`; a new device subtype belongs in `zap/devices/`; a new solver hook belongs near the existing solver code. Use `repo_overview` / `grep` first; don't guess.
2. **Implement.** Write the Python. Follow zap's existing patterns (attrs dataclasses, CVXPY expressions, torch tensors for differentiable code paths). Keep changes minimal and additive — don't rewrite adjacent code.
3. **Verify.** Run the smoke ladder above. For differentiable code, test the autograd path with a tiny finite-difference check (zap's tests already do this — see `zap/tests/test_network.py`).
4. **Register as a Skill.** Create `.opencode/skills/<feature-slug>/SKILL.md` with frontmatter `{ name, description }` and a body explaining what the feature does, what its inputs are, and a one-line example invocation. The Skill is how the feature persists across sessions and shows up in the UI.
5. **Commit.** One commit per feature: code + tests + SKILL.md together. Conventional commit style (`feat(planning): nitrogen emissions objective`).

## Style

Match zap's existing style:
- 100-char lines (`ruff` is configured).
- attrs `@define` / `@attrs.define` for device dataclasses.
- snake_case Python; lowercase module names.
- Type hints where they help readability; don't fight Python's looser typing where zap already runs un-typed.
- Don't add try/except around things that already work; let exceptions propagate.

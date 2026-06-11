# Ralph Research Loop: JEPA-style World Models as AC-OPF Surrogates / Latent-Space Planners

You are one iteration of a continuous deep-research loop ("Ralph loop") running inside the
`zap` repository (Degleris, Rajagopal et al. — GPU-accelerated, differentiable, vectorized
DC-OPF and multi-period/multi-value network expansion planning via gradient methods).

## The research question

Can **LeJEPA** (Balestriero & LeCun, 2025 — isotropic-Gaussian-regularized joint-embedding
predictive architecture) and/or **"LeWorldModel"**-style JEPA world models be used as an
**AC-OPF surrogate and latent-space planner** for power grids, such that the system:

1. Achieves **compositional generalization** (new topologies, N-k contingencies, unseen
   load/renewable patterns, varying network sizes) — the "foundation model" generality that
   Microsoft's recently announced **grid foundation model** claims (identify the exact model:
   verify via web search what Microsoft released for grid/power-flow foundation modeling,
   what it's trained on, what tasks it supports, and its reported accuracy/speedups).
2. Supports **deeper physics understanding** than pure supervised input→output OPF proxies
   (i.e., latent dynamics consistent with AC power-flow physics, energy balance, line limits).
3. Performs **optimal planning in latent space** (JEPA-style planning à la LeCun's
   world-model agenda: optimize actions/decisions by gradient descent through a frozen
   latent predictor).
4. Outputs (or makes recoverable) **dual economic variables** — LMPs/congestion prices,
   investment-signal duals — and supports **multi-period / multi-value expansion planning**
   in the spirit of:
   - Degleris, Rosenberg, Bouffard, Rajagopal et al., "Gradient methods for [multi-value /
     scalable] electricity network expansion planning" (the math behind this repo: implicit
     differentiation / backprop through DC-OPF, KKT-based sensitivities, vectorized GPU layers).
   - Rajagopal-group GPU-accelerated DC-OPF (this repo, `zap`).

The target artifact concept: **a hybrid that has the quality/speed/generality of a grid
foundation-model surrogate (à la Microsoft) AND the planning capability, dual variables,
and differentiability of the backprop-able DC-OPF stack (zap / Degleris et al.).**

## Sub-questions to research and verify (with citations)

- LeJEPA: exact mechanism (SIGReg / sketched isotropic Gaussian regularization), what it
  fixes vs. prior JEPAs (collapse, heuristics), evidence of transfer; is there any public
  "LeWorldModel" artifact (check LeCun's post-Meta startup, e.g. AMI/Advanced Machine
  Intelligence labs) — distinguish what exists vs. vaporware as of today.
- Grid foundation models: Microsoft's model (verify name, paper/blog, date), IBM/GridFM,
  PowerGraph/GraphNeuralSolver lines, OPFData (DeepMind), topology-aware GNN OPF surrogates.
  What generality do they actually demonstrate (cross-grid? cross-task?)?
- AC-OPF learning literature: DC3, DeepOPF, E2ELR (end-to-end learning w/ repair), 
  primal-dual learning (PDL), Lagrangian-dual deep learning (Fioretto et al.), 
  learning-to-optimize warm starts, feasibility restoration layers. Specifically: who
  recovers **dual variables / LMPs** from learned surrogates, and how accurate?
- Latent planning: JEPA planning results (V-JEPA-2-AC, DINO-WM, PLDM/navigation-world-models),
  MPC-in-latent-space precedents; what would "planning in latent space" mean when the
  decision variables are continuous dispatch/investment vectors with hard constraints?
- The differentiable-optimization bridge: OptNet/cvxpylayers/diffcp, implicit function
  theorem on KKT systems, zap's approach; can a JEPA encoder feed a differentiable OPF
  layer so duals come from the layer (exact) while the JEPA gives speed/generality?
  Compare against: amortized optimization, learned warm-starts for ADMM (zap has ADMM).
- Compositional generalization evidence: graph-based encoders vs. fixed-size MLPs; any
  results on transfer across PGLib cases / grid sizes; permutation/topology equivariance.
- Multi-period & expansion: how Degleris et al. formulate multi-value expansion planning
  (emissions, cost, curtailment as objectives; duals as investment signals); could a latent
  world model amortize the inner dispatch problem inside the outer planning loop, and what
  accuracy do the planning gradients need (bias in surrogate gradients vs. exact KKT grads)?
- Honest failure-mode analysis: where a JEPA surrogate will break (hard feasibility, duals
  are derivative objects — small primal error ≠ small dual error; degeneracy of LMPs;
  AC non-convexity; data coverage of binding-constraint patterns).

## Deliverables (maintain these files; improve them every iteration)

- `ralph/REPORT.md` — the main report. Structured, cited (URLs + paper titles +
  dates), with: executive summary; landscape; the proposed hybrid architecture(s) (at least
  2–3 concrete designs, e.g. (a) JEPA encoder + differentiable OPF head, (b) pure latent
  planner with dual-decoder + KKT-residual training loss, (c) foundation-model warm-start
  for zap's ADMM/gradient planner); feasibility verdict per design; experiment plan that
  could be run in this repo (`zap`) with PyPSA/PGLib data; open risks.
- `ralph/NOTES.md` — raw verified findings with citations (append, then prune).
- `ralph/TODO.md` — checklist of remaining research tasks. Maintain it: check off
  what you complete, add new tasks you discover.
- `ralph/SOURCES.md` — bibliography: every claim-bearing source, with URL, date,
  and a one-line credibility note.

## Rules for each iteration

1. Read `TODO.md`, `NOTES.md`, and `REPORT.md` first (if they exist) — do NOT redo done work.
2. Pick the highest-value 2–4 open TODO items. Use web search/fetch aggressively to verify
   facts; prefer primary sources (arXiv, official blogs, this repo's code/papers). Read the
   actual zap code (`zap/`, `experiments/`) when making claims about what zap can do.
3. Adversarially check claims: if you can't verify something (e.g., "LeWorldModel" exists),
   say so explicitly in the report rather than hallucinating.
4. Update all four deliverable files. Keep REPORT.md the single polished artifact.
5. Commit your changes: `git add research/ && git commit -m "ralph: <what you advanced>"`.
   Do NOT push; the outer loop pushes. Do NOT touch files outside `ralph/`.
6. **Termination**: when REPORT.md is comprehensive, fully cited, internally consistent,
   and TODO.md has no remaining high-value items, write a file `ralph/DONE`
   containing a one-paragraph completion summary. Only do this when genuinely done —
   a reviewer should find no major unanswered sub-question above.

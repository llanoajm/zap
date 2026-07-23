# Assumptions

Every place information was missing, a defensible choice was made, recorded here
with its reasoning, and the build continued. Grouped by area. Where a number is a
literature nominal or an engineering default rather than a measurement, it says
so.

## Repository context
- **`llanoajm/zap` is a power-systems optimization codebase**, not a microfluidics
  repo. The microfluidics project was added as self-contained top-level
  directories (`docs/`, `models/`, `bench/`, `normalization/`, `reference_stack/`,
  `synthetic/`, `data/`) on branch `claude/microfluidic-droplet-control-c0z612`,
  leaving the existing `zap/` code untouched. Rationale: the task specifies this
  exact repo layout and branch; the two projects don't interact.
- **No source PDFs in `docs/context/`.** Proceeded from the technical content
  reproduced in the task, as instructed. `docs/context/` is left as a placeholder.

## Physics / normalization
- **ℓ\* uses the continuous phase**: ℓ\* = η_c²/(ρ_c·γ). The source doc's two
  reference examples were reconciled as (a) an aqueous-continuous system
  (η_c ≈ 1 mPa·s → ℓ\* ≈ 0.2 µm, small) and (b) a fluorocarbon-continuous system
  (HFE-7500 → ℓ\* ≈ µm-scale). This is the only reading under which the doc's
  "water/oil small, fluorocarbon comparable-to-channel" narrative is internally
  consistent.
- **Correction propagated: W/ℓ\* = Oh⁻²**, not "Oh²". Derivation:
  Oh² = η_c²/(ρ_c γ W) = ℓ\*/W, so W/ℓ\* = Oh⁻². Asserted in
  `test_w_over_ell_is_oh_minus_2`. Also W/ℓ\* = Re/Ca (used in the audit).
- **Correction: HFE-7500 reference ℓ\* ≈ 5.3 µm, not 14 µm.** The doc's 14 µm is
  inconsistent with its own quoted η ≈ 1.6 mPa·s and γ ≈ 0.3 mN/m, which imply
  ρ ≈ 609 kg/m³; HFE-7500 is 1614 kg/m³. With the correct density the same η, γ
  give 5.3 µm. The qualitative claim survives. Flagged for the authors
  (`normalization/RESULTS.md`).
- **Fluid property library** (`reference_stack/physics.py`, ~25 °C nominal
  literature values; sources are datasheets / standard references, not measured
  here):
  - water η 1.0 mPa·s, ρ 998; mineral oil (light) η 25 mPa·s, ρ 840;
    HFE-7500 (3M Novec) η 1.6 mPa·s (datasheet range 1.24–1.77), ρ 1614;
    silicone oil 10 cSt η 9.3 mPa·s, ρ 930.
  - γ values are surfactant-laden nominals: water/mineral-oil+Span80 5 mN/m;
    water/HFE low-surfactant 4 mN/m, high-surfactant 0.3 mN/m; water/silicone
    3 mN/m; aqueous o/w 8 mN/m. Real values must be measured (pendant-drop
    tensiometry) per fluid batch — these are placeholders for design, flagged as
    such in the device spec.
- **Ca_crit prefactor = 1.0** in Ca_crit ~ Oh⁻¹. The O(1) fluid/geometry constant
  is left to be fit from data; 1.0 is the neutral placeholder.
- **Predicted droplet diameter band = 0.5–1.5 × W_or** (dripping regime scaling,
  Anna 2003; consistent with DAFD's observed 15–250 µm over 15–175 µm orifices).
  Frequency then follows from exact volume conservation F = Q_d/[(π/6)D³]. Marked
  as a derived estimate in `docs/device_spec.md`, not a measurement.

## Device matrix
- **8 devices, geometry × fluid chosen to span W/ℓ\* ~0.1–1,200.** Freezing
  geometry and sweeping flow maximizes data per fabricated chip for the dynamics
  work; spanning geometry × fluid is required for the normalization study. Both
  served by one mask set (`docs/device_spec.md`).
- **Fabrication limits** (`data/build_device_matrix.py`), conservative
  single-layer-PDMS / maskless-writer defaults: min in-plane feature 8 µm; height
  15–200 µm; roof-collapse guard width < 10×H; mold-release guard H < 8×min-width.
  These are standard soft-lithography rules of thumb; tighten to Bernardo's
  measured process capability when known.
- **Design flow rates** per device were chosen to place the design operating point
  inside dripping (Ca < Ca_crit) — G15's flow was reduced specifically because its
  high Oh (3.15) gives a narrow dripping window.

## Models / architecture
- **State layout: OBS(7) = [D_um, log10 F, CV, regime one-hot×4], ACT(2) =
  [Q_c, Q_d], window 16.** log10 F because generation rate spans decades. A
  reasonable, model-agnostic encoding; revisit if a target needs U_drop or Ca as
  explicit state.
- **Model widths** (regime CNN base width 16, LSTM hidden 64, world-model latent
  16, etc.) are chosen to be representative-but-small so benchmarks reflect a
  realistic lightweight deployment. They are dummy/untrained by design — the point
  is throughput, not accuracy.
- **Chosen dynamics model: per-fluid linear.** World model deferred
  (`docs/proposals/world_models.md`). Hidden parameters (θ, γ(t), compliance, τ)
  are absorbed as per-fluid residual structure into A, B and the MPC's re-planned
  process noise, rather than modeled explicitly at current data volume.
- **CoreML prediction is pending macOS.** Conversion runs on Linux and the saved
  `.mlpackage` spec is validated (declared I/O shapes), but `predict()` needs
  macOS; the exact command is stored in `models/exported/export_report.json`.
  ONNX is fully verified here (load + run + match-eager, 6/6).
- **Pi 5 timing estimates use a ~5–7× slowdown vs this x86 host.** An estimate;
  the real numbers come from `python bench/benchmark.py` on the board. Every Pi 5
  figure in the docs is labelled estimate/pending.

## Normalization study
- **ℓ\* imputed per viscosity-ratio cluster** from literature fluids, because the
  DAFD release lacks γ/ρ/absolute-viscosity. These imputations are explicitly
  literature estimates (`normalization/study.py:IMPUTED_FLUIDS`), and the whole
  point of the study is that this imputation is the weakness that makes the data
  unable to test the claim.
- **Model = GradientBoostingRegressor**, identical pipeline for both
  normalizations, so any difference is attributable to the feature change alone.
  Grouped CV (`GroupKFold`) for fluid- and geometry-held-out splits.
- **Datasets are not vendored.** `normalization/fetch_data.py` pulls them from OSF
  938rs on demand; they are other researchers' data (cite, don't redistribute).

## Control / safety
- **Hard actuator limits** (`reference_stack/actuator.py`), non-overridable:
  FLOW_MAX 5000 µL/h, flow slew 2000 µL/h/s, PRESSURE_MAX 2000 mbar, pressure slew
  800 mbar/s, watchdog timeout 2 s, safe state (0, 0). These are conservative
  engineering defaults sized to keep every device in the matrix below jetting and
  to protect bonded single-layer PDMS from delamination; **tighten per device**
  once real limits are known. Deliberately constants, not parameters.
- **Convergence tolerances**: diameter ±6 µm, frequency ±15 %, stability CV ≤ 0.07
  over a 5-step window. Physically reasonable for a droplet system (5 % native
  polydispersity); loosen/tighten per application spec.
- **Mock plant = the MPC's own steady-state map** in the closed-loop demo, so the
  mock loop is a fair perfect-model test of the loop mechanics. Real deployment
  fits the dynamics model from logged data; the controller is unchanged.

## Synthetic data
- **Blender is not installed here**, so `synthetic/render_droplet.py` has two
  paths: the real Cycles ray-tracer when run under `blender --python`, and a
  numpy+PIL fallback (physically-motivated brightfield appearance) so the script
  is genuinely runnable and testable in CI. Both emit identical PNG + JSON label
  schemas. The fallback was executed for all four regimes; the Blender path is
  structured but pending a Blender install to render.
- **Refractive indices** (water 1.333, mineral oil 1.467, HFE-7500 1.29, silicone
  1.40, air 1.00, PDMS 1.41, glass 1.52): standard literature values.

## Process
- **This ran unattended.** No questions were asked; every fork above was decided on
  the evidence and recorded. Anything a real measurement would change is labelled
  pending with the command that produces it.

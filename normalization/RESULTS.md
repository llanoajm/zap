# Normalization validation — results

**Verdict, first line: the published DAFD 3.0 dataset cannot test whether
W/ℓ\* beats W/W_or for cross-fluid generalization, and no split design rescues
it.** The release does not contain the fluid properties ℓ\* requires, its fluid
diversity is dominated by a single system, and the two normalizations are
mathematically indistinguishable within any one fluid system. Below are the
numbers behind that call and the experiment that would actually settle it.

Reproduce: `python normalization/fetch_data.py && python normalization/study.py`
(writes `normalization/artifacts/results.json` and `normalization_study.png`).

---

## Why the derivation is not the question

The physics is settled and is implemented + asserted in
`reference_stack/physics.py`: ℓ\* = η_c²/(ρ_c·γ), and **W/ℓ\* = Oh⁻²**
(the source doc's "Oh²" is inverted — `test_w_over_ell_is_oh_minus_2` fails if
that regresses). Across realistic fluid systems ℓ\* spans **0.125 µm
(aqueous-continuous) to 149 µm (mineral-oil-continuous), ~1,200×**, so the group
is genuinely discriminating in principle. The open question is purely empirical:
does substituting it improve generalization *across fluid systems*? That needs
data with fluid-system diversity **and** the properties to compute ℓ\* per row.

## The audit: what the DAFD release actually contains

`Comprehensive_normalized.xlsx`, 868 usable single-emulsion rows. Columns:
orifice width, normalized channel depth / inlet / outlet (all normalized by
W_or), flow-rate ratio, capillary number, viscosity ratio, hydraulic diameter,
observed diameter, observed generation rate. Three findings kill the test:

1. **ℓ\* is not computable from the release.** There is no interfacial-tension
   column, no density column, no absolute continuous-phase viscosity. From the
   available Ca and λ you can recover η_c/γ per row, but ℓ\* = η_c²/(ρ_c γ) needs
   one further independent property (equivalently ℓ\* = Re/Ca needs the kinematic
   viscosity). It is absent. Any ℓ\* used here is **imputed** from the DAFD
   papers' described fluids, per viscosity-ratio cluster — a literature estimate,
   not data.

2. **Fluid diversity is dominated by one system.** Of 868 rows, **474 (54.6 %)**
   share viscosity ratio λ = 57.2 — a single high-viscosity carrier. The
   remaining 45 % spread thinly across ten more λ clusters. Under imputation those
   collapse to **~5 distinct ℓ\* values.** Five points, heavily imbalanced, is
   not enough to fit or falsify a cross-fluid normalization trend.

3. **Within one fluid system the two normalizations are the same feature up to a
   constant.** ℓ\* is fixed per system, so W/ℓ\* is just W/W_or rescaled by a
   per-system scalar — and every geometry ratio likewise. A model with per-group
   intercepts cannot tell them apart *within* a group; only *across* groups with
   different, known ℓ\* can the substitution matter. That is exactly the axis the
   data is thinnest on.

## The strongest test the data supports, run anyway

Identical `GradientBoostingRegressor` pipelines on two feature sets that differ
**only** in geometry normalization (W_or ratios vs ℓ\*-imputed ratios), sharing
Ca, λ, Φ; scored under three splits. Targets: droplet diameter (µm) and
log₁₀ generation rate.

| split | target | W/W_or R² | W/ℓ\* R² |
|---|---|---|---|
| random 5-fold | diameter | 0.957 ± 0.007 | 0.956 ± 0.005 |
| random 5-fold | log F | 0.990 ± 0.002 | 0.990 ± 0.001 |
| leave-one-**fluid**-out | diameter | 0.178 ± 0.671 | −10.6 ± 19.8 |
| leave-one-**fluid**-out | log F | −3.04 ± 4.88 | −3.82 ± 6.21 |
| leave-one-**geometry**-out | diameter | −90.8 ± 177.9 | −272 ± 534 |

Reading the table:

- **Random split: identical, both excellent.** The model is fine and the features
  are fine; a random split just leaks each fluid system across train and test, so
  it answers the wrong question (interpolation, not cross-fluid generalization).
  This is why a random split is worthless here.
- **Leave-one-fluid-out: both collapse, and the difference is pure noise.** The
  ℓ\*−W_or gap on diameter is **−10.8 against a pooled σ of 19.8** — well inside
  the noise, i.e. **not discriminating.** The enormous variances come directly
  from the imbalance in finding (2): holding out the 55 % system, or a singleton
  cluster, produces wildly different folds. This is what "underpowered" looks like
  numerically.
- **Leave-one-geometry-out: uninformative.** The geometry groups are few and
  confounded with fluid and flow, so grouped CV degenerates.

`verdict.discriminating = false` in `results.json`. ℓ\* is neither confirmed nor
refuted on this data — the data simply cannot see it.

## What Bernardo should run instead

The remedy is a designed experiment, and it is already specified in
`data/device_matrix.csv` and `docs/device_spec.md`:

- **Hold geometry fixed, sweep fluid systems.** Fabricate the matrix once, run
  each geometry across **≥ 3 fluid systems whose η_c, ρ_c, γ are measured**
  (rheometer / pendant-drop tensiometry), chosen to span ℓ\* by orders of
  magnitude. The matrix already pairs geometries with fluids spanning ℓ\*
  0.13–149 µm and W/ℓ\* 0.10–1,198.
- **Score by leave-one-fluid-system-out**, with ℓ\* now *measured* per system, not
  imputed. Then the per-system constant that defeats the DAFD data becomes a
  known, varying quantity and the substitution is finally testable.
- **Every startup run is a time series**, so each fabricated device yields
  hundreds of labelled observations, not one row — the cost is dominated by
  fabrication, not by data.

## Questions for the DAFD authors (Bernardo owns the contact)

1. Are the per-experiment continuous-phase viscosity, density, and interfacial
   tension available anywhere? Publishing them would make the release testable for
   ℓ\*-type normalization without new fabrication.
2. Confirm the fluid identity behind each viscosity-ratio cluster, especially
   λ = 57.2 (55 % of the SE set) — which carrier, which surfactant, which γ?
3. Was interfacial tension measured at equilibrium or estimated? At kHz generation
   the dynamic value γ(t) differs; which enters the reported Ca?
4. In the DAFD normalization, is there any evidence the W_or convention fails
   across the fluid systems you *did* vary — i.e. a residual that correlates with
   a fluid property? That residual is the signal ℓ\* is meant to remove.
5. Would you be open to cross-validating a jointly-collected ℓ\*-spanning dataset
   (our device matrix) against DAFD 3.0 predictions?

## One physics correction to carry into that conversation

Independent of the data: the source document states "W/ℓ\* is equivalent to Oh²"
— it is **Oh⁻²**. And its fluorocarbon reference ℓ\* ≈ 14 µm is not
self-consistent with HFE-7500's density (1614 kg/m³); the quoted η ≈ 1.6 mPa·s and
γ ≈ 0.3 mN/m give **ℓ\* ≈ 5.3 µm** (their 14 µm implies ρ ≈ 609 kg/m³, which no
fluorocarbon has). The qualitative claim — fluorocarbon ℓ\* is O(1–10 µm),
comparable to channel dimensions, unlike the sub-micron aqueous case — is intact.
Both corrections are asserted in `reference_stack/tests/test_stack.py`.

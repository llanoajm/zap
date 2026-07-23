# Device specification

Companion to `data/device_matrix.csv`. Regenerate both with
`python data/build_device_matrix.py` (every row is checked against fabrication
limits before it is written; unfabricable rows are dropped with a printed reason,
never emitted). Bernardo should be able to take the CSV into the cleanroom without
a follow-up call.

## The question Bernardo asked: sweep geometry, or freeze geometry and sweep flow?

**Both, for different reasons — and the matrix is designed so one fabrication run
serves both.**

- **For the dynamics model and controller: freeze geometry, sweep flow rates.**
  Here every startup run is a time series, so a single device yields hundreds of
  (state, action, next-state) tuples per run — orders of magnitude more labelled
  data per fabricated chip than a DAFD-style one-row-per-device steady-state
  sweep. Information-per-fabricated-device is maximized by re-using a few good
  geometries across many flow trajectories and deliberate perturbations, not by
  fabricating many geometries once. Fabrication is the expensive step; data is
  cheap once a chip exists.
- **For the normalization study: you must sweep geometry × fluid**, because W/ℓ\*
  cannot vary on a single frozen geometry with a single fluid. This is the axis
  the DAFD data was blind to (`normalization/RESULTS.md`).

The resolution: **a small set of geometries (8), each individually a good
controller test-bed, chosen so that when paired with fluid systems spanning ℓ\*
they collectively span W/ℓ\* by ~4 orders of magnitude.** Freeze each geometry and
sweep flow for the dynamics data; compare across geometry×fluid for the
normalization study. Eight devices, one mask set, both programs served.

## The matrix

All flow-focusing, PDMS, dripping at the design operating point. Lengths in µm.

| id | W_or | H | CIW | DIW | OCW | angle° | fluid system | ℓ\* µm | D_h µm | W/ℓ\* | Oh | Ca(des) | purpose |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| G15 | 15 | 25 | 30 | 22 | 45 | 90 | mineral+Span80 | 148.8 | 18.8 | 0.10 | 3.15 | 0.22 | low-W/ℓ\* anchor (W≪ℓ\*), narrow dripping window |
| G25 | 25 | 30 | 50 | 38 | 75 | 90 | silicone 10cSt | 31.0 | 27.3 | 0.81 | 1.11 | 0.34 | W≈ℓ\* (Oh≈1): transition physics |
| G40 | 40 | 45 | 80 | 60 | 120 | 90 | HFE-7500 hi-surf | 5.29 | 42.4 | 7.57 | 0.36 | 0.33 | fluorocarbon, encapsulation-relevant |
| G40_shallow | 40 | 20 | 80 | 60 | 120 | 90 | HFE-7500 hi-surf | 5.29 | 26.7 | 7.57 | 0.36 | 0.74 | aspect-ratio control vs G40 |
| G60 | 60 | 60 | 120 | 90 | 180 | 60 | HFE-7500 lo-surf | 0.40 | 60.0 | 151 | 0.081 | 0.019 | mid W/ℓ\* |
| G100 | 100 | 90 | 150 | 120 | 250 | 60 | aqueous o/w | 0.125 | 94.7 | 798 | 0.035 | 0.0035 | DAFD-like aqueous |
| G150 | 150 | 120 | 225 | 180 | 375 | 45 | aqueous o/w | 0.125 | 133 | 1198 | 0.029 | 0.0027 | high-W/ℓ\*, high-throughput |
| G100_deep | 100 | 150 | 150 | 120 | 250 | 60 | mineral+Span80 | 148.8 | 120 | 0.67 | 1.22 | 0.083 | high aspect ratio, W≈ℓ\* |

**W/ℓ\* spans 0.10 → 1,198 (~12,000×); Oh spans 0.029 → 3.15.** That is the spread
the normalization study needs to discriminate — deliberately achieved by pairing
geometry and fluid together, not by geometry alone. Note G40/G40_shallow (same
W_or, same fluid, different H) and G100/G100_deep (aspect-ratio pair) isolate
channel aspect ratio at fixed W/ℓ\*; G15/G100_deep both sit at W≈ℓ\* via the
viscous mineral-oil system at opposite absolute scales.

## Predicted operating envelope (derived estimates — refine with device data)

Per device, from `data/device_matrix.csv`. **These are model estimates, not
measurements.** The diameter band is a dripping-regime scaling (droplets are
0.5–1.5× the orifice; Anna 2003; consistent with DAFD's 15–250 µm over 15–175 µm
orifices); the frequency band then follows from **exact volume conservation
F = Q_d / [(π/6) D³]** — only the diameter band is a model assumption, the
diameter↔frequency trade is exact. The device's own startup data collapses these
bands; the command to do it is the reference-stack measurement path.

| id | design Q_c / Q_d (µL/h) | pred D drop (µm) | pred F gen (Hz) | pred regime | Ca_crit (drip→jet) |
|---|---|---|---|---|---|
| G15 | 60 / 12 | 7.5–22.5 | 559–15,090 | dripping | 0.32 |
| G25 | 300 / 60 | 12.5–37.5 | 604–16,298 | dripping | 0.90 |
| G40 | 400 / 80 | 20–60 | 197–5,305 | dripping | 2.75 |
| G40_shallow | 400 / 80 | 20–60 | 197–5,305 | dripping | 2.75 |
| G60 | 600 / 120 | 30–90 | 87–2,358 | dripping | 12.3 |
| G100 | 900 / 180 | 50–150 | 28–764 | dripping | 28.3 |
| G150 | 1400 / 280 | 75–225 | 13–352 | dripping | 34.6 |
| G100_deep | 900 / 180 | 50–150 | 28–764 | dripping | 0.82 |

The **dripping→jetting boundary** is `Ca_crit ~ Oh⁻¹` (Utada 2008; prefactor 1,
absorbing the O(1) fluid/geometry constant to be fit). Note how it shifts with
ℓ\* exactly as the normalization argument predicts: the high-Oh devices (G15
Oh 3.15, G25 Oh 1.11, G100_deep Oh 1.22 — all W≈ℓ\*) have a **narrow** dripping
window (Ca_crit 0.32–0.90), so their design points sit deliberately low in Ca;
the low-Oh devices (G100, G150 — W≫ℓ\*) tolerate Ca up to ~30 before jetting. A
model taking W/ℓ\* rather than W/W_or should track that shift across the fluid
systems — which is the whole hypothesis, now fabricable.

## Fabrication constraints (single-layer PDMS, maskless / ML3 microwriter)

Enforced in `data/build_device_matrix.py:fab_check`; every emitted row passes all
of them:

- **Minimum in-plane feature ≥ 8 µm** — reliable limit for the maskless writer +
  PDMS replication. Smallest feature in the matrix is G15's 15 µm orifice; its
  22 µm dispersed inlet is the tightest, still comfortably above the floor.
- **Channel height 15–200 µm** — SU-8 single spin-coat: below ~15 µm depth control
  degrades; above ~200 µm a single coat won't hold vertical walls. Matrix uses
  20–150 µm.
- **Roof-collapse guard: every in-plane width < 10 × height.** A wide, shallow
  channel sags and bonds shut. The binding case is G40_shallow (H 20 µm) — its
  120 µm outlet is 6× the height, under the 10× limit. G15/G25 outlets are ~1.8×
  and 2.5× their heights.
- **Mold-release / tall-wall guard: height < 8 × min width.** The binding case is
  G100_deep (H 150 µm) — its 100 µm min feature gives a 1.5× ratio, safe.
- **Bonding.** Single-layer devices bonded to PDMS-coated glass; standard plasma
  activation. No feature in the matrix demands a high-aspect or suspended
  structure that would stress the bond. The 45–90° inlet attack angles are all
  moldable.

Check the CSV values against these limits mentally before committing a cleanroom
session; the generator already did, but an unfabricable row costs a session, so it
is worth the second look. If a geometry needs changing, edit `GEOMS`/`DESIGN` in
`data/build_device_matrix.py` and rerun — the fab check will reject anything out
of bounds and print why.

# Synthetic droplet imagery for regime classification: a Blender ray-tracing pipeline

**Audience:** Bernardo, Andrea, Antonio (modeling lead)
**Status:** proposal. Sim-to-real metrics are marked *pending* against real acquisition.

## Problem and the one thing this buys us

We need labeled brightfield frames to train the regime classifier (`regime_classifier`, CNN, input `(B,1,128,128)` → 4-way regime one-hot, 98k params). Real labeled frames are cheap for **dripping** — it is steady, and Andrea can hand-label thousands from a few minutes of video. They are expensive for the **tail**: jetting onset and satellite formation. Those are transient (they happen over a handful of frames), they are rare at any fixed operating point, and labeling them by hand means scrubbing video frame-by-frame guessing where "onset" starts. That is exactly the data the classifier is starved for, and exactly the data a renderer can produce on demand with a perfect label attached.

So the justification is narrow and concrete: **synthesize the tail, label the head from real video.** We are not trying to replace real data. We are filling the classes where real labels are the bottleneck.

The second thing this buys us is that we can render the *specific* fluid systems in the device matrix, including the ones we know will be hard to segment because their phases barely differ in refractive index. More on that below — it is the reason to render optics physically rather than paste ellipses onto noise.

## Why brightfield separates phases at all (no dye)

We image transmitted light through a monochrome sensor. There is no fluorophore and no absorbing dye. Contrast comes entirely from what happens to light at an interface between two media of different refractive index `n`. At every such interface:

- **Refraction (Snell):** `n1 sin θ1 = n2 sin θ2`. A curved droplet cap acts as a lens; rays bend toward or away from the optical axis depending on the sign of `n2 − n1`, redistributing intensity and producing the characteristic bright center / dark rim (or the inverse) of a droplet in brightfield.
- **Fresnel reflection:** at normal incidence the reflected fraction is `R = ((n1 − n2)/(n1 + n2))²`. This is what draws the interface edge. It scales with the **index contrast** `|n1 − n2|`, so the edge sharpness of a droplet is set by which two fluids meet, not by any pigment.

This is a ray-optics regime — droplets are tens of microns, illumination is incoherent broadband approximated as monochrome — so a path-tracer with correct IORs and a physically-sane light path reproduces the real image formation. That is why Blender (Cycles) is the right tool: it does Fresnel + refraction natively, and we control every `n`.

## Refractive indices and which pairs are hard

IOR values (from the locked facts):

| Medium | n |
|---|---|
| air | 1.00 |
| HFE-7500 | 1.29 |
| water | 1.333 |
| silicone / PDMS oil | 1.40 |
| PDMS (device wall) | 1.41 |
| mineral oil | 1.467 |
| glass (coverslip) | 1.52 |

The device matrix pairs a continuous phase with a dispersed phase per fluid system. The contrast that the classifier and any segmenter actually see is `|n_disp − n_cont|`:

| Device(s) | System | Continuous / dispersed | \|Δn\| | Edge contrast |
|---|---|---|---|---|
| G15, G100_deep | mineral oil + Span 80 | mineral oil 1.467 / water 1.333 | **0.134** | strong |
| G25 | silicone 10 cSt + surf | silicone 1.40 / water 1.333 | 0.067 | moderate |
| G40, G40_shallow, G60 | HFE-7500 + fluorosurf | HFE-7500 1.29 / water 1.333 | **0.043** | **weak — hard to segment** |
| G100, G150 | aqueous o/w | water 1.333 / oil (dispersed) | ~0.07–0.13 | moderate–strong |

Reference edges also matter: the water/PDMS wall boundary is `|1.333 − 1.41| = 0.077`, the HFE/PDMS wall is `0.12`, and any trapped **air** against water is `0.333` — air slugs and bubbles are the highest-contrast feature in the whole scene, which is useful for detecting them and a nuisance when they are contamination.

The load-bearing observation for modeling: **the HFE-7500 devices (G40, G40_shallow, G60) sit at Δn ≈ 0.043.** That is roughly a third of the mineral-oil contrast and near the noise floor for a classical Otsu-and-contour segmenter. The Fresnel edge is faint, refraction is weak, and an out-of-focus HFE droplet can vanish into the background. This is precisely where we want the classifier to lean on learned texture rather than a hard threshold, and precisely where synthetic data with exact masks lets us train for the low-contrast case instead of hoping the real data happens to cover it. If we rendered fake data with arbitrary IORs we would miss this entirely; rendering the real `n` pairs surfaces it for free.

## Scene and optics

**Geometry.** The scene is a flow-focusing junction built from the device matrix columns: orifice width `W_or`, channel height `H`, continuous-inlet width `CIW`, dispersed-inlet width `DIW`, outlet-channel width `OCW`, and the `inlet_attack_angle_deg`. The channel is a rectangular cross-section extruded to height `H`, cut in PDMS (wall `n = 1.41`), capped by a glass coverslip (`n = 1.52`). Every device in the matrix is one parameter set; `G40` is `W_or=40, H=45, CIW=80, DIW=60, OCW=120, angle=90°`, `G15` is the tight `W_or=15, H=25` anchor, `G150` the wide `W_or=150, H=120` throughput case. The fluids fill the channel volume as separate media with their IORs; droplets are dispersed-phase bodies with a curved cap whose radius follows the target diameter.

**Transmitted-light path (brightfield).** A source below the chip, chip in the middle, objective and sensor above — light passes *through* the sample. We approximate **Köhler illumination**: a uniform, incoherent, effectively collimated-then-condensed field across the field of view, so intensity variation in the image comes from the sample optics rather than from an imaged filament. In Cycles this is an area emitter behind a condenser aperture; we do not need a literal lamp filament, we need a flat illuminated background against which the index-mismatch edges appear.

**Sensor.** Target is the ZWO ASI174MM: 1936×1216, monochrome, 8-bit, global shutter. We render the field at the working ROI and reduce to the classifier's `128×128` monochrome input (center-crop the junction, resample). Rendering monochrome matches the sensor — there is no color channel to lean on, contrast has to come from the index physics, which is the point.

**Depth of field.** The objective has a finite numerical aperture `NA`; the focal plane sits at channel **mid-height** (`H/2`). Depth of field `≈ λ/NA² + (e·n)/(M·NA)` is shallow relative to `H` for the taller devices (`G100_deep` at `H=150`, `G150` at `H=120`), so a droplet riding high or low in the channel is genuinely out of focus. We model this with a camera aperture (finite f-number) so out-of-plane droplets blur exactly as they do on the scope, rather than being uniformly sharp. This is not cosmetic: a blurred low-contrast HFE droplet is the single hardest thing the classifier will face, and it must be in the training set.

## Domain randomization

Each rendered scene draws parameters from ranges chosen to bracket real operating conditions, so the network cannot latch onto a synthetic-only regularity:

- **Droplet diameter:** per-device from the matrix `pred_D_drop_um` band (e.g. G40 20–60 µm, G150 75–225 µm), jittered ±15%.
- **Spacing / count:** number of droplets in the field and inter-droplet gap, from tight monodisperse trains to sparse.
- **Flow speed → motion blur:** exposure-time smear along the flow axis, scaled from `U_c` (design 21–139 mm/s across the matrix); fast devices get visible streak.
- **Illumination:** overall intensity, plus a low-order spatial **gradient** and vignette to mimic imperfect Köhler alignment.
- **Focus offset:** focal plane displaced from `H/2` by a random fraction of `H`; droplet z-position within the channel randomized so DoF blur varies.
- **Sensor model:** additive read noise, shot noise, gain/black-level, mild fixed-pattern; 8-bit quantization.
- **IOR jitter:** each `n` perturbed within a small band (temperature, surfactant loading, batch) — critically this pushes the low-contrast HFE pair across the range where it is nearly invisible to slightly-more-visible, so the classifier sees the full difficulty spread.
- **Wall contamination:** thin wetting films, debris specks, and residue on the channel walls.
- **Satellite droplets:** small secondary droplets trailing the primaries, at controllable size and frequency — this is a first-class knob because satellites *are* one of the tail regimes we are here to synthesize.

## Labels are free from the scene graph

Because we place every droplet, we know its exact center, cap radius, and diameter, and we know the regime we asked the renderer to produce. So each frame ships with:

- a per-droplet **binary mask** (rasterized from the known geometry),
- **centroid** `(x, y)` and **diameter** per droplet,
- the **regime label** for the frame.

No hand labeling, no annotation drift, no ambiguity about where onset begins. The mask/diameter/centroid feed `tiny_detector` (`(B,1,128,128)` → `(B,5,16,16)`); the frame regime feeds `regime_classifier`. This is the whole efficiency argument: the label is an input to the render, not an interpretation of the output.

## The runnable script

`synthetic/render_droplet.py` is a headless Blender Python script that renders **one** parameterized droplet scene with its labels. It is invoked as:

```
blender --background --python synthetic/render_droplet.py -- \
    --out <dir> --regime dripping --n-droplets 5 --diameter-um 80
```

Everything after `--` is passed to the script, not to Blender. It builds the junction geometry, assigns fluid IORs, sets the brightfield light path and camera DoF, applies the domain-randomization draws, renders, and writes:

- a **monochrome PNG** — the rendered brightfield frame, and
- a **JSON label sidecar** — per-droplet `{centroid, diameter_um}` plus the frame `regime`, alongside the render parameters used.

One invocation = one labeled example. A dataset is a sweep over `--regime`, `--diameter-um`, `--n-droplets`, and device geometry, parallelized across processes. Rendering one scene at a time keeps the script trivial to reason about and trivial to fan out on a cluster or overnight on a workstation.

## Sim-to-real validation

The plan is **train on synthetic, test on real.** We hold out a small set of **real** frames — hand-labeled by regime, pending acquisition on the actual chips — and never train on them. We report **per-class accuracy and macro-F1** on that held-out real set, with particular attention to the tail classes (jetting onset, satellite) since that is where synthetic is doing the work; overall accuracy will be dominated by dripping and can look good while the tail fails.

Pending command (produces the metric once the real held-out set exists):

```
python synthetic/eval_sim2real.py \
    --model models/regime_classifier.onnx \
    --real-frames data/real_heldout/ \
    --report metrics/sim2real.json
```

The metric is **pending real acquisition** — no accuracy number is claimed here.

**The reality gap** is the risk: a network trained on renders can exploit synthetic-only cues (too-clean edges, wrong noise texture, unrealistic illumination) and then fail on the scope. Domain randomization is the mitigation — by varying illumination, focus, noise, IOR, contamination, and motion blur widely, the real scope becomes just another sample from the training distribution rather than a distribution shift. Concretely, if the sim-to-real number is weak we widen the ranges (especially noise and the low-contrast IOR band) and re-render, rather than reaching for real labels we do not have. The honest fallback if the gap stays wide for a specific fluid — most likely the Δn=0.043 HFE case — is to spend a small real-labeling budget *only* on that system, using synthetic to cover the rest.

## Summary

Render the device-matrix fluid systems with correct IORs so contrast comes from Fresnel + Snell, not dye; randomize the acquisition physics hard; take exact masks and regime labels for free from the scene graph; and spend the synthetic budget on the expensive tail — jetting onset and satellite formation — while labeling cheap dripping from real video. `synthetic/render_droplet.py` is the unit of production; `eval_sim2real.py` is the pending judge.

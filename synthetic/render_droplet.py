"""Render one parameterized flow-focusing droplet scene to a monochrome
brightfield image + a JSON label sidecar.

Two execution paths, same CLI and same outputs:

  * Under Blender (real ray tracing):
        blender --background --python synthetic/render_droplet.py -- \
            --out out/ --regime dripping --n-droplets 5 --diameter-um 80 \
            --fluid-pair water_in_hfe
    Builds a Cycles scene with index-matched glass BSDFs for each fluid (contrast
    from Fresnel + refraction at index-mismatched interfaces, no dye), a brightfield
    transmitted-light source, and a monochrome camera; renders to PNG.

  * Without Blender (numpy + PIL fallback, so the script is runnable anywhere and
    in CI):
        python synthetic/render_droplet.py --out out/ --regime dripping \
            --n-droplets 5 --diameter-um 80 --fluid-pair water_in_hfe
    Rasterizes the same scene with a physically-motivated brightfield droplet
    appearance (bright lensed core + dark refraction rim, contrast scaled by the
    fluid index mismatch).  Produces a byte-identical label schema so downstream
    training code does not care which path made the frame.

Label JSON schema (both paths):
    {"regime": str, "microns_per_pixel": float, "image": "<png>",
     "fluids": {"continuous": str, "dispersed": str, "n_c": float, "n_d": float,
                "index_contrast": float},
     "droplets": [{"cx_px","cy_px","diameter_px","diameter_um"}...]}

The rare regimes (jetting onset, satellite formation) are the point: real labels
for them are expensive; here they come free from the scene graph.  See
docs/proposals/synthetic_data.md.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

# Fluid refractive indices (see docs/proposals/synthetic_data.md / FACTS)
IOR = {"water": 1.333, "mineral_oil": 1.467, "hfe7500": 1.29,
       "silicone_oil": 1.40, "air": 1.00, "pdms": 1.41, "glass": 1.52}

FLUID_PAIRS = {
    # name            : (continuous, dispersed)
    "water_in_hfe":    ("hfe7500", "water"),        # low contrast 0.043
    "water_in_mineral": ("mineral_oil", "water"),   # higher contrast 0.134
    "water_in_silicone": ("silicone_oil", "water"),
    "oil_in_water":    ("water", "mineral_oil"),
}

REGIMES = ("dripping", "jetting", "no_droplets", "satellite")


def _argv_after_dashdash():
    # Blender passes script args after a lone "--"; plain python has none.
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1:]
    return sys.argv[1:]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="synthetic/out")
    ap.add_argument("--regime", choices=REGIMES, default="dripping")
    ap.add_argument("--n-droplets", type=int, default=5)
    ap.add_argument("--diameter-um", type=float, default=80.0)
    ap.add_argument("--fluid-pair", choices=list(FLUID_PAIRS), default="water_in_mineral")
    ap.add_argument("--microns-per-pixel", type=float, default=2.0)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--wor-um", type=float, default=40.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--focus-offset", type=float, default=0.0,
                    help="DoF blur radius in px for out-of-plane droplets")
    ap.add_argument("--noise", type=float, default=3.0)
    return ap.parse_args(_argv_after_dashdash())


def scene_droplets(args):
    """Compute droplet placements + labels for the requested regime.  Pure
    geometry; used by BOTH the Blender and fallback paths so labels match."""
    import numpy as np
    rng = np.random.default_rng(args.seed)
    d_um = args.diameter_um
    d_px = d_um / args.microns_per_pixel
    h, w = args.height, args.width
    cy0 = h // 2
    drops = []
    if args.regime == "no_droplets":
        return drops
    if args.regime == "jetting":
        # an unbroken thread from the orifice with a bulbous tip, few/no discrete drops
        # represent as a very elongated "droplet" plus 0-1 pinch-off beads
        for k in range(max(1, args.n_droplets // 3)):
            cx = int(0.2 * w + k * 0.25 * w)
            drops.append({"cx_px": cx, "cy_px": cy0,
                          "diameter_px": float(d_px * (1.6 - 0.1 * k)),
                          "diameter_um": float(d_um * (1.6 - 0.1 * k)),
                          "elongated": True})
        return drops
    spacing = max(d_px * 1.8, w / (args.n_droplets + 1))
    for k in range(args.n_droplets):
        cx = int((k + 1) * spacing % (w - d_px) + d_px / 2)
        cy = cy0 + int(rng.normal(0, 1.0))
        drops.append({"cx_px": cx, "cy_px": cy, "diameter_px": float(d_px),
                      "diameter_um": float(d_um), "elongated": False})
        if args.regime == "satellite":
            # a small satellite trailing each main drop
            sd = d_px * rng.uniform(0.25, 0.4)
            drops.append({"cx_px": int(cx - spacing * 0.4),
                          "cy_px": cy + int(rng.normal(0, 0.5)),
                          "diameter_px": float(sd),
                          "diameter_um": float(sd * args.microns_per_pixel),
                          "satellite": True})
    return drops


# --------------------------------------------------------------------------- #
# Fallback renderer (numpy + PIL): physically-motivated brightfield appearance
# --------------------------------------------------------------------------- #
def render_fallback(args, drops, n_c, n_d):
    import numpy as np
    from PIL import Image

    h, w = args.height, args.width
    contrast = abs(n_c - n_d)
    # brightfield: bright background; index-mismatched droplet = bright lensed
    # core ringed by a dark refraction rim whose depth scales with index contrast.
    img = np.full((h, w), 205.0, np.float64)

    # channel walls (flow-focusing): two horizontal PDMS walls bounding the outlet
    wall_half = int(1.4 * args.wor_um / args.microns_per_pixel)
    img[: max(0, h // 2 - wall_half), :] *= 0.86
    img[min(h, h // 2 + wall_half):, :] *= 0.86

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    rim_gain = 120.0 * min(1.0, contrast / 0.15)   # weak contrast -> faint rim
    for d in drops:
        cx, cy = d["cx_px"], d["cy_px"]
        r = d["diameter_px"] / 2.0
        if r < 1:
            continue
        if d.get("elongated"):
            # jetting thread: an ellipse elongated along flow (x)
            rr = ((xx - cx) / (2.4 * r)) ** 2 + ((yy - cy) / (0.6 * r)) ** 2
        else:
            rr = ((xx - cx) ** 2 + (yy - cy) ** 2) / (r * r)
        inside = rr <= 1.0
        # lensed bright core
        img[inside] += 18.0 * (1.0 - rr[inside])
        # dark refraction rim near rr ~ 1
        rim = np.exp(-((np.sqrt(np.clip(rr, 0, 4)) - 1.0) ** 2) / (2 * 0.08 ** 2))
        img -= rim_gain * rim

    # depth-of-field blur for out-of-plane droplets
    if args.focus_offset > 0:
        from scipy.ndimage import gaussian_filter  # optional
        try:
            img = gaussian_filter(img, sigma=args.focus_offset)
        except Exception:
            pass

    rng = np.random.default_rng(args.seed + 1)
    img += rng.normal(0, args.noise, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)

    os.makedirs(args.out, exist_ok=True)
    png = os.path.join(args.out, f"{args.regime}_{args.fluid_pair}_{args.seed}.png")
    Image.fromarray(img, mode="L").save(png)
    return png, "fallback-numpy"


# --------------------------------------------------------------------------- #
# Blender renderer (real ray tracing) -- only runs under `blender --python`
# --------------------------------------------------------------------------- #
def render_blender(args, drops, n_c, n_d):  # pragma: no cover - needs Blender
    import bpy
    import numpy as np

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.image_settings.color_mode = "BW"
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.cycles.samples = 128

    # clear default objects
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    um = args.microns_per_pixel * 1e-3  # mm per px (arbitrary world scale)

    def glass(name, ior):
        m = bpy.data.materials.new(name)
        m.use_nodes = True
        bsdf = m.node_tree.nodes.get("Principled BSDF")
        bsdf.inputs["Transmission Weight"].default_value = 1.0
        bsdf.inputs["IOR"].default_value = ior
        bsdf.inputs["Roughness"].default_value = 0.0
        return m

    cont_mat = glass("continuous", n_c)
    disp_mat = glass("dispersed", n_d)

    # continuous-phase slab (the channel volume)
    bpy.ops.mesh.primitive_cube_add(size=1)
    slab = bpy.context.active_object
    slab.scale = (args.width * um, args.height * um, 0.2)
    slab.data.materials.append(cont_mat)

    # droplets as spheres/ellipsoids
    for d in drops:
        x = (d["cx_px"] - args.width / 2) * um
        y = (args.height / 2 - d["cy_px"]) * um
        r = d["diameter_px"] / 2 * um
        bpy.ops.mesh.primitive_uv_sphere_add(radius=r, location=(x, y, 0))
        s = bpy.context.active_object
        if d.get("elongated"):
            s.scale = (2.4, 0.6, 1.0)
        s.data.materials.append(disp_mat)

    # brightfield: area light BEHIND the slab, camera in front (transmitted)
    bpy.ops.object.light_add(type="AREA", location=(0, 0, -3))
    light = bpy.context.active_object
    light.data.energy = 2000
    light.data.size = args.width * um * 2
    bpy.ops.object.camera_add(location=(0, 0, 4), rotation=(0, 0, 0))
    scene.camera = bpy.context.active_object
    scene.camera.data.dof.use_dof = args.focus_offset > 0

    os.makedirs(args.out, exist_ok=True)
    png = os.path.join(args.out, f"{args.regime}_{args.fluid_pair}_{args.seed}.png")
    scene.render.filepath = png
    bpy.ops.render.render(write_still=True)
    return png, "blender-cycles"


def main():
    args = parse_args()
    cont, disp = FLUID_PAIRS[args.fluid_pair]
    n_c, n_d = IOR[cont], IOR[disp]
    drops = scene_droplets(args)

    try:
        import bpy  # noqa: F401
        png, backend = render_blender(args, drops, n_c, n_d)
    except ImportError:
        png, backend = render_fallback(args, drops, n_c, n_d)

    labels = {
        "regime": args.regime,
        "backend": backend,
        "microns_per_pixel": args.microns_per_pixel,
        "image": os.path.basename(png),
        "fluids": {"continuous": cont, "dispersed": disp, "n_c": n_c, "n_d": n_d,
                   "index_contrast": round(abs(n_c - n_d), 4)},
        "droplets": [{"cx_px": d["cx_px"], "cy_px": d["cy_px"],
                      "diameter_px": round(d["diameter_px"], 2),
                      "diameter_um": round(d["diameter_um"], 2)} for d in drops],
    }
    jpath = os.path.splitext(png)[0] + ".json"
    with open(jpath, "w") as fh:
        json.dump(labels, fh, indent=2)
    print(f"[{backend}] wrote {png} and {jpath}  "
          f"({len(drops)} labelled droplets, regime={args.regime}, "
          f"index_contrast={labels['fluids']['index_contrast']})")


if __name__ == "__main__":
    main()

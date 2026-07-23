"""Generate the fabricable device matrix (data/device_matrix.csv) and the
per-device predicted operating envelope used by docs/device_spec.md.

Design rationale (defended in docs/device_spec.md):

*Freeze geometry, sweep flow rates* is the right move for the dynamics/controller
data, because every startup run yields hundreds of (state, action, next-state)
tuples for free -- transients are the training signal, so a handful of devices
gives orders of magnitude more labelled data than one-row-per-device steady-state
sweeps.  BUT the normalization study needs geometry x fluid coverage that spans
W/ell*, and a single frozen geometry cannot provide that.  So the matrix is a
small set of geometries, each a good controller test-bed on its own, chosen so
that when paired with the project's fluid systems they collectively span ~3
orders of magnitude in W/ell*.

Every row is checked against maskless-lithography / single-layer-PDMS fabrication
limits before it is emitted.  An unfabricable row costs a cleanroom session, so
rows that fail a limit are dropped with a printed reason rather than written.

Run:  python data/build_device_matrix.py   ->  writes data/device_matrix.csv
"""
from __future__ import annotations

import csv
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "reference_stack"))
import physics as ph  # noqa: E402

UM = 1e-6
UL_PER_H = 1e-9 / 3600.0  # uL/h -> m^3/s


# --------------------------------------------------------------------------- #
# Fabrication limits (single-layer PDMS, maskless / ML3 microwriter prototyping)
# --------------------------------------------------------------------------- #
MIN_FEATURE_UM = 8.0        # reliable min in-plane feature, maskless writer + PDMS
MIN_HEIGHT_UM = 15.0        # reliable SU-8 spin/expose floor for a controlled depth
MAX_HEIGHT_UM = 200.0       # single spin-coat, keep vertical walls
# Roof collapse: a wide, shallow channel sags.  Keep every in-plane width below
# ROOF_AR * height.  Conservative single-layer PDMS value.
ROOF_AR = 10.0
# Mold release / tall-wall aspect: keep height below MOLD_AR * min width.
MOLD_AR = 8.0


def fab_check(w_or, h, ciw, diw, ocw):
    """Return (ok, reason).  All args in um."""
    widths = {"W_or": w_or, "CIW": ciw, "DIW": diw, "OCW": ocw}
    for name, w in widths.items():
        if w < MIN_FEATURE_UM:
            return False, f"{name}={w:.1f}um < min feature {MIN_FEATURE_UM}um"
    if h < MIN_HEIGHT_UM:
        return False, f"H={h:.1f}um < min height {MIN_HEIGHT_UM}um"
    if h > MAX_HEIGHT_UM:
        return False, f"H={h:.1f}um > max single-spin height {MAX_HEIGHT_UM}um"
    for name, w in widths.items():
        if w > ROOF_AR * h:
            return False, f"{name}={w:.0f}um > {ROOF_AR}x H ({h:.0f}um): roof-collapse risk"
    if h > MOLD_AR * min(widths.values()):
        return False, f"H={h:.0f}um > {MOLD_AR}x min width: mold-release/tall-wall risk"
    return True, "ok"


# --------------------------------------------------------------------------- #
# Predicted envelope (derived, not measured -- flagged as such in the CSV)
# --------------------------------------------------------------------------- #
def diameter_band_um(w_or_um):
    """Dripping-regime droplet diameter band, as a fraction of orifice width.

    In flow-focusing dripping, droplet diameter is comparable to the orifice:
    roughly 0.5-1.5 x W_or across the accessible flow-ratio range (Anna 2003;
    Lashkaripour 2021 report 15-250 um for 15-175 um orifices).  We report the
    band; the device's own startup data will collapse it.
    """
    return 0.5 * w_or_um, 1.5 * w_or_um


def frequency_band_hz(d_lo_um, d_hi_um, q_d_m3s):
    """Generation frequency from exact volume conservation F = Q_d / V_d.

    Larger droplets -> lower frequency, so the low-diameter edge gives the high-F
    edge.  This relation is exact; only the diameter band is a model estimate.
    """
    v_lo = ph.droplet_volume(d_lo_um * UM)  # smallest drop -> highest F
    v_hi = ph.droplet_volume(d_hi_um * UM)
    return q_d_m3s / v_hi, q_d_m3s / v_lo  # (F_lo, F_hi)


# --------------------------------------------------------------------------- #
# Geometry seeds.  Attack angle and inlet ratios follow DAFD-typical values.
# (W_or, H, CIW, DIW, OCW) in um, attack angle in deg.
# --------------------------------------------------------------------------- #
GEOMS = [
    # key            W_or   H    CIW   DIW   OCW   angle
    ("G15",          15,   25,   30,   22,   45,   90),
    ("G25",          25,   30,   50,   38,   75,   90),
    ("G40",          40,   45,   80,   60,  120,   90),
    ("G40_shallow",  40,   20,   80,   60,  120,   90),   # low aspect ratio
    ("G60",          60,   60,  120,   90,  180,   60),
    ("G100",        100,   90,  150,  120,  250,   60),
    ("G150",        150,  120,  225,  180,  375,   45),
    ("G100_deep",   100,  150,  150,  120,  250,   60),   # high aspect ratio
]

# Fluid systems paired to span W/ell*.  Each geometry is fabricated once; the
# fluid it is *designed to run* is chosen to place it at a target W/ell* decade.
# Continuous-phase flow-rate design point (uL/h) for the envelope calc.
DESIGN = [
    # geom key      fluid key         Q_c(uL/h)  Q_d(uL/h)  purpose
    ("G15",         "mineral_span80",    60,       12,   "low W/ell*~0.1 anchor: channel far smaller than ell*; narrow dripping window (Oh>1)"),
    ("G25",         "si10_surf",        300,       60,   "W~ell* regime (Oh~1): transition physics visible"),
    ("G40",         "hfe_highsurf",     400,       80,   "W/ell*~8: fluorocarbon, drug-encapsulation-relevant"),
    ("G40_shallow", "hfe_highsurf",     400,       80,   "aspect-ratio control vs G40, same fluid"),
    ("G60",         "hfe_lowsurf",      600,      120,   "W/ell*~150: low-surfactant fluorocarbon"),
    ("G100",        "aq_surf",          900,      180,   "W/ell*~800: DAFD-like aqueous, o/w"),
    ("G150",        "aq_surf",         1400,      280,   "W/ell*~1200: large aqueous, high-throughput"),
    ("G100_deep",   "mineral_span80",  900,       180,  "W/ell*~0.7: viscous carrier, tiny relative to ell*"),
]


def build():
    rows = []
    dropped = []
    for geom_key, fluid_key, qc_ulh, qd_ulh, purpose in DESIGN:
        g = {k: v for k, v in zip(
            ["key", "W_or", "H", "CIW", "DIW", "OCW", "angle"],
            next(x for x in GEOMS if x[0] == geom_key))}
        ok, reason = fab_check(g["W_or"], g["H"], g["CIW"], g["DIW"], g["OCW"])
        if not ok:
            dropped.append((geom_key, reason))
            continue
        fs = ph.FLUID_SYSTEMS[fluid_key]
        w_or, h = g["W_or"] * UM, g["H"] * UM
        ciw, diw, ocw = g["CIW"] * UM, g["DIW"] * UM, g["OCW"] * UM
        qc, qd = qc_ulh * UL_PER_H, qd_ulh * UL_PER_H

        ell = fs.ell_star()
        d_h = ph.hydraulic_diameter(w_or, h)
        u_c = ph.u_continuous(qc, w_or, h)
        oh = ph.ohnesorge_number(fs, w_or)
        ca = ph.capillary_number(fs, u_c)
        re = ph.reynolds_number(fs, u_c, w_or)
        we = ph.weber_number(fs, u_c, w_or)
        ca_c = ph.ca_crit(fs, w_or)

        d_lo, d_hi = diameter_band_um(g["W_or"])
        f_lo, f_hi = frequency_band_hz(d_lo, d_hi, qd)

        rows.append({
            # --- fabricable geometry (what Bernardo needs) ---
            "device_id": geom_key,
            "junction_type": "flow-focusing",
            "device_material": "PDMS",
            "W_or_um": g["W_or"],
            "H_um": g["H"],
            "CIW_um": g["CIW"],
            "DIW_um": g["DIW"],
            "OCW_um": g["OCW"],
            "inlet_attack_angle_deg": g["angle"],
            # --- intended fluid pair ---
            "fluid_system": fluid_key,
            "fluid_label": fs.label,
            "ell_star_um": round(ell * 1e6, 4),
            # --- derived geometry (both normalizations) ---
            "D_h_um": round(d_h * 1e6, 3),
            "H_tilde_H_over_ell": round(h / ell, 4),
            "CIW_tilde_over_ell": round(ciw / ell, 4),
            "DIW_tilde_over_ell": round(diw / ell, 4),
            "OCW_tilde_over_ell": round(ocw / ell, 4),
            "W_over_ell_star": round(w_or / ell, 4),
            "Oh": round(oh, 5),
            "lambda_visc_ratio": round(fs.lam, 4),
            # --- design operating point ---
            "Q_c_uL_h": qc_ulh,
            "Q_d_uL_h": qd_ulh,
            "Phi_Qc_over_Qd": round(qc / qd, 3),
            "U_c_mm_s_design": round(u_c * 1e3, 2),
            "Ca_design": round(ca, 5),
            "Re_design": round(re, 4),
            "We_design": round(we, 6),
            # --- predicted envelope (DERIVED estimate, refine with device data) ---
            "pred_Ca_crit_dripping_to_jetting": round(ca_c, 4),
            "pred_regime_at_design": "dripping" if ca < ca_c else "jetting/unstable",
            "pred_D_drop_um_lo": round(d_lo, 1),
            "pred_D_drop_um_hi": round(d_hi, 1),
            "pred_F_gen_Hz_lo": round(f_lo, 1),
            "pred_F_gen_Hz_hi": round(f_hi, 1),
            # --- provenance ---
            "measurement_purpose": purpose,
        })
    return rows, dropped


def main():
    rows, dropped = build()
    out = os.path.join(os.path.dirname(__file__), "device_matrix.csv")
    fieldnames = list(rows[0].keys())
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} fabricable devices -> {out}")
    if dropped:
        print("DROPPED (unfabricable):")
        for k, r in dropped:
            print(f"  {k}: {r}")
    wl = [r["W_over_ell_star"] for r in rows]
    print(f"W/ell* span: {min(wl):.3g} .. {max(wl):.4g}  ({max(wl)/min(wl):.0f}x)")
    print(f"Oh span:     {min(r['Oh'] for r in rows):.3g} .. "
          f"{max(r['Oh'] for r in rows):.3g}")
    print("\nper-device summary:")
    print(f"{'id':<12}{'W_or':>6}{'H':>5}{'fluid':<16}"
          f"{'ell*[um]':>10}{'W/ell*':>10}{'Oh':>8}{'Ca_des':>9}{'regime':>10}")
    for r in rows:
        print(f"{r['device_id']:<12}{r['W_or_um']:>6}{r['H_um']:>5}"
              f"{r['fluid_system']:<16}{r['ell_star_um']:>10.4g}"
              f"{r['W_over_ell_star']:>10.4g}{r['Oh']:>8.3g}"
              f"{r['Ca_design']:>9.4g}{r['pred_regime_at_design']:>10}")


if __name__ == "__main__":
    main()

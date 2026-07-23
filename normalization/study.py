"""Normalization audit + best-effort validation of the W/ell* hypothesis.

Question (empirical, not derivational): does replacing DAFD's W/W_or geometry
normalization with W/ell* improve *cross-fluid-system* generalization?

This script (1) audits the DAFD single-emulsion release for the fluid-property
information and fluid-system diversity the test requires, and (2) runs the
strongest test the data can support: identical models on W/W_or- vs
ell*-normalized features under random, leave-one-fluid-system-out, and
leave-one-geometry-out splits.

It writes numbers + a plot consumed by normalization/RESULTS.md.  The verdict is
computed here, not asserted in prose.

Data: DAFD 3.0, OSF 938rs, `Comprehensive_normalized.xlsx`.  Download with
    python normalization/fetch_data.py
(kept out of git; see normalization/fetch_data.py for the exact OSF URLs).

Usage:  python normalization/study.py
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "reference_stack"))
import physics as ph  # noqa: E402

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "data", "Comprehensive_normalized.xlsx")
OUT = os.path.join(HERE, "artifacts")
os.makedirs(OUT, exist_ok=True)


# --------------------------------------------------------------------------- #
# IMPUTED fluid properties per viscosity-ratio cluster.
# The release carries NO gamma, NO density, NO absolute continuous viscosity, so
# ell* cannot be computed from the columns.  To attempt the test at all we impute
# (eta_c, rho_c, gamma) per lambda-cluster from the DAFD papers' described fluids.
# These are LITERATURE ESTIMATES, not dataset values -- flagged everywhere.  The
# whole point of RESULTS.md is that this imputation is exactly the weakness that
# makes the DAFD release unable to test the claim.
# --------------------------------------------------------------------------- #
IMPUTED_FLUIDS = {
    # lambda_cluster : (eta_c [Pa.s], rho_c [kg/m3], gamma [N/m], description)
    0.4703: (1.0e-3, 998.0, 8.0e-3, "aqueous continuous (o/w), low-visc carrier"),
    0.8078: (1.0e-3, 998.0, 8.0e-3, "aqueous continuous (o/w)"),
    0.8749: (1.6e-3, 1614.0, 4.0e-3, "fluorocarbon continuous, low surfactant"),
    0.9687: (1.6e-3, 1614.0, 4.0e-3, "fluorocarbon continuous"),
    1.6090: (25.0e-3, 840.0, 5.0e-3, "light mineral oil + Span80 (DAFD base)"),
    1.6339: (25.0e-3, 840.0, 5.0e-3, "light mineral oil + Span80 (DAFD base)"),
    1.6689: (25.0e-3, 840.0, 5.0e-3, "light mineral oil + Span80 (DAFD base)"),
    1.7329: (25.0e-3, 840.0, 5.0e-3, "light mineral oil + Span80 (DAFD base)"),
    1.8729: (25.0e-3, 840.0, 5.0e-3, "mineral oil variant"),
    3.9634: (50.0e-3, 850.0, 5.0e-3, "heavier mineral oil"),
    57.2:   (350.0e-3, 950.0, 3.0e-3, "high-viscosity carrier (silicone/PDMS oil)"),
}


def nearest_lambda_key(lam):
    return min(IMPUTED_FLUIDS, key=lambda k: abs(k - lam))


def load():
    df = pd.read_excel(DATA).dropna(subset=["Orifice width (um)"]).reset_index(drop=True)
    # numeric coercion
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Observed droplet diameter (um)",
                           "Observed generation rate (Hz)",
                           "Capillary number", "viscosity ratio",
                           "Flow rate ratio"]).reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# 1. AUDIT
# --------------------------------------------------------------------------- #
def audit(df):
    lam = df["viscosity ratio"].round(4)
    vc = lam.value_counts().sort_index()
    n = len(df)
    dominant = vc.max()
    report = {
        "n_rows": int(n),
        "n_distinct_lambda": int(lam.nunique()),
        "dominant_lambda": float(vc.idxmax()),
        "dominant_lambda_fraction": round(float(dominant) / n, 3),
        "lambda_counts": {float(k): int(v) for k, v in vc.items()},
        "Ca_min": float(df["Capillary number"].min()),
        "Ca_max": float(df["Capillary number"].max()),
        "has_gamma_column": False,
        "has_density_column": False,
        "has_abs_viscosity_column": False,
        "W_or_range_um": [float(df["Orifice width (um)"].min()),
                          float(df["Orifice width (um)"].max())],
    }
    # ell* spread IF we impute
    ells = []
    for l in vc.index:
        eta, rho, gam, _ = IMPUTED_FLUIDS[nearest_lambda_key(l)]
        ells.append(eta ** 2 / (rho * gam) * 1e6)
    report["imputed_ell_star_um_distinct"] = sorted(round(e, 4) for e in set(np.round(ells, 4)))
    report["n_distinct_imputed_ell_star"] = len(set(np.round(ells, 2)))
    return report


# --------------------------------------------------------------------------- #
# 2. FEATURE BUILD
# --------------------------------------------------------------------------- #
def build_features(df):
    """Return (X_wor, X_ell, y_D, y_F, groups_fluid, groups_geom, colnames).

    X_wor: geometry normalized by W_or (DAFD convention, as released).
    X_ell: same geometry normalized by ell* (imputed per fluid system).
    Shared physics features (Ca, lambda, Phi, Re-proxy) appear in both so the
    ONLY difference between the two feature sets is the geometry normalization.
    """
    w_or_um = df["Orifice width (um)"].values
    # imputed fluid props per row
    eta = np.empty(len(df)); rho = np.empty(len(df)); gam = np.empty(len(df))
    for i, l in enumerate(df["viscosity ratio"].values):
        e, r, g, _ = IMPUTED_FLUIDS[nearest_lambda_key(round(l, 4))]
        eta[i], rho[i], gam[i] = e, r, g
    ell_um = (eta ** 2 / (rho * gam)) * 1e6

    # absolute geometry reconstructed from released normalized columns x W_or
    depth_um = df["Normalized channel depth"].values * w_or_um
    ciw_um = df["Normalized continuous inlet"].values * w_or_um
    diw_um = df["Normalized dispersed inlet"].values * w_or_um
    ocw_um = df["Normalized outlet width"].values * w_or_um

    # shared physics features (identical in both sets)
    Ca = df["Capillary number"].values
    lam = df["viscosity ratio"].values
    Phi = df["Flow rate ratio"].values
    shared = np.column_stack([np.log10(Ca), np.log10(lam), np.log10(Phi)])
    shared_names = ["log_Ca", "log_lambda", "log_Phi"]

    # DAFD convention: geometry as ratios to W_or (dimensionless), + log W_or scale
    X_wor = np.column_stack([
        df["Normalized channel depth"].values,
        df["Normalized continuous inlet"].values,
        df["Normalized dispersed inlet"].values,
        df["Normalized outlet width"].values,
        np.log10(w_or_um),
        shared,
    ])
    wor_names = ["H/Wor", "CIW/Wor", "DIW/Wor", "OCW/Wor", "log_Wor"] + shared_names

    # ell* convention: geometry as ratios to ell*
    X_ell = np.column_stack([
        np.log10(w_or_um / ell_um),        # W/ell* = Oh^-2
        np.log10(depth_um / ell_um),
        np.log10(ciw_um / ell_um),
        np.log10(diw_um / ell_um),
        np.log10(ocw_um / ell_um),
        shared,
    ])
    ell_names = ["log_W/ell", "log_H/ell", "log_CIW/ell", "log_DIW/ell",
                 "log_OCW/ell"] + shared_names

    y_D = df["Observed droplet diameter (um)"].values
    y_F = np.log10(df["Observed generation rate (Hz)"].values)  # spans decades

    groups_fluid = df["viscosity ratio"].round(4).values
    # geometry group = (W_or, depth ratio) signature
    groups_geom = (df["Orifice width (um)"].round(1).astype(str) + "_" +
                   df["Normalized channel depth"].round(2).astype(str)).values
    return (X_wor, wor_names, X_ell, ell_names, y_D, y_F,
            groups_fluid, groups_geom)


# --------------------------------------------------------------------------- #
# 3. MODEL + SPLIT EVALUATION
# --------------------------------------------------------------------------- #
def eval_grouped(X, y, groups, kind):
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import GroupKFold, KFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import mean_absolute_error, r2_score

    y = np.asarray(y)
    if kind == "random":
        splitter = KFold(n_splits=5, shuffle=True, random_state=0).split(X)
    else:
        ug = len(np.unique(groups))
        splitter = GroupKFold(n_splits=min(5, ug)).split(X, y, groups)
    maes, r2s = [], []
    for tr, te in splitter:
        model = make_pipeline(
            StandardScaler(),
            GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                      learning_rate=0.05, random_state=0))
        model.fit(X[tr], y[tr])
        p = model.predict(X[te])
        maes.append(mean_absolute_error(y[te], p))
        r2s.append(r2_score(y[te], p))
    return {"mae_mean": float(np.mean(maes)), "mae_std": float(np.std(maes)),
            "r2_mean": float(np.mean(r2s)), "r2_std": float(np.std(r2s))}


def main():
    if not os.path.exists(DATA):
        print(f"MISSING {DATA}\nRun: python normalization/fetch_data.py")
        sys.exit(2)
    df = load()
    aud = audit(df)
    print("=== AUDIT ===")
    print(json.dumps(aud, indent=2))

    (Xw, wn, Xe, en, yD, yF, gf, gg) = build_features(df)
    results = {"audit": aud, "targets": {}}
    for tname, y in [("droplet_diameter_um", yD), ("log10_generation_Hz", yF)]:
        results["targets"][tname] = {}
        for split in ["random", "leave_one_fluid_out", "leave_one_geometry_out"]:
            groups = gf if split == "leave_one_fluid_out" else (
                gg if split == "leave_one_geometry_out" else None)
            rw = eval_grouped(Xw, y, groups, "random" if split == "random" else split)
            re = eval_grouped(Xe, y, groups, "random" if split == "random" else split)
            results["targets"][tname][split] = {"W_or_norm": rw, "ell_star_norm": re}
            print(f"\n[{tname}] {split}")
            print(f"  W/W_or : R2={rw['r2_mean']:.3f}+/-{rw['r2_std']:.3f}  "
                  f"MAE={rw['mae_mean']:.3f}")
            print(f"  W/ell* : R2={re['r2_mean']:.3f}+/-{re['r2_std']:.3f}  "
                  f"MAE={re['mae_mean']:.3f}")

    # verdict: does ell* improve the *cross-fluid* R2 beyond noise?
    lf = results["targets"]["droplet_diameter_um"]["leave_one_fluid_out"]
    delta = lf["ell_star_norm"]["r2_mean"] - lf["W_or_norm"]["r2_mean"]
    noise = math.hypot(lf["ell_star_norm"]["r2_std"], lf["W_or_norm"]["r2_std"])
    results["verdict"] = {
        "cross_fluid_R2_delta_ell_minus_wor": round(delta, 4),
        "cross_fluid_R2_pooled_std": round(noise, 4),
        "discriminating": bool(abs(delta) > noise),
        "n_distinct_fluid_systems": aud["n_distinct_lambda"],
        "n_distinct_imputed_ell_star": aud["n_distinct_imputed_ell_star"],
        "dominant_system_fraction": aud["dominant_lambda_fraction"],
    }
    print("\n=== VERDICT ===")
    print(json.dumps(results["verdict"], indent=2))

    with open(os.path.join(OUT, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    _plot(results, aud)
    print(f"\nwrote {OUT}/results.json and plots")


def _plot(results, aud):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    # (a) fluid diversity
    ax = axes[0]
    ks = list(aud["lambda_counts"].keys())
    vs = list(aud["lambda_counts"].values())
    ax.bar([f"{k:.2g}" for k in ks], vs, color="#4477aa")
    ax.set_title(f"Fluid-system diversity\n{aud['n_distinct_lambda']} systems, "
                 f"top = {aud['dominant_lambda_fraction']*100:.0f}% of rows")
    ax.set_xlabel("viscosity ratio lambda (fluid-system proxy)")
    ax.set_ylabel("rows"); ax.tick_params(axis="x", rotation=45)

    # (b) cross-fluid R2 comparison for diameter
    ax = axes[1]
    for i, split in enumerate(["random", "leave_one_fluid_out",
                               "leave_one_geometry_out"]):
        d = results["targets"]["droplet_diameter_um"][split]
        ax.bar(i - 0.18, d["W_or_norm"]["r2_mean"], 0.36,
               yerr=d["W_or_norm"]["r2_std"], color="#ccbb44",
               label="W/W_or" if i == 0 else None)
        ax.bar(i + 0.18, d["ell_star_norm"]["r2_mean"], 0.36,
               yerr=d["ell_star_norm"]["r2_std"], color="#66ccee",
               label="W/ell*" if i == 0 else None)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["random", "leave-fluid-out", "leave-geom-out"], rotation=15)
    ax.set_ylabel("R^2 (droplet diameter)"); ax.legend()
    ax.set_title("Does ell* normalization help generalization?")
    ax.axhline(0, color="k", lw=0.5)

    # (c) identifiability: imputed ell* clusters
    ax = axes[2]
    ax.bar(range(len(aud["imputed_ell_star_um_distinct"])),
           sorted(aud["imputed_ell_star_um_distinct"]), color="#ee6677")
    ax.set_yscale("log")
    ax.set_title(f"Imputed ell* clusters\n{aud['n_distinct_imputed_ell_star']} "
                 f"distinct values -> low DoF")
    ax.set_ylabel("ell* [um] (IMPUTED)"); ax.set_xlabel("fluid-system index")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "normalization_study.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()

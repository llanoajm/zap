"""Dimensionless-group and fluid-length-scale physics for microfluidic droplet control.

This module is the single source of truth for every derived quantity used across
the project: the visco-capillary length ell* that replaces the DAFD orifice-width
normalization, the hydraulic diameter, and the standard dimensionless groups
(Oh, Ca, Re, We, lambda, Phi).

It is deliberately dependency-free (standard-library ``math`` only) so it runs on
a Raspberry Pi, inside a Blender Python, or before the ML stack is installed.

Two corrections to the source design document are propagated here and asserted in
``_self_test`` so they cannot silently regress:

1.  W / ell*  ==  Oh**-2   (the source doc wrote "equivalent to Oh**2"; it is the
    inverse).  Since Oh**2 = eta_c**2 / (rho_c gamma W) = ell*/W, dividing W by ell*
    gives Oh**-2.  Same group up to inversion; the physics is unaffected, but the
    statement must be right before it goes to the paper's authors.

2.  The source doc's fluorocarbon reference ell* ~ 14 um is not self-consistent
    with its own quoted properties (eta_c ~ 1.6 mPa.s, gamma ~ 0.3 mN/m): those
    imply a continuous-phase density of ~609 kg/m^3, whereas HFE-7500 is
    1614 kg/m^3.  With the correct density the same eta and gamma give
    ell* ~ 5.3 um.  The qualitative claim (fluorocarbon ell* is O(1-10 um),
    comparable to channel dimensions, unlike the sub-micron aqueous system) is
    intact; the specific number is corrected.  See ``FLUID_SYSTEMS`` and
    docs/decisions.md.

All SI units unless a name carries a suffix (``_um``, ``_uLmin``, ``_mPas``, ...).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

G = 9.80665  # m s^-2


# --------------------------------------------------------------------------- #
# Fluid property library
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FluidSystem:
    """A continuous/dispersed fluid pair with the properties ell* needs.

    ell* = eta_c^2 / (rho_c * gamma) depends only on the *continuous* phase
    viscosity/density and the interfacial tension; the dispersed phase enters
    the model through the viscosity ratio lambda = eta_c / eta_d.
    """

    key: str
    label: str
    eta_c: float      # continuous-phase dynamic viscosity  [Pa.s]
    rho_c: float      # continuous-phase density             [kg/m^3]
    eta_d: float      # dispersed-phase dynamic viscosity    [Pa.s]
    rho_d: float      # dispersed-phase density              [kg/m^3]
    gamma: float      # interfacial tension                  [N/m]
    note: str = ""

    # --- intrinsic length scales -------------------------------------------- #
    def ell_star(self) -> float:
        """Visco-capillary length eta_c^2 / (rho_c gamma)  [m]."""
        return self.eta_c ** 2 / (self.rho_c * self.gamma)

    def ell_cap(self) -> float:
        """Capillary length sqrt(gamma / (rho_c g))  [m]  (rejected normalization)."""
        return math.sqrt(self.gamma / (self.rho_c * G))

    @property
    def lam(self) -> float:
        """Viscosity ratio lambda = eta_c / eta_d."""
        return self.eta_c / self.eta_d

    def ell_star_um(self) -> float:
        return self.ell_star() * 1e6


# Property values are datasheet / literature nominal figures at ~25 C.  Sources
# and the rationale for each choice are logged in ASSUMPTIONS.md.  Surfactant
# lowers gamma dramatically; the "loading" column names the assumed regime.
FLUID_SYSTEMS: dict[str, FluidSystem] = {
    # --- aqueous-continuous (oil-in-water); the small-ell* end -------------- #
    "aq_surf": FluidSystem(
        key="aq_surf", label="DI water + surfactant (aqueous continuous, o/w)",
        eta_c=1.0e-3, rho_c=998.0, eta_d=25.0e-3, rho_d=840.0, gamma=8.0e-3,
        note="water carrier, mineral-oil droplets; gamma ~8 mN/m at moderate loading",
    ),
    # --- fluorocarbon-continuous (water-in-oil); low-mid ell* -------------- #
    "hfe_lowsurf": FluidSystem(
        key="hfe_lowsurf", label="HFE-7500, low fluorosurfactant (water-in-oil)",
        eta_c=1.6e-3, rho_c=1614.0, eta_d=1.0e-3, rho_d=998.0, gamma=4.0e-3,
        note="3M Novec 7500; eta_c=1.6 mPa.s (source-doc value, within datasheet "
             "range 1.24-1.77); gamma ~4 mN/m at low fluorosurfactant loading",
    ),
    "hfe_highsurf": FluidSystem(
        key="hfe_highsurf", label="HFE-7500, high fluorosurfactant (water-in-oil)",
        eta_c=1.6e-3, rho_c=1614.0, eta_d=1.0e-3, rho_d=998.0, gamma=0.3e-3,
        note="the source-doc reference (eta_c=1.6 mPa.s, gamma=0.3 mN/m). With the "
             "CORRECT HFE density 1614 kg/m^3 these give ell*=5.3 um, NOT the 14 um "
             "in the source doc (whose number implies rho~609 kg/m^3, impossible "
             "for a fluorocarbon).",
    ),
    # --- silicone-oil-continuous; mid-high ell* ----------------------------- #
    "si10_surf": FluidSystem(
        key="si10_surf", label="Silicone oil 10 cSt + surfactant (water-in-oil)",
        eta_c=9.3e-3, rho_c=930.0, eta_d=1.0e-3, rho_d=998.0, gamma=3.0e-3,
        note="PDMS oil 10 cSt carrier; gamma ~3 mN/m",
    ),
    # --- mineral-oil-continuous; the large-ell* end ------------------------ #
    "mineral_span80": FluidSystem(
        key="mineral_span80", label="Light mineral oil + 2% Span 80 (water-in-oil)",
        eta_c=25.0e-3, rho_c=840.0, eta_d=1.0e-3, rho_d=998.0, gamma=5.0e-3,
        note="classic w/o system; eta_c ~25 mPa.s dominates ell*",
    ),
}


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def hydraulic_diameter(w_or: float, h: float) -> float:
    """D_h = 2 W H / (W + H)  for a rectangular channel  [m]."""
    return 2.0 * w_or * h / (w_or + h)


# --------------------------------------------------------------------------- #
# Operating-point quantities (need a flow rate / velocity)
# --------------------------------------------------------------------------- #
def u_continuous(q_c: float, w_or: float, h: float) -> float:
    """Continuous-phase superficial velocity U_c = Q_c / (W H)  [m/s].

    q_c in m^3/s, w_or & h in m.
    """
    return q_c / (w_or * h)


def capillary_number(fs: FluidSystem, u_c: float) -> float:
    """Ca = eta_c U_c / gamma."""
    return fs.eta_c * u_c / fs.gamma


def reynolds_number(fs: FluidSystem, u_c: float, w_or: float) -> float:
    """Re = rho_c U_c W / eta_c."""
    return fs.rho_c * u_c * w_or / fs.eta_c


def weber_number(fs: FluidSystem, u_c: float, w_or: float) -> float:
    """We = rho_c U_c^2 W / gamma  (== Ca * Re)."""
    return fs.rho_c * u_c ** 2 * w_or / fs.gamma


def ohnesorge_number(fs: FluidSystem, w_or: float) -> float:
    """Oh = eta_c / sqrt(rho_c gamma W) = sqrt(ell*/W)."""
    return fs.eta_c / math.sqrt(fs.rho_c * fs.gamma * w_or)


def w_over_ellstar(fs: FluidSystem, w_or: float) -> float:
    """The proposed geometry normalization W / ell*  ==  Oh**-2."""
    return w_or / fs.ell_star()


# --------------------------------------------------------------------------- #
# Dripping -> jetting transition
# --------------------------------------------------------------------------- #
def ca_crit(fs: FluidSystem, w_or: float, prefactor: float = 1.0) -> float:
    """Critical capillary number for dripping->jetting, Ca_crit ~ Oh^-1.

    Utada et al. (2008) place the transition at a critical capillary number that
    scales inversely with Ohnesorge.  ``prefactor`` absorbs the O(1) constant
    that is fluid/geometry specific and must be fit from data; default 1.0.
    """
    return prefactor / ohnesorge_number(fs, w_or)


# --------------------------------------------------------------------------- #
# Droplet result quantities
# --------------------------------------------------------------------------- #
def droplet_volume(diameter: float) -> float:
    """V_d = (pi/6) D^3."""
    return math.pi / 6.0 * diameter ** 3


def d_tilde(diameter: float, w_or: float, h: float) -> float:
    """Normalized droplet diameter D / D_h."""
    return diameter / hydraulic_diameter(w_or, h)


# --------------------------------------------------------------------------- #
# Convenience: full derived-feature vector for one operating point
# --------------------------------------------------------------------------- #
def derived_features(fs: FluidSystem, w_or: float, h: float,
                     ciw: float, diw: float, ocw: float,
                     q_c: float, q_d: float) -> dict[str, float]:
    """Every derived quantity from the taxonomy for a single (device, op-point).

    Lengths in m, flow rates in m^3/s.  Returns a flat dict of dimensionless /
    physical derived features, including geometry under BOTH normalization
    conventions (W_or and ell*).
    """
    ell = fs.ell_star()
    d_h = hydraulic_diameter(w_or, h)
    u_c = u_continuous(q_c, w_or, h)
    oh = ohnesorge_number(fs, w_or)
    feats = {
        # intrinsic
        "ell_star_m": ell,
        "D_h_m": d_h,
        "lambda": fs.lam,
        "Oh": oh,
        # operating
        "U_c_m_s": u_c,
        "Ca": capillary_number(fs, u_c),
        "Re": reynolds_number(fs, u_c, w_or),
        "We": weber_number(fs, u_c, w_or),
        "Phi": q_c / q_d,
        # geometry under DAFD convention (normalize by W_or)
        "H_over_Wor": h / w_or,
        "CIW_over_Wor": ciw / w_or,
        "DIW_over_Wor": diw / w_or,
        "OCW_over_Wor": ocw / w_or,
        # geometry under proposed convention (normalize by ell*)
        "Wor_over_ell": w_or / ell,
        "H_over_ell": h / ell,
        "CIW_over_ell": ciw / ell,
        "DIW_over_ell": diw / ell,
        "OCW_over_ell": ocw / ell,
        # transition
        "Ca_crit": ca_crit(fs, w_or),
    }
    return feats


# --------------------------------------------------------------------------- #
# Self-test: asserts the two corrections and reproduces the ell* table
# --------------------------------------------------------------------------- #
def _self_test() -> None:
    # (1) W/ell* == Oh^-2, exactly, for every fluid/geometry pair.
    for fs in FLUID_SYSTEMS.values():
        for w in (15e-6, 50e-6, 200e-6):
            lhs = w_over_ellstar(fs, w)
            rhs = ohnesorge_number(fs, w) ** -2
            assert math.isclose(lhs, rhs, rel_tol=1e-12), (fs.key, w, lhs, rhs)

    # (2) We == Ca * Re, identically.
    fs = FLUID_SYSTEMS["hfe_lowsurf"]
    u = u_continuous(1e-9, 50e-6, 40e-6)  # 60 uL/... arbitrary
    assert math.isclose(
        weber_number(fs, u, 50e-6),
        capillary_number(fs, u) * reynolds_number(fs, u, 50e-6),
        rel_tol=1e-12,
    )

    # (3) The corrected HFE reference: ell* ~ 5.3 um, NOT 14 um.
    hfe = FLUID_SYSTEMS["hfe_highsurf"]
    assert 4.5 < hfe.ell_star_um() < 6.0, hfe.ell_star_um()

    print("physics self-test: PASS")
    print(f"{'system':<40}{'ell* [um]':>12}{'ell_cap [mm]':>14}{'lambda':>10}")
    for fs in FLUID_SYSTEMS.values():
        print(f"{fs.label:<40}{fs.ell_star_um():>12.4g}"
              f"{fs.ell_cap()*1e3:>14.3g}{fs.lam:>10.3g}")
    print(f"\nell* spread: "
          f"{min(f.ell_star_um() for f in FLUID_SYSTEMS.values()):.3g} um "
          f"to {max(f.ell_star_um() for f in FLUID_SYSTEMS.values()):.4g} um "
          f"({max(f.ell_star_um() for f in FLUID_SYSTEMS.values())/min(f.ell_star_um() for f in FLUID_SYSTEMS.values()):.0f}x)")


if __name__ == "__main__":
    _self_test()

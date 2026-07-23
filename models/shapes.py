"""Canonical tensor shapes shared by every model, benchmark, and export.

Fixing these in one place is what lets Andrea benchmark all candidates against
identical I/O and lets the controller and dynamics model agree on a state vector.

State / action layout
---------------------
OBS  (7): [D_um, log10_F_Hz, CV, regime_p0, regime_p1, regime_p2, regime_p3]
          regime probs order: dripping, jetting, no_droplets, satellite
ACT  (2): [Q_c, Q_d]  (continuous-, dispersed-phase flow rate, uL/h)
STEP (9): concat(OBS, ACT) -- one timestep fed to a dynamics model
WINDOW (16): number of past steps a recurrent/world model conditions on

Vision
------
ROI is a single-channel (monochrome camera) crop of the junction.
ROI_HW default 128x128; the classifier and detector accept (B,1,H,W).
"""

OBS_DIM = 7
ACT_DIM = 2
STEP_DIM = OBS_DIM + ACT_DIM  # 9
WINDOW = 16

N_REGIMES = 4
REGIME_NAMES = ("dripping", "jetting", "no_droplets", "satellite")

ROI_H = 128
ROI_W = 128

# latent size for the world model / neural-ODE latent
LATENT_DIM = 16

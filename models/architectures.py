"""Runnable, randomly-initialized dummy models with the exact I/O shapes of every
candidate in docs/architecture.md.  No training -- these exist so Andrea can
benchmark throughput and latency on his own hardware immediately, and so the
export/profile scripts have concrete graphs to measure.

Every class has a fixed forward signature and a ``.example_inputs()`` classmethod
returning correctly-shaped random tensors, which the benchmark and export scripts
call generically.

Candidates
----------
RegimeClassifier   : lightweight CNN, (B,1,128,128) -> (B,4) regime logits
TinyDetector       : conv detector stand-in for a learned diameter path,
                     (B,1,128,128) -> (B,5,16,16) [obj,x,y,w,h] per cell
LSTMDynamics       : recurrent dynamics, (B,16,9) -> (B,7) next OBS
LinearDynamics     : per-fluid LDS  x' = A x + B u, (B,7)+(B,2) -> (B,7)
NeuralODEDynamics  : MLP vector field + RK4, (B,7)+(B,2) -> (B,7)
LatentWorldModel   : encoder/transition/decoder, additive-noise world model
"""
from __future__ import annotations

import torch
import torch.nn as nn

from shapes import (OBS_DIM, ACT_DIM, STEP_DIM, WINDOW, N_REGIMES,
                    ROI_H, ROI_W, LATENT_DIM)


# --------------------------------------------------------------------------- #
# Vision
# --------------------------------------------------------------------------- #
class RegimeClassifier(nn.Module):
    """Lightweight CNN classifying a junction ROI into 4 flow regimes.

    (B,1,128,128) -> (B,4) logits.  Sized to run at kHz on modest hardware; see
    models/profile.py for the exact FLOP/param budget.
    """

    def __init__(self, n_classes: int = N_REGIMES, width: int = 16):
        super().__init__()
        c = width
        self.features = nn.Sequential(
            nn.Conv2d(1, c, 3, stride=2, padding=1), nn.BatchNorm2d(c), nn.ReLU(),      # 64
            nn.Conv2d(c, 2 * c, 3, stride=2, padding=1), nn.BatchNorm2d(2 * c), nn.ReLU(),  # 32
            nn.Conv2d(2 * c, 4 * c, 3, stride=2, padding=1), nn.BatchNorm2d(4 * c), nn.ReLU(),  # 16
            nn.Conv2d(4 * c, 8 * c, 3, stride=2, padding=1), nn.BatchNorm2d(8 * c), nn.ReLU(),  # 8
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Linear(8 * c, n_classes))

    def forward(self, x):
        return self.head(self.features(x))

    @classmethod
    def example_inputs(cls, batch=1):
        return (torch.randn(batch, 1, ROI_H, ROI_W),)


class TinyDetector(nn.Module):
    """Single-scale YOLO-style detector stand-in for a *learned* diameter path.

    (B,1,128,128) -> (B,5,16,16): per 8x8-px cell, [objectness,x,y,w,h].
    This is the candidate to benchmark against classical CV -- if CV throughput
    is the binding constraint, a detector at this size is what would replace it.
    """

    def __init__(self, width: int = 16):
        super().__init__()
        c = width
        self.backbone = nn.Sequential(
            nn.Conv2d(1, c, 3, 2, 1), nn.ReLU(),        # 64
            nn.Conv2d(c, 2 * c, 3, 2, 1), nn.ReLU(),    # 32
            nn.Conv2d(2 * c, 4 * c, 3, 2, 1), nn.ReLU(),  # 16
            nn.Conv2d(4 * c, 4 * c, 3, 1, 1), nn.ReLU(),
        )
        self.head = nn.Conv2d(4 * c, 5, 1)

    def forward(self, x):
        return self.head(self.backbone(x))

    @classmethod
    def example_inputs(cls, batch=1):
        return (torch.randn(batch, 1, ROI_H, ROI_W),)


# --------------------------------------------------------------------------- #
# Dynamics candidates
# --------------------------------------------------------------------------- #
class LSTMDynamics(nn.Module):
    """Recurrent dynamics over a window of (obs, action) pairs.

    (B, WINDOW, STEP_DIM) -> (B, OBS_DIM) predicted next observation.
    """

    def __init__(self, hidden: int = 64, layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(STEP_DIM, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, OBS_DIM)

    def forward(self, seq):
        out, _ = self.lstm(seq)
        return self.head(out[:, -1, :])

    @classmethod
    def example_inputs(cls, batch=1):
        return (torch.randn(batch, WINDOW, STEP_DIM),)


class LinearDynamics(nn.Module):
    """Per-fluid-system linear dynamical system  x_{t+1} = A x_t + B u_t.

    (B, OBS_DIM), (B, ACT_DIM) -> (B, OBS_DIM).  The baseline the fancier models
    must beat near steady state.  A,B are the only parameters.
    """

    def __init__(self):
        super().__init__()
        self.A = nn.Linear(OBS_DIM, OBS_DIM, bias=False)
        self.B = nn.Linear(ACT_DIM, OBS_DIM, bias=True)

    def forward(self, x, u):
        return self.A(x) + self.B(u)

    @classmethod
    def example_inputs(cls, batch=1):
        return (torch.randn(batch, OBS_DIM), torch.randn(batch, ACT_DIM))


class NeuralODEDynamics(nn.Module):
    """Continuous-time dynamics: MLP vector field integrated with fixed-step RK4.

    (B, OBS_DIM), (B, ACT_DIM) -> (B, OBS_DIM) after ``steps`` RK4 steps of the
    learned field f(x,u).  No torchdiffeq dependency -- fixed-step keeps it
    ONNX/CoreML-exportable and deterministic for benchmarking.
    """

    def __init__(self, hidden: int = 64, steps: int = 4, dt: float = 0.25):
        super().__init__()
        self.steps, self.dt = steps, dt
        self.f = nn.Sequential(
            nn.Linear(OBS_DIM + ACT_DIM, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, OBS_DIM))

    def field(self, x, u):
        return self.f(torch.cat([x, u], dim=-1))

    def forward(self, x, u):
        dt = self.dt
        for _ in range(self.steps):
            k1 = self.field(x, u)
            k2 = self.field(x + 0.5 * dt * k1, u)
            k3 = self.field(x + 0.5 * dt * k2, u)
            k4 = self.field(x + dt * k3, u)
            x = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return x

    @classmethod
    def example_inputs(cls, batch=1):
        return (torch.randn(batch, OBS_DIM), torch.randn(batch, ACT_DIM))


class LatentWorldModel(nn.Module):
    """Additive-noise latent world model (the proposal under scrutiny).

    encoder:    (B, WINDOW, STEP_DIM) -> z (B, LATENT_DIM)
    transition: (z, action) -> z'      (additive-noise: z' = z + g(z,u))
    decoder:    z' -> next obs (B, OBS_DIM)

    forward returns (next_obs, z') so the controller can roll the latent forward
    under candidate action sequences without decoding every step.
    """

    def __init__(self, latent: int = LATENT_DIM, hidden: int = 64):
        super().__init__()
        self.encoder = nn.GRU(STEP_DIM, latent, batch_first=True)
        self.transition = nn.Sequential(
            nn.Linear(latent + ACT_DIM, hidden), nn.Tanh(),
            nn.Linear(hidden, latent))
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden), nn.Tanh(),
            nn.Linear(hidden, OBS_DIM))

    def forward(self, seq, action):
        _, h = self.encoder(seq)
        z = h[-1]
        z_next = z + self.transition(torch.cat([z, action], dim=-1))  # additive noise
        return self.decoder(z_next), z_next

    @classmethod
    def example_inputs(cls, batch=1):
        return (torch.randn(batch, WINDOW, STEP_DIM),
                torch.randn(batch, ACT_DIM))


# registry consumed by profile.py / export.py / benchmark.py
REGISTRY = {
    "regime_classifier": RegimeClassifier,
    "tiny_detector": TinyDetector,
    "lstm_dynamics": LSTMDynamics,
    "linear_dynamics": LinearDynamics,
    "neural_ode_dynamics": NeuralODEDynamics,
    "latent_world_model": LatentWorldModel,
}

# which models are "per-frame" (vision) vs "per-control-step" (dynamics)
PER_FRAME = {"regime_classifier", "tiny_detector"}
PER_STEP = {"lstm_dynamics", "linear_dynamics", "neural_ode_dynamics",
            "latent_world_model"}


def build(name: str) -> nn.Module:
    m = REGISTRY[name]()
    m.eval()
    return m


if __name__ == "__main__":
    for name, ctor in REGISTRY.items():
        m = ctor().eval()
        xs = ctor.example_inputs(batch=2)
        with torch.no_grad():
            out = m(*xs)
        outs = out if isinstance(out, tuple) else (out,)
        shapes = ", ".join(str(tuple(o.shape)) for o in outs)
        n = sum(p.numel() for p in m.parameters())
        print(f"{name:22s} in={[tuple(x.shape) for x in xs]}  out=({shapes})  "
              f"params={n:,}")

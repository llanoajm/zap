"""Regime estimation: dripping / jetting / no_droplets / satellite.

Two backends behind one interface:
  * HeuristicRegime  -- frame-differencing + periodicity, zero training, the
                        default and probably sufficient (see docs/architecture.md).
  * CNNRegime        -- wraps models.RegimeClassifier for when the heuristic is
                        not discriminative enough; loads a checkpoint if given,
                        else runs the randomly-initialized dummy (benchmark only).
"""
from __future__ import annotations

import numpy as np

from measurement import frame_diff_regime_hint

REGIMES = ("dripping", "jetting", "no_droplets", "satellite")


class HeuristicRegime:
    backend = "heuristic"

    def classify(self, frames: np.ndarray) -> tuple[str, dict]:
        hint = frame_diff_regime_hint(frames)
        # heuristic cannot see satellites reliably; map unknown/unstable -> jetting
        label = hint if hint in REGIMES else ("jetting" if hint == "unstable"
                                              else "no_droplets")
        conf = {r: (1.0 if r == label else 0.0) for r in REGIMES}
        return label, conf


class CNNRegime:
    backend = "cnn"

    def __init__(self, checkpoint: str | None = None, device: str = "cpu",
                 roi=(128, 128)):
        import os
        import sys
        import torch
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
        from architectures import RegimeClassifier
        self.torch = torch
        self.roi = roi
        self.device = device
        self.model = RegimeClassifier().to(device).eval()
        if checkpoint:
            self.model.load_state_dict(torch.load(checkpoint, map_location=device))

    def _prep(self, frames: np.ndarray):
        import cv2
        # use the median frame as a representative ROI-resized input
        rep = np.median(frames[:: max(1, len(frames) // 8)], axis=0).astype(np.uint8)
        rep = cv2.resize(rep, self.roi)
        t = self.torch.from_numpy(rep).float()[None, None] / 255.0
        return t.to(self.device)

    def classify(self, frames: np.ndarray) -> tuple[str, dict]:
        with self.torch.no_grad():
            logits = self.model(self._prep(frames))
            probs = self.torch.softmax(logits, dim=-1)[0].cpu().numpy()
        conf = {r: float(p) for r, p in zip(REGIMES, probs)}
        return REGIMES[int(probs.argmax())], conf

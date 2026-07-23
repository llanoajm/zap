"""Classical-CV measurement of droplet diameter, generation frequency, and
polydispersity from a batch of monochrome frames.

Salvaged in spirit from uDROP: keep pixel-to-micron calibration from a known
reference distance, thresholding + contour detection for diameter, and
inter-droplet timing for frequency.  Discard uDROP's ffmpeg batch architecture
and GUI -- here frames arrive as in-memory numpy arrays and nothing touches disk.

The whole module is pure numpy + OpenCV and has no hardware dependency, so it is
the step the benchmark profiles as "the CV path that must finish inside the
control interval."
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    import cv2
    _HAVE_CV2 = True
except Exception:  # pragma: no cover
    _HAVE_CV2 = False


@dataclass(frozen=True)
class Calibration:
    """Pixel-to-micron scale from a known reference distance in the image.

    e.g. image a stage micrometer or a channel of known width once, measure it in
    pixels, and record microns_per_pixel = known_um / measured_px.
    """
    microns_per_pixel: float

    @classmethod
    def from_reference(cls, known_um: float, measured_px: float) -> "Calibration":
        if measured_px <= 0:
            raise ValueError("measured_px must be > 0")
        return cls(known_um / measured_px)


@dataclass
class Measurement:
    diameter_um: float          # mean equivalent-circle diameter
    diameter_cv: float          # polydispersity: std/mean over detected droplets
    frequency_hz: float         # generation frequency
    n_droplets: int             # droplets used for the diameter estimate
    regime_hint: str            # cheap frame-diff regime hint (not the classifier)
    per_droplet_um: np.ndarray = field(default_factory=lambda: np.empty(0))


# --------------------------------------------------------------------------- #
# Diameter from a single frame
# --------------------------------------------------------------------------- #
def detect_droplets(frame: np.ndarray, calib: Calibration,
                    min_area_px: int = 25,
                    invert: bool = True) -> np.ndarray:
    """Return equivalent-circle diameters (um) of droplets in one frame.

    frame: 2-D uint8 grayscale.  Uses Otsu threshold + external contours; each
    contour's area gives an equivalent-circle diameter d = 2*sqrt(A/pi).
    """
    if not _HAVE_CV2:
        raise RuntimeError("OpenCV not available")
    if frame.ndim != 2:
        raise ValueError("expected 2-D grayscale frame")
    img = frame if frame.dtype == np.uint8 else _to_u8(frame)
    thr_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, mask = cv2.threshold(img, 0, 255, thr_type + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    di: list[float] = []
    for c in contours:
        a = cv2.contourArea(c)
        if a < min_area_px:
            continue
        # circularity gate: keep round-ish blobs (drops), drop channel walls
        peri = cv2.arcLength(c, True)
        if peri <= 0:
            continue
        circularity = 4 * np.pi * a / (peri * peri)
        if circularity < 0.6:
            continue
        d_px = 2.0 * np.sqrt(a / np.pi)
        di.append(d_px * calib.microns_per_pixel)
    return np.asarray(di, dtype=float)


def _to_u8(frame: np.ndarray) -> np.ndarray:
    f = frame.astype(np.float32)
    lo, hi = float(f.min()), float(f.max())
    if hi <= lo:
        return np.zeros_like(frame, dtype=np.uint8)
    return ((f - lo) / (hi - lo) * 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Frequency from a batch of frames
# --------------------------------------------------------------------------- #
def frequency_from_batch(frames: np.ndarray, fps: float,
                         roi_col: int | None = None) -> float:
    """Estimate generation frequency (Hz) from a downstream-signal time series.

    Takes the mean intensity in a 1-px-wide column downstream of the junction as
    a periodic signal (a droplet passing dips/raises brightness), and finds its
    dominant frequency by FFT.  Robust, cheap, and needs no per-droplet tracking.
    """
    frames = np.asarray(frames)
    n = frames.shape[0]
    if n < 4:
        return 0.0
    col = roi_col if roi_col is not None else frames.shape[2] // 2
    signal = frames[:, :, col].mean(axis=1).astype(np.float64)
    signal -= signal.mean()
    if np.allclose(signal, 0):
        return 0.0
    spec = np.abs(np.fft.rfft(signal * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    spec[0] = 0.0  # kill DC
    return float(freqs[int(np.argmax(spec))])


def frame_diff_regime_hint(frames: np.ndarray, thresh: float = 3.0) -> str:
    """Cheap regime hint from frame-to-frame differences (NOT the CNN classifier).

    High, periodic difference energy -> dripping; near-zero -> no_droplets; high
    but aperiodic -> jetting/unstable.  A placeholder for the learned classifier,
    useful as a sanity cross-check and a watchdog input.
    """
    frames = np.asarray(frames).astype(np.float32)
    if frames.shape[0] < 3:
        return "unknown"
    diffs = np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2))
    energy = diffs.mean()
    if energy < thresh:
        return "no_droplets"
    # periodicity via autocorrelation peak sharpness
    d = diffs - diffs.mean()
    ac = np.correlate(d, d, mode="full")[len(d) - 1:]
    if ac[0] <= 0:
        return "unstable"
    ac = ac / ac[0]
    peak = ac[1:].max() if len(ac) > 1 else 0.0
    return "dripping" if peak > 0.3 else "jetting"


# --------------------------------------------------------------------------- #
# Full batch measurement
# --------------------------------------------------------------------------- #
def measure_batch(frames: np.ndarray, fps: float, calib: Calibration,
                  diameter_stride: int = 5) -> Measurement:
    """Measure (D, CV, F, regime hint) from a batch of frames.

    diameter_stride: sample every k-th frame for diameter (droplets move slowly
    relative to frame rate; no need to segment every frame).
    """
    frames = np.asarray(frames)
    all_d: list[float] = []
    for i in range(0, frames.shape[0], diameter_stride):
        try:
            all_d.extend(detect_droplets(frames[i], calib).tolist())
        except RuntimeError:
            break
    per = np.asarray(all_d, dtype=float)
    if per.size:
        d_mean = float(per.mean())
        d_cv = float(per.std() / per.mean()) if per.mean() > 0 else 0.0
    else:
        d_mean, d_cv = 0.0, 0.0
    freq = frequency_from_batch(frames, fps)
    hint = frame_diff_regime_hint(frames)
    return Measurement(diameter_um=d_mean, diameter_cv=d_cv, frequency_hz=freq,
                       n_droplets=int(per.size), regime_hint=hint,
                       per_droplet_um=per)


# --------------------------------------------------------------------------- #
# Synthetic frame generator (for tests + benchmark, no camera needed)
# --------------------------------------------------------------------------- #
def synthetic_dripping(n_frames: int, hw=(256, 256), fps: float = 250.0,
                       drop_freq_hz: float = 40.0, diameter_px: float = 24.0,
                       spacing_px: float | None = None, seed: int = 0) -> np.ndarray:
    """Generate a batch of frames of dark droplets marching across a bright
    channel so that exactly ``drop_freq_hz`` droplets cross the centre column per
    second (i.e. the generation frequency equals the column-passage frequency).
    Deterministic given seed."""
    rng = np.random.default_rng(seed)
    h, w = hw
    frames = np.full((n_frames, h, w), 220, np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    r = diameter_px / 2.0
    pad = int(diameter_px) + 2
    span = w + 2 * pad
    spacing = spacing_px if spacing_px is not None else 3.0 * diameter_px
    # velocity such that one spacing passes a fixed point every 1/drop_freq s
    v_px_per_frame = spacing * drop_freq_hz / fps
    n_drop = int(span / spacing) + 2
    for t in range(n_frames):
        f = frames[t]
        shift = (t * v_px_per_frame) % spacing
        cy = h // 2 + int(rng.normal(0, 0.5))
        for k in range(n_drop + 1):
            cx = int(k * spacing + shift) - pad
            if -pad <= cx <= w + pad:
                f[(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r] = 40
    return frames


if __name__ == "__main__":
    cal = Calibration.from_reference(known_um=100.0, measured_px=50.0)  # 2 um/px
    frames = synthetic_dripping(60, drop_freq_hz=40.0, diameter_px=24.0)
    m = measure_batch(frames, fps=250.0, calib=cal)
    print(f"diameter={m.diameter_um:.1f} um (expect ~{24*2:.0f}), "
          f"CV={m.diameter_cv:.3f}, freq={m.frequency_hz:.1f} Hz (expect ~40), "
          f"n={m.n_droplets}, regime={m.regime_hint}")

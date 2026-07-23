"""Camera abstraction: ZWO ASI174MM over the ZWO ASI SDK, and a MockCamera that
serves synthetic frames so the whole stack runs with no hardware.

Frames are returned as in-memory numpy uint8 arrays; nothing here writes to disk.
The real driver path uses the ``zwoasi`` bindings (deferred import) so this module
loads anywhere.
"""
from __future__ import annotations

import numpy as np

from measurement import synthetic_dripping


class Camera:
    """Interface.  ``grab(n)`` returns an (n, H, W) uint8 batch."""

    def grab(self, n: int) -> np.ndarray:  # pragma: no cover - overridden
        raise NotImplementedError

    def close(self):
        pass


class MockCamera(Camera):
    """Serves synthetic dripping frames at a configurable generation frequency.

    The generation frequency and diameter can be updated to emulate the chip
    responding to pump commands, so a closed-loop test can watch the measured
    state track a target.
    """

    def __init__(self, hw=(256, 256), fps=250.0, drop_freq_hz=40.0,
                 diameter_px=24.0, seed=0):
        self.hw = hw
        self.fps = fps
        self.drop_freq_hz = drop_freq_hz
        self.diameter_px = diameter_px
        self._seed = seed

    def set_state(self, drop_freq_hz=None, diameter_px=None):
        if drop_freq_hz is not None:
            self.drop_freq_hz = float(drop_freq_hz)
        if diameter_px is not None:
            self.diameter_px = float(diameter_px)

    def grab(self, n: int) -> np.ndarray:
        self._seed += 1
        return synthetic_dripping(n, hw=self.hw, fps=self.fps,
                                  drop_freq_hz=self.drop_freq_hz,
                                  diameter_px=self.diameter_px, seed=self._seed)


class ZwoAsiCamera(Camera):
    """ZWO ASI174MM via the ``zwoasi`` SDK bindings.  Deferred import.

    Resolution x framerate is the binding constraint (Andrea: ~250 fps at
    800x600, ~1000 fps at small ROI).  Set a small ROI for high frame rate.
    """

    def __init__(self, roi=(800, 600), exposure_us=2000, gain=200,
                 sdk_lib=None):
        import zwoasi as asi  # deferred
        if sdk_lib:
            asi.init(sdk_lib)
        if asi.get_num_cameras() == 0:
            raise RuntimeError("no ZWO ASI camera found")
        self._cam = asi.Camera(0)
        self._cam.set_roi(width=roi[0], height=roi[1])
        self._cam.set_image_type(asi.ASI_IMG_RAW8)
        self._cam.set_control_value(asi.ASI_EXPOSURE, exposure_us)
        self._cam.set_control_value(asi.ASI_GAIN, gain)
        self._cam.start_video_capture()
        self.roi = roi

    def grab(self, n: int) -> np.ndarray:
        h, w = self.roi[1], self.roi[0]
        out = np.empty((n, h, w), np.uint8)
        for i in range(n):
            out[i] = self._cam.capture_video_frame().reshape(h, w)
        return out

    def close(self):
        try:
            self._cam.stop_video_capture()
            self._cam.close()
        except Exception:
            pass

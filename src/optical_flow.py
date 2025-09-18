from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("OpenCV is required for optical flow calculations") from exc

logger = logging.getLogger(__name__)


@dataclass
class OpticalFlowConfig:
    algorithm: str = "DIS"
    preset: str = "MEDIUM"
    pyr_scale: float = 0.5
    levels: int = 3
    winsize: int = 15
    iterations: int = 3
    poly_n: int = 5
    poly_sigma: float = 1.2
    flags: int = 0


class OpticalFlowCalculator:
    def __init__(self, config: OpticalFlowConfig) -> None:
        self.config = config
        algo = config.algorithm.upper()
        if algo not in {"DIS", "FARNEBACK"}:
            raise ValueError(f"Unsupported optical flow algorithm: {config.algorithm}")
        self.algorithm = algo
        self._dis = None
        if self.algorithm == "DIS":
            self._dis = self._create_dis(preset=config.preset)
        logger.debug("OpticalFlowCalculator initialized with %s", self.config)

    def _create_dis(self, preset: str):
        preset_map: Dict[str, int] = {
            "ULTRAFAST": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
            "FAST": cv2.DISOPTICAL_FLOW_PRESET_FAST,
            "MEDIUM": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
        }
        preset_key = preset_map.get(preset.upper(), cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        dis = cv2.DISOpticalFlow_create(preset_key)
        return dis

    def compute(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
        if prev_gray.ndim != 2 or curr_gray.ndim != 2:
            raise ValueError("Optical flow expects grayscale images")

        if self.algorithm == "DIS":
            flow = self._dis.calc(prev_gray, curr_gray, None)
        else:
            cfg = self.config
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray,
                curr_gray,
                None,
                pyr_scale=cfg.pyr_scale,
                levels=cfg.levels,
                winsize=cfg.winsize,
                iterations=cfg.iterations,
                poly_n=cfg.poly_n,
                poly_sigma=cfg.poly_sigma,
                flags=cfg.flags,
            )
        return flow.astype(np.float32)


def to_grayscale(frame_bgr: np.ndarray) -> np.ndarray:
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("Expected BGR frame with three channels")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

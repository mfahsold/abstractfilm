from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("OpenCV required for morph warp operations") from exc

logger = logging.getLogger(__name__)


def warp_with_flow(frame: np.ndarray, flow: np.ndarray, scale: float = 1.0) -> np.ndarray:
    if frame.shape[:2] != flow.shape[:2]:
        raise ValueError("Frame and flow dimensions must match")
    h, w = flow.shape[:2]
    flow_map = -flow.copy() * scale
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (flow_map[..., 0] + grid_x).astype(np.float32)
    map_y = (flow_map[..., 1] + grid_y).astype(np.float32)
    warped = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return warped


def blend_frames(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    alpha = np.clip(alpha, 0.0, 1.0)
    blended = cv2.addWeighted(a, alpha, b, 1.0 - alpha, 0.0)
    return blended


def compute_morph_pair(prev_frame: np.ndarray, curr_frame: np.ndarray, flow: np.ndarray, *, scale: float, alpha: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    warped_prev = warp_with_flow(prev_frame, flow, scale=scale)
    warped_current = warp_with_flow(curr_frame, -flow, scale=scale)
    morphed = blend_frames(warped_prev, warped_current, alpha)
    return warped_prev, warped_current, morphed

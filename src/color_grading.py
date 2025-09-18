from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("OpenCV required for color grading") from exc

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from .effects_profiles import GradingSettings


@dataclass
class Lut3D:
    size: int
    table: np.ndarray  # shape (size, size, size, 3)
    domain_min: np.ndarray
    domain_max: np.ndarray


def load_cube_lut(path: Path) -> Lut3D:
    size: Optional[int] = None
    domain_min = np.zeros(3, dtype=np.float32)
    domain_max = np.ones(3, dtype=np.float32)
    entries = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.split()
            key = tokens[0].upper()
            if key == "TITLE":
                continue
            if key == "DOMAIN_MIN":
                domain_min = np.array(list(map(float, tokens[1:4])), dtype=np.float32)
                continue
            if key == "DOMAIN_MAX":
                domain_max = np.array(list(map(float, tokens[1:4])), dtype=np.float32)
                continue
            if key == "LUT_3D_SIZE":
                size = int(tokens[1])
                continue
            if size is None:
                continue
            values = list(map(float, tokens[:3]))
            entries.append(values)
    if size is None or not entries:
        raise ValueError(f"Invalid LUT file: {path}")
    table = np.array(entries, dtype=np.float32)
    table = table.reshape((size, size, size, 3))
    return Lut3D(size=size, table=table, domain_min=domain_min, domain_max=domain_max)


def apply_lut(frame_bgr: np.ndarray, lut: Lut3D) -> np.ndarray:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    domain_scale = lut.domain_max - lut.domain_min
    domain_scale[domain_scale == 0] = 1.0
    coords = (frame_rgb - lut.domain_min) / domain_scale
    coords = np.clip(coords, 0.0, 1.0) * (lut.size - 1)
    indices = np.rint(coords).astype(int)
    r_idx = indices[..., 0]
    g_idx = indices[..., 1]
    b_idx = indices[..., 2]
    mapped = lut.table[r_idx, g_idx, b_idx]
    mapped = np.clip(mapped, 0.0, 1.0)
    out_rgb = (mapped * 255.0).astype(np.uint8)
    return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)


def retro_neon_grade(
    frame_bgr: np.ndarray,
    saturation_boost: float = 1.2,
    shadow_tint: Tuple[float, float, float] = (0.8, 0.9, 1.2),
    highlight_tint: Tuple[float, float, float] = (1.2, 1.0, 0.8),
    grading_settings: Optional["GradingSettings"] = None,
) -> np.ndarray:
    if grading_settings is not None:
        saturation_boost = grading_settings.saturation_boost
        shadow_tint = grading_settings.shadow_tint
        highlight_tint = grading_settings.highlight_tint
    frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    frame_hsv[..., 1] *= saturation_boost
    frame_hsv[..., 1] = np.clip(frame_hsv[..., 1], 0, 255)
    frame_hsv[..., 2] = np.clip(frame_hsv[..., 2], 0, 255)
    graded = cv2.cvtColor(frame_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0
    luminance = np.dot(graded, np.array([0.299, 0.587, 0.114]))
    tint = np.zeros_like(graded)
    shadow = np.array(shadow_tint, dtype=np.float32)
    highlight = np.array(highlight_tint, dtype=np.float32)
    luminance = np.clip(luminance, 0.0, 1.0)
    mix = luminance[..., None]
    tint = shadow * (1.0 - mix) + highlight * mix
    out = np.clip(graded * tint, 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)

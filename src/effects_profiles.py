from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml

logger = logging.getLogger(__name__)


@dataclass
class ReactionDiffusionSettings:
    diffusion_base: float = 0.8
    diffusion_scale: float = 0.2
    feedback_base: float = 0.4
    feedback_scale: float = 0.1
    time_step: float = 0.6


@dataclass
class FeedbackSettings:
    rd_mix_base: float = 0.35
    rd_mix_scale: float = 0.25
    glow_strength: float = 1.0


@dataclass
class GradingSettings:
    saturation_boost: float = 1.2
    shadow_tint: Tuple[float, float, float] = (0.8, 0.9, 1.2)
    highlight_tint: Tuple[float, float, float] = (1.2, 1.0, 0.8)


@dataclass
class EffectSettings:
    reaction_diffusion: ReactionDiffusionSettings = field(
        default_factory=ReactionDiffusionSettings
    )
    feedback: FeedbackSettings = field(default_factory=FeedbackSettings)
    grading: GradingSettings = field(default_factory=GradingSettings)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EffectSettings":
        rd_data = data.get("reaction_diffusion", {})
        fb_data = data.get("feedback", {})
        grading_data = data.get("grading", {})
        rd = ReactionDiffusionSettings(
            diffusion_base=float(rd_data.get("diffusion_base", 0.8)),
            diffusion_scale=float(rd_data.get("diffusion_scale", 0.2)),
            feedback_base=float(rd_data.get("feedback_base", 0.4)),
            feedback_scale=float(rd_data.get("feedback_scale", 0.1)),
            time_step=float(rd_data.get("time_step", 0.6)),
        )
        fb = FeedbackSettings(
            rd_mix_base=float(fb_data.get("rd_mix_base", 0.35)),
            rd_mix_scale=float(fb_data.get("rd_mix_scale", 0.25)),
            glow_strength=float(fb_data.get("glow_strength", 1.0)),
        )
        grading = GradingSettings(
            saturation_boost=float(grading_data.get("saturation_boost", 1.2)),
            shadow_tint=_ensure_tuple(grading_data.get("shadow_tint", (0.8, 0.9, 1.2))),
            highlight_tint=_ensure_tuple(grading_data.get("highlight_tint", (1.2, 1.0, 0.8))),
        )
        return cls(reaction_diffusion=rd, feedback=fb, grading=grading)


def _ensure_tuple(values: Any) -> Tuple[float, float, float]:
    if isinstance(values, (list, tuple)) and len(values) == 3:
        return tuple(float(v) for v in values)
    raise ValueError(f"Expected list/tuple of length 3, got {values!r}")


def load_effect_profile(path: Path) -> EffectSettings:
    path = path.expanduser()
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as fh:
        if path.suffix.lower() in {".yaml", ".yml"}:
            raw = yaml.safe_load(fh) or {}
        else:
            raw = json.load(fh)
    settings = EffectSettings.from_dict(raw)
    logger.debug("Loaded effect profile %s", path)
    return settings

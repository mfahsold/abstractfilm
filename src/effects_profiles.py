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
    diffusion_u_base: float = 0.16
    diffusion_u_scale: float = 0.04
    diffusion_v_base: float = 0.08
    diffusion_v_scale: float = 0.03
    feed_base: float = 0.055
    feed_scale: float = 0.02
    kill_base: float = 0.062
    kill_scale: float = 0.02
    advection_base: float = 0.18
    advection_scale: float = 0.28
    drive_influence: float = 0.08
    motion_floor: float = 0.18
    time_step: float = 1.0


@dataclass
class FeedbackSettings:
    rd_mix_base: float = 0.25
    rd_mix_scale: float = 0.25
    rd_mix_min: float = 0.1
    rd_mix_max: float = 0.6
    glow_strength: float = 1.0
    glow_color: Tuple[float, float, float] = (1.0, 0.35, 0.85)
    luminance_influence: float = 0.15
    displacement_strength: float = 0.02
    height_scale: float = 1.5
    light_dir: Tuple[float, float, float] = (0.25, 0.6, 1.0)


@dataclass
class GradingSettings:
    saturation_boost: float = 1.2
    shadow_tint: Tuple[float, float, float] = (0.8, 0.9, 1.2)
    highlight_tint: Tuple[float, float, float] = (1.2, 1.0, 0.8)


@dataclass
class AnalogSettings:
    film_grain: float = 0.05
    vignette: float = 0.1
    flicker: float = 0.04


@dataclass
class EffectSettings:
    reaction_diffusion: ReactionDiffusionSettings = field(
        default_factory=ReactionDiffusionSettings
    )
    feedback: FeedbackSettings = field(default_factory=FeedbackSettings)
    grading: GradingSettings = field(default_factory=GradingSettings)
    analog: AnalogSettings = field(default_factory=AnalogSettings)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EffectSettings":
        rd_data = data.get("reaction_diffusion", {})
        fb_data = data.get("feedback", {})
        grading_data = data.get("grading", {})
        analog_data = data.get("analog", {})
        rd = ReactionDiffusionSettings(
            diffusion_u_base=float(rd_data.get("diffusion_u_base", rd_data.get("diffusion_base", 0.16))),
            diffusion_u_scale=float(rd_data.get("diffusion_u_scale", rd_data.get("diffusion_scale", 0.04))),
            diffusion_v_base=float(rd_data.get("diffusion_v_base", rd_data.get("diffusion_secondary_base", 0.08))),
            diffusion_v_scale=float(rd_data.get("diffusion_v_scale", rd_data.get("diffusion_secondary_scale", 0.03))),
            feed_base=float(rd_data.get("feed_base", rd_data.get("feedback_base", 0.055))),
            feed_scale=float(rd_data.get("feed_scale", rd_data.get("feedback_scale", 0.02))),
            kill_base=float(rd_data.get("kill_base", 0.062)),
            kill_scale=float(rd_data.get("kill_scale", 0.02)),
            advection_base=float(rd_data.get("advection_base", 0.18)),
            advection_scale=float(rd_data.get("advection_scale", 0.28)),
            drive_influence=float(rd_data.get("drive_influence", 0.08)),
            motion_floor=float(rd_data.get("motion_floor", 0.18)),
            time_step=float(rd_data.get("time_step", 1.0)),
        )
        fb = FeedbackSettings(
            rd_mix_base=float(fb_data.get("rd_mix_base", 0.25)),
            rd_mix_scale=float(fb_data.get("rd_mix_scale", 0.25)),
            rd_mix_min=float(fb_data.get("rd_mix_min", 0.1)),
            rd_mix_max=float(fb_data.get("rd_mix_max", 0.6)),
            glow_strength=float(fb_data.get("glow_strength", 1.0)),
            glow_color=_ensure_tuple(fb_data.get("glow_color", (1.0, 0.35, 0.85))),
            luminance_influence=float(fb_data.get("luminance_influence", 0.15)),
            displacement_strength=float(fb_data.get("displacement_strength", 0.02)),
            height_scale=float(fb_data.get("height_scale", 1.5)),
            light_dir=_ensure_tuple(fb_data.get("light_dir", (0.25, 0.6, 1.0))),
        )
        grading = GradingSettings(
            saturation_boost=float(grading_data.get("saturation_boost", 1.2)),
            shadow_tint=_ensure_tuple(grading_data.get("shadow_tint", (0.8, 0.9, 1.2))),
            highlight_tint=_ensure_tuple(grading_data.get("highlight_tint", (1.2, 1.0, 0.8))),
        )
        analog = AnalogSettings(
            film_grain=float(analog_data.get("film_grain", 0.05)),
            vignette=float(analog_data.get("vignette", 0.1)),
            flicker=float(analog_data.get("flicker", 0.04)),
        )
        return cls(reaction_diffusion=rd, feedback=fb, grading=grading, analog=analog)


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

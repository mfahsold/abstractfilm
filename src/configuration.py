from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Set, TYPE_CHECKING

import yaml

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from .effects_profiles import EffectSettings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROFILE_DIR = PROJECT_ROOT / "effects_profiles"
DEFAULT_PROFILES = {
    "preview": DEFAULT_PROFILE_DIR / "preview.yaml",
    "full": DEFAULT_PROFILE_DIR / "full.yaml",
}


@dataclass
class SceneParameterSet:
    name: str = "default"
    flow_strength: float = 1.0
    morph_alpha: float = 0.5
    rd_steps: Optional[int] = None
    glow_boost: Optional[float] = None
    lut_path: Optional[Path] = None
    effects_profile_path: Optional[Path] = None
    effects_profile: Optional["EffectSettings"] = None

    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> "SceneParameterSet":
        profile_path = data.get("effects_profile") or data.get("effect_profile")
        return cls(
            name=name,
            flow_strength=float(data.get("flow_strength", 1.0)),
            morph_alpha=float(data.get("morph_alpha", 0.5)),
            rd_steps=(
                int(data["rd_steps"]) if data.get("rd_steps") is not None else None
            ),
            glow_boost=(
                float(data["glow_boost"]) if data.get("glow_boost") is not None else None
            ),
            lut_path=(
                Path(data["lut_path"]).expanduser()
                if data.get("lut_path")
                else None
            ),
            effects_profile_path=(
                Path(profile_path).expanduser() if profile_path else None
            ),
        )


@dataclass
class PipelineConfig:
    input_path: Path
    output_path: Path
    mode: str = "auto"
    scene_detect: bool = False
    include_audio: bool = False
    start_time: Optional[float] = None
    duration: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    flow_algo: str = "DIS"
    flow_preset: str = "MEDIUM"
    rd_steps: int = 3
    rd_iterations_preview: int = 1
    glow: bool = True
    glow_strength: float = 1.0
    use_lut: Optional[Path] = None
    morph_alpha: float = 0.5
    flow_scale: float = 1.0
    random_seed: Optional[int] = None
    preview_mode: bool = False
    full_mode: bool = False
    effects_profile_path: Optional[Path] = None
    effects_profile: Optional["EffectSettings"] = None
    output_frame_pattern: Optional[Path] = None
    frame_export: bool = False
    config_name: str = "cli"
    log_level: str = "INFO"
    scene_params: Dict[str, SceneParameterSet] = field(default_factory=dict)
    cli_overrides: Set[str] = field(default_factory=set, repr=False)

    def resolve(self) -> None:
        self.input_path = self.input_path.expanduser()
        self.output_path = self.output_path.expanduser()
        if self.use_lut:
            self.use_lut = self.use_lut.expanduser()
        if self.effects_profile_path:
            self.effects_profile_path = self.effects_profile_path.expanduser()
        if self.output_frame_pattern:
            self.output_frame_pattern = self.output_frame_pattern.expanduser()
        for key, params in list(self.scene_params.items()):
            if params.lut_path:
                params.lut_path = params.lut_path.expanduser()
            if params.effects_profile_path:
                params.effects_profile_path = params.effects_profile_path.expanduser()

    def apply_mode_defaults(self) -> None:
        target_mode = self.mode
        if self.preview_mode and target_mode == "auto":
            target_mode = "preview"
        if self.full_mode:
            target_mode = "full"
        if target_mode == "auto":
            target_mode = "full"

        if target_mode == "preview":
            if "width" not in self.cli_overrides and self.width is None:
                self.width = 640
            if "height" not in self.cli_overrides and self.height is None:
                self.height = 360
            if "rd_steps" not in self.cli_overrides:
                self.rd_steps = 2
            if "glow" not in self.cli_overrides:
                self.glow = False
            if "flow_algo" not in self.cli_overrides:
                self.flow_algo = "DIS"
            if "flow_preset" not in self.cli_overrides:
                self.flow_preset = "ULTRAFAST"
        elif target_mode == "full":
            if "rd_steps" not in self.cli_overrides:
                self.rd_steps = max(self.rd_steps, 5)
            if "glow" not in self.cli_overrides:
                self.glow = True
        self.mode = target_mode

    def resolve_effect_profile_path(self) -> Path:
        if self.effects_profile_path:
            return self.effects_profile_path
        default_path = DEFAULT_PROFILES.get(self.mode)
        if default_path and default_path.exists():
            return default_path
        raise FileNotFoundError(f"No effect profile defined for mode '{self.mode}'")


def load_preset(path: Path) -> Dict[str, Any]:
    path = path.expanduser()
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as fh:
        if path.suffix.lower() in {".yaml", ".yml"}:
            data = yaml.safe_load(fh)
        else:
            data = json.load(fh)
    logger.debug("Loaded preset %s", path)
    return data or {}


def build_config(cli_args: Dict[str, Any], preset: Optional[Dict[str, Any]] = None) -> PipelineConfig:
    merged: Dict[str, Any] = {}
    preset = preset or {}
    merged.update(preset)
    merged.update({k: v for k, v in cli_args.items() if v is not None})

    cli_overrides: Set[str] = {k for k, v in cli_args.items() if v is not None}

    mode = merged.get("mode", "auto")
    if merged.get("preview"):
        mode = "preview"
    if merged.get("full"):
        mode = "full"

    scene_param_defs = merged.pop("scene_params", {})
    scene_params = {
        name: SceneParameterSet.from_dict(name, payload)
        for name, payload in scene_param_defs.items()
    }

    config = PipelineConfig(
        input_path=Path(merged["input"]),
        output_path=Path(merged["output"]),
        mode=str(mode),
        scene_detect=bool(merged.get("scene_detect", False)),
        include_audio=bool(merged.get("include_audio", False)),
        start_time=merged.get("start"),
        duration=merged.get("duration"),
        width=merged.get("width"),
        height=merged.get("height"),
        flow_algo=str(merged.get("flow_algo", "DIS")).upper(),
        flow_preset=str(merged.get("flow_preset", "MEDIUM")).upper(),
        rd_steps=int(merged.get("rd_steps", 3)),
        rd_iterations_preview=int(merged.get("rd_iterations_preview", 1)),
        glow=bool(merged.get("glow", True)),
        glow_strength=float(merged.get("glow_strength", 1.0)),
        use_lut=Path(merged["use_lut"]) if merged.get("use_lut") else None,
        morph_alpha=float(merged.get("morph_alpha", 0.5)),
        flow_scale=float(merged.get("flow_scale", 1.0)),
        random_seed=merged.get("random_seed"),
        preview_mode=bool(merged.get("preview_mode", False) or merged.get("preview", False)),
        full_mode=bool(merged.get("full", False)),
        effects_profile_path=(
            Path(merged["effects_profile"]) if merged.get("effects_profile") else None
        ),
        output_frame_pattern=(
            Path(merged["output_frame_pattern"]) if merged.get("output_frame_pattern") else None
        ),
        frame_export=bool(merged.get("frame_export", False)),
        config_name=str(merged.get("config_name", "cli")),
        log_level=str(merged.get("log_level", "INFO"))
    )
    config.scene_params = scene_params
    config.cli_overrides = cli_overrides
    config.resolve()
    config.apply_mode_defaults()

    from .effects_profiles import load_effect_profile

    profile_path = config.resolve_effect_profile_path()
    config.effects_profile_path = profile_path
    config.effects_profile = load_effect_profile(profile_path)

    for params in config.scene_params.values():
        if params.effects_profile_path:
            params.effects_profile = load_effect_profile(params.effects_profile_path)
    return config

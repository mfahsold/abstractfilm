from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("OpenCV required for pipeline") from exc

from .configuration import PipelineConfig, SceneParameterSet
from .encode import VideoEncoder, extract_audio
from .effects_gl import ArtEffectProcessor
from .morph_warp import compute_morph_pair
from .optical_flow import OpticalFlowCalculator, OpticalFlowConfig, to_grayscale
from .scene_detect import Shot, detect_scenes, chunk_frames
from . import color_grading
from .effects_profiles import load_effect_profile, EffectSettings, AnalogSettings

logger = logging.getLogger(__name__)


class VideoPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._capture: Optional[cv2.VideoCapture] = None
        self._shots: List[Shot] = []
        self._encoder: Optional[VideoEncoder] = None
        self._effect_processor: Optional[ArtEffectProcessor] = None
        self._lut_cache: Dict[Path, color_grading.Lut3D] = {}
        self._current_shot: Optional[Shot] = None
        self._rng = np.random.default_rng(config.random_seed if config.random_seed is not None else None)
        self._flow_norm = 5.0
        self._low_motion_threshold = 0.45
        self._variant_profiles: List[Tuple[Path, EffectSettings]] = []
        self._vignette_cache: Dict[Tuple[int, int], np.ndarray] = {}
        self._load_variant_profiles()

    # ------------------------------------------------------------------
    def run(self) -> None:
        cfg = self.config
        random.seed(cfg.random_seed)
        if cfg.random_seed is not None:
            np.random.seed(cfg.random_seed)

        self._open_capture(cfg.input_path)
        fps = self._capture.get(cv2.CAP_PROP_FPS) or 30.0
        orig_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        start_frame = int(fps * cfg.start_time) if cfg.start_time else 0
        if cfg.duration:
            end_frame = start_frame + int(math.ceil(fps * cfg.duration))
        else:
            end_frame = total_frames if total_frames > 0 else None

        if start_frame:
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        target_width = cfg.width or orig_width
        target_height = cfg.height or orig_height

        if cfg.scene_detect:
            self._shots = detect_scenes(str(cfg.input_path))
        else:
            self._shots = [
                Shot(
                    index=0,
                    start_frame=start_frame,
                    end_frame=(end_frame - 1) if end_frame else total_frames - 1,
                    start_time=cfg.start_time or 0.0,
                    end_time=(cfg.start_time or 0.0) + (cfg.duration or 0.0),
                )
            ]
        logger.info("Using %d shots", len(self._shots))

        of_config = OpticalFlowConfig(algorithm=cfg.flow_algo, preset=cfg.flow_preset)
        flow_calc = OpticalFlowCalculator(of_config)

        if cfg.include_audio:
            audio_temp = cfg.output_path.with_suffix(".temp_audio.aac")
            extract_audio(cfg.input_path, audio_temp)
        else:
            audio_temp = None

        self._encoder = VideoEncoder(
            cfg.output_path,
            fps=fps,
            frame_size=(target_width, target_height),
            include_audio=cfg.include_audio and audio_temp is not None,
            audio_source=audio_temp,
        )

        prev_frame = None
        prev_gray = None
        frame_index = start_frame
        processed_frames = 0
        total_work = (end_frame - start_frame) if end_frame else (total_frames - start_frame)
        progress = tqdm(total=total_work if total_work > 0 else None, desc="Rendering", unit="frame")

        while True:
            if end_frame is not None and frame_index >= end_frame:
                break
            success, frame_bgr = self._capture.read()
            if not success or frame_bgr is None:
                break

            frame_bgr = cv2.resize(frame_bgr, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

            shot = self._get_shot_for_frame(frame_index)
            if shot and (self._current_shot is None or shot.index != self._current_shot.index):
                self._switch_shot(
                    shot,
                    width=target_width,
                    height=target_height,
                    initial_frame=frame_bgr,
                )
            params = self._resolve_params_for_shot(shot)

            curr_gray = to_grayscale(frame_bgr)
            if prev_frame is None or prev_gray is None or self._effect_processor is None:
                prev_frame = frame_bgr
                prev_gray = curr_gray
                frame_index += 1
                progress.update(1)
                continue

            flow = flow_calc.compute(prev_gray, curr_gray)
            flow_norm = np.linalg.norm(flow, axis=2)
            flow_magnitude = float(np.mean(flow_norm))
            morph_strength = np.clip(flow_magnitude / self._flow_norm, 0.0, 1.0)
            dynamic_alpha = self._compute_dynamic_alpha(params.morph_alpha, morph_strength)
            warped_prev, warped_curr, morphed = compute_morph_pair(
                prev_frame,
                frame_bgr,
                flow * params.flow_strength,
                scale=cfg.flow_scale,
                alpha=dynamic_alpha,
            )
            morphed, morph_strength = self._apply_low_motion_variation(
                morphed, flow_magnitude, frame_index, morph_strength
            )

            flow_texture = self._prepare_flow_texture(flow)
            luminance = self._compute_luminance(frame_bgr)

            processed = self._effect_processor.process(
                morphed,
                flow_texture,
                frame_index=frame_index,
                morph_strength=morph_strength,
                luminance=luminance,
            )
            graded = self._apply_grading(processed, params, morph_strength, frame_index)

            self._encoder.write(graded)
            if cfg.frame_export and cfg.output_frame_pattern:
                self._export_frame(cfg.output_frame_pattern, processed_frames, graded)

            prev_frame = frame_bgr
            prev_gray = curr_gray
            frame_index += 1
            processed_frames += 1
            progress.update(1)

        progress.close()
        if self._encoder:
            self._encoder.close()
        if self._effect_processor:
            self._effect_processor.release()
        if self._capture:
            self._capture.release()

    # ------------------------------------------------------------------
    def _open_capture(self, path: Path) -> None:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open input video: {path}")
        self._capture = capture

    def _get_shot_for_frame(self, frame_index: int) -> Optional[Shot]:
        if not self._shots:
            return None
        shot = chunk_frames(frame_index, self._shots)
        return shot

    def _switch_shot(
        self,
        shot: Shot,
        *,
        width: int,
        height: int,
        initial_frame: Optional[np.ndarray] = None,
    ) -> None:
        logger.info("Entering shot %d (%d-%d)", shot.index, shot.start_frame, shot.end_frame)
        if self._effect_processor:
            self._effect_processor.release()
        self._current_shot = shot
        params = self._resolve_params_for_shot(shot)
        rd_steps = params.rd_steps if params.rd_steps is not None else self.config.rd_steps
        glow_strength = params.glow_boost if params.glow_boost is not None else self.config.glow_strength
        effect_settings = (
            params.effects_profile
            or self._choose_effect_profile_for_shot(shot.index)
            or self.config.effects_profile
        )
        self._effect_processor = ArtEffectProcessor(
            width=width,
            height=height,
            rd_steps=rd_steps,
            glow=self.config.glow,
            glow_strength=glow_strength,
            enable_gpu=True,
            effect_settings=effect_settings,
        )
        if initial_frame is not None and self._effect_processor:
            self._effect_processor.prime_state(initial_frame)

    def _resolve_params_for_shot(self, shot: Optional[Shot]) -> SceneParameterSet:
        cfg = self.config
        if shot is None:
            return cfg.scene_params.get("default", SceneParameterSet())
        key = f"shot_{shot.index}"
        if key in cfg.scene_params:
            return cfg.scene_params[key]
        return cfg.scene_params.get("default", SceneParameterSet())

    def _apply_grading(
        self,
        frame_bgr: np.ndarray,
        params: SceneParameterSet,
        morph_strength: float,
        frame_index: int,
    ) -> np.ndarray:
        lut_path = params.lut_path or self.config.use_lut
        if lut_path:
            lut = self._lut_cache.get(lut_path)
            if lut is None:
                lut = color_grading.load_cube_lut(lut_path)
                self._lut_cache[lut_path] = lut
            frame_bgr = color_grading.apply_lut(frame_bgr, lut)
        profile = None
        if self._effect_processor is not None:
            profile = self._effect_processor.effect_settings
        if profile is None:
            profile = params.effects_profile or self.config.effects_profile
        grading_settings = profile.grading if profile else None
        frame_bgr = color_grading.retro_neon_grade(frame_bgr, grading_settings=grading_settings)
        if params.glow_boost:
            glow = cv2.GaussianBlur(frame_bgr, (0, 0), sigmaX=5.0)
            frame_bgr = cv2.addWeighted(frame_bgr, 1.0, glow, params.glow_boost, 0)
        analog_settings = profile.analog if profile else None
        frame_bgr = self._apply_analog_overlay(frame_bgr, analog_settings, frame_index, morph_strength)
        return frame_bgr

    def _export_frame(self, pattern: Path, index: int, frame_bgr: np.ndarray) -> None:
        target_path = Path(str(pattern).format(index=index, i=index, frame=index))
        target_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(target_path), frame_bgr)

    def _compute_dynamic_alpha(self, base_alpha: float, morph_strength: float) -> float:
        base = float(np.clip(base_alpha, 0.1, 0.8))
        adjusted = base - 0.18 * morph_strength + 0.05
        return float(np.clip(adjusted, 0.25, 0.6))

    def _apply_low_motion_variation(
        self,
        frame_bgr: np.ndarray,
        flow_magnitude: float,
        frame_index: int,
        morph_strength: float,
    ) -> Tuple[np.ndarray, float]:
        if flow_magnitude >= self._low_motion_threshold:
            return frame_bgr, morph_strength
        deficit = (self._low_motion_threshold - flow_magnitude) / max(self._low_motion_threshold, 1e-6)
        amplitude = 1.6 * deficit
        if amplitude <= 1e-3:
            return frame_bgr, morph_strength
        h, w = frame_bgr.shape[:2]
        noise_x = self._rng.normal(0.0, 1.0, (h, w)).astype(np.float32)
        noise_y = self._rng.normal(0.0, 1.0, (h, w)).astype(np.float32)
        dx = cv2.GaussianBlur(noise_x, (0, 0), sigmaX=1.2) * amplitude
        dy = cv2.GaussianBlur(noise_y, (0, 0), sigmaX=1.2) * amplitude
        yy, xx = np.indices((h, w), dtype=np.float32)
        map_x = np.clip(xx + dx, 0, w - 1).astype(np.float32)
        map_y = np.clip(yy + dy, 0, h - 1).astype(np.float32)
        remapped = cv2.remap(
            frame_bgr,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        boosted = morph_strength + 0.25 * deficit
        return remapped, float(np.clip(boosted, 0.0, 1.0))

    def _apply_analog_overlay(
        self,
        frame_bgr: np.ndarray,
        analog_settings: Optional[AnalogSettings],
        frame_index: int,
        morph_strength: float,
    ) -> np.ndarray:
        if analog_settings is None:
            return frame_bgr
        frame = frame_bgr.astype(np.float32) / 255.0
        if analog_settings.flicker > 0.0:
            flicker_strength = analog_settings.flicker * (0.5 + 0.5 * morph_strength)
            flicker = 1.0 + flicker_strength * math.sin(frame_index * 0.12)
            frame *= flicker
        if analog_settings.film_grain > 0.0:
            grain = self._rng.normal(0.0, 1.0, frame.shape[:2]).astype(np.float32)
            grain = cv2.GaussianBlur(grain, (0, 0), sigmaX=0.8)
            grain_gain = analog_settings.film_grain * (0.5 + 0.5 * (1.0 - morph_strength))
            frame += grain_gain * grain[..., None]
        if analog_settings.vignette > 0.0:
            mask = self._get_vignette_mask(frame.shape[1], frame.shape[0])
            frame *= 1.0 - analog_settings.vignette * mask[..., None]
        frame = np.clip(frame, 0.0, 1.0)
        return (frame * 255.0).astype(np.uint8)

    def _get_vignette_mask(self, width: int, height: int) -> np.ndarray:
        key = (width, height)
        if key not in self._vignette_cache:
            yy, xx = np.indices((height, width), dtype=np.float32)
            cx = (width - 1) / 2.0
            cy = (height - 1) / 2.0
            distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            max_radius = np.sqrt(cx**2 + cy**2)
            mask = np.clip(distance / max_radius, 0.0, 1.0) ** 1.5
            self._vignette_cache[key] = mask
        return self._vignette_cache[key]

    def _load_variant_profiles(self) -> None:
        base_dir = Path(__file__).resolve().parent.parent / "effects_profiles"
        candidate_files = [
            "variant_a.yaml",
            "variant_b.yaml",
            "variant_c.yaml",
        ]
        for filename in candidate_files:
            path = base_dir / filename
            if not path.exists():
                continue
            try:
                profile = load_effect_profile(path)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to load effect profile %s: %s", path, exc)
                continue
            self._variant_profiles.append((path, profile))
        if self._variant_profiles:
            logger.info(
                "Loaded %d variant effect profiles", len(self._variant_profiles)
            )

    def _choose_effect_profile_for_shot(self, shot_index: int) -> Optional[EffectSettings]:
        if not self._variant_profiles:
            return None
        _, profile = self._variant_profiles[shot_index % len(self._variant_profiles)]
        return profile

    def _compute_luminance(self, frame_bgr: np.ndarray) -> float:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray) / 255.0)

    def _prepare_flow_texture(self, flow: np.ndarray) -> np.ndarray:
        if flow.size == 0:
            if self._effect_processor is not None:
                return np.zeros((self._effect_processor.height, self._effect_processor.width, 2), dtype=np.float32)
            return np.zeros((1, 1, 2), dtype=np.float32)
        flow_norm = np.linalg.norm(flow, axis=2)
        scale = np.percentile(flow_norm, 95)
        if not np.isfinite(scale) or scale < 1e-3:
            scale = 1.0
        normalized = np.clip(flow / scale, -1.0, 1.0)
        return normalized.astype(np.float32)

def pipeline_from_config(config: PipelineConfig) -> VideoPipeline:
    logger.info("Pipeline configuration: %s", json.dumps(asdict(config), default=str, indent=2))
    return VideoPipeline(config)

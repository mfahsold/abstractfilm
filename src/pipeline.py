from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

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
                self._switch_shot(shot, width=target_width, height=target_height)
            params = self._resolve_params_for_shot(shot)

            curr_gray = to_grayscale(frame_bgr)
            if prev_frame is None or prev_gray is None or self._effect_processor is None:
                prev_frame = frame_bgr
                prev_gray = curr_gray
                frame_index += 1
                progress.update(1)
                continue

            flow = flow_calc.compute(prev_gray, curr_gray)
            warped_prev, warped_curr, morphed = compute_morph_pair(
                prev_frame,
                frame_bgr,
                flow * params.flow_strength,
                scale=cfg.flow_scale,
                alpha=params.morph_alpha,
            )

            flow_magnitude = float(np.mean(np.linalg.norm(flow, axis=2)))
            morph_strength = np.clip(flow_magnitude / 5.0, 0.0, 1.0)

            processed = self._effect_processor.process(
                morphed,
                frame_index=frame_index,
                morph_strength=morph_strength,
            )
            graded = self._apply_grading(processed, params)

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

    def _switch_shot(self, shot: Shot, *, width: int, height: int) -> None:
        logger.info("Entering shot %d (%d-%d)", shot.index, shot.start_frame, shot.end_frame)
        if self._effect_processor:
            self._effect_processor.release()
        self._current_shot = shot
        params = self._resolve_params_for_shot(shot)
        rd_steps = params.rd_steps if params.rd_steps is not None else self.config.rd_steps
        glow_strength = params.glow_boost if params.glow_boost is not None else self.config.glow_strength
        effect_settings = params.effects_profile or self.config.effects_profile
        self._effect_processor = ArtEffectProcessor(
            width=width,
            height=height,
            rd_steps=rd_steps,
            glow=self.config.glow,
            glow_strength=glow_strength,
            enable_gpu=True,
            effect_settings=effect_settings,
        )

    def _resolve_params_for_shot(self, shot: Optional[Shot]) -> SceneParameterSet:
        cfg = self.config
        if shot is None:
            return cfg.scene_params.get("default", SceneParameterSet())
        key = f"shot_{shot.index}"
        if key in cfg.scene_params:
            return cfg.scene_params[key]
        return cfg.scene_params.get("default", SceneParameterSet())

    def _apply_grading(self, frame_bgr: np.ndarray, params: SceneParameterSet) -> np.ndarray:
        lut_path = params.lut_path or self.config.use_lut
        if lut_path:
            lut = self._lut_cache.get(lut_path)
            if lut is None:
                lut = color_grading.load_cube_lut(lut_path)
                self._lut_cache[lut_path] = lut
            frame_bgr = color_grading.apply_lut(frame_bgr, lut)
        profile = params.effects_profile or self.config.effects_profile
        grading_settings = profile.grading if profile else None
        frame_bgr = color_grading.retro_neon_grade(frame_bgr, grading_settings=grading_settings)
        if params.glow_boost:
            glow = cv2.GaussianBlur(frame_bgr, (0, 0), sigmaX=5.0)
            frame_bgr = cv2.addWeighted(frame_bgr, 1.0, glow, params.glow_boost, 0)
        return frame_bgr

    def _export_frame(self, pattern: Path, index: int, frame_bgr: np.ndarray) -> None:
        target_path = Path(str(pattern).format(index=index, i=index, frame=index))
        target_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(target_path), frame_bgr)


def pipeline_from_config(config: PipelineConfig) -> VideoPipeline:
    logger.info("Pipeline configuration: %s", json.dumps(asdict(config), default=str, indent=2))
    return VideoPipeline(config)

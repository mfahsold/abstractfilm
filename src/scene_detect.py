from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)

try:
    from scenedetect import SceneManager
    from scenedetect.detectors import AdaptiveDetector, ContentDetector
    from scenedetect.video_manager import VideoManager
except ImportError:  # pragma: no cover - optional dependency
    SceneManager = None
    AdaptiveDetector = None
    ContentDetector = None
    VideoManager = None

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


@dataclass
class Shot:
    index: int
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float

    @property
    def frame_count(self) -> int:
        return max(0, self.end_frame - self.start_frame + 1)


def _fallback_single_shot(video_path: str) -> List[Shot]:
    if cv2 is None:
        logger.warning("No OpenCV available, returning default single-shot placeholder")
        return [Shot(index=0, start_frame=0, end_frame=-1, start_time=0.0, end_time=-1.0)]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning("Unable to open video for fallback shot detection: %s", video_path)
        return [Shot(index=0, start_frame=0, end_frame=-1, start_time=0.0, end_time=-1.0)]

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) - 1
    duration = frame_count / fps if fps else 0
    cap.release()
    return [Shot(index=0, start_frame=0, end_frame=frame_count, start_time=0.0, end_time=duration)]


def detect_scenes(video_path: str, adaptive: bool = True, threshold: float = 30.0) -> List[Shot]:
    if SceneManager is None or VideoManager is None:
        logger.info("PySceneDetect not available, using fallback shot covering entire video")
        return _fallback_single_shot(video_path)

    video_manager = VideoManager([video_path])
    stats_file_path = None
    try:
        video_manager.set_downscale_factor()
        video_manager.start()
        scene_manager = SceneManager()
        if adaptive and AdaptiveDetector is not None:
            scene_manager.add_detector(AdaptiveDetector())
        else:
            scene_manager.add_detector(ContentDetector(threshold=threshold))

        scene_manager.detect_scenes(frame_source=video_manager)
        scene_list = scene_manager.get_scene_list()
        shots: List[Shot] = []
        for idx, (start_time, end_time) in enumerate(scene_list):
            start_frame = start_time.get_frames()
            end_frame = end_time.get_frames() - 1
            shots.append(
                Shot(
                    index=idx,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    start_time=start_time.get_seconds(),
                    end_time=end_time.get_seconds(),
                )
            )
        if not shots:
            shots = _fallback_single_shot(video_path)
        logger.info("Detected %d scenes", len(shots))
        return shots
    finally:
        video_manager.release()
        if stats_file_path:
            try:
                stats_file_path.unlink()
            except OSError:
                pass


def chunk_frames(frame_index: int, shots: List[Shot]) -> Optional[Shot]:
    for shot in shots:
        if shot.start_frame <= frame_index <= shot.end_frame:
            return shot
    return shots[-1] if shots else None

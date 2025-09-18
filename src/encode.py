from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Optional

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("OpenCV required for encoding") from exc

logger = logging.getLogger(__name__)


class VideoEncoder:
    def __init__(
        self,
        output_path: Path,
        *,
        fps: float,
        frame_size: tuple[int, int],
        crf: int = 18,
        include_audio: bool = False,
        audio_source: Optional[Path] = None,
    ) -> None:
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size
        self.crf = crf
        self.include_audio = include_audio
        self.audio_source = audio_source
        self._writer_path = output_path.with_suffix(".temp.mp4") if include_audio else output_path
        self._writer = self._create_writer(self._writer_path)
        if self._writer is None:
            raise RuntimeError(
                "Failed to open VideoWriter. Ensure FFmpeg with MP4 support is installed or adjust output codec."
            )
        logger.info("VideoEncoder initialised: %s", self._writer_path)

    def _create_writer(self, path: Path) -> Optional[cv2.VideoWriter]:
        fourcc_candidates = ["avc1", "H264", "mp4v", "XVID"]
        for code in fourcc_candidates:
            fourcc = cv2.VideoWriter_fourcc(*code)
            writer = cv2.VideoWriter(str(path), fourcc, self.fps, self.frame_size)
            if writer.isOpened():
                logger.info("Using VideoWriter fourcc=%s", code)
                return writer
            writer.release()
        return None

    def write(self, frame_bgr: np.ndarray) -> None:
        if frame_bgr.shape[1] != self.frame_size[0] or frame_bgr.shape[0] != self.frame_size[1]:
            frame_bgr = cv2.resize(frame_bgr, self.frame_size, interpolation=cv2.INTER_LINEAR)
        self._writer.write(frame_bgr)

    def close(self) -> None:
        if self._writer:
            self._writer.release()
            self._writer = None
        if self.include_audio and self.audio_source:
            self._mux_audio()

    def _mux_audio(self) -> None:
        final_path = self.output_path
        temp_path = self._writer_path
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(temp_path),
            "-i",
            str(self.audio_source),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "256k",
            str(final_path),
        ]
        logger.info("Muxing audio via FFmpeg -> %s", final_path)
        subprocess.run(cmd, check=True)
        temp_path.unlink(missing_ok=True)


def extract_audio(input_path: Path, output_path: Path) -> Path:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-acodec",
        "copy",
        str(output_path),
    ]
    logger.info("Extracting audio: %s -> %s", input_path, output_path)
    subprocess.run(cmd, check=True)
    return output_path

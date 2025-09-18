from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("OpenCV required for effects pipeline") from exc

try:
    import moderngl
except ImportError:  # pragma: no cover - runtime fallback
    moderngl = None

from .effects_profiles import EffectSettings

logger = logging.getLogger(__name__)


class ArtEffectProcessor:
    def __init__(
        self,
        width: int,
        height: int,
        *,
        shader_dir: Optional[Path] = None,
        rd_steps: int = 3,
        glow: bool = True,
        glow_strength: float = 1.0,
        enable_gpu: bool = True,
        effect_settings: Optional[EffectSettings] = None,
    ) -> None:
        self.width = width
        self.height = height
        self.rd_steps = max(1, rd_steps)
        self.glow = glow
        self.glow_strength = glow_strength
        self.shader_dir = shader_dir or Path(__file__).resolve().parent.parent / "shaders"
        self.enable_gpu = enable_gpu and moderngl is not None
        self.effect_settings = effect_settings or EffectSettings()
        self._ctx: Optional[moderngl.Context] = None
        self._quad_vbo = None
        self._frame_tex = None
        self._flow_tex = None
        self._compose_tex = None
        self._state_textures = []
        self._state_fbos = []
        self._compose_fbo = None
        self._rd_program = None
        self._feedback_program = None
        self._rd_vao = None
        self._feedback_vao = None
        self._active_state_index = 0
        if self.enable_gpu:
            try:
                self._init_gpu()
            except Exception as exc:  # pragma: no cover - GPU fallback
                logger.warning("Falling back to CPU effects due to GPU init error: %s", exc)
                self.enable_gpu = False

    # ---------------------------------------------------------------------
    # GPU INITIALISATION
    # ---------------------------------------------------------------------
    def _init_gpu(self) -> None:
        assert moderngl is not None
        self._ctx = moderngl.create_context(standalone=True)
        self._ctx.enable(moderngl.BLEND)

        quad_data = np.array(
            [
                -1.0,
                -1.0,
                0.0,
                0.0,
                1.0,
                -1.0,
                1.0,
                0.0,
                -1.0,
                1.0,
                0.0,
                1.0,
                -1.0,
                1.0,
                0.0,
                1.0,
                1.0,
                -1.0,
                1.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
            ],
            dtype="f4",
        )
        self._quad_vbo = self._ctx.buffer(quad_data.tobytes())

        def load_shader(path: Path) -> str:
            with path.open("r", encoding="utf-8") as fh:
                return fh.read()

        vertex_src = load_shader(self.shader_dir / "fullscreen_quad.vs")
        rd_src = load_shader(self.shader_dir / "reaction_diffusion.fs")
        feedback_src = load_shader(self.shader_dir / "feedback.fs")

        state_kwargs = dict(size=(self.width, self.height), components=4, dtype="f4")
        zero_state = np.zeros((self.height, self.width, 4), dtype="f4")
        zero_state[..., 0] = 1.0  # Gray-Scott starts with U ~ 1.0
        self._state_textures = [
            self._ctx.texture(**state_kwargs),
            self._ctx.texture(**state_kwargs),
        ]
        for tex in self._state_textures:
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            tex.repeat_x = True
            tex.repeat_y = True
            tex.write(zero_state.tobytes())

        self._state_fbos = [self._ctx.framebuffer(color_attachments=[tex]) for tex in self._state_textures]

        self._frame_tex = self._ctx.texture(size=(self.width, self.height), components=4, dtype="f4")
        self._frame_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._frame_tex.repeat_x = False
        self._frame_tex.repeat_y = False

        self._flow_tex = self._ctx.texture(size=(self.width, self.height), components=2, dtype="f4")
        self._flow_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._flow_tex.repeat_x = False
        self._flow_tex.repeat_y = False

        self._compose_tex = self._ctx.texture(size=(self.width, self.height), components=4, dtype="f4")
        self._compose_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._compose_tex.repeat_x = False
        self._compose_tex.repeat_y = False
        self._compose_fbo = self._ctx.framebuffer(color_attachments=[self._compose_tex])

        self._rd_program = self._ctx.program(vertex_shader=vertex_src, fragment_shader=rd_src)
        self._feedback_program = self._ctx.program(vertex_shader=vertex_src, fragment_shader=feedback_src)

        self._rd_vao = self._ctx.simple_vertex_array(
            self._rd_program, self._quad_vbo, "in_pos", "in_uv"
        )
        self._feedback_vao = self._ctx.simple_vertex_array(
            self._feedback_program, self._quad_vbo, "in_pos", "in_uv"
        )
        self._ctx.clear(0.0, 0.0, 0.0, 1.0)
        logger.info("ModernGL effect processor initialised (%sx%s)", self.width, self.height)

    # ---------------------------------------------------------------------
    # PROCESSING
    # ---------------------------------------------------------------------
    def process(
        self,
        frame_bgr: np.ndarray,
        flow_map: np.ndarray,
        *,
        frame_index: int,
        morph_strength: float,
        luminance: float,
    ) -> np.ndarray:
        if frame_bgr.shape[1] != self.width or frame_bgr.shape[0] != self.height:
            frame_bgr = cv2.resize(frame_bgr, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        if not self.enable_gpu:
            return self._cpu_fallback(frame_bgr)
        return self._gpu_process(
            frame_bgr,
            flow_map,
            frame_index=frame_index,
            morph_strength=morph_strength,
            luminance=luminance,
        )

    def _gpu_process(
        self,
        frame_bgr: np.ndarray,
        flow_map: np.ndarray,
        *,
        frame_index: int,
        morph_strength: float,
        luminance: float,
    ) -> np.ndarray:
        assert self._ctx is not None
        rgba = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA).astype("f4") / 255.0
        rgba = np.flipud(rgba)
        self._frame_tex.write(rgba.tobytes())

        if self._flow_tex is not None:
            flow_bytes = np.flipud(flow_map).astype("f4").tobytes()
            self._flow_tex.write(flow_bytes)

        rd = self.effect_settings.reaction_diffusion
        motion_level = max(float(morph_strength), rd.motion_floor)
        diffusion_u = rd.diffusion_u_base + rd.diffusion_u_scale * motion_level
        diffusion_v = rd.diffusion_v_base + rd.diffusion_v_scale * motion_level
        feed = rd.feed_base + rd.feed_scale * motion_level
        kill = rd.kill_base + rd.kill_scale * motion_level
        advection = rd.advection_base + rd.advection_scale * motion_level
        drive_influence = rd.drive_influence
        time_step = rd.time_step / max(float(self.rd_steps), 1.0)

        fb = self.effect_settings.feedback
        rd_mix_value = fb.rd_mix_base + fb.rd_mix_scale * motion_level + fb.luminance_influence * (1.0 - luminance)
        rd_mix_value = float(np.clip(rd_mix_value, fb.rd_mix_min, fb.rd_mix_max))
        glow_value = fb.glow_strength * self.glow_strength if self.glow else 0.0
        glow_value *= 0.6 + 0.4 * (1.0 - luminance)
        displacement_value = fb.displacement_strength
        height_scale_value = fb.height_scale
        light_dir = np.array(fb.light_dir, dtype=np.float32)
        if np.linalg.norm(light_dir) == 0.0:
            light_dir = np.array([0.25, 0.6, 1.0], dtype=np.float32)
        light_dir = light_dir / np.linalg.norm(light_dir)

        active = self._active_state_index
        passive = 1 - active

        self._rd_program["state_tex"].value = 0
        self._rd_program["drive_tex"].value = 1
        if self._flow_tex is not None:
            self._rd_program["flow_tex"].value = 2
        self._rd_program["diffusion_u"].value = diffusion_u
        self._rd_program["diffusion_v"].value = diffusion_v
        self._rd_program["feed"].value = feed
        self._rd_program["kill"].value = kill
        self._rd_program["advection"].value = advection
        self._rd_program["drive_influence"].value = drive_influence
        self._rd_program["time_step"].value = time_step

        for step in range(self.rd_steps):
            target_index = passive if step == 0 else active
            state_tex = self._state_textures[active]
            target_fbo = self._state_fbos[target_index]

            target_fbo.use()
            state_tex.use(location=0)
            self._frame_tex.use(location=1)
            if self._flow_tex is not None:
                self._flow_tex.use(location=2)
            self._rd_vao.render()
            active, passive = target_index, active

        self._active_state_index = active

        self._compose_fbo.use()
        self._state_textures[self._active_state_index].use(location=0)
        self._frame_tex.use(location=1)
        if self._flow_tex is not None:
            self._flow_tex.use(location=2)
        self._feedback_program["rd_tex"].value = 0
        self._feedback_program["video_tex"].value = 1
        if self._flow_tex is not None:
            self._feedback_program["flow_tex"].value = 2
        self._feedback_program["rd_mix"].value = rd_mix_value
        self._feedback_program["glow_strength"].value = glow_value if self.glow else 0.0
        glow_color = np.clip(self.effect_settings.feedback.glow_color, 0.0, 2.0)
        self._feedback_program["glow_color"].value = tuple(float(c) for c in glow_color)
        self._feedback_program["displacement"].value = displacement_value
        self._feedback_program["height_scale"].value = height_scale_value
        self._feedback_program["light_dir"].value = tuple(float(c) for c in light_dir)
        self._feedback_vao.render()

        data = self._compose_fbo.read(components=4, dtype="f4")
        frame = np.frombuffer(data, dtype=np.float32).reshape((self.height, self.width, 4))
        frame = np.flipud(frame)
        frame = np.clip(frame, 0.0, 1.0)
        frame = (frame * 255.0).astype(np.uint8)
        return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

    def _cpu_fallback(self, frame_bgr: np.ndarray) -> np.ndarray:
        fb = self.effect_settings.feedback
        glow_value = fb.glow_strength * self.glow_strength if self.glow else 0.0
        blur = cv2.GaussianBlur(frame_bgr, (0, 0), sigmaX=3.0)
        glow = cv2.addWeighted(frame_bgr, 1.0, blur, glow_value * 0.5, 0)

        grading = self.effect_settings.grading
        hsv = cv2.cvtColor(glow, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * grading.saturation_boost, 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2], 0, 255)
        saturated = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0

        luminance = np.dot(saturated, np.array([0.299, 0.587, 0.114]))
        mix = luminance[..., None]
        shadow = np.array(grading.shadow_tint, dtype=np.float32)
        highlight = np.array(grading.highlight_tint, dtype=np.float32)
        tinted = saturated * (shadow * (1.0 - mix) + highlight * mix)
        tinted = np.clip(tinted, 0.0, 1.0)

        analog = self.effect_settings.analog
        if analog.film_grain > 0.0:
            grain = np.random.default_rng().normal(0.0, 1.0, tinted.shape[:2]).astype(np.float32)
            grain = cv2.GaussianBlur(grain, (0, 0), sigmaX=0.8)
            tinted += analog.film_grain * grain[..., None]
        tinted = np.clip(tinted, 0.0, 1.0)
        return (tinted * 255.0).astype(np.uint8)

    def prime_state(self, frame_bgr: np.ndarray) -> None:
        if not self.enable_gpu or self._ctx is None:
            return
        blurred = cv2.GaussianBlur(frame_bgr, (0, 0), sigmaX=2.0)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY).astype("f4") / 255.0
        u = np.clip(1.0 - 0.7 * gray, 0.0, 1.0)
        v = np.clip(0.9 * gray, 0.0, 1.0)
        state = np.zeros((self.height, self.width, 4), dtype="f4")
        state[..., 0] = u
        state[..., 1] = v
        state[..., 2] = gray
        state[..., 3] = 1.0
        state = np.flipud(state)
        for tex in self._state_textures:
            tex.write(state.tobytes())

    def release(self) -> None:
        if not self.enable_gpu or self._ctx is None:
            return
        self._ctx.release()
        self._ctx = None

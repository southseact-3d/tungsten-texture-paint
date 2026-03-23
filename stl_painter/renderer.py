from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import moderngl
import numpy as np

from .camera import OrbitCamera
from .mesh_model import MeshModel

logger = logging.getLogger(__name__)

SHADER_DIR = Path(__file__).with_name("shaders")


@dataclass(slots=True)
class RenderSnapshot:
    rgba: np.ndarray
    viewport_size: tuple[int, int]


def make_grid_snapshot(viewport_size: tuple[int, int]) -> RenderSnapshot:
    """Build a non-GL placeholder image so the UI is never all-black."""
    width, height = viewport_size
    rgba = MeshRenderer.create_grid_texture(width, height)
    return RenderSnapshot(rgba=rgba, viewport_size=viewport_size)


class MeshRenderer:
    def __init__(
        self,
        ctx: moderngl.Context,
        mesh_model: MeshModel,
        viewport_size: tuple[int, int],
    ) -> None:
        self.ctx = ctx
        self.mesh_model = mesh_model
        self.viewport_size = viewport_size
        self.program = self.ctx.program(
            vertex_shader=(SHADER_DIR / "mesh.vert").read_text(encoding="utf-8"),
            fragment_shader=(SHADER_DIR / "mesh.frag").read_text(encoding="utf-8"),
        )
        self.pick_program = self.ctx.program(
            vertex_shader=(SHADER_DIR / "pick.vert").read_text(encoding="utf-8"),
            fragment_shader=(SHADER_DIR / "pick.frag").read_text(encoding="utf-8"),
        )
        self._build_buffers()
        self._build_framebuffers()

    @staticmethod
    def create_grid_texture(width: int, height: int) -> np.ndarray:
        """Create a grid texture for display when no model is loaded."""
        # Create a light gray background
        grid = np.ones((height, width, 4), dtype=np.uint8) * 240
        # Draw grid lines every 40 pixels
        for x in range(0, width, 40):
            grid[:, x, :3] = 220  # Light gray lines
        for y in range(0, height, 40):
            grid[y, :, :3] = 220  # Light gray lines
        return grid

    def _build_buffers(self) -> None:
        positions, normals, colours = self.mesh_model.expanded_arrays()
        face_ids = np.repeat(np.arange(self.mesh_model.face_count, dtype=np.int32), 3)
        self.position_vbo = self.ctx.buffer(positions.astype("f4").tobytes())
        self.normal_vbo = self.ctx.buffer(normals.astype("f4").tobytes())
        self.colour_vbo = self.ctx.buffer(colours.astype("f4").tobytes())
        self.face_id_vbo = self.ctx.buffer(face_ids.astype("i4").tobytes())
        self.vao = self.ctx.vertex_array(
            self.program,
            [
                (self.position_vbo, "3f", "in_position"),
                (self.normal_vbo, "3f", "in_normal"),
                (self.colour_vbo, "4f", "in_colour"),
            ],
        )
        self.pick_vao = self.ctx.vertex_array(
            self.pick_program,
            [
                (self.position_vbo, "3f", "in_position"),
                (self.face_id_vbo, "i", "in_face_id"),
            ],
        )

    def _build_framebuffers(self) -> None:
        width, height = self.viewport_size
        colour = self.ctx.texture((width, height), 4, dtype="f1")
        depth = self.ctx.depth_texture((width, height))
        self.framebuffer = self.ctx.framebuffer(
            color_attachments=[colour], depth_attachment=depth
        )
        pick_colour = self.ctx.texture((width, height), 4, dtype="f1")
        pick_depth = self.ctx.depth_texture((width, height))
        self.pick_framebuffer = self.ctx.framebuffer(
            color_attachments=[pick_colour], depth_attachment=pick_depth
        )

    def resize(self, viewport_size: tuple[int, int]) -> None:
        if viewport_size == self.viewport_size:
            return
        self.viewport_size = viewport_size
        self._build_framebuffers()

    def update_face_colour(self, face_id: int) -> None:
        rgba = np.array(self.mesh_model.face_colour(face_id), dtype=np.float32) / 255.0
        colour_data = np.tile(rgba, (3, 1)).astype("f4")
        offset = face_id * 3 * 4 * 4
        self.colour_vbo.write(colour_data.tobytes(), offset=offset)

    def render(
        self,
        camera: OrbitCamera,
        light_dir: tuple[float, float, float] = (0.3, 0.8, 0.4),
    ) -> RenderSnapshot:
        # Use light background color by default
        self.ctx.viewport = (0, 0, self.viewport_size[0], self.viewport_size[1])
        self.framebuffer.use()
        self.framebuffer.clear(0.95, 0.95, 0.95, 1.0, depth=1.0)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.program["mvp"].write(
            camera.mvp_matrix(self.viewport_size).astype("f4").tobytes()
        )
        self.program["light_dir"].value = light_dir
        self.vao.render(moderngl.TRIANGLES)
        data = self.framebuffer.read(components=4, dtype="f1")
        rgba = np.frombuffer(data, dtype=np.uint8).reshape(
            self.viewport_size[1], self.viewport_size[0], 4
        )
        rgba = np.flipud(rgba.copy())
        return RenderSnapshot(rgba=rgba, viewport_size=self.viewport_size)

    def pick_face(self, camera: OrbitCamera, mouse_x: int, mouse_y: int) -> int | None:
        self.ctx.viewport = (0, 0, self.viewport_size[0], self.viewport_size[1])
        self.pick_framebuffer.use()
        self.pick_framebuffer.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.pick_program["mvp"].write(
            camera.mvp_matrix(self.viewport_size).astype("f4").tobytes()
        )
        self.pick_vao.render(moderngl.TRIANGLES)
        read_y = self.viewport_size[1] - mouse_y - 1
        pixel = self.pick_framebuffer.read(
            viewport=(mouse_x, read_y, 1, 1), components=4, dtype="f1"
        )
        r, g, b, _ = pixel
        face_id = (r << 16) | (g << 8) | b
        if face_id >= self.mesh_model.face_count:
            return None
        return face_id

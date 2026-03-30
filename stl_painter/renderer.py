from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import moderngl
import numpy as np
from PIL import Image, ImageDraw

from .camera import OrbitCamera
from .mesh_model import MeshModel

logger = logging.getLogger(__name__)

VIEWPORT_BG = np.array((237, 240, 245, 255), dtype=np.uint8)
SOFTWARE_GL_MARKERS = (
    "software",
    "swiftshader",
    "llvmpipe",
    "softpipe",
    "gdi generic",
    "microsoft basic render",
    "microsoft basic render driver",
    "mesa",
)


@dataclass(slots=True)
class RenderSnapshot:
    rgba: np.ndarray
    viewport_size: tuple[int, int]


def make_grid_snapshot(viewport_size: tuple[int, int]) -> RenderSnapshot:
    width, height = viewport_size
    rgba = MeshRenderer.create_grid_texture(width, height)
    return RenderSnapshot(rgba=rgba, viewport_size=viewport_size)


def probe_gpu_support() -> tuple[bool, str]:
    try:
        ctx = moderngl.create_standalone_context()
        info = getattr(ctx, "info", {}) or {}
        vendor = str(info.get("GL_VENDOR", "unknown")).strip()
        renderer = str(info.get("GL_RENDERER", "unknown")).strip()
        combined = f"{vendor} | {renderer}".strip()
        lowered = combined.lower()
        is_software = any(marker in lowered for marker in SOFTWARE_GL_MARKERS)
        return (not is_software), combined
    except Exception as exc:
        return False, f"unavailable ({exc})"


class MeshRenderer:
    def __init__(
        self,
        ctx: object | None,
        mesh_model: MeshModel,
        viewport_size: tuple[int, int],
        prefer_gpu: bool | None = None,
    ) -> None:
        self.ctx = ctx
        self.mesh_model = mesh_model
        self.viewport_size = viewport_size
        self.prefer_gpu = self._resolve_gpu_preference(prefer_gpu)
        self._last_render_key: tuple[tuple[int, int], bytes] | None = None
        self._last_pick_image: np.ndarray | None = None
        self._gpu_ready = False
        self._mesh_program: moderngl.Program | None = None
        self._edge_program: moderngl.Program | None = None
        self._pick_program: moderngl.Program | None = None
        self._mesh_vbo: moderngl.Buffer | None = None
        self._edge_vbo: moderngl.Buffer | None = None
        self._position_vbo: moderngl.Buffer | None = None
        self._face_id_vbo: moderngl.Buffer | None = None
        self._mesh_vao: moderngl.VertexArray | None = None
        self._edge_vao: moderngl.VertexArray | None = None
        self._pick_vao: moderngl.VertexArray | None = None
        self._colour_fbo: moderngl.Framebuffer | None = None
        self._pick_fbo: moderngl.Framebuffer | None = None
        self._colour_tex: moderngl.Texture | None = None
        self._pick_tex: moderngl.Texture | None = None
        self._colour_vertices: np.ndarray | None = None
        self._face_centers = (
            self.mesh_model.vertices[self.mesh_model.faces]
            .mean(axis=1)
            .astype(np.float32)
        )
        scale_env = os.environ.get("STL_TEXTURE_PAINTER_SOFTWARE_SCALE", "").strip()
        if scale_env:
            self._software_scale = float(scale_env)
        elif self.mesh_model.face_count > 50000:
            self._software_scale = 0.85
        elif self.mesh_model.face_count > 25000:
            self._software_scale = 0.95
        else:
            self._software_scale = 1.0
        logger.info(
            "Initializing renderer | viewport=%s | faces=%s | vertices=%s | prefer_gpu=%s | software_scale=%.2f",
            viewport_size,
            mesh_model.face_count,
            mesh_model.vertex_count,
            self.prefer_gpu,
            self._software_scale,
        )
        if self.prefer_gpu:
            self._init_gpu_renderer()
        else:
            logger.info("Renderer starting in software mode")

    def _resolve_gpu_preference(self, prefer_gpu: bool | None) -> bool:
        if prefer_gpu is not None:
            return prefer_gpu
        force_software = (
            os.environ.get("STL_TEXTURE_PAINTER_FORCE_SOFTWARE", "").strip().lower()
        )
        if force_software in {"1", "true", "yes", "on"}:
            return False
        env_value = os.environ.get("STL_TEXTURE_PAINTER_ENABLE_GPU", "").strip().lower()
        if env_value in {"1", "true", "yes", "on"}:
            return True
        if env_value in {"0", "false", "no", "off"}:
            return False
        return True

    @staticmethod
    def create_grid_texture(width: int, height: int) -> np.ndarray:
        grid = np.ones((height, width, 4), dtype=np.uint8) * 242
        for x in range(0, width, 40):
            grid[:, x, :3] = 214
        for y in range(0, height, 40):
            grid[y, :, :3] = 214
        return grid

    def _shader_source(self, name: str) -> str:
        return (Path(__file__).resolve().parent / "shaders" / name).read_text(
            encoding="utf-8"
        )

    def _init_gpu_renderer(self) -> None:
        try:
            self.ctx = self.ctx or moderngl.create_standalone_context()
            assert isinstance(self.ctx, moderngl.Context)
            if self._context_is_software(self.ctx):
                logger.warning(
                    "OpenGL context is software-rendered; using software renderer instead"
                )
                self._gpu_ready = False
                return
            self._mesh_program = self.ctx.program(
                vertex_shader=self._shader_source("mesh.vert"),
                fragment_shader=self._shader_source("mesh.frag"),
            )
            self._edge_program = self.ctx.program(
                vertex_shader=self._shader_source("edge.vert"),
                fragment_shader=self._shader_source("edge.frag"),
            )
            self._pick_program = self.ctx.program(
                vertex_shader=self._shader_source("pick.vert"),
                fragment_shader=self._shader_source("pick.frag"),
            )

            positions = (
                self.mesh_model.vertices[self.mesh_model.faces]
                .reshape(-1, 3)
                .astype("f4")
            )
            face_vertices = self.mesh_model.vertices[self.mesh_model.faces].astype("f4")
            normals = np.repeat(self.mesh_model.normals, 3, axis=0).astype("f4")
            colours = np.repeat(
                np.array(
                    [
                        self.mesh_model.face_colour(face_id)
                        for face_id in range(self.mesh_model.face_count)
                    ],
                    dtype=np.float32,
                )
                / 255.0,
                3,
                axis=0,
            ).astype("f4")
            face_ids = np.repeat(
                np.arange(self.mesh_model.face_count, dtype=np.int32), 3
            )

            self._colour_vertices = np.concatenate(
                [positions, normals, colours], axis=1
            ).astype("f4")
            self._mesh_vbo = self.ctx.buffer(self._colour_vertices.tobytes())
            edge_positions = np.stack(
                [
                    face_vertices[:, 0],
                    face_vertices[:, 1],
                    face_vertices[:, 1],
                    face_vertices[:, 2],
                    face_vertices[:, 2],
                    face_vertices[:, 0],
                ],
                axis=1,
            ).reshape(-1, 3)
            self._edge_vbo = self.ctx.buffer(edge_positions.astype("f4").tobytes())
            self._position_vbo = self.ctx.buffer(positions.tobytes())
            self._face_id_vbo = self.ctx.buffer(face_ids.tobytes())
            self._mesh_vao = self.ctx.vertex_array(
                self._mesh_program,
                [
                    (
                        self._mesh_vbo,
                        "3f 3f 4f",
                        "in_position",
                        "in_normal",
                        "in_colour",
                    )
                ],
            )
            self._edge_vao = self.ctx.vertex_array(
                self._edge_program,
                [(self._edge_vbo, "3f", "in_position")],
            )
            self._pick_vao = self.ctx.vertex_array(
                self._pick_program,
                [
                    (self._position_vbo, "3f", "in_position"),
                    (self._face_id_vbo, "1i", "in_face_id"),
                ],
            )
            self._create_framebuffers()
            self._gpu_ready = True
            logger.info("GPU renderer ready | version=%s", self.ctx.version_code)
        except Exception:
            self._gpu_ready = False
            logger.exception(
                "Falling back to software renderer because GPU setup failed"
            )

    def _context_is_software(self, ctx: moderngl.Context) -> bool:
        info = getattr(ctx, "info", {}) or {}
        vendor = str(info.get("GL_VENDOR", "")).lower()
        renderer = str(info.get("GL_RENDERER", "")).lower()
        combined = f"{vendor} {renderer}"
        return any(marker in combined for marker in SOFTWARE_GL_MARKERS)

    def _create_framebuffers(self) -> None:
        if not self._gpu_ready and not isinstance(self.ctx, moderngl.Context):
            return
        assert isinstance(self.ctx, moderngl.Context)
        width, height = self.viewport_size
        self._colour_tex = self.ctx.texture((width, height), 4, dtype="f1")
        self._colour_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        colour_depth = self.ctx.depth_renderbuffer((width, height))
        self._colour_fbo = self.ctx.framebuffer(
            color_attachments=[self._colour_tex], depth_attachment=colour_depth
        )
        self._pick_tex = self.ctx.texture((width, height), 3, dtype="f1")
        self._pick_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        pick_depth = self.ctx.depth_renderbuffer((width, height))
        self._pick_fbo = self.ctx.framebuffer(
            color_attachments=[self._pick_tex], depth_attachment=pick_depth
        )

    def rebuild_edge_buffer(self) -> None:
        if not self._gpu_ready or self.ctx is None:
            return
        internal_edges = self.mesh_model.get_internal_group_edges()
        edge_positions = []

        mesh = self.mesh_model.mesh()
        if mesh and mesh.face_adjacency is not None:
            vertex_pairs = [mesh.edges[i] for i in mesh.face_adjacency_edges]
            face_pairs = mesh.face_adjacency

            for (face1, face2), (v1, v2) in zip(face_pairs, vertex_pairs):
                edge = (min(face1, face2), max(face1, face2))
                if edge in internal_edges:
                    continue
                edge_positions.append(self.mesh_model.vertices[v1])
                edge_positions.append(self.mesh_model.vertices[v2])

        if edge_positions:
            edge_positions_arr = np.array(edge_positions, dtype=np.float32)
            self._edge_vbo = self.ctx.buffer(edge_positions_arr.tobytes())
            self._edge_vao = self.ctx.vertex_array(
                self._edge_program,
                [(self._edge_vbo, "3f", "in_position")],
            )
        else:
            self._edge_vbo = None
            self._edge_vao = None

    def _find_edge_neighbor(self, face_id: int, v0: int, v1: int) -> int | None:
        face_vertices = self.mesh_model.faces
        for other_face in range(self.mesh_model.face_count):
            if other_face == face_id:
                continue
            ov = face_vertices[other_face]
            if (ov[0] == v0 or ov[1] == v0 or ov[2] == v0) and (
                ov[0] == v1 or ov[1] == v1 or ov[2] == v1
            ):
                return other_face
        return None

    def resize(self, viewport_size: tuple[int, int]) -> None:
        if viewport_size != self.viewport_size:
            self.viewport_size = viewport_size
            self._last_render_key = None
            self._last_pick_image = None
            if self._gpu_ready:
                self._create_framebuffers()

    def update_face_colour(self, face_id: int) -> None:
        self._last_render_key = None
        self._last_pick_image = None
        if (
            not self._gpu_ready
            or self._colour_vertices is None
            or self._mesh_vbo is None
        ):
            return
        base_colour = (
            np.array(self.mesh_model.face_colour(face_id), dtype=np.float32) / 255.0
        )
        start = face_id * 3
        end = start + 3
        self._colour_vertices[start:end, 6:10] = base_colour
        self._mesh_vbo.write(self._colour_vertices.tobytes())

    def update_face_colours(
        self, face_ids: list[int] | tuple[int, ...] | set[int]
    ) -> None:
        if not face_ids:
            return
        self._last_render_key = None
        self._last_pick_image = None
        if (
            not self._gpu_ready
            or self._colour_vertices is None
            or self._mesh_vbo is None
        ):
            return
        for face_id in face_ids:
            base_colour = (
                np.array(self.mesh_model.face_colour(face_id), dtype=np.float32) / 255.0
            )
            start = face_id * 3
            end = start + 3
            self._colour_vertices[start:end, 6:10] = base_colour
        self._mesh_vbo.write(self._colour_vertices.tobytes())

    def _camera_key(self, camera: OrbitCamera) -> tuple[tuple[int, int], bytes]:
        return self.viewport_size, camera.mvp_matrix(self.viewport_size).astype(
            "f4"
        ).tobytes()

    def _mvp_bytes(self, camera: OrbitCamera) -> bytes:
        return camera.mvp_matrix(self.viewport_size).astype("f4").T.tobytes()

    def render(
        self,
        camera: OrbitCamera,
        light_dir: tuple[float, float, float] = (0.3, 0.8, 0.4),
        show_triangle_edges: bool = False,
    ) -> RenderSnapshot:
        if self._gpu_ready:
            return self._render_gpu(camera, light_dir, show_triangle_edges)
        return self._render_software(camera, light_dir, show_triangle_edges)

    def _render_gpu(
        self,
        camera: OrbitCamera,
        light_dir: tuple[float, float, float],
        show_triangle_edges: bool,
    ) -> RenderSnapshot:
        assert self._gpu_ready
        assert isinstance(self.ctx, moderngl.Context)
        assert self._colour_fbo is not None
        assert self._colour_tex is not None
        assert self._mesh_program is not None
        assert self._mesh_vao is not None
        self._colour_fbo.use()
        bg = VIEWPORT_BG.astype(np.float32) / 255.0
        self._colour_fbo.clear(
            float(bg[0]), float(bg[1]), float(bg[2]), float(bg[3]), depth=1.0
        )
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self._mesh_program["mvp"].write(self._mvp_bytes(camera))
        self._mesh_program["light_dir"].value = tuple(float(v) for v in light_dir)
        self._mesh_vao.render(mode=moderngl.TRIANGLES)
        if show_triangle_edges:
            assert self._edge_program is not None
            assert self._edge_vao is not None
            self.ctx.disable(moderngl.CULL_FACE)
            self._edge_program["mvp"].write(self._mvp_bytes(camera))
            self._edge_program["depth_bias"].value = 0.0006
            self._edge_program["line_colour"].value = (0.0, 0.0, 0.0, 1.0)
            self._edge_vao.render(mode=moderngl.LINES)
        rgba = np.frombuffer(
            self._colour_fbo.read(components=4, alignment=1), dtype=np.uint8
        ).reshape((self.viewport_size[1], self.viewport_size[0], 4))
        rgba = np.flipud(rgba).copy()
        self._last_render_key = self._camera_key(camera)
        self._last_pick_image = None
        return RenderSnapshot(rgba=rgba, viewport_size=self.viewport_size)

    def _render_pick_gpu(self, camera: OrbitCamera) -> None:
        assert self._gpu_ready
        assert isinstance(self.ctx, moderngl.Context)
        assert self._pick_fbo is not None
        assert self._pick_program is not None
        assert self._pick_vao is not None
        self._pick_fbo.use()
        self._pick_fbo.clear(0.0, 0.0, 0.0, 1.0, depth=1.0)
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self._pick_program["mvp"].write(self._mvp_bytes(camera))
        self._pick_vao.render(mode=moderngl.TRIANGLES)

    def _build_pick_image(self, camera: OrbitCamera) -> np.ndarray:
        if self._gpu_ready:
            self._render_pick_gpu(camera)
            assert self._pick_fbo is not None
            pick_image = np.frombuffer(
                self._pick_fbo.read(components=3, alignment=1), dtype=np.uint8
            ).reshape((self.viewport_size[1], self.viewport_size[0], 3))
            self._last_pick_image = np.flipud(pick_image).copy()
            return self._last_pick_image
        return self._build_pick_image_software(camera)

    def pick_face(self, camera: OrbitCamera, mouse_x: int, mouse_y: int) -> int | None:
        if mouse_x < 0 or mouse_y < 0:
            return None
        width, height = self.viewport_size
        if mouse_x >= width or mouse_y >= height:
            return None
        if self._gpu_ready:
            self._render_pick_gpu(camera)
            assert self._pick_fbo is not None
            read_y = height - 1 - mouse_y
            pixel = self._pick_fbo.read(
                viewport=(mouse_x, read_y, 1, 1), components=3, alignment=1
            )
            r, g, b = pixel[0], pixel[1], pixel[2]
            face_id = (int(r) << 16) | (int(g) << 8) | int(b)
            if face_id >= self.mesh_model.face_count:
                return None
            return face_id
        current_key = self._camera_key(camera)
        if self._last_pick_image is None or self._last_render_key != current_key:
            self._build_pick_image(camera)
            self._last_render_key = current_key
        assert self._last_pick_image is not None
        r, g, b = self._last_pick_image[mouse_y, mouse_x]
        face_id = (int(r) << 16) | (int(g) << 8) | int(b)
        if face_id == 0 and self.mesh_model.face_count > 0:
            sample = self._last_pick_image[mouse_y, mouse_x]
            if np.all(sample == 0):
                return (
                    0 if self._point_hits_face_zero(camera, mouse_x, mouse_y) else None
                )
        if face_id >= self.mesh_model.face_count:
            return None
        return face_id

    def _render_size(self) -> tuple[int, int]:
        width, height = self.viewport_size
        scale = min(max(self._software_scale, 0.35), 1.0)
        if scale >= 0.999:
            return width, height
        return max(320, int(width * scale)), max(240, int(height * scale))

    def _project_faces(
        self,
        camera: OrbitCamera,
        screen_size: tuple[int, int],
        *,
        min_area: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        vertices = self.mesh_model.vertices
        faces = self.mesh_model.faces
        width, height = screen_size
        mvp = camera.mvp_matrix(self.viewport_size)
        clip = (mvp @ np.c_[vertices, np.ones(len(vertices), dtype=np.float32)].T).T
        w = clip[:, 3:4]
        safe_w = np.where(np.abs(w) < 1e-6, 1e-6, w)
        ndc = clip[:, :3] / safe_w

        valid_vertices = np.isfinite(ndc).all(axis=1)
        face_vertices = ndc[faces]
        valid_faces = valid_vertices[faces].all(axis=1)
        valid_faces &= (np.abs(face_vertices[:, :, 2]) <= 1.5).any(axis=1)

        to_camera = camera.position().astype(np.float32) - self._face_centers
        facing = np.einsum("ij,ij->i", self.mesh_model.normals, to_camera) > 0.0
        valid_faces &= facing

        screen = np.empty((len(vertices), 2), dtype=np.float32)
        screen[:, 0] = (ndc[:, 0] * 0.5 + 0.5) * width
        screen[:, 1] = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * height

        projected = screen[faces]
        mins = projected.min(axis=1)
        maxs = projected.max(axis=1)
        valid_faces &= maxs[:, 0] >= 0.0
        valid_faces &= mins[:, 0] < float(width)
        valid_faces &= maxs[:, 1] >= 0.0
        valid_faces &= mins[:, 1] < float(height)
        if min_area > 0.0:
            edge_a = projected[:, 1] - projected[:, 0]
            edge_b = projected[:, 2] - projected[:, 0]
            double_area = np.abs(
                edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0]
            )
            valid_faces &= double_area >= (min_area * 2.0)
        depths = face_vertices[:, :, 2].mean(axis=1)
        return projected, depths, valid_faces

    def _sorted_face_indices(
        self,
        camera: OrbitCamera,
        screen_size: tuple[int, int],
        *,
        min_area: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        projected, depths, valid_faces = self._project_faces(
            camera, screen_size, min_area=min_area
        )
        visible = np.where(valid_faces)[0]
        if len(visible) == 0:
            return projected, visible
        order = visible[np.argsort(depths[visible])[::-1]]
        return projected, order

    def _render_software(
        self,
        camera: OrbitCamera,
        light_dir: tuple[float, float, float] = (0.3, 0.8, 0.4),
        show_triangle_edges: bool = False,
    ) -> RenderSnapshot:
        render_size = self._render_size()
        projected, order = self._sorted_face_indices(camera, render_size, min_area=1.25)
        width, height = render_size
        image = Image.new("RGBA", (width, height), tuple(int(v) for v in VIEWPORT_BG))
        draw = ImageDraw.Draw(image, "RGBA")

        view_light = camera.position() - camera.target
        light = (
            np.asarray(light_dir, dtype=np.float32)
            + view_light.astype(np.float32) * 0.12
        )
        light_norm = max(float(np.linalg.norm(light)), 1e-6)
        light = light / light_norm

        for face_id in order:
            points = [tuple(point) for point in projected[face_id]]
            normal = self.mesh_model.normals[face_id]
            normal_norm = max(float(np.linalg.norm(normal)), 1e-6)
            normal = normal / normal_norm
            diffuse = max(float(np.dot(normal, light)), 0.0)
            diffuse = 0.18 + diffuse * 0.82
            base = np.asarray(
                self.mesh_model.face_colour(int(face_id)), dtype=np.float32
            )
            lit = np.clip(base[:3] * diffuse + 26.0, 0.0, 255.0).astype(np.uint8)
            draw.polygon(
                points, fill=(int(lit[0]), int(lit[1]), int(lit[2]), int(base[3]))
            )
            if show_triangle_edges:
                draw.line(
                    points + [points[0]],
                    fill=(0, 0, 0, 255),
                    width=1,
                )

        if render_size != self.viewport_size:
            image = image.resize(self.viewport_size, Image.Resampling.BILINEAR)
        rgba = np.asarray(image, dtype=np.uint8)
        self._last_render_key = self._camera_key(camera)
        self._last_pick_image = None
        logger.debug(
            "Software render complete | visible_faces=%s | viewport=%s",
            len(order),
            self.viewport_size,
        )
        return RenderSnapshot(rgba=rgba, viewport_size=self.viewport_size)

    def _build_pick_image_software(self, camera: OrbitCamera) -> np.ndarray:
        projected, order = self._sorted_face_indices(camera, self.viewport_size)
        width, height = self.viewport_size
        image = Image.new("RGB", (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(image, "RGB")
        for face_id in order:
            encoded = (
                (int(face_id) >> 16) & 255,
                (int(face_id) >> 8) & 255,
                int(face_id) & 255,
            )
            points = [tuple(point) for point in projected[face_id]]
            draw.polygon(points, fill=encoded)
        pick_image = np.asarray(image, dtype=np.uint8)
        self._last_pick_image = pick_image
        return pick_image

    def _point_hits_face_zero(
        self, camera: OrbitCamera, mouse_x: int, mouse_y: int
    ) -> bool:
        projected, order = self._sorted_face_indices(camera, self.viewport_size)
        if len(order) == 0 or int(order[-1]) != 0:
            return False
        triangle = projected[0]
        point = np.array([mouse_x, mouse_y], dtype=np.float32)
        a, b, c = triangle
        v0 = c - a
        v1 = b - a
        v2 = point - a
        denom = v0[0] * v1[1] - v1[0] * v0[1]
        if abs(float(denom)) < 1e-6:
            return False
        inv = 1.0 / denom
        u = (v2[0] * v1[1] - v1[0] * v2[1]) * inv
        v = (v0[0] * v2[1] - v2[0] * v0[1]) * inv
        return u >= 0.0 and v >= 0.0 and (u + v) <= 1.0

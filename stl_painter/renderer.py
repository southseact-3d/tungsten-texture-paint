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
        self._front_edge_vao: moderngl.VertexArray | None = None
        self._front_edge_key: bytes | None = None
        self._front_cad_edge_vao: moderngl.VertexArray | None = None
        self._front_cad_edge_key: bytes | None = None
        self._edge_incidents_cache: dict[tuple[int, int], list[int]] | None = None
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
            edge_positions = self._edge_positions()
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

    def _full_triangle_edge_positions(self) -> np.ndarray:
        """All triangle edges as line-list positions (2 verts per edge).

        Source data for paint-mode outlines. The GPU path filters these to
        front-facing faces per camera (see ``_front_edge_vao_for_camera``)
        so only visible triangles are outlined — never an X-ray wireframe.
        This is topology-only and never depends on face groups.
        """
        face_vertices = self.mesh_model.vertices[self.mesh_model.faces].astype("f4")
        return np.stack(
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

    def _cad_boundary_edge_positions(self) -> np.ndarray:
        """CAD-face outline edges as line-list positions (2 verts per edge).

        STEP models hide triangulation: only edges between different CAD
        faces (plus naked borders) are drawn, so rectangles render as
        rectangles and circles as circles.
        """
        edges = self.mesh_model.cad_boundary_edges()
        if len(edges) == 0:
            return np.zeros((0, 3), dtype=np.float32)
        return self.mesh_model.vertices[np.asarray(edges)].reshape(-1, 3).astype(
            np.float32
        )

    def _occlusion_size(self) -> tuple[int, int]:
        """Low-res size for the CPU pick image used in edge occlusion tests.

        Visibility is coarse, so the longest side is capped at 480px to
        keep per-camera-move cost small on large viewports. World-space
        edge endpoints are projected at this same size before sampling
        the pick image (mirrors ``_render_size`` capping in software).
        """
        width, height = self.viewport_size
        longest = max(int(width), int(height), 1)
        scale = min(1.0, 480.0 / float(longest))
        return (
            max(160, int(int(width) * scale)),
            max(120, int(int(height) * scale)),
        )

    def _edge_incidents(self) -> dict[tuple[int, int], list[int]]:
        """Cached map of undirected mesh edge -> incident triangle ids.

        Topology never changes after import (uniform scaling keeps
        indices), so this is built once and shared by the GPU visible-edge
        filter and the software CAD-edge pass instead of rebuilding a
        dict per frame in three places.
        """
        if self._edge_incidents_cache is not None:
            return self._edge_incidents_cache
        faces = self.mesh_model.faces
        edge_to_faces: dict[tuple[int, int], list[int]] = {}
        for face_id in range(len(faces)):
            a, b, c = (
                int(faces[face_id, 0]),
                int(faces[face_id, 1]),
                int(faces[face_id, 2]),
            )
            for u, v in ((a, b), (b, c), (c, a)):
                key = (u, v) if u < v else (v, u)
                entry = edge_to_faces.get(key)
                if entry is None:
                    edge_to_faces[key] = [face_id]
                else:
                    entry.append(face_id)
        self._edge_incidents_cache = edge_to_faces
        return edge_to_faces

    def _visible_cad_edges(self, camera: OrbitCamera) -> np.ndarray:
        """CAD boundary edges that are truly visible (not just front-facing).

        A front-facing edge can still sit behind another part of a concave
        STEP model; drawing it on top looks like X-ray wireframe. Each
        front-facing candidate is occlusion-tested against a CPU pick image
        via ``_cad_edge_visible_in_pick`` (the same test the software
        renderer uses), so only edges whose samples resolve to their own
        CAD face are kept. Pure CPU/PIL — safe on the Dear PyGui thread.
        """
        if not self.mesh_model.has_cad_faces:
            return np.zeros((0, 2), dtype=np.int32)
        edges = self.mesh_model.cad_boundary_edges()
        if len(edges) == 0:
            return np.zeros((0, 2), dtype=np.int32)
        mask = self._front_facing_mask(camera)
        edge_to_faces = self._edge_incidents()
        edge_list = np.asarray(edges, dtype=np.int32)
        front = np.array(
            [
                bool(mask[edge_to_faces.get((int(u), int(v)) if int(u) < int(v) else (int(v), int(u)), [])].any())
                for u, v in edge_list.tolist()
            ],
            dtype=bool,
        )
        if not front.any():
            return np.zeros((0, 2), dtype=np.int32)
        occ_size = self._occlusion_size()
        screen, valid = self._project_vertices(camera, occ_size)
        pick_image = self._build_pick_image_software(camera, occ_size)
        # Fully vectorized window test across all front-facing candidates:
        # 5 sample points x 3x3 windows per edge resolve with a handful of
        # numpy ops (a Python loop here costs ~3s on a 5k-face STEP model).
        # Integer ``tri_to_cad`` indices stand in for ``cad_id_for_face``
        # (which sorts the whole ``cad_faces`` dict per call): neighbours
        # on one CAD plane share the same index, matching the acceptance
        # rule of ``_cad_edge_visible_in_pick`` used by the software path.
        tri_cad = self.mesh_model.tri_to_cad
        tri_cad_ints = (
            np.asarray(tri_cad, dtype=np.int64).reshape(-1)
            if tri_cad is not None
            else None
        )
        face_count = int(self.mesh_model.face_count)
        pick_h, pick_w = int(pick_image.shape[0]), int(pick_image.shape[1])
        cand_idx = np.flatnonzero(front)
        if cand_idx.size == 0:
            return np.zeros((0, 2), dtype=np.int32)
        cand = edge_list[cand_idx]
        cand_valid = valid[cand[:, 0]] & valid[cand[:, 1]]
        # Incident faces per candidate, padded with -1 (STEP boundary
        # edges almost always have exactly one incident face).
        members_per_edge = [
            edge_to_faces.get(
                (int(u), int(v)) if int(u) < int(v) else (int(v), int(u)), []
            )
            for u, v in cand.tolist()
        ]
        max_members = max(1, max(len(m) for m in members_per_edge))
        member_mat = np.full((len(cand), max_members), -1, dtype=np.int64)
        for i, members in enumerate(members_per_edge):
            for j, fid in enumerate(members[:max_members]):
                member_mat[i, j] = int(fid)
        has_members = (member_mat >= 0).any(axis=1)
        wanted_mat = np.full((len(cand), max_members), -1, dtype=np.int64)
        if tri_cad_ints is not None:
            clipped = np.clip(member_mat, 0, len(tri_cad_ints) - 1)
            wanted_mat = np.where(
                member_mat >= 0, tri_cad_ints[clipped], -1
            )
        pick_ids = (
            pick_image[:, :, 0].astype(np.int32) << 16
            | pick_image[:, :, 1].astype(np.int32) << 8
            | pick_image[:, :, 2].astype(np.int32)
        )
        fractions = np.asarray((0.1, 0.3, 0.5, 0.7, 0.9), dtype=np.float32)
        p0 = screen[cand[:, 0]].astype(np.float32)
        delta = screen[cand[:, 1]].astype(np.float32) - p0
        samples = (
            p0[:, None, :] + fractions[None, :, None] * delta[:, None, :]
        )
        cxs = np.round(samples[:, :, 0]).astype(np.int32)
        cys = np.round(samples[:, :, 1]).astype(np.int32)
        offsets = np.asarray(
            [(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)],
            dtype=np.int32,
        )
        win_x = cxs[:, :, None] + offsets[None, None, :, 0]
        win_y = cys[:, :, None] + offsets[None, None, :, 1]
        win_ok = (
            (win_x >= 0) & (win_x < pick_w) & (win_y >= 0) & (win_y < pick_h)
        )
        safe_x = np.clip(win_x, 0, pick_w - 1)
        safe_y = np.clip(win_y, 0, pick_h - 1)
        shown = np.where(win_ok, pick_ids[safe_y, safe_x], -1)
        member_hit = (
            (shown[..., None] == member_mat[:, None, None, :])
            & win_ok[..., None]
        ).any(axis=(1, 2, 3))
        in_mesh = (shown >= 0) & (shown < face_count) & win_ok
        if tri_cad_ints is not None and wanted_mat.shape[1] > 0:
            safe_shown = np.clip(shown, 0, len(tri_cad_ints) - 1)
            shown_cad = np.where(in_mesh, tri_cad_ints[safe_shown], -1)
            cad_hit = (
                (shown_cad[..., None] == wanted_mat[:, None, None, :])
                & in_mesh[..., None]
            ).any(axis=(1, 2, 3))
        else:
            cad_hit = np.zeros(len(cand), dtype=bool)
        edge_visible = (
            np.asarray(member_hit).reshape(-1)
            | np.asarray(cad_hit).reshape(-1)
        ) & np.asarray(cand_valid).reshape(-1) & np.asarray(has_members).reshape(-1)
        kept = cand[np.flatnonzero(edge_visible)]
        if len(kept) == 0:
            return np.zeros((0, 2), dtype=np.int32)
        return np.asarray(kept, dtype=np.int32)

    def _front_cad_edge_positions(self, camera: OrbitCamera) -> np.ndarray:
        """CAD boundary edges filtered to truly visible edges only.

        Front-facing filter first (cheap), then a true occlusion test per
        edge against a pick image: on concave STEP models a front-facing
        edge can sit behind another part of the mesh, and drawing it on top
        looks like X-ray wireframe. Same test the software renderer uses.
        """
        if not self.mesh_model.has_cad_faces:
            return np.zeros((0, 3), dtype=np.float32)
        visible = self._visible_cad_edges(camera)
        if len(visible) == 0:
            return np.zeros((0, 3), dtype=np.float32)
        return self.mesh_model.vertices[visible].reshape(-1, 3).astype(np.float32)

    def _front_cad_edge_vao_for_camera(
        self, camera: OrbitCamera
    ) -> moderngl.VertexArray | None:
        """Cached VAO of truly visible CAD-boundary edges only.

        Rebuilt only when the camera position or viewport changes, so
        per-frame cost in paint mode is one vectorized facing test plus a
        low-res pick render on cache miss, then a cache hit. Returns None
        when no CAD edges are visible.
        """
        assert isinstance(self.ctx, moderngl.Context)
        assert self._edge_program is not None
        key = (
            camera.position().astype("f4").tobytes()
            + bytes(bytearray(f"{self.viewport_size}".encode()))
        )
        if self._front_cad_edge_key != key or self._front_cad_edge_vao is None:
            positions = self._front_cad_edge_positions(camera)
            if len(positions) == 0:
                self._front_cad_edge_key = key
                self._front_cad_edge_vao = None
                return None
            vbo = self.ctx.buffer(positions.tobytes())
            self._front_cad_edge_vao = self.ctx.vertex_array(
                self._edge_program,
                [(vbo, "3f", "in_position")],
            )
            self._front_cad_edge_key = key
        return self._front_cad_edge_vao

    def _edge_positions(self) -> np.ndarray:
        if self.mesh_model.has_cad_faces:
            return self._cad_boundary_edge_positions()
        return self._full_triangle_edge_positions()

    def _front_facing_mask(self, camera: OrbitCamera) -> np.ndarray:
        """Front-facing face mask for the current camera position.

        Vectorized ``dot(normal, to_camera) > 0`` test. Used to restrict
        paint-mode triangle outlines to visible faces so back-facing edges
        never shine through the model (X-ray wireframe). This is the same
        facing test the software renderer uses in ``_project_faces``.
        """
        to_camera = camera.position().astype(np.float32) - self._face_centers
        return (
            np.einsum(
                "ij,ij->i", self.mesh_model.normals.astype(np.float32), to_camera
            )
            > 0.0
        )

    def _front_edge_vao_for_camera(
        self, camera: OrbitCamera
    ) -> moderngl.VertexArray | None:
        """Cached VAO of front-facing triangle edges only.

        Rebuilt only when the camera position changes, so per-frame cost in
        paint mode is a single vectorized facing test plus a cache hit.
        Returns None when no faces front the camera.
        """
        assert isinstance(self.ctx, moderngl.Context)
        assert self._edge_program is not None
        key = camera.position().astype("f4").tobytes()
        if self._front_edge_key != key or self._front_edge_vao is None:
            mask = self._front_facing_mask(camera)
            if not bool(mask.any()):
                self._front_edge_key = key
                self._front_edge_vao = None
                return None
            face_vertices = self.mesh_model.vertices[self.mesh_model.faces].astype(
                "f4"
            )[mask]
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
            vbo = self.ctx.buffer(edge_positions.astype("f4").tobytes())
            self._front_edge_vao = self.ctx.vertex_array(
                self._edge_program,
                [(vbo, "3f", "in_position")],
            )
            self._front_edge_key = key
        return self._front_edge_vao

    def rebuild_edge_buffer(self) -> None:
        if not self._gpu_ready or self.ctx is None:
            logger.debug("rebuild_edge_buffer: GPU not ready or no context")
            return
        self._front_edge_key = None
        self._front_edge_vao = None
        self._front_cad_edge_key = None
        self._front_cad_edge_vao = None
        edge_positions_arr = self._edge_positions().astype("f4")
        self._edge_vbo = self.ctx.buffer(edge_positions_arr.tobytes())
        self._edge_vao = self.ctx.vertex_array(
            self._edge_program,
            [(self._edge_vbo, "3f", "in_position")],
        )
        logger.debug(
            "rebuild_edge_buffer: created %d %s edges for %d faces",
            len(edge_positions_arr) // 2,
            "CAD boundary" if self.mesh_model.has_cad_faces else "triangle",
            self.mesh_model.face_count,
        )

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
            # Visibility decisions now depend on the viewport (occlusion
            # pick size), so drop cached edge VAOs on resize.
            self._front_edge_key = None
            self._front_edge_vao = None
            self._front_cad_edge_key = None
            self._front_cad_edge_vao = None
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
        # STEP tessellation is per-CAD-face with duplicated vertices and
        # untrusted winding, so backface culling can carve holes in the
        # outer shell (see-through mesh + leaked edges). Draw both faces
        # for CAD models; depth testing still resolves nearest-wins.
        if self.mesh_model.has_cad_faces:
            self.ctx.enable(moderngl.DEPTH_TEST)
            self.ctx.disable(moderngl.CULL_FACE)
        else:
            self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self._mesh_program["mvp"].write(self._mvp_bytes(camera))
        self._mesh_program["light_dir"].value = tuple(float(v) for v in light_dir)
        self._mesh_vao.render(mode=moderngl.TRIANGLES)
        if show_triangle_edges:
            assert self._edge_program is not None
            self._edge_program["mvp"].write(self._mvp_bytes(camera))
            self._edge_program["line_colour"].value = (0.0, 0.0, 0.0, 1.0)
            if self.mesh_model.has_cad_faces:
                # STEP models: CAD-boundary outlines on truly visible
                # (occlusion-tested) edges only, so occluded interior edges
                # cannot shine through the mesh like X-ray wireframe.
                front_cad_vao = self._front_cad_edge_vao_for_camera(camera)
                if front_cad_vao is not None:
                    self.ctx.enable(moderngl.DEPTH_TEST)
                    self.ctx.disable(moderngl.CULL_FACE)
                    self._edge_program["depth_bias"].value = 0.0006
                    front_cad_vao.render(mode=moderngl.LINES)
            else:
                # Triangle meshes: outline front-facing triangles only so
                # back edges never shine through (X-ray wireframe). Depth
                # testing stays on; the small bias lifts front edges just
                # off the surface to avoid z-fighting.
                front_vao = self._front_edge_vao_for_camera(camera)
                if front_vao is not None:
                    self.ctx.enable(moderngl.DEPTH_TEST)
                    self.ctx.disable(moderngl.CULL_FACE)
                    self._edge_program["depth_bias"].value = 0.0006
                    front_vao.render(mode=moderngl.LINES)
        rgba = np.frombuffer(
            self._colour_fbo.read(components=4, alignment=1), dtype=np.uint8
        ).reshape((self.viewport_size[1], self.viewport_size[0], 4))
        rgba = np.flipud(rgba).copy()
        self._last_render_key = self._camera_key(camera)
        self._last_pick_image = None
        return RenderSnapshot(rgba=rgba, viewport_size=self.viewport_size)

    def _render_pick_gpu(self, camera: OrbitCamera) -> None:
        """Render the face-ID pick buffer (headless/self-test only).

        Must NOT be called from the Dear PyGui frame thread: binding this
        framebuffer there access-violates on some drivers (observed on AMD
        Radeon) and kills the app. GUI picking always uses the CPU ray path.
        """
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
        """Pick a face at a viewport pixel.

        The GPU branch is for headless use (e.g. ``self_test``) with a
        standalone context only. GUI code must use CPU picking
        (:func:`stl_painter.picking.pick_face_location_cpu`); calling the
        GPU branch from the Dear PyGui frame thread can access-violate.
        """
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

    def _project_vertices(
        self, camera: OrbitCamera, screen_size: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project all mesh vertices to screen pixels.

        Returns ``(screen, valid)`` with ``screen`` shaped ``(V, 2)`` and a
        per-vertex validity mask. Used by face projection and by CAD
        boundary-edge drawing in the software renderer.
        """
        vertices = self.mesh_model.vertices
        width, height = screen_size
        mvp = camera.mvp_matrix(self.viewport_size)
        clip = (mvp @ np.c_[vertices, np.ones(len(vertices), dtype=np.float32)].T).T
        w = clip[:, 3:4]
        safe_w = np.where(np.abs(w) < 1e-6, 1e-6, w)
        ndc = clip[:, :3] / safe_w
        valid_vertices = np.isfinite(ndc).all(axis=1)
        screen = np.empty((len(vertices), 2), dtype=np.float32)
        screen[:, 0] = (ndc[:, 0] * 0.5 + 0.5) * width
        screen[:, 1] = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * height
        return screen, valid_vertices

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
        # No minimum-area culling: on dense meshes a painted face can be a
        # few pixels, and culling it makes click-paint look like a no-op.
        projected, order = self._sorted_face_indices(camera, render_size, min_area=0.0)
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
            if show_triangle_edges and not self.mesh_model.has_cad_faces:
                draw.line(
                    points + [points[0]],
                    fill=(0, 0, 0, 255),
                    width=1,
                )

        if show_triangle_edges and self.mesh_model.has_cad_faces:
            # STEP models: outline CAD faces only, never triangulation.
            # Visibility decisions come from the shared occlusion-tested
            # helper (front-facing filter + pick-image test, integer fast
            # path); only the drawing projection uses render_size.
            visible = self._visible_cad_edges(camera)
            if len(visible) > 0:
                draw_screen, draw_valid = self._project_vertices(
                    camera, render_size
                )
                for u, v in visible.tolist():
                    u, v = int(u), int(v)
                    if not (bool(draw_valid[u]) and bool(draw_valid[v])):
                        continue
                    draw.line(
                        [tuple(draw_screen[u]), tuple(draw_screen[v])],
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

    def _build_pick_image_software(
        self,
        camera: OrbitCamera,
        screen_size: tuple[int, int] | None = None,
    ) -> np.ndarray:
        size = tuple(screen_size) if screen_size is not None else self.viewport_size
        projected, order = self._sorted_face_indices(camera, size)
        width, height = size
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
        if screen_size is None:
            self._last_pick_image = pick_image
        return pick_image

    def _cad_edge_visible_in_pick(
        self,
        pick_image: np.ndarray,
        p0: np.ndarray,
        p1: np.ndarray,
        members: list[int],
        face_zero_front: bool = False,
    ) -> bool:
        """True occlusion test for one CAD boundary edge.

        The front-facing filter alone is not enough on concave models: a
        front-facing edge can sit behind another part of the mesh. The
        edge counts as visible when a sample point along it resolves to
        one of its incident faces (or a triangulation neighbour on the
        same CAD face) in the pick image. Without this, occluded edges
        are drawn on top of the model like an X-ray wireframe.
        """
        height, width = int(pick_image.shape[0]), int(pick_image.shape[1])
        member_set = set(int(m) for m in members)
        wanted_cads: set[object] = set()
        if self.mesh_model.has_cad_faces:
            for member in member_set:
                try:
                    wanted_cads.add(self.mesh_model.cad_id_for_face(member))
                except Exception:
                    continue
            wanted_cads.discard(None)
        for t in (0.1, 0.3, 0.5, 0.7, 0.9):
            cx = int(round(float(p0[0]) + (float(p1[0]) - float(p0[0])) * t))
            cy = int(round(float(p0[1]) + (float(p1[1]) - float(p0[1])) * t))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    x, y = cx + dx, cy + dy
                    if not 0 <= x < width and 0 <= y < height:
                        continue
                    r, g, b = (int(v) for v in pick_image[y, x])
                    shown = (r << 16) | (g << 8) | b
                    if shown in member_set:
                        return True
                    if shown >= self.mesh_model.face_count:
                        continue
                    if wanted_cads:
                        try:
                            if self.mesh_model.cad_id_for_face(int(shown)) in wanted_cads:
                                return True
                        except Exception:
                            pass
                    # Encoded black is ambiguous between background and
                    # face 0; only count it when face 0 owns this edge and
                    # faces the camera.
                    if (
                        shown == 0
                        and 0 in member_set
                        and (r, g, b) == (0, 0, 0)
                        and face_zero_front
                    ):
                        return True
        return False

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

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from .camera import OrbitCamera
from .mesh_model import MeshModel

logger = logging.getLogger(__name__)

VIEWPORT_BG = np.array((237, 240, 245, 255), dtype=np.uint8)


@dataclass(slots=True)
class RenderSnapshot:
    rgba: np.ndarray
    viewport_size: tuple[int, int]


def make_grid_snapshot(viewport_size: tuple[int, int]) -> RenderSnapshot:
    width, height = viewport_size
    rgba = MeshRenderer.create_grid_texture(width, height)
    return RenderSnapshot(rgba=rgba, viewport_size=viewport_size)


class MeshRenderer:
    def __init__(
        self,
        ctx: object | None,
        mesh_model: MeshModel,
        viewport_size: tuple[int, int],
    ) -> None:
        self.ctx = ctx
        self.mesh_model = mesh_model
        self.viewport_size = viewport_size
        self._last_render_key: tuple[tuple[int, int], bytes] | None = None
        self._last_pick_image: np.ndarray | None = None
        logger.info(
            "Initializing software renderer | viewport=%s | faces=%s | vertices=%s",
            viewport_size,
            mesh_model.face_count,
            mesh_model.vertex_count,
        )

    @staticmethod
    def create_grid_texture(width: int, height: int) -> np.ndarray:
        grid = np.ones((height, width, 4), dtype=np.uint8) * 242
        for x in range(0, width, 40):
            grid[:, x, :3] = 214
        for y in range(0, height, 40):
            grid[y, :, :3] = 214
        return grid

    def resize(self, viewport_size: tuple[int, int]) -> None:
        if viewport_size != self.viewport_size:
            self.viewport_size = viewport_size
            self._last_render_key = None
            self._last_pick_image = None

    def update_face_colour(self, face_id: int) -> None:
        self._last_render_key = None
        self._last_pick_image = None

    def _camera_key(self, camera: OrbitCamera) -> tuple[tuple[int, int], bytes]:
        return self.viewport_size, camera.mvp_matrix(self.viewport_size).astype("f4").tobytes()

    def _project_faces(self, camera: OrbitCamera) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        vertices = self.mesh_model.vertices
        faces = self.mesh_model.faces
        width, height = self.viewport_size
        mvp = camera.mvp_matrix(self.viewport_size)
        clip = (mvp @ np.c_[vertices, np.ones(len(vertices), dtype=np.float32)].T).T
        w = clip[:, 3:4]
        safe_w = np.where(np.abs(w) < 1e-6, 1e-6, w)
        ndc = clip[:, :3] / safe_w

        valid_vertices = np.isfinite(ndc).all(axis=1)
        face_vertices = ndc[faces]
        valid_faces = valid_vertices[faces].all(axis=1)
        valid_faces &= (np.abs(face_vertices[:, :, 2]) <= 1.5).any(axis=1)

        screen = np.empty((len(vertices), 2), dtype=np.float32)
        screen[:, 0] = (ndc[:, 0] * 0.5 + 0.5) * width
        screen[:, 1] = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * height

        projected = screen[faces]
        depths = face_vertices[:, :, 2].mean(axis=1)
        return projected, depths, valid_faces

    def _sorted_face_indices(self, camera: OrbitCamera) -> tuple[np.ndarray, np.ndarray]:
        projected, depths, valid_faces = self._project_faces(camera)
        visible = np.where(valid_faces)[0]
        if len(visible) == 0:
            return projected, visible
        order = visible[np.argsort(depths[visible])[::-1]]
        return projected, order

    def render(
        self,
        camera: OrbitCamera,
        light_dir: tuple[float, float, float] = (0.3, 0.8, 0.4),
    ) -> RenderSnapshot:
        projected, order = self._sorted_face_indices(camera)
        width, height = self.viewport_size
        image = Image.new("RGBA", (width, height), tuple(int(v) for v in VIEWPORT_BG))
        draw = ImageDraw.Draw(image, "RGBA")

        light = np.asarray(light_dir, dtype=np.float32)
        light_norm = max(float(np.linalg.norm(light)), 1e-6)
        light = light / light_norm

        for face_id in order:
            points = [tuple(point) for point in projected[face_id]]
            normal = self.mesh_model.normals[face_id]
            normal_norm = max(float(np.linalg.norm(normal)), 1e-6)
            normal = normal / normal_norm
            diffuse = max(float(np.dot(normal, light)), 0.22)
            base = np.asarray(self.mesh_model.face_colour(int(face_id)), dtype=np.float32)
            lit = np.clip(base[:3] * diffuse + 20.0, 0.0, 255.0).astype(np.uint8)
            draw.polygon(points, fill=(int(lit[0]), int(lit[1]), int(lit[2]), int(base[3])))

        rgba = np.asarray(image, dtype=np.uint8)
        self._last_render_key = self._camera_key(camera)
        self._last_pick_image = None
        logger.info(
            "Software render complete | visible_faces=%s | viewport=%s",
            len(order),
            self.viewport_size,
        )
        return RenderSnapshot(rgba=rgba, viewport_size=self.viewport_size)

    def _build_pick_image(self, camera: OrbitCamera) -> np.ndarray:
        projected, order = self._sorted_face_indices(camera)
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

    def pick_face(self, camera: OrbitCamera, mouse_x: int, mouse_y: int) -> int | None:
        if mouse_x < 0 or mouse_y < 0:
            return None
        width, height = self.viewport_size
        if mouse_x >= width or mouse_y >= height:
            return None
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
                return 0 if self._point_hits_face_zero(camera, mouse_x, mouse_y) else None
        if face_id >= self.mesh_model.face_count:
            return None
        return face_id

    def _point_hits_face_zero(self, camera: OrbitCamera, mouse_x: int, mouse_y: int) -> bool:
        projected, order = self._sorted_face_indices(camera)
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

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .camera import OrbitCamera
from .color_utils import Color, blend_over, clamp_color
from .mesh_model import MeshModel, Stroke


ASSET_DIR = Path(__file__).with_name("assets")


@dataclass(slots=True)
class SketchState:
    image: Image.Image
    view_projection: np.ndarray
    viewport_size: tuple[int, int]


def _find_font() -> str | None:
    candidates = [
        ASSET_DIR / "DejaVuSans.ttf",
        Path("C:/Windows/Fonts/DejaVuSans.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/Library/Fonts/Arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


class SketchTool:
    def __init__(self, viewport_size: tuple[int, int]) -> None:
        self.viewport_size = viewport_size
        self._font_path = _find_font()
        self.current: SketchState | None = None

    def begin(self, camera: OrbitCamera) -> SketchState:
        image = Image.new("RGBA", self.viewport_size, (0, 0, 0, 0))
        self.current = SketchState(image=image, view_projection=camera.mvp_matrix(self.viewport_size), viewport_size=self.viewport_size)
        return self.current

    def ensure_state(self, camera: OrbitCamera) -> SketchState:
        if self.current is None or self.current.viewport_size != self.viewport_size:
            return self.begin(camera)
        return self.current

    def resize(self, viewport_size: tuple[int, int]) -> None:
        self.viewport_size = viewport_size
        self.current = None

    def add_text(self, mesh_model: MeshModel, camera: OrbitCamera, position: tuple[float, float], text: str, colour: Color, size: int) -> None:
        state = self.ensure_state(camera)
        draw = ImageDraw.Draw(state.image)
        if self._font_path:
            font = ImageFont.truetype(self._font_path, size=size)
        else:
            font = ImageFont.load_default()
        draw.text(position, text, fill=clamp_color(colour), font=font)
        mesh_model.overlay_strokes.append(
            Stroke(kind="text", data={"position": list(position), "text": text, "colour": list(colour), "size": size}, camera_matrix=state.view_projection.copy())
        )

    def add_rectangle(
        self,
        mesh_model: MeshModel,
        camera: OrbitCamera,
        bounds: tuple[float, float, float, float],
        outline: Color,
        fill: Color | None = None,
        width: int = 2,
    ) -> None:
        state = self.ensure_state(camera)
        draw = ImageDraw.Draw(state.image)
        draw.rectangle(bounds, outline=clamp_color(outline), fill=clamp_color(fill) if fill else None, width=width)
        mesh_model.overlay_strokes.append(
            Stroke(
                kind="rect",
                data={"bounds": list(bounds), "outline": list(outline), "fill": list(fill) if fill else None, "width": width},
                camera_matrix=state.view_projection.copy(),
            )
        )

    def add_line(
        self,
        mesh_model: MeshModel,
        camera: OrbitCamera,
        points: Iterable[tuple[float, float]],
        colour: Color,
        width: int = 3,
    ) -> None:
        state = self.ensure_state(camera)
        points_list = [tuple(point) for point in points]
        draw = ImageDraw.Draw(state.image)
        if len(points_list) >= 2:
            draw.line(points_list, fill=clamp_color(colour), width=width, joint="curve")
        mesh_model.overlay_strokes.append(
            Stroke(kind="line", data={"points": [list(point) for point in points_list], "colour": list(colour), "width": width}, camera_matrix=state.view_projection.copy())
        )

    def add_freehand(
        self,
        mesh_model: MeshModel,
        camera: OrbitCamera,
        points: Iterable[tuple[float, float]],
        colour: Color,
        width: int = 4,
    ) -> None:
        self.add_line(mesh_model, camera, points, colour, width=width)
        if mesh_model.overlay_strokes:
            mesh_model.overlay_strokes[-1].kind = "freehand"

    def clear(self, mesh_model: MeshModel) -> None:
        mesh_model.overlay_strokes.clear()
        self.current = None


def project_vertices_to_screen(vertices: np.ndarray, view_projection: np.ndarray, viewport_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    verts4 = np.hstack([vertices.astype(np.float32), np.ones((len(vertices), 1), dtype=np.float32)])
    clip = (view_projection @ verts4.T).T
    w = clip[:, 3:4]
    valid = w[:, 0] > 0
    ndc = clip[:, :3] / np.where(w == 0, 1.0, w)
    width, height = viewport_size
    x = (ndc[:, 0] + 1.0) * 0.5 * width
    y = (1.0 - (ndc[:, 1] + 1.0) * 0.5) * height
    screen = np.column_stack([x, y]).astype(np.float32)
    return screen, valid


def _face_overlay_colour(sketch_image: Image.Image, uv_verts: np.ndarray) -> Color | None:
    bounds = (
        max(0, int(np.floor(np.min(uv_verts[:, 0])))),
        max(0, int(np.floor(np.min(uv_verts[:, 1])))),
        min(sketch_image.width, int(np.ceil(np.max(uv_verts[:, 0])) + 1)),
        min(sketch_image.height, int(np.ceil(np.max(uv_verts[:, 1])) + 1)),
    )
    if bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
        return None
    cropped = sketch_image.crop(bounds)
    mask = Image.new("L", cropped.size, 0)
    shifted = [(float(x - bounds[0]), float(y - bounds[1])) for x, y in uv_verts]
    ImageDraw.Draw(mask).polygon(shifted, fill=255)
    pixels = np.asarray(cropped, dtype=np.uint8)
    mask_arr = np.asarray(mask, dtype=np.uint8)
    covered = pixels[(mask_arr > 0) & (pixels[:, :, 3] > 0)]
    if covered.size == 0:
        return None
    mean = covered.mean(axis=0)
    return clamp_color(mean.tolist())


def bake_sketch_to_faces(mesh_model: MeshModel, sketch_image: Image.Image, view_projection: np.ndarray, viewport_size: tuple[int, int]) -> dict[int, Color]:
    screen_vertices, valid_vertices = project_vertices_to_screen(mesh_model.vertices, view_projection, viewport_size)
    baked: dict[int, Color] = {}
    for face_id, face in enumerate(mesh_model.faces):
        if not valid_vertices[face].all():
            continue
        uv_verts = screen_vertices[face]
        overlay_colour = _face_overlay_colour(sketch_image, uv_verts)
        if overlay_colour is None:
            continue
        baked[face_id] = blend_over(mesh_model.face_colour(face_id), overlay_colour)
    return baked

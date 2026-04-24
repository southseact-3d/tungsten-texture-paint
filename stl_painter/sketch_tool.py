from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .camera import OrbitCamera
from .color_utils import Color, blend_over, clamp_color
from .mesh_model import MeshModel, SketchDocument, SketchEntity, SketchPlane
from .svg_tool import load_svg_as_image


ASSET_DIR = Path(__file__).with_name("assets")


@dataclass(slots=True)
class SketchHit:
    world: np.ndarray
    plane_uv: np.ndarray
    snapped: bool
    snap_label: str | None = None


def _normalize(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length <= 1e-8:
        return vector.astype(np.float32)
    return (vector / length).astype(np.float32)


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
    def __init__(self) -> None:
        self._font_path = _find_font()

    def create_plane_from_face(self, mesh_model: MeshModel, face_id: int) -> SketchDocument:
        face = mesh_model.face_vertices(face_id)
        origin = face.mean(axis=0).astype(np.float32)
        normal = _normalize(mesh_model.normals[int(face_id)])
        edge = face[1] - face[0]
        if float(np.linalg.norm(edge)) <= 1e-8:
            edge = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        tangent_u = _normalize(edge)
        tangent_v = _normalize(np.cross(normal, tangent_u))
        if float(np.linalg.norm(tangent_v)) <= 1e-8:
            fallback = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            tangent_u = _normalize(np.cross(fallback, normal))
            tangent_v = _normalize(np.cross(normal, tangent_u))
        plane = SketchPlane(
            origin=origin,
            normal=normal,
            tangent_u=tangent_u,
            tangent_v=tangent_v,
            anchor_face_id=int(face_id),
        )
        return SketchDocument(plane=plane)

    def world_to_plane(self, plane: SketchPlane, point: np.ndarray) -> np.ndarray:
        offset = point.astype(np.float32) - plane.origin
        return np.asarray(
            [
                float(np.dot(offset, plane.tangent_u)),
                float(np.dot(offset, plane.tangent_v)),
            ],
            dtype=np.float32,
        )

    def plane_to_world(self, plane: SketchPlane, uv: np.ndarray) -> np.ndarray:
        return (
            plane.origin
            + plane.tangent_u * float(uv[0])
            + plane.tangent_v * float(uv[1])
        ).astype(np.float32)

    def ray_to_plane(
        self,
        plane: SketchPlane,
        camera: OrbitCamera,
        mouse_pos: tuple[float, float],
        viewport_size: tuple[int, int],
    ) -> SketchHit | None:
        origin, direction = camera.unproject_ray(mouse_pos[0], mouse_pos[1], viewport_size)
        denom = float(np.dot(direction, plane.normal))
        if abs(denom) <= 1e-6:
            return None
        t = float(np.dot(plane.origin - origin, plane.normal) / denom)
        if t < 0.0:
            return None
        world = origin + direction * t
        uv = self.world_to_plane(plane, world)
        return SketchHit(world=world, plane_uv=uv, snapped=False, snap_label=None)

    def snap_point(
        self,
        mesh_model: MeshModel,
        document: SketchDocument,
        uv: np.ndarray,
        *,
        shift_lock_axis: np.ndarray | None = None,
    ) -> SketchHit:
        best_uv = uv.astype(np.float32)
        best_distance = float("inf")
        best_label: str | None = None

        def consider(candidate: np.ndarray, label: str) -> None:
            nonlocal best_uv, best_distance, best_label
            distance = float(np.linalg.norm(candidate - uv))
            if distance < best_distance:
                best_uv = candidate.astype(np.float32)
                best_distance = distance
                best_label = label

        grid_candidate: np.ndarray | None = None
        if document.snap_to_grid and document.grid_size > 1e-8:
            grid = document.grid_size
            grid_candidate = (np.round(uv / grid) * grid).astype(np.float32)
            if float(np.linalg.norm(grid_candidate - uv)) <= max(document.grid_size * 0.6, 0.05):
                return SketchHit(
                    world=self.plane_to_world(document.plane, grid_candidate),
                    plane_uv=grid_candidate,
                    snapped=True,
                    snap_label="grid",
                )
            consider(grid_candidate, "grid")

        if document.snap_to_vertices or document.snap_to_edges:
            anchor_face = mesh_model.face_vertices(document.plane.anchor_face_id)
            anchor_uv = np.asarray(
                [self.world_to_plane(document.plane, point) for point in anchor_face],
                dtype=np.float32,
            )
            if document.snap_to_vertices:
                for point_uv in anchor_uv:
                    consider(point_uv, "vertex")
            if document.snap_to_edges:
                for index in range(3):
                    a = anchor_uv[index]
                    b = anchor_uv[(index + 1) % 3]
                    midpoint = (a + b) * 0.5
                    consider(midpoint, "edge midpoint")

        if document.snap_to_entities:
            for entity in document.entities:
                for point_uv in entity_snap_points(entity):
                    consider(point_uv, f"{entity.kind} snap")

        if shift_lock_axis is not None:
            if abs(float(shift_lock_axis[0])) >= abs(float(shift_lock_axis[1])):
                consider(np.asarray([uv[0], shift_lock_axis[1]], dtype=np.float32), "ortho")
            else:
                consider(np.asarray([shift_lock_axis[0], uv[1]], dtype=np.float32), "ortho")

        snapped = best_distance <= max(document.grid_size * 0.6, 0.05)
        final_uv = best_uv if snapped else uv.astype(np.float32)
        return SketchHit(
            world=self.plane_to_world(document.plane, final_uv),
            plane_uv=final_uv,
            snapped=snapped,
            snap_label=best_label if snapped else None,
        )

    def create_entity(
        self,
        kind: str,
        start_uv: np.ndarray,
        end_uv: np.ndarray,
        colour: Color,
        *,
        text: str = "Text",
        stroke_width: float = 0.02,
        text_size: int = 28,
    ) -> SketchEntity:
        data: dict[str, object]
        if kind == "rect":
            mins = np.minimum(start_uv, end_uv)
            maxs = np.maximum(start_uv, end_uv)
            data = {
                "min": mins.tolist(),
                "max": maxs.tolist(),
                "colour": list(colour),
                "stroke_width": stroke_width,
            }
        elif kind == "line":
            data = {
                "start": start_uv.tolist(),
                "end": end_uv.tolist(),
                "colour": list(colour),
                "stroke_width": stroke_width,
            }
        elif kind == "circle":
            radius = float(np.linalg.norm(end_uv - start_uv))
            data = {
                "center": start_uv.tolist(),
                "radius": radius,
                "colour": list(colour),
                "stroke_width": stroke_width,
            }
        elif kind == "text":
            data = {
                "position": start_uv.tolist(),
                "text": text,
                "colour": list(colour),
                "size": int(text_size),
            }
        elif kind == "svg":
            mins = np.minimum(start_uv, end_uv)
            maxs = np.maximum(start_uv, end_uv)
            data = {
                "min": mins.tolist(),
                "max": maxs.tolist(),
                "path": "",
                "tint": list(colour),
            }
        else:
            data = {
                "start": start_uv.tolist(),
                "end": end_uv.tolist(),
                "colour": list(colour),
                "stroke_width": stroke_width,
            }
        return SketchEntity(entity_id=uuid4().hex[:8], kind=kind, data=data)

    def resize_entity(
        self, entity: SketchEntity, handle: str, new_uv: np.ndarray
    ) -> dict[str, object]:
        data = dict(entity.data)
        if entity.kind == "rect":
            min_uv = np.asarray(data["min"], dtype=np.float32)
            max_uv = np.asarray(data["max"], dtype=np.float32)
            if handle == "min":
                min_uv = new_uv
            elif handle == "max":
                max_uv = new_uv
            else:
                center = (min_uv + max_uv) * 0.5
                size = (max_uv - min_uv) * 0.5
                center = new_uv
                min_uv = center - size
                max_uv = center + size
            data["min"] = np.minimum(min_uv, max_uv).tolist()
            data["max"] = np.maximum(min_uv, max_uv).tolist()
        elif entity.kind == "line":
            data["end" if handle == "end" else "start"] = new_uv.tolist()
        elif entity.kind == "circle":
            center = np.asarray(data["center"], dtype=np.float32)
            data["radius"] = float(np.linalg.norm(new_uv - center))
        elif entity.kind == "text":
            data["position"] = new_uv.tolist()
        elif entity.kind == "svg":
            min_uv = np.asarray(data.get("min", [0.0, 0.0]), dtype=np.float32)
            max_uv = np.asarray(data.get("max", [0.1, 0.1]), dtype=np.float32)
            if handle == "min":
                min_uv = new_uv
            elif handle == "max":
                max_uv = new_uv
            else:
                center = new_uv
                size = (max_uv - min_uv) * 0.5
                min_uv = center - size
                max_uv = center + size
            data["min"] = np.minimum(min_uv, max_uv).tolist()
            data["max"] = np.maximum(min_uv, max_uv).tolist()
        return data

    def render_entity_to_image(
        self,
        entity: SketchEntity,
        document: SketchDocument,
        image: Image.Image,
        bounds: tuple[float, float, float, float],
        pixels_per_unit: float,
    ) -> None:
        draw = ImageDraw.Draw(image)
        text_size = int(entity.data.get("size", 28))
        if self._font_path:
            font = ImageFont.truetype(self._font_path, size=max(8, text_size))
        else:
            font = ImageFont.load_default()
        def to_px(uv: np.ndarray) -> tuple[float, float]:
            x = (float(uv[0]) - bounds[0]) * pixels_per_unit
            y = (bounds[3] - float(uv[1])) * pixels_per_unit
            return x, y
        colour = tuple(int(value) for value in entity.data.get("colour", [255, 0, 0, 255]))
        stroke_width = max(1, int(round(float(entity.data.get("stroke_width", 0.02)) * pixels_per_unit)))
        if entity.kind == "rect":
            min_uv = np.asarray(entity.data["min"], dtype=np.float32)
            max_uv = np.asarray(entity.data["max"], dtype=np.float32)
            p0 = to_px(min_uv)
            p1 = to_px(max_uv)
            draw.rectangle(
                [
                    (min(p0[0], p1[0]), min(p0[1], p1[1])),
                    (max(p0[0], p1[0]), max(p0[1], p1[1])),
                ],
                outline=colour,
                width=stroke_width,
            )
        elif entity.kind == "line":
            start_uv = np.asarray(entity.data["start"], dtype=np.float32)
            end_uv = np.asarray(entity.data["end"], dtype=np.float32)
            draw.line([to_px(start_uv), to_px(end_uv)], fill=colour, width=stroke_width)
        elif entity.kind == "circle":
            center = np.asarray(entity.data["center"], dtype=np.float32)
            radius = float(entity.data["radius"])
            min_uv = center - np.asarray([radius, radius], dtype=np.float32)
            max_uv = center + np.asarray([radius, radius], dtype=np.float32)
            p0 = to_px(min_uv)
            p1 = to_px(max_uv)
            draw.ellipse(
                [
                    (min(p0[0], p1[0]), min(p0[1], p1[1])),
                    (max(p0[0], p1[0]), max(p0[1], p1[1])),
                ],
                outline=colour,
                width=stroke_width,
            )
        elif entity.kind == "text":
            position = np.asarray(entity.data["position"], dtype=np.float32)
            draw.text(to_px(position), str(entity.data.get("text", "Text")), fill=colour, font=font)
        elif entity.kind == "svg":
            min_uv = np.asarray(entity.data.get("min", [0.0, 0.0]), dtype=np.float32)
            max_uv = np.asarray(entity.data.get("max", [0.1, 0.1]), dtype=np.float32)
            p0 = to_px(min_uv)
            p1 = to_px(max_uv)
            left = int(min(p0[0], p1[0]))
            right = int(max(p0[0], p1[0]))
            top = int(min(p0[1], p1[1]))
            bottom = int(max(p0[1], p1[1]))
            width = max(1, right - left)
            height = max(1, bottom - top)
            svg_image = load_svg_as_image(str(entity.data.get("path", "")), (width, height))
            if svg_image is None:
                return
            tint = np.asarray(entity.data.get("tint", [255, 255, 255, 255]), dtype=np.uint8)
            arr = np.asarray(svg_image, dtype=np.uint8).copy()
            arr[:, :, :3] = ((arr[:, :, :3].astype(np.float32) * (tint[:3].astype(np.float32) / 255.0))).astype(np.uint8)
            arr[:, :, 3] = ((arr[:, :, 3].astype(np.float32) * (float(tint[3]) / 255.0))).astype(np.uint8)
            image.alpha_composite(Image.fromarray(arr, mode="RGBA"), (left, top))

    def bake_document_to_faces(
        self,
        mesh_model: MeshModel,
        document: SketchDocument,
        base_colours: dict[int, Color] | None = None,
        *,
        front_facing_only: bool = True,
    ) -> dict[int, Color]:
        """Bake sketch entities onto mesh face colours using orthographic projection.

        Each face centroid is projected onto the sketch plane via dot-product
        (orthographic, not ray-cast), so the bake works reliably on curved surfaces,
        thick solids, and any face regardless of its distance from the plane.

        Args:
            mesh_model: The mesh to paint.
            document: The sketch document containing entities.
            base_colours: Optional existing face-colour dict to blend onto.
            front_facing_only: When True (default), only faces whose normal has a
                positive dot-product with the sketch plane normal are painted,
                preventing the bake from "bleeding through" to back faces.
        """
        if not document.entities:
            return {}

        colours = base_colours or {
            face_id: mesh_model.face_colour(face_id)
            for face_id in range(mesh_model.face_count)
        }

        # -- Orthographic projection of all face centroids onto the sketch plane --
        # centroids shape: (F, 3)
        centroids = mesh_model.vertices[mesh_model.faces].mean(axis=1).astype(np.float32)
        offsets = centroids - document.plane.origin          # (F, 3)
        u_proj = (offsets * document.plane.tangent_u).sum(axis=1)   # (F,)
        v_proj = (offsets * document.plane.tangent_v).sum(axis=1)   # (F,)

        # -- Optional: skip back-facing faces --
        if front_facing_only:
            dots = (mesh_model.normals * document.plane.normal).sum(axis=1)  # (F,)
            front_mask = dots >= -0.1  # allow nearly-perpendicular faces too
        else:
            front_mask = np.ones(mesh_model.face_count, dtype=bool)

        baked: dict[int, Color] = {}

        for entity in document.entities:
            raw_colour = entity.data.get("colour", [255, 0, 0, 255])
            colour = clamp_color(tuple(int(c) for c in raw_colour))

            if entity.kind in ("text", "svg"):
                # Raster path for text/SVG: render to an image then sample
                face_ids = _raster_bake_entity(self, entity, document, mesh_model,
                                               u_proj, v_proj, front_mask, colours)
            else:
                face_ids = _faces_covered_by_entity(entity, u_proj, v_proj)
                face_ids = [fid for fid in face_ids if front_mask[fid]]

            for face_id in face_ids:
                baked[int(face_id)] = colour

        return baked


# ---------------------------------------------------------------------------
# Module-level baking helpers (used by SketchTool.bake_document_to_faces)
# ---------------------------------------------------------------------------

def _faces_covered_by_entity(
    entity: SketchEntity,
    u_proj: np.ndarray,
    v_proj: np.ndarray,
) -> list[int]:
    """Return face indices (into u_proj / v_proj) whose projected centroid falls
    inside the geometric bounds of *entity*.

    This uses pure NumPy broadcasting and works for **any** face regardless of
    its 3-D distance from the sketch plane.
    """
    kind = entity.kind

    if kind == "rect":
        min_u, min_v = float(entity.data["min"][0]), float(entity.data["min"][1])
        max_u, max_v = float(entity.data["max"][0]), float(entity.data["max"][1])
        if min_u > max_u:
            min_u, max_u = max_u, min_u
        if min_v > max_v:
            min_v, max_v = max_v, min_v
        mask = (
            (u_proj >= min_u) & (u_proj <= max_u)
            & (v_proj >= min_v) & (v_proj <= max_v)
        )
        return [int(i) for i in np.where(mask)[0]]

    if kind == "circle":
        cx = float(entity.data["center"][0])
        cy = float(entity.data["center"][1])
        radius = float(entity.data["radius"])
        du = u_proj - cx
        dv = v_proj - cy
        mask = (du * du + dv * dv) <= (radius * radius)
        return [int(i) for i in np.where(mask)[0]]

    if kind == "line":
        start = np.asarray(entity.data["start"], dtype=np.float32)
        end = np.asarray(entity.data["end"], dtype=np.float32)
        half_w = max(float(entity.data.get("stroke_width", 0.02)) * 0.5, 1e-6)
        line_vec = end - start
        length = float(np.linalg.norm(line_vec))
        if length < 1e-8:
            return []
        line_n = line_vec / length
        pts = np.column_stack([u_proj, v_proj]).astype(np.float32)
        rel = pts - start
        t = np.clip(rel @ line_n, 0.0, length)
        proj = start + np.outer(t, line_n)
        dist = np.linalg.norm(pts - proj, axis=1)
        return [int(i) for i in np.where(dist <= half_w)[0]]

    return []


def _raster_bake_entity(
    tool: "SketchTool",
    entity: "SketchEntity",
    document: "SketchDocument",
    mesh_model: "MeshModel",
    u_proj: np.ndarray,
    v_proj: np.ndarray,
    front_mask: np.ndarray,
    colours: dict[int, "Color"],
) -> list[int]:
    """Rasterise a single text/SVG entity to a small image and return the face IDs
    whose projected centroid lands inside a painted pixel.

    Falls back gracefully: if rasterisation produces no pixels, returns an empty list.
    """
    snap_pts = entity_snap_points(entity)
    if not snap_pts:
        return []

    # Build tight UV bounding box for the entity
    uv_arr = np.asarray(snap_pts, dtype=np.float32)
    min_uv = uv_arr.min(axis=0) - 0.05
    max_uv = uv_arr.max(axis=0) + 0.05
    bounds = (float(min_uv[0]), float(min_uv[1]), float(max_uv[0]), float(max_uv[1]))
    size_uv = np.maximum(max_uv - min_uv, 1e-3)
    pixels_per_unit = max(256.0 / float(max(size_uv)), 128.0)
    width  = max(64, int(np.ceil(size_uv[0] * pixels_per_unit)))
    height = max(64, int(np.ceil(size_uv[1] * pixels_per_unit)))

    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    tool.render_entity_to_image(entity, document, image, bounds, pixels_per_unit)

    # Find faces whose UV projection lands inside the image and hits a painted pixel
    result: list[int] = []
    for face_id in range(mesh_model.face_count):
        if not front_mask[face_id]:
            continue
        u = float(u_proj[face_id])
        v = float(v_proj[face_id])
        # Convert UV → pixel coordinate
        px = int((u - bounds[0]) * pixels_per_unit)
        py = int((bounds[3] - v) * pixels_per_unit)
        if 0 <= px < width and 0 <= py < height:
            pixel = image.getpixel((px, py))
            if pixel[3] > 16:  # has a meaningful alpha → inside painted region
                result.append(face_id)

    return result


def entity_snap_points(entity: SketchEntity) -> list[np.ndarray]:
    if entity.kind == "rect":
        min_uv = np.asarray(entity.data["min"], dtype=np.float32)
        max_uv = np.asarray(entity.data["max"], dtype=np.float32)
        center = (min_uv + max_uv) * 0.5
        return [
            min_uv,
            max_uv,
            np.asarray([min_uv[0], max_uv[1]], dtype=np.float32),
            np.asarray([max_uv[0], min_uv[1]], dtype=np.float32),
            center,
            np.asarray([center[0], min_uv[1]], dtype=np.float32),
            np.asarray([center[0], max_uv[1]], dtype=np.float32),
            np.asarray([min_uv[0], center[1]], dtype=np.float32),
            np.asarray([max_uv[0], center[1]], dtype=np.float32),
        ]
    if entity.kind == "line":
        start = np.asarray(entity.data["start"], dtype=np.float32)
        end = np.asarray(entity.data["end"], dtype=np.float32)
        return [start, end, (start + end) * 0.5]
    if entity.kind == "circle":
        center = np.asarray(entity.data["center"], dtype=np.float32)
        radius = float(entity.data["radius"])
        return [
            center,
            center + np.asarray([radius, 0.0], dtype=np.float32),
            center + np.asarray([-radius, 0.0], dtype=np.float32),
            center + np.asarray([0.0, radius], dtype=np.float32),
            center + np.asarray([0.0, -radius], dtype=np.float32),
        ]
    if entity.kind == "text":
        return [np.asarray(entity.data["position"], dtype=np.float32)]
    if entity.kind == "svg":
        min_uv = np.asarray(entity.data.get("min", [0.0, 0.0]), dtype=np.float32)
        max_uv = np.asarray(entity.data.get("max", [0.1, 0.1]), dtype=np.float32)
        center = (min_uv + max_uv) * 0.5
        return [min_uv, max_uv, center]
    return []


def _sample_polygon(
    image: Image.Image,
    uv_verts: np.ndarray,
    bounds: tuple[float, float, float, float],
    pixels_per_unit: float,
) -> Color | None:
    def to_px(uv: np.ndarray) -> tuple[float, float]:
        x = (float(uv[0]) - bounds[0]) * pixels_per_unit
        y = (bounds[3] - float(uv[1])) * pixels_per_unit
        return x, y

    px = np.asarray([to_px(vertex) for vertex in uv_verts], dtype=np.float32)
    min_x = max(0, int(np.floor(px[:, 0].min())))
    max_x = min(image.width, int(np.ceil(px[:, 0].max()) + 1))
    min_y = max(0, int(np.floor(px[:, 1].min())))
    max_y = min(image.height, int(np.ceil(px[:, 1].max()) + 1))
    if min_x >= max_x or min_y >= max_y:
        return None
    cropped = image.crop((min_x, min_y, max_x, max_y))
    mask = Image.new("L", cropped.size, 0)
    shifted = [(float(x - min_x), float(y - min_y)) for x, y in px]
    ImageDraw.Draw(mask).polygon(shifted, fill=255)
    pixels = np.asarray(cropped, dtype=np.uint8)
    mask_arr = np.asarray(mask, dtype=np.uint8)
    covered = pixels[(mask_arr > 0) & (pixels[:, :, 3] > 0)]
    if covered.size == 0:
        return None
    return clamp_color(covered.mean(axis=0))


def project_vertices_to_screen(
    vertices: np.ndarray,
    view_projection: np.ndarray,
    viewport_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    verts4 = np.hstack(
        [vertices.astype(np.float32), np.ones((len(vertices), 1), dtype=np.float32)]
    )
    clip = (view_projection @ verts4.T).T
    w = clip[:, 3:4]
    valid = w[:, 0] > 0
    ndc = clip[:, :3] / np.where(w == 0, 1.0, w)
    width, height = viewport_size
    x = (ndc[:, 0] + 1.0) * 0.5 * width
    y = (1.0 - (ndc[:, 1] + 1.0) * 0.5) * height
    screen = np.column_stack([x, y]).astype(np.float32)
    return screen, valid


def bake_sketch_to_faces(
    mesh_model: MeshModel,
    sketch_image: Image.Image,
    view_projection: np.ndarray,
    viewport_size: tuple[int, int],
) -> dict[int, Color]:
    screen_vertices, valid_vertices = project_vertices_to_screen(
        mesh_model.vertices, view_projection, viewport_size
    )
    baked: dict[int, Color] = {}
    for face_id, face in enumerate(mesh_model.faces):
        if not valid_vertices[face].all():
            continue
        uv_verts = screen_vertices[face]
        overlay_colour = _sample_polygon(
            sketch_image,
            uv_verts,
            (0.0, 0.0, float(sketch_image.width), float(sketch_image.height)),
            1.0,
        )
        if overlay_colour is None:
            continue
        baked[face_id] = blend_over(mesh_model.face_colour(face_id), overlay_colour)
    return baked

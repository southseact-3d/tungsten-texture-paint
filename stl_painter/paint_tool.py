from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import acos

import numpy as np

from .color_utils import Color, blend_over, clamp_color
from .mesh_model import MeshModel


@dataclass(slots=True)
class PaintChange:
    face_id: int
    old_colour: Color
    new_colour: Color


class UndoStack:
    def __init__(self, max_size: int = 200) -> None:
        self.stack: deque[list[PaintChange]] = deque(maxlen=max_size)

    def push(self, changes: list[PaintChange]) -> None:
        if changes:
            self.stack.append(changes)

    def undo(self, mesh_model: MeshModel) -> list[int]:
        if not self.stack:
            return []
        changes = self.stack.pop()
        touched: list[int] = []
        for change in changes:
            mesh_model.set_face_colour(change.face_id, change.old_colour)
            touched.append(change.face_id)
        return touched


def _normalize(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length <= 1e-8:
        return vector.astype(np.float32)
    return (vector / length).astype(np.float32)


def _falloff_weight(distance: float, radius: float, mode: str) -> float:
    if radius <= 1e-8:
        return 1.0 if distance <= 1e-8 else 0.0
    t = max(0.0, min(1.0, 1.0 - distance / radius))
    if mode == "constant":
        return 1.0 if distance <= radius else 0.0
    if mode == "linear":
        return t
    return t * t * (3.0 - 2.0 * t)


class PaintTool:
    def __init__(self, mesh_model: MeshModel, undo_stack: UndoStack | None = None) -> None:
        self.mesh_model = mesh_model
        self.undo_stack = undo_stack or UndoStack()

    def paint_face(self, face_id: int, colour: Color) -> list[int]:
        old_colour = self.mesh_model.face_colour(face_id)
        if old_colour == colour:
            return []
        self.mesh_model.set_face_colour(face_id, colour)
        self.undo_stack.push(
            [PaintChange(face_id=face_id, old_colour=old_colour, new_colour=colour)]
        )
        return [face_id]

    def paint_faces(self, updates: dict[int, Color]) -> list[int]:
        changes: list[PaintChange] = []
        for face_id, colour in updates.items():
            old_colour = self.mesh_model.face_colour(face_id)
            if old_colour == colour:
                continue
            self.mesh_model.set_face_colour(face_id, colour)
            changes.append(
                PaintChange(
                    face_id=int(face_id),
                    old_colour=old_colour,
                    new_colour=clamp_color(colour),
                )
            )
        self.undo_stack.push(changes)
        return [change.face_id for change in changes]

    def flood_fill(
        self, start_face_id: int, colour: Color, tolerance: float = 0.0
    ) -> list[int]:
        updates = self.flood_fill_updates(start_face_id, colour, tolerance=tolerance)
        return self.paint_faces(updates)

    def flood_fill_updates(
        self, start_face_id: int, colour: Color, tolerance: float = 0.0
    ) -> dict[int, Color]:
        """Flood-fill connected faces similar to the seed face's colour.

        ``tolerance`` is a max RGB Euclidean distance (0 = exact match).
        Non-zero tolerance lets fill cross the ±1-2 LSB variance left by
        baked texture colours on pretextured imports.
        """
        target = np.asarray(
            self.mesh_model.face_colour(start_face_id)[:3], dtype=np.float32
        )
        if np.linalg.norm(
            target - np.asarray(colour[:3], dtype=np.float32)
        ) <= max(0.0, float(tolerance)):
            return {}
        adjacency = self.mesh_model.adjacency_map()
        queue = deque([start_face_id])
        visited = {start_face_id}
        updates: dict[int, Color] = {}
        limit = max(0.0, float(tolerance))
        while queue:
            face_id = queue.popleft()
            candidate = np.asarray(
                self.mesh_model.face_colour(face_id)[:3], dtype=np.float32
            )
            if float(np.linalg.norm(candidate - target)) > limit:
                continue
            if face_id not in self.mesh_model.masked_faces:
                updates[face_id] = colour
            for neighbour in adjacency[face_id]:
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append(neighbour)
        return updates

    def sample_colour(self, face_id: int) -> Color:
        return self.mesh_model.face_colour(face_id)

    def mask_faces(self, face_ids: set[int]) -> None:
        self.mesh_model.masked_faces.update(int(face_id) for face_id in face_ids)

    def unmask_faces(self, face_ids: set[int]) -> None:
        self.mesh_model.masked_faces.difference_update(int(face_id) for face_id in face_ids)

    def brush_paint(
        self,
        center_face_id: int,
        hit_point: np.ndarray,
        colour: Color,
        *,
        radius: float,
        opacity: float,
        falloff: str = "smooth",
        front_faces_only: bool = False,
        angle_tolerance_degrees: float = 65.0,
        erase: bool = False,
    ) -> list[int]:
        updates = self.brush_updates(
            center_face_id,
            hit_point,
            colour,
            radius=radius,
            opacity=opacity,
            falloff=falloff,
            front_faces_only=front_faces_only,
            angle_tolerance_degrees=angle_tolerance_degrees,
            erase=erase,
        )
        return self.paint_faces(updates)

    def brush_updates(
        self,
        center_face_id: int,
        hit_point: np.ndarray,
        colour: Color,
        *,
        radius: float,
        opacity: float,
        falloff: str = "smooth",
        front_faces_only: bool = False,
        angle_tolerance_degrees: float = 65.0,
        erase: bool = False,
    ) -> dict[int, Color]:
        adjacency = self.mesh_model.adjacency_map()
        radius = max(1e-6, float(radius))
        opacity = max(0.0, min(1.0, float(opacity)))
        max_angle = np.deg2rad(angle_tolerance_degrees)
        target_normal = _normalize(self.mesh_model.normals[int(center_face_id)])
        updates: dict[int, Color] = {}
        queue = deque([int(center_face_id)])
        visited: set[int] = set()
        while queue:
            face_id = queue.popleft()
            if face_id in visited:
                continue
            visited.add(face_id)
            if face_id in self.mesh_model.masked_faces:
                continue
            center = self.mesh_model.face_center(face_id)
            distance = float(np.linalg.norm(center - hit_point))
            if distance > radius:
                continue
            normal = _normalize(self.mesh_model.normals[face_id])
            dot = float(np.clip(np.dot(normal, target_normal), -1.0, 1.0))
            angle = acos(dot)
            if angle > max_angle:
                continue
            if front_faces_only and dot < 0.25:
                continue
            weight = _falloff_weight(distance, radius, falloff) * opacity
            if face_id == int(center_face_id):
                weight = max(weight, 1.0)
            if weight <= 0.0:
                continue
            old_colour = self.mesh_model.face_colour(face_id)
            paint_colour = self.mesh_model.default_colour if erase else colour
            applied = blend_over(
                old_colour,
                clamp_color(
                    (
                        paint_colour[0],
                        paint_colour[1],
                        paint_colour[2],
                        int(round(255 * weight)),
                    )
                ),
            )
            if applied != old_colour:
                updates[face_id] = applied
            for neighbour in adjacency[face_id]:
                if neighbour not in visited:
                    queue.append(neighbour)
        return updates

    def undo(self) -> list[int]:
        return self.undo_stack.undo(self.mesh_model)

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .color_utils import Color
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


class PaintTool:
    def __init__(self, mesh_model: MeshModel, undo_stack: UndoStack | None = None) -> None:
        self.mesh_model = mesh_model
        self.undo_stack = undo_stack or UndoStack()

    def paint_face(self, face_id: int, colour: Color) -> list[int]:
        old_colour = self.mesh_model.face_colour(face_id)
        if old_colour == colour:
            return []
        self.mesh_model.set_face_colour(face_id, colour)
        self.undo_stack.push([PaintChange(face_id=face_id, old_colour=old_colour, new_colour=colour)])
        return [face_id]

    def flood_fill(self, start_face_id: int, colour: Color) -> list[int]:
        target_colour = self.mesh_model.face_colour(start_face_id)
        if target_colour == colour:
            return []
        adjacency = self.mesh_model.adjacency_map()
        queue = deque([start_face_id])
        visited = {start_face_id}
        changes: list[PaintChange] = []
        while queue:
            face_id = queue.popleft()
            if self.mesh_model.face_colour(face_id) != target_colour:
                continue
            changes.append(PaintChange(face_id=face_id, old_colour=target_colour, new_colour=colour))
            self.mesh_model.set_face_colour(face_id, colour)
            for neighbour in adjacency[face_id]:
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append(neighbour)
        self.undo_stack.push(changes)
        return [change.face_id for change in changes]

    def undo(self) -> list[int]:
        return self.undo_stack.undo(self.mesh_model)

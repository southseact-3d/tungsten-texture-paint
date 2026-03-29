from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .color_utils import Color
from .mesh_model import MeshModel, SketchDocument, SketchEntity
from .paint_tool import PaintTool


class Command(Protocol):
    def apply(self) -> list[int]: ...

    def undo(self) -> list[int]: ...

    @property
    def description(self) -> str: ...


@dataclass(slots=True)
class PaintFacesCommand:
    mesh_model: MeshModel
    updates: dict[int, Color]
    _previous: dict[int, Color] = field(default_factory=dict)
    description: str = "Paint faces"

    def apply(self) -> list[int]:
        touched: list[int] = []
        for face_id, colour in self.updates.items():
            self._previous[int(face_id)] = self.mesh_model.face_colour(face_id)
            self.mesh_model.set_face_colour(face_id, colour)
            touched.append(int(face_id))
        return touched

    def undo(self) -> list[int]:
        touched: list[int] = []
        for face_id, colour in self._previous.items():
            self.mesh_model.set_face_colour(face_id, colour)
            touched.append(int(face_id))
        return touched


@dataclass(slots=True)
class SketchEntityCommand:
    document: SketchDocument
    entity: SketchEntity
    description: str = "Add sketch entity"

    def apply(self) -> list[int]:
        self.document.entities.append(self.entity)
        self.document.selected_entity_id = self.entity.entity_id
        return []

    def undo(self) -> list[int]:
        self.document.entities = [
            item for item in self.document.entities if item.entity_id != self.entity.entity_id
        ]
        if self.document.selected_entity_id == self.entity.entity_id:
            self.document.selected_entity_id = None
        return []


@dataclass(slots=True)
class UpdateSketchEntityCommand:
    document: SketchDocument
    entity_id: str
    new_data: dict[str, object]
    _old_data: dict[str, object] = field(default_factory=dict)
    description: str = "Update sketch entity"

    def apply(self) -> list[int]:
        for entity in self.document.entities:
            if entity.entity_id == self.entity_id:
                self._old_data = dict(entity.data)
                entity.data = dict(self.new_data)
                self.document.selected_entity_id = entity.entity_id
                break
        return []

    def undo(self) -> list[int]:
        for entity in self.document.entities:
            if entity.entity_id == self.entity_id:
                entity.data = dict(self._old_data)
                break
        return []


@dataclass(slots=True)
class DeleteSketchEntityCommand:
    document: SketchDocument
    entity_id: str
    _removed: SketchEntity | None = None
    description: str = "Delete sketch entity"

    def apply(self) -> list[int]:
        kept: list[SketchEntity] = []
        for entity in self.document.entities:
            if entity.entity_id == self.entity_id:
                self._removed = entity
            else:
                kept.append(entity)
        self.document.entities = kept
        if self.document.selected_entity_id == self.entity_id:
            self.document.selected_entity_id = None
        return []

    def undo(self) -> list[int]:
        if self._removed is not None:
            self.document.entities.append(self._removed)
        return []


@dataclass(slots=True)
class SetMaskedFacesCommand:
    mesh_model: MeshModel
    new_masked_faces: set[int]
    _previous_masked_faces: set[int] = field(default_factory=set)
    description: str = "Update masked faces"

    def apply(self) -> list[int]:
        self._previous_masked_faces = set(self.mesh_model.masked_faces)
        self.mesh_model.masked_faces = {int(face_id) for face_id in self.new_masked_faces}
        touched = self._previous_masked_faces.union(self.mesh_model.masked_faces)
        return sorted(touched)

    def undo(self) -> list[int]:
        current = set(self.mesh_model.masked_faces)
        self.mesh_model.masked_faces = set(self._previous_masked_faces)
        touched = current.union(self.mesh_model.masked_faces)
        return sorted(touched)


class CommandManager:
    def __init__(self) -> None:
        self._undo_stack: list[Command] = []
        self._redo_stack: list[Command] = []

    def execute(self, command: Command) -> list[int]:
        touched = command.apply()
        self._undo_stack.append(command)
        self._redo_stack.clear()
        return touched

    def undo(self) -> list[int]:
        if not self._undo_stack:
            return []
        command = self._undo_stack.pop()
        touched = command.undo()
        self._redo_stack.append(command)
        return touched

    def redo(self) -> list[int]:
        if not self._redo_stack:
            return []
        command = self._redo_stack.pop()
        touched = command.apply()
        self._undo_stack.append(command)
        return touched

    def can_undo(self) -> bool:
        return bool(self._undo_stack)

    def can_redo(self) -> bool:
        return bool(self._redo_stack)


class AppCommands:
    def __init__(self, mesh_model: MeshModel | None, paint_tool: PaintTool | None) -> None:
        self.mesh_model = mesh_model
        self.paint_tool = paint_tool
        self.history = CommandManager()

    def attach(self, mesh_model: MeshModel, paint_tool: PaintTool) -> None:
        self.mesh_model = mesh_model
        self.paint_tool = paint_tool
        self.history = CommandManager()

    def paint_faces(self, updates: dict[int, Color], description: str = "Paint faces") -> list[int]:
        if self.mesh_model is None:
            return []
        return self.history.execute(
            PaintFacesCommand(mesh_model=self.mesh_model, updates=updates, description=description)
        )

    def add_sketch_entity(self, document: SketchDocument, entity: SketchEntity) -> list[int]:
        return self.history.execute(SketchEntityCommand(document=document, entity=entity))

    def update_sketch_entity(
        self, document: SketchDocument, entity_id: str, new_data: dict[str, object]
    ) -> list[int]:
        return self.history.execute(
            UpdateSketchEntityCommand(document=document, entity_id=entity_id, new_data=new_data)
        )

    def delete_sketch_entity(self, document: SketchDocument, entity_id: str) -> list[int]:
        return self.history.execute(DeleteSketchEntityCommand(document=document, entity_id=entity_id))

    def set_masked_faces(
        self,
        face_ids: set[int],
        description: str = "Update masked faces",
    ) -> list[int]:
        if self.mesh_model is None:
            return []
        return self.history.execute(
            SetMaskedFacesCommand(
                mesh_model=self.mesh_model,
                new_masked_faces={int(face_id) for face_id in face_ids},
                description=description,
            )
        )

    def undo(self) -> list[int]:
        return self.history.undo()

    def redo(self) -> list[int]:
        return self.history.redo()

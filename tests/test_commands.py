from __future__ import annotations

from stl_painter.commands import AppCommands
from stl_painter.paint_tool import PaintTool
from stl_painter.sketch_tool import SketchTool


def test_command_history_undo_redo_paint(square_mesh) -> None:
    commands = AppCommands(square_mesh, PaintTool(square_mesh))

    touched = commands.paint_faces({0: (255, 0, 0, 255)})
    assert touched == [0]
    assert square_mesh.face_colour(0) == (255, 0, 0, 255)

    commands.undo()
    assert square_mesh.face_colour(0) == square_mesh.default_colour

    commands.redo()
    assert square_mesh.face_colour(0) == (255, 0, 0, 255)


def test_command_history_sketch_entity(square_mesh) -> None:
    tool = SketchTool()
    document = tool.create_plane_from_face(square_mesh, 0)
    commands = AppCommands(square_mesh, PaintTool(square_mesh))
    entity = tool.create_entity(
        "line",
        start_uv=tool.world_to_plane(document.plane, square_mesh.face_vertices(0)[0]),
        end_uv=tool.world_to_plane(document.plane, square_mesh.face_vertices(0)[1]),
        colour=(0, 0, 255, 255),
    )

    commands.add_sketch_entity(document, entity)
    assert len(document.entities) == 1

    commands.undo()
    assert document.entities == []


def test_command_history_mask_set_undo_redo(square_mesh) -> None:
    commands = AppCommands(square_mesh, PaintTool(square_mesh))

    touched = commands.set_masked_faces({0}, description="Mask one face")
    assert touched == [0]
    assert square_mesh.masked_faces == {0}

    commands.undo()
    assert square_mesh.masked_faces == set()

    commands.redo()
    assert square_mesh.masked_faces == {0}


def test_timeline_jump_and_export(square_mesh) -> None:
    commands = AppCommands(square_mesh, PaintTool(square_mesh))
    commands.paint_faces({0: (255, 0, 0, 255)}, description="First")
    commands.paint_faces({1: (0, 255, 0, 255)}, description="Second")
    commands.jump_to_timeline_index(1)
    assert square_mesh.face_colour(0) == (255, 0, 0, 255)
    assert square_mesh.face_colour(1) == square_mesh.default_colour
    payload = commands.export_timeline()
    assert payload["current_index"] == 1
    assert len(payload["descriptions"]) == 3

from __future__ import annotations

import numpy as np

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


def test_command_manager_undo_redo(square_mesh) -> None:
    from stl_painter.commands import CommandManager, PaintFacesCommand

    manager = CommandManager()

    cmd = PaintFacesCommand(mesh_model=square_mesh, updates={0: (255, 0, 0, 255)})
    manager.execute(cmd)
    assert square_mesh.face_colour(0) == (255, 0, 0, 255)

    manager.undo()
    assert square_mesh.face_colour(0) == square_mesh.default_colour

    manager.redo()
    assert square_mesh.face_colour(0) == (255, 0, 0, 255)


def test_command_manager_can_undo_redo(square_mesh) -> None:
    from stl_painter.commands import CommandManager, PaintFacesCommand

    manager = CommandManager()

    assert manager.can_undo() is False
    assert manager.can_redo() is False

    cmd = PaintFacesCommand(mesh_model=square_mesh, updates={0: (255, 0, 0, 255)})
    manager.execute(cmd)

    assert manager.can_undo() is True
    assert manager.can_redo() is False


def test_command_manager_jump(square_mesh) -> None:
    from stl_painter.commands import CommandManager, PaintFacesCommand

    manager = CommandManager()

    manager.execute(
        PaintFacesCommand(
            mesh_model=square_mesh, updates={0: (255, 0, 0, 255)}, description="First"
        )
    )
    manager.execute(
        PaintFacesCommand(
            mesh_model=square_mesh, updates={1: (0, 255, 0, 255)}, description="Second"
        )
    )
    manager.jump_to(0)


def test_command_manager_timeline_descriptions(square_mesh) -> None:
    from stl_painter.commands import CommandManager, PaintFacesCommand

    manager = CommandManager()

    manager.execute(
        PaintFacesCommand(
            mesh_model=square_mesh,
            updates={0: (255, 0, 0, 255)},
            description="Paint red",
        )
    )
    manager.execute(
        PaintFacesCommand(
            mesh_model=square_mesh,
            updates={1: (0, 255, 0, 255)},
            description="Paint green",
        )
    )

    descs = manager.timeline_descriptions()
    assert "Paint red" in descs
    assert "Paint green" in descs


def test_app_commands_attach(square_mesh) -> None:
    from stl_painter.commands import AppCommands

    commands = AppCommands(None, None)
    commands.attach(square_mesh, PaintTool(square_mesh))
    assert commands.mesh_model is not None
    assert commands.paint_tool is not None


def test_app_commands_undo_redo(square_mesh) -> None:
    from stl_painter.commands import AppCommands

    commands = AppCommands(square_mesh, PaintTool(square_mesh))
    commands.paint_faces({0: (255, 0, 0, 255)}, description="Paint red")
    commands.undo()
    assert square_mesh.face_colour(0) == square_mesh.default_colour
    commands.redo()
    assert square_mesh.face_colour(0) == (255, 0, 0, 255)


def test_delete_sketch_entity_command(square_mesh) -> None:
    from stl_painter.commands import AppCommands
    from stl_painter.sketch_tool import SketchTool
    import numpy as np

    tool = SketchTool()
    document = tool.create_plane_from_face(square_mesh, 0)
    entity = tool.create_entity(
        "rect",
        np.asarray([0.0, 0.0], dtype=np.float32),
        np.asarray([1.0, 1.0], dtype=np.float32),
        (255, 0, 0, 255),
    )
    commands = AppCommands(square_mesh, PaintTool(square_mesh))
    commands.add_sketch_entity(document, entity)
    assert len(document.entities) == 1

    commands.delete_sketch_entity(document, entity.entity_id)
    assert len(document.entities) == 0

    commands.undo()
    assert len(document.entities) == 1

    commands.delete_sketch_entity(document, entity.entity_id)
    assert len(document.entities) == 0

    commands.undo()
    assert len(document.entities) == 1


def test_update_sketch_entity_command(square_mesh) -> None:
    from stl_painter.commands import AppCommands
    from stl_painter.sketch_tool import SketchTool

    tool = SketchTool()
    document = tool.create_plane_from_face(square_mesh, 0)
    entity = tool.create_entity(
        "rect",
        np.asarray([0.0, 0.0], dtype=np.float32),
        np.asarray([1.0, 1.0], dtype=np.float32),
        (255, 0, 0, 255),
    )
    commands = AppCommands(square_mesh, PaintTool(square_mesh))
    commands.add_sketch_entity(document, entity)

    new_data = {
        "min": [0.0, 0.0],
        "max": [2.0, 2.0],
        "colour": [0, 255, 0, 255],
        "stroke_width": 0.02,
    }
    commands.update_sketch_entity(document, entity.entity_id, new_data)

    commands.undo()
    assert document.entities[0].data["max"] == [1.0, 1.0]

    commands.redo()
    assert document.entities[0].data["max"] == [2.0, 2.0]


def test_app_commands_timeline_index(square_mesh) -> None:
    from stl_painter.commands import AppCommands

    commands = AppCommands(square_mesh, PaintTool(square_mesh))
    assert commands.timeline_index() == 0

    commands.paint_faces({0: (255, 0, 0, 255)})
    assert commands.timeline_index() == 1

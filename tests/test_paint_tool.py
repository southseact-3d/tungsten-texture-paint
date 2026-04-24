from __future__ import annotations

import numpy as np

from stl_painter.paint_tool import PaintTool


def test_paint_face_and_undo(square_mesh) -> None:
    tool = PaintTool(square_mesh)
    changed = tool.paint_face(0, (255, 0, 0, 255))
    assert changed == [0]
    assert square_mesh.face_colour(0) == (255, 0, 0, 255)

    restored = tool.undo()
    assert restored == [0]
    assert square_mesh.face_colour(0) == square_mesh.default_colour


def test_flood_fill_updates_connected_region(square_mesh) -> None:
    tool = PaintTool(square_mesh)
    changed = tool.flood_fill(0, (0, 255, 0, 255))
    assert sorted(changed) == [0, 1]
    assert square_mesh.face_colour(0) == (0, 255, 0, 255)
    assert square_mesh.face_colour(1) == (0, 255, 0, 255)


def test_brush_updates_always_paints_center_face(square_mesh) -> None:
    tool = PaintTool(square_mesh)

    updates = tool.brush_updates(
        0,
        np.asarray(square_mesh.face_center(0), dtype=np.float32),
        (255, 0, 0, 255),
        radius=0.01,
        opacity=0.1,
        falloff="smooth",
    )

    assert updates[0] == (255, 0, 0, 255)


def test_paint_faces_batch(square_mesh) -> None:
    tool = PaintTool(square_mesh)
    updates = {0: (255, 0, 0, 255), 1: (0, 255, 0, 255)}
    changed = tool.paint_faces(updates)
    assert len(changed) == 2
    assert square_mesh.face_colour(0) == (255, 0, 0, 255)
    assert square_mesh.face_colour(1) == (0, 255, 0, 255)


def test_paint_faces_ignores_unchanged(square_mesh) -> None:
    tool = PaintTool(square_mesh)
    default_colour = square_mesh.default_colour
    updates = {0: default_colour}
    changed = tool.paint_faces(updates)
    assert changed == []


def test_flood_fill_updates_same_colour_returns_empty(square_mesh) -> None:
    tool = PaintTool(square_mesh)
    current_colour = square_mesh.face_colour(0)
    updates = tool.flood_fill_updates(0, current_colour)
    assert updates == {}


def test_flood_fill_respects_mask(square_mesh) -> None:
    tool = PaintTool(square_mesh)
    square_mesh.masked_faces.add(1)
    updates = tool.flood_fill_updates(0, (255, 0, 0, 255))
    assert 1 not in updates


def test_sample_colour(square_mesh) -> None:
    tool = PaintTool(square_mesh)
    colour = tool.sample_colour(0)
    assert colour == square_mesh.default_colour


def test_mask_faces(square_mesh) -> None:
    tool = PaintTool(square_mesh)
    tool.mask_faces({0, 1})
    assert 0 in square_mesh.masked_faces
    assert 1 in square_mesh.masked_faces


def test_unmask_faces(square_mesh) -> None:
    tool = PaintTool(square_mesh)
    square_mesh.masked_faces.add(0)
    square_mesh.masked_faces.add(1)
    tool.unmask_faces({0})
    assert 0 not in square_mesh.masked_faces
    assert 1 in square_mesh.masked_faces


def test_brush_paint_with_erase(square_mesh) -> None:
    tool = PaintTool(square_mesh)
    square_mesh.set_face_colour(0, (255, 0, 0, 255))
    hit_point = square_mesh.face_center(0)
    changed = tool.brush_paint(
        0,
        hit_point,
        (0, 0, 0, 0),
        radius=0.5,
        opacity=1.0,
        erase=True,
    )
    assert len(changed) > 0


def test_brush_paint_with_front_faces_only(square_mesh) -> None:
    tool = PaintTool(square_mesh)
    hit_point = square_mesh.face_center(0)
    changed = tool.brush_paint(
        0,
        hit_point,
        (255, 0, 0, 255),
        radius=0.5,
        opacity=1.0,
        front_faces_only=True,
    )
    assert len(changed) > 0


def test_brush_updates_ignores_masked_faces(square_mesh) -> None:
    tool = PaintTool(square_mesh)
    square_mesh.masked_faces.add(0)
    hit_point = square_mesh.face_center(0)
    updates = tool.brush_updates(
        0,
        hit_point,
        (255, 0, 0, 255),
        radius=0.5,
        opacity=1.0,
    )
    assert 0 not in updates


def test_undo_stack_max_size(square_mesh) -> None:
    from stl_painter.paint_tool import UndoStack

    stack = UndoStack(max_size=3)
    tool = PaintTool(square_mesh, stack)
    tool.paint_face(0, (255, 0, 0, 255))
    tool.paint_face(0, (0, 255, 0, 255))
    tool.paint_face(0, (0, 0, 255, 255))
    tool.paint_face(0, (255, 255, 0, 255))
    assert len(stack.stack) == 3

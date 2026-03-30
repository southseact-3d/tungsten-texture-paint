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

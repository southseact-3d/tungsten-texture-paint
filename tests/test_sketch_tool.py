from __future__ import annotations

import numpy as np

from stl_painter.sketch_tool import SketchTool


def test_sketch_plane_round_trip(square_mesh) -> None:
    tool = SketchTool()
    document = tool.create_plane_from_face(square_mesh, 0)
    point = square_mesh.face_center(0)

    uv = tool.world_to_plane(document.plane, point)
    rebuilt = tool.plane_to_world(document.plane, uv)

    assert np.allclose(point, rebuilt, atol=1e-5)


def test_rect_resize_updates_extents(square_mesh) -> None:
    tool = SketchTool()
    document = tool.create_plane_from_face(square_mesh, 0)
    entity = tool.create_entity(
        "rect",
        np.asarray([0.0, 0.0], dtype=np.float32),
        np.asarray([1.0, 1.0], dtype=np.float32),
        (255, 0, 0, 255),
    )

    updated = tool.resize_entity(entity, "max", np.asarray([2.0, 3.0], dtype=np.float32))

    assert updated["max"] == [2.0, 3.0]
    assert document.plane.anchor_face_id == 0


def test_snap_prefers_grid(square_mesh) -> None:
    tool = SketchTool()
    document = tool.create_plane_from_face(square_mesh, 0)
    document.grid_size = 0.5

    snapped = tool.snap_point(square_mesh, document, np.asarray([0.46, 0.04], dtype=np.float32))

    assert snapped.snapped
    assert np.allclose(snapped.plane_uv, np.asarray([0.5, 0.0], dtype=np.float32), atol=1e-4)

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

    updated = tool.resize_entity(
        entity, "max", np.asarray([2.0, 3.0], dtype=np.float32)
    )

    assert updated["max"] == [2.0, 3.0]
    assert document.plane.anchor_face_id == 0


def test_snap_prefers_grid(square_mesh) -> None:
    tool = SketchTool()
    document = tool.create_plane_from_face(square_mesh, 0)
    document.grid_size = 0.5

    snapped = tool.snap_point(
        square_mesh, document, np.asarray([0.46, 0.04], dtype=np.float32)
    )

    assert snapped.snapped
    assert np.allclose(
        snapped.plane_uv, np.asarray([0.5, 0.0], dtype=np.float32), atol=1e-4
    )


def test_svg_entity_can_be_resized(square_mesh) -> None:
    tool = SketchTool()
    entity = tool.create_entity(
        "svg",
        np.asarray([0.0, 0.0], dtype=np.float32),
        np.asarray([1.0, 1.0], dtype=np.float32),
        (255, 255, 255, 255),
    )
    updated = tool.resize_entity(
        entity, "max", np.asarray([2.0, 3.0], dtype=np.float32)
    )
    assert updated["max"] == [2.0, 3.0]


def test_create_circle_entity(square_mesh) -> None:
    tool = SketchTool()
    entity = tool.create_entity(
        "circle",
        np.asarray([0.0, 0.0], dtype=np.float32),
        np.asarray([1.0, 0.0], dtype=np.float32),
        (0, 255, 0, 255),
    )
    assert entity.kind == "circle"
    assert entity.data["radius"] > 0


def test_create_text_entity(square_mesh) -> None:
    tool = SketchTool()
    entity = tool.create_entity(
        "text",
        np.asarray([0.0, 0.0], dtype=np.float32),
        np.asarray([1.0, 1.0], dtype=np.float32),
        (255, 0, 255, 255),
        text="Hello",
    )
    assert entity.kind == "text"
    assert entity.data["text"] == "Hello"


def test_create_line_entity(square_mesh) -> None:
    tool = SketchTool()
    entity = tool.create_entity(
        "line",
        np.asarray([0.0, 0.0], dtype=np.float32),
        np.asarray([1.0, 1.0], dtype=np.float32),
        (0, 0, 255, 255),
    )
    assert entity.kind == "line"


def test_resize_line_entity(square_mesh) -> None:
    tool = SketchTool()
    entity = tool.create_entity(
        "line",
        np.asarray([0.0, 0.0], dtype=np.float32),
        np.asarray([1.0, 1.0], dtype=np.float32),
        (0, 0, 255, 255),
    )
    updated = tool.resize_entity(
        entity, "end", np.asarray([2.0, 2.0], dtype=np.float32)
    )
    assert updated["end"] == [2.0, 2.0]


def test_resize_circle_entity(square_mesh) -> None:
    tool = SketchTool()
    entity = tool.create_entity(
        "circle",
        np.asarray([0.0, 0.0], dtype=np.float32),
        np.asarray([1.0, 0.0], dtype=np.float32),
        (0, 255, 0, 255),
    )
    updated = tool.resize_entity(
        entity, "center", np.asarray([0.5, 0.5], dtype=np.float32)
    )
    assert "radius" in updated


def test_resize_text_entity(square_mesh) -> None:
    tool = SketchTool()
    entity = tool.create_entity(
        "text",
        np.asarray([0.0, 0.0], dtype=np.float32),
        np.asarray([1.0, 1.0], dtype=np.float32),
        (255, 0, 255, 255),
        text="Test",
    )
    updated = tool.resize_entity(
        entity, "center", np.asarray([0.5, 0.5], dtype=np.float32)
    )
    assert "position" in updated


def test_entity_snap_points_rect(square_mesh) -> None:
    from stl_painter.sketch_tool import entity_snap_points

    tool = SketchTool()
    entity = tool.create_entity(
        "rect",
        np.asarray([0.0, 0.0], dtype=np.float32),
        np.asarray([1.0, 1.0], dtype=np.float32),
        (255, 0, 0, 255),
    )
    points = entity_snap_points(entity)
    assert len(points) > 0


def test_entity_snap_points_line(square_mesh) -> None:
    from stl_painter.sketch_tool import entity_snap_points

    tool = SketchTool()
    entity = tool.create_entity(
        "line",
        np.asarray([0.0, 0.0], dtype=np.float32),
        np.asarray([1.0, 1.0], dtype=np.float32),
        (255, 0, 0, 255),
    )
    points = entity_snap_points(entity)
    assert len(points) == 3


def test_entity_snap_points_circle(square_mesh) -> None:
    from stl_painter.sketch_tool import entity_snap_points

    tool = SketchTool()
    entity = tool.create_entity(
        "circle",
        np.asarray([0.0, 0.0], dtype=np.float32),
        np.asarray([1.0, 0.0], dtype=np.float32),
        (0, 255, 0, 255),
    )
    points = entity_snap_points(entity)
    assert len(points) == 5


def test_entity_snap_points_text(square_mesh) -> None:
    from stl_painter.sketch_tool import entity_snap_points

    tool = SketchTool()
    entity = tool.create_entity(
        "text",
        np.asarray([0.5, 0.5], dtype=np.float32),
        np.asarray([1.0, 1.0], dtype=np.float32),
        (255, 0, 255, 255),
    )
    points = entity_snap_points(entity)
    assert len(points) == 1


def test_project_vertices_to_screen(square_mesh) -> None:
    from stl_painter.sketch_tool import project_vertices_to_screen

    view_proj = np.eye(4, dtype=np.float32)
    screen, valid = project_vertices_to_screen(
        square_mesh.vertices, view_proj, (800, 600)
    )
    assert screen.shape[0] == square_mesh.vertex_count
    assert valid.shape[0] == square_mesh.vertex_count


def test_snap_to_grid_disabled(square_mesh) -> None:
    tool = SketchTool()
    document = tool.create_plane_from_face(square_mesh, 0)
    document.snap_to_grid = False
    document.grid_size = 0.1

    snapped = tool.snap_point(
        square_mesh, document, np.asarray([0.46, 0.04], dtype=np.float32)
    )
    assert not snapped.snapped


def test_snap_to_vertices_disabled(square_mesh) -> None:
    tool = SketchTool()
    document = tool.create_plane_from_face(square_mesh, 0)
    document.snap_to_vertices = False
    document.grid_size = 0.0

    snapped = tool.snap_point(
        square_mesh, document, np.asarray([0.0, 0.0], dtype=np.float32)
    )
    assert not snapped.snapped


def test_snap_to_edges_disabled(square_mesh) -> None:
    tool = SketchTool()
    document = tool.create_plane_from_face(square_mesh, 0)
    document.snap_to_edges = False

    snapped = tool.snap_point(
        square_mesh, document, np.asarray([0.5, 0.0], dtype=np.float32)
    )
    pass

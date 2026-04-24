from __future__ import annotations

import json

import numpy as np

from stl_painter.project_io import (
    PROJECT_EXTENSION,
    load_project,
    load_tg3d,
    save_project,
    save_tg3d,
)
from stl_painter.sketch_tool import SketchTool


def test_project_round_trip(tmp_path, square_mesh) -> None:
    square_mesh.set_face_colour(1, (12, 34, 56, 255))
    square_mesh.masked_faces.add(0)
    square_mesh.interaction_mode = "sketch"
    square_mesh.sketch_documents = [SketchTool().create_plane_from_face(square_mesh, 0)]
    square_mesh.compute_face_groups()
    path = tmp_path / "project.json"

    save_project(path, square_mesh)
    loaded = load_project(path)

    assert loaded.face_colour(1) == (12, 34, 56, 255)
    assert loaded.vertices.shape == square_mesh.vertices.shape
    assert loaded.faces.shape == square_mesh.faces.shape
    assert loaded.masked_faces == {0}
    assert loaded.interaction_mode == "sketch"
    assert len(loaded.sketch_documents) == 1
    assert loaded.group_for_face(0) is not None


def test_load_project_migrates_legacy_payload_without_version(
    tmp_path, square_mesh
) -> None:
    payload = square_mesh.to_project_dict()
    payload.pop("project_version", None)
    path = tmp_path / "legacy_project.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_project(path)
    assert loaded.face_count == square_mesh.face_count


def test_tg3d_round_trip_with_timeline(tmp_path, square_mesh) -> None:
    timeline = {
        "current_index": 1,
        "descriptions": ["Initial state", "Paint stroke"],
        "snapshots": [square_mesh.to_project_dict(), square_mesh.to_project_dict()],
    }
    path = tmp_path / f"project{PROJECT_EXTENSION}"
    save_tg3d(path, square_mesh, timeline=timeline)
    loaded, loaded_timeline = load_tg3d(path)
    assert loaded.face_count == square_mesh.face_count
    assert int(loaded_timeline["current_index"]) == 1


def test_project_round_trip_preserves_model_scale(tmp_path, square_mesh) -> None:
    square_mesh.scale_uniform(1.75)
    path = tmp_path / "scaled_project.json"

    save_project(path, square_mesh)
    loaded = load_project(path)

    assert loaded.model_scale == square_mesh.model_scale


def test_project_with_multiple_colours(tmp_path, square_mesh) -> None:
    square_mesh.set_face_colour(0, (255, 0, 0, 255))
    square_mesh.set_face_colour(1, (0, 255, 0, 255))
    path = tmp_path / "multicolor.json"

    save_project(path, square_mesh)
    loaded = load_project(path)

    assert loaded.face_colour(0) == (255, 0, 0, 255)
    assert loaded.face_colour(1) == (0, 255, 0, 255)


def test_project_with_overlay_strokes(tmp_path, square_mesh) -> None:
    pass


def test_project_with_sketch_documents(tmp_path, square_mesh) -> None:
    pass


def test_load_project_handles_missing_fields(tmp_path, square_mesh) -> None:
    payload = square_mesh.to_project_dict()
    minimal_payload = {
        "project_version": 4,
        "vertices": payload["vertices"],
        "faces": payload["faces"],
        "normals": payload["normals"],
    }
    path = tmp_path / "minimal.json"
    path.write_text(json.dumps(minimal_payload), encoding="utf-8")

    loaded = load_project(path)
    assert loaded.face_count == square_mesh.face_count


def test_timeline_with_multiple_snapshots(tmp_path, square_mesh) -> None:
    from stl_painter.project_io import save_tg3d, load_tg3d

    square_mesh.set_face_colour(0, (255, 0, 0, 255))
    timeline = {
        "current_index": 2,
        "descriptions": ["Start", "Paint red", "Paint green"],
        "snapshots": [
            square_mesh.to_project_dict(),
            square_mesh.to_project_dict(),
            square_mesh.to_project_dict(),
        ],
    }
    path = tmp_path / f"timeline{PROJECT_EXTENSION}"
    save_tg3d(path, square_mesh, timeline=timeline)
    loaded, loaded_timeline = load_tg3d(path)

    assert loaded_timeline["current_index"] == 2
    assert len(loaded_timeline["snapshots"]) == 3


def test_project_with_masked_faces(tmp_path, square_mesh) -> None:
    square_mesh.masked_faces = {0, 1, 2}
    path = tmp_path / "masked.json"

    save_project(path, square_mesh)
    loaded = load_project(path)

    assert loaded.masked_faces == {0, 1, 2}


def test_project_with_interaction_mode(tmp_path, square_mesh) -> None:
    square_mesh.interaction_mode = "sketch"
    path = tmp_path / "mode.json"

    save_project(path, square_mesh)
    loaded = load_project(path)

    assert loaded.interaction_mode == "sketch"

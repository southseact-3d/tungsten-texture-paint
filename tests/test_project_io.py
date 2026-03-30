from __future__ import annotations

import json

from stl_painter.sketch_tool import SketchTool
from stl_painter.project_io import PROJECT_EXTENSION, load_project, load_tg3d, save_project, save_tg3d


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


def test_load_project_migrates_legacy_payload_without_version(tmp_path, square_mesh) -> None:
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

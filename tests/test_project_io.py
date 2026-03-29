from __future__ import annotations

import json

from stl_painter.sketch_tool import SketchTool
from stl_painter.project_io import load_project, save_project


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

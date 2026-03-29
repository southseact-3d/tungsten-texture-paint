from __future__ import annotations

from stl_painter.sketch_tool import SketchTool
from stl_painter.project_io import load_project, save_project


def test_project_round_trip(tmp_path, square_mesh) -> None:
    square_mesh.set_face_colour(1, (12, 34, 56, 255))
    square_mesh.masked_faces.add(0)
    square_mesh.interaction_mode = "sketch"
    square_mesh.sketch_documents = [SketchTool().create_plane_from_face(square_mesh, 0)]
    path = tmp_path / "project.json"

    save_project(path, square_mesh)
    loaded = load_project(path)

    assert loaded.face_colour(1) == (12, 34, 56, 255)
    assert loaded.vertices.shape == square_mesh.vertices.shape
    assert loaded.faces.shape == square_mesh.faces.shape
    assert loaded.masked_faces == {0}
    assert loaded.interaction_mode == "sketch"
    assert len(loaded.sketch_documents) == 1

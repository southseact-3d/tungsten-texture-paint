from __future__ import annotations

import numpy as np
import trimesh

from stl_painter.importer import load_model
from stl_painter.mesh_model import MeshModel


def test_load_model_obj_reads_mesh(tmp_path) -> None:
    mesh = trimesh.creation.box(extents=(1.0, 2.0, 3.0))
    path = tmp_path / "box.obj"
    mesh.export(path)

    loaded = load_model(path)

    assert loaded.face_count > 0
    assert loaded.vertex_count > 0
    assert loaded.source_path == str(path)


def test_map_colours_from_handles_scaled_variant(square_mesh) -> None:
    source = square_mesh
    source.set_face_colour(0, (255, 0, 0, 255))
    source.set_face_colour(1, (0, 255, 0, 255))

    scaled = MeshModel(
        vertices=(source.vertices * 2.5).astype(np.float32),
        faces=source.faces.copy(),
        normals=source.normals.copy(),
    )
    mapped = scaled.map_colours_from(source, normalize_scale=True)

    assert mapped == scaled.face_count
    assert scaled.face_colour(0) == (255, 0, 0, 255)
    assert scaled.face_colour(1) == (0, 255, 0, 255)

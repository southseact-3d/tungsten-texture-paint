from __future__ import annotations

import numpy as np
import pytest
import trimesh

from stl_painter.mesh_model import MeshModel


def test_from_trimesh_generates_valid_normals_for_basic_mesh() -> None:
    mesh = trimesh.Trimesh(
        vertices=np.asarray(
            [
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
            ],
            dtype=np.float32,
        ),
        faces=np.asarray([(0, 1, 2)], dtype=np.int32),
        process=False,
    )

    model = MeshModel.from_trimesh(mesh)

    assert model.face_count == 1
    assert model.normals.shape == (1, 3)
    assert np.isfinite(model.normals).all()
    assert pytest.approx(np.linalg.norm(model.normals[0]), rel=1e-4) == 1.0


def test_mesh_model_rejects_empty_face_list() -> None:
    with pytest.raises(ValueError, match="does not contain any faces"):
        MeshModel(
            vertices=np.asarray([(0.0, 0.0, 0.0)], dtype=np.float32),
            faces=np.empty((0, 3), dtype=np.int32),
            normals=np.empty((0, 3), dtype=np.float32),
        )


def test_compute_face_groups_returns_connected_groups(square_mesh) -> None:
    groups = square_mesh.compute_face_groups()
    assert len(groups) == 1
    group_id = square_mesh.group_for_face(0)
    assert group_id is not None
    assert square_mesh.faces_for_group(group_id) == [0, 1]

from __future__ import annotations

import numpy as np
import pytest

from stl_painter.mesh_model import MeshModel


@pytest.fixture()
def square_mesh() -> MeshModel:
    vertices = np.asarray(
        [
            (-0.5, -0.5, 0.0),
            (0.5, -0.5, 0.0),
            (0.5, 0.5, 0.0),
            (-0.5, 0.5, 0.0),
        ],
        dtype=np.float32,
    )
    faces = np.asarray([(0, 1, 2), (0, 2, 3)], dtype=np.int32)
    normals = np.asarray([(0.0, 0.0, 1.0), (0.0, 0.0, 1.0)], dtype=np.float32)
    return MeshModel(vertices=vertices, faces=faces, normals=normals)

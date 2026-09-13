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


def test_mesh_model_face_colour_default() -> None:
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    model = MeshModel.from_trimesh(mesh)
    colour = model.face_colour(0)
    assert colour[0] >= 0 and colour[1] >= 0 and colour[2] >= 0
    assert len(colour) == 4


def test_mesh_model_set_face_colour(square_mesh) -> None:
    square_mesh.set_face_colour(0, (255, 128, 64, 200))
    assert square_mesh.face_colour(0) == (255, 128, 64, 200)


def test_mesh_model_adjacency_map(square_mesh) -> None:
    adj = square_mesh.adjacency_map()
    assert 0 in adj
    assert 1 in adj
    assert 1 in adj[0] or 0 in adj[1]


def test_mesh_model_face_vertices(square_mesh) -> None:
    verts = square_mesh.face_vertices(0)
    assert verts.shape == (3, 3)


def test_mesh_model_face_center(square_mesh) -> None:
    center = square_mesh.face_center(0)
    assert center.shape == (3,)
    assert center[2] == 0.0


def test_mesh_model_expanded_arrays(square_mesh) -> None:
    positions, normals, colours = square_mesh.expanded_arrays()
    assert positions.shape[0] == 6
    assert normals.shape[0] == 6
    assert colours.shape[0] == 6
    assert colours.shape[1] == 4


def test_mesh_model_mesh_extents(square_mesh) -> None:
    extents = square_mesh.mesh_extents()
    assert extents.shape == (3,)
    assert extents[0] == 1.0
    assert extents[1] == 1.0


def test_mesh_model_mesh_diagonal(square_mesh) -> None:
    diagonal = square_mesh.mesh_diagonal()
    assert diagonal > 0.0


def test_mesh_model_scale_uniform(square_mesh) -> None:
    original_extent = square_mesh.mesh_extents()[0]
    square_mesh.scale_uniform(2.0)
    assert square_mesh.mesh_extents()[0] == pytest.approx(original_extent * 2.0)
    assert square_mesh.model_scale == 2.0


def test_mesh_model_scale_uniform_rejects_zero(square_mesh) -> None:
    with pytest.raises(ValueError, match="must be greater than zero"):
        square_mesh.scale_uniform(0.0)


def test_mesh_model_scale_uniform_rejects_negative(square_mesh) -> None:
    with pytest.raises(ValueError, match="must be greater than zero"):
        square_mesh.scale_uniform(-1.0)


def test_mesh_model_get_internal_group_edges(square_mesh) -> None:
    square_mesh.compute_face_groups()
    edges = square_mesh.get_internal_group_edges()
    assert isinstance(edges, set)


def test_mesh_model_rejects_invalid_vertex_shape() -> None:
    with pytest.raises(ValueError, match="Expected vertices shaped"):
        MeshModel(
            vertices=np.asarray([(0.0, 0.0), (1.0, 0.0)], dtype=np.float32),
            faces=np.asarray([[0, 1, 2]], dtype=np.int32),
            normals=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
        )


def test_mesh_model_rejects_invalid_face_shape() -> None:
    with pytest.raises(ValueError, match="Expected triangular faces"):
        MeshModel(
            vertices=np.asarray(
                [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)], dtype=np.float32
            ),
            faces=np.asarray([[0, 1]], dtype=np.int32),
            normals=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
        )


def test_mesh_model_rejects_invalid_normals_shape() -> None:
    with pytest.raises(ValueError, match="Expected normals shaped"):
        MeshModel(
            vertices=np.asarray(
                [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)], dtype=np.float32
            ),
            faces=np.asarray([[0, 1, 2]], dtype=np.int32),
            normals=np.asarray(
                [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32
            ),
        )


def test_mesh_model_sketch_plane_serialization() -> None:
    from stl_painter.mesh_model import SketchPlane

    plane = SketchPlane(
        origin=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        normal=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        tangent_u=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        tangent_v=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        anchor_face_id=0,
    )
    data = plane.to_dict()
    restored = SketchPlane.from_dict(data)
    assert np.allclose(restored.origin, plane.origin)
    assert np.allclose(restored.normal, plane.normal)
    assert restored.anchor_face_id == 0


def test_mesh_model_sketch_entity_serialization(square_mesh) -> None:
    from stl_painter.mesh_model import SketchEntity

    entity = SketchEntity(
        entity_id="test123",
        kind="rect",
        data={"min": [0.0, 0.0], "max": [1.0, 1.0], "colour": [255, 0, 0, 255]},
    )
    data = entity.to_dict()
    restored = SketchEntity.from_dict(data)
    assert restored.entity_id == "test123"
    assert restored.kind == "rect"


def test_mesh_model_sketch_document_serialization(square_mesh) -> None:
    from stl_painter.mesh_model import SketchDocument, SketchPlane
    from stl_painter.sketch_tool import SketchTool

    tool = SketchTool()
    document = tool.create_plane_from_face(square_mesh, 0)
    entity = tool.create_entity(
        "rect",
        np.asarray([0.0, 0.0], dtype=np.float32),
        np.asarray([1.0, 1.0], dtype=np.float32),
        (255, 0, 0, 255),
    )
    document.entities.append(entity)
    data = document.to_dict()
    restored = SketchDocument.from_dict(data)
    assert len(restored.entities) == 1
    assert restored.plane.anchor_face_id == 0


def test_mesh_model_to_project_dict(square_mesh) -> None:
    from stl_painter.mesh_model import PROJECT_VERSION

    square_mesh.set_face_colour(0, (255, 0, 0, 255))
    data = square_mesh.to_project_dict()
    assert data["project_version"] == PROJECT_VERSION
    assert "vertices" in data
    assert "faces" in data
    assert "face_colours" in data


def test_project_dict_v4_without_cad_fields_loads(square_mesh) -> None:
    """Pre-STEP (v4) payloads without CAD keys still load with no CAD mapping."""
    data = square_mesh.to_project_dict()
    data["project_version"] = 4
    data.pop("tri_to_cad", None)
    data.pop("cad_faces", None)
    restored = MeshModel.from_project_dict(data)
    assert restored.face_count == square_mesh.face_count
    assert not restored.has_cad_faces


def test_mesh_model_from_project_dict(square_mesh) -> None:
    square_mesh.set_face_colour(0, (255, 0, 0, 255))
    data = square_mesh.to_project_dict()
    restored = MeshModel.from_project_dict(data)
    assert restored.face_count == square_mesh.face_count
    assert restored.face_colour(0) == (255, 0, 0, 255)


def test_mesh_model_to_project_delta(square_mesh) -> None:
    square_mesh.set_face_colour(0, (255, 0, 0, 255))
    delta = square_mesh.to_project_delta()
    assert "face_colours" in delta
    assert "default_colour" in delta


def test_mesh_model_from_project_delta(square_mesh) -> None:
    square_mesh.set_face_colour(0, (255, 0, 0, 255))
    delta = square_mesh.to_project_delta()
    restored = MeshModel.from_project_delta(square_mesh, delta)
    assert restored.face_colour(0) == (255, 0, 0, 255)

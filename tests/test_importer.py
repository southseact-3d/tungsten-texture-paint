from __future__ import annotations

import numpy as np
import pytest
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


def test_load_stl_binary_format(tmp_path) -> None:
    mesh = trimesh.creation.box(extents=(1.0, 2.0, 3.0))
    path = tmp_path / "box.stl"
    mesh.export(path)

    loaded = load_model(path)

    assert loaded.face_count > 0
    assert loaded.vertex_count > 0
    assert loaded.source_path == str(path)


def test_load_stl_ascii_format(tmp_path) -> None:
    mesh = trimesh.creation.box(extents=(1.0, 2.0, 3.0))
    path = tmp_path / "box_ascii.stl"
    mesh.export(path, file_type="stl_ascii")

    loaded = load_model(path)

    assert loaded.face_count > 0
    assert loaded.vertex_count > 0


def test_load_model_rejects_unsupported_extension(tmp_path) -> None:
    from stl_painter.importer import load_model

    path = tmp_path / "mesh.xyz"
    path.write_text("dummy")

    with pytest.raises(ValueError, match="Unsupported import format"):
        load_model(path)


def test_load_model_glb_format(tmp_path) -> None:
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    path = tmp_path / "box.glb"
    mesh.export(path)

    loaded = load_model(path)

    assert loaded.face_count > 0
    assert loaded.vertex_count > 0


def test_load_model_gltf_format(tmp_path) -> None:
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    path = tmp_path / "box.gltf"
    mesh.export(path)

    loaded = load_model(path)

    assert loaded.face_count > 0
    assert loaded.vertex_count > 0


def test_load_model_empty_file(tmp_path) -> None:
    from stl_painter.importer import load_model

    path = tmp_path / "empty.stl"
    path.touch()

    with pytest.raises(ValueError, match="did not contain any mesh"):
        load_model(path)


def test_import_stl_alias(tmp_path) -> None:
    from stl_painter.importer import import_stl

    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    path = tmp_path / "test.stl"
    mesh.export(path)

    loaded = import_stl(path)

    assert loaded.face_count > 0


def test_map_colours_empty_target(square_mesh) -> None:
    source = square_mesh
    source.set_face_colour(0, (255, 0, 0, 255))

    with pytest.raises(ValueError, match="does not contain any faces"):
        MeshModel(
            vertices=np.empty((0, 3), dtype=np.float32),
            faces=np.empty((0, 3), dtype=np.int32),
            normals=np.empty((0, 3), dtype=np.float32),
        )


def test_map_colours_without_normalize(square_mesh) -> None:
    source = square_mesh
    source.set_face_colour(0, (255, 0, 0, 255))

    scaled = MeshModel(
        vertices=(source.vertices * 2.5).astype(np.float32),
        faces=source.faces.copy(),
        normals=source.normals.copy(),
    )
    mapped = scaled.map_colours_from(source, normalize_scale=False)
    assert mapped >= 0

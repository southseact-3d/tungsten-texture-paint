from __future__ import annotations

import json

import numpy as np
import pytest

from stl_painter.importer import (
    STEP_IMPORT_EXTENSIONS,
    SUPPORTED_IMPORT_EXTENSIONS,
    load_model,
    load_step_npz,
)
from stl_painter.mesh_model import MeshModel, PROJECT_VERSION


def _two_quad_cad():
    """Two disjoint quads (4 triangles) as two CAD faces."""
    vertices = np.asarray(
        [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (2.0, 0.0, 0.0),
            (3.0, 0.0, 0.0),
            (3.0, 1.0, 0.0),
            (2.0, 1.0, 0.0),
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [(0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7)], dtype=np.int64
    )
    tri_to_cad = np.asarray([0, 0, 1, 1], dtype=np.int64)
    cad_meta = [
        {"id": "cad_s0_f0", "solid": 0, "name": "solid_0 / face_0",
         "surface": "PLANE", "count": 2},
        {"id": "cad_s0_f1", "solid": 0, "name": "solid_0 / face_1",
         "surface": "PLANE", "count": 2},
    ]
    return vertices, faces, tri_to_cad, cad_meta


def test_step_extensions_supported() -> None:
    assert ".step" in SUPPORTED_IMPORT_EXTENSIONS
    assert ".stp" in SUPPORTED_IMPORT_EXTENSIONS
    assert STEP_IMPORT_EXTENSIONS == {".step", ".stp"}


def test_from_cad_arrays_builds_groups() -> None:
    vertices, faces, tri_to_cad, cad_meta = _two_quad_cad()
    model = MeshModel.from_cad_arrays(vertices, faces, tri_to_cad, cad_meta)

    assert model.face_count == 4
    assert model.has_cad_faces
    assert model.cad_face_count == 2
    assert model.face_groups["cad_s0_f0"] == [0, 1]
    assert model.face_groups["cad_s0_f1"] == [2, 3]
    assert model.cad_id_for_face(0) == "cad_s0_f0"
    assert model.cad_id_for_face(3) == "cad_s0_f1"
    assert model.faces_for_cad("cad_s0_f0") == [0, 1]
    assert model.faces_for_cad("missing") == []


def test_cad_boundary_edges_hide_diagonals() -> None:
    vertices, faces, tri_to_cad, cad_meta = _two_quad_cad()
    model = MeshModel.from_cad_arrays(vertices, faces, tri_to_cad, cad_meta)

    boundary = model.cad_boundary_edges()
    # Two quads -> 8 outline edges; the 2 triangulation diagonals are hidden.
    assert len(boundary) == 8
    keys = {tuple(sorted(edge)) for edge in boundary.tolist()}
    assert (0, 2) not in keys  # first quad diagonal
    assert (4, 6) not in keys  # second quad diagonal

    face_edges = model.cad_face_boundary_edges("cad_s0_f0")
    assert len(face_edges) == 4


def test_from_cad_arrays_rejects_bad_mapping() -> None:
    vertices, faces, _, cad_meta = _two_quad_cad()
    with pytest.raises(ValueError, match="tri_to_cad"):
        MeshModel.from_cad_arrays(vertices, faces, [0, 0, 0], cad_meta)
    with pytest.raises(ValueError, match="unknown CAD face"):
        MeshModel.from_cad_arrays(vertices, faces, [0, 0, 5, 5], cad_meta)


def test_cad_project_roundtrip() -> None:
    vertices, faces, tri_to_cad, cad_meta = _two_quad_cad()
    model = MeshModel.from_cad_arrays(vertices, faces, tri_to_cad, cad_meta)
    model.set_face_colour(0, (255, 0, 0, 255))

    assert PROJECT_VERSION >= 5
    restored = MeshModel.from_project_dict(model.to_project_dict())

    assert restored.has_cad_faces
    assert restored.cad_face_count == 2
    assert list(restored.tri_to_cad) == [0, 0, 1, 1]
    assert restored.face_groups["cad_s0_f1"] == [2, 3]
    assert restored.face_colour(0) == (255, 0, 0, 255)
    assert len(restored.cad_boundary_edges()) == 8


def test_cad_project_delta_keeps_mapping() -> None:
    vertices, faces, tri_to_cad, cad_meta = _two_quad_cad()
    model = MeshModel.from_cad_arrays(vertices, faces, tri_to_cad, cad_meta)

    merged = MeshModel.from_project_delta(model, model.to_project_delta())
    assert merged.has_cad_faces
    assert merged.cad_face_count == 2


def _write_npz(path, vertices, faces, tri_to_cad, cad_meta) -> None:
    np.savez(
        path,
        vertices=np.asarray(vertices),
        faces=np.asarray(faces),
        tri_cad=np.asarray(tri_to_cad),
        cad_meta=np.array(json.dumps(cad_meta)),
    )


def test_load_step_npz_sets_source(tmp_path) -> None:
    vertices, faces, tri_to_cad, cad_meta = _two_quad_cad()
    path = tmp_path / "model.npz"
    _write_npz(path, vertices, faces, tri_to_cad, cad_meta)

    model = load_step_npz(path, source_path="model.step")

    assert model.face_count == 4
    assert model.cad_face_count == 2
    assert model.source_path == "model.step"


def test_load_model_step_branch_uses_converter(tmp_path, monkeypatch) -> None:
    import stl_painter.importer as importer_module

    vertices, faces, tri_to_cad, cad_meta = _two_quad_cad()
    step_path = tmp_path / "part.step"
    step_path.write_text("dummy step")

    def fake_convert(source, **kwargs):
        from contextlib import contextmanager

        @contextmanager
        def _ctx():
            npz_path = tmp_path / "part.npz"
            _write_npz(npz_path, vertices, faces, tri_to_cad, cad_meta)
            yield str(npz_path)

        return _ctx()

    monkeypatch.setattr(
        "stl_painter.step_convert.convert_step_to_npz", fake_convert
    )
    # load_model imports convert_step_to_npz lazily from stl_painter.step_convert,
    # so patch the attribute on that module (also visible via importer import).
    assert importer_module.load_model is load_model

    model = load_model(step_path)

    assert model.has_cad_faces
    assert model.cad_face_count == 2
    assert model.source_path == str(step_path)


def test_renderer_cad_edges_and_software_render() -> None:
    from stl_painter.camera import OrbitCamera
    from stl_painter.renderer import MeshRenderer

    vertices, faces, tri_to_cad, cad_meta = _two_quad_cad()
    model = MeshModel.from_cad_arrays(vertices, faces, tri_to_cad, cad_meta)
    renderer = MeshRenderer(None, model, (320, 240), prefer_gpu=False)

    assert len(renderer._cad_boundary_edge_positions()) == 16  # 8 edges x 2 verts
    assert len(renderer._full_triangle_edge_positions()) == 24  # 4 tris x 3 x 2

    camera = OrbitCamera.for_mesh(model.vertices)
    plain = renderer.render(camera, show_triangle_edges=False)
    assert plain.rgba.shape == (240, 320, 4)
    outlined = renderer.render(camera, show_triangle_edges=True)
    assert outlined.rgba.shape == (240, 320, 4)

    def _black_pixels(image):
        return int(np.count_nonzero((image[:, :, :3] == 0).all(axis=2)))

    # CAD outlines are drawn (more edge pixels than the plain shaded view).
    assert _black_pixels(outlined.rgba) > _black_pixels(plain.rgba)

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


def _front_back_cad():
    """Two separated triangles with opposite normals as two CAD faces."""
    vertices = np.asarray(
        [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.5, 1.0, 0.0),
            (2.5, 0.0, 0.0),
            (3.5, 0.0, 0.0),
            (3.0, 1.0, 0.0),
        ],
        dtype=np.float32,
    )
    faces = np.asarray([(0, 1, 2), (3, 5, 4)], dtype=np.int32)
    tri_to_cad = np.asarray([0, 1], dtype=np.int64)
    cad_meta = [
        {"id": "front", "solid": 0, "name": "front", "surface": "PLANE", "count": 1},
        {"id": "back", "solid": 0, "name": "back", "surface": "PLANE", "count": 1},
    ]
    return vertices, faces, tri_to_cad, cad_meta


def test_front_cad_edge_positions_filters_backfaces() -> None:
    from stl_painter.camera import OrbitCamera
    from stl_painter.renderer import MeshRenderer

    vertices, faces, tri_to_cad, cad_meta = _front_back_cad()
    model = MeshModel.from_cad_arrays(vertices, faces, tri_to_cad, cad_meta)
    renderer = MeshRenderer(None, model, (200, 200), prefer_gpu=False)

    camera = OrbitCamera(
        target=np.array([1.75, 0.5, 0.0], dtype=np.float32),
        distance=5.0,
        azimuth=0.0,
        elevation=90.0,
    )
    positions = renderer._front_cad_edge_positions(camera)
    assert len(positions) == 6


def test_front_cad_edge_vao_caches_per_camera() -> None:
    try:
        import moderngl
    except Exception:
        import pytest
        pytest.skip("moderngl not available")

    from stl_painter.camera import OrbitCamera
    from stl_painter.renderer import MeshRenderer

    vertices, faces, tri_to_cad, cad_meta = _front_back_cad()
    model = MeshModel.from_cad_arrays(vertices, faces, tri_to_cad, cad_meta)
    ctx = moderngl.create_standalone_context()
    renderer = MeshRenderer(ctx, model, (200, 200), prefer_gpu=True)
    if not renderer._gpu_ready:
        import pytest
        pytest.skip("GPU renderer not available")

    camera = OrbitCamera(
        target=np.array([1.75, 0.5, 0.0], dtype=np.float32),
        distance=5.0,
        azimuth=0.0,
        elevation=90.0,
    )
    vao1 = renderer._front_cad_edge_vao_for_camera(camera)
    assert vao1 is not None
    vao2 = renderer._front_cad_edge_vao_for_camera(camera)
    assert vao2 is vao1


def test_software_render_hides_backface_cad_edges() -> None:
    from stl_painter.camera import OrbitCamera
    from stl_painter.renderer import MeshRenderer

    vertices, faces, tri_to_cad, cad_meta = _front_back_cad()
    model = MeshModel.from_cad_arrays(vertices, faces, tri_to_cad, cad_meta)
    renderer = MeshRenderer(None, model, (200, 200), prefer_gpu=False)

    # NOTE: elevation=90 is a degenerate pole view (zero view matrix),
    # so use a real orbit view here. With for_mesh (azimuth 45), the
    # back-facing triangle projects left (x < 100) and the front-facing
    # triangle projects right: only the left region must stay edge-free.
    camera = OrbitCamera.for_mesh(model.vertices)
    plain = renderer.render(camera, show_triangle_edges=False)
    result = renderer.render(camera, show_triangle_edges=True)

    def _black_pixels(image):
        return int(np.count_nonzero((image[:, :, :3] == 0).all(axis=2)))

    assert _black_pixels(result.rgba) > _black_pixels(plain.rgba)
    left_region = result.rgba[:, :100, :]
    black_pixels = int(np.count_nonzero((left_region[:, :, :3] == 0).all(axis=2)))
    assert black_pixels == 0


def test_gpu_render_hides_backface_cad_edges() -> None:
    try:
        import moderngl
    except Exception:
        import pytest
        pytest.skip("moderngl not available")

    from stl_painter.camera import OrbitCamera
    from stl_painter.renderer import MeshRenderer

    vertices, faces, tri_to_cad, cad_meta = _front_back_cad()
    model = MeshModel.from_cad_arrays(vertices, faces, tri_to_cad, cad_meta)
    ctx = moderngl.create_standalone_context()
    renderer = MeshRenderer(ctx, model, (200, 200), prefer_gpu=True)
    if not renderer._gpu_ready:
        import pytest
        pytest.skip("GPU renderer not available")

    camera = OrbitCamera.for_mesh(model.vertices)
    plain = renderer.render(camera, show_triangle_edges=False)
    result = renderer.render(camera, show_triangle_edges=True)

    def _black_pixels(image):
        return int(np.count_nonzero((image[:, :, :3] == 0).all(axis=2)))

    assert _black_pixels(result.rgba) > _black_pixels(plain.rgba)
    left_region = result.rgba[:, :100, :]
    black_pixels = int(np.count_nonzero((left_region[:, :, :3] == 0).all(axis=2)))
    assert black_pixels == 0


def _two_cubes_in_row_cad():
    """Front unit cube (z 2..3) fully occluding a smaller back cube.

    The back cube's +z faces point at a camera on +z, so the facing-only
    filter keeps their edges: only a true occlusion test hides them.
    Windings are outward (verified), 12 triangles, 12 CAD faces.
    """
    base = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    cube_faces = [
        (0, 2, 1), (0, 3, 2),
        (4, 5, 6), (4, 6, 7),
        (0, 1, 5), (0, 5, 4),
        (3, 7, 6), (3, 6, 2),
        (0, 4, 7), (0, 7, 3),
        (1, 2, 6), (1, 6, 5),
    ]
    front = base + np.asarray([0.0, 0.0, 2.0], dtype=np.float32)
    small = (base - 0.5) * 0.6 + np.asarray([0.5, 0.5, 0.5], dtype=np.float32)
    vertices = np.vstack([front, small]).astype(np.float32)
    faces = np.asarray(
        cube_faces + [(a + 8, b + 8, c + 8) for a, b, c in cube_faces],
        dtype=np.int32,
    )
    tri_to_cad = np.asarray(list(range(6)) * 2 + list(range(6, 12)) * 2, dtype=np.int64)
    cad_meta = [
        {
            "id": f"cube_{'front' if i < 6 else 'back'}_f{i % 6}",
            "solid": 0 if i < 6 else 1,
            "name": f"cube_{'front' if i < 6 else 'back'}_f{i % 6}",
            "surface": "PLANE",
            "count": 2,
        }
        for i in range(12)
    ]
    return vertices, faces, tri_to_cad, cad_meta


def test_software_render_hides_occluded_cad_edges() -> None:
    """Occluded-but-front-facing CAD edges must not draw (X-ray)."""
    from stl_painter.camera import OrbitCamera
    from stl_painter.renderer import MeshRenderer

    vertices, faces, tri_to_cad, cad_meta = _two_cubes_in_row_cad()
    model = MeshModel.from_cad_arrays(vertices, faces, tri_to_cad, cad_meta)
    renderer = MeshRenderer(None, model, (400, 300), prefer_gpu=False)

    camera = OrbitCamera(
        target=np.array([0.5, 0.5, 1.5], dtype=np.float32),
        distance=8.0,
        azimuth=0.0,
        elevation=85.0,
    )
    # The back cube's +z faces (14, 15) face the camera, so the cheap
    # facing filter alone keeps their edges: this test is only meaningful
    # if that holds.
    mask = renderer._front_facing_mask(camera)
    assert bool(mask[14]) and bool(mask[15])

    plain = renderer.render(camera, show_triangle_edges=False)
    outlined = renderer.render(camera, show_triangle_edges=True)

    def _black_pixels(image):
        return int(np.count_nonzero((image[:, :, :3] == 0).all(axis=2)))

    # Front-cube outlines are drawn (guards against a vacuous render).
    assert _black_pixels(outlined.rgba) > _black_pixels(plain.rgba)

    # Back-cube +z corners project inside the front silhouette: the pick
    # image must show front-cube faces there (occluded), and the outlined
    # render must match the plain render (no edge drawn on top).
    mvp = camera.mvp_matrix((400, 300))
    pick = renderer._build_pick_image_software(camera)
    corners = np.asarray(
        [(0.2, 0.2, 0.8), (0.8, 0.2, 0.8), (0.8, 0.8, 0.8), (0.2, 0.8, 0.8)],
        dtype=np.float32,
    )
    clip = (mvp @ np.c_[corners, np.ones(4, dtype=np.float32)].T).T
    ndc = clip[:, :3] / clip[:, 3:4]
    xs = ((ndc[:, 0] * 0.5 + 0.5) * 400).astype(int)
    ys = ((1.0 - (ndc[:, 1] * 0.5 + 0.5)) * 300).astype(int)
    covered = 0
    for x, y in zip(xs.tolist(), ys.tolist()):
        assert 0 <= x < 400 and 0 <= y < 300
        if tuple(plain.rgba[y, x, :3].tolist()) == (237, 240, 245):
            continue  # corner off the model: not a usable sample
        shown = (int(pick[y, x, 0]) << 16) | (int(pick[y, x, 1]) << 8) | int(
            pick[y, x, 2]
        )
        if shown >= 12:  # must be a front-cube face occluding the corner
            continue
        covered += 1
        assert tuple(outlined.rgba[y, x].tolist()) == tuple(
            plain.rgba[y, x].tolist()
        )
        assert not bool((outlined.rgba[y, x, :3] == 0).all())
    assert covered >= 3


def test_step_paint_click_applies_colour_to_cad_group() -> None:
    vertices, faces, tri_to_cad, cad_meta = _two_quad_cad()
    model = MeshModel.from_cad_arrays(vertices, faces, tri_to_cad, cad_meta)
    new_colour = (255, 0, 0, 255)
    face_id = 0
    cad_id = model.cad_id_for_face(face_id)
    assert cad_id is not None
    group_faces = model.faces_for_cad(cad_id)
    assert len(group_faces) == 2
    for fid in group_faces:
        model.set_face_colour(fid, new_colour)
    for fid in group_faces:
        assert model.face_colour(fid) == new_colour


def test_step_paint_click_via_commands_touches_cad_group() -> None:
    """The exact CAD branch used by paint clicks must report touches.

    ``TexturePainterApp._apply_paint_at_pick`` builds one update per CAD
    group face and relies on ``AppCommands.paint_faces`` returning the
    touched faces; an empty return shows "No change" and nothing refreshes.
    """
    from stl_painter.commands import AppCommands
    from stl_painter.paint_tool import PaintTool

    vertices, faces, tri_to_cad, cad_meta = _two_quad_cad()
    model = MeshModel.from_cad_arrays(vertices, faces, tri_to_cad, cad_meta)
    commands = AppCommands(None, None)
    commands.attach(model, PaintTool(model))

    cad_id = model.cad_id_for_face(0)
    assert cad_id is not None
    updates = {
        fid: (0, 255, 0, 255)
        for fid in model.faces_for_cad(cad_id)
        if fid not in model.masked_faces
    }
    assert updates
    touched = commands.paint_faces(updates, description="Brush stroke")
    assert sorted(touched) == sorted(updates)
    for fid in model.faces_for_cad(cad_id):
        assert model.face_colour(fid) == (0, 255, 0, 255)


def test_cad_edge_visible_in_pick() -> None:
    """Per-edge occlusion: visible edges pass, occluded edges fail."""
    from stl_painter.camera import OrbitCamera
    from stl_painter.renderer import MeshRenderer

    vertices, faces, tri_to_cad, cad_meta = _two_cubes_in_row_cad()
    model = MeshModel.from_cad_arrays(vertices, faces, tri_to_cad, cad_meta)
    renderer = MeshRenderer(None, model, (400, 300), prefer_gpu=False)
    camera = OrbitCamera(
        target=np.array([0.5, 0.5, 1.5], dtype=np.float32),
        distance=8.0,
        azimuth=0.0,
        elevation=85.0,
    )
    screen, _ = renderer._project_vertices(camera, (400, 300))
    pick = renderer._build_pick_image_software(camera)
    faces_arr = model.faces
    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    for face_id in range(len(faces_arr)):
        a, b, c = (int(faces_arr[face_id, 0]), int(faces_arr[face_id, 1]), int(
            faces_arr[face_id, 2]))
        for u, v in ((a, b), (b, c), (c, a)):
            edge_to_faces.setdefault((u, v) if u < v else (v, u), []).append(face_id)

    # A front-cube top edge is visible ...
    assert renderer._cad_edge_visible_in_pick(
        pick, screen[4], screen[5], edge_to_faces[(4, 5)]
    )
    # ... while a back-cube top edge is occluded by the front cube.
    assert not renderer._cad_edge_visible_in_pick(
        pick, screen[12], screen[13], edge_to_faces[(12, 13)]
    )


def test_visible_cad_edges_hide_occluded_without_gl() -> None:
    """GPU helper parity: occluded-but-front-facing edges are dropped.

    The back cube's +z faces point at the camera, so the cheap facing
    filter alone keeps their edges (asserted below): only the pick-based
    occlusion test in ``_visible_cad_edges`` hides them. Needs no GL
    context, so it guards the GPU path on headless CI too.
    """
    from stl_painter.camera import OrbitCamera
    from stl_painter.renderer import MeshRenderer

    vertices, faces, tri_to_cad, cad_meta = _two_cubes_in_row_cad()
    model = MeshModel.from_cad_arrays(vertices, faces, tri_to_cad, cad_meta)
    renderer = MeshRenderer(None, model, (400, 300), prefer_gpu=False)
    camera = OrbitCamera(
        target=np.array([0.5, 0.5, 1.5], dtype=np.float32),
        distance=8.0,
        azimuth=0.0,
        elevation=85.0,
    )
    mask = renderer._front_facing_mask(camera)
    assert bool(mask[14]) and bool(mask[15])

    visible = renderer._visible_cad_edges(camera)
    keys = {tuple(sorted(edge)) for edge in visible.tolist()}
    # Front-cube top edge stays; back-cube top edge (occluded) goes.
    assert (4, 5) in keys
    assert (12, 13) not in keys

    # The occlusion filter must actually remove something vs facing-only.
    incidents = renderer._edge_incidents()
    edges = model.cad_boundary_edges()
    front_only = sum(
        1
        for u, v in edges.tolist()
        if bool(mask[incidents.get((u, v) if u < v else (v, u), [])].any())
    )
    assert len(visible) < front_only


def test_front_cad_edge_positions_hide_occluded() -> None:
    """End of the GPU chain: positions exclude occluded back-cube edges."""
    from stl_painter.camera import OrbitCamera
    from stl_painter.renderer import MeshRenderer

    vertices, faces, tri_to_cad, cad_meta = _two_cubes_in_row_cad()
    model = MeshModel.from_cad_arrays(vertices, faces, tri_to_cad, cad_meta)
    renderer = MeshRenderer(None, model, (400, 300), prefer_gpu=False)
    camera = OrbitCamera(
        target=np.array([0.5, 0.5, 1.5], dtype=np.float32),
        distance=8.0,
        azimuth=0.0,
        elevation=85.0,
    )
    positions = renderer._front_cad_edge_positions(camera)
    assert len(positions) % 2 == 0
    assert len(positions) > 0
    # Back-cube corner (0.2, 0.2, 0.8) must not anchor any drawn edge.
    back_corner = np.asarray([0.2, 0.2, 0.8], dtype=np.float32)
    assert not any(
        np.allclose(positions[i], back_corner, atol=1e-6)
        for i in range(len(positions))
    )


def test_occlusion_size_caps_large_viewports() -> None:
    """Occlusion pick stays cheap on large viewports."""
    from stl_painter.renderer import MeshRenderer

    vertices, faces, tri_to_cad, cad_meta = _two_quad_cad()
    model = MeshModel.from_cad_arrays(vertices, faces, tri_to_cad, cad_meta)
    renderer = MeshRenderer(None, model, (1920, 1080), prefer_gpu=False)
    assert renderer._occlusion_size() == (480, 270)

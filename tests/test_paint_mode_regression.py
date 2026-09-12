"""Regression tests for the texture-paint-mode regression.

Entering texture paint mode must visibly change the model (triangle
edges) and faces must remain selectable via picking.
"""

from __future__ import annotations

import numpy as np

from stl_painter.camera import OrbitCamera
from stl_painter.picking import _pick_face_location_fallback, pick_face_location_cpu
from stl_painter.renderer import MeshRenderer


def test_full_triangle_edge_positions_cover_all_faces(square_mesh) -> None:
    renderer = MeshRenderer(None, square_mesh, (100, 100), prefer_gpu=False)
    edges = renderer._full_triangle_edge_positions()
    # 3 edges x 2 verts per face.
    assert edges.shape == (square_mesh.face_count * 6, 3)
    # Every face contributes its own vertices (not group-filtered).
    first_tri = square_mesh.vertices[square_mesh.faces[0]]
    for corner in first_tri:
        assert np.any(np.all(np.isclose(edges, corner), axis=1))


def test_software_render_edges_differ_from_preview(square_mesh) -> None:
    renderer = MeshRenderer(None, square_mesh, (100, 100), prefer_gpu=False)
    camera = OrbitCamera.for_mesh(square_mesh.vertices)
    plain = renderer.render(camera, show_triangle_edges=False)
    paint = renderer.render(camera, show_triangle_edges=True)
    assert plain.viewport_size == paint.viewport_size == (100, 100)
    assert np.abs(plain.rgba.astype(int) - paint.rgba.astype(int)).sum() > 0


def test_cpu_pick_hits_face_at_projected_center(square_mesh) -> None:
    """A ray through a visible face center must return a face (not None)."""
    size = (200, 200)
    camera = OrbitCamera.for_mesh(square_mesh.vertices)
    renderer = MeshRenderer(None, square_mesh, size, prefer_gpu=False)
    _, order = renderer._sorted_face_indices(camera, size)
    assert len(order) > 0
    # Front-most visible face: project its center and pick there.
    top_face = int(order[-1])
    center = square_mesh.vertices[square_mesh.faces[top_face]].mean(axis=0)
    mvp = camera.mvp_matrix(size)
    clip = mvp @ np.array([center[0], center[1], center[2], 1.0], dtype=np.float32)
    ndc = clip[:3] / clip[3]
    x = float((ndc[0] * 0.5 + 0.5) * size[0])
    y = float((1.0 - (ndc[1] * 0.5 + 0.5)) * size[1])
    hit = pick_face_location_cpu(square_mesh, camera, x, y, size)
    assert hit is not None
    assert 0 <= hit.face_id < square_mesh.face_count
    fallback = _pick_face_location_fallback(
        square_mesh, *camera.unproject_ray(x, y, size)
    )
    assert fallback is not None


def test_display_to_render_mapping_scales_and_passes_through() -> None:
    from stl_painter.app import TexturePainterApp

    # Scaled display rect maps into render space.
    x, y, size = TexturePainterApp._map_display_to_render(
        (100.0, 50.0), (800, 600), (400, 300)
    )
    assert (x, y, size) == (50.0, 25.0, (400, 300))
    # Identical sizes pass through untouched.
    x, y, size = TexturePainterApp._map_display_to_render(
        (12.0, 34.0), (400, 300), (400, 300)
    )
    assert (x, y, size) == (12.0, 34.0, (400, 300))
    # Degenerate sizes never divide by zero.
    x, y, size = TexturePainterApp._map_display_to_render(
        (12.0, 34.0), (0, 0), (400, 300)
    )
    assert (x, y, size) == (12.0, 34.0, (400, 300))


def test_hover_cpu_pick_sweep_over_real_mesh() -> None:
    """The hover path must resolve every viewport sample without error.

    ``_update_hover_face`` is CPU-picking-only (the GPU pick-FBO render
    access-violates from the GUI thread on some drivers), so sweep a grid
    over a real 36k-face mesh exactly as hover does and require each
    sample to return a valid face or a clean miss.
    """
    from pathlib import Path

    from stl_painter.importer import import_stl

    mesh = import_stl(Path(__file__).resolve().parent.parent / "dart.stl")
    size = (960, 720)
    camera = OrbitCamera.for_mesh(mesh.vertices)
    hits = 0
    samples = 0
    for gx in range(80, 881, 100):
        for gy in range(60, 661, 100):
            result = pick_face_location_cpu(
                mesh, camera, float(gx), float(gy), size
            )
            samples += 1
            if result is not None:
                hits += 1
                assert 0 <= result.face_id < mesh.face_count
                assert result.location.shape == (3,)
                assert result.distance > 0.0
    assert samples == 63
    assert hits > 0


def test_gui_never_calls_gpu_pick_face() -> None:
    """Tripwire: GUI code must not call the GPU pick-FBO path.

    See ``MeshRenderer.pick_face`` / ``_render_pick_gpu`` docstrings.
    """
    from pathlib import Path

    app_source = (
        Path(__file__).resolve().parent.parent / "stl_painter" / "app.py"
    ).read_text(encoding="utf-8")
    assert ".pick_face(" not in app_source
    assert "_render_pick_gpu" not in app_source

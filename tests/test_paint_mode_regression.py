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
    """Tripwire: GPU pick-FBO calls are main-loop only, never in callbacks.

    See ``MeshRenderer.pick_face`` / ``_render_pick_gpu`` docstrings:
    binding the pick framebuffer from inside a Dear PyGui frame callback
    access-violates on some drivers. ``TexturePainterApp`` may therefore
    only call ``.pick_face(`` from ``_process_pending_paint`` (which runs
    in the main loop next to the viewport render, same GL context).
    """
    from pathlib import Path

    app_source = (
        Path(__file__).resolve().parent.parent / "stl_painter" / "app.py"
    ).read_text(encoding="utf-8")
    assert "_render_pick_gpu" not in app_source
    lines = app_source.splitlines()
    current_method: str | None = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("def "):
            current_method = stripped[4:].split("(")[0]
        if ".pick_face(" in line:
            assert current_method == "_process_pending_paint", (
                f"GPU pick must stay in _process_pending_paint, found in {current_method}"
            )


def test_paint_click_uses_display_consistent_picker() -> None:
    """Paint clicks must resolve the displayed face in the main loop.

    The CPU raycaster can disagree with the rasterizer on coplanar or
    edge-on triangles, so clicks painted hidden faces while the viewport
    showed no change. Mouse handlers must therefore only queue the click
    (``_pending_paint_click``); ``_process_pending_paint`` resolves it
    with the display-consistent pick in the GL-safe main loop.
    """
    from pathlib import Path

    app_source = (
        Path(__file__).resolve().parent.parent / "stl_painter" / "app.py"
    ).read_text(encoding="utf-8")
    assert "_pending_paint_click" in app_source
    assert "def _process_pending_paint" in app_source
    assert "_pick_result_software" in app_source


def test_software_pick_paint_updates_rendered_pixel(square_mesh) -> None:
    """End-to-end click flow: pick pixel -> paint face -> pixel changes.

    Mirrors ``TexturePainterApp`` paint clicks (software pick, paint
    command, ``update_face_colours``, re-render) headlessly and requires
    the clicked pixel to actually change colour.
    """
    from stl_painter.commands import AppCommands
    from stl_painter.paint_tool import PaintTool

    size = (200, 200)
    camera = OrbitCamera.for_mesh(square_mesh.vertices)
    renderer = MeshRenderer(None, square_mesh, size, prefer_gpu=False)

    pick = renderer._build_pick_image_software(camera, size)
    before_snap = renderer.render(camera, show_triangle_edges=False)
    target = None
    for y in range(size[1]):
        for x in range(size[0]):
            if tuple(before_snap.rgba[y, x, :3].tolist()) == (237, 240, 245):
                continue  # background pixel (also decodes as face 0)
            face_id = (int(pick[y, x, 0]) << 16) | (
                int(pick[y, x, 1]) << 8
            ) | int(pick[y, x, 2])
            if 0 <= face_id < square_mesh.face_count:
                target = (x, y, face_id)
                break
        if target is not None:
            break
    assert target is not None
    x, y, face_id = target

    before = before_snap.rgba[y, x].tolist()

    tool = PaintTool(square_mesh)
    commands = AppCommands(None, None)
    commands.attach(square_mesh, tool)
    hit_point = square_mesh.face_center(face_id).astype(np.float32)
    updates = tool.brush_updates(
        face_id,
        hit_point,
        (255, 0, 0, 255),
        radius=0.0001,
        opacity=1.0,
        falloff="smooth",
        front_faces_only=False,
        angle_tolerance_degrees=65.0,
        erase=False,
    )
    assert face_id in updates
    touched = commands.paint_faces(updates, description="Brush stroke")
    assert face_id in touched
    renderer.update_face_colours(touched)

    after = renderer.render(camera, show_triangle_edges=False).rgba[y, x].tolist()
    assert after != before
    assert after[0] > 150 and after[1] < 150 and after[2] < 150


def test_display_pick_paint_updates_rendered_pixel(square_mesh) -> None:
    """End-to-end main-loop click flow using the display pick.

    Mirrors ``TexturePainterApp._process_pending_paint`` headlessly:
    ``renderer.pick_face`` (GPU buffer when available, software image
    otherwise — the same call the main loop makes), paint command,
    ``update_face_colours``, re-render. The clicked pixel must change.
    """
    from stl_painter.commands import AppCommands
    from stl_painter.paint_tool import PaintTool

    size = (200, 200)
    camera = OrbitCamera.for_mesh(square_mesh.vertices)
    renderer = MeshRenderer(None, square_mesh, size, prefer_gpu=False)

    before_snap = renderer.render(camera, show_triangle_edges=False)
    target = None
    for y in range(size[1]):
        for x in range(size[0]):
            if tuple(before_snap.rgba[y, x, :3].tolist()) == (237, 240, 245):
                continue  # background pixel
            face_id = renderer.pick_face(camera, x, y)
            if face_id is not None and 0 <= face_id < square_mesh.face_count:
                target = (x, y, face_id)
                break
        if target is not None:
            break
    assert target is not None
    x, y, face_id = target

    before = before_snap.rgba[y, x].tolist()

    tool = PaintTool(square_mesh)
    commands = AppCommands(None, None)
    commands.attach(square_mesh, tool)
    hit_point = square_mesh.face_center(face_id).astype(np.float32)
    updates = tool.brush_updates(
        face_id,
        hit_point,
        (255, 0, 0, 255),
        radius=0.0001,
        opacity=1.0,
        falloff="smooth",
        front_faces_only=False,
        angle_tolerance_degrees=65.0,
        erase=False,
    )
    assert face_id in updates
    touched = commands.paint_faces(updates, description="Brush stroke")
    assert face_id in touched
    renderer.update_face_colours(touched)

    after = renderer.render(camera, show_triangle_edges=False).rgba[y, x].tolist()
    assert after != before
    assert after[0] > 150 and after[1] < 150 and after[2] < 150

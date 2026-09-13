"""Paint-mode wireframe occlusion fix: front-facing outlines only, no X-ray."""

from __future__ import annotations

import numpy as np

from stl_painter.camera import OrbitCamera
from stl_painter.interaction_state import InteractionState


def _app_without_init():
    from stl_painter.app import TexturePainterApp

    app = TexturePainterApp.__new__(TexturePainterApp)
    return app


def test_paint_mode_requests_wireframe() -> None:
    app = _app_without_init()
    app.state = InteractionState()
    app.state.workspace_mode = "paint"
    assert app.should_show_triangle_edges() is True
    app.state.workspace_mode = "preview"
    assert app.should_show_triangle_edges() is False


def test_front_visible_filter_hides_backface(square_mesh) -> None:
    from stl_painter.app import TexturePainterApp
    from stl_painter.renderer import MeshRenderer

    app = TexturePainterApp.__new__(TexturePainterApp)
    app.mesh_model = square_mesh
    app.camera = OrbitCamera.for_mesh(square_mesh.vertices)
    app.renderer = MeshRenderer(
        None, square_mesh, (200, 200), prefer_gpu=False
    )
    app.state = InteractionState()
    app.state.viewport_size = (200, 200)

    # Front-most visible face from the renderer ordering must pass.
    _, order = app.renderer._sorted_face_indices(app.camera, (200, 200))
    assert len(order) > 0
    assert app._is_face_visible(int(order[-1])) is True

    # A face pointing away from the camera must fail.
    normals = square_mesh.normals
    to_cams = app.camera.position().astype(np.float32) - np.asarray(
        [square_mesh.face_center(i) for i in range(square_mesh.face_count)],
        dtype=np.float32,
    )
    dots = np.einsum("ij,ij->i", normals.astype(np.float32), to_cams)
    back = np.where(dots <= 0.0)[0]
    if len(back):
        assert app._is_face_visible(int(back[0])) is False


def test_gpu_front_edge_vao_filters_backfaces(square_mesh) -> None:
    try:
        import moderngl
    except Exception:
        import pytest
        pytest.skip("moderngl not available")

    from stl_painter.renderer import MeshRenderer

    ctx = moderngl.create_standalone_context()
    renderer = MeshRenderer(ctx, square_mesh, (200, 200), prefer_gpu=True)
    if not renderer._gpu_ready:
        import pytest
        pytest.skip("GPU renderer not available")

    camera = OrbitCamera.for_mesh(square_mesh.vertices)
    vao = renderer._front_edge_vao_for_camera(camera)
    # At least some edges should be generated for front-facing faces.
    assert vao is not None

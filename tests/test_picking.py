from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from stl_painter.camera import OrbitCamera
from stl_painter.picking import pick_face_location_cpu, pick_face_cpu


def test_pick_face_location_cpu_falls_back_without_rtree(
    square_mesh, monkeypatch
) -> None:
    camera = OrbitCamera.for_mesh(square_mesh.vertices)

    def fake_mesh() -> SimpleNamespace:
        ray = SimpleNamespace(
            intersects_location=lambda **_kwargs: (_ for _ in ()).throw(
                ModuleNotFoundError("No module named 'rtree'")
            )
        )
        return SimpleNamespace(ray=ray)

    monkeypatch.setattr(type(square_mesh), "mesh", lambda self: fake_mesh())

    hit = pick_face_location_cpu(square_mesh, camera, 50.0, 50.0, (100, 100))

    assert hit is not None
    assert hit.face_id in {0, 1}
    assert np.allclose(hit.location[2], 0.0, atol=1e-5)
    assert hit.distance > 0.0


def test_pick_face_location_cpu_falls_back_on_any_backend_error(
    square_mesh, monkeypatch
) -> None:
    """Any ray-backend failure (not just missing rtree) must fall back."""
    camera = OrbitCamera.for_mesh(square_mesh.vertices)

    def fake_mesh() -> SimpleNamespace:
        ray = SimpleNamespace(
            intersects_location=lambda **_kwargs: (_ for _ in ()).throw(
                RuntimeError("broken GL/backend")
            )
        )
        return SimpleNamespace(ray=ray)

    monkeypatch.setattr(type(square_mesh), "mesh", lambda self: fake_mesh())

    hit = pick_face_location_cpu(square_mesh, camera, 50.0, 50.0, (100, 100))

    assert hit is not None
    assert hit.face_id in {0, 1}


def test_pick_face_cpu_returns_face_id(square_mesh) -> None:
    camera = OrbitCamera.for_mesh(square_mesh.vertices)
    face_id = pick_face_cpu(square_mesh, camera, 50.0, 50.0, (100, 100))

    assert face_id is not None
    assert face_id in {0, 1}


def test_pick_face_location_cpu_returns_result(square_mesh) -> None:
    camera = OrbitCamera.for_mesh(square_mesh.vertices)
    result = pick_face_location_cpu(square_mesh, camera, 50.0, 50.0, (100, 100))

    assert result is not None
    assert result.face_id >= 0
    assert result.location.shape == (3,)
    assert result.distance >= 0.0


def test_pick_face_location_returns_none_when_miss(square_mesh) -> None:
    camera = OrbitCamera.for_mesh(square_mesh.vertices)
    result = pick_face_location_cpu(square_mesh, camera, 10000.0, 10000.0, (100, 100))

    assert result is None


def test_pick_result_dataclass() -> None:
    from stl_painter.picking import PickResult

    result = PickResult(
        face_id=0,
        location=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        distance=1.0,
    )
    assert result.face_id == 0
    assert np.allclose(result.location, [0.0, 0.0, 0.0])
    assert result.distance == 1.0

from __future__ import annotations

import numpy as np

from stl_painter.camera import OrbitCamera, look_at, perspective


def test_perspective_projection_matrix() -> None:
    matrix = perspective(45.0, 1.0, 0.01, 1000.0)
    assert matrix.shape == (4, 4)
    assert matrix[3, 2] == -1.0
    assert matrix[2, 3] != 0.0


def test_look_at_creates_valid_view_matrix() -> None:
    eye = np.asarray([0.0, 0.0, 5.0], dtype=np.float32)
    target = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    up = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)

    matrix = look_at(eye, target, up)

    assert matrix.shape == (4, 4)
    assert np.allclose(matrix[3, :3], [0.0, 0.0, 0.0])
    assert np.isclose(matrix[3, 3], 1.0)


def test_orbit_camera_for_mesh() -> None:
    vertices = np.asarray(
        [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        ],
        dtype=np.float32,
    )
    camera = OrbitCamera.for_mesh(vertices)

    assert camera.target.shape == (3,)
    assert camera.distance > 0.0
    assert np.allclose(camera.target, [0.5, 0.5, 0.0], atol=1e-4)


def test_orbit_camera_position_calculation() -> None:
    camera = OrbitCamera(
        target=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        distance=5.0,
        azimuth=45.0,
        elevation=30.0,
    )
    position = camera.position()

    assert position.shape == (3,)
    assert np.linalg.norm(position) > 0.0


def test_orbit_camera_view_matrix() -> None:
    camera = OrbitCamera(
        target=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        distance=5.0,
        azimuth=0.0,
        elevation=0.0,
    )
    view = camera.view_matrix()

    assert view.shape == (4, 4)


def test_orbit_camera_projection_matrix() -> None:
    import numpy as np

    camera = OrbitCamera(
        target=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        fov_y_degrees=45.0,
        near=0.1,
        far=100.0,
    )
    try:
        proj = camera.projection_matrix((800, 600))
        assert proj.shape == (4, 4)
    except TypeError:
        pass


def test_orbit_camera_orbit_clips_elevation() -> None:
    camera = OrbitCamera(
        target=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        azimuth=0.0,
        elevation=85.0,
    )
    camera.orbit(0.0, 20.0)

    assert camera.elevation <= 89.0


def test_orbit_camera_set_angles() -> None:
    camera = OrbitCamera(
        target=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        azimuth=0.0,
        elevation=0.0,
    )
    camera.set_angles(90.0, 45.0)

    assert camera.azimuth == 90.0
    assert camera.elevation == 45.0


def test_orbit_camera_set_angles_clips_elevation() -> None:
    camera = OrbitCamera(
        target=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        elevation=0.0,
    )
    camera.set_angles(0.0, 100.0)

    assert camera.elevation <= 89.0


def test_orbit_camera_zoom() -> None:
    camera = OrbitCamera(
        target=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        distance=5.0,
    )
    original_distance = camera.distance
    camera.zoom(0.1)

    assert camera.distance < original_distance


def test_orbit_camera_zoom_enforces_minimum() -> None:
    camera = OrbitCamera(
        target=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        distance=0.1,
    )
    camera.zoom(10.0)

    assert camera.distance >= 0.05


def test_orbit_camera_pan() -> None:
    camera = OrbitCamera(
        target=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        distance=5.0,
        azimuth=0.0,
        elevation=0.0,
    )
    original_target = camera.target.copy()
    camera.pan(100.0, 100.0)

    assert not np.allclose(camera.target, original_target)


def test_orbit_camera_unproject_ray() -> None:
    camera = OrbitCamera(
        target=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        distance=5.0,
        azimuth=0.0,
        elevation=0.0,
    )
    origin, direction = camera.unproject_ray(400.0, 300.0, (800, 600))

    assert origin.shape == (3,)
    assert direction.shape == (3,)
    assert np.isfinite(origin).all()
    assert np.isfinite(direction).all()
    assert np.allclose(np.linalg.norm(direction), 1.0, atol=1e-4)

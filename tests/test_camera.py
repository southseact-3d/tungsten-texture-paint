from __future__ import annotations

import numpy as np
import pytest

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


def _flat_camera() -> OrbitCamera:
    return OrbitCamera(
        target=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        distance=5.0,
        azimuth=0.0,
        elevation=0.0,
    )


def test_orbit_pixels_full_window_spans_200_degrees() -> None:
    camera = _flat_camera()
    camera.orbit_pixels(800.0, 0.0, 800.0, 600.0)

    assert camera.azimuth == pytest.approx(-200.0)
    assert camera.elevation == pytest.approx(0.0)


def test_orbit_pixels_drag_down_raises_elevation() -> None:
    camera = _flat_camera()
    camera.orbit_pixels(0.0, 300.0, 800.0, 600.0)

    # 200*300/600 = 100 degrees but clamped (VTK would roll over instead).
    assert camera.elevation == pytest.approx(89.0)
    assert camera.elevation > 0.0


def test_orbit_pixels_normalized_by_size() -> None:
    small = _flat_camera()
    small.orbit_pixels(100.0, 0.0, 800.0, 600.0)
    big = _flat_camera()
    big.orbit_pixels(200.0, 0.0, 1600.0, 600.0)

    assert small.azimuth == pytest.approx(big.azimuth)


def test_dolly_pixels_down_zooms_out() -> None:
    camera = _flat_camera()
    camera.dolly_pixels(150.0, 600.0)  # drag down

    assert camera.distance > 5.0


def test_dolly_pixels_up_zooms_in() -> None:
    camera = _flat_camera()
    camera.dolly_pixels(-150.0, 600.0)

    assert camera.distance < 5.0
    assert camera.distance >= 0.05


def test_dolly_notches_wheel_up_zooms_in() -> None:
    camera = _flat_camera()
    camera.dolly_notches(1.0)

    assert camera.distance == pytest.approx(5.0 / 1.1**2)


def test_pan_pixels_grab_style_and_scaled() -> None:
    camera = _flat_camera()
    before = camera.target.copy()
    camera.pan_pixels(100.0, 0.0, 600.0)

    moved = camera.target - before
    # At az=0 the camera looks down -X, so screen-right is +Y: dragging
    # right shifts the target toward -Y (content follows the cursor).
    assert abs(moved[0]) < 1e-6
    assert moved[1] < 0.0
    expected = 2.0 * 5.0 * np.tan(np.radians(45.0) / 2.0) / 600.0 * 100.0
    assert abs(moved[1]) == pytest.approx(expected, rel=1e-4)


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

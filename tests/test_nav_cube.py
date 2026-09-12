from __future__ import annotations

import numpy as np

from stl_painter import nav_cube


def _identity() -> np.ndarray:
    return np.eye(3, dtype=np.float32)


def test_project_cube_front_face_visible() -> None:
    faces = nav_cube.project_cube(_identity(), (66.0, 66.0), 36.0)

    assert len(faces) == 6
    by_axis = {str(face["axis"]): face for face in faces}
    assert by_axis["zp"]["visible"] is True
    assert by_axis["zn"]["visible"] is False
    # Painter's order: far faces first.
    depths = [float(face["depth"]) for face in faces]
    assert depths == sorted(depths)
    # Front face polygon is centred on the gizmo centre.
    center = np.asarray(by_axis["zp"]["polygon"], dtype=np.float32).mean(axis=0)
    assert abs(center[0] - 66.0) < 1.0
    assert abs(center[1] - 66.0) < 1.0


def test_project_cube_rotates_with_camera() -> None:
    rotation = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    faces = nav_cube.project_cube(rotation, (66.0, 66.0), 36.0)
    by_axis = {str(face["axis"]): face for face in faces}

    assert by_axis["zp"]["visible"] is True
    assert by_axis["zn"]["visible"] is False


def test_hit_test_clicks_topmost_visible_face() -> None:
    faces = nav_cube.project_cube(_identity(), (66.0, 66.0), 36.0)

    assert nav_cube.hit_test((66.0, 66.0), faces, 52.0, (66.0, 66.0)) == (
        "axis",
        "zp",
    )


def test_hit_test_orbit_and_miss() -> None:
    faces = nav_cube.project_cube(_identity(), (66.0, 66.0), 36.0)

    # Inside the orbit radius but off the cube (corner gap).
    assert nav_cube.hit_test((66.0 + 48.0, 66.0 + 48.0), faces, 52.0, (66.0, 66.0))[
        0
    ] in {"orbit", "none"}
    assert nav_cube.hit_test((200.0, 200.0), faces, 52.0, (66.0, 66.0)) == (
        "none",
        None,
    )


def test_snap_angles_cover_all_axes() -> None:
    assert nav_cube.snap_angles("xp") == (0.0, 0.0)
    assert nav_cube.snap_angles("xn") == (180.0, 0.0)
    assert nav_cube.snap_angles("yp") == (0.0, 89.0)
    assert nav_cube.snap_angles("yn") == (0.0, -89.0)
    assert nav_cube.snap_angles("zp") == (90.0, 0.0)
    assert nav_cube.snap_angles("zn") == (-90.0, 0.0)


def test_face_centers_cover_all_axes() -> None:
    faces = nav_cube.project_cube(_identity(), (66.0, 66.0), 36.0)
    centers = nav_cube.face_centers_2d(faces)

    assert sorted(centers) == ["xn", "xp", "yn", "yp", "zn", "zp"]

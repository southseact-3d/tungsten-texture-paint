from __future__ import annotations

import numpy as np

from stl_painter import navcube


def test_bounds_top_right_corner() -> None:
    x0, y0, x1, y1 = navcube.navcube_bounds(960, 720)
    assert (x1 - x0) == navcube.NAV_CUBE_SIZE
    assert (y1 - y0) == navcube.NAV_CUBE_SIZE
    assert x1 == 960 - navcube.NAV_CUBE_MARGIN
    assert y0 == navcube.NAV_CUBE_MARGIN


def test_point_in_navcube_inside_and_outside() -> None:
    x0, y0, x1, y1 = navcube.navcube_bounds(960, 720)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    assert navcube.point_in_navcube(cx, cy, 960, 720)
    assert navcube.point_in_navcube(x0, y0, 960, 720)
    assert not navcube.point_in_navcube(10.0, 10.0, 960, 720)
    assert not navcube.point_in_navcube(480.0, 360.0, 960, 720)


def test_drag_threshold_matches_slate_manhattan_3px() -> None:
    press = (100.0, 100.0)
    assert not navcube.drag_engaged(press, (102.0, 101.0))  # manhattan == 3
    assert navcube.drag_engaged(press, (102.0, 102.0))  # manhattan == 4
    assert not navcube.drag_engaged(press, press)


def test_azimuth_for_drag_matches_slate_sensitivity() -> None:
    # Slate: new_az = press_azimuth - dx * 0.4 with a fixed press anchor.
    assert navcube.azimuth_for_drag(45.0, 100.0, 150.0) == 25.0
    assert navcube.azimuth_for_drag(45.0, 100.0, 50.0) == 65.0
    assert navcube.azimuth_for_drag(45.0, 100.0, 100.0) == 45.0


def test_project_cube_returns_six_quads_in_bounds() -> None:
    quads = navcube.project_cube(45.0, 30.0)
    assert len(quads) == 6
    for _normal, quad in quads:
        assert len(quad) == 4
        for x, y in quad:
            assert 0 <= x < navcube.NAV_CUBE_SIZE
            assert 0 <= y < navcube.NAV_CUBE_SIZE


def test_project_cube_tracks_camera() -> None:
    quads_a = navcube.project_cube(0.0, 30.0)
    quads_b = navcube.project_cube(90.0, 30.0)
    flat = lambda qs: [(x, y) for _, q in qs for x, y in q]
    assert flat(quads_a) != flat(quads_b)


def test_draw_navcube_paints_top_right_corner() -> None:
    rgba = np.zeros((720, 960, 4), dtype=np.float32)
    bounds = navcube.draw_navcube(rgba, 45.0, 30.0)
    x0, y0, x1, y1 = bounds
    corner = rgba[y0:y1, x0:x1, :]
    assert corner[..., 3].max() > 0.0  # cube pixels written with alpha
    # Outside the corner the buffer is untouched.
    assert rgba[360, 480, 3] == 0.0


def test_draw_navcube_changes_with_azimuth() -> None:
    rgba_a = np.zeros((720, 960, 4), dtype=np.float32)
    rgba_b = np.zeros((720, 960, 4), dtype=np.float32)
    navcube.draw_navcube(rgba_a, 0.0, 30.0)
    navcube.draw_navcube(rgba_b, 90.0, 30.0)
    assert not np.allclose(rgba_a, rgba_b)


def test_draw_navcube_tiny_viewport_is_noop() -> None:
    rgba = np.zeros((40, 40, 4), dtype=np.float32)
    navcube.draw_navcube(rgba, 45.0, 30.0)
    assert rgba[..., 3].max() == 0.0

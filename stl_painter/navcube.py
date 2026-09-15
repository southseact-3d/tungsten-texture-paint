"""NavCube -- software-rendered camera-orientation widget (orbit gizmo).

Ports the SlateWallTexturer orbiting widget (``slate_app/navcube.py``) to
the STL Texture Painter's numpy viewport-compositing pipeline.

Behaviour (identical to the Slate widget, drag-to-orbit only):

* A small 3-D cube is drawn in the top-right corner of the viewport whose
  orientation mirrors the main camera azimuth/elevation.
* Faces are coloured X=red, Y=green, Z=blue with dark outlines and axis
  labels (R/L/B/F/T/Bot).
* Press-and-drag on the cube orbits the main camera: a 3px press-drag
  threshold engages the drag (``manhattan > 3``), then horizontal motion
  rotates azimuth at 0.4 degrees per pixel
  (``new_az = press_azimuth - dx * 0.4``). Elevation, distance and target
  are preserved (elevation stays clamped to +/-89 by the camera).
* Release without a drag does nothing (no click-to-snap).

Coordinate system (identical to :class:`OrbitCamera` / VTK trackball):
    X = forward, Y = right, Z = up.

This module is deliberately free of Dear PyGui/Qt dependencies so it can
be unit-tested headless. Rendering targets the float32 RGBA viewport
buffer (H, W, 4) used by ``TexturePainterApp._render_viewport``.
"""

from __future__ import annotations

import math

import numpy as np

NAV_CUBE_SIZE = 110
NAV_CUBE_MARGIN = 8
NAV_CUBE_FOV_DEGREES = 45.0

# Virtual camera distance and cube half-size for the orientation render.
# The cube orientation mirrors the main camera; scale is fixed (rather than
# the Slate main-camera-distance formula) so the virtual camera always sits
# well outside the cube and every face projects in front of it.
_VIRTUAL_DISTANCE = 5.0
_CUBE_HALF_SIZE = 1.0

# Press-drag engage threshold: Manhattan distance in pixels (Slate parity:
# ``(ev.pos() - press_pos).manhattanLength() > 3``).
DRAG_THRESHOLD_PX = 3.0

# Horizontal drag sensitivity in degrees per pixel (Slate parity:
# ``delta_azimuth = -delta.x() * 0.4``).
DRAG_SENSITIVITY_DEG_PER_PX = 0.4


# ---------------------------------------------------------------------------
# 3-D maths helpers (identical to slate_app/navcube.py)
# ---------------------------------------------------------------------------

def _normalize(v: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = v
    length = math.sqrt(x * x + y * y + z * z)
    if length < 1e-9:
        return (0.0, 0.0, 0.0)
    return (x / length, y / length, z / length)


def _cross(
    a: tuple[float, float, float], b: tuple[float, float, float]
) -> tuple[float, float, float]:
    ax, ay, az = a
    bx, by, bz = b
    return (ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx)


def _sub(
    a: tuple[float, float, float], b: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(
    a: tuple[float, float, float], b: tuple[float, float, float]
) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _mul(s: float, v: tuple[float, float, float]) -> tuple[float, float, float]:
    return (s * v[0], s * v[1], s * v[2])


def _add(
    a: tuple[float, float, float], b: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _view_matrix(
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
    world_up: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> tuple[float, ...]:
    """4x4 view matrix as a flat 16-element tuple (row-major).

    Identical logic to VTK's ``vtkCamera::ViewMatrix`` / ``look_at``:
        forward  = normalize(target - eye)
        right    = normalize(cross(forward, world_up))
        cam_up   = cross(right, forward)
    """
    fwd = _normalize(_sub(target, eye))
    right = _normalize(_cross(fwd, world_up))
    cam_up = _cross(right, fwd)

    ex, ey, ez = eye
    tx = -(right[0] * ex + right[1] * ey + right[2] * ez)
    ty = -(cam_up[0] * ex + cam_up[1] * ey + cam_up[2] * ez)
    tz = -(-fwd[0] * ex + -fwd[1] * ey + -fwd[2] * ez)

    return (
        right[0], right[1], right[2], 0.0,
        cam_up[0], cam_up[1], cam_up[2], 0.0,
        -fwd[0], -fwd[1], -fwd[2], 0.0,
        tx, ty, tz, 1.0,
    )


def _transform_point(
    m: tuple[float, ...], point: tuple[float, float, float]
) -> tuple[float, float, float] | None:
    """Multiply a 4x4 row-major matrix by a homogeneous point."""
    x, y, z = point
    w = m[3] * x + m[7] * y + m[11] * z + m[15]
    if abs(w) < 1e-9:
        return None
    return (
        (m[0] * x + m[1] * y + m[2] * z + m[12]) / w,
        (m[4] * x + m[5] * y + m[6] * z + m[13]) / w,
        (m[8] * x + m[9] * y + m[10] * z + m[14]) / w,
    )


def _perspective_project_viewspace(
    point_vs: tuple[float, float, float],
    fov_deg: float,
    width: int,
    height: int,
) -> tuple[int, int] | None:
    """Project a view-space point to 2-D pixel coords (origin top-left).

    View space follows the OpenGL convention (camera looks down ``-Z``),
    so points in front have negative Z and depth is ``-z``.
    """
    x_vs, y_vs, z_vs = point_vs
    depth = -z_vs
    if depth < 0.05:  # behind the camera
        return None

    f = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
    aspect = max(width, 1) / max(height, 1)
    ndc_x = (x_vs * f / aspect) / depth
    ndc_y = (y_vs * f) / depth
    x = int((ndc_x * 0.5 + 0.5) * width)
    y = int((1.0 - (ndc_y * 0.5 + 0.5)) * height)
    return (x, y)


def _view_depth(point_vs: tuple[float, float, float] | None) -> float:
    """Positive distance in front of the camera (<= 0 when behind)."""
    if point_vs is None:
        return 0.0
    return -point_vs[2]


# ---------------------------------------------------------------------------
# Cube geometry (unit cube centred at origin; identical to slate_app/navcube.py)
# ---------------------------------------------------------------------------

_CUBE_VERTS: list[tuple[float, float, float]] = [
    (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),  # 0..3  z=-1
    (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1),  # 4..7  z=+1
]

# CCW order when viewed from outside (+X right, +Y back, +Z top).
_CUBE_FACES: list[tuple[int, int, int, int]] = [
    (4, 5, 6, 7),  # +X  right
    (0, 3, 2, 1),  # -X  left
    (5, 1, 2, 6),  # +Y  back
    (0, 4, 7, 3),  # -Y  front
    (7, 6, 2, 3),  # +Z  top
    (0, 1, 5, 4),  # -Z  bottom
]

_FACE_NORMALS: list[tuple[int, int, int]] = [
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
]

_FACE_LABEL_POS: list[tuple[float, float, float]] = [
    (1.4, 0.0, 0.0),
    (-1.4, 0.0, 0.0),
    (0.0, 1.4, 0.0),
    (0.0, -1.4, 0.0),
    (0.0, 0.0, 1.4),
    (0.0, 0.0, -1.4),
]

_FACE_LABEL_TEXT = ["R", "L", "B", "F", "T", "Bot"]

# (fill_rgba, outline_rgba, label_rgb) as 0..1 floats. Converted 1:1 from
# the QColor values in slate_app/navcube.py `_FACE_STYLE`.
_FACE_STYLE: dict[tuple[int, int, int], tuple[tuple, tuple, tuple]] = {
    (1, 0, 0): (
        (210 / 255, 80 / 255, 80 / 255, 90 / 255),
        (180 / 255, 40 / 255, 40 / 255, 220 / 255),
        (240 / 255, 70 / 255, 70 / 255),
    ),
    (-1, 0, 0): (
        (170 / 255, 50 / 255, 50 / 255, 80 / 255),
        (130 / 255, 25 / 255, 25 / 255, 200 / 255),
        (200 / 255, 55 / 255, 55 / 255),
    ),
    (0, 1, 0): (
        (80 / 255, 190 / 255, 80 / 255, 90 / 255),
        (40 / 255, 150 / 255, 40 / 255, 220 / 255),
        (70 / 255, 220 / 255, 70 / 255),
    ),
    (0, -1, 0): (
        (50 / 255, 140 / 255, 50 / 255, 80 / 255),
        (25 / 255, 100 / 255, 25 / 255, 200 / 255),
        (55 / 255, 170 / 255, 55 / 255),
    ),
    (0, 0, 1): (
        (80 / 255, 80 / 255, 210 / 255, 90 / 255),
        (40 / 255, 40 / 255, 170 / 255, 220 / 255),
        (70 / 255, 70 / 255, 240 / 255),
    ),
    (0, 0, -1): (
        (50 / 255, 50 / 255, 160 / 255, 80 / 255),
        (25 / 255, 25 / 255, 120 / 255, 200 / 255),
        (55 / 255, 55 / 255, 190 / 255),
    ),
}


# ---------------------------------------------------------------------------
# Hit-testing + drag maths (identical to slate_app/navcube.py interaction)
# ---------------------------------------------------------------------------

def navcube_bounds(
    viewport_width: int,
    viewport_height: int,
    size: int = NAV_CUBE_SIZE,
    margin: int = NAV_CUBE_MARGIN,
) -> tuple[int, int, int, int]:
    """Pixel bounds ``(x0, y0, x1, y1)`` of the cube in the top-right corner."""
    x1 = int(viewport_width) - int(margin)
    y0 = int(margin)
    x0 = x1 - int(size)
    y1 = y0 + int(size)
    return (x0, y0, x1, y1)


def point_in_navcube(
    x: float,
    y: float,
    viewport_width: int,
    viewport_height: int,
    size: int = NAV_CUBE_SIZE,
    margin: int = NAV_CUBE_MARGIN,
) -> bool:
    """True when render-space point (x, y) lies inside the cube bounds."""
    x0, y0, x1, y1 = navcube_bounds(viewport_width, viewport_height, size, margin)
    return x0 <= x <= x1 and y0 <= y <= y1


def drag_engaged(
    press_pos: tuple[float, float],
    current_pos: tuple[float, float],
    threshold_px: float = DRAG_THRESHOLD_PX,
) -> bool:
    """3px press-drag threshold (Slate parity: ``manhattanLength() > 3``)."""
    return (
        abs(current_pos[0] - press_pos[0]) + abs(current_pos[1] - press_pos[1])
        > threshold_px
    )


def azimuth_for_drag(
    press_azimuth: float,
    press_x: float,
    current_x: float,
    sensitivity: float = DRAG_SENSITIVITY_DEG_PER_PX,
) -> float:
    """Azimuth for an in-progress cube drag (Slate parity).

    ``delta_azimuth = -dx * 0.4`` where ``dx`` is the total horizontal
    displacement from the press point; the press anchor is fixed for the
    whole gesture (no origin reset), giving an absolute mapping.
    """
    return float(press_azimuth) - (float(current_x) - float(press_x)) * float(
        sensitivity
    )


# ---------------------------------------------------------------------------
# Projection (identical to slate_app/navcube.py paintEvent)
# ---------------------------------------------------------------------------

def project_cube(
    azimuth_deg: float,
    elevation_deg: float,
    size: int = NAV_CUBE_SIZE,
    fov_deg: float = NAV_CUBE_FOV_DEGREES,
) -> list[tuple[tuple[int, int, int], list[tuple[int, int]]]]:
    """Project the cube for a camera orientation.

    Returns back-to-front ``[(normal, quad), ...]`` with widget-local pixel
    coords (origin top-left). Faces behind the camera are skipped.
    """
    from math import cos as _cos
    from math import radians as _radians
    from math import sin as _sin

    w = h = int(size)
    az_r = _radians(float(azimuth_deg))
    el_r = _radians(float(elevation_deg))
    dist = _VIRTUAL_DISTANCE
    eye = (
        dist * _cos(el_r) * _cos(az_r),
        dist * _cos(el_r) * _sin(az_r),
        dist * _sin(el_r),
    )
    target = (0.0, 0.0, 0.0)

    m = _view_matrix(eye, target, world_up=(0.0, 0.0, 1.0))

    scale = _CUBE_HALF_SIZE

    def project(
        point: tuple[float, float, float],
    ) -> tuple[int, int] | None:
        vs = _transform_point(m, point)
        if vs is None:
            return None
        return _perspective_project_viewspace(vs, fov_deg, w, h)

    screen = [project(_mul(scale, v)) for v in _CUBE_VERTS]

    faces_with_depth: list[
        tuple[float, tuple[int, int, int, int], tuple[int, int, int]]
    ] = []
    for face, normal in zip(_CUBE_FACES, _FACE_NORMALS):
        center = _mul(0.0, _CUBE_VERTS[0])
        for vi in face:
            center = _add(center, _CUBE_VERTS[vi])
        center = _mul(0.25, center)
        vs_center = _transform_point(m, center)
        faces_with_depth.append((_view_depth(vs_center), face, normal))

    faces_with_depth.sort(key=lambda t: t[0], reverse=True)

    quads: list[tuple[tuple[int, int, int], list[tuple[int, int]]]] = []
    for _depth, face, normal in faces_with_depth:
        quad = [screen[vi] for vi in face]
        if any(p is None for p in quad):
            continue
        quads.append((tuple(normal), [q for q in quad if q is not None]))
    return quads


def label_anchor(
    face_index: int,
    azimuth_deg: float,
    elevation_deg: float,
    size: int = NAV_CUBE_SIZE,
    fov_deg: float = NAV_CUBE_FOV_DEGREES,
) -> tuple[int, int] | None:
    """Widget-local pixel anchor for a face label (Slate parity)."""
    from math import cos as _cos
    from math import radians as _radians
    from math import sin as _sin

    w = h = int(size)
    az_r = _radians(float(azimuth_deg))
    el_r = _radians(float(elevation_deg))
    dist = _VIRTUAL_DISTANCE
    eye = (
        dist * _cos(el_r) * _cos(az_r),
        dist * _cos(el_r) * _sin(az_r),
        dist * _sin(el_r),
    )
    m = _view_matrix(eye, (0.0, 0.0, 0.0), world_up=(0.0, 0.0, 1.0))
    scale = _CUBE_HALF_SIZE
    vs = _transform_point(m, _mul(scale, _FACE_LABEL_POS[face_index]))
    if vs is None:
        return None
    return _perspective_project_viewspace(vs, fov_deg, w, h)


# ---------------------------------------------------------------------------
# Rasterization into the float32 RGBA viewport buffer
# ---------------------------------------------------------------------------

def _blend_pixel(
    rgba: np.ndarray, x: int, y: int, colour: tuple[float, float, float], alpha: float
) -> None:
    h, w, _ = rgba.shape
    if 0 <= x < w and 0 <= y < h:
        rgba[y, x, :3] = rgba[y, x, :3] * (1.0 - alpha) + np.asarray(
            colour, dtype=np.float32
        ) * alpha
        rgba[y, x, 3] = 1.0


def _draw_line_rgba(
    rgba: np.ndarray,
    p0: tuple[int, int],
    p1: tuple[int, int],
    colour: tuple[float, float, float],
    alpha: float = 1.0,
) -> None:
    x0, y0 = int(p0[0]), int(p0[1])
    x1, y1 = int(p1[0]), int(p1[1])
    steps = max(abs(x1 - x0), abs(y1 - y0), 1)
    xs = np.linspace(x0, x1, steps + 1).astype(np.int32)
    ys = np.linspace(y0, y1, steps + 1).astype(np.int32)
    h, w, _ = rgba.shape
    for x, y in zip(xs.tolist(), ys.tolist()):
        _blend_pixel(rgba, int(x), int(y), colour, alpha)


def _fill_convex_quad(
    rgba: np.ndarray,
    quad: list[tuple[int, int]],
    colour: tuple[float, float, float],
    alpha: float,
    offset_x: int = 0,
    offset_y: int = 0,
) -> None:
    """Scanline fill of a convex quad with per-pixel alpha blending."""
    pts = [(int(x) + offset_x, int(y) + offset_y) for x, y in quad]
    ys = [p[1] for p in pts]
    y_min, y_max = min(ys), max(ys)
    h, w, _ = rgba.shape
    y_min = max(y_min, 0)
    y_max = min(y_max, h - 1)
    edges = [(pts[i], pts[(i + 1) % 4]) for i in range(4)]
    for y in range(y_min, y_max + 1):
        xs: list[float] = []
        for (x0, y0), (x1, y1) in edges:
            if y0 == y1:
                continue
            if (y0 <= y < y1) or (y1 <= y < y0):
                t = (y - y0) / (y1 - y0)
                xs.append(x0 + t * (x1 - x0))
        if len(xs) < 2:
            continue
        x_start, x_end = int(math.ceil(min(xs))), int(math.floor(max(xs)))
        x_start = max(x_start, 0)
        x_end = min(x_end, w - 1)
        for x in range(x_start, x_end + 1):
            _blend_pixel(rgba, x, y, colour, alpha)


def _draw_labels(
    rgba: np.ndarray,
    quads: list[tuple[tuple[int, int, int], list[tuple[int, int]]]],
    azimuth_deg: float,
    elevation_deg: float,
    offset_x: int,
    offset_y: int,
    size: int = NAV_CUBE_SIZE,
) -> None:
    """Draw face labels (R/L/B/F/T/Bot) with PIL (Slate parity)."""
    from PIL import Image, ImageDraw, ImageFont

    h, w, _ = rgba.shape
    x0c = max(offset_x, 0)
    y0c = max(offset_y, 0)
    x1c = min(offset_x + size, w)
    y1c = min(offset_y + size, h)
    if x1c <= x0c or y1c <= y0c:
        return
    region = np.clip(rgba[y0c:y1c, x0c:x1c, :], 0.0, 1.0)
    img = Image.fromarray((region * 255.0).astype(np.uint8), mode="RGBA")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    normal_to_index = {n: i for i, n in enumerate(_FACE_NORMALS)}
    for normal_key, _quad in quads:
        idx = normal_to_index.get(tuple(normal_key))
        if idx is None:
            continue
        anchor = label_anchor(idx, azimuth_deg, elevation_deg, size)
        if anchor is None:
            continue
        lx = anchor[0] + offset_x - x0c
        ly = anchor[1] + offset_y - y0c
        text = _FACE_LABEL_TEXT[idx]
        label_rgb = _FACE_STYLE[tuple(normal_key)][2]
        fill = (
            int(label_rgb[0] * 255),
            int(label_rgb[1] * 255),
            int(label_rgb[2] * 255),
            255,
        )
        draw.text((float(lx), float(ly)), text, fill=fill, font=font, anchor="mm")
    back = np.asarray(img).astype(np.float32) / 255.0
    rgba[y0c:y1c, x0c:x1c, :] = back


def draw_navcube(
    rgba: np.ndarray,
    azimuth_deg: float,
    elevation_deg: float,
    size: int = NAV_CUBE_SIZE,
    margin: int = NAV_CUBE_MARGIN,
) -> tuple[int, int, int, int]:
    """Draw the orientation cube into the top-right of ``rgba``.

    Returns the pixel bounds ``(x0, y0, x1, y1)`` used (same convention as
    :func:`navcube_bounds`). Faces are drawn back-to-front with outlines,
    then labels -- mirroring ``NavCube.paintEvent`` draw order.
    """
    h, w, _ = rgba.shape
    if w < size + margin or h < size + margin:
        return navcube_bounds(w, h, size, margin)
    x0, y0, _x1, _y1 = navcube_bounds(w, h, size, margin)
    quads = project_cube(azimuth_deg, elevation_deg, size)

    for normal_key, quad in quads:
        fill, outline, _label = _FACE_STYLE.get(
            tuple(normal_key),
            ((0.4, 0.4, 0.4, 0.35), (0.25, 0.25, 0.25, 0.8), (0.3, 0.3, 0.3)),
        )
        _fill_convex_quad(rgba, quad, fill[:3], fill[3], x0, y0)

    for normal_key, quad in quads:
        _fill, outline, _label = _FACE_STYLE.get(
            tuple(normal_key),
            ((0.4, 0.4, 0.4, 0.35), (0.25, 0.25, 0.25, 0.8), (0.3, 0.3, 0.3)),
        )
        pts = [(int(x) + x0, int(y) + y0) for x, y in quad]
        for i in range(4):
            _draw_line_rgba(rgba, pts[i], pts[(i + 1) % 4], outline[:3], outline[3])

    _draw_labels(rgba, quads, azimuth_deg, elevation_deg, x0, y0, size)
    return (x0, y0, x0 + size, y0 + size)

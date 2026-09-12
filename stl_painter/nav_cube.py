"""Viewport orientation cube (VTK ``vtkCameraOrientationWidget`` style).

Drawn on a Dear PyGui drawlist by the app; all geometry here is pure NumPy
so the projection / hit-test / snap math is unit-testable without a GUI.

Behaviour mirrors the reference widget:
- annotated cube with axis-coloured faces (+X red, +Y green, +Z blue, with
  dimmed -X/-Y/-Z backs) and white labels,
- click a face to snap the camera to that axis,
- drag anywhere on the cube to orbit,
- hover highlights the face under the cursor.
"""

from __future__ import annotations

import numpy as np

# Face definitions: (axis id, outward normal, corner indices, label).
# Corners of a unit cube, x right, y up, z toward viewer in cube space.
CORNERS = np.array(
    [
        (-1.0, -1.0, -1.0),  # 0
        (1.0, -1.0, -1.0),  # 1
        (1.0, 1.0, -1.0),  # 2
        (-1.0, 1.0, -1.0),  # 3
        (-1.0, -1.0, 1.0),  # 4
        (1.0, -1.0, 1.0),  # 5
        (1.0, 1.0, 1.0),  # 6
        (-1.0, 1.0, 1.0),  # 7
    ],
    dtype=np.float32,
)

# VTK-style axis colours (front faces full, back faces dimmed).
_AXIS_COLOURS: dict[str, tuple[int, int, int, int]] = {
    "xp": (227, 74, 89, 255),
    "xn": (110, 36, 43, 255),
    "yp": (130, 196, 55, 255),
    "yn": (63, 95, 27, 255),
    "zp": (64, 150, 255, 255),
    "zn": (31, 73, 124, 255),
}

FACES: tuple[dict[str, object], ...] = (
    {"axis": "xp", "normal": (1.0, 0.0, 0.0), "corners": (1, 2, 6, 5), "label": "X"},
    {"axis": "xn", "normal": (-1.0, 0.0, 0.0), "corners": (0, 4, 7, 3), "label": "X"},
    {"axis": "yp", "normal": (0.0, 1.0, 0.0), "corners": (3, 7, 6, 2), "label": "Y"},
    {"axis": "yn", "normal": (0.0, -1.0, 0.0), "corners": (0, 1, 5, 4), "label": "Y"},
    {"axis": "zp", "normal": (0.0, 0.0, 1.0), "corners": (4, 5, 6, 7), "label": "Z"},
    {"axis": "zn", "normal": (0.0, 0.0, -1.0), "corners": (0, 3, 2, 1), "label": "Z"},
)

# Camera azimuth/elevation targets per snapped face (matches OrbitCamera).
SNAP_ANGLES: dict[str, tuple[float, float]] = {
    "xp": (0.0, 0.0),
    "xn": (180.0, 0.0),
    "yp": (0.0, 89.0),
    "yn": (0.0, -89.0),
    "zp": (90.0, 0.0),
    "zn": (-90.0, 0.0),
}


def project_cube(
    view_rotation: np.ndarray,
    center: tuple[float, float],
    half_size: float,
) -> list[dict[str, object]]:
    """Project cube faces into 2D, sorted far-to-near for painter's order.

    ``view_rotation`` is the camera view matrix 3x3 (world -> camera space,
    camera looking down -z). Returns one dict per face with ``axis``,
    ``label``, ``polygon`` (4x2 px), ``depth``, ``visible`` and ``fill``.
    """
    rotation = np.asarray(view_rotation, dtype=np.float32).reshape(3, 3)
    camera_space = CORNERS @ rotation.T
    projected = np.empty((len(CORNERS), 2), dtype=np.float32)
    projected[:, 0] = center[0] + camera_space[:, 0] * half_size
    projected[:, 1] = center[1] - camera_space[:, 1] * half_size

    faces: list[dict[str, object]] = []
    for spec in FACES:
        normal = np.asarray(spec["normal"], dtype=np.float32)
        camera_normal = rotation @ normal
        corners = [int(index) for index in spec["corners"]]  # type: ignore[union-attr]
        depth = float(camera_space[corners, 2].mean())
        faces.append(
            {
                "axis": spec["axis"],
                "label": spec["label"],
                "polygon": projected[corners],
                "depth": depth,
                "visible": bool(camera_normal[2] > 1e-6),
                "fill": _AXIS_COLOURS[str(spec["axis"])],
            }
        )
    faces.sort(key=lambda face: float(face["depth"]))
    return faces


def _point_in_polygon(point: tuple[float, float], polygon: np.ndarray) -> bool:
    inside = False
    total = len(polygon)
    x, y = point
    for i in range(total):
        x0, y0 = float(polygon[i][0]), float(polygon[i][1])
        x1, y1 = float(polygon[(i + 1) % total][0]), float(polygon[(i + 1) % total][1])
        if (y0 > y) != (y1 > y):
            intersect = x0 + (x1 - x0) * (y - y0) / (y1 - y0)
            if x < intersect:
                inside = not inside
    return inside


def hit_test(
    mouse_pos: tuple[float, float],
    faces: list[dict[str, object]],
    orbit_radius: float,
    center: tuple[float, float],
) -> tuple[str, str | None]:
    """Hit-test the projected cube.

    Returns ``("axis", axis_id)`` for the topmost visible face under the
    cursor, ``("orbit", None)`` inside the cube radius, else ``("none", None)``.
    """
    for face in sorted(faces, key=lambda item: float(item["depth"]), reverse=True):
        if not bool(face["visible"]):
            continue
        polygon = np.asarray(face["polygon"], dtype=np.float32)
        if _point_in_polygon(mouse_pos, polygon):
            return "axis", str(face["axis"])
    dx = mouse_pos[0] - center[0]
    dy = mouse_pos[1] - center[1]
    if dx * dx + dy * dy <= (orbit_radius + 8.0) ** 2:
        return "orbit", None
    return "none", None


def snap_angles(axis: str) -> tuple[float, float]:
    """Camera (azimuth, elevation) for a snapped cube face."""
    return SNAP_ANGLES[axis]


def face_centers_2d(
    faces: list[dict[str, object]],
) -> dict[str, tuple[float, float]]:
    """Mean 2D position of each projected face (label anchor points)."""
    centers: dict[str, tuple[float, float]] = {}
    for face in faces:
        polygon = np.asarray(face["polygon"], dtype=np.float32)
        mean = polygon.mean(axis=0)
        centers[str(face["axis"])] = (float(mean[0]), float(mean[1]))
    return centers

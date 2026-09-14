from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .camera import OrbitCamera
from .mesh_model import MeshModel

logger = logging.getLogger(__name__)
_CPU_PICK_WARNING_EMITTED = False


@dataclass(slots=True)
class PickResult:
    face_id: int
    location: np.ndarray
    distance: float


def _pick_face_location_fallback(
    mesh_model: MeshModel, origin: np.ndarray, direction: np.ndarray
) -> PickResult | None:
    triangles = mesh_model.vertices[mesh_model.faces].astype(np.float32, copy=False)
    v0 = triangles[:, 0]
    edge1 = triangles[:, 1] - v0
    edge2 = triangles[:, 2] - v0

    # Vectorized Moller-Trumbore ray/triangle test so painting still works
    # when trimesh's accelerated ray helper depends on an unavailable package.
    pvec = np.cross(np.broadcast_to(direction, edge2.shape), edge2)
    det = np.einsum("ij,ij->i", edge1, pvec)
    epsilon = 1e-8
    valid = np.abs(det) > epsilon
    if not np.any(valid):
        return None

    inv_det = np.zeros_like(det)
    inv_det[valid] = 1.0 / det[valid]
    tvec = origin.astype(np.float32) - v0
    u = np.einsum("ij,ij->i", tvec, pvec) * inv_det
    valid &= (u >= 0.0) & (u <= 1.0)
    if not np.any(valid):
        return None

    qvec = np.cross(tvec, edge1)
    direction_stack = np.broadcast_to(direction, qvec.shape)
    v = np.einsum("ij,ij->i", direction_stack, qvec) * inv_det
    valid &= (v >= 0.0) & ((u + v) <= 1.0)
    if not np.any(valid):
        return None

    t = np.einsum("ij,ij->i", edge2, qvec) * inv_det
    valid &= t >= epsilon
    if not np.any(valid):
        return None

    candidates = np.flatnonzero(valid)
    nearest_index = int(candidates[np.argmin(t[candidates])])
    distance = float(t[nearest_index])
    location = origin + direction * distance
    return PickResult(
        face_id=nearest_index,
        location=location.astype(np.float32),
        distance=distance,
    )


def pick_face_cpu(mesh_model: MeshModel, camera: OrbitCamera, mouse_x: float, mouse_y: float, viewport_size: tuple[int, int]) -> int | None:
    result = pick_face_location_cpu(mesh_model, camera, mouse_x, mouse_y, viewport_size)
    return result.face_id if result is not None else None


def pick_face_location_cpu(
    mesh_model: MeshModel,
    camera: OrbitCamera,
    mouse_x: float,
    mouse_y: float,
    viewport_size: tuple[int, int],
) -> PickResult | None:
    global _CPU_PICK_WARNING_EMITTED
    origin, direction = camera.unproject_ray(mouse_x, mouse_y, viewport_size)
    # Shared cached mesh: mesh() copies per call, forcing trimesh/rtree to
    # rebuild its spatial index (~3.4s on 285k faces) on every hover move.
    mesh = mesh_model.mesh_for_ray()
    try:
        locations, _, face_ids = mesh.ray.intersects_location(
            ray_origins=np.asarray([origin], dtype=np.float32),
            ray_directions=np.asarray([direction], dtype=np.float32),
            multiple_hits=True,
        )
    except Exception as exc:
        # Never fail a pick silently: the accelerated ray backend depends on
        # optional native packages (rtree/embree/scipy) that may be missing
        # or broken in frozen builds, so always fall back to the dependency-
        # free vectorized Moller-Trumbore test.
        if not _CPU_PICK_WARNING_EMITTED:
            logger.warning(
                "Accelerated picking unavailable (%s); using fallback ray test",
                exc,
            )
            _CPU_PICK_WARNING_EMITTED = True
        return _pick_face_location_fallback(mesh_model, origin, direction)
    if len(face_ids) == 0:
        return None
    distances = np.linalg.norm(locations - origin[None, :], axis=1)
    index = int(np.argmin(distances))
    return PickResult(
        face_id=int(face_ids[index]),
        location=locations[index].astype(np.float32),
        distance=float(distances[index]),
    )

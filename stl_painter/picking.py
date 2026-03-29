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
    mesh = mesh_model.mesh()
    try:
        locations, _, face_ids = mesh.ray.intersects_location(
            ray_origins=np.asarray([origin], dtype=np.float32),
            ray_directions=np.asarray([direction], dtype=np.float32),
            multiple_hits=True,
        )
    except ModuleNotFoundError as exc:
        if not _CPU_PICK_WARNING_EMITTED:
            logger.warning(
                "CPU picking disabled because an optional dependency is missing: %s",
                exc,
            )
            _CPU_PICK_WARNING_EMITTED = True
        return None
    except Exception:
        logger.exception("CPU picking failed unexpectedly")
        return None
    if len(face_ids) == 0:
        return None
    distances = np.linalg.norm(locations - origin[None, :], axis=1)
    index = int(np.argmin(distances))
    return PickResult(
        face_id=int(face_ids[index]),
        location=locations[index].astype(np.float32),
        distance=float(distances[index]),
    )

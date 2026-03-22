from __future__ import annotations

import numpy as np

from .camera import OrbitCamera
from .mesh_model import MeshModel


def pick_face_cpu(mesh_model: MeshModel, camera: OrbitCamera, mouse_x: float, mouse_y: float, viewport_size: tuple[int, int]) -> int | None:
    origin, direction = camera.unproject_ray(mouse_x, mouse_y, viewport_size)
    mesh = mesh_model.mesh()
    locations, _, face_ids = mesh.ray.intersects_location(
        ray_origins=np.asarray([origin], dtype=np.float32),
        ray_directions=np.asarray([direction], dtype=np.float32),
        multiple_hits=True,
    )
    if len(face_ids) == 0:
        return None
    distances = np.linalg.norm(locations - origin[None, :], axis=1)
    index = int(np.argmin(distances))
    return int(face_ids[index])

from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians, sin, tan

import numpy as np


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def perspective(fov_y_degrees: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / tan(radians(fov_y_degrees) / 2.0)
    matrix = np.zeros((4, 4), dtype=np.float32)
    matrix[0, 0] = f / aspect
    matrix[1, 1] = f
    matrix[2, 2] = (far + near) / (near - far)
    matrix[2, 3] = (2 * far * near) / (near - far)
    matrix[3, 2] = -1.0
    return matrix


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = _normalize(target - eye)
    right = _normalize(np.cross(forward, up))
    camera_up = np.cross(right, forward)
    matrix = np.eye(4, dtype=np.float32)
    matrix[0, :3] = right
    matrix[1, :3] = camera_up
    matrix[2, :3] = -forward
    matrix[:3, 3] = -matrix[:3, :3] @ eye
    return matrix


@dataclass(slots=True)
class OrbitCamera:
    target: np.ndarray
    distance: float = 3.0
    azimuth: float = 45.0
    elevation: float = 30.0
    fov_y_degrees: float = 45.0
    near: float = 0.01
    far: float = 1000.0

    def __post_init__(self) -> None:
        self.target = np.asarray(self.target, dtype=np.float32)

    @classmethod
    def for_mesh(cls, vertices: np.ndarray) -> "OrbitCamera":
        mins = vertices.min(axis=0)
        maxs = vertices.max(axis=0)
        target = ((mins + maxs) / 2.0).astype(np.float32)
        diagonal = np.linalg.norm(maxs - mins)
        return cls(target=target, distance=max(1.0, float(diagonal) * 1.5))

    def position(self) -> np.ndarray:
        azimuth = radians(self.azimuth)
        elevation = radians(self.elevation)
        x = self.distance * cos(elevation) * cos(azimuth)
        y = self.distance * sin(elevation)
        z = self.distance * cos(elevation) * sin(azimuth)
        return self.target + np.array([x, y, z], dtype=np.float32)

    def view_matrix(self) -> np.ndarray:
        return look_at(self.position(), self.target, np.array([0.0, 1.0, 0.0], dtype=np.float32))

    def projection_matrix(self, viewport_size: tuple[int, int]) -> np.ndarray:
        width, height = viewport_size
        aspect = max(width, 1) / max(height, 1)
        return perspective(self.fov_y_degrees, aspect, self.near, self.far)

    def mvp_matrix(self, viewport_size: tuple[int, int]) -> np.ndarray:
        return self.projection_matrix(viewport_size) @ self.view_matrix()

    def orbit(self, delta_x: float, delta_y: float) -> None:
        self.azimuth += delta_x
        self.elevation = float(np.clip(self.elevation + delta_y, -89.0, 89.0))

    def zoom(self, delta: float) -> None:
        self.distance = max(0.05, self.distance * (1.0 - delta))

    def pan(self, delta_x: float, delta_y: float) -> None:
        eye = self.position()
        forward = _normalize(self.target - eye)
        right = _normalize(np.cross(forward, np.array([0.0, 1.0, 0.0], dtype=np.float32)))
        up = _normalize(np.cross(right, forward))
        scale = self.distance * 0.0025
        offset = (-right * delta_x + up * delta_y) * scale
        self.target += offset

    def unproject_ray(self, mouse_x: float, mouse_y: float, viewport_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        width, height = viewport_size
        x = (2.0 * mouse_x) / max(width, 1) - 1.0
        y = 1.0 - (2.0 * mouse_y) / max(height, 1)
        near_point = np.array([x, y, -1.0, 1.0], dtype=np.float32)
        far_point = np.array([x, y, 1.0, 1.0], dtype=np.float32)
        inv = np.linalg.inv(self.projection_matrix(viewport_size) @ self.view_matrix())
        near_world = inv @ near_point
        far_world = inv @ far_point
        near_world /= near_world[3]
        far_world /= far_world[3]
        origin = near_world[:3].astype(np.float32)
        direction = _normalize((far_world[:3] - near_world[:3]).astype(np.float32))
        return origin, direction

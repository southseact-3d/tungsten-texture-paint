from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import trimesh

from .color_utils import DEFAULT_COLOR, Color, clamp_color


@dataclass(slots=True)
class Stroke:
    kind: str
    data: dict[str, Any]
    camera_matrix: np.ndarray


@dataclass(slots=True)
class MeshModel:
    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray
    face_colours: dict[int, Color] = field(default_factory=dict)
    overlay_strokes: list[Stroke] = field(default_factory=list)
    default_colour: Color = DEFAULT_COLOR
    source_path: str | None = None
    _adjacency: dict[int, set[int]] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=np.float32)
        self.faces = np.asarray(self.faces, dtype=np.int32)
        self.normals = np.asarray(self.normals, dtype=np.float32)
        self.face_colours = {int(face_id): clamp_color(colour) for face_id, colour in self.face_colours.items()}
        self.default_colour = clamp_color(self.default_colour)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError(f"Expected vertices shaped (n, 3), got {self.vertices.shape}")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError(f"Expected triangular faces shaped (n, 3), got {self.faces.shape}")
        if self.face_count == 0:
            raise ValueError("Mesh does not contain any faces after import cleanup")
        if self.normals.shape != (self.face_count, 3):
            raise ValueError(
                f"Expected normals shaped ({self.face_count}, 3), got {self.normals.shape}"
            )

    @classmethod
    def from_trimesh(cls, mesh: trimesh.Trimesh, *, source_path: str | None = None) -> "MeshModel":
        deduped = mesh.copy()
        deduped.update_faces(deduped.unique_faces())
        deduped.remove_unreferenced_vertices()
        deduped.merge_vertices()
        deduped.update_faces(deduped.nondegenerate_faces())
        deduped.remove_unreferenced_vertices()
        deduped.remove_infinite_values()
        unique_faces = deduped.unique_faces()
        if unique_faces is not None:
            deduped.update_faces(unique_faces)
            deduped.remove_unreferenced_vertices()
        normals = np.asarray(deduped.face_normals, dtype=np.float32)
        if normals.shape != (len(deduped.faces), 3) or not np.isfinite(normals).all():
            normals = np.asarray(deduped.triangles_cross, dtype=np.float32)
            lengths = np.linalg.norm(normals, axis=1, keepdims=True)
            safe_lengths = np.where(lengths > 1e-8, lengths, 1.0)
            normals = normals / safe_lengths
        return cls(
            vertices=deduped.vertices.view(np.ndarray),
            faces=deduped.faces.view(np.ndarray),
            normals=normals,
            source_path=source_path,
        )

    @property
    def face_count(self) -> int:
        return int(self.faces.shape[0])

    @property
    def vertex_count(self) -> int:
        return int(self.vertices.shape[0])

    def mesh(self) -> trimesh.Trimesh:
        return trimesh.Trimesh(vertices=self.vertices.copy(), faces=self.faces.copy(), process=False)

    def face_colour(self, face_id: int) -> Color:
        return self.face_colours.get(int(face_id), self.default_colour)

    def set_face_colour(self, face_id: int, colour: Color) -> None:
        self.face_colours[int(face_id)] = clamp_color(colour)

    def expanded_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        positions = self.vertices[self.faces].reshape(-1, 3).astype(np.float32)
        normals = np.repeat(self.normals, 3, axis=0).astype(np.float32)
        colours = np.vstack(
            [np.tile(np.array(self.face_colour(face_id), dtype=np.float32) / 255.0, (3, 1)) for face_id in range(self.face_count)]
        ).astype(np.float32)
        return positions, normals, colours

    def adjacency_map(self) -> dict[int, set[int]]:
        if self._adjacency is not None:
            return self._adjacency
        mesh = self.mesh()
        adjacency: dict[int, set[int]] = {face_id: set() for face_id in range(self.face_count)}
        for left, right in mesh.face_adjacency:
            adjacency[int(left)].add(int(right))
            adjacency[int(right)].add(int(left))
        self._adjacency = adjacency
        return adjacency

    def to_project_dict(self) -> dict[str, Any]:
        return {
            "vertices": self.vertices.tolist(),
            "faces": self.faces.tolist(),
            "normals": self.normals.tolist(),
            "face_colours": {str(face_id): list(colour) for face_id, colour in self.face_colours.items()},
            "default_colour": list(self.default_colour),
            "overlay_strokes": [
                {"kind": stroke.kind, "data": stroke.data, "camera_matrix": stroke.camera_matrix.tolist()}
                for stroke in self.overlay_strokes
            ],
            "source_path": self.source_path,
        }

    @classmethod
    def from_project_dict(cls, payload: dict[str, Any]) -> "MeshModel":
        overlay_strokes = [
            Stroke(kind=item["kind"], data=item["data"], camera_matrix=np.asarray(item["camera_matrix"], dtype=np.float32))
            for item in payload.get("overlay_strokes", [])
        ]
        return cls(
            vertices=np.asarray(payload["vertices"], dtype=np.float32),
            faces=np.asarray(payload["faces"], dtype=np.int32),
            normals=np.asarray(payload["normals"], dtype=np.float32),
            face_colours={int(face_id): tuple(colour) for face_id, colour in payload.get("face_colours", {}).items()},
            default_colour=tuple(payload.get("default_colour", DEFAULT_COLOR)),
            overlay_strokes=overlay_strokes,
            source_path=payload.get("source_path"),
        )

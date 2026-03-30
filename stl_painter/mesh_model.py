from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import trimesh

from .color_utils import DEFAULT_COLOR, Color, clamp_color

PROJECT_VERSION = 4


@dataclass(slots=True)
class Stroke:
    kind: str
    data: dict[str, Any]
    camera_matrix: np.ndarray


@dataclass(slots=True)
class SketchPlane:
    origin: np.ndarray
    normal: np.ndarray
    tangent_u: np.ndarray
    tangent_v: np.ndarray
    anchor_face_id: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "origin": self.origin.tolist(),
            "normal": self.normal.tolist(),
            "tangent_u": self.tangent_u.tolist(),
            "tangent_v": self.tangent_v.tolist(),
            "anchor_face_id": self.anchor_face_id,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SketchPlane":
        return cls(
            origin=np.asarray(payload["origin"], dtype=np.float32),
            normal=np.asarray(payload["normal"], dtype=np.float32),
            tangent_u=np.asarray(payload["tangent_u"], dtype=np.float32),
            tangent_v=np.asarray(payload["tangent_v"], dtype=np.float32),
            anchor_face_id=int(payload["anchor_face_id"]),
        )


@dataclass(slots=True)
class SketchEntity:
    entity_id: str
    kind: str
    data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "kind": self.kind,
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SketchEntity":
        return cls(
            entity_id=str(payload["entity_id"]),
            kind=str(payload["kind"]),
            data=dict(payload["data"]),
        )


@dataclass(slots=True)
class SketchDocument:
    plane: SketchPlane
    entities: list[SketchEntity] = field(default_factory=list)
    selected_entity_id: str | None = None
    grid_size: float = 0.1
    snap_to_grid: bool = True
    snap_to_vertices: bool = True
    snap_to_edges: bool = True
    snap_to_entities: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "plane": self.plane.to_dict(),
            "entities": [entity.to_dict() for entity in self.entities],
            "selected_entity_id": self.selected_entity_id,
            "grid_size": self.grid_size,
            "snap_to_grid": self.snap_to_grid,
            "snap_to_vertices": self.snap_to_vertices,
            "snap_to_edges": self.snap_to_edges,
            "snap_to_entities": self.snap_to_entities,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SketchDocument":
        return cls(
            plane=SketchPlane.from_dict(payload["plane"]),
            entities=[
                SketchEntity.from_dict(item) for item in payload.get("entities", [])
            ],
            selected_entity_id=payload.get("selected_entity_id"),
            grid_size=float(payload.get("grid_size", 0.1)),
            snap_to_grid=bool(payload.get("snap_to_grid", True)),
            snap_to_vertices=bool(payload.get("snap_to_vertices", True)),
            snap_to_edges=bool(payload.get("snap_to_edges", True)),
            snap_to_entities=bool(payload.get("snap_to_entities", True)),
        )


@dataclass(slots=True)
class MeshModel:
    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray
    face_colours: dict[int, Color] = field(default_factory=dict)
    overlay_strokes: list[Stroke] = field(default_factory=list)
    legacy_overlay_strokes: list[Stroke] = field(default_factory=list)
    sketch_documents: list[SketchDocument] = field(default_factory=list)
    masked_faces: set[int] = field(default_factory=set)
    interaction_mode: str = "paint"
    default_colour: Color = DEFAULT_COLOR
    source_path: str | None = None
    face_groups: dict[str, list[int]] = field(default_factory=dict)
    face_to_group: dict[int, str] = field(default_factory=dict)
    model_scale: float = 1.0
    _adjacency: dict[int, set[int]] | None = field(default=None, init=False, repr=False)
    _mesh_cache: trimesh.Trimesh | None = field(default=None, init=False, repr=False)
    _face_centers: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=np.float32)
        self.faces = np.asarray(self.faces, dtype=np.int32)
        self.normals = np.asarray(self.normals, dtype=np.float32)
        self.face_colours = {
            int(face_id): clamp_color(colour)
            for face_id, colour in self.face_colours.items()
        }
        self.masked_faces = {int(face_id) for face_id in self.masked_faces}
        self.default_colour = clamp_color(self.default_colour)
        self.face_groups = {
            str(group_id): sorted({int(face_id) for face_id in faces})
            for group_id, faces in self.face_groups.items()
        }
        self.face_to_group = {
            int(face_id): str(group_id)
            for face_id, group_id in self.face_to_group.items()
        }
        self.model_scale = float(self.model_scale)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError(
                f"Expected vertices shaped (n, 3), got {self.vertices.shape}"
            )
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError(
                f"Expected triangular faces shaped (n, 3), got {self.faces.shape}"
            )
        if self.face_count == 0:
            raise ValueError("Mesh does not contain any faces after import cleanup")
        if self.normals.shape != (self.face_count, 3):
            raise ValueError(
                f"Expected normals shaped ({self.face_count}, 3), got {self.normals.shape}"
            )

    @classmethod
    def from_trimesh(
        cls, mesh: trimesh.Trimesh, *, source_path: str | None = None
    ) -> "MeshModel":
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
        face_colours: dict[int, Color] = {}
        face_rgba = getattr(getattr(deduped, "visual", None), "face_colors", None)
        if face_rgba is not None and len(face_rgba) == len(deduped.faces):
            raw = np.asarray(face_rgba, dtype=np.uint8)
            if raw.ndim == 2 and raw.shape[1] >= 3:
                if raw.shape[1] == 3:
                    alpha = np.full((raw.shape[0], 1), 255, dtype=np.uint8)
                    raw = np.hstack([raw, alpha])
                for face_id, colour in enumerate(raw):
                    rgba = (
                        int(colour[0]),
                        int(colour[1]),
                        int(colour[2]),
                        int(colour[3]),
                    )
                    if rgba != DEFAULT_COLOR:
                        face_colours[face_id] = rgba
        return cls(
            vertices=deduped.vertices.view(np.ndarray),
            faces=deduped.faces.view(np.ndarray),
            normals=normals,
            face_colours=face_colours,
            source_path=source_path,
        )

    def scale_uniform(self, factor: float) -> None:
        if factor <= 0:
            raise ValueError("Scale factor must be greater than zero")
        self.vertices = (self.vertices * float(factor)).astype(np.float32)
        self.model_scale *= float(factor)
        self._mesh_cache = None
        self._face_centers = None

    def map_colours_from(
        self, source: "MeshModel", *, normalize_scale: bool = True
    ) -> int:
        if self.face_count == 0 or source.face_count == 0:
            return 0

        source_centers = source.vertices[source.faces].mean(axis=1).astype(np.float32)
        target_centers = self.vertices[self.faces].mean(axis=1).astype(np.float32)

        if normalize_scale:
            src_min = source_centers.min(axis=0)
            src_extent = np.maximum(source_centers.max(axis=0) - src_min, 1e-6)
            source_centers = (source_centers - src_min) / src_extent

            tgt_min = target_centers.min(axis=0)
            tgt_extent = np.maximum(target_centers.max(axis=0) - tgt_min, 1e-6)
            target_centers = (target_centers - tgt_min) / tgt_extent

        source_colours = np.array(
            [source.face_colour(face_id) for face_id in range(source.face_count)],
            dtype=np.int32,
        )
        updates = 0
        chunk_size = 1024
        for start in range(0, self.face_count, chunk_size):
            end = min(self.face_count, start + chunk_size)
            batch = target_centers[start:end]
            deltas = batch[:, None, :] - source_centers[None, :, :]
            distances = np.einsum("ijk,ijk->ij", deltas, deltas)
            nearest = np.argmin(distances, axis=1)
            for offset, source_face_id in enumerate(nearest):
                target_face_id = start + offset
                colour = tuple(int(c) for c in source_colours[int(source_face_id)])
                if colour != self.face_colour(target_face_id):
                    self.face_colours[target_face_id] = colour
                    updates += 1
        return updates

    @property
    def face_count(self) -> int:
        return int(self.faces.shape[0])

    @property
    def vertex_count(self) -> int:
        return int(self.vertices.shape[0])

    def mesh(self) -> trimesh.Trimesh:
        if self._mesh_cache is None:
            self._mesh_cache = trimesh.Trimesh(
                vertices=self.vertices.copy(),
                faces=self.faces.copy(),
                process=False,
            )
        return self._mesh_cache.copy()

    def face_colour(self, face_id: int) -> Color:
        return self.face_colours.get(int(face_id), self.default_colour)

    def set_face_colour(self, face_id: int, colour: Color) -> None:
        self.face_colours[int(face_id)] = clamp_color(colour)

    def expanded_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        positions = self.vertices[self.faces].reshape(-1, 3).astype(np.float32)
        normals = np.repeat(self.normals, 3, axis=0).astype(np.float32)
        colours = np.vstack(
            [
                np.tile(
                    np.array(self.face_colour(face_id), dtype=np.float32) / 255.0,
                    (3, 1),
                )
                for face_id in range(self.face_count)
            ]
        ).astype(np.float32)
        return positions, normals, colours

    def adjacency_map(self) -> dict[int, set[int]]:
        if self._adjacency is not None:
            return self._adjacency
        mesh = self.mesh()
        adjacency: dict[int, set[int]] = {
            face_id: set() for face_id in range(self.face_count)
        }
        for left, right in mesh.face_adjacency:
            adjacency[int(left)].add(int(right))
            adjacency[int(right)].add(int(left))
        self._adjacency = adjacency
        return adjacency

    def compute_face_groups(
        self, angle_tolerance_degrees: float = 180.0
    ) -> dict[str, list[int]]:
        adjacency = self.adjacency_map()
        max_angle = np.deg2rad(float(max(0.0, min(180.0, angle_tolerance_degrees))))
        visited: set[int] = set()
        groups: dict[str, list[int]] = {}
        face_to_group: dict[int, str] = {}
        group_index = 0
        for start_face in range(self.face_count):
            if start_face in visited:
                continue
            group_id = f"group_{group_index}"
            group_index += 1
            queue = [start_face]
            members: list[int] = []
            while queue:
                face_id = queue.pop()
                if face_id in visited:
                    continue
                visited.add(face_id)
                members.append(face_id)
                n0 = self.normals[face_id]
                for neighbour in adjacency.get(face_id, set()):
                    if neighbour in visited:
                        continue
                    n1 = self.normals[neighbour]
                    dot = float(np.clip(np.dot(n0, n1), -1.0, 1.0))
                    if float(np.arccos(dot)) <= max_angle:
                        queue.append(neighbour)
            members.sort()
            groups[group_id] = members
            for face_id in members:
                face_to_group[face_id] = group_id
        self.face_groups = groups
        self.face_to_group = face_to_group
        return groups

    def group_for_face(self, face_id: int) -> str | None:
        if not self.face_to_group:
            self.compute_face_groups()
        return self.face_to_group.get(int(face_id))

    def faces_for_group(self, group_id: str) -> list[int]:
        if not self.face_groups:
            self.compute_face_groups()
        return list(self.face_groups.get(str(group_id), []))

    def get_internal_group_edges(self) -> set[tuple[int, int]]:
        if not self.face_groups or not self.face_to_group:
            return set()
        adjacency = self.adjacency_map()
        internal_edges: set[tuple[int, int]] = set()
        for face_id in range(self.face_count):
            group_id = self.face_to_group.get(face_id)
            if group_id is None:
                continue
            for neighbor in adjacency.get(face_id, set()):
                neighbor_group = self.face_to_group.get(neighbor)
                if neighbor_group == group_id:
                    edge = (min(face_id, neighbor), max(face_id, neighbor))
                    internal_edges.add(edge)
        return internal_edges

    def face_vertices(self, face_id: int) -> np.ndarray:
        return self.vertices[self.faces[int(face_id)]]

    def face_center(self, face_id: int) -> np.ndarray:
        if self._face_centers is None:
            self._face_centers = (
                self.vertices[self.faces].mean(axis=1).astype(np.float32)
            )
        return self._face_centers[int(face_id)]

    def mesh_extents(self) -> np.ndarray:
        return (self.vertices.max(axis=0) - self.vertices.min(axis=0)).astype(
            np.float32
        )

    def mesh_diagonal(self) -> float:
        return float(np.linalg.norm(self.mesh_extents()))

    def to_project_dict(self) -> dict[str, Any]:
        return {
            "project_version": PROJECT_VERSION,
            "vertices": self.vertices.tolist(),
            "faces": self.faces.tolist(),
            "normals": self.normals.tolist(),
            "face_colours": {
                str(face_id): list(colour)
                for face_id, colour in self.face_colours.items()
            },
            "default_colour": list(self.default_colour),
            "overlay_strokes": [
                {
                    "kind": stroke.kind,
                    "data": stroke.data,
                    "camera_matrix": stroke.camera_matrix.tolist(),
                }
                for stroke in self.overlay_strokes
            ],
            "legacy_overlay_strokes": [
                {
                    "kind": stroke.kind,
                    "data": stroke.data,
                    "camera_matrix": stroke.camera_matrix.tolist(),
                }
                for stroke in self.legacy_overlay_strokes
            ],
            "sketch_documents": [
                document.to_dict() for document in self.sketch_documents
            ],
            "masked_faces": sorted(self.masked_faces),
            "interaction_mode": self.interaction_mode,
            "source_path": self.source_path,
            "face_groups": {
                group_id: list(faces) for group_id, faces in self.face_groups.items()
            },
            "face_to_group": {
                str(face_id): group_id
                for face_id, group_id in self.face_to_group.items()
            },
            "model_scale": self.model_scale,
        }

    @classmethod
    def from_project_dict(cls, payload: dict[str, Any]) -> "MeshModel":
        overlay_strokes = [
            Stroke(
                kind=item["kind"],
                data=item["data"],
                camera_matrix=np.asarray(item["camera_matrix"], dtype=np.float32),
            )
            for item in payload.get("overlay_strokes", [])
        ]
        return cls(
            vertices=np.asarray(payload["vertices"], dtype=np.float32),
            faces=np.asarray(payload["faces"], dtype=np.int32),
            normals=np.asarray(payload["normals"], dtype=np.float32),
            face_colours={
                int(face_id): tuple(colour)
                for face_id, colour in payload.get("face_colours", {}).items()
            },
            default_colour=tuple(payload.get("default_colour", DEFAULT_COLOR)),
            overlay_strokes=[],
            legacy_overlay_strokes=overlay_strokes
            + [
                Stroke(
                    kind=item["kind"],
                    data=item["data"],
                    camera_matrix=np.asarray(item["camera_matrix"], dtype=np.float32),
                )
                for item in payload.get("legacy_overlay_strokes", [])
            ],
            sketch_documents=[
                SketchDocument.from_dict(item)
                for item in payload.get("sketch_documents", [])
            ],
            masked_faces=set(payload.get("masked_faces", [])),
            interaction_mode=str(payload.get("interaction_mode", "paint")),
            source_path=payload.get("source_path"),
            face_groups={
                str(group_id): [int(face_id) for face_id in faces]
                for group_id, faces in payload.get("face_groups", {}).items()
            },
            face_to_group={
                int(face_id): str(group_id)
                for face_id, group_id in payload.get("face_to_group", {}).items()
            },
            model_scale=float(payload.get("model_scale", 1.0)),
        )

    def to_project_delta(
        self, exclude_colors: set[tuple[int, int, int, int]] | None = None
    ) -> dict[str, Any]:
        if exclude_colors is None:
            exclude_colors = set()
        filtered_colors = {
            str(face_id): list(colour)
            for face_id, colour in self.face_colours.items()
            if colour != self.default_colour
            and colour not in exclude_colors
            and not (colour[0] == colour[1] == colour[2] and colour[3] == 255)
        }
        return {
            "face_colours": filtered_colors,
            "default_colour": list(self.default_colour),
            "overlay_strokes": [
                {
                    "kind": stroke.kind,
                    "data": stroke.data,
                    "camera_matrix": stroke.camera_matrix.tolist(),
                }
                for stroke in self.overlay_strokes
            ],
            "sketch_documents": [
                document.to_dict() for document in self.sketch_documents
            ],
            "masked_faces": sorted(self.masked_faces),
            "interaction_mode": self.interaction_mode,
            "face_groups": {
                group_id: list(faces) for group_id, faces in self.face_groups.items()
            },
            "face_to_group": {
                str(face_id): group_id
                for face_id, group_id in self.face_to_group.items()
            },
            "model_scale": self.model_scale,
        }

    @classmethod
    def from_project_delta(
        cls, base: "MeshModel", delta: dict[str, Any]
    ) -> "MeshModel":
        merged_colors = dict(base.face_colours)
        delta_colors = delta.get("face_colours", {})
        for face_id, colour in delta_colors.items():
            merged_colors[int(face_id)] = tuple(colour)
        overlay_strokes = [
            Stroke(
                kind=item["kind"],
                data=item["data"],
                camera_matrix=np.asarray(item["camera_matrix"], dtype=np.float32),
            )
            for item in delta.get("overlay_strokes", [])
        ]
        return cls(
            vertices=base.vertices.copy(),
            faces=base.faces.copy(),
            normals=base.normals.copy(),
            face_colours=merged_colors,
            default_colour=tuple(delta.get("default_colour", base.default_colour)),
            overlay_strokes=[],
            legacy_overlay_strokes=overlay_strokes,
            sketch_documents=[
                SketchDocument.from_dict(item)
                for item in delta.get("sketch_documents", [])
            ],
            masked_faces=set(delta.get("masked_faces", base.masked_faces)),
            interaction_mode=str(delta.get("interaction_mode", base.interaction_mode)),
            source_path=base.source_path,
            face_groups=delta.get("face_groups", base.face_groups),
            face_to_group=delta.get("face_to_group", base.face_to_group),
            model_scale=float(delta.get("model_scale", base.model_scale)),
        )

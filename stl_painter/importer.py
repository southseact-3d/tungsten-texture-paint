from __future__ import annotations

import logging
from pathlib import Path

import trimesh

from .mesh_model import MeshModel

logger = logging.getLogger(__name__)
SUPPORTED_IMPORT_EXTENSIONS = {".stl", ".obj", ".glb", ".gltf"}


def _coerce_trimesh(loaded: trimesh.Trimesh | trimesh.Scene) -> trimesh.Trimesh:
    if isinstance(loaded, trimesh.Trimesh):
        return loaded
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError("The selected file did not contain any mesh geometry")
        combined = trimesh.util.concatenate(tuple(loaded.geometry.values()))
        if not isinstance(combined, trimesh.Trimesh):
            raise ValueError("Failed to combine scene geometry into a mesh")
        return combined
    raise ValueError(f"Expected a mesh, got {type(loaded)!r}")


def load_stl(path: str | Path) -> MeshModel:
    source_path = str(path)
    logger.info("Loading STL from %s", source_path)
    loaded = _coerce_trimesh(trimesh.load(source_path, force="mesh"))
    logger.info(
        "Raw STL stats | faces=%s | vertices=%s | watertight=%s | extents=%s",
        len(loaded.faces),
        len(loaded.vertices),
        loaded.is_watertight,
        tuple(float(value) for value in loaded.extents),
    )
    if len(loaded.faces) == 0 or len(loaded.vertices) == 0:
        raise ValueError("The selected STL did not contain any renderable triangles")
    logger.info(
        "Successfully loaded STL with %s faces and %s vertices",
        len(loaded.faces),
        len(loaded.vertices),
    )
    mesh_model = MeshModel.from_trimesh(loaded, source_path=source_path)
    logger.info(
        "Prepared mesh model | faces=%s | vertices=%s | source=%s",
        mesh_model.face_count,
        mesh_model.vertex_count,
        source_path,
    )
    return mesh_model


def import_stl(path: str | Path) -> MeshModel:
    return load_stl(path)


def load_model(path: str | Path) -> MeshModel:
    source_path = str(path)
    extension = Path(path).suffix.lower()
    if extension not in SUPPORTED_IMPORT_EXTENSIONS:
        raise ValueError(
            f"Unsupported import format '{extension}'. Supported: {sorted(SUPPORTED_IMPORT_EXTENSIONS)}"
        )
    logger.info("Loading model from %s", source_path)
    mesh = _coerce_trimesh(trimesh.load(source_path, force="scene"))
    if len(mesh.faces) == 0 or len(mesh.vertices) == 0:
        raise ValueError("The selected model did not contain any renderable triangles")
    mesh_model = MeshModel.from_trimesh(mesh, source_path=source_path)
    logger.info(
        "Prepared mesh model | faces=%s | vertices=%s | source=%s",
        mesh_model.face_count,
        mesh_model.vertex_count,
        source_path,
    )
    return mesh_model

from __future__ import annotations

import logging
from pathlib import Path

import trimesh

from .mesh_model import MeshModel

logger = logging.getLogger(__name__)


def load_stl(path: str | Path) -> MeshModel:
    source_path = str(path)
    logger.info("Loading STL from %s", source_path)
    loaded = trimesh.load(source_path, force="mesh")
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f"Expected a mesh, got {type(loaded)!r}")
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

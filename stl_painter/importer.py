from __future__ import annotations

import logging
from pathlib import Path

import trimesh

from .mesh_model import MeshModel

logger = logging.getLogger(__name__)


def load_stl(path: str | Path) -> MeshModel:
    source_path = str(path)
    logger.info(f"Loading STL from {source_path}")
    loaded = trimesh.load(source_path, force="mesh")
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f"Expected a mesh, got {type(loaded)!r}")
    logger.info(
        f"Successfully loaded STL with {len(loaded.faces)} faces and {len(loaded.vertices)} vertices"
    )
    return MeshModel.from_trimesh(loaded, source_path=source_path)

from __future__ import annotations

from pathlib import Path

import trimesh

from .mesh_model import MeshModel


def load_stl(path: str | Path) -> MeshModel:
    source_path = str(path)
    loaded = trimesh.load(source_path, force="mesh")
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f"Expected a mesh, got {type(loaded)!r}")
    return MeshModel.from_trimesh(loaded, source_path=source_path)

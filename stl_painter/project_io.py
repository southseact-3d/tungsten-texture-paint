from __future__ import annotations

import json
from pathlib import Path

from .mesh_model import MeshModel


def save_project(path: str | Path, mesh_model: MeshModel) -> None:
    Path(path).write_text(json.dumps(mesh_model.to_project_dict(), indent=2), encoding="utf-8")


def load_project(path: str | Path) -> MeshModel:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return MeshModel.from_project_dict(payload)

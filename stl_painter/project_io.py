from __future__ import annotations

import json
from pathlib import Path

from .mesh_model import MeshModel, PROJECT_VERSION


def migrate_project_payload(payload: dict[str, object]) -> dict[str, object]:
    version = int(payload.get("project_version", 1))
    migrated = dict(payload)
    if version < 2:
        migrated["project_version"] = 2
        version = 2
    if version < 3:
        migrated.setdefault("face_groups", {})
        migrated.setdefault("face_to_group", {})
        migrated["project_version"] = 3
    if int(migrated.get("project_version", 1)) != PROJECT_VERSION:
        raise ValueError(
            f"Unsupported project version {migrated.get('project_version')} (expected {PROJECT_VERSION})"
        )
    return migrated


def save_project(path: str | Path, mesh_model: MeshModel) -> None:
    Path(path).write_text(json.dumps(mesh_model.to_project_dict(), indent=2), encoding="utf-8")


def load_project(path: str | Path) -> MeshModel:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return MeshModel.from_project_dict(migrate_project_payload(payload))

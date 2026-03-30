from __future__ import annotations

import json
from pathlib import Path

from .mesh_model import MeshModel, PROJECT_VERSION

PROJECT_EXTENSION = ".tg3d"


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
        version = 3
    if version < 4:
        migrated.setdefault("model_scale", 1.0)
        migrated["project_version"] = 4
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


def save_tg3d(path: str | Path, mesh_model: MeshModel, timeline: dict[str, object] | None = None) -> None:
    payload = {
        "tg3d_version": 1,
        "model": mesh_model.to_project_dict(),
        "timeline": timeline or {"current_index": 0, "descriptions": ["Initial state"], "snapshots": [mesh_model.to_project_dict()]},
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_tg3d(path: str | Path) -> tuple[MeshModel, dict[str, object]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    version = int(payload.get("tg3d_version", 0))
    if version != 1:
        raise ValueError(f"Unsupported .tg3d version {version}")
    model_payload = migrate_project_payload(dict(payload["model"]))
    timeline = dict(payload.get("timeline", {}))
    return MeshModel.from_project_dict(model_payload), timeline

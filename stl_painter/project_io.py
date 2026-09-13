from __future__ import annotations

import json
from collections import Counter
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
        version = 4
    if version < 5:
        migrated.setdefault("tri_to_cad", None)
        migrated.setdefault("cad_faces", {})
        migrated["project_version"] = 5
    if int(migrated.get("project_version", 1)) != PROJECT_VERSION:
        raise ValueError(
            f"Unsupported project version {migrated.get('project_version')} (expected {PROJECT_VERSION})"
        )
    return migrated


def save_project(path: str | Path, mesh_model: MeshModel) -> None:
    Path(path).write_text(
        json.dumps(mesh_model.to_project_dict(), indent=2), encoding="utf-8"
    )


def load_project(path: str | Path) -> MeshModel:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return MeshModel.from_project_dict(migrate_project_payload(payload))


def _snapshot_to_delta(
    prev_snapshot: dict[str, object] | None,
    current: MeshModel,
    exclude_colors: set[tuple[int, int, int, int]] | None = None,
) -> dict[str, object]:
    current_delta = current.to_project_delta(exclude_colors)
    if prev_snapshot is None:
        return current_delta
    delta = {}
    prev_raw_colors = prev_snapshot.get("face_colours", {})
    prev_colors = {
        k: tuple(v)
        for k, v in prev_raw_colors.items()
        if tuple(v) not in (exclude_colors or set())
    }
    curr_colors = current_delta.get("face_colours", {})
    changed_colors = {
        str(face_id): list(colour)
        for face_id, colour in curr_colors.items()
        if prev_colors.get(str(face_id)) is None
        or tuple(prev_colors.get(str(face_id))) != tuple(colour)
    }
    if changed_colors:
        delta["face_colours"] = changed_colors
    prev_default = prev_snapshot.get("default_colour")
    curr_default = current_delta.get("default_colour")
    if curr_default != prev_default:
        delta["default_colour"] = curr_default
    prev_masks = sorted(prev_snapshot.get("masked_faces", []))
    curr_masks = current_delta.get("masked_faces", [])
    if curr_masks != prev_masks:
        delta["masked_faces"] = curr_masks
    prev_mode = prev_snapshot.get("interaction_mode")
    curr_mode = current_delta.get("interaction_mode")
    if curr_mode != prev_mode:
        delta["interaction_mode"] = curr_mode
    prev_groups = prev_snapshot.get("face_groups", {})
    curr_groups = current_delta.get("face_groups", {})
    if curr_groups != prev_groups:
        delta["face_groups"] = curr_groups
    prev_f2g = prev_snapshot.get("face_to_group", {})
    curr_f2g = current_delta.get("face_to_group", {})
    if curr_f2g != prev_f2g:
        delta["face_to_group"] = curr_f2g
    prev_scale = prev_snapshot.get("model_scale")
    curr_scale = current_delta.get("model_scale")
    if curr_scale != prev_scale:
        delta["model_scale"] = curr_scale
    prev_sketches = prev_snapshot.get("sketch_documents", [])
    curr_sketches = current_delta.get("sketch_documents", [])
    if curr_sketches != prev_sketches:
        delta["sketch_documents"] = curr_sketches
    prev_strokes = prev_snapshot.get("overlay_strokes", [])
    curr_strokes = current_delta.get("overlay_strokes", [])
    if curr_strokes != prev_strokes:
        delta["overlay_strokes"] = curr_strokes
    return delta


def save_tg3d(
    path: str | Path, mesh_model: MeshModel, timeline: dict[str, object] | None = None
) -> None:
    timeline_data = timeline or {
        "current_index": 0,
        "descriptions": ["Initial state"],
        "snapshots": [],
    }
    raw_snapshots = timeline_data.get("snapshots", [])
    deltas: list[dict[str, object]] = []
    prev_snapshot: dict[str, object] | None = None
    color_counts = Counter(mesh_model.face_colours.values())
    exclude_colors: set[tuple[int, int, int, int]] = set()
    if color_counts:
        common_color, count = color_counts.most_common(1)[0]
        if count > len(mesh_model.face_colours) * 0.5:
            exclude_colors = {common_color}
    for idx, snapshot in enumerate(raw_snapshots):
        if isinstance(snapshot, dict) and "vertices" in snapshot:
            snapshot_model = MeshModel.from_project_dict(
                migrate_project_payload(snapshot)
            )
            if idx == 0:
                delta = _snapshot_to_delta(None, snapshot_model, exclude_colors)
            else:
                delta = _snapshot_to_delta(
                    prev_snapshot, snapshot_model, exclude_colors
                )
        elif isinstance(snapshot, dict):
            delta = snapshot
        else:
            delta = {}
        deltas.append(delta)
        prev_snapshot = snapshot if isinstance(snapshot, dict) else None
    model_dict = mesh_model.to_project_dict()
    model_dict["face_colours"] = {}
    payload = {
        "tg3d_version": 2,
        "model": model_dict,
        "timeline": {
            "current_index": timeline_data.get("current_index", 0),
            "descriptions": timeline_data.get("descriptions", ["Initial state"]),
            "snapshots": deltas,
        },
    }
    Path(path).write_text(json.dumps(payload), encoding="utf-8")


def load_tg3d(path: str | Path) -> tuple[MeshModel, dict[str, object]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    version = int(payload.get("tg3d_version", 0))
    if version == 0:
        version = int(payload.get("tg3d_version", 1))
    if version not in (1, 2):
        raise ValueError(f"Unsupported .tg3d version {version}")
    model_payload = migrate_project_payload(dict(payload["model"]))
    base_model = MeshModel.from_project_dict(model_payload)
    timeline = dict(payload.get("timeline", {}))
    raw_snapshots = timeline.get("snapshots", [])
    if version == 1:
        timeline["snapshots"] = raw_snapshots
        return base_model, timeline
    deltas = raw_snapshots
    reconstructed_snapshots: list[dict[str, object]] = []
    current_model = base_model
    for delta in deltas:
        current_model = MeshModel.from_project_delta(current_model, delta)
        reconstructed_snapshots.append(current_model.to_project_dict())
    timeline["snapshots"] = reconstructed_snapshots
    return base_model, timeline

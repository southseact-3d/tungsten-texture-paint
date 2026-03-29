from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def app_config_path() -> Path:
    return Path.cwd() / ".stl_texture_painter.local.json"


def load_local_config() -> dict[str, Any]:
    path = app_config_path()
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_local_config(payload: dict[str, Any]) -> None:
    path = app_config_path()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

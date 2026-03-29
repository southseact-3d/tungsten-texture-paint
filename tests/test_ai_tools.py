from __future__ import annotations

import json
from pathlib import Path

import pytest

from stl_painter.ai_tools import AIToolContext, run_tool
from stl_painter.commands import AppCommands
from stl_painter.interaction_state import InteractionState
from stl_painter.paint_tool import PaintTool
from stl_painter.sketch_tool import SketchTool


def _context(square_mesh) -> AIToolContext:
    return AIToolContext(
        mesh_model=square_mesh,
        interaction_state=InteractionState(),
        commands=AppCommands(square_mesh, PaintTool(square_mesh)),
        paint_tool=PaintTool(square_mesh),
        sketch_tool=SketchTool(),
        capture_viewport=lambda: Path("snapshot.png"),
    )


def test_ai_tool_rejects_invalid_face(square_mesh) -> None:
    context = _context(square_mesh)
    with pytest.raises(ValueError, match="out of range"):
        run_tool(context, "create_sketch_plane", json.dumps({"face_id": 99}))


def test_ai_tool_creates_sketch_and_bakes(square_mesh) -> None:
    context = _context(square_mesh)
    run_tool(context, "create_sketch_plane", json.dumps({"face_id": 0}))
    run_tool(context, "add_rect_sketch", json.dumps({"min_uv": [0.0, 0.0], "max_uv": [0.5, 0.5]}))
    result = json.loads(run_tool(context, "bake_sketch", "{}"))

    assert result["ok"] is True

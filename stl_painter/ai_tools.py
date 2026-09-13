from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from .commands import AppCommands
from .color_utils import clamp_color
from .exporter import export_3mf
from .interaction_state import InteractionState
from .mesh_model import MeshModel
from .paint_tool import PaintTool
from .sketch_tool import SketchTool


@dataclass(slots=True)
class AIToolContext:
    mesh_model: MeshModel | None
    interaction_state: InteractionState
    commands: AppCommands
    paint_tool: PaintTool | None
    sketch_tool: SketchTool
    capture_viewport: Callable[[], Path]


def tool_specs() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "get_mesh_state",
                "description": "Inspect the currently loaded mesh and active tool state.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "set_mode",
                "description": "Switch between paint and sketch modes.",
                "parameters": {
                    "type": "object",
                    "properties": {"mode": {"type": "string", "enum": ["paint", "sketch"]}},
                    "required": ["mode"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "set_active_color",
                "description": "Set the active RGBA color.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "rgba": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 4,
                            "maxItems": 4,
                        }
                    },
                    "required": ["rgba"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "create_sketch_plane",
                "description": "Create a sketch plane from a face id.",
                "parameters": {
                    "type": "object",
                    "properties": {"face_id": {"type": "integer"}},
                    "required": ["face_id"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "add_rect_sketch",
                "description": "Add a rectangle entity in plane coordinates.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "min_uv": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                        "max_uv": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                    },
                    "required": ["min_uv", "max_uv"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "bake_sketch",
                "description": "Bake the active sketch document onto mesh face colors.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "paint_faces",
                "description": "Paint explicit face ids with the active color.",
                "parameters": {
                    "type": "object",
                    "properties": {"face_ids": {"type": "array", "items": {"type": "integer"}}},
                    "required": ["face_ids"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "export_project_snapshot",
                "description": "Capture the current viewport snapshot to a PNG path and return that path.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "export_3mf_file",
                "description": "Export the current mesh to a 3MF file path.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "undo",
                "description": "Undo the last command.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        },
    ]


def _require_mesh(context: AIToolContext) -> MeshModel:
    if context.mesh_model is None:
        raise ValueError("No mesh is currently loaded")
    return context.mesh_model


def run_tool(context: AIToolContext, name: str, arguments_json: str) -> str:
    arguments = json.loads(arguments_json or "{}")
    if name == "get_mesh_state":
        mesh = context.mesh_model
        if mesh is None:
            return json.dumps({"loaded": False})
        return json.dumps(
            {
                "loaded": True,
                "faces": mesh.face_count,
                "vertices": mesh.vertex_count,
                "masked_faces": len(mesh.masked_faces),
                "mode": context.interaction_state.interaction_mode,
                "paint_tool": context.interaction_state.paint_tool,
                "sketch_tool": context.interaction_state.sketch_tool,
                "active_colour": list(context.interaction_state.active_colour),
            }
        )
    if name == "set_mode":
        context.interaction_state.interaction_mode = arguments["mode"]
        return json.dumps({"ok": True, "mode": context.interaction_state.interaction_mode})
    if name == "set_active_color":
        from .color_utils import rgba_hex, rgb_hex

        context.interaction_state.active_colour = clamp_color(arguments["rgba"])
        try:
            import dearpygui.dearpygui as dpg

            active = list(context.interaction_state.active_colour)
            for tag, value in (
                ("active_colour_preview", active),
                ("active_colour_picker", active),
            ):
                try:
                    if dpg.does_item_exist(tag):
                        dpg.set_value(tag, value)
                except Exception:
                    pass
            try:
                if dpg.does_item_exist("active_colour_hex_label"):
                    dpg.set_value(
                        "active_colour_hex_label",
                        rgb_hex(context.interaction_state.active_colour),
                    )
            except Exception:
                pass
            try:
                if dpg.does_item_exist("colour_popup_hex_label"):
                    dpg.set_value(
                        "colour_popup_hex_label",
                        rgba_hex(context.interaction_state.active_colour),
                    )
            except Exception:
                pass
            try:
                if dpg.does_item_exist("active_colour_hex_input"):
                    dpg.set_value(
                        "active_colour_hex_input",
                        rgba_hex(context.interaction_state.active_colour),
                    )
            except Exception:
                pass
        except Exception:
            pass
        return json.dumps({"ok": True, "rgba": list(context.interaction_state.active_colour)})
    if name == "create_sketch_plane":
        mesh = _require_mesh(context)
        face_id = int(arguments["face_id"])
        if not 0 <= face_id < mesh.face_count:
            raise ValueError("face_id is out of range")
        document = context.sketch_tool.create_plane_from_face(mesh, face_id)
        mesh.sketch_documents = [document]
        return json.dumps({"ok": True, "anchor_face_id": face_id})
    if name == "add_rect_sketch":
        mesh = _require_mesh(context)
        if not mesh.sketch_documents:
            raise ValueError("No active sketch document")
        document = mesh.sketch_documents[-1]
        entity = context.sketch_tool.create_entity(
            "rect",
            np.asarray(arguments["min_uv"], dtype=np.float32),
            np.asarray(arguments["max_uv"], dtype=np.float32),
            context.interaction_state.active_colour,
        )
        context.commands.add_sketch_entity(document, entity)
        return json.dumps({"ok": True, "entity_id": entity.entity_id})
    if name == "bake_sketch":
        mesh = _require_mesh(context)
        if not mesh.sketch_documents:
            raise ValueError("No active sketch document")
        baked = context.sketch_tool.bake_document_to_faces(mesh, mesh.sketch_documents[-1])
        context.commands.paint_faces(baked, description="AI bake sketch")
        return json.dumps({"ok": True, "faces": len(baked)})
    if name == "paint_faces":
        mesh = _require_mesh(context)
        updates = {}
        for face_id in arguments["face_ids"]:
            if not 0 <= int(face_id) < mesh.face_count:
                raise ValueError(f"Invalid face_id {face_id}")
            updates[int(face_id)] = context.interaction_state.active_colour
        touched = context.commands.paint_faces(updates, description="AI paint faces")
        return json.dumps({"ok": True, "faces": touched})
    if name == "export_project_snapshot":
        path = context.capture_viewport()
        return json.dumps({"ok": True, "path": str(path)})
    if name == "export_3mf_file":
        mesh = _require_mesh(context)
        output = Path(arguments["path"])
        export_3mf(output, mesh)
        return json.dumps({"ok": True, "path": str(output)})
    if name == "undo":
        return json.dumps({"ok": True, "faces": context.commands.undo()})
    raise ValueError(f"Unknown AI tool: {name}")

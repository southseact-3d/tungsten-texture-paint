from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .color_utils import Color

InteractionMode = Literal["paint", "sketch"]
PaintMode = Literal["brush", "fill", "sample", "erase", "mask", "select"]
SketchMode = Literal["select", "line", "rect", "circle", "text"]
WorkspaceMode = Literal["preview", "paint"]
NavAxis = Literal["xp", "xn", "yp", "yn", "zp", "zn"]


@dataclass(slots=True)
class BrushSettings:
    radius: float = 0.15
    opacity: float = 0.8
    falloff: Literal["constant", "linear", "smooth"] = "smooth"
    spacing: float = 0.2
    front_faces_only: bool = False
    angle_tolerance_degrees: float = 65.0


@dataclass(slots=True)
class ToolCallLogEntry:
    name: str
    arguments: str
    ok: bool
    summary: str


@dataclass(slots=True)
class AISettings:
    provider_base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4.1-mini"
    timeout: float = 60.0
    system_prompt_version: str = "v1"
    image_path: str = ""


@dataclass(slots=True)
class InteractionState:
    workspace_mode: WorkspaceMode = "preview"
    interaction_mode: InteractionMode = "paint"
    paint_tool: PaintMode = "brush"
    sketch_tool: SketchMode = "select"
    active_colour: Color = (255, 80, 80, 255)
    paint_linked_faces: bool = False
    hovered_face: int | None = None
    drag_origin_screen: tuple[float, float] | None = None
    drag_origin_plane: np.ndarray | None = None
    dragging: bool = False
    drag_button: int | None = None
    nav_dragging: bool = False
    nav_drag_origin: tuple[float, float] | None = None
    nav_hover_axis: NavAxis | None = None
    nav_pressed_axis: NavAxis | None = None
    nav_drag_moved: bool = False
    shift_down: bool = False
    ctrl_down: bool = False
    tab_down: bool = False
    viewport_size: tuple[int, int] = (960, 720)
    viewport_dirty: bool = True
    selected_entity_id: str | None = None
    active_handle: str | None = None
    preview_entity_id: str | None = None
    selected_faces: set[int] = field(default_factory=set)
    marquee_start: tuple[float, float] | None = None
    marquee_end: tuple[float, float] | None = None
    svg_tint: Color = (255, 255, 255, 255)
    recent_svgs: list[str] = field(default_factory=list)
    recent_projects: list[str] = field(default_factory=list)
    freehand_points: list[tuple[float, float]] = field(default_factory=list)
    brush: BrushSettings = field(default_factory=BrushSettings)
    ai_settings: AISettings = field(default_factory=AISettings)
    ai_messages: list[tuple[str, str]] = field(default_factory=list)
    tool_logs: list[ToolCallLogEntry] = field(default_factory=list)

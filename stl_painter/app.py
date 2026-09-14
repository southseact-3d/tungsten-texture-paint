from __future__ import annotations

import ctypes
import logging
import os
import time
from math import ceil
from pathlib import Path
from tempfile import NamedTemporaryFile
from tkinter import Tk, filedialog

import dearpygui.dearpygui as dpg
import numpy as np
from PIL import Image

from .ai_agent import AIAgent, build_user_prompt
from .ai_tools import AIToolContext
from .camera import OrbitCamera
from .color_utils import (
    DEFAULT_COLOR,
    Color,
    clamp_color,
    parse_hex_color,
    rgb_hex,
    rgba_hex,
)
from .commands import AppCommands
from .config import load_local_config, save_local_config
from .exporter import SUPPORTED_EXPORT_EXTENSIONS, export_model
from .importer import load_model
from .interaction_state import InteractionState
from .logging_utils import log_file_path
from .mesh_model import MeshModel
from . import nav_cube
from .paint_tool import PaintTool
from .picking import PickResult, pick_face_location_cpu
from .project_io import (
    PROJECT_EXTENSION,
    load_project,
    load_tg3d,
    reconstruct_current_model,
    save_tg3d,
)
from .renderer import (
    MeshRenderer,
    RenderSnapshot,
    make_grid_snapshot,
    probe_gpu_support,
)
from .sketch_tool import SketchTool, entity_snap_points
from .ui_panels import compute_workspace_layout, sync_mode_sections

logger = logging.getLogger(__name__)
_USER32 = ctypes.windll.user32

# File-dialog request kinds served by the main loop (see _process_pending_dialog).
_DIALOG_OPEN = "open"
_DIALOG_VARIANT = "variant"
_DIALOG_EXPORT = "export"
_DIALOG_SAVE = "save"


class _POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class _RECT(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]


PALETTE: list[Color] = [
    (220, 220, 220, 255),
    (255, 80, 80, 255),
    (255, 179, 71, 255),
    (77, 184, 72, 255),
    (65, 105, 225, 255),
    (157, 78, 221, 255),
]

ICON = {
    "open": "Open",
    "export": "Export",
    "save": "Save",
    "bake": "Bake",
    "undo": "Undo",
    "redo": "Redo",
    "clear": "Clear",
    "invert": "Invert",
    "paint": "Paint",
    "mask": "Mask",
    "select": "Select",
    "image": "Image",
    "send": "Send",
    "svg": "SVG",
    "group": "Group",
}


def _normalise_picker_colour(app_data: list[float] | tuple[float, ...]) -> Color:
    if not app_data:
        return (255, 80, 80, 255)
    values = [float(item) for item in list(app_data)[:4]]
    while len(values) < 4:
        values.append(255.0 if len(values) == 3 else 0.0)
    if max(values) <= 1.0:
        values = [value * 255.0 for value in values]
    return clamp_color(tuple(int(round(value)) for value in values))


class TexturePainterApp:
    def __init__(self) -> None:
        self.state = InteractionState()
        self.mesh_model: MeshModel | None = None
        self.camera = OrbitCamera(np.array([0.0, 0.0, 0.0], dtype=np.float32), 5.0)
        self.renderer: MeshRenderer | None = None
        self.paint_tool: PaintTool | None = None
        self.sketch_tool = SketchTool()
        self.commands = AppCommands(None, None)
        self._key_tab = getattr(dpg, "mvKey_Tab", 512)
        self._key_lshift = getattr(dpg, "mvKey_LShift", 528)
        self._key_rshift = getattr(dpg, "mvKey_RShift", 532)
        self._key_lctrl = getattr(dpg, "mvKey_LControl", 527)
        self._key_rctrl = getattr(dpg, "mvKey_RControl", 531)
        self._key_z = getattr(dpg, "mvKey_Z", 571)
        self._viewport_texture_tag: int | str | None = None
        self._texture_data = np.zeros(
            (self.state.viewport_size[1], self.state.viewport_size[0], 4),
            dtype=np.float32,
        )
        self._last_brush_face: int | None = None
        self._vis_pick_key: object = None
        self._vis_pick_image = None
        self._click_pick_key: object = None
        self._click_pick_image: np.ndarray | None = None
        # Deferred paint request from Dear PyGui mouse callbacks:
        # (mouse_pos, is_drag). Resolved in the main loop where GPU
        # calls are safe (see _process_pending_paint).
        self._pending_paint_click: tuple[tuple[float, float], bool] | None = None
        self._timeline_context_index: int = 0
        self._pending_dialog: str | None = None
        self._last_hover_px: tuple[int, int] | None = None
        self._last_hover_time: float = 0.0
        self._load_ai_settings()
        self._load_svg_settings()
        self._load_recent_projects()
        self._create_ui()

    def _pick_path_native(
        self, *, save: bool, title: str, filetypes: list[tuple[str, str]]
    ) -> str | None:
        # NOTE: must only run from the main-loop dialog pump
        # (_process_pending_dialog), never from inside a Dear PyGui frame
        # callback: the Tk modal pumps window messages, and doing that
        # re-entrantly mid-frame corrupts the GL context (native crash in
        # opengl32.dll shortly after opening a file).
        root = Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        try:
            value = (
                filedialog.asksaveasfilename(title=title, filetypes=filetypes)
                if save
                else filedialog.askopenfilename(title=title, filetypes=filetypes)
            )
        finally:
            root.destroy()
        return str(value) if value else None

    def _request_dialog(self, kind: str) -> None:
        """Defer a native file dialog to the main-loop pump (see run())."""
        self._pending_dialog = kind
        self._mark_viewport_dirty()

    def _process_pending_dialog(self) -> None:
        kind, self._pending_dialog = self._pending_dialog, None
        if kind is None:
            return
        if kind == _DIALOG_OPEN:
            path = self._pick_path_native(
                save=False,
                title="Open model or project",
                filetypes=[
                    (
                        "3D Files",
                        "*.stl *.obj *.glb *.gltf *.3mf *.ply *.fbx *.blend "
                        "*.step *.stp *.tg3d *.json",
                    ),
                    ("All files", "*.*"),
                ],
            )
            if path:
                self._on_open_selected(path)
        elif kind == _DIALOG_VARIANT:
            if self.mesh_model is None:
                self._set_status("Load a base model before importing a variant")
                return
            path = self._pick_path_native(
                save=False,
                title="Import variant model",
                filetypes=[
                    (
                        "3D Files",
                        "*.stl *.obj *.glb *.gltf *.3mf *.ply *.fbx *.blend "
                        "*.step *.stp",
                    ),
                    ("All files", "*.*"),
                ],
            )
            if path:
                self._on_import_variant_selected(path)
        elif kind == _DIALOG_EXPORT:
            if self.mesh_model is None:
                self._set_status("No mesh loaded")
                return
            formats = " ".join(
                f"*{ext}" for ext in sorted(SUPPORTED_EXPORT_EXTENSIONS)
            )
            path = self._pick_path_native(
                save=True,
                title="Export model",
                filetypes=[("3D Files", formats), ("All files", "*.*")],
            )
            if path:
                self._on_export_selected(path)
        elif kind == _DIALOG_SAVE:
            if self.mesh_model is None:
                self._set_status("No mesh loaded")
                return
            path = self._pick_path_native(
                save=True,
                title="Save TG3D Project",
                filetypes=[
                    ("Tungsten Project", f"*{PROJECT_EXTENSION}"),
                    ("All files", "*.*"),
                ],
            )
            if path:
                if Path(path).suffix.lower() != PROJECT_EXTENSION:
                    path = f"{path}{PROJECT_EXTENSION}"
                self._on_save_project_selected(path)

    def _load_ai_settings(self) -> None:
        payload = load_local_config()
        ai = payload.get("ai_settings", {})
        if not ai:
            return
        self.state.ai_settings.provider_base_url = str(
            ai.get("provider_base_url", self.state.ai_settings.provider_base_url)
        )
        self.state.ai_settings.api_key = str(ai.get("api_key", ""))
        self.state.ai_settings.model = str(
            ai.get("model", self.state.ai_settings.model)
        )
        self.state.ai_settings.timeout = float(
            ai.get("timeout", self.state.ai_settings.timeout)
        )
        self.state.ai_settings.system_prompt_version = str(
            ai.get(
                "system_prompt_version", self.state.ai_settings.system_prompt_version
            )
        )
        self.state.ai_settings.image_path = str(ai.get("image_path", ""))

    def _save_ai_settings(self) -> None:
        save_local_config(
            {
                "ai_settings": {
                    "provider_base_url": self.state.ai_settings.provider_base_url,
                    "api_key": self.state.ai_settings.api_key,
                    "model": self.state.ai_settings.model,
                    "timeout": self.state.ai_settings.timeout,
                    "system_prompt_version": self.state.ai_settings.system_prompt_version,
                    "image_path": self.state.ai_settings.image_path,
                }
            }
        )

    def _load_svg_settings(self) -> None:
        payload = load_local_config()
        self.state.recent_svgs = [
            str(path)
            for path in payload.get("recent_svgs", [])
            if Path(str(path)).exists()
        ]
        tint = payload.get("last_svg_tint", list(self.state.svg_tint))
        self.state.svg_tint = clamp_color(tuple(tint))

    def _load_recent_projects(self) -> None:
        payload = load_local_config()
        self.state.recent_projects = [
            str(path)
            for path in payload.get("recent_projects", [])
            if Path(str(path)).exists()
            and Path(str(path)).suffix.lower() == PROJECT_EXTENSION
        ]

    def _save_recent_projects(self) -> None:
        save_local_config({"recent_projects": self.state.recent_projects[:20]})

    def _save_svg_settings(self) -> None:
        save_local_config(
            {
                "recent_svgs": self.state.recent_svgs[:10],
                "last_svg_tint": list(self.state.svg_tint),
            }
        )

    def _create_ui(self) -> None:
        dpg.create_context()
        self._apply_light_theme()
        self._load_default_font()
        with dpg.texture_registry(show=False, tag="texture_registry"):
            self._viewport_texture_tag = dpg.add_raw_texture(
                width=self.state.viewport_size[0],
                height=self.state.viewport_size[1],
                default_value=self._texture_data,
                parent="texture_registry",
                format=dpg.mvFormat_Float_rgba,
            )
        self._create_file_dialogs()
        self._build_colour_popup()
        with dpg.window(
            label="Tungsten Texture Paint",
            tag="home_window",
            width=720,
            height=520,
            show=False,
        ):
            dpg.add_text("Welcome to Tungsten Texture Paint", color=(17, 24, 39))
            dpg.add_text(
                "Open a model/project or continue from a recent .tg3d session.",
                wrap=680,
            )
            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Open Model / Project",
                    callback=self._on_open_click,
                    width=220,
                )
                dpg.add_button(
                    label="Start Empty Viewer",
                    callback=lambda: self._show_main_window(),
                    width=180,
                )
            dpg.add_spacer(height=8)
            dpg.add_text("Recent Projects", color=(17, 24, 39))
            dpg.add_listbox(
                items=self.state.recent_projects or ["No recent projects yet"],
                tag="recent_projects_list",
                num_items=10,
                width=-1,
            )
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Open Selected Recent Project",
                    callback=self._open_selected_recent_project,
                    width=260,
                )
                dpg.add_button(
                    label="Refresh",
                    callback=lambda: self._refresh_recent_project_list(),
                    width=110,
                )
        with dpg.window(label="STL Texture Painter", tag="main_window", show=False):
            with dpg.group(horizontal=True):
                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label="Open",
                        callback=self._on_open_click,
                        width=70,
                    )
                    dpg.add_button(
                        label="Export",
                        callback=self._on_export_click,
                        width=70,
                    )
                    dpg.add_button(
                        label="Save",
                        callback=self._on_save_project_click,
                        width=70,
                    )
                    dpg.add_button(
                        label="Bake",
                        callback=self._on_bake_sketch,
                        width=70,
                    )
                dpg.add_spacer(width=12)
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Undo", callback=self._on_undo, width=70)
                    dpg.add_button(label="Redo", callback=self._on_redo, width=70)
                dpg.add_spacer(width=12)
                dpg.add_text("Preview Mode", tag="workspace_mode_label")
                dpg.add_text("", tag="status_text")
            dpg.add_separator()
            dpg.add_text(
                "Preview mode: orbit the model here. Press Tab to switch into texture paint mode.",
                tag="workspace_hint_text",
                wrap=0,
            )
            with dpg.group(horizontal=True):
                self._build_left_sidebar()
                with dpg.child_window(
                    width=self.state.viewport_size[0],
                    height=self.state.viewport_size[1],
                    border=False,
                    tag="viewport_panel",
                ):
                    dpg.add_image(self._viewport_texture_tag, tag="viewport_image")
                    with dpg.window(
                        tag="viewport_nav",
                        label="",
                        no_title_bar=True,
                        no_move=True,
                        no_resize=True,
                        no_scrollbar=True,
                        no_collapse=True,
                        autosize=True,
                    ):
                        dpg.add_text("View")
                        dpg.add_drawlist(width=132, height=132, tag="viewport_nav_pad")
                        dpg.add_button(
                            label="Home", width=72, callback=self._reset_view
                        )
                    with dpg.handler_registry():
                        dpg.add_mouse_down_handler(callback=self._on_mouse_down)
                        dpg.add_mouse_release_handler(callback=self._on_mouse_release)
                        dpg.add_mouse_move_handler(callback=self._on_mouse_move)
                        dpg.add_mouse_wheel_handler(callback=self._on_mouse_wheel)
                        dpg.add_key_down_handler(callback=self._on_key_down)
                        dpg.add_key_release_handler(callback=self._on_key_release)
                self._build_right_sidebar()
            dpg.add_separator()
            self._build_timeline_panel()
        dpg.create_viewport(title="STL Texture Painter", width=1680, height=920)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        self._show_main_window()
        self._set_status("Ready")
        self._sync_tool_panels()
        self._sync_workspace_layout(force=True)

    def _apply_light_theme(self) -> None:
        with dpg.theme(tag="light_app_theme"):
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 10, 10)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 6, 5)
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 6, 6)
                dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 6)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 4)
                dpg.add_theme_style(dpg.mvStyleVar_GrabRounding, 4)
                dpg.add_theme_style(dpg.mvStyleVar_TabRounding, 4)
                dpg.add_theme_style(dpg.mvStyleVar_ScrollbarRounding, 4)
                dpg.add_theme_style(dpg.mvStyleVar_PopupRounding, 6)
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (245, 247, 250, 255))
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (255, 255, 255, 255))
                dpg.add_theme_color(dpg.mvThemeCol_PopupBg, (255, 255, 255, 255))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (238, 242, 248, 255))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (228, 235, 248, 255))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (215, 228, 250, 255))
                dpg.add_theme_color(dpg.mvThemeCol_Button, (59, 130, 246, 255))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (37, 99, 235, 255))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (29, 78, 216, 255))
                dpg.add_theme_color(dpg.mvThemeCol_Header, (219, 234, 254, 255))
                dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, (191, 219, 254, 255))
                dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, (147, 197, 253, 255))
                dpg.add_theme_color(dpg.mvThemeCol_Text, (31, 41, 55, 255))
                dpg.add_theme_color(dpg.mvThemeCol_Separator, (203, 213, 225, 255))
                dpg.add_theme_color(dpg.mvThemeCol_Tab, (243, 244, 246, 255))
                dpg.add_theme_color(dpg.mvThemeCol_TabHovered, (229, 231, 235, 255))
                dpg.add_theme_color(dpg.mvThemeCol_TabActive, (229, 231, 235, 255))
                dpg.add_theme_color(dpg.mvThemeCol_TitleBg, (243, 244, 246, 255))
                dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, (243, 244, 246, 255))
                dpg.add_theme_color(dpg.mvThemeCol_MenuBarBg, (249, 250, 251, 255))
                dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg, (243, 244, 246, 255))
                dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab, (203, 213, 225, 255))
                dpg.add_theme_color(
                    dpg.mvThemeCol_ScrollbarGrabHovered, (148, 163, 184, 255)
                )
                dpg.add_theme_color(
                    dpg.mvThemeCol_ScrollbarGrabActive, (107, 114, 128, 255)
                )
                dpg.add_theme_color(dpg.mvThemeCol_CheckMark, (59, 130, 246, 255))
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, (59, 130, 246, 255))
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, (37, 99, 235, 255))
                dpg.add_theme_color(dpg.mvThemeCol_TableHeaderBg, (243, 244, 246, 255))
                dpg.add_theme_color(
                    dpg.mvThemeCol_TableBorderStrong, (203, 213, 225, 255)
                )
                dpg.add_theme_color(
                    dpg.mvThemeCol_TableBorderLight, (229, 231, 235, 255)
                )
            with dpg.theme_component(dpg.mvButton, tag="toolbar_button_theme"):
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 10, 6)
                dpg.add_theme_style(dpg.mvStyleVar_ItemInnerSpacing, 8, 4)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 4)
                dpg.add_theme_color(dpg.mvThemeCol_Button, (243, 244, 246, 255))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (229, 231, 235, 255))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (209, 213, 219, 255))
                dpg.add_theme_color(dpg.mvThemeCol_Text, (31, 41, 55, 255))
            with dpg.theme_component(dpg.mvButton, tag="primary_button_theme"):
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 10, 6)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 4)
                dpg.add_theme_color(dpg.mvThemeCol_Button, (59, 130, 246, 255))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (37, 99, 235, 255))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (29, 78, 216, 255))
                dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 255, 255, 255))
            with dpg.theme_component(dpg.mvText, tag="section_header_theme"):
                dpg.add_theme_color(dpg.mvThemeCol_Text, (17, 24, 39, 255))
            with dpg.theme_component(dpg.mvText, tag="hint_text_theme"):
                dpg.add_theme_color(dpg.mvThemeCol_Text, (107, 114, 128, 255))
        dpg.bind_theme("light_app_theme")
        with dpg.theme(tag="colour_swatch_theme"):
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 4, 4)
                dpg.add_theme_color(dpg.mvThemeCol_Button, self.state.active_colour)
                dpg.add_theme_color(
                    dpg.mvThemeCol_ButtonHovered, self.state.active_colour
                )
                dpg.add_theme_color(
                    dpg.mvThemeCol_ButtonActive, self.state.active_colour
                )
                dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 255, 255, 255))

    def _load_default_font(self) -> None:
        font_candidates = [
            "C:\\Windows\\Fonts\\segoeui.ttf",
            "C:\\Windows\\Fonts\\arial.ttf",
            "C:\\Windows\\Fonts\\calibri.ttf",
            "/System/Library/Fonts/SF Pro Text.ttc",
            "/System/Library/Fonts/Helvetica.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
        ]
        with dpg.font_registry():
            for font_path in font_candidates:
                if Path(font_path).exists():
                    font_id = dpg.add_font(font_path, 18)
                    dpg.bind_font(font_id)
                    return
            font_id = dpg.add_font(dpg.mvFont_None, 18)
            dpg.bind_font(font_id)

    def _build_left_sidebar(self) -> None:
        with dpg.child_window(width=300, height=-1, tag="tools_panel"):
            dpg.add_text("Workspace", color=(17, 24, 39))
            dpg.add_combo(
                items=["paint", "sketch"],
                default_value=self.state.interaction_mode,
                label="Mode",
                callback=self._set_interaction_mode,
                tag="interaction_mode_combo",
            )
            dpg.add_separator()
            dpg.add_text("Colour", color=(17, 24, 39))
            dpg.add_text("No mesh loaded", tag="mesh_info_text", wrap=260)
            dpg.add_text(
                rgb_hex(self.state.active_colour),
                tag="active_colour_hex_label",
                color=(72, 86, 110),
            )
            dpg.add_button(
                label="",
                width=-1,
                height=40,
                tag="active_colour_preview",
                callback=lambda *_: self._open_colour_popup(),
            )
            dpg.bind_item_theme("active_colour_preview", "colour_swatch_theme")
            dpg.add_text(
                "Click the swatch to change colour.",
                color=(107, 114, 128),
                wrap=260,
            )
            dpg.add_separator()
            dpg.add_text("Model Transform", color=(17, 24, 39))
            dpg.add_input_float(
                label="Scale Multiple",
                default_value=1.0,
                min_value=0.001,
                min_clamped=True,
                tag="model_scale_factor",
            )
            dpg.add_button(
                label="Apply Scale", callback=self._on_apply_scale_click, width=-1
            )
            dpg.add_button(
                label="Import Variant & Transfer Paint",
                callback=self._on_import_variant_click,
                width=-1,
            )
            with dpg.group(tag="paint_section"):
                dpg.add_separator()
                dpg.add_text("Paint Tools", color=(17, 24, 39))
                dpg.add_combo(
                    items=["brush", "fill", "sample", "erase", "mask", "select"],
                    default_value=self.state.paint_tool,
                    label="Tool",
                    callback=self._set_paint_tool,
                    tag="paint_tool_combo",
                )
                dpg.add_checkbox(
                    label="Paint Linked Faces as Group",
                    default_value=self.state.paint_linked_faces,
                    callback=lambda _s, value: setattr(
                        self.state, "paint_linked_faces", bool(value)
                    ),
                )
                dpg.add_slider_float(
                    label="Radius",
                    min_value=0.01,
                    max_value=0.6,
                    default_value=self.state.brush.radius,
                    callback=lambda _s, value: setattr(
                        self.state.brush, "radius", float(value)
                    ),
                )
                dpg.add_slider_float(
                    label="Opacity",
                    min_value=0.05,
                    max_value=1.0,
                    default_value=self.state.brush.opacity,
                    callback=lambda _s, value: setattr(
                        self.state.brush, "opacity", float(value)
                    ),
                )
                dpg.add_combo(
                    items=["constant", "linear", "smooth"],
                    default_value=self.state.brush.falloff,
                    label="Falloff",
                    callback=lambda _s, value: setattr(
                        self.state.brush, "falloff", value
                    ),
                )
                dpg.add_checkbox(
                    label="Front Faces Only",
                    default_value=self.state.brush.front_faces_only,
                    callback=lambda _s, value: setattr(
                        self.state.brush, "front_faces_only", bool(value)
                    ),
                )
                dpg.add_slider_float(
                    label="Angle Tolerance",
                    min_value=5.0,
                    max_value=180.0,
                    default_value=self.state.brush.angle_tolerance_degrees,
                    callback=lambda _s, value: setattr(
                        self.state.brush, "angle_tolerance_degrees", float(value)
                    ),
                )
                dpg.add_slider_float(
                    label="Fill Tolerance",
                    min_value=0.0,
                    max_value=100.0,
                    default_value=self.state.fill_tolerance,
                    callback=lambda _s, value: setattr(
                        self.state, "fill_tolerance", float(value)
                    ),
                )
                dpg.add_button(
                    label="Group Faces",
                    callback=self._on_group_faces,
                    width=-1,
                )
                dpg.add_text("Groups: 0", tag="face_groups_count_text")
                dpg.add_button(
                    label="Clear Mask",
                    callback=self._on_clear_mask,
                    width=-1,
                )
                dpg.add_button(
                    label="Invert Mask",
                    callback=self._on_invert_mask,
                    width=-1,
                )
                dpg.add_button(
                    label="Paint Selected",
                    callback=self._paint_selected_faces,
                    width=-1,
                )
                dpg.add_button(
                    label="Mask Selected",
                    callback=self._mask_selected_faces,
                    width=-1,
                )
                dpg.add_button(
                    label="Clear Selection",
                    callback=self._clear_face_selection,
                    width=-1,
                )
                dpg.add_text("Masked faces: 0", tag="mask_count_text")
                dpg.add_text("Selected faces: 0", tag="selected_count_text")
            with dpg.group(tag="sketch_section", show=False):
                dpg.add_separator()
                dpg.add_text("Sketch Tools", color=(17, 24, 39))
                dpg.add_combo(
                    items=["select", "line", "rect", "circle", "text", "svg"],
                    default_value=self.state.sketch_tool,
                    label="Tool",
                    callback=self._set_sketch_tool,
                    tag="sketch_tool_combo",
                )
                dpg.add_button(
                    label="Import SVG",
                    callback=lambda: dpg.show_item("svg_dialog"),
                    width=-1,
                )
                dpg.add_combo(
                    items=self.state.recent_svgs or [""],
                    label="Recent SVGs",
                    tag="recent_svg_combo",
                    callback=self._on_recent_svg_selected,
                )
                dpg.add_input_text(
                    label="Text", default_value="Text", tag="sketch_text_value"
                )
                dpg.add_slider_int(
                    label="Text Size",
                    min_value=8,
                    max_value=120,
                    default_value=28,
                    tag="sketch_text_size",
                )
                dpg.add_slider_float(
                    label="Stroke Width",
                    min_value=0.001,
                    max_value=0.2,
                    default_value=0.02,
                    tag="sketch_stroke_width",
                )
                dpg.add_slider_float(
                    label="Grid",
                    min_value=0.01,
                    max_value=1.0,
                    default_value=0.1,
                    callback=self._set_sketch_grid,
                )
                dpg.add_checkbox(
                    label="Snap To Grid",
                    default_value=True,
                    callback=self._set_snap_flag,
                    user_data="grid",
                )
                dpg.add_checkbox(
                    label="Snap To Vertices",
                    default_value=True,
                    callback=self._set_snap_flag,
                    user_data="vertices",
                )
                dpg.add_checkbox(
                    label="Snap To Edges",
                    default_value=True,
                    callback=self._set_snap_flag,
                    user_data="edges",
                )
                dpg.add_checkbox(
                    label="Snap To Entities",
                    default_value=True,
                    callback=self._set_snap_flag,
                    user_data="entities",
                )
                dpg.add_text("Click a face first to start a sketch plane.", wrap=260)

    def _build_colour_popup(self) -> None:
        with dpg.window(
            label="Colour Picker",
            tag="colour_picker_popup",
            show=False,
            width=440,
            height=560,
            no_collapse=True,
            popup=True,
        ):
            dpg.add_text("Active Colour", color=(17, 24, 39))
            dpg.add_text(
                rgba_hex(self.state.active_colour),
                tag="colour_popup_hex_label",
                color=(72, 86, 110),
            )
            dpg.add_text("Palette", color=(17, 24, 39))
            with dpg.group(horizontal=True, tag="colour_popup_palette"):
                for index, colour in enumerate(PALETTE):
                    palette_theme_tag = f"colour_popup_palette_theme_{index}"
                    with dpg.theme(tag=palette_theme_tag):
                        with dpg.theme_component(dpg.mvButton):
                            dpg.add_theme_color(dpg.mvThemeCol_Button, colour)
                            dpg.add_theme_color(
                                dpg.mvThemeCol_ButtonHovered, colour
                            )
                            dpg.add_theme_color(
                                dpg.mvThemeCol_ButtonActive, colour
                            )
                            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 2, 2)
                    dpg.add_button(
                        label="",
                        width=44,
                        height=30,
                        tag=f"colour_popup_palette_{index}",
                        callback=lambda _s, _a, _u=None, user_data=index: (
                            self._select_palette_colour(user_data)
                        ),
                    )
                    dpg.bind_item_theme(
                        f"colour_popup_palette_{index}", palette_theme_tag
                    )
            dpg.add_color_picker(
                label="Active Color",
                default_value=list(self.state.active_colour),
                callback=self._on_custom_colour,
                tag="active_colour_picker",
                alpha_bar=True,
                display_hex=True,
            )
            dpg.add_input_text(
                label="Hex",
                hint="#RRGGBB or #RRGGBBAA",
                default_value=rgba_hex(self.state.active_colour),
                tag="active_colour_hex_input",
                on_enter=True,
                callback=self._on_hex_entered,
            )
            dpg.add_text(
                "Tip: press Enter in the hex field to apply. Changes apply live.",
                color=(107, 114, 128),
                wrap=400,
            )
            dpg.add_button(
                label="Done",
                callback=lambda *_: self._close_colour_popup(),
                width=-1,
            )

    def _open_colour_popup(self) -> None:
        if not dpg.does_item_exist("colour_picker_popup"):
            return
        self._sync_active_colour_ui()
        try:
            viewport_width = int(dpg.get_viewport_client_width())
            viewport_height = int(dpg.get_viewport_client_height())
            popup_width = 440
            popup_height = 560
            dpg.set_item_pos(
                "colour_picker_popup",
                [max(0, (viewport_width - popup_width) // 2), max(0, (viewport_height - popup_height) // 2)],
            )
        except Exception:
            pass
        dpg.show_item("colour_picker_popup")
        try:
            dpg.focus_item("colour_picker_popup")
        except Exception:
            pass

    def _close_colour_popup(self) -> None:
        if dpg.does_item_exist("colour_picker_popup"):
            dpg.hide_item("colour_picker_popup")

    def _sync_active_colour_ui(self) -> None:
        active = self.state.active_colour
        if dpg.does_item_exist("active_colour_preview"):
            if dpg.does_item_exist("colour_swatch_theme"):
                dpg.set_value(
                    "colour_swatch_theme",
                    {
                        dpg.mvThemeCol_Button: active,
                        dpg.mvThemeCol_ButtonHovered: active,
                        dpg.mvThemeCol_ButtonActive: active,
                    },
                )
        if dpg.does_item_exist("active_colour_picker"):
            dpg.set_value("active_colour_picker", list(active))
        if dpg.does_item_exist("active_colour_hex_label"):
            dpg.set_value("active_colour_hex_label", rgb_hex(active))
        if dpg.does_item_exist("colour_popup_hex_label"):
            dpg.set_value("colour_popup_hex_label", rgba_hex(active))
        if dpg.does_item_exist("active_colour_hex_input"):
            dpg.set_value("active_colour_hex_input", rgba_hex(active))

    def _on_hex_entered(self, _sender: int, app_data: str) -> None:
        parsed = parse_hex_color(str(app_data or ""))
        if parsed is None:
            self._set_status("Invalid hex colour — use #RRGGBB or #RRGGBBAA")
            self._sync_active_colour_ui()
            return
        self.state.active_colour = parsed
        self._sync_active_colour_ui()

    def _build_right_sidebar(self) -> None:
        with dpg.child_window(width=360, height=-1, tag="ai_panel"):
            dpg.add_text("AI Assistant", color=(17, 24, 39))
            dpg.add_text(
                "Prompt is optional. If you provide only a reference image, the assistant will try to recreate that texture.",
                wrap=320,
                color=(72, 86, 110),
            )
            dpg.add_input_text(
                label="Base URL",
                default_value=self.state.ai_settings.provider_base_url,
                callback=lambda _s, value: setattr(
                    self.state.ai_settings, "provider_base_url", value
                ),
            )
            dpg.add_input_text(
                label="API Key",
                password=True,
                default_value=self.state.ai_settings.api_key,
                callback=lambda _s, value: setattr(
                    self.state.ai_settings, "api_key", value
                ),
            )
            dpg.add_input_text(
                label="Model",
                default_value=self.state.ai_settings.model,
                callback=lambda _s, value: setattr(
                    self.state.ai_settings, "model", value
                ),
            )
            dpg.add_combo(
                label="Provider Preset",
                items=["opencode.ai (default)", "chutes.ai (alt)"],
                default_value="opencode.ai (default)",
                callback=lambda _s, value: self._apply_provider_preset(value),
            )
            dpg.add_input_text(
                label="Reference Image",
                default_value=self.state.ai_settings.image_path,
                tag="ai_image_path",
            )
            dpg.add_button(
                label="Choose Image",
                callback=lambda: dpg.show_item("image_dialog"),
                width=-1,
            )
            dpg.add_separator()
            dpg.add_text("Optional Prompt", color=(17, 24, 39))
            dpg.add_input_text(
                multiline=True,
                height=150,
                hint="Optional: describe the texture or sketch you want. Leave blank to recreate from the reference image only.",
                tag="ai_prompt_input",
            )
            dpg.add_button(
                label="Send To Assistant",
                callback=self._on_ai_send,
                width=-1,
            )
            dpg.add_input_text(
                multiline=True, readonly=True, height=220, tag="ai_chat_transcript"
            )
            dpg.add_input_text(
                multiline=True, readonly=True, height=180, tag="ai_tool_log"
            )

    def _create_file_dialogs(self) -> None:
        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            callback=self._on_image_selected,
            tag="image_dialog",
            width=700,
            height=400,
        ):
            dpg.add_file_extension(".png")
            dpg.add_file_extension(".jpg")
            dpg.add_file_extension(".jpeg")
        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            callback=self._on_svg_selected,
            tag="svg_dialog",
            width=700,
            height=400,
        ):
            dpg.add_file_extension(".svg")

    def _set_status(self, text: str) -> None:
        dpg.set_value("status_text", text)
        logger.info("STATUS | %s", text)

    def _mark_viewport_dirty(self) -> None:
        self.state.viewport_dirty = True

    def _set_interaction_mode(self, _sender: int, app_data: str) -> None:
        self.state.interaction_mode = app_data
        if self.mesh_model is not None:
            self.mesh_model.interaction_mode = app_data
        self._sync_tool_panels()
        self._mark_viewport_dirty()

    def _set_paint_tool(self, _sender: int, app_data: str) -> None:
        self.state.paint_tool = app_data
        self._set_status(f"Paint tool: {app_data}")

    def _set_sketch_tool(self, _sender: int, app_data: str) -> None:
        self.state.sketch_tool = app_data

    def _set_sketch_grid(self, _sender: int, value: float) -> None:
        if self._active_document() is not None:
            self._active_document().grid_size = float(value)

    def _set_snap_flag(self, _sender: int, value: bool, user_data: str) -> None:
        document = self._active_document()
        if document is None:
            return
        if user_data == "grid":
            document.snap_to_grid = bool(value)
        elif user_data == "vertices":
            document.snap_to_vertices = bool(value)
        elif user_data == "edges":
            document.snap_to_edges = bool(value)
        elif user_data == "entities":
            document.snap_to_entities = bool(value)

    def _sync_tool_panels(self) -> None:
        sync_mode_sections(self.state.interaction_mode)
        self._sync_workspace_ui()

    def _sync_workspace_ui(self) -> None:
        in_paint_workspace = self.state.workspace_mode == "paint"
        if dpg.does_item_exist("tools_panel"):
            dpg.configure_item("tools_panel", show=in_paint_workspace)
        if dpg.does_item_exist("ai_panel"):
            dpg.configure_item("ai_panel", show=True)
        if dpg.does_item_exist("workspace_mode_label"):
            dpg.set_value(
                "workspace_mode_label",
                "Texture Paint Mode" if in_paint_workspace else "Preview Mode",
            )
        if dpg.does_item_exist("workspace_hint_text"):
            hint = (
                "Texture paint mode: paint, fill, and sketch on the model. Press Tab to return to preview mode."
                if in_paint_workspace
                else "Preview mode: orbit the model here. Press Tab to switch into texture paint mode."
            )
            dpg.set_value("workspace_hint_text", hint)
        self._sync_workspace_layout(force=True)

    def _set_workspace_mode(self, mode: str) -> None:
        if self.state.workspace_mode == mode:
            return
        self.state.workspace_mode = mode  # type: ignore[assignment]
        self.state.hovered_face = None
        self._sync_workspace_ui()
        self._mark_viewport_dirty()
        logger.info("WORKSPACE_MODE | mode=%s", mode)
        self._set_status(
            "Texture paint mode enabled" if mode == "paint" else "Preview mode enabled"
        )

    def _toggle_workspace_mode(self) -> None:
        next_mode = "paint" if self.state.workspace_mode == "preview" else "preview"
        self._set_workspace_mode(next_mode)

    def _poll_keyboard_shortcuts(self) -> None:
        tab_down = bool(dpg.is_key_down(self._key_tab))
        if tab_down and not self.state.tab_down:
            self._toggle_workspace_mode()
        self.state.tab_down = tab_down

    def _reset_view(self) -> None:
        if self.mesh_model is not None:
            self.camera = OrbitCamera.for_mesh(self.mesh_model.vertices)
        else:
            self.camera = OrbitCamera(np.array([0.0, 0.0, 0.0], dtype=np.float32), 5.0)
        self._mark_viewport_dirty()

    def _position_nav_widget(self) -> None:
        if not dpg.does_item_exist("viewport_nav") or not dpg.does_item_exist(
            "viewport_image"
        ):
            return
        image_pos = dpg.get_item_rect_min("viewport_image")
        image_size = dpg.get_item_rect_size("viewport_image")
        nav_size = dpg.get_item_rect_size("viewport_nav")
        if image_size[0] <= 0 or image_size[1] <= 0:
            return
        x = int(image_pos[0] + image_size[0] - nav_size[0] - 12)
        y = int(image_pos[1] + 12)
        dpg.set_item_pos("viewport_nav", [x, y])

    def _nav_gizmo_geometry(self) -> dict[str, object]:
        width, height = 132, 132
        return {
            "size": (width, height),
            "center": (width / 2.0, height / 2.0),
            "radius": 60.0,
            "orbit_radius": 52.0,
            "half_size": 36.0,
        }

    def _nav_cube_faces(self) -> list[dict[str, object]]:
        gizmo = self._nav_gizmo_geometry()
        center = gizmo["center"]
        assert isinstance(center, tuple)
        return nav_cube.project_cube(
            self.camera.view_matrix()[:3, :3],
            (float(center[0]), float(center[1])),
            float(gizmo["half_size"]),
        )

    def _draw_nav_pad(self) -> None:
        if not dpg.does_item_exist("viewport_nav_pad"):
            return
        dpg.delete_item("viewport_nav_pad", children_only=True)
        gizmo = self._nav_gizmo_geometry()
        center = gizmo["center"]
        radius = gizmo["radius"]
        dpg.draw_circle(
            center,
            radius,
            color=(118, 126, 140, 220),
            fill=(243, 245, 248, 210),
            thickness=2,
            parent="viewport_nav_pad",
        )
        faces = self._nav_cube_faces()
        centers = nav_cube.face_centers_2d(faces)
        for face in faces:
            axis = str(face["axis"])
            polygon = [
                (float(point[0]), float(point[1]))
                for point in np.asarray(face["polygon"], dtype=np.float32)
            ]
            hovered = self.state.nav_hover_axis == axis
            fill = tuple(face["fill"])  # type: ignore[union-attr]
            if hovered:
                fill = tuple(min(255, component + 45) for component in fill[:3]) + (255,)
            outline = (24, 24, 28, 255) if hovered else (90, 96, 106, 255)
            dpg.draw_polygon(
                polygon,
                color=outline,
                fill=fill,
                thickness=2 if hovered else 1,
                parent="viewport_nav_pad",
            )
            label_pos = centers[axis]
            dpg.draw_text(
                (label_pos[0] - 4, label_pos[1] - 7),
                str(face["label"]),
                color=(255, 255, 255, 255),
                size=12,
                parent="viewport_nav_pad",
            )
        dpg.draw_text(
            (28, 112),
            "Drag to rotate",
            color=(82, 88, 96, 255),
            size=13,
            parent="viewport_nav_pad",
        )

    def _nav_pad_hovered(self) -> bool:
        return dpg.does_item_exist("viewport_nav_pad") and bool(
            dpg.is_item_hovered("viewport_nav_pad")
        )

    def _nav_pad_mouse_position(self) -> tuple[float, float] | None:
        if not dpg.does_item_exist("viewport_nav_pad"):
            return None
        mouse_x, mouse_y = dpg.get_mouse_pos(local=False)
        pad_x, pad_y = dpg.get_item_rect_min("viewport_nav_pad")
        return mouse_x - pad_x, mouse_y - pad_y

    def _global_mouse_position(self) -> tuple[float, float]:
        mouse_x, mouse_y = dpg.get_mouse_pos(local=False)
        return float(mouse_x), float(mouse_y)

    def _client_to_screen(self, x: float, y: float) -> tuple[float, float]:
        hwnd = _USER32.GetActiveWindow()
        if not hwnd:
            return x, y
        point = _POINT(int(round(x)), int(round(y)))
        _USER32.ClientToScreen(hwnd, ctypes.byref(point))
        return float(point.x), float(point.y)

    def _nav_pad_screen_rect(self) -> tuple[int, int, int, int] | None:
        if not dpg.does_item_exist("viewport_nav_pad"):
            return None
        local_x, local_y = dpg.get_item_rect_min("viewport_nav_pad")
        width, height = dpg.get_item_rect_size("viewport_nav_pad")
        screen_x, screen_y = self._client_to_screen(local_x, local_y)
        inset = 12
        return (
            int(round(screen_x + inset)),
            int(round(screen_y + inset)),
            int(round(screen_x + width - inset)),
            int(round(screen_y + height - inset)),
        )

    def _nav_pad_screen_center(self) -> tuple[float, float] | None:
        rect = self._nav_pad_screen_rect()
        if rect is None:
            return None
        left, top, right, bottom = rect
        return (left + right) / 2.0, (top + bottom) / 2.0

    def _clip_nav_cursor(self) -> None:
        rect = self._nav_pad_screen_rect()
        if rect is None:
            return
        left, top, right, bottom = rect
        clip = _RECT(left=left, top=top, right=right, bottom=bottom)
        _USER32.ClipCursor(ctypes.byref(clip))

    def _center_nav_cursor(self) -> tuple[float, float] | None:
        center = self._nav_pad_screen_center()
        if center is None:
            return None
        x, y = center
        _USER32.SetCursorPos(int(round(x)), int(round(y)))
        return center

    def _release_cursor_clip(self) -> None:
        _USER32.ClipCursor(None)

    def _nav_hit_test(self, mouse_pos: tuple[float, float]) -> tuple[str, str | None]:
        gizmo = self._nav_gizmo_geometry()
        center = gizmo["center"]
        assert isinstance(center, tuple)
        return nav_cube.hit_test(
            (float(mouse_pos[0]), float(mouse_pos[1])),
            self._nav_cube_faces(),
            float(gizmo["orbit_radius"]),
            (float(center[0]), float(center[1])),
        )

    def _snap_camera_to_axis(self, axis: str) -> None:
        azimuth, elevation = nav_cube.snap_angles(axis)
        self.camera.set_angles(azimuth, elevation)
        self._mark_viewport_dirty()

    def _draw_world_grid(self, rgba: np.ndarray) -> None:
        width, _ = self.state.viewport_size
        if self.mesh_model is not None:
            mins = self.mesh_model.vertices.min(axis=0)
            maxs = self.mesh_model.vertices.max(axis=0)
            extent = float(np.linalg.norm(maxs - mins))
            size = max(2.0, extent * 1.2)
        else:
            size = 4.0
        major_lines = 8
        step = max(size / major_lines, 0.25)
        line_count = int(ceil(size / step))
        for i in range(-line_count, line_count + 1):
            v = i * step
            p0 = self._project_world_to_screen(
                np.array([-size, 0.0, v], dtype=np.float32)
            )
            p1 = self._project_world_to_screen(
                np.array([size, 0.0, v], dtype=np.float32)
            )
            p2 = self._project_world_to_screen(
                np.array([v, 0.0, -size], dtype=np.float32)
            )
            p3 = self._project_world_to_screen(
                np.array([v, 0.0, size], dtype=np.float32)
            )
            if p0 and p1:
                strength = 0.32 if i % 5 == 0 else 0.16
                self._draw_line(rgba, p0, p1, (0.30, 0.34, 0.38), alpha=strength)
            if p2 and p3:
                strength = 0.32 if i % 5 == 0 else 0.16
                self._draw_line(rgba, p2, p3, (0.30, 0.34, 0.38), alpha=strength)
        ax0 = self._project_world_to_screen(
            np.array([-size, 0.0, 0.0], dtype=np.float32)
        )
        ax1 = self._project_world_to_screen(
            np.array([size, 0.0, 0.0], dtype=np.float32)
        )
        az0 = self._project_world_to_screen(
            np.array([0.0, 0.0, -size], dtype=np.float32)
        )
        az1 = self._project_world_to_screen(
            np.array([0.0, 0.0, size], dtype=np.float32)
        )
        if ax0 and ax1:
            self._draw_line(rgba, ax0, ax1, (0.82, 0.22, 0.22), alpha=0.65)
        if az0 and az1:
            self._draw_line(rgba, az0, az1, (0.20, 0.35, 0.82), alpha=0.65)
        center = self._project_world_to_screen(
            np.array([0.0, 0.0, 0.0], dtype=np.float32)
        )
        if center:
            cx, cy = center
            if 2 <= cx < width - 2 and 2 <= cy < rgba.shape[0] - 2:
                rgba[cy - 2 : cy + 3, cx - 2 : cx + 3, :3] = np.array(
                    [0.15, 0.15, 0.15], dtype=np.float32
                )
                rgba[cy - 2 : cy + 3, cx - 2 : cx + 3, 3] = 1.0

    def _select_palette_colour(self, index: int) -> None:
        self.state.active_colour = PALETTE[index]
        self._sync_active_colour_ui()

    def _on_custom_colour(self, _sender: int, app_data: list[float]) -> None:
        self.state.active_colour = _normalise_picker_colour(app_data)
        self._sync_active_colour_ui()

    def _clear_face_selection(self) -> None:
        self.state.selected_faces.clear()
        self._refresh_selected_count()
        self._mark_viewport_dirty()

    def _refresh_selected_count(self) -> None:
        dpg.set_value(
            "selected_count_text", f"Selected faces: {len(self.state.selected_faces)}"
        )

    def _mask_selected_faces(self) -> None:
        if self.mesh_model is None or not self.state.selected_faces:
            return
        masked = set(self.mesh_model.masked_faces).union(self.state.selected_faces)
        touched = self.commands.set_masked_faces(
            masked, description="Mask selected faces"
        )
        self._apply_updates(touched)
        self._refresh_mask_count()

    def _paint_selected_faces(self) -> None:
        if not self.state.selected_faces:
            return
        updates = {
            face_id: self.state.active_colour for face_id in self.state.selected_faces
        }
        touched = self.commands.paint_faces(updates, description="Paint selected faces")
        self._apply_updates(touched)

    def _active_document(self):
        if self.mesh_model is None or not self.mesh_model.sketch_documents:
            return None
        return self.mesh_model.sketch_documents[-1]

    def _show_main_window(self) -> None:
        if dpg.does_item_exist("home_window"):
            dpg.hide_item("home_window")
        dpg.show_item("main_window")
        dpg.set_primary_window("main_window", True)
        self._sync_workspace_ui()

    def _show_home_window(self) -> None:
        if dpg.does_item_exist("main_window"):
            dpg.hide_item("main_window")
        dpg.show_item("home_window")
        dpg.set_primary_window("home_window", True)

    def _resize_viewport_texture(self, width: int, height: int) -> None:
        new_size = (max(1, int(width)), max(1, int(height)))
        if new_size == self.state.viewport_size:
            return
        self.state.viewport_size = new_size
        self._texture_data = np.zeros(
            (self.state.viewport_size[1], self.state.viewport_size[0], 4),
            dtype=np.float32,
        )
        if self._viewport_texture_tag is not None and dpg.does_item_exist(
            self._viewport_texture_tag
        ):
            dpg.delete_item(self._viewport_texture_tag)
        self._viewport_texture_tag = dpg.add_raw_texture(
            width=self.state.viewport_size[0],
            height=self.state.viewport_size[1],
            default_value=self._texture_data,
            parent="texture_registry",
            format=dpg.mvFormat_Float_rgba,
        )
        if dpg.does_item_exist("viewport_image"):
            dpg.configure_item(
                "viewport_image",
                texture_tag=self._viewport_texture_tag,
                width=self.state.viewport_size[0],
                height=self.state.viewport_size[1],
            )
        if self.renderer is not None:
            self.renderer.resize(self.state.viewport_size)
        self._mark_viewport_dirty()

    def _sync_workspace_layout(self, *, force: bool = False) -> None:
        if not dpg.does_item_exist("main_window") or not dpg.is_item_shown(
            "main_window"
        ):
            return
        window_width = int(dpg.get_viewport_client_width())
        window_height = int(dpg.get_viewport_client_height())
        layout = compute_workspace_layout(
            window_width,
            window_height,
            show_tools_panel=self.state.workspace_mode == "paint",
        )
        if dpg.does_item_exist("tools_panel"):
            dpg.configure_item(
                "tools_panel",
                width=layout.left_panel_width,
                height=layout.workspace_height,
            )
        if dpg.does_item_exist("ai_panel"):
            dpg.configure_item(
                "ai_panel",
                width=layout.right_panel_width,
                height=layout.workspace_height,
            )
        if dpg.does_item_exist("viewport_panel"):
            dpg.configure_item(
                "viewport_panel",
                width=layout.viewport_width,
                height=layout.workspace_height,
            )
        if (
            force
            or layout.viewport_width != self.state.viewport_size[0]
            or layout.workspace_height != self.state.viewport_size[1]
        ):
            self._resize_viewport_texture(
                layout.viewport_width, layout.workspace_height
            )

    def _refresh_recent_project_list(self) -> None:
        self.state.recent_projects = [
            path for path in self.state.recent_projects if Path(path).exists()
        ]
        dpg.configure_item(
            "recent_projects_list",
            items=self.state.recent_projects or ["No recent projects yet"],
        )
        self._save_recent_projects()

    def _push_recent_project(self, path: str) -> None:
        normalized = str(Path(path))
        self.state.recent_projects = [
            item for item in self.state.recent_projects if item != normalized
        ]
        self.state.recent_projects.insert(0, normalized)
        self.state.recent_projects = [
            item
            for item in self.state.recent_projects
            if Path(item).exists() and Path(item).suffix.lower() == PROJECT_EXTENSION
        ][:20]
        self._refresh_recent_project_list()

    def _open_selected_recent_project(
        self, _sender: int | None = None, _app_data: object | None = None
    ) -> None:
        selected = dpg.get_value("recent_projects_list")
        if not selected or selected == "No recent projects yet":
            self._set_status("Select a recent project first")
            return
        self._on_open_selected(str(selected))

    def _load_mesh_model(self, mesh_model: MeshModel) -> None:
        self.mesh_model = mesh_model
        self._vis_pick_key = None
        self._vis_pick_image = None
        self._click_pick_key = None
        self._click_pick_image = None
        if not mesh_model.face_groups:
            mesh_model.compute_face_groups()
        group_count = len(mesh_model.face_groups)
        self.camera = OrbitCamera.for_mesh(mesh_model.vertices)
        gpu_available, gpu_info = probe_gpu_support()
        logger.info("GPU detection: available=%s, info=%s", gpu_available, gpu_info)
        self.renderer = MeshRenderer(
            None, mesh_model, self.state.viewport_size, prefer_gpu=gpu_available
        )
        if group_count > 0 and self.renderer is not None:
            self.renderer.rebuild_edge_buffer()
        self.paint_tool = PaintTool(mesh_model)
        self.commands.attach(mesh_model, self.paint_tool)
        self._mark_viewport_dirty()
        info_lines = [
            f"{Path(mesh_model.source_path or 'project').name}",
            f"Faces: {mesh_model.face_count}",
            f"Vertices: {mesh_model.vertex_count}",
        ]
        if mesh_model.has_cad_faces:
            info_lines.append(
                f"CAD faces: {mesh_model.cad_face_count} "
                f"({mesh_model.cad_solid_count} solids)"
            )
        info_lines.append(f"Masked: {len(mesh_model.masked_faces)}")
        info_lines.append(f"Scale: {mesh_model.model_scale:.4f}x")
        renderer_mode = (
            "GPU"
            if self.renderer is not None
            and bool(getattr(self.renderer, "_gpu_ready", False))
            else "Software"
        )
        info_lines.append(f"Renderer: {renderer_mode}")
        dpg.set_value("mesh_info_text", "\n".join(info_lines))
        dpg.set_value("model_scale_factor", 1.0)
        dpg.set_value("face_groups_count_text", f"Groups: {group_count}")
        self._refresh_mask_count()
        self._refresh_selected_count()
        self._refresh_timeline()
        self._show_main_window()
        self._set_workspace_mode("preview")
        self._set_status(
            f"Loaded {Path(mesh_model.source_path or 'project').name} | Preview mode active, press Tab to paint"
        )

    def _refresh_timeline(self) -> None:
        descriptions = self.commands.timeline_descriptions()
        current = self.commands.timeline_index()
        max_index = max(0, len(descriptions) - 1)
        dpg.configure_item("timeline_slider", max_value=max_index)
        dpg.set_value("timeline_slider", min(current, max_index))
        current_desc = (
            descriptions[min(current, max_index)] if descriptions else "Initial state"
        )
        dpg.set_value("timeline_status", f"Step {current}/{max_index} | {current_desc}")
        self._draw_timeline()

    def _build_timeline_panel(self) -> None:
        with dpg.child_window(
            height=110,
            border=True,
            tag="timeline_panel",
            no_scrollbar=False,
        ):
            with dpg.group(horizontal=True):
                dpg.add_text("History", color=(55, 65, 81))
                dpg.add_text("", tag="timeline_status")
            dpg.add_spacer(height=4)
            dpg.add_drawlist(
                width=-1,
                height=50,
                tag="timeline_drawlist",
            )
            dpg.add_slider_int(
                tag="timeline_slider",
                min_value=0,
                max_value=0,
                default_value=0,
                callback=self._on_timeline_change,
                width=-1,
            )
            with dpg.handler_registry(tag="timeline_handler"):
                dpg.add_mouse_click_handler(
                    button=dpg.mvMouseButton_Right,
                    callback=self._on_timeline_right_click,
                )
        dpg.add_text("", tag="timeline_message")

    def _draw_timeline(self) -> None:
        if not dpg.does_item_exist("timeline_drawlist"):
            return
        dpg.delete_item("timeline_drawlist", children_only=True)
        descriptions = self.commands.timeline_descriptions()
        current = self.commands.timeline_index()
        if not descriptions:
            descriptions = ["Initial state"]
            current = 0
        drawlist_width = dpg.get_item_rect_size("timeline_drawlist")[0]
        if drawlist_width <= 0:
            drawlist_width = 800
        item_width = max(
            60, min(120, (drawlist_width - 40) // max(1, len(descriptions)))
        )
        total_width = item_width * len(descriptions)
        start_x = max(20, (drawlist_width - total_width) // 2)
        y_center = 25
        y_top = 8
        y_bottom = 42
        for i, desc in enumerate(descriptions):
            x = start_x + i * item_width
            is_current = i == current
            is_past = i < current
            if is_current:
                fill_colour = (59, 130, 246, 255)
                text_colour = (255, 255, 255, 255)
                border_colour = (37, 99, 235, 255)
            elif is_past:
                fill_colour = (219, 234, 254, 255)
                text_colour = (31, 41, 55, 255)
                border_colour = (147, 197, 253, 255)
            else:
                fill_colour = (243, 244, 246, 255)
                text_colour = (107, 114, 128, 255)
                border_colour = (209, 213, 219, 255)
            rect_x1 = x + 2
            rect_y1 = y_top
            rect_x2 = x + item_width - 2
            rect_y2 = y_bottom
            dpg.draw_rectangle(
                [rect_x1, rect_y1],
                [rect_x2, rect_y2],
                fill=fill_colour,
                color=border_colour,
                thickness=2,
                rounding=4,
                parent="timeline_drawlist",
            )
            short_desc = desc[:12] if len(desc) > 12 else desc
            text_width = len(short_desc) * 5
            text_x = x + (item_width - text_width) // 2
            dpg.draw_text(
                (text_x, y_center - 6),
                short_desc,
                color=text_colour,
                size=14,
                parent="timeline_drawlist",
            )
            if i < len(descriptions) - 1:
                line_start_x = x + item_width
                line_end_x = x + item_width + 2
                dpg.draw_line(
                    (line_start_x, y_center),
                    (line_end_x, y_center),
                    color=(148, 163, 184, 180),
                    thickness=2,
                    parent="timeline_drawlist",
                )
        arrow_x = start_x + current * item_width + item_width // 2
        arrow_y = y_bottom + 2
        dpg.draw_triangle(
            (arrow_x - 5, arrow_y),
            (arrow_x + 5, arrow_y),
            (arrow_x, arrow_y + 6),
            fill=(59, 130, 246, 255),
            color=(37, 99, 235, 255),
            thickness=1,
            parent="timeline_drawlist",
        )

    def _on_timeline_right_click(self, _sender: int, _app_data: object) -> None:
        if not dpg.does_item_exist("timeline_drawlist"):
            return
        if not dpg.is_item_hovered("timeline_drawlist"):
            return
        mouse_x, mouse_y = dpg.get_mouse_pos(local=True)
        descriptions = self.commands.timeline_descriptions()
        if not descriptions:
            descriptions = ["Initial state"]
        drawlist_width = dpg.get_item_rect_size("timeline_drawlist")[0]
        if drawlist_width <= 0:
            drawlist_width = 800
        item_width = max(
            60, min(120, (drawlist_width - 40) // max(1, len(descriptions)))
        )
        total_width = item_width * len(descriptions)
        start_x = max(20, (drawlist_width - total_width) // 2)
        clicked_index = int((mouse_x - start_x) // item_width)
        if clicked_index < 0 or clicked_index >= len(descriptions):
            return
        self._timeline_context_index = clicked_index
        if dpg.does_item_exist("timeline_context_menu"):
            dpg.delete_item("timeline_context_menu")
        with dpg.window(
            tag="timeline_context_menu",
            label="",
            no_title_bar=True,
            no_move=False,
            no_resize=True,
            popup=True,
            autosize=True,
            no_scrollbar=True,
        ):
            dpg.add_text(f"Action: {descriptions[clicked_index]}", color=(55, 65, 81))
            dpg.add_separator()
            dpg.add_menu_item(
                label="Jump to here",
                callback=lambda: self._timeline_jump_to(self._timeline_context_index),
            )
            current = self.commands.timeline_index()
            if clicked_index < current:
                dpg.add_menu_item(
                    label="Undo to here",
                    callback=lambda: self._timeline_undo_to(
                        self._timeline_context_index
                    ),
                )
            if clicked_index > current:
                dpg.add_menu_item(
                    label="Redo to here",
                    callback=lambda: self._timeline_redo_to(
                        self._timeline_context_index
                    ),
                )
            if clicked_index > 0:
                dpg.add_menu_item(
                    label="Undo this action",
                    callback=lambda: self._timeline_undo_one(
                        self._timeline_context_index
                    ),
                )
        dpg.configure_item("timeline_context_menu", pos=[mouse_x, mouse_y])
        dpg.show_item("timeline_context_menu")

    def _timeline_jump_to(self, index: int) -> None:
        touched = self.commands.jump_to_timeline_index(index)
        self._apply_updates(touched)
        self._refresh_mask_count()
        self._set_status(f"Jumped to step {index}")

    def _timeline_undo_to(self, index: int) -> None:
        touched = self.commands.jump_to_timeline_index(index)
        self._apply_updates(touched)
        self._refresh_mask_count()
        self._set_status(f"Undid to step {index}")

    def _timeline_redo_to(self, index: int) -> None:
        touched = self.commands.jump_to_timeline_index(index)
        self._apply_updates(touched)
        self._refresh_mask_count()
        self._set_status(f"Redid to step {index}")

    def _timeline_undo_one(self, index: int) -> None:
        current = self.commands.timeline_index()
        if index == current:
            self._on_undo()
        else:
            touched = self.commands.jump_to_timeline_index(index - 1)
            self._apply_updates(touched)
            self._refresh_mask_count()
            self._set_status(f"Undid action at step {index}")

    def should_show_triangle_edges(self) -> bool:
        """Paint-mode outline switch.

        Paint mode outlines triangles so the mesh visibly changes when
        entering texture-paint mode. The outlines are occlusion-correct:
        the GPU path draws front-facing edges only (see
        ``MeshRenderer._front_edge_vao_for_camera``) and the software path
        paints back-to-front, so back edges never shine through the model
        like transparency. Kept as a helper so it is unit-testable.
        """
        try:
            return bool(self.state.workspace_mode == "paint")
        except Exception:
            return False

    def _is_face_front_visible(self, face_id: int) -> bool:
        """Cheap front-facing check (backface cull, not true occlusion)."""
        if self.mesh_model is None:
            return False
        if not 0 <= int(face_id) < self.mesh_model.face_count:
            return False
        normal = self.mesh_model.normals[int(face_id)].astype(np.float32)
        to_camera = self.camera.position().astype(
            np.float32
        ) - self.mesh_model.face_center(int(face_id)).astype(np.float32)
        return bool(float(np.dot(normal, to_camera)) > 0.0)

    def _cached_pick_image(self) -> np.ndarray | None:
        """Software face-ID image for occlusion tests, cached by camera key.

        Pure-PIL render (no GL calls, so safe on the Dear PyGui thread).
        Only available when the renderer is in software mode; in GPU mode
        returns None and callers fall back to the facing test. Rebuilt at
        most once per camera pose.
        """
        if self.mesh_model is None or self.renderer is None:
            return None
        try:
            if bool(getattr(self.renderer, "_gpu_ready", False)):
                return None
            key = (
                self.renderer._camera_key(self.camera),
                int(self.mesh_model.face_count),
            )
            if (
                getattr(self, "_vis_pick_key", None) != key
                or getattr(self, "_vis_pick_image", None) is None
            ):
                self._vis_pick_image = (
                    self.renderer._build_pick_image_software(self.camera)
                )
                self._vis_pick_key = key
            image = self._vis_pick_image
            width, height = tuple(self.state.viewport_size)
            if image.shape[1] != int(width) or image.shape[0] != int(height):
                return None
            return image
        except Exception:
            logger.debug("pick-image visibility unavailable", exc_info=True)
            return None

    def _is_face_visible(
        self,
        face_id: int,
        screen_point: tuple[int, int] | None = None,
        pick_image: np.ndarray | None = None,
    ) -> bool:
        """True occlusion test for overlay dots and marquee selection.

        With a software pick image, the face counts as visible when its
        projected center pixel shows that same face (or the same CAD face,
        so triangulation neighbours on one flat CAD plane pass). Without a
        pick image (GPU mode / headless) falls back to the front-facing
        check. When ``pick_image`` is omitted the cached image is fetched
        automatically; pass an explicit image only to share one render
        across many lookups.
        """
        if self.mesh_model is None:
            return False
        fid = int(face_id)
        if not 0 <= fid < self.mesh_model.face_count:
            return False
        point = screen_point
        if point is None:
            point = self._project_world_to_screen(self.mesh_model.face_center(fid))
        if point is None:
            return False
        if pick_image is None:
            pick_image = self._cached_pick_image()
        if pick_image is None:
            return self._is_face_front_visible(fid)
        try:
            cx, cy = int(point[0]), int(point[1])
            height, width = int(pick_image.shape[0]), int(pick_image.shape[1])
            want_cad: object = None
            if self.mesh_model.has_cad_faces:
                try:
                    want_cad = self.mesh_model.cad_id_for_face(fid)
                except Exception:
                    want_cad = None
            # 3x3 window: dense-mesh faces can be a few pixels, so the exact
            # center pixel may rasterize to a neighbour even when the face
            # itself is visible.
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    x, y = cx + dx, cy + dy
                    if not 0 <= x < width and 0 <= y < height:
                        continue
                    r, g, b = (int(v) for v in pick_image[y, x])
                    shown = (r << 16) | (g << 8) | b
                    if shown == fid:
                        return True
                    if shown >= self.mesh_model.face_count:
                        continue
                    if want_cad is not None:
                        try:
                            if self.mesh_model.cad_id_for_face(int(shown)) == want_cad:
                                return True
                        except Exception:
                            pass
            # Encoded black is ambiguous between background and face 0.
            if fid == 0 and self._is_face_front_visible(fid):
                x0 = max(0, min(cx, width - 1))
                y0 = max(0, min(cy, height - 1))
                if tuple(int(v) for v in pick_image[y0, x0]) == (0, 0, 0):
                    return True
            return False
        except Exception:
            logger.debug("face visibility lookup failed", exc_info=True)
            return self._is_face_front_visible(fid)

    def _render_snapshot(self) -> RenderSnapshot:
        if self.mesh_model is None or self.renderer is None:
            return make_grid_snapshot(self.state.viewport_size)
        return self.renderer.render(
            self.camera,
            show_triangle_edges=self.should_show_triangle_edges(),
        )

    def _project_world_to_screen(self, point: np.ndarray) -> tuple[int, int] | None:
        mvp = self.camera.mvp_matrix(self.state.viewport_size)
        clip = mvp @ np.array([point[0], point[1], point[2], 1.0], dtype=np.float32)
        w = float(clip[3])
        if abs(w) < 1e-6:
            return None
        ndc = clip[:3] / w
        width, height = self.state.viewport_size
        x = int((ndc[0] * 0.5 + 0.5) * width)
        y = int((1.0 - (ndc[1] * 0.5 + 0.5)) * height)
        return x, y

    def _project_world_batch_to_screen(
        self, points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project many world points at once (vectorized hover outlines).

        Returns ``(pixels, valid)`` with ``pixels`` shaped ``(N, 2)`` int
        and a bool validity mask. One ``mvp`` multiply for all points
        instead of thousands of per-vertex Python calls.
        """
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        if len(pts) == 0:
            return np.zeros((0, 2), dtype=np.int32), np.zeros((0,), dtype=bool)
        mvp = self.camera.mvp_matrix(self.state.viewport_size)
        ones = np.ones((len(pts), 1), dtype=np.float32)
        clip = (mvp @ np.hstack([pts, ones]).T).T
        w = clip[:, 3:4]
        safe = np.where(np.abs(w) < 1e-6, 1e-6, w)
        ndc = clip[:, :3] / safe
        valid = np.isfinite(ndc).all(axis=1) & (np.abs(w[:, 0]) >= 1e-6)
        width, height = self.state.viewport_size
        pixels = np.empty((len(pts), 2), dtype=np.int32)
        pixels[:, 0] = ((ndc[:, 0] * 0.5 + 0.5) * width).astype(np.int32)
        pixels[:, 1] = ((1.0 - (ndc[:, 1] * 0.5 + 0.5)) * height).astype(np.int32)
        return pixels, valid

    def _draw_line(
        self,
        rgba: np.ndarray,
        p0: tuple[int, int],
        p1: tuple[int, int],
        colour: tuple[float, float, float],
        alpha: float = 1.0,
    ) -> None:
        x0, y0 = p0
        x1, y1 = p1
        dx = x1 - x0
        dy = y1 - y0
        steps = max(abs(dx), abs(dy), 1)
        xs = np.linspace(x0, x1, steps + 1, dtype=np.int32)
        ys = np.linspace(y0, y1, steps + 1, dtype=np.int32)
        h, w, _ = rgba.shape
        mask = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        xs = xs[mask]
        ys = ys[mask]
        rgba[ys, xs, :3] = (
            rgba[ys, xs, :3] * (1.0 - alpha)
            + np.array(colour, dtype=np.float32) * alpha
        )
        rgba[ys, xs, 3] = 1.0

    def _draw_point(
        self,
        rgba: np.ndarray,
        point: tuple[int, int],
        colour: tuple[float, float, float],
    ) -> None:
        x, y = point
        h, w, _ = rgba.shape
        if 1 <= x < w - 1 and 1 <= y < h - 1:
            rgba[y - 1 : y + 2, x - 1 : x + 2, :3] = np.array(colour, dtype=np.float32)
            rgba[y - 1 : y + 2, x - 1 : x + 2, 3] = 1.0

    def _overlay_sketch_entities(self, rgba: np.ndarray) -> None:
        document = self._active_document()
        if document is None:
            return
        preview_entity = None
        if (
            self.state.dragging
            and self.state.interaction_mode == "sketch"
            and self.state.sketch_tool in {"rect", "line", "circle"}
            and self.state.drag_origin_plane is not None
        ):
            current_uv = self._screen_hit_on_plane(self._viewport_mouse_position())
            if current_uv is not None:
                preview_entity = self.sketch_tool.create_entity(
                    self.state.sketch_tool,
                    self.state.drag_origin_plane,
                    current_uv,
                    self.state.active_colour,
                    text=str(dpg.get_value("sketch_text_value")),
                    stroke_width=float(dpg.get_value("sketch_stroke_width")),
                    text_size=int(dpg.get_value("sketch_text_size")),
                )
        for entity in document.entities + (
            [preview_entity] if preview_entity is not None else []
        ):
            colour = (
                np.array(
                    entity.data.get("colour", [255, 80, 80, 255]), dtype=np.float32
                )[:3]
                / 255.0
            )
            alpha = (
                0.45
                if preview_entity is not None
                and entity.entity_id == preview_entity.entity_id
                else 1.0
            )
            if entity.kind == "rect":
                min_uv = np.asarray(entity.data["min"], dtype=np.float32)
                max_uv = np.asarray(entity.data["max"], dtype=np.float32)
                corners = [
                    min_uv,
                    np.asarray([max_uv[0], min_uv[1]], dtype=np.float32),
                    max_uv,
                    np.asarray([min_uv[0], max_uv[1]], dtype=np.float32),
                ]
                screens = [
                    self._project_world_to_screen(
                        self.sketch_tool.plane_to_world(document.plane, uv)
                    )
                    for uv in corners
                ]
                if all(screen is not None for screen in screens):
                    for index in range(4):
                        self._draw_line(
                            rgba,
                            screens[index],
                            screens[(index + 1) % 4],
                            tuple(colour),
                            alpha=alpha,
                        )  # type: ignore[arg-type]
            elif entity.kind == "line":
                start = self.sketch_tool.plane_to_world(
                    document.plane, np.asarray(entity.data["start"], dtype=np.float32)
                )
                end = self.sketch_tool.plane_to_world(
                    document.plane, np.asarray(entity.data["end"], dtype=np.float32)
                )
                p0 = self._project_world_to_screen(start)
                p1 = self._project_world_to_screen(end)
                if p0 and p1:
                    self._draw_line(rgba, p0, p1, tuple(colour), alpha=alpha)
            elif entity.kind == "circle":
                center = np.asarray(entity.data["center"], dtype=np.float32)
                radius = float(entity.data["radius"])
                ring = [
                    self.sketch_tool.plane_to_world(
                        document.plane,
                        center
                        + np.asarray(
                            [np.cos(angle) * radius, np.sin(angle) * radius],
                            dtype=np.float32,
                        ),
                    )
                    for angle in np.linspace(0.0, np.pi * 2.0, 24, endpoint=False)
                ]
                screens = [self._project_world_to_screen(point) for point in ring]
                valid = [screen for screen in screens if screen is not None]
                if len(valid) >= 2:
                    for index in range(len(valid)):
                        self._draw_line(
                            rgba,
                            valid[index],
                            valid[(index + 1) % len(valid)],
                            tuple(colour),
                            alpha=alpha,
                        )
            elif entity.kind == "text":
                point = self.sketch_tool.plane_to_world(
                    document.plane,
                    np.asarray(entity.data["position"], dtype=np.float32),
                )
                screen = self._project_world_to_screen(point)
                if screen:
                    self._draw_point(rgba, screen, tuple(colour))
            if entity.entity_id == document.selected_entity_id:
                for snap_uv in entity_snap_points(entity):
                    point = self._project_world_to_screen(
                        self.sketch_tool.plane_to_world(document.plane, snap_uv)
                    )
                    if point:
                        self._draw_point(rgba, point, (1.0, 1.0, 1.0))

    def _overlay_masked_faces(self, rgba: np.ndarray) -> None:
        if self.mesh_model is None:
            return
        faces = list(self.mesh_model.masked_faces)[:250]
        pick_image = self._cached_pick_image() if faces else None
        for face_id in faces:
            point = self._project_world_to_screen(self.mesh_model.face_center(face_id))
            if point and self._is_face_visible(int(face_id), point, pick_image):
                self._draw_point(rgba, point, (0.0, 0.0, 0.0))

    def _update_hover_face(self, mouse_pos: tuple[float, float]) -> None:
        """Refresh the idle-hover face highlight (paint workspace only)."""
        if (
            self.mesh_model is None
            or self.state.workspace_mode != "paint"
            or self.state.interaction_mode != "paint"
            or self.state.dragging
            or self.state.nav_dragging
        ):
            if self.state.hovered_face is not None:
                self.state.hovered_face = None
                self._mark_viewport_dirty()
            return
        if self._nav_pad_hovered():
            if self.state.hovered_face is not None:
                self.state.hovered_face = None
                self._mark_viewport_dirty()
            return
        pixel = (int(mouse_pos[0]), int(mouse_pos[1]))
        now = time.monotonic()
        if pixel == self._last_hover_px and now - self._last_hover_time < 0.05:
            return
        self._last_hover_px = pixel
        self._last_hover_time = now
        face_id: int | None = None
        try:
            # Fast path: sample the GPU pick cache warmed by the main loop
            # (no GL calls, so safe on the Dear PyGui frame thread).
            # Falls back to the CPU raycaster when the cache is stale
            # (camera just moved and the next frame has not warmed it yet).
            # Direct GPU renders here are forbidden: binding the pick
            # framebuffer mid-frame access-violates on some drivers
            # (observed on AMD Radeon), so never call renderer.pick_face
            # or warm_pick_cache from this handler.
            if self.renderer is not None:
                x, y, _ = self._to_render_coords(
                    (float(pixel[0]), float(pixel[1]))
                )
                face_id = self.renderer.cached_pick_face(
                    self.camera, int(x), int(y)
                )
            if face_id is None:
                pick = self._pick_result((float(pixel[0]), float(pixel[1])))
                face_id = pick.face_id if pick is not None else None
        except Exception:
            logger.debug("Hover pick failed", exc_info=True)
            face_id = None
        logger.info("HOVER_PICK | pixel=%s face=%s", pixel, face_id)
        if face_id != self.state.hovered_face:
            self.state.hovered_face = face_id
            self._mark_viewport_dirty()

    def _overlay_hovered_face(self, rgba: np.ndarray) -> None:
        if self.mesh_model is None or self.state.hovered_face is None:
            return
        face_id = int(self.state.hovered_face)
        if not 0 <= face_id < self.mesh_model.face_count:
            return
        if self.mesh_model.has_cad_faces:
            self._overlay_hovered_cad_face(rgba, face_id)
            return
        corners = [
            self._project_world_to_screen(vertex)
            for vertex in self.mesh_model.face_vertices(face_id)
        ]
        if any(point is None for point in corners):
            return
        p0, p1, p2 = corners
        assert p0 is not None and p1 is not None and p2 is not None
        self._draw_line(rgba, p0, p1, (0.2, 0.95, 0.95), alpha=0.95)
        self._draw_line(rgba, p1, p2, (0.2, 0.95, 0.95), alpha=0.95)
        self._draw_line(rgba, p2, p0, (0.2, 0.95, 0.95), alpha=0.95)

    def _overlay_hovered_cad_face(self, rgba: np.ndarray, face_id: int) -> None:
        """Outline the whole hovered CAD face (boundary loop, no diagonals)."""
        assert self.mesh_model is not None
        cad_id = self.mesh_model.cad_id_for_face(face_id)
        if cad_id is None:
            return
        edges = self.mesh_model.cad_face_boundary_edges(cad_id)
        if len(edges) == 0:
            return
        edges = edges[:4000]
        pairs = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
        points = self.mesh_model.vertices[pairs.reshape(-1)]
        pixels, valid = self._project_world_batch_to_screen(points)
        pixels = pixels.reshape(-1, 2, 2)
        valid = valid.reshape(-1, 2)
        for index in range(len(pairs)):
            if not bool(valid[index].all()):
                continue
            p0 = (int(pixels[index, 0, 0]), int(pixels[index, 0, 1]))
            p1 = (int(pixels[index, 1, 0]), int(pixels[index, 1, 1]))
            self._draw_line(rgba, p0, p1, (0.2, 0.95, 0.95), alpha=0.95)

    def _overlay_selected_faces(self, rgba: np.ndarray) -> None:
        if self.mesh_model is None:
            return
        faces = list(self.state.selected_faces)[:500]
        pick_image = self._cached_pick_image() if faces else None
        for face_id in faces:
            point = self._project_world_to_screen(self.mesh_model.face_center(face_id))
            if point and self._is_face_visible(int(face_id), point, pick_image):
                self._draw_point(rgba, point, (1.0, 1.0, 0.0))
        if self.state.marquee_start is not None and self.state.marquee_end is not None:
            x0, y0 = int(self.state.marquee_start[0]), int(self.state.marquee_start[1])
            x1, y1 = int(self.state.marquee_end[0]), int(self.state.marquee_end[1])
            p00 = (x0, y0)
            p10 = (x1, y0)
            p11 = (x1, y1)
            p01 = (x0, y1)
            self._draw_line(rgba, p00, p10, (1.0, 1.0, 0.2), alpha=0.8)
            self._draw_line(rgba, p10, p11, (1.0, 1.0, 0.2), alpha=0.8)
            self._draw_line(rgba, p11, p01, (1.0, 1.0, 0.2), alpha=0.8)
            self._draw_line(rgba, p01, p00, (1.0, 1.0, 0.2), alpha=0.8)

    def _render_viewport(self) -> None:
        if not self.state.viewport_dirty:
            return
        snapshot = self._render_snapshot()
        rgba = snapshot.rgba.astype(np.float32) / 255.0
        if self.mesh_model is None or self.state.workspace_mode in ("preview", "paint"):
            self._draw_world_grid(rgba)
        if self.mesh_model is not None:
            self._overlay_sketch_entities(rgba)
            self._overlay_masked_faces(rgba)
            self._overlay_selected_faces(rgba)
            self._overlay_hovered_face(rgba)
        self._texture_data[:, :, :] = rgba
        if self._viewport_texture_tag is not None:
            dpg.set_value(self._viewport_texture_tag, self._texture_data)
        self._position_nav_widget()
        self._draw_nav_pad()
        self.state.viewport_dirty = False
        # Refresh the GPU pick cache for paint-mode hover/drag sampling.
        # This runs on the main loop thread (GL-safe, same as
        # _process_pending_paint); the Dear PyGui hover handler only
        # samples the cache via cached_pick_face().
        if (
            self.mesh_model is not None
            and self.renderer is not None
            and self.state.workspace_mode == "paint"
            and self.state.interaction_mode == "paint"
            and bool(getattr(self.renderer, "_gpu_ready", False))
        ):
            try:
                self.renderer.warm_pick_cache(self.camera)
            except Exception:
                logger.debug("Pick cache warm skipped", exc_info=True)

    def _mouse_inside_viewport(self) -> bool:
        mouse_x, mouse_y = dpg.get_mouse_pos(local=False)
        image_x, image_y = dpg.get_item_rect_min("viewport_image")
        width, height = dpg.get_item_rect_size("viewport_image")
        within = (
            image_x <= mouse_x < image_x + width
            and image_y <= mouse_y < image_y + height
        )
        return within or self._nav_pad_hovered()

    def _viewport_mouse_position(self) -> tuple[float, float]:
        mouse_x, mouse_y = dpg.get_mouse_pos(local=False)
        image_x, image_y = dpg.get_item_rect_min("viewport_image")
        return mouse_x - image_x, mouse_y - image_y

    def _pick_viewport_size(self) -> tuple[int, int]:
        """Render-space size matching mouse coordinates.

        Mouse positions are measured against the displayed image rect, which
        can drift from ``state.viewport_size`` after layout changes; picking
        against a stale size shifts every ray off the model.
        """
        if dpg.does_item_exist("viewport_image"):
            width, height = dpg.get_item_rect_size("viewport_image")
            if width > 0 and height > 0:
                return int(width), int(height)
        return self.state.viewport_size

    @staticmethod
    def _map_display_to_render(
        mouse_pos: tuple[float, float],
        display_size: tuple[int, int],
        render_size: tuple[int, int],
    ) -> tuple[float, float, tuple[int, int]]:
        """Map a display-space pointer position into render space.

        The displayed image rect can drift from the render size after
        layout changes; picking/projection against a stale size shifts
        every ray off the model. Pure function so it is unit-testable
        without a Dear PyGui context.
        """
        x, y = float(mouse_pos[0]), float(mouse_pos[1])
        if (
            display_size != render_size
            and render_size[0] > 0
            and render_size[1] > 0
            and display_size[0] > 0
            and display_size[1] > 0
        ):
            x = x * render_size[0] / display_size[0]
            y = y * render_size[1] / display_size[1]
        return x, y, render_size

    def _to_render_coords(
        self, mouse_pos: tuple[float, float]
    ) -> tuple[float, float, tuple[int, int]]:
        """Display-space mouse position mapped to renderer coordinates."""
        display_size = self._pick_viewport_size()
        render_size = self.state.viewport_size
        mapped_x, mapped_y, size = self._map_display_to_render(
            mouse_pos, display_size, render_size
        )
        if (display_size != render_size) and (
            render_size[0] > 0 and render_size[1] > 0
        ):
            logger.debug(
                "Pick size drift | display=%s render=%s", display_size, render_size
            )
        return mapped_x, mapped_y, size

    def _pick_result(self, mouse_pos: tuple[float, float]) -> PickResult | None:
        if self.mesh_model is None:
            return None
        x, y, size = self._to_render_coords(mouse_pos)
        result = pick_face_location_cpu(
            self.mesh_model,
            self.camera,
            x,
            y,
            size,
        )
        logger.info(
            "PICK_RESULT | mouse=%s render=%s,%s size=%s face=%s",
            mouse_pos,
            x,
            y,
            size,
            result.face_id if result is not None else None,
        )
        return result

    def _pick_result_software(self, mouse_pos: tuple[float, float]) -> PickResult | None:
        """Accurate pick for paint clicks using the software pick image.

        The CPU raycaster can disagree with the GPU rasterizer on coplanar
        or edge-on triangles, causing clicks to paint a hidden face while
        the user sees no change. The software pick image uses the same
        painter's-algorithm rasterization as the viewport, so it agrees
        with what the user sees. The image is cached per camera pose so
        repeated clicks stay responsive on dense meshes.
        """
        if self.mesh_model is None or self.renderer is None:
            return None
        x, y, size = self._to_render_coords(mouse_pos)
        px, py = int(x), int(y)
        if px < 0 or py < 0 or px >= size[0] or py >= size[1]:
            return None
        try:
            cache_key = None
            try:
                cache_key = (
                    self.renderer._camera_key(self.camera),
                    int(self.mesh_model.face_count),
                    tuple(size),
                )
            except Exception:
                cache_key = None
            pick_image = None
            if (
                cache_key is not None
                and getattr(self, "_click_pick_key", None) == cache_key
                and getattr(self, "_click_pick_image", None) is not None
            ):
                pick_image = self._click_pick_image
            else:
                pick_image = self.renderer._build_pick_image_software(
                    self.camera, size
                )
                if cache_key is not None:
                    self._click_pick_key = cache_key
                    self._click_pick_image = pick_image
        except Exception:
            logger.debug("Software pick failed", exc_info=True)
            return None
        if py >= pick_image.shape[0] or px >= pick_image.shape[1]:
            return None
        r_val, g_val, b_val = (int(v) for v in pick_image[py, px])
        face_id = (r_val << 16) | (g_val << 8) | b_val
        if face_id >= self.mesh_model.face_count:
            return None
        if face_id == 0 and (r_val, g_val, b_val) == (0, 0, 0):
            # Encoded black is ambiguous between background and face 0.
            # Resolve it the same way the renderer does, falling back to
            # the CPU picker when the render sizes differ.
            try:
                if tuple(size) == tuple(self.renderer.viewport_size):
                    if not self.renderer._point_hits_face_zero(
                        self.camera, px, py
                    ):
                        return None
                else:
                    return self._pick_result(mouse_pos)
            except Exception:
                pass
        location = self.mesh_model.face_center(face_id).astype(np.float32)
        distance = float(np.linalg.norm(location - self.camera.position()))
        logger.info(
            "PICK_RESULT_SOFTWARE | mouse=%s render=%s,%s size=%s face=%s",
            mouse_pos,
            px,
            py,
            size,
            face_id,
        )
        return PickResult(face_id=face_id, location=location, distance=distance)

    def _process_pending_paint(self) -> None:
        """Apply a queued paint click/drag in the main loop (GL-safe).

        Must run outside Dear PyGui frame callbacks: it resolves the
        clicked face with the GPU pick buffer — the same rasterizer and
        depth test as the viewport render, so the painted face is always
        the displayed one — and binding that framebuffer mid-frame
        access-violates on some drivers. Falls back to the software pick
        image, then the CPU raycaster, so a pick failure can never lose
        a stroke outright.
        """
        pending = self._pending_paint_click
        self._pending_paint_click = None
        if pending is None:
            return
        mouse_pos, is_drag = pending
        if (
            self.mesh_model is None
            or self.renderer is None
            or self.state.workspace_mode != "paint"
            or self.state.interaction_mode != "paint"
            or self.state.paint_tool in {"select"}
        ):
            return
        x, y, size = self._to_render_coords(mouse_pos)
        face_id: int | None = None
        try:
            face_id = self.renderer.pick_face(self.camera, int(x), int(y))
        except Exception:
            logger.debug("Display pick failed; falling back", exc_info=True)
            face_id = None
        pick: PickResult | None = None
        if face_id is not None:
            location = self.mesh_model.face_center(face_id).astype(np.float32)
            distance = float(np.linalg.norm(location - self.camera.position()))
            pick = PickResult(face_id=face_id, location=location, distance=distance)
            logger.info(
                "PAINT_CLICK | mouse=%s pick=%s (display)", mouse_pos, face_id
            )
        else:
            pick = self._pick_result_software(mouse_pos)
            if pick is None:
                pick = self._pick_result(mouse_pos)
            logger.info(
                "PAINT_CLICK | mouse=%s pick=%s (fallback)",
                mouse_pos,
                pick.face_id if pick else None,
            )
        if pick is None:
            return
        if is_drag and pick.face_id == self._last_brush_face:
            return
        self._apply_paint_at_pick(pick)
        self._last_brush_face = pick.face_id

    def _apply_updates(self, touched: list[int]) -> None:
        if self.renderer is not None and touched:
            self.renderer.update_face_colours(touched)
        self._mark_viewport_dirty()
        self._refresh_timeline()

    def _apply_paint_at_pick(self, pick: PickResult) -> None:
        if self.mesh_model is None or self.paint_tool is None:
            logger.info("APPLY_PAINTA_PICK | early exit mesh=%s pt=%s", self.mesh_model is not None, self.paint_tool is not None)
            return
        face_id = int(pick.face_id)
        logger.info("APPLY_PAINTA_PICK | face=%s faces=%s tool=%s", face_id, self.mesh_model.face_count, self.state.paint_tool)
        if not 0 <= face_id < self.mesh_model.face_count:
            logger.warning(
                "Paint click ignored | face=%s out of range (faces=%s)",
                pick.face_id,
                self.mesh_model.face_count,
            )
            self._set_status("Pick missed the model — no face under the cursor")
            return
        tool = self.state.paint_tool
        if tool == "sample":
            self.state.active_colour = self.mesh_model.face_colour(face_id)
            self._sync_active_colour_ui()
            self._set_status(f"Sampled face {face_id}")
            return
        if tool == "mask":
            updated = set(self.mesh_model.masked_faces)
            if face_id in updated:
                updated.remove(face_id)
            else:
                updated.add(face_id)
            touched = self.commands.set_masked_faces(
                updated, description="Toggle face mask"
            )
            self._apply_updates(touched)
            self._refresh_mask_count()
            return
        cad_painted = False
        if self.mesh_model.has_cad_faces and tool in ("brush", "erase", "fill"):
            # STEP models paint whole CAD faces so colour boundaries follow
            # the CAD shape (rectangles/circles), never triangulation.
            cad_id = self.mesh_model.cad_id_for_face(face_id)
            if cad_id is None:
                logger.warning(
                    "Paint click ignored | face=%s has no CAD group (faces=%s)",
                    face_id,
                    self.mesh_model.face_count,
                )
                self._set_status("Face has no CAD group — cannot paint it")
                return
            group_colour = (
                self.state.active_colour
                if tool != "erase"
                else self.mesh_model.default_colour
            )
            updates = {
                member_id: group_colour
                for member_id in self.mesh_model.faces_for_cad(cad_id)
                if member_id not in self.mesh_model.masked_faces
            }
            if not updates:
                logger.info(
                    "Paint click | tool=%s face=%s cad=%s all masked",
                    tool,
                    face_id,
                    cad_id,
                )
                self._set_status("Face is masked — unmask or Clear Mask to paint it")
                return
            cad_painted = True
        elif tool == "fill":
            updates = self.paint_tool.flood_fill_updates(
                face_id,
                self.state.active_colour,
                tolerance=self.state.fill_tolerance,
            )
        else:
            # Clicks dab with the full brush radius (not a single face), so a
            # click stays visible on dense meshes where one face is ~pixels.
            # The centre face always takes full weight (see brush_updates).
            # (Only brush/erase reach here; sample/mask/select return above.)
            radius = max(
                0.0001,
                self.state.brush.radius * max(1.0, self.mesh_model.mesh_diagonal()),
            )
            updates = self.paint_tool.brush_updates(
                face_id,
                pick.location,
                self.state.active_colour,
                radius=radius,
                opacity=self.state.brush.opacity,
                falloff=self.state.brush.falloff,
                front_faces_only=self.state.brush.front_faces_only,
                angle_tolerance_degrees=self.state.brush.angle_tolerance_degrees,
                erase=(tool == "erase"),
            )
        if (
            not cad_painted
            and self.state.paint_linked_faces
            and self.mesh_model is not None
            and updates
        ):
            group_id = self.mesh_model.group_for_face(face_id)
            if group_id is not None:
                grouped_faces = self.mesh_model.faces_for_group(group_id)
                group_colour = (
                    self.state.active_colour
                    if tool != "erase"
                    else self.mesh_model.default_colour
                )
                for member_id in grouped_faces:
                    if member_id not in self.mesh_model.masked_faces:
                        updates[member_id] = group_colour
        touched = self.commands.paint_faces(
            updates, description=f"{tool.title()} stroke"
        )
        logger.info(
            "Paint click | tool=%s face=%s candidates=%s touched=%s active=%s",
            tool,
            face_id,
            len(updates),
            len(touched),
            self.state.active_colour,
        )
        if not touched:
            logger.info("PAINT_CLICK_EMPTY | face=%s masked=%s", face_id, face_id in self.mesh_model.masked_faces)
            if face_id in self.mesh_model.masked_faces:
                self._set_status("Face is masked — unmask or Clear Mask to paint it")
            else:
                self._set_status("No change — face already has this colour")
            return
        logger.info("PAINT_CLICK_APPLY | face=%s touched=%s", face_id, len(touched))
        self._apply_updates(touched)
        self._set_status(f"Painted {len(touched)} face(s)")

    def _start_sketch_plane(self, face_id: int) -> None:
        if self.mesh_model is None:
            return
        document = self.sketch_tool.create_plane_from_face(self.mesh_model, face_id)
        self.mesh_model.sketch_documents = [document]
        self._mark_viewport_dirty()
        self._set_status(f"Sketch plane anchored to face {face_id}")

    def _screen_hit_on_plane(self, mouse_pos: tuple[float, float]) -> np.ndarray | None:
        document = self._active_document()
        if document is None or self.mesh_model is None:
            return None
        hit = self.sketch_tool.ray_to_plane(
            document.plane, self.camera, mouse_pos, self.state.viewport_size
        )
        if hit is None:
            return None
        shift_lock = self.state.drag_origin_plane if self.state.shift_down else None
        return self.sketch_tool.snap_point(
            self.mesh_model, document, hit.plane_uv, shift_lock_axis=shift_lock
        ).plane_uv

    def _find_entity_handle(
        self, mouse_pos: tuple[float, float]
    ) -> tuple[str, str] | None:
        document = self._active_document()
        if document is None:
            return None
        best: tuple[str, str] | None = None
        best_distance = 18.0
        handle_names = {
            "rect": ["min", "max", "center"],
            "line": ["start", "end"],
            "circle": ["radius"],
            "text": ["position"],
            "svg": ["min", "max", "center"],
        }
        for entity in document.entities:
            names = handle_names.get(entity.kind, ["position"])
            for index, point_uv in enumerate(entity_snap_points(entity)[: len(names)]):
                screen = self._project_world_to_screen(
                    self.sketch_tool.plane_to_world(document.plane, point_uv)
                )
                if screen is None:
                    continue
                distance = float(
                    np.linalg.norm(
                        np.asarray(screen, dtype=np.float32)
                        - np.asarray(mouse_pos, dtype=np.float32)
                    )
                )
                if distance < best_distance:
                    best = (
                        entity.entity_id,
                        names[index] if index < len(names) else "position",
                    )
                    best_distance = distance
        return best

    def _select_entity_at_mouse(self, mouse_pos: tuple[float, float]) -> None:
        document = self._active_document()
        if document is None:
            return
        handle = self._find_entity_handle(mouse_pos)
        document.selected_entity_id = handle[0] if handle is not None else None
        self._mark_viewport_dirty()

    def _on_mouse_down(self, _sender: int, app_data: tuple[int, float]) -> None:
        button = app_data[0] if isinstance(app_data, (tuple, list)) else int(app_data)
        logger.info(
            "MOUSE_DOWN | button=%s viewport=%s paint_mode=%s interaction=%s tool=%s",
            button,
            self._mouse_inside_viewport(),
            self.state.workspace_mode,
            self.state.interaction_mode,
            self.state.paint_tool,
        )
        if self._nav_pad_hovered() and button == 0:
            nav_pos = self._nav_pad_mouse_position()
            if nav_pos is not None:
                action, axis = self._nav_hit_test(nav_pos)
                if action != "none":
                    self.state.nav_dragging = True
                    self.state.nav_pressed_axis = axis
                    self.state.nav_drag_moved = False
                    self._clip_nav_cursor()
                    self.state.nav_drag_origin = self._center_nav_cursor()
                    if self.state.nav_drag_origin is None:
                        self.state.nav_drag_origin = self._global_mouse_position()
                    return
        if not self._mouse_inside_viewport():
            return
        mouse_pos = self._viewport_mouse_position()
        self.state.dragging = True
        self.state.drag_button = button
        self.state.drag_origin_screen = mouse_pos
        self.state.drag_engaged = False
        if button != 0:
            return
        if self.state.workspace_mode == "preview":
            return
        if self.state.interaction_mode == "paint":
            if self.state.paint_tool == "select":
                self.state.marquee_start = mouse_pos
                self.state.marquee_end = mouse_pos
                self._mark_viewport_dirty()
            else:
                # Queue for the main loop: picking must match the displayed
                # pixels (GPU pick buffer), which is only safe to touch
                # outside Dear PyGui frame callbacks (see
                # _process_pending_paint).
                self._pending_paint_click = (mouse_pos, False)
            return
        if self._active_document() is None:
            pick = self._pick_result(mouse_pos)
            if pick is not None:
                self._start_sketch_plane(pick.face_id)
            return
        if self.state.sketch_tool == "select":
            handle = self._find_entity_handle(mouse_pos)
            if handle is not None:
                self.state.selected_entity_id = handle[0]
                self.state.active_handle = handle[1]
                self._active_document().selected_entity_id = handle[0]
            else:
                self._select_entity_at_mouse(mouse_pos)
            return
        start_uv = self._screen_hit_on_plane(mouse_pos)
        if start_uv is None:
            return
        self.state.drag_origin_plane = start_uv
        if self.state.sketch_tool == "text":
            entity = self.sketch_tool.create_entity(
                "text",
                start_uv,
                start_uv,
                self.state.active_colour,
                text=str(dpg.get_value("sketch_text_value")),
                text_size=int(dpg.get_value("sketch_text_size")),
            )
            self.commands.add_sketch_entity(self._active_document(), entity)
            self._mark_viewport_dirty()
        elif self.state.sketch_tool == "svg":
            path = str(dpg.get_value("recent_svg_combo")).strip()
            if not path:
                self._set_status("Import/select an SVG first")
                return
            entity = self.sketch_tool.create_entity(
                "svg",
                start_uv,
                start_uv + np.asarray([0.2, 0.2], dtype=np.float32),
                self.state.svg_tint,
            )
            entity.data["path"] = path
            entity.data["tint"] = list(self.state.svg_tint)
            self.commands.add_sketch_entity(self._active_document(), entity)
            self._mark_viewport_dirty()

    def _on_mouse_release(
        self, _sender: int, app_data: tuple[int, float] | int
    ) -> None:
        button = app_data[0] if isinstance(app_data, (tuple, list)) else int(app_data)
        if self.state.nav_dragging:
            if (
                button == 0
                and not self.state.nav_drag_moved
                and self.state.nav_pressed_axis is not None
            ):
                self._snap_camera_to_axis(self.state.nav_pressed_axis)
            self.state.nav_dragging = False
            self.state.nav_drag_origin = None
            self.state.nav_pressed_axis = None
            self.state.nav_drag_moved = False
            self._release_cursor_clip()
            return
        if not self.state.dragging:
            return
        mouse_pos = self._viewport_mouse_position()
        if (
            button == 0
            and self.state.workspace_mode == "paint"
            and self.state.interaction_mode == "sketch"
            and self._active_document() is not None
        ):
            document = self._active_document()
            if (
                self.state.sketch_tool in {"rect", "line", "circle"}
                and self.state.drag_origin_plane is not None
            ):
                end_uv = self._screen_hit_on_plane(mouse_pos)
                if end_uv is not None:
                    entity = self.sketch_tool.create_entity(
                        self.state.sketch_tool,
                        self.state.drag_origin_plane,
                        end_uv,
                        self.state.active_colour,
                        text=str(dpg.get_value("sketch_text_value")),
                        stroke_width=float(dpg.get_value("sketch_stroke_width")),
                        text_size=int(dpg.get_value("sketch_text_size")),
                    )
                    self.commands.add_sketch_entity(document, entity)
                    self._mark_viewport_dirty()
            elif (
                self.state.sketch_tool == "select"
                and self.state.selected_entity_id is not None
                and self.state.active_handle is not None
            ):
                entity = next(
                    (
                        item
                        for item in document.entities
                        if item.entity_id == self.state.selected_entity_id
                    ),
                    None,
                )
                end_uv = self._screen_hit_on_plane(mouse_pos)
                if entity is not None and end_uv is not None:
                    new_data = self.sketch_tool.resize_entity(
                        entity, self.state.active_handle, end_uv
                    )
                    self.commands.update_sketch_entity(
                        document, entity.entity_id, new_data
                    )
                    self._mark_viewport_dirty()
        if (
            button == 0
            and self.state.workspace_mode == "paint"
            and self.state.interaction_mode == "paint"
            and self.state.paint_tool == "select"
        ):
            if self.state.drag_engaged:
                self._finish_marquee_selection()
            else:
                self.state.marquee_start = None
                self.state.marquee_end = None
        self.state.dragging = False
        self.state.drag_button = None
        self.state.drag_origin_screen = None
        self.state.drag_origin_plane = None
        self.state.active_handle = None
        self.state.drag_engaged = False

    def _finish_marquee_selection(self) -> None:
        if (
            self.mesh_model is None
            or self.state.marquee_start is None
            or self.state.marquee_end is None
        ):
            return
        # Marquee corners are display-space; projected face centers are
        # render-space, so map the rect before comparing (same mapping as
        # picking) or a scaled viewport selects nothing.
        start_x, start_y, _ = self._to_render_coords(
            (float(self.state.marquee_start[0]), float(self.state.marquee_start[1]))
        )
        end_x, end_y, _ = self._to_render_coords(
            (float(self.state.marquee_end[0]), float(self.state.marquee_end[1]))
        )
        min_x = min(start_x, end_x)
        max_x = max(start_x, end_x)
        min_y = min(start_y, end_y)
        max_y = max(start_y, end_y)
        selected: set[int] = set()
        # Single pick-image render gives true occlusion for every
        # candidate (falls back to the facing test in GPU mode).
        pick_image = self._cached_pick_image()
        for face_id in range(self.mesh_model.face_count):
            point = self._project_world_to_screen(self.mesh_model.face_center(face_id))
            if point is None:
                continue
            if not min_x <= point[0] <= max_x and min_y <= point[1] <= max_y:
                continue
            # Only visible faces: without this the marquee selects
            # backside faces through the model (transparent/X-ray effect).
            if not self._is_face_visible(face_id, point, pick_image):
                continue
            selected.add(face_id)
        self.state.selected_faces = selected
        self.state.marquee_start = None
        self.state.marquee_end = None
        self._refresh_selected_count()
        self._mark_viewport_dirty()

    def _on_mouse_move(self, _sender: int, _app_data: tuple[float, float]) -> None:
        nav_pos = self._nav_pad_mouse_position() if self._nav_pad_hovered() else None
        if nav_pos is not None:
            action, axis = self._nav_hit_test(nav_pos)
            new_hover_axis = axis if action == "axis" else None
            if new_hover_axis != self.state.nav_hover_axis:
                self.state.nav_hover_axis = new_hover_axis
                self._draw_nav_pad()
        elif self.state.nav_hover_axis is not None:
            self.state.nav_hover_axis = None
            self._draw_nav_pad()
        if self.state.nav_dragging and self.state.nav_drag_origin is not None:
            mouse_pos = self._global_mouse_position()
            dx = mouse_pos[0] - self.state.nav_drag_origin[0]
            dy = mouse_pos[1] - self.state.nav_drag_origin[1]
            if dx * dx + dy * dy >= 9.0:
                self.state.nav_drag_moved = True
            if self.state.nav_drag_moved:
                self.camera.orbit_pixels(dx, dy, 132.0, 132.0)
                new_origin = self._center_nav_cursor()
                self.state.nav_drag_origin = new_origin or self.state.nav_drag_origin
                self._mark_viewport_dirty()
            return
        if (
            not self._mouse_inside_viewport()
            or not self.state.dragging
            or self.state.drag_origin_screen is None
        ):
            if not self.state.dragging and not self.state.nav_dragging:
                self._update_hover_face(self._viewport_mouse_position())
            return
        mouse_pos = self._viewport_mouse_position()
        dx = mouse_pos[0] - self.state.drag_origin_screen[0]
        dy = mouse_pos[1] - self.state.drag_origin_screen[1]
        width, height = self._pick_viewport_size()
        if self.state.drag_button == 1:
            # VTK right-drag: dolly zoom in any mode.
            if self._engage_drag(mouse_pos):
                self.camera.dolly_pixels(dy, height)
                self.state.drag_origin_screen = mouse_pos
                self._mark_viewport_dirty()
            return
        if self.state.drag_button == 2 or (
            self.state.drag_button == 0 and self.state.shift_down
        ):
            # VTK middle-drag (or Shift+Left): grab-style pan.
            if self._engage_drag(mouse_pos):
                self.camera.pan_pixels(dx, dy, height)
                self.state.drag_origin_screen = mouse_pos
                self._mark_viewport_dirty()
            return
        if self.state.drag_button != 0:
            return
        if self.state.workspace_mode == "preview":
            # VTK left-drag: orbit (viewport-normalized).
            if self._engage_drag(mouse_pos):
                self.camera.orbit_pixels(dx, dy, width, height)
                self.state.drag_origin_screen = mouse_pos
                self._mark_viewport_dirty()
            return
        if (
            self.state.workspace_mode == "paint"
            and self.state.interaction_mode == "paint"
            and self.state.paint_tool in {"brush", "erase"}
        ):
            # Queue for the main loop (see _process_pending_paint); the
            # latest position wins so strokes stay smooth at frame rate.
            self._pending_paint_click = (mouse_pos, True)
        elif (
            self.state.workspace_mode == "paint"
            and self.state.interaction_mode == "paint"
            and self.state.paint_tool == "select"
        ):
            if self._engage_drag(mouse_pos):
                self.state.marquee_end = mouse_pos
                self._mark_viewport_dirty()

    def _engage_drag(self, mouse_pos: tuple[float, float]) -> bool:
        """3px press-drag threshold (Slate/VTK parity).

        Returns True once the pointer has moved far enough that the gesture
        counts as a drag; resets the origin on crossing so the view does not
        jump. Paint dabs happen on press, so strokes are unaffected.
        """
        if self.state.drag_engaged:
            return True
        origin = self.state.drag_origin_screen
        if origin is None:
            return False
        dx = mouse_pos[0] - origin[0]
        dy = mouse_pos[1] - origin[1]
        if dx * dx + dy * dy < 9.0:
            return False
        self.state.drag_engaged = True
        self.state.drag_origin_screen = mouse_pos
        return True

    def _on_mouse_wheel(self, _sender: int, app_data: float) -> None:
        if not self._mouse_inside_viewport():
            return
        # VTK wheel zoom: 1.1^2 per notch, scroll up = closer.
        self.camera.dolly_notches(float(app_data))
        self._mark_viewport_dirty()

    def _on_key_down(self, _sender: int, app_data: int) -> None:
        if app_data in (self._key_lctrl, self._key_rctrl):
            self.state.ctrl_down = True
        if app_data in (self._key_lshift, self._key_rshift):
            self.state.shift_down = True
        if self.state.ctrl_down and app_data == self._key_z:
            if self.state.shift_down:
                self._on_redo()
            else:
                self._on_undo()

    def _on_timeline_change(self, _sender: int, app_data: int) -> None:
        touched = self.commands.jump_to_timeline_index(int(app_data))
        self._apply_updates(touched)
        self._refresh_mask_count()

    def _on_key_release(self, _sender: int, app_data: int) -> None:
        if app_data in (self._key_lctrl, self._key_rctrl):
            self.state.ctrl_down = False
        if app_data in (self._key_lshift, self._key_rshift):
            self.state.shift_down = False

    def _on_open_selected(self, path: str) -> None:
        try:
            suffix = Path(path).suffix.lower()
            if suffix == PROJECT_EXTENSION:
                base_model, file_timeline = load_tg3d(path)
                # The stored model block carries no face colours (paint
                # lives in the timeline snapshots), so reconstruct the
                # saved current state instead of the bare base model.
                mesh_model = reconstruct_current_model(base_model, file_timeline)
                self._push_recent_project(path)
            elif suffix == ".json":
                mesh_model = load_project(path)
            else:
                mesh_model = load_model(path)
            self._load_mesh_model(mesh_model)
        except Exception as exc:
            logger.exception("Open failed")
            self._set_status(f"Open failed: {exc} | See log: {log_file_path().name}")

    def _on_apply_scale_click(
        self, _sender: int | None = None, _app_data: object | None = None
    ) -> None:
        if self.mesh_model is None:
            self._set_status("Load a model before scaling")
            return
        try:
            factor = float(dpg.get_value("model_scale_factor"))
            self.mesh_model.scale_uniform(factor)
            self._load_mesh_model(self.mesh_model)
            self._set_status(f"Scaled model by {factor:.4f}x")
        except Exception as exc:
            self._set_status(f"Scale failed: {exc}")

    def _on_import_variant_click(
        self, _sender: int | None = None, _app_data: object | None = None
    ) -> None:
        self._request_dialog(_DIALOG_VARIANT)

    def _on_import_variant_selected(self, path: str) -> None:
        if self.mesh_model is None:
            self._set_status("Load a base model before importing a variant")
            return
        try:
            variant = load_model(path)
            mapped = variant.map_colours_from(self.mesh_model, normalize_scale=True)
            self._load_mesh_model(variant)
            self._set_status(f"Imported variant and mapped paint to {mapped} faces")
        except Exception as exc:
            self._set_status(f"Import variant failed: {exc}")

    def _on_export_selected(self, path: str) -> None:
        if self.mesh_model is None:
            self._set_status("No mesh loaded")
            return
        export_model(path, self.mesh_model)
        self._set_status(f"Exported {Path(path).name}")

    def _on_save_project_selected(self, path: str) -> None:
        if self.mesh_model is None:
            self._set_status("No mesh loaded")
            return
        save_tg3d(path, self.mesh_model, timeline=self.commands.export_timeline())
        self._push_recent_project(path)
        self._set_status("Saved project .tg3d")

    def _on_open_click(
        self, _sender: int | None = None, _app_data: object | None = None
    ) -> None:
        self._request_dialog(_DIALOG_OPEN)

    def _on_export_click(
        self, _sender: int | None = None, _app_data: object | None = None
    ) -> None:
        self._request_dialog(_DIALOG_EXPORT)

    def _on_save_project_click(
        self, _sender: int | None = None, _app_data: object | None = None
    ) -> None:
        self._request_dialog(_DIALOG_SAVE)

    def _on_image_selected(self, _sender: int, app_data: dict[str, object]) -> None:
        self.state.ai_settings.image_path = str(app_data["file_path_name"])
        dpg.set_value("ai_image_path", self.state.ai_settings.image_path)
        self._save_ai_settings()

    def _on_svg_selected(self, _sender: int, app_data: dict[str, object]) -> None:
        path = str(app_data["file_path_name"])
        self._push_recent_svg(path)
        dpg.set_value("recent_svg_combo", path)
        self._set_status(f"Selected SVG {Path(path).name}")

    def _on_recent_svg_selected(self, _sender: int, app_data: str) -> None:
        if app_data:
            self._push_recent_svg(str(app_data))

    def _push_recent_svg(self, path: str) -> None:
        normalized = str(Path(path))
        self.state.recent_svgs = [
            item for item in self.state.recent_svgs if item != normalized
        ]
        self.state.recent_svgs.insert(0, normalized)
        self.state.recent_svgs = [
            item for item in self.state.recent_svgs if Path(item).exists()
        ][:10]
        dpg.configure_item("recent_svg_combo", items=self.state.recent_svgs or [""])
        self._save_svg_settings()

    def _on_bake_sketch(self) -> None:
        if self.mesh_model is None or self._active_document() is None:
            self._set_status("No sketch document to bake")
            return
        baked = self.sketch_tool.bake_document_to_faces(
            self.mesh_model, self._active_document()
        )
        touched = self.commands.paint_faces(baked, description="Bake sketch")
        self._apply_updates(touched)
        self._set_status(f"Baked {len(touched)} face colours")

    def _on_undo(
        self, _sender: int | None = None, _app_data: object | None = None
    ) -> None:
        touched = self.commands.undo()
        self._apply_updates(touched)
        self._refresh_mask_count()
        self._set_status("Undo")

    def _on_redo(
        self, _sender: int | None = None, _app_data: object | None = None
    ) -> None:
        touched = self.commands.redo()
        self._apply_updates(touched)
        self._refresh_mask_count()
        self._set_status("Redo")

    def _refresh_mask_count(self) -> None:
        if self.mesh_model is None:
            dpg.set_value("mask_count_text", "Masked faces: 0")
            return
        dpg.set_value(
            "mask_count_text", f"Masked faces: {len(self.mesh_model.masked_faces)}"
        )

    def _on_clear_mask(self) -> None:
        if self.mesh_model is None or not self.mesh_model.masked_faces:
            return
        touched = self.commands.set_masked_faces(set(), description="Clear mask")
        self._apply_updates(touched)
        self._refresh_mask_count()
        self._set_status("Cleared mask")

    def _on_invert_mask(self) -> None:
        if self.mesh_model is None:
            return
        full_set = set(range(self.mesh_model.face_count))
        updated = full_set.difference(self.mesh_model.masked_faces)
        touched = self.commands.set_masked_faces(updated, description="Invert mask")
        self._apply_updates(touched)
        self._refresh_mask_count()
        self._set_status("Inverted mask")

    def _on_group_faces(self) -> None:
        if self.mesh_model is None:
            self._set_status("No mesh loaded")
            return
        if self.mesh_model.has_cad_faces:
            self._set_status(
                "STEP model already groups per CAD face — regroup skipped"
            )
            return
        angle_tolerance = self.state.brush.angle_tolerance_degrees
        self.mesh_model.compute_face_groups(angle_tolerance_degrees=angle_tolerance)
        group_count = len(self.mesh_model.face_groups)
        dpg.set_value("face_groups_count_text", f"Groups: {group_count}")
        if self.renderer is not None:
            self.renderer.rebuild_edge_buffer()
        self._mark_viewport_dirty()
        self._set_status(f"Created {group_count} face groups")

    def _capture_viewport_to_file(self) -> Path:
        self._render_viewport()
        image = (np.clip(self._texture_data, 0.0, 1.0) * 255).astype(np.uint8)
        path = Path.cwd() / "ai_viewport_snapshot.png"
        Image.fromarray(image, mode="RGBA").save(path)
        return path

    def _refresh_ai_panels(self) -> None:
        transcript = "\n\n".join(
            f"{role.upper()}: {content}"
            for role, content in self.state.ai_messages[-12:]
        )
        dpg.set_value("ai_chat_transcript", transcript)
        tool_text = "\n".join(
            f"{entry.name} | {'OK' if entry.ok else 'ERR'} | {entry.summary}"
            for entry in self.state.tool_logs[-12:]
        )
        dpg.set_value("ai_tool_log", tool_text)

    def _on_ai_send(self) -> None:
        raw_prompt = str(dpg.get_value("ai_prompt_input")).strip()
        self.state.ai_settings.image_path = str(dpg.get_value("ai_image_path")).strip()
        prompt = build_user_prompt(raw_prompt, self.state.ai_settings.image_path)
        if not prompt:
            self._set_status("Enter a prompt or choose a reference image")
            return
        self._save_ai_settings()
        transcript_prompt = (
            raw_prompt
            if raw_prompt
            else "[Image-only request] Recreate the texture from the selected reference image."
        )
        self.state.ai_messages.append(("user", transcript_prompt))
        self._refresh_ai_panels()
        try:
            agent = AIAgent(
                AIToolContext(
                    mesh_model=self.mesh_model,
                    interaction_state=self.state,
                    commands=self.commands,
                    paint_tool=self.paint_tool,
                    sketch_tool=self.sketch_tool,
                    capture_viewport=self._capture_viewport_to_file,
                )
            )
            response = agent.run(prompt)
            self.state.ai_messages.append(("assistant", response.text))
            self.state.tool_logs.extend(response.tool_logs)
            if self.renderer is not None and self.mesh_model is not None:
                self.renderer.update_face_colours(
                    list(range(self.mesh_model.face_count))
                )
            self._mark_viewport_dirty()
            self._refresh_ai_panels()
            self._set_status("AI assistant completed")
        except Exception as exc:
            logger.exception("AI assistant failed")
            self.state.ai_messages.append(("assistant", f"Error: {exc}"))
            self._refresh_ai_panels()
            self._set_status(f"AI error: {exc}")

    def _test_native_dialog(self) -> None:
        """Exercise the Tk native file dialog path headlessly (test hook).

        Opens the real dialog (loading shell extensions like a user click),
        auto-dismisses it with ESC from a helper thread, then continues.
        """
        import threading
        import time

        def _dismiss() -> None:
            time.sleep(4.0)
            try:
                _USER32.keybd_event(0x1B, 0, 0, 0)  # VK_ESCAPE down
                _USER32.keybd_event(0x1B, 0, 2, 0)  # VK_ESCAPE up
            except Exception:
                pass

        helper = threading.Thread(target=_dismiss, daemon=True)
        helper.start()
        self._pending_dialog = _DIALOG_OPEN
        self._process_pending_dialog()
        helper.join(timeout=10.0)
        logger.info("Native dialog smoke test done")

    def run(self) -> None:
        try:
            # Headless smoke-test hooks (also useful for CI):
            #   STL_TEXTURE_PAINTER_AUTOLOAD=<path>  open a model at startup
            #   STL_TEXTURE_PAINTER_SPIN=1            orbit a little every frame
            #   STL_TEXTURE_PAINTER_MAX_FRAMES=<n>    quit after n frames
            if (
                os.environ.get("STL_TEXTURE_PAINTER_TEST_DIALOG", "").strip().lower()
                in {"1", "true", "yes", "on"}
            ):
                self._test_native_dialog()
            autoload = os.environ.get("STL_TEXTURE_PAINTER_AUTOLOAD", "").strip()
            if autoload:
                self._on_open_selected(autoload)
            spin = (
                os.environ.get("STL_TEXTURE_PAINTER_SPIN", "").strip().lower()
                in {"1", "true", "yes", "on"}
            )
            try:
                max_frames = int(
                    os.environ.get("STL_TEXTURE_PAINTER_MAX_FRAMES", "0") or 0
                )
            except ValueError:
                max_frames = 0
            frames = 0
            while dpg.is_dearpygui_running():
                # Serve deferred native dialogs here, outside Dear PyGui
                # frame rendering (see _pick_path_native).
                self._process_pending_dialog()
                self._poll_keyboard_shortcuts()
                self._sync_workspace_layout()
                if spin and self.mesh_model is not None:
                    self.camera.orbit_pixels(2.0, 1.0, 960.0, 720.0)
                    self._mark_viewport_dirty()
                self._process_pending_paint()
                self._render_viewport()
                dpg.render_dearpygui_frame()
                frames += 1
                if max_frames and frames >= max_frames:
                    logger.info("Smoke-test frame budget reached | frames=%s", frames)
                    dpg.stop_dearpygui()
        finally:
            self._release_cursor_clip()
            dpg.destroy_context()


def export_demo_mesh(path: str | Path) -> None:
    cube = MeshModel(
        vertices=np.asarray(
            [
                (-1, -1, -1),
                (1, -1, -1),
                (1, 1, -1),
                (-1, 1, -1),
                (-1, -1, 1),
                (1, -1, 1),
                (1, 1, 1),
                (-1, 1, 1),
            ],
            dtype=np.float32,
        ),
        faces=np.asarray(
            [
                (0, 1, 2),
                (0, 2, 3),
                (1, 5, 6),
                (1, 6, 2),
                (5, 4, 7),
                (5, 7, 6),
                (4, 0, 3),
                (4, 3, 7),
                (3, 2, 6),
                (3, 6, 7),
                (4, 5, 1),
                (4, 1, 0),
            ],
            dtype=np.int32,
        ),
        normals=np.zeros((12, 3), dtype=np.float32),
        default_colour=DEFAULT_COLOR,
    )
    with NamedTemporaryFile(suffix=".3mf", delete=False) as handle:
        export_model(handle.name, cube)

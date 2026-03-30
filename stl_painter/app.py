from __future__ import annotations

import logging
from pathlib import Path
from tempfile import NamedTemporaryFile

import dearpygui.dearpygui as dpg
import numpy as np
from PIL import Image

from .ai_agent import AIAgent
from .ai_tools import AIToolContext
from .camera import OrbitCamera
from .color_utils import DEFAULT_COLOR, Color, clamp_color
from .commands import AppCommands
from .config import load_local_config, save_local_config
from .exporter import export_3mf
from .importer import load_stl
from .interaction_state import InteractionState
from .logging_utils import log_file_path
from .mesh_model import MeshModel
from .paint_tool import PaintTool
from .picking import PickResult, pick_face_location_cpu
from .project_io import load_project, save_project
from .renderer import MeshRenderer, RenderSnapshot, make_grid_snapshot
from .sketch_tool import SketchTool, entity_snap_points
from .ui_panels import sync_mode_sections

logger = logging.getLogger(__name__)

PALETTE: list[Color] = [
    (220, 220, 220, 255),
    (255, 80, 80, 255),
    (255, 179, 71, 255),
    (77, 184, 72, 255),
    (65, 105, 225, 255),
    (157, 78, 221, 255),
]

ICON = {
    "open": "📂",
    "export": "📤",
    "save": "💾",
    "bake": "🔥",
    "undo": "↶",
    "redo": "↷",
    "clear": "🧹",
    "invert": "🔁",
    "paint": "🖌",
    "mask": "🎭",
    "select": "🔲",
    "image": "🖼",
    "send": "✉",
    "svg": "📐",
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
        self._texture_data = np.zeros(
            (self.state.viewport_size[1], self.state.viewport_size[0], 4),
            dtype=np.float32,
        )
        self._last_brush_face: int | None = None
        self._load_ai_settings()
        self._load_svg_settings()
        self._create_ui()

    def _load_ai_settings(self) -> None:
        payload = load_local_config()
        ai = payload.get("ai_settings", {})
        if not ai:
            return
        self.state.ai_settings.provider_base_url = str(
            ai.get("provider_base_url", self.state.ai_settings.provider_base_url)
        )
        self.state.ai_settings.api_key = str(ai.get("api_key", ""))
        self.state.ai_settings.model = str(ai.get("model", self.state.ai_settings.model))
        self.state.ai_settings.timeout = float(ai.get("timeout", self.state.ai_settings.timeout))
        self.state.ai_settings.system_prompt_version = str(
            ai.get("system_prompt_version", self.state.ai_settings.system_prompt_version)
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
        self.state.recent_svgs = [str(path) for path in payload.get("recent_svgs", []) if Path(str(path)).exists()]
        tint = payload.get("last_svg_tint", list(self.state.svg_tint))
        self.state.svg_tint = clamp_color(tuple(tint))

    def _save_svg_settings(self) -> None:
        save_local_config({"recent_svgs": self.state.recent_svgs[:10], "last_svg_tint": list(self.state.svg_tint)})

    def _create_ui(self) -> None:
        dpg.create_context()
        self._apply_light_theme()
        self._load_default_font()
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                width=self.state.viewport_size[0],
                height=self.state.viewport_size[1],
                default_value=self._texture_data,
                format=dpg.mvFormat_Float_rgba,
                tag="viewport_texture",
            )
        self._create_file_dialogs()
        with dpg.window(label="STL Texture Painter", tag="main_window"):
            with dpg.group(horizontal=True):
                dpg.add_button(label=f"{ICON['open']} Open STL / Project", callback=lambda: dpg.show_item("open_dialog"), width=180)
                dpg.add_button(label=f"{ICON['export']} Export 3MF", callback=lambda: dpg.show_item("export_dialog"), width=130)
                dpg.add_button(label=f"{ICON['save']} Save Project", callback=lambda: dpg.show_item("save_project_dialog"), width=140)
                dpg.add_button(label=f"{ICON['bake']} Bake Sketch", callback=self._on_bake_sketch, width=130)
                dpg.add_button(label=f"{ICON['undo']} Undo", callback=self._on_undo, width=90)
                dpg.add_button(label=f"{ICON['redo']} Redo", callback=self._on_redo, width=90)
                dpg.add_text("", tag="status_text")
            dpg.add_separator()
            with dpg.group(horizontal=True):
                self._build_left_sidebar()
                with dpg.child_window(autosize_x=True, autosize_y=True):
                    dpg.add_image("viewport_texture", tag="viewport_image")
                    with dpg.handler_registry():
                        dpg.add_mouse_down_handler(callback=self._on_mouse_down)
                        dpg.add_mouse_release_handler(callback=self._on_mouse_release)
                        dpg.add_mouse_move_handler(callback=self._on_mouse_move)
                        dpg.add_mouse_wheel_handler(callback=self._on_mouse_wheel)
                        dpg.add_key_down_handler(callback=self._on_key_down)
                        dpg.add_key_release_handler(callback=self._on_key_release)
                self._build_right_sidebar()
        dpg.create_viewport(title="STL Texture Painter", width=1680, height=920)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("main_window", True)
        self._set_status("Ready")
        self._sync_tool_panels()

    def _apply_light_theme(self) -> None:
        with dpg.theme(tag="light_app_theme"):
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 12, 12)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 8, 6)
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 8, 8)
                dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 8)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 6)
                dpg.add_theme_style(dpg.mvStyleVar_GrabRounding, 6)
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (246, 248, 252, 255))
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (255, 255, 255, 255))
                dpg.add_theme_color(dpg.mvThemeCol_PopupBg, (255, 255, 255, 255))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (240, 244, 250, 255))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (229, 238, 250, 255))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (217, 232, 252, 255))
                dpg.add_theme_color(dpg.mvThemeCol_Button, (64, 123, 255, 255))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (48, 107, 232, 255))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (38, 88, 204, 255))
                dpg.add_theme_color(dpg.mvThemeCol_Header, (219, 232, 255, 255))
                dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, (204, 223, 252, 255))
                dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, (186, 212, 252, 255))
                dpg.add_theme_color(dpg.mvThemeCol_Text, (31, 41, 55, 255))
                dpg.add_theme_color(dpg.mvThemeCol_Separator, (206, 216, 230, 255))
        dpg.bind_theme("light_app_theme")

    def _load_default_font(self) -> None:
        font_candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
        ]
        for font_path in font_candidates:
            if Path(font_path).exists():
                with dpg.font_registry():
                    font_id = dpg.add_font(font_path, 17)
                dpg.bind_font(font_id)
                return

    def _build_left_sidebar(self) -> None:
        with dpg.child_window(width=300, autosize_y=True):
            dpg.add_text("Workspace", color=(33, 55, 102))
            dpg.add_combo(
                items=["paint", "sketch"],
                default_value=self.state.interaction_mode,
                label="Mode",
                callback=self._set_interaction_mode,
                tag="interaction_mode_combo",
            )
            dpg.add_separator()
            dpg.add_text("Colour", color=(33, 55, 102))
            dpg.add_text("No mesh loaded", tag="mesh_info_text", wrap=260)
            with dpg.group(horizontal=True):
                for index, colour in enumerate(PALETTE):
                    dpg.add_color_button(
                        default_value=list(colour),
                        width=40,
                        height=30,
                        callback=lambda _s, _a, user_data=index: self._select_palette_colour(user_data),
                    )
            dpg.add_color_picker(
                label="Active Color",
                default_value=list(self.state.active_colour),
                callback=self._on_custom_colour,
                tag="active_colour_picker",
                alpha_bar=True,
            )
            with dpg.group(tag="paint_section"):
                dpg.add_separator()
                dpg.add_text("Paint Tools", color=(33, 55, 102))
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
                    callback=lambda _s, value: setattr(self.state, "paint_linked_faces", bool(value)),
                )
                dpg.add_slider_float(
                    label="Radius",
                    min_value=0.01,
                    max_value=0.6,
                    default_value=self.state.brush.radius,
                    callback=lambda _s, value: setattr(self.state.brush, "radius", float(value)),
                )
                dpg.add_slider_float(
                    label="Opacity",
                    min_value=0.05,
                    max_value=1.0,
                    default_value=self.state.brush.opacity,
                    callback=lambda _s, value: setattr(self.state.brush, "opacity", float(value)),
                )
                dpg.add_combo(
                    items=["constant", "linear", "smooth"],
                    default_value=self.state.brush.falloff,
                    label="Falloff",
                    callback=lambda _s, value: setattr(self.state.brush, "falloff", value),
                )
                dpg.add_checkbox(
                    label="Front Faces Only",
                    default_value=self.state.brush.front_faces_only,
                    callback=lambda _s, value: setattr(self.state.brush, "front_faces_only", bool(value)),
                )
                dpg.add_slider_float(
                    label="Angle Tolerance",
                    min_value=5.0,
                    max_value=180.0,
                    default_value=self.state.brush.angle_tolerance_degrees,
                    callback=lambda _s, value: setattr(self.state.brush, "angle_tolerance_degrees", float(value)),
                )
                dpg.add_button(label=f"{ICON['clear']} Clear Mask", callback=self._on_clear_mask, width=-1)
                dpg.add_button(label=f"{ICON['invert']} Invert Mask", callback=self._on_invert_mask, width=-1)
                dpg.add_button(label=f"{ICON['paint']} Paint Selected", callback=self._paint_selected_faces, width=-1)
                dpg.add_button(label=f"{ICON['mask']} Mask Selected", callback=self._mask_selected_faces, width=-1)
                dpg.add_button(label=f"{ICON['select']} Clear Selection", callback=self._clear_face_selection, width=-1)
                dpg.add_text("Masked faces: 0", tag="mask_count_text")
                dpg.add_text("Selected faces: 0", tag="selected_count_text")
            with dpg.group(tag="sketch_section", show=False):
                dpg.add_separator()
                dpg.add_text("Sketch Tools", color=(33, 55, 102))
                dpg.add_combo(
                    items=["select", "line", "rect", "circle", "text", "svg"],
                    default_value=self.state.sketch_tool,
                    label="Tool",
                    callback=self._set_sketch_tool,
                    tag="sketch_tool_combo",
                )
                dpg.add_button(label=f"{ICON['svg']} Import SVG", callback=lambda: dpg.show_item("svg_dialog"), width=-1)
                dpg.add_combo(items=self.state.recent_svgs or [""], label="Recent SVGs", tag="recent_svg_combo", callback=self._on_recent_svg_selected)
                dpg.add_input_text(label="Text", default_value="Text", tag="sketch_text_value")
                dpg.add_slider_int(label="Text Size", min_value=8, max_value=120, default_value=28, tag="sketch_text_size")
                dpg.add_slider_float(
                    label="Stroke Width",
                    min_value=0.001,
                    max_value=0.2,
                    default_value=0.02,
                    tag="sketch_stroke_width",
                )
                dpg.add_slider_float(label="Grid", min_value=0.01, max_value=1.0, default_value=0.1, callback=self._set_sketch_grid)
                dpg.add_checkbox(label="Snap To Grid", default_value=True, callback=self._set_snap_flag, user_data="grid")
                dpg.add_checkbox(label="Snap To Vertices", default_value=True, callback=self._set_snap_flag, user_data="vertices")
                dpg.add_checkbox(label="Snap To Edges", default_value=True, callback=self._set_snap_flag, user_data="edges")
                dpg.add_checkbox(label="Snap To Entities", default_value=True, callback=self._set_snap_flag, user_data="entities")
                dpg.add_text("Click a face first to start a sketch plane.", wrap=260)

    def _build_right_sidebar(self) -> None:
        with dpg.child_window(width=360, autosize_y=True):
            dpg.add_text("AI Assistant", color=(33, 55, 102))
            dpg.add_input_text(
                label="Base URL",
                default_value=self.state.ai_settings.provider_base_url,
                callback=lambda _s, value: setattr(self.state.ai_settings, "provider_base_url", value),
            )
            dpg.add_input_text(
                label="API Key",
                password=True,
                default_value=self.state.ai_settings.api_key,
                callback=lambda _s, value: setattr(self.state.ai_settings, "api_key", value),
            )
            dpg.add_input_text(
                label="Model",
                default_value=self.state.ai_settings.model,
                callback=lambda _s, value: setattr(self.state.ai_settings, "model", value),
            )
            dpg.add_input_text(label="Image", default_value=self.state.ai_settings.image_path, tag="ai_image_path")
            dpg.add_button(label=f"{ICON['image']} Choose Image", callback=lambda: dpg.show_item("image_dialog"), width=-1)
            dpg.add_input_text(multiline=True, height=120, hint="Describe the paint or sketch you want...", tag="ai_prompt_input")
            dpg.add_button(label=f"{ICON['send']} Send To Assistant", callback=self._on_ai_send, width=-1)
            dpg.add_input_text(multiline=True, readonly=True, height=220, tag="ai_chat_transcript")
            dpg.add_input_text(multiline=True, readonly=True, height=180, tag="ai_tool_log")

    def _create_file_dialogs(self) -> None:
        with dpg.file_dialog(directory_selector=False, show=False, callback=self._on_open_selected, tag="open_dialog", width=700, height=400):
            dpg.add_file_extension(".stl")
            dpg.add_file_extension(".json")
        with dpg.file_dialog(directory_selector=False, show=False, callback=self._on_export_selected, tag="export_dialog", width=700, height=400):
            dpg.add_file_extension(".3mf")
        with dpg.file_dialog(directory_selector=False, show=False, callback=self._on_save_project_selected, tag="save_project_dialog", width=700, height=400):
            dpg.add_file_extension(".json")
        with dpg.file_dialog(directory_selector=False, show=False, callback=self._on_image_selected, tag="image_dialog", width=700, height=400):
            dpg.add_file_extension(".png")
            dpg.add_file_extension(".jpg")
            dpg.add_file_extension(".jpeg")
        with dpg.file_dialog(directory_selector=False, show=False, callback=self._on_svg_selected, tag="svg_dialog", width=700, height=400):
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

    def _select_palette_colour(self, index: int) -> None:
        self.state.active_colour = PALETTE[index]
        dpg.set_value("active_colour_picker", list(self.state.active_colour))

    def _on_custom_colour(self, _sender: int, app_data: list[float]) -> None:
        self.state.active_colour = _normalise_picker_colour(app_data)

    def _clear_face_selection(self) -> None:
        self.state.selected_faces.clear()
        self._refresh_selected_count()
        self._mark_viewport_dirty()

    def _refresh_selected_count(self) -> None:
        dpg.set_value("selected_count_text", f"Selected faces: {len(self.state.selected_faces)}")

    def _mask_selected_faces(self) -> None:
        if self.mesh_model is None or not self.state.selected_faces:
            return
        masked = set(self.mesh_model.masked_faces).union(self.state.selected_faces)
        touched = self.commands.set_masked_faces(masked, description="Mask selected faces")
        self._apply_updates(touched)
        self._refresh_mask_count()

    def _paint_selected_faces(self) -> None:
        if not self.state.selected_faces:
            return
        updates = {face_id: self.state.active_colour for face_id in self.state.selected_faces}
        touched = self.commands.paint_faces(updates, description="Paint selected faces")
        self._apply_updates(touched)

    def _active_document(self):
        if self.mesh_model is None or not self.mesh_model.sketch_documents:
            return None
        return self.mesh_model.sketch_documents[-1]

    def _load_mesh_model(self, mesh_model: MeshModel) -> None:
        self.mesh_model = mesh_model
        if not mesh_model.face_groups:
            mesh_model.compute_face_groups()
        self.camera = OrbitCamera.for_mesh(mesh_model.vertices)
        self.renderer = MeshRenderer(None, mesh_model, self.state.viewport_size, prefer_gpu=False)
        self.paint_tool = PaintTool(mesh_model)
        self.commands.attach(mesh_model, self.paint_tool)
        self._mark_viewport_dirty()
        dpg.set_value(
            "mesh_info_text",
            f"{Path(mesh_model.source_path or 'project').name}\nFaces: {mesh_model.face_count}\nVertices: {mesh_model.vertex_count}\nMasked: {len(mesh_model.masked_faces)}",
        )
        self._refresh_mask_count()
        self._refresh_selected_count()
        self._set_status(f"Loaded {Path(mesh_model.source_path or 'project').name}")

    def _render_snapshot(self) -> RenderSnapshot:
        if self.mesh_model is None or self.renderer is None:
            return make_grid_snapshot(self.state.viewport_size)
        return self.renderer.render(self.camera, show_triangle_edges=True)

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

    def _draw_line(self, rgba: np.ndarray, p0: tuple[int, int], p1: tuple[int, int], colour: tuple[float, float, float], alpha: float = 1.0) -> None:
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
        rgba[ys, xs, :3] = rgba[ys, xs, :3] * (1.0 - alpha) + np.array(colour, dtype=np.float32) * alpha
        rgba[ys, xs, 3] = 1.0

    def _draw_point(self, rgba: np.ndarray, point: tuple[int, int], colour: tuple[float, float, float]) -> None:
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
        if self.state.dragging and self.state.interaction_mode == "sketch" and self.state.sketch_tool in {"rect", "line", "circle"} and self.state.drag_origin_plane is not None:
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
        for entity in document.entities + ([preview_entity] if preview_entity is not None else []):
            colour = np.array(entity.data.get("colour", [255, 80, 80, 255]), dtype=np.float32)[:3] / 255.0
            alpha = 0.45 if preview_entity is not None and entity.entity_id == preview_entity.entity_id else 1.0
            if entity.kind == "rect":
                min_uv = np.asarray(entity.data["min"], dtype=np.float32)
                max_uv = np.asarray(entity.data["max"], dtype=np.float32)
                corners = [
                    min_uv,
                    np.asarray([max_uv[0], min_uv[1]], dtype=np.float32),
                    max_uv,
                    np.asarray([min_uv[0], max_uv[1]], dtype=np.float32),
                ]
                screens = [self._project_world_to_screen(self.sketch_tool.plane_to_world(document.plane, uv)) for uv in corners]
                if all(screen is not None for screen in screens):
                    for index in range(4):
                        self._draw_line(rgba, screens[index], screens[(index + 1) % 4], tuple(colour), alpha=alpha)  # type: ignore[arg-type]
            elif entity.kind == "line":
                start = self.sketch_tool.plane_to_world(document.plane, np.asarray(entity.data["start"], dtype=np.float32))
                end = self.sketch_tool.plane_to_world(document.plane, np.asarray(entity.data["end"], dtype=np.float32))
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
                        center + np.asarray([np.cos(angle) * radius, np.sin(angle) * radius], dtype=np.float32),
                    )
                    for angle in np.linspace(0.0, np.pi * 2.0, 24, endpoint=False)
                ]
                screens = [self._project_world_to_screen(point) for point in ring]
                valid = [screen for screen in screens if screen is not None]
                if len(valid) >= 2:
                    for index in range(len(valid)):
                        self._draw_line(rgba, valid[index], valid[(index + 1) % len(valid)], tuple(colour), alpha=alpha)
            elif entity.kind == "text":
                point = self.sketch_tool.plane_to_world(document.plane, np.asarray(entity.data["position"], dtype=np.float32))
                screen = self._project_world_to_screen(point)
                if screen:
                    self._draw_point(rgba, screen, tuple(colour))
            if entity.entity_id == document.selected_entity_id:
                for snap_uv in entity_snap_points(entity):
                    point = self._project_world_to_screen(self.sketch_tool.plane_to_world(document.plane, snap_uv))
                    if point:
                        self._draw_point(rgba, point, (1.0, 1.0, 1.0))

    def _overlay_masked_faces(self, rgba: np.ndarray) -> None:
        if self.mesh_model is None:
            return
        for face_id in list(self.mesh_model.masked_faces)[:250]:
            point = self._project_world_to_screen(self.mesh_model.face_center(face_id))
            if point:
                self._draw_point(rgba, point, (0.0, 0.0, 0.0))

    def _overlay_selected_faces(self, rgba: np.ndarray) -> None:
        if self.mesh_model is None:
            return
        for face_id in list(self.state.selected_faces)[:500]:
            point = self._project_world_to_screen(self.mesh_model.face_center(face_id))
            if point:
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
        if self.mesh_model is not None:
            self._overlay_sketch_entities(rgba)
            self._overlay_masked_faces(rgba)
            self._overlay_selected_faces(rgba)
        self._texture_data[:, :, :] = rgba
        dpg.set_value("viewport_texture", self._texture_data)
        self.state.viewport_dirty = False

    def _mouse_inside_viewport(self) -> bool:
        mouse_x, mouse_y = dpg.get_mouse_pos(local=False)
        image_x, image_y = dpg.get_item_rect_min("viewport_image")
        width, height = dpg.get_item_rect_size("viewport_image")
        return image_x <= mouse_x < image_x + width and image_y <= mouse_y < image_y + height

    def _viewport_mouse_position(self) -> tuple[float, float]:
        mouse_x, mouse_y = dpg.get_mouse_pos(local=False)
        image_x, image_y = dpg.get_item_rect_min("viewport_image")
        return mouse_x - image_x, mouse_y - image_y

    def _pick_result(self, mouse_pos: tuple[float, float]) -> PickResult | None:
        if self.mesh_model is None:
            return None
        return pick_face_location_cpu(self.mesh_model, self.camera, mouse_pos[0], mouse_pos[1], self.state.viewport_size)

    def _apply_updates(self, touched: list[int]) -> None:
        if self.renderer is not None and touched:
            self.renderer.update_face_colours(touched)
        self._mark_viewport_dirty()

    def _apply_paint_at_pick(self, pick: PickResult) -> None:
        if self.mesh_model is None or self.paint_tool is None:
            return
        tool = self.state.paint_tool
        if tool == "sample":
            self.state.active_colour = self.mesh_model.face_colour(pick.face_id)
            dpg.set_value("active_colour_picker", list(self.state.active_colour))
            self._set_status(f"Sampled face {pick.face_id}")
            return
        if tool == "mask":
            updated = set(self.mesh_model.masked_faces)
            if pick.face_id in updated:
                updated.remove(pick.face_id)
            else:
                updated.add(pick.face_id)
            touched = self.commands.set_masked_faces(updated, description="Toggle face mask")
            self._apply_updates(touched)
            self._refresh_mask_count()
            return
        if tool == "fill":
            updates = self.paint_tool.flood_fill_updates(pick.face_id, self.state.active_colour)
        else:
            radius = max(0.0001, self.state.brush.radius * max(1.0, self.mesh_model.mesh_diagonal()))
            updates = self.paint_tool.brush_updates(
                pick.face_id,
                pick.location,
                self.state.active_colour,
                radius=radius,
                opacity=self.state.brush.opacity,
                falloff=self.state.brush.falloff,
                front_faces_only=self.state.brush.front_faces_only,
                angle_tolerance_degrees=self.state.brush.angle_tolerance_degrees,
                erase=(tool == "erase"),
            )
        if self.state.paint_linked_faces and self.mesh_model is not None and updates:
            group_id = self.mesh_model.group_for_face(pick.face_id)
            if group_id is not None:
                grouped_faces = self.mesh_model.faces_for_group(group_id)
                group_colour = self.state.active_colour if tool != "erase" else self.mesh_model.default_colour
                for face_id in grouped_faces:
                    if face_id not in self.mesh_model.masked_faces:
                        updates[face_id] = group_colour
        touched = self.commands.paint_faces(updates, description=f"{tool.title()} stroke")
        self._apply_updates(touched)

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
        hit = self.sketch_tool.ray_to_plane(document.plane, self.camera, mouse_pos, self.state.viewport_size)
        if hit is None:
            return None
        shift_lock = self.state.drag_origin_plane if self.state.shift_down else None
        return self.sketch_tool.snap_point(self.mesh_model, document, hit.plane_uv, shift_lock_axis=shift_lock).plane_uv

    def _find_entity_handle(self, mouse_pos: tuple[float, float]) -> tuple[str, str] | None:
        document = self._active_document()
        if document is None:
            return None
        best: tuple[str, str] | None = None
        best_distance = 18.0
        handle_names = {"rect": ["min", "max", "center"], "line": ["start", "end"], "circle": ["radius"], "text": ["position"], "svg": ["min", "max", "center"]}
        for entity in document.entities:
            names = handle_names.get(entity.kind, ["position"])
            for index, point_uv in enumerate(entity_snap_points(entity)[: len(names)]):
                screen = self._project_world_to_screen(self.sketch_tool.plane_to_world(document.plane, point_uv))
                if screen is None:
                    continue
                distance = float(np.linalg.norm(np.asarray(screen, dtype=np.float32) - np.asarray(mouse_pos, dtype=np.float32)))
                if distance < best_distance:
                    best = (entity.entity_id, names[index] if index < len(names) else "position")
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
        if not self._mouse_inside_viewport():
            return
        button = app_data[0] if isinstance(app_data, (tuple, list)) else int(app_data)
        mouse_pos = self._viewport_mouse_position()
        self.state.dragging = True
        self.state.drag_button = button
        self.state.drag_origin_screen = mouse_pos
        if button != 0:
            return
        if self.state.interaction_mode == "paint":
            if self.state.paint_tool == "select":
                self.state.marquee_start = mouse_pos
                self.state.marquee_end = mouse_pos
                self._mark_viewport_dirty()
            else:
                pick = self._pick_result(mouse_pos)
                if pick is not None:
                    self._apply_paint_at_pick(pick)
                    self._last_brush_face = pick.face_id
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
            entity = self.sketch_tool.create_entity("svg", start_uv, start_uv + np.asarray([0.2, 0.2], dtype=np.float32), self.state.svg_tint)
            entity.data["path"] = path
            entity.data["tint"] = list(self.state.svg_tint)
            self.commands.add_sketch_entity(self._active_document(), entity)
            self._mark_viewport_dirty()

    def _on_mouse_release(self, _sender: int, app_data: tuple[int, float] | int) -> None:
        if not self.state.dragging:
            return
        button = app_data[0] if isinstance(app_data, (tuple, list)) else int(app_data)
        mouse_pos = self._viewport_mouse_position()
        if button == 0 and self.state.interaction_mode == "sketch" and self._active_document() is not None:
            document = self._active_document()
            if self.state.sketch_tool in {"rect", "line", "circle"} and self.state.drag_origin_plane is not None:
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
            elif self.state.sketch_tool == "select" and self.state.selected_entity_id is not None and self.state.active_handle is not None:
                entity = next((item for item in document.entities if item.entity_id == self.state.selected_entity_id), None)
                end_uv = self._screen_hit_on_plane(mouse_pos)
                if entity is not None and end_uv is not None:
                    new_data = self.sketch_tool.resize_entity(entity, self.state.active_handle, end_uv)
                    self.commands.update_sketch_entity(document, entity.entity_id, new_data)
                    self._mark_viewport_dirty()
        if button == 0 and self.state.interaction_mode == "paint" and self.state.paint_tool == "select":
            self._finish_marquee_selection()
        self.state.dragging = False
        self.state.drag_button = None
        self.state.drag_origin_screen = None
        self.state.drag_origin_plane = None
        self.state.active_handle = None

    def _finish_marquee_selection(self) -> None:
        if self.mesh_model is None or self.state.marquee_start is None or self.state.marquee_end is None:
            return
        min_x = min(self.state.marquee_start[0], self.state.marquee_end[0])
        max_x = max(self.state.marquee_start[0], self.state.marquee_end[0])
        min_y = min(self.state.marquee_start[1], self.state.marquee_end[1])
        max_y = max(self.state.marquee_start[1], self.state.marquee_end[1])
        selected: set[int] = set()
        for face_id in range(self.mesh_model.face_count):
            point = self._project_world_to_screen(self.mesh_model.face_center(face_id))
            if point is None:
                continue
            if min_x <= point[0] <= max_x and min_y <= point[1] <= max_y:
                selected.add(face_id)
        self.state.selected_faces = selected
        self.state.marquee_start = None
        self.state.marquee_end = None
        self._refresh_selected_count()
        self._mark_viewport_dirty()

    def _on_mouse_move(self, _sender: int, _app_data: tuple[float, float]) -> None:
        if not self._mouse_inside_viewport() or not self.state.dragging or self.state.drag_origin_screen is None:
            return
        mouse_pos = self._viewport_mouse_position()
        dx = mouse_pos[0] - self.state.drag_origin_screen[0]
        dy = mouse_pos[1] - self.state.drag_origin_screen[1]
        if self.state.drag_button == 1:
            self.camera.orbit(dx * 0.4, -dy * 0.4)
            self.state.drag_origin_screen = mouse_pos
            self._mark_viewport_dirty()
            return
        if self.state.drag_button == 2:
            self.camera.pan(dx, dy)
            self.state.drag_origin_screen = mouse_pos
            self._mark_viewport_dirty()
            return
        if self.state.drag_button == 0 and self.state.interaction_mode == "paint" and self.state.paint_tool in {"brush", "erase"}:
            pick = self._pick_result(mouse_pos)
            if pick is not None and pick.face_id != self._last_brush_face:
                self._apply_paint_at_pick(pick)
                self._last_brush_face = pick.face_id
        elif self.state.drag_button == 0 and self.state.interaction_mode == "paint" and self.state.paint_tool == "select":
            self.state.marquee_end = mouse_pos
            self._mark_viewport_dirty()

    def _on_mouse_wheel(self, _sender: int, app_data: float) -> None:
        if not self._mouse_inside_viewport():
            return
        self.camera.zoom(app_data * 0.1)
        self._mark_viewport_dirty()

    def _on_key_down(self, _sender: int, app_data: int) -> None:
        if app_data in (527, 531):
            self.state.ctrl_down = True
        if app_data in (528, 532):
            self.state.shift_down = True
        if self.state.ctrl_down and app_data == 571:
            self._on_undo()

    def _on_key_release(self, _sender: int, app_data: int) -> None:
        if app_data in (527, 531):
            self.state.ctrl_down = False
        if app_data in (528, 532):
            self.state.shift_down = False

    def _on_open_selected(self, _sender: int, app_data: dict[str, object]) -> None:
        try:
            path = str(app_data["file_path_name"])
            mesh_model = load_project(path) if Path(path).suffix.lower() == ".json" else load_stl(path)
            self._load_mesh_model(mesh_model)
        except Exception as exc:
            logger.exception("Open failed")
            self._set_status(f"Open failed: {exc} | See log: {log_file_path().name}")

    def _on_export_selected(self, _sender: int, app_data: dict[str, object]) -> None:
        if self.mesh_model is None:
            self._set_status("No mesh loaded")
            return
        path = str(app_data["file_path_name"])
        export_3mf(path, self.mesh_model)
        self._set_status(f"Exported {Path(path).name}")

    def _on_save_project_selected(self, _sender: int, app_data: dict[str, object]) -> None:
        if self.mesh_model is None:
            self._set_status("No mesh loaded")
            return
        save_project(str(app_data["file_path_name"]), self.mesh_model)
        self._set_status("Saved project JSON")

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
        self.state.recent_svgs = [item for item in self.state.recent_svgs if item != normalized]
        self.state.recent_svgs.insert(0, normalized)
        self.state.recent_svgs = [item for item in self.state.recent_svgs if Path(item).exists()][:10]
        dpg.configure_item("recent_svg_combo", items=self.state.recent_svgs or [""])
        self._save_svg_settings()

    def _on_bake_sketch(self) -> None:
        if self.mesh_model is None or self._active_document() is None:
            self._set_status("No sketch document to bake")
            return
        baked = self.sketch_tool.bake_document_to_faces(self.mesh_model, self._active_document())
        touched = self.commands.paint_faces(baked, description="Bake sketch")
        self._apply_updates(touched)
        self._set_status(f"Baked {len(touched)} face colours")

    def _on_undo(self, _sender: int | None = None, _app_data: object | None = None) -> None:
        touched = self.commands.undo()
        self._apply_updates(touched)
        self._refresh_mask_count()
        self._set_status("Undo")

    def _on_redo(self, _sender: int | None = None, _app_data: object | None = None) -> None:
        touched = self.commands.redo()
        self._apply_updates(touched)
        self._refresh_mask_count()
        self._set_status("Redo")

    def _refresh_mask_count(self) -> None:
        if self.mesh_model is None:
            dpg.set_value("mask_count_text", "Masked faces: 0")
            return
        dpg.set_value("mask_count_text", f"Masked faces: {len(self.mesh_model.masked_faces)}")

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

    def _capture_viewport_to_file(self) -> Path:
        self._render_viewport()
        image = (np.clip(self._texture_data, 0.0, 1.0) * 255).astype(np.uint8)
        path = Path.cwd() / "ai_viewport_snapshot.png"
        Image.fromarray(image, mode="RGBA").save(path)
        return path

    def _refresh_ai_panels(self) -> None:
        transcript = "\n\n".join(f"{role.upper()}: {content}" for role, content in self.state.ai_messages[-12:])
        dpg.set_value("ai_chat_transcript", transcript)
        tool_text = "\n".join(
            f"{entry.name} | {'OK' if entry.ok else 'ERR'} | {entry.summary}"
            for entry in self.state.tool_logs[-12:]
        )
        dpg.set_value("ai_tool_log", tool_text)

    def _on_ai_send(self) -> None:
        prompt = str(dpg.get_value("ai_prompt_input")).strip()
        if not prompt:
            self._set_status("Enter a prompt for the assistant")
            return
        self.state.ai_settings.image_path = str(dpg.get_value("ai_image_path")).strip()
        self._save_ai_settings()
        self.state.ai_messages.append(("user", prompt))
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
                self.renderer.update_face_colours(list(range(self.mesh_model.face_count)))
            self._mark_viewport_dirty()
            self._refresh_ai_panels()
            self._set_status("AI assistant completed")
        except Exception as exc:
            logger.exception("AI assistant failed")
            self.state.ai_messages.append(("assistant", f"Error: {exc}"))
            self._refresh_ai_panels()
            self._set_status(f"AI error: {exc}")

    def run(self) -> None:
        try:
            while dpg.is_dearpygui_running():
                self._render_viewport()
                dpg.render_dearpygui_frame()
        finally:
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
        export_3mf(handle.name, cube)

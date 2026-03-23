from __future__ import annotations

import logging
from math import ceil
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

import dearpygui.dearpygui as dpg
import moderngl
import numpy as np

from .camera import OrbitCamera
from .color_utils import DEFAULT_COLOR, Color, clamp_color
from .exporter import export_3mf
from .importer import load_stl
from .mesh_model import MeshModel
from .paint_tool import PaintTool
from .picking import pick_face_cpu
from .project_io import load_project, save_project
from .renderer import MeshRenderer, RenderSnapshot, make_grid_snapshot
from .sketch_tool import SketchTool, bake_sketch_to_faces

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

ToolMode = Literal["paint", "fill", "sketch"]
SketchPrimitive = Literal["text", "rect", "line", "freehand"]

# Light and dark theme colors
LIGHT_BG_COLOR = (0.95, 0.95, 0.95, 1.0)  # Light gray
DARK_BG_COLOR = (0.08, 0.1, 0.12, 1.0)  # Original dark blue-gray
CURRENT_BG_COLOR = LIGHT_BG_COLOR  # Default to light mode

PALETTE: list[Color] = [
    (220, 220, 220, 255),
    (255, 80, 80, 255),
    (255, 179, 71, 255),
    (77, 184, 72, 255),
    (65, 105, 225, 255),
    (157, 78, 221, 255),
]


@dataclass(slots=True)
class AppState:
    mesh_model: MeshModel | None = None
    camera: OrbitCamera | None = None
    renderer: MeshRenderer | None = None
    paint_tool: PaintTool | None = None
    sketch_tool: SketchTool | None = None
    tool_mode: ToolMode = "paint"
    sketch_primitive: SketchPrimitive = "text"
    active_colour: Color = PALETTE[1]
    hovered_face: int | None = None
    viewport_size: tuple[int, int] = (960, 720)
    shift_down: bool = False
    ctrl_down: bool = False
    dragging: bool = False
    drag_button: int | None = None
    drag_origin: tuple[float, float] | None = None
    freehand_points: list[tuple[float, float]] | None = None


class TexturePainterApp:
    def __init__(self) -> None:
        self.state = AppState(sketch_tool=SketchTool((960, 720)))
        self.state.camera = OrbitCamera(np.array([0.0, 0.0, 0.0], dtype=np.float32), 5.0)
        self._ctx: moderngl.Context | None = None
        self._texture_data = np.zeros(
            (self.state.viewport_size[1], self.state.viewport_size[0], 4),
            dtype=np.float32,
        )
        self._create_ui()

    def _create_context(self) -> moderngl.Context:
        if self._ctx is None:
            last_error: Exception | None = None
            # Try explicit Windows backend first; fall back to the default backend.
            for backend in ("wgl", None):
                try:
                    if backend is None:
                        self._ctx = moderngl.create_standalone_context()
                    else:
                        self._ctx = moderngl.create_standalone_context(backend=backend)
                    logger.info("Created OpenGL context with backend=%s", backend or "default")
                    break
                except Exception as exc:
                    last_error = exc
                    logger.warning(
                        "Failed to create OpenGL context with backend=%s: %s",
                        backend or "default",
                        exc,
                    )
            if self._ctx is None and last_error is not None:
                raise RuntimeError(f"Unable to create OpenGL context: {last_error}")
        return self._ctx

    def _create_ui(self) -> None:
        dpg.create_context()
        with dpg.texture_registry(show=False):
            dpg.add_dynamic_texture(
                width=self.state.viewport_size[0],
                height=self.state.viewport_size[1],
                default_value=self._texture_data.flatten().tolist(),
                tag="viewport_texture",
            )

        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            callback=self._on_open_selected,
            tag="open_dialog",
            width=700,
            height=400,
        ):
            dpg.add_file_extension(".stl", color=(150, 200, 255, 255))
            dpg.add_file_extension(".json", color=(150, 255, 170, 255))

        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            callback=self._on_export_selected,
            tag="export_dialog",
            width=700,
            height=400,
        ):
            dpg.add_file_extension(".3mf", color=(255, 210, 150, 255))

        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            callback=self._on_save_project_selected,
            tag="save_project_dialog",
            width=700,
            height=400,
        ):
            dpg.add_file_extension(".json", color=(150, 255, 170, 255))

        with dpg.window(label="STL Texture Painter", tag="main_window"):
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Open STL / Project",
                    callback=lambda: dpg.show_item("open_dialog"),
                )
                dpg.add_button(
                    label="Export 3MF", callback=lambda: dpg.show_item("export_dialog")
                )
                dpg.add_button(
                    label="Save Project",
                    callback=lambda: dpg.show_item("save_project_dialog"),
                )
                dpg.add_button(label="Bake Sketch", callback=self._on_bake_sketch)
                dpg.add_button(label="Undo", callback=self._on_undo)
                dpg.add_text("", tag="status_text")
            with dpg.group(horizontal=True):
                with dpg.child_window(width=280, autosize_y=True):
                    dpg.add_text("Tools")
                    dpg.add_radio_button(
                        items=["paint", "fill", "sketch"],
                        default_value="paint",
                        callback=self._set_tool_mode,
                    )
                    dpg.add_separator()
                    dpg.add_text("Colour")
                    for index, colour in enumerate(PALETTE):
                        dpg.add_color_button(
                            default_value=[int(c / 255.0 * 255) for c in colour],
                            width=40,
                            height=40,
                            callback=lambda sender, app_data, user_data=index: (
                                self._select_palette_colour(user_data)
                            ),
                        )
                    dpg.add_color_picker(
                        default_value=[
                            int(c / 255.0 * 255) for c in self.state.active_colour
                        ],
                        alpha_bar=True,
                        display_rgb=True,
                        callback=self._on_custom_colour,
                        tag="active_colour_picker",
                    )
                    dpg.add_separator()
                    dpg.add_text("Sketch")
                    dpg.add_radio_button(
                        items=["text", "rect", "line", "freehand"],
                        default_value="text",
                        callback=self._set_sketch_primitive,
                    )
                    dpg.add_input_text(
                        label="Text", default_value="Sample", tag="sketch_text_value"
                    )
                    dpg.add_input_int(
                        label="Text Size",
                        default_value=28,
                        min_value=8,
                        min_clamped=True,
                        tag="sketch_text_size",
                    )
                    dpg.add_input_int(
                        label="Stroke Width",
                        default_value=4,
                        min_value=1,
                        min_clamped=True,
                        tag="sketch_stroke_width",
                    )
                    dpg.add_text("Sketch usage:")
                    dpg.add_text("Text: click once")
                    dpg.add_text("Rect/Line: drag")
                    dpg.add_text("Freehand: drag")
                with dpg.child_window(
                    tag="viewport_panel", autosize_x=True, autosize_y=True
                ):
                    dpg.add_image("viewport_texture", tag="viewport_image")
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
                        with dpg.group(horizontal=True):
                            dpg.add_button(label="L", width=34, callback=lambda: self._orbit_step(-12.0, 0.0))
                            dpg.add_button(label="R", width=34, callback=lambda: self._orbit_step(12.0, 0.0))
                        with dpg.group(horizontal=True):
                            dpg.add_button(label="U", width=34, callback=lambda: self._orbit_step(0.0, -10.0))
                            dpg.add_button(label="D", width=34, callback=lambda: self._orbit_step(0.0, 10.0))
                        dpg.add_button(label="Home", width=72, callback=self._reset_view)
                    with dpg.handler_registry():
                        dpg.add_mouse_down_handler(callback=self._on_mouse_down)
                        dpg.add_mouse_release_handler(callback=self._on_mouse_release)
                        dpg.add_mouse_move_handler(callback=self._on_mouse_move)
                        dpg.add_mouse_wheel_handler(callback=self._on_mouse_wheel)
                        dpg.add_key_down_handler(callback=self._on_key_down)
                        dpg.add_key_release_handler(callback=self._on_key_release)

        dpg.create_viewport(title="STL Texture Painter", width=1440, height=920)
        dpg.setup_dearpygui()
        # Set viewport clear color (light gray background for light mode)
        dpg.set_viewport_clear_color([242, 242, 242, 255])
        dpg.show_viewport()
        dpg.set_primary_window("main_window", True)

    def _select_palette_colour(self, index: int) -> None:
        self.state.active_colour = PALETTE[index]
        dpg.set_value(
            "active_colour_picker",
            [int(c / 255.0 * 255) for c in self.state.active_colour],
        )

    def _on_custom_colour(self, sender: int, app_data: list[float]) -> None:
        self.state.active_colour = clamp_color(
            int(round(component * 255)) for component in app_data
        )

    def _set_tool_mode(self, sender: int, app_data: str) -> None:
        self.state.tool_mode = app_data  # type: ignore[assignment]
        self._set_status(f"Mode: {app_data}")

    def _set_sketch_primitive(self, sender: int, app_data: str) -> None:
        self.state.sketch_primitive = app_data  # type: ignore[assignment]

    def _set_status(self, text: str) -> None:
        dpg.set_value("status_text", text)

    def _reset_view(self) -> None:
        if self.state.mesh_model is not None:
            self.state.camera = OrbitCamera.for_mesh(self.state.mesh_model.vertices)
        else:
            self.state.camera = OrbitCamera(np.array([0.0, 0.0, 0.0], dtype=np.float32), 5.0)
        self._render_viewport()

    def _orbit_step(self, delta_azimuth: float, delta_elevation: float) -> None:
        if self.state.camera is None:
            return
        self.state.camera.orbit(delta_azimuth, delta_elevation)
        self._render_viewport()

    def _position_nav_widget(self) -> None:
        if not dpg.does_item_exist("viewport_nav") or not dpg.does_item_exist("viewport_image"):
            return
        image_pos = dpg.get_item_rect_min("viewport_image")
        image_size = dpg.get_item_rect_size("viewport_image")
        nav_size = dpg.get_item_rect_size("viewport_nav")
        if image_size[0] <= 0 or image_size[1] <= 0:
            return
        x = int(image_pos[0] + image_size[0] - nav_size[0] - 12)
        y = int(image_pos[1] + 12)
        dpg.set_item_pos("viewport_nav", [x, y])

    def _project_world_to_screen(
        self, camera: OrbitCamera, point: np.ndarray
    ) -> tuple[int, int] | None:
        mvp = camera.mvp_matrix(self.state.viewport_size)
        clip = mvp @ np.array([point[0], point[1], point[2], 1.0], dtype=np.float32)
        w = float(clip[3])
        if abs(w) < 1e-6:
            return None
        ndc = clip[:3] / w
        if ndc[2] < -1.2 or ndc[2] > 1.2:
            return None
        width, height = self.state.viewport_size
        sx = int((ndc[0] * 0.5 + 0.5) * width)
        sy = int((1.0 - (ndc[1] * 0.5 + 0.5)) * height)
        return sx, sy

    def _draw_line_on_rgba(
        self,
        rgba: np.ndarray,
        p0: tuple[int, int],
        p1: tuple[int, int],
        colour: tuple[float, float, float],
        alpha: float,
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
        if not np.any(mask):
            return
        xs = xs[mask]
        ys = ys[mask]
        rgb = np.array(colour, dtype=np.float32)
        rgba[ys, xs, :3] = rgba[ys, xs, :3] * (1.0 - alpha) + rgb * alpha
        rgba[ys, xs, 3] = 1.0

    def _draw_world_grid(self, rgba: np.ndarray) -> None:
        if self.state.camera is None:
            return
        camera = self.state.camera
        width, _ = self.state.viewport_size
        if self.state.mesh_model is not None:
            mins = self.state.mesh_model.vertices.min(axis=0)
            maxs = self.state.mesh_model.vertices.max(axis=0)
            extent = float(np.linalg.norm(maxs - mins))
            size = max(2.0, extent * 1.2)
        else:
            size = 4.0
        major_lines = 8
        step = max(size / major_lines, 0.25)
        line_count = int(ceil(size / step))

        # XZ-plane grid at y=0, Blender-style world floor.
        for i in range(-line_count, line_count + 1):
            v = i * step
            p0 = self._project_world_to_screen(camera, np.array([-size, 0.0, v], dtype=np.float32))
            p1 = self._project_world_to_screen(camera, np.array([size, 0.0, v], dtype=np.float32))
            p2 = self._project_world_to_screen(camera, np.array([v, 0.0, -size], dtype=np.float32))
            p3 = self._project_world_to_screen(camera, np.array([v, 0.0, size], dtype=np.float32))
            if p0 and p1:
                strength = 0.32 if i % 5 == 0 else 0.16
                self._draw_line_on_rgba(rgba, p0, p1, (0.30, 0.34, 0.38), strength)
            if p2 and p3:
                strength = 0.32 if i % 5 == 0 else 0.16
                self._draw_line_on_rgba(rgba, p2, p3, (0.30, 0.34, 0.38), strength)

        # Axis tint like DCC viewports.
        ax0 = self._project_world_to_screen(camera, np.array([-size, 0.0, 0.0], dtype=np.float32))
        ax1 = self._project_world_to_screen(camera, np.array([size, 0.0, 0.0], dtype=np.float32))
        az0 = self._project_world_to_screen(camera, np.array([0.0, 0.0, -size], dtype=np.float32))
        az1 = self._project_world_to_screen(camera, np.array([0.0, 0.0, size], dtype=np.float32))
        if ax0 and ax1:
            self._draw_line_on_rgba(rgba, ax0, ax1, (0.82, 0.22, 0.22), 0.65)
        if az0 and az1:
            self._draw_line_on_rgba(rgba, az0, az1, (0.20, 0.35, 0.82), 0.65)

        # Draw a small center marker.
        center = self._project_world_to_screen(camera, np.array([0.0, 0.0, 0.0], dtype=np.float32))
        if center:
            cx, cy = center
            if 2 <= cx < width - 2 and 2 <= cy < rgba.shape[0] - 2:
                rgba[cy - 2 : cy + 3, cx - 2 : cx + 3, :3] = np.array([0.15, 0.15, 0.15], dtype=np.float32)
                rgba[cy - 2 : cy + 3, cx - 2 : cx + 3, 3] = 1.0

    def _load_mesh_model(self, mesh_model: MeshModel) -> None:
        viewport_size = self.state.viewport_size
        try:
            camera = OrbitCamera.for_mesh(mesh_model.vertices)
            renderer = MeshRenderer(self._create_context(), mesh_model, viewport_size)
        except Exception as exc:
            logger.exception("Failed to initialize renderer for loaded mesh")
            self._set_status(f"Mesh loaded but renderer failed: {exc}")
            return
        self.state.mesh_model = mesh_model
        self.state.camera = camera
        self.state.renderer = renderer
        self.state.paint_tool = PaintTool(mesh_model)
        self.state.sketch_tool = SketchTool(viewport_size)
        self.state.viewport_size = viewport_size
        self._render_viewport()
        self._set_status(f"Loaded mesh with {mesh_model.face_count} faces")

    def _viewport_size(self) -> tuple[int, int]:
        return self.state.viewport_size

    def _ensure_texture_size(self, size: tuple[int, int]) -> None:
        # Dynamic texture recreation during the render loop can crash in DearPyGui.
        # Keep a fixed render target size for stability.
        if size != self.state.viewport_size:
            self.state.viewport_size = size
            if self.state.renderer:
                self.state.renderer.resize(size)
            if self.state.sketch_tool:
                self.state.sketch_tool.resize(size)

    def _render_snapshot(self) -> RenderSnapshot | None:
        if (
            not self.state.mesh_model
            or not self.state.camera
            or not self.state.renderer
        ):
            # Show a software grid when no model is loaded to avoid a black viewport.
            return make_grid_snapshot(self.state.viewport_size)
        return self.state.renderer.render(self.state.camera)

    def _render_viewport(self) -> None:
        snapshot = self._render_snapshot()
        if snapshot is None:
            return
        rgba = snapshot.rgba.astype(np.float32) / 255.0
        self._draw_world_grid(rgba)
        if self.state.sketch_tool and self.state.sketch_tool.current:
            overlay = (
                np.asarray(self.state.sketch_tool.current.image, dtype=np.float32)
                / 255.0
            )
            alpha = overlay[:, :, 3:4]
            rgba = overlay[:, :, :4] * alpha + rgba * (1.0 - alpha)
            rgba[:, :, 3] = 1.0
        dpg.set_value("viewport_texture", rgba.flatten().tolist())
        self._position_nav_widget()

    def _mouse_inside_viewport(self) -> bool:
        return bool(dpg.is_item_hovered("viewport_image"))

    def _viewport_mouse_position(self) -> tuple[float, float]:
        mouse_x, mouse_y = dpg.get_mouse_pos(local=False)
        image_x, image_y = dpg.get_item_rect_min("viewport_image")
        return mouse_x - image_x, mouse_y - image_y

    def _mouse_within_frame(self, mouse_pos: tuple[float, float]) -> bool:
        x, y = mouse_pos
        width, height = self.state.viewport_size
        return 0.0 <= x < float(width) and 0.0 <= y < float(height)

    def _pick_face(self, mouse_pos: tuple[float, float]) -> int | None:
        if (
            not self.state.mesh_model
            or not self.state.camera
            or not self._mouse_within_frame(mouse_pos)
        ):
            return None
        face_id = pick_face_cpu(
            self.state.mesh_model,
            self.state.camera,
            mouse_pos[0],
            mouse_pos[1],
            self.state.viewport_size,
        )
        if face_id is None and self.state.renderer:
            try:
                face_id = self.state.renderer.pick_face(
                    self.state.camera, int(mouse_pos[0]), int(mouse_pos[1])
                )
            except Exception:
                face_id = None
        self.state.hovered_face = face_id
        return face_id

    def _apply_paint(self, face_id: int) -> None:
        if not self.state.paint_tool or not self.state.renderer:
            return
        if self.state.tool_mode == "fill" or (
            self.state.tool_mode == "paint" and self.state.shift_down
        ):
            touched = self.state.paint_tool.flood_fill(
                face_id, self.state.active_colour
            )
        else:
            touched = self.state.paint_tool.paint_face(
                face_id, self.state.active_colour
            )
        for touched_face in touched:
            self.state.renderer.update_face_colour(touched_face)
        if touched:
            self._render_viewport()
            self._set_status(f"Painted {len(touched)} face(s)")

    def _apply_sketch_point(self, mouse_pos: tuple[float, float]) -> None:
        if (
            not self.state.mesh_model
            or not self.state.camera
            or not self.state.sketch_tool
        ):
            return
        primitive = self.state.sketch_primitive
        width = int(dpg.get_value("sketch_stroke_width"))
        if primitive == "text":
            text = str(dpg.get_value("sketch_text_value"))
            size = int(dpg.get_value("sketch_text_size"))
            self.state.sketch_tool.add_text(
                self.state.mesh_model,
                self.state.camera,
                mouse_pos,
                text,
                self.state.active_colour,
                size,
            )
            self._render_viewport()

    def _on_mouse_down(self, sender: int, app_data: tuple[int, float]) -> None:
        if not self._mouse_inside_viewport():
            return
        mouse_pos = self._viewport_mouse_position()
        if not self._mouse_within_frame(mouse_pos):
            return
        button = app_data[0]
        self.state.dragging = True
        self.state.drag_button = button
        self.state.drag_origin = mouse_pos
        if button == 0:
            if self.state.tool_mode == "sketch":
                if self.state.sketch_primitive == "text":
                    self._apply_sketch_point(mouse_pos)
                elif self.state.sketch_primitive == "freehand":
                    self.state.freehand_points = [mouse_pos]
            else:
                face_id = self._pick_face(mouse_pos)
                if face_id is not None:
                    self._apply_paint(face_id)

    def _on_mouse_release(self, sender: int, app_data: tuple[int, float]) -> None:
        if not self.state.dragging or self.state.drag_origin is None:
            return
        mouse_pos = self._viewport_mouse_position()
        button = app_data[0]
        if (
            button == 0
            and self.state.tool_mode == "sketch"
            and self.state.mesh_model
            and self.state.camera
            and self.state.sketch_tool
        ):
            primitive = self.state.sketch_primitive
            width = int(dpg.get_value("sketch_stroke_width"))
            if primitive == "rect":
                x0, y0 = self.state.drag_origin
                x1, y1 = mouse_pos
                self.state.sketch_tool.add_rectangle(
                    self.state.mesh_model,
                    self.state.camera,
                    (x0, y0, x1, y1),
                    self.state.active_colour,
                    None,
                    width,
                )
                self._render_viewport()
            elif primitive == "line":
                self.state.sketch_tool.add_line(
                    self.state.mesh_model,
                    self.state.camera,
                    [self.state.drag_origin, mouse_pos],
                    self.state.active_colour,
                    width,
                )
                self._render_viewport()
            elif primitive == "freehand" and self.state.freehand_points:
                self.state.freehand_points.append(mouse_pos)
                self.state.sketch_tool.add_freehand(
                    self.state.mesh_model,
                    self.state.camera,
                    self.state.freehand_points,
                    self.state.active_colour,
                    width,
                )
                self.state.freehand_points = None
                self._render_viewport()
        self.state.dragging = False
        self.state.drag_button = None
        self.state.drag_origin = None

    def _on_mouse_move(self, sender: int, app_data: tuple[float, float]) -> None:
        if not self._mouse_inside_viewport():
            return
        mouse_pos = self._viewport_mouse_position()
        if not self._mouse_within_frame(mouse_pos):
            return
        if (
            self.state.dragging
            and self.state.camera
            and self.state.drag_origin is not None
        ):
            dx = mouse_pos[0] - self.state.drag_origin[0]
            dy = mouse_pos[1] - self.state.drag_origin[1]
            if self.state.drag_button == 1:
                self.state.camera.orbit(dx * 0.4, dy * 0.4)
                self.state.drag_origin = mouse_pos
                self._render_viewport()
            elif self.state.drag_button == 2:
                self.state.camera.pan(dx, dy)
                self.state.drag_origin = mouse_pos
                self._render_viewport()
            elif self.state.drag_button == 0 and self.state.tool_mode == "paint":
                face_id = self._pick_face(mouse_pos)
                if face_id is not None:
                    self._apply_paint(face_id)
            elif (
                self.state.drag_button == 0
                and self.state.tool_mode == "sketch"
                and self.state.sketch_primitive == "freehand"
            ):
                if self.state.freehand_points is not None:
                    self.state.freehand_points.append(mouse_pos)
        else:
            face_id = self._pick_face(mouse_pos)
            if face_id is not None:
                self._set_status(
                    f"Hovered face: {face_id} | Mode: {self.state.tool_mode}"
                )

    def _on_mouse_wheel(self, sender: int, app_data: float) -> None:
        if not self.state.camera or not self._mouse_inside_viewport():
            return
        self.state.camera.zoom(app_data * 0.1)
        self._render_viewport()

    def _on_key_down(self, sender: int, app_data: int) -> None:
        if app_data in (340, 344):
            self.state.shift_down = True
        if app_data in (341, 345):
            self.state.ctrl_down = True
        if self.state.ctrl_down and app_data == 90:
            self._on_undo()

    def _on_key_release(self, sender: int, app_data: int) -> None:
        if app_data in (340, 344):
            self.state.shift_down = False
        if app_data in (341, 345):
            self.state.ctrl_down = False

    def _on_open_selected(self, sender: int, app_data: dict[str, object]) -> None:
        try:
            path = str(app_data["file_path_name"])
            suffix = Path(path).suffix.lower()
            mesh_model = load_project(path) if suffix == ".json" else load_stl(path)
            self._load_mesh_model(mesh_model)
        except Exception as exc:
            logger.exception("Failed to open file")
            self._set_status(f"Open failed: {exc}")

    def _on_export_selected(self, sender: int, app_data: dict[str, object]) -> None:
        if not self.state.mesh_model:
            self._set_status("No mesh loaded")
            return
        path = str(app_data["file_path_name"])
        sketch_image = (
            self.state.sketch_tool.current.image
            if self.state.sketch_tool and self.state.sketch_tool.current
            else None
        )
        view_projection = (
            self.state.sketch_tool.current.view_projection
            if self.state.sketch_tool and self.state.sketch_tool.current
            else None
        )
        issues = export_3mf(
            path,
            self.state.mesh_model,
            sketch_image,
            view_projection,
            self.state.viewport_size if sketch_image is not None else None,
        )
        self._set_status("Exported 3MF | " + " ".join(issues))

    def _on_save_project_selected(
        self, sender: int, app_data: dict[str, object]
    ) -> None:
        if not self.state.mesh_model:
            self._set_status("No mesh loaded")
            return
        save_project(str(app_data["file_path_name"]), self.state.mesh_model)
        self._set_status("Saved project JSON")

    def _on_bake_sketch(self) -> None:
        if (
            not self.state.mesh_model
            or not self.state.renderer
            or not self.state.sketch_tool
            or not self.state.sketch_tool.current
        ):
            self._set_status("No sketch to bake")
            return
        baked = bake_sketch_to_faces(
            self.state.mesh_model,
            self.state.sketch_tool.current.image,
            self.state.sketch_tool.current.view_projection,
            self.state.viewport_size,
        )
        for face_id, colour in baked.items():
            self.state.mesh_model.set_face_colour(face_id, colour)
            self.state.renderer.update_face_colour(face_id)
        self.state.sketch_tool.clear(self.state.mesh_model)
        self._render_viewport()
        self._set_status(f"Baked {len(baked)} face colours")

    def _on_undo(
        self, sender: int | None = None, app_data: object | None = None
    ) -> None:
        if not self.state.paint_tool or not self.state.renderer:
            self._set_status("Nothing to undo")
            return
        touched = self.state.paint_tool.undo()
        for face_id in touched:
            self.state.renderer.update_face_colour(face_id)
        self._render_viewport()
        self._set_status(f"Undo restored {len(touched)} face(s)")

    def run(self) -> None:
        while dpg.is_dearpygui_running():
            try:
                self._render_viewport()
            except Exception as exc:
                logger.exception("Render loop failure")
                self._set_status(f"Render error: {exc}")
            dpg.render_dearpygui_frame()
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

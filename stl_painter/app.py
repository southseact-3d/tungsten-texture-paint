from __future__ import annotations

import ctypes
import logging
from math import ceil
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

import dearpygui.dearpygui as dpg
import numpy as np

from .camera import OrbitCamera
from .color_utils import DEFAULT_COLOR, Color, clamp_color
from .exporter import export_3mf
from .importer import load_stl
from .mesh_model import MeshModel
from .logging_utils import log_file_path
from .paint_tool import PaintTool
from .picking import pick_face_cpu
from .project_io import load_project, save_project
from .renderer import MeshRenderer, RenderSnapshot, make_grid_snapshot, probe_gpu_support
from .sketch_tool import SketchTool, bake_sketch_to_faces

logger = logging.getLogger(__name__)
_USER32 = ctypes.windll.user32


class _POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class _RECT(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]

ToolMode = Literal["paint", "fill", "sketch"]
SketchPrimitive = Literal["text", "rect", "line", "freehand"]
NavAxis = Literal["xp", "xn", "yp", "yn", "zp", "zn"]

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
    nav_dragging: bool = False
    nav_drag_origin: tuple[float, float] | None = None
    nav_hover_axis: NavAxis | None = None
    nav_pressed_axis: NavAxis | None = None
    nav_drag_moved: bool = False
    freehand_points: list[tuple[float, float]] | None = None
    viewport_dirty: bool = True


class TexturePainterApp:
    def __init__(self) -> None:
        self.state = AppState(sketch_tool=SketchTool((960, 720)))
        self.state.camera = OrbitCamera(np.array([0.0, 0.0, 0.0], dtype=np.float32), 5.0)
        self._texture_data = np.zeros(
            (self.state.viewport_size[1], self.state.viewport_size[0], 4),
            dtype=np.float32,
        )
        self._create_ui()

    def _create_ui(self) -> None:
        dpg.create_context()
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                width=self.state.viewport_size[0],
                height=self.state.viewport_size[1],
                default_value=self._texture_data,
                format=dpg.mvFormat_Float_rgba,
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
                        dpg.add_drawlist(width=132, height=132, tag="viewport_nav_pad")
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
        self._draw_nav_pad()

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
        logger.info("STATUS | %s", text)

    def _mark_viewport_dirty(self) -> None:
        self.state.viewport_dirty = True

    def _reset_view(self) -> None:
        if self.state.mesh_model is not None:
            self.state.camera = OrbitCamera.for_mesh(self.state.mesh_model.vertices)
        else:
            self.state.camera = OrbitCamera(np.array([0.0, 0.0, 0.0], dtype=np.float32), 5.0)
        self._mark_viewport_dirty()

    def _orbit_step(self, delta_azimuth: float, delta_elevation: float) -> None:
        if self.state.camera is None:
            return
        self.state.camera.orbit(delta_azimuth, delta_elevation)
        self._mark_viewport_dirty()

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

    def _draw_nav_pad(self) -> None:
        if not dpg.does_item_exist("viewport_nav_pad"):
            return
        dpg.delete_item("viewport_nav_pad", children_only=True)
        gizmo = self._nav_gizmo_geometry()
        center = gizmo["center"]
        radius = gizmo["radius"]
        orbit_radius = gizmo["orbit_radius"]
        dpg.draw_circle(
            center,
            radius,
            color=(118, 126, 140, 220),
            fill=(243, 245, 248, 210),
            thickness=2,
            parent="viewport_nav_pad",
        )
        dpg.draw_circle(
            center,
            orbit_radius,
            color=(208, 212, 220, 180),
            thickness=1,
            parent="viewport_nav_pad",
        )
        dpg.draw_circle(
            center,
            4,
            color=(84, 90, 100, 255),
            fill=(84, 90, 100, 255),
            parent="viewport_nav_pad",
        )
        if self.state.camera is None:
            dpg.draw_text((40, 112), "Drag to rotate", color=(82, 88, 96, 255), size=13, parent="viewport_nav_pad")
            return

        axis_items = self._nav_axis_items()
        for item in sorted(axis_items, key=lambda axis: axis["depth"], reverse=True):
            start = center
            end = item["point"]
            line_colour = item["line_colour"]
            if not item["positive"]:
                line_colour = (*line_colour[:3], 90)
            dpg.draw_line(start, end, color=line_colour, thickness=2, parent="viewport_nav_pad")

        for item in sorted(axis_items, key=lambda axis: axis["depth"], reverse=True):
            point = item["point"]
            hovered = self.state.nav_hover_axis == item["axis"]
            if item["positive"]:
                fill = item["fill"]
                outline = (24, 24, 28, 255) if hovered else item["outline"]
                radius_px = 16 if hovered else 14
                dpg.draw_circle(
                    point,
                    radius_px,
                    color=outline,
                    fill=fill,
                    thickness=2,
                    parent="viewport_nav_pad",
                )
                text_pos = (point[0] - 6, point[1] - 10)
                dpg.draw_text(text_pos, item["label"], color=(255, 255, 255, 255), size=18, parent="viewport_nav_pad")
            else:
                radius_px = 11 if hovered else 9
                dpg.draw_circle(
                    point,
                    radius_px,
                    color=item["outline"],
                    fill=(0, 0, 0, 0),
                    thickness=2,
                    parent="viewport_nav_pad",
                )

        dpg.draw_text((28, 112), "Drag to rotate", color=(82, 88, 96, 255), size=13, parent="viewport_nav_pad")

    def _nav_pad_hovered(self) -> bool:
        return dpg.does_item_exist("viewport_nav_pad") and bool(dpg.is_item_hovered("viewport_nav_pad"))

    def _nav_gizmo_geometry(self) -> dict[str, object]:
        width, height = 132, 132
        return {
            "size": (width, height),
            "center": (width / 2.0, height / 2.0),
            "radius": 44.0,
            "orbit_radius": 34.0,
            "axis_length": 32.0,
            "handle_radius": 14.0,
        }

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
        left = int(round(screen_x + inset))
        top = int(round(screen_y + inset))
        right = int(round(screen_x + width - inset))
        bottom = int(round(screen_y + height - inset))
        return left, top, right, bottom

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

    def _nav_axis_items(self) -> list[dict[str, object]]:
        if self.state.camera is None:
            return []
        gizmo = self._nav_gizmo_geometry()
        center_x, center_y = gizmo["center"]  # type: ignore[misc]
        axis_length = float(gizmo["axis_length"])
        rotation = self.state.camera.view_matrix()[:3, :3]
        axes: list[tuple[NavAxis, np.ndarray, tuple[int, int, int, int], str, bool]] = [
            ("xp", np.array([1.0, 0.0, 0.0], dtype=np.float32), (227, 74, 89, 255), "X", True),
            ("xn", np.array([-1.0, 0.0, 0.0], dtype=np.float32), (227, 74, 89, 255), "X", False),
            ("yp", np.array([0.0, 1.0, 0.0], dtype=np.float32), (130, 196, 55, 255), "Y", True),
            ("yn", np.array([0.0, -1.0, 0.0], dtype=np.float32), (130, 196, 55, 255), "Y", False),
            ("zp", np.array([0.0, 0.0, 1.0], dtype=np.float32), (64, 150, 255, 255), "Z", True),
            ("zn", np.array([0.0, 0.0, -1.0], dtype=np.float32), (64, 150, 255, 255), "Z", False),
        ]
        items: list[dict[str, object]] = []
        for axis_name, direction, colour, label, positive in axes:
            camera_space = rotation @ direction
            point = (
                center_x + float(camera_space[0]) * axis_length,
                center_y - float(camera_space[1]) * axis_length,
            )
            items.append(
                {
                    "axis": axis_name,
                    "label": label,
                    "positive": positive,
                    "point": point,
                    "depth": float(camera_space[2]),
                    "fill": colour if positive else (0, 0, 0, 0),
                    "outline": colour,
                    "line_colour": colour,
                }
            )
        return items

    def _nav_hit_test(self, mouse_pos: tuple[float, float]) -> tuple[str, NavAxis | None]:
        gizmo = self._nav_gizmo_geometry()
        handle_radius = float(gizmo["handle_radius"])
        center_x, center_y = gizmo["center"]  # type: ignore[misc]
        dx = mouse_pos[0] - center_x
        dy = mouse_pos[1] - center_y
        for item in self._nav_axis_items():
            point_x, point_y = item["point"]  # type: ignore[misc]
            if (mouse_pos[0] - point_x) ** 2 + (mouse_pos[1] - point_y) ** 2 <= (handle_radius + 5.0) ** 2:
                return "axis", item["axis"]  # type: ignore[return-value]
        if dx * dx + dy * dy <= (float(gizmo["radius"]) + 8.0) ** 2:
            return "orbit", None
        return "none", None

    def _mouse_button_from_event(self, app_data: object) -> int | None:
        if isinstance(app_data, int):
            return app_data
        if isinstance(app_data, (tuple, list)) and app_data:
            first = app_data[0]
            if isinstance(first, int):
                return first
        return None

    def _snap_camera_to_axis(self, axis: NavAxis) -> None:
        if self.state.camera is None:
            return
        targets: dict[NavAxis, tuple[float, float]] = {
            "xp": (0.0, 0.0),
            "xn": (180.0, 0.0),
            "yp": (0.0, 89.0),
            "yn": (0.0, -89.0),
            "zp": (90.0, 0.0),
            "zn": (-90.0, 0.0),
        }
        azimuth, elevation = targets[axis]
        self.state.camera.set_angles(azimuth, elevation)
        self._mark_viewport_dirty()

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
        logger.info(
            "Loading mesh into viewport | faces=%s | vertices=%s | source=%s",
            mesh_model.face_count,
            mesh_model.vertex_count,
            mesh_model.source_path,
        )
        try:
            gpu_available, gpu_description = probe_gpu_support()
            logger.info(
                "Viewport GPU probe | available=%s | description=%s",
                gpu_available,
                gpu_description,
            )
            camera = OrbitCamera.for_mesh(mesh_model.vertices)
            renderer = MeshRenderer(None, mesh_model, viewport_size, prefer_gpu=False)
        except Exception as exc:
            logger.exception("Failed to initialize renderer for loaded mesh")
            self._set_status(
                f"Mesh loaded but renderer failed: {exc} | See log: {log_file_path().name}"
            )
            return
        self.state.mesh_model = mesh_model
        self.state.camera = camera
        self.state.renderer = renderer
        self.state.paint_tool = PaintTool(mesh_model)
        self.state.sketch_tool = SketchTool(viewport_size)
        self.state.viewport_size = viewport_size
        self._mark_viewport_dirty()
        self._set_status(
            f"Loaded mesh with {mesh_model.face_count} faces from {Path(mesh_model.source_path or 'project').name}"
        )

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
            self._mark_viewport_dirty()

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
        if not self.state.viewport_dirty:
            return
        snapshot = self._render_snapshot()
        if snapshot is None:
            return
        rgba = snapshot.rgba.astype(np.float32) / 255.0
        if self.state.mesh_model is None:
            self._draw_world_grid(rgba)
        if self.state.sketch_tool and self.state.sketch_tool.current:
            overlay = (
                np.asarray(self.state.sketch_tool.current.image, dtype=np.float32)
                / 255.0
            )
            alpha = overlay[:, :, 3:4]
            rgba = overlay[:, :, :4] * alpha + rgba * (1.0 - alpha)
            rgba[:, :, 3] = 1.0
        self._texture_data[:, :, :] = rgba
        dpg.set_value("viewport_texture", self._texture_data)
        self._position_nav_widget()
        self._draw_nav_pad()
        self.state.viewport_dirty = False

    def _mouse_inside_viewport(self) -> bool:
        if not dpg.does_item_exist("viewport_image"):
            return False
        mouse_x, mouse_y = dpg.get_mouse_pos(local=False)
        image_x, image_y = dpg.get_item_rect_min("viewport_image")
        image_width, image_height = dpg.get_item_rect_size("viewport_image")
        within_image = (
            image_x <= mouse_x < image_x + image_width
            and image_y <= mouse_y < image_y + image_height
        )
        return within_image or self._nav_pad_hovered()

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
        face_id = None
        if self.state.renderer:
            try:
                face_id = self.state.renderer.pick_face(
                    self.state.camera, int(mouse_pos[0]), int(mouse_pos[1])
                )
            except Exception:
                logger.exception("GPU face picking failed")
                face_id = None
        if face_id is None:
            face_id = pick_face_cpu(
                self.state.mesh_model,
                self.state.camera,
                mouse_pos[0],
                mouse_pos[1],
                self.state.viewport_size,
            )
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
            self._mark_viewport_dirty()
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
            self._mark_viewport_dirty()

    def _on_mouse_down(self, sender: int, app_data: tuple[int, float]) -> None:
        button = self._mouse_button_from_event(app_data)
        if self._nav_pad_hovered() and self.state.camera is not None and button == 0:
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
        if not self._mouse_within_frame(mouse_pos):
            return
        if button is None:
            return
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

    def _on_mouse_release(self, sender: int, app_data: tuple[int, float] | int) -> None:
        button = self._mouse_button_from_event(app_data)
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
        if not self.state.dragging or self.state.drag_origin is None:
            return
        mouse_pos = self._viewport_mouse_position()
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
                self._mark_viewport_dirty()
            elif primitive == "line":
                self.state.sketch_tool.add_line(
                    self.state.mesh_model,
                    self.state.camera,
                    [self.state.drag_origin, mouse_pos],
                    self.state.active_colour,
                    width,
                )
                self._mark_viewport_dirty()
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
                self._mark_viewport_dirty()
        self.state.dragging = False
        self.state.drag_button = None
        self.state.drag_origin = None

    def _on_mouse_move(self, sender: int, app_data: tuple[float, float]) -> None:
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
        if self.state.nav_dragging and self.state.camera and self.state.nav_drag_origin is not None:
            mouse_pos = self._global_mouse_position()
            dx = mouse_pos[0] - self.state.nav_drag_origin[0]
            dy = mouse_pos[1] - self.state.nav_drag_origin[1]
            if dx * dx + dy * dy >= 1.0:
                self.state.nav_drag_moved = True
            if self.state.nav_drag_moved:
                self.state.camera.orbit(dx * 2.4, -dy * 2.4)
                new_origin = self._center_nav_cursor()
                self.state.nav_drag_origin = new_origin or self.state.nav_drag_origin
            if self.state.nav_drag_moved:
                self._mark_viewport_dirty()
            return
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
                self.state.camera.orbit(dx * 0.4, -dy * 0.4)
                self.state.drag_origin = mouse_pos
                self._mark_viewport_dirty()
            elif self.state.drag_button == 2:
                self.state.camera.pan(dx, dy)
                self.state.drag_origin = mouse_pos
                self._mark_viewport_dirty()
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
            return

    def _on_mouse_wheel(self, sender: int, app_data: float) -> None:
        if not self.state.camera or not self._mouse_inside_viewport():
            return
        self.state.camera.zoom(app_data * 0.1)
        self._mark_viewport_dirty()

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
            logger.info("Open requested | path=%s | suffix=%s", path, suffix)
            mesh_model = load_project(path) if suffix == ".json" else load_stl(path)
            self._load_mesh_model(mesh_model)
        except Exception as exc:
            logger.exception("Failed to open file")
            self._set_status(f"Open failed: {exc} | See log: {log_file_path().name}")

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
        self._mark_viewport_dirty()
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
        self._mark_viewport_dirty()
        self._set_status(f"Undo restored {len(touched)} face(s)")

    def run(self) -> None:
        try:
            while dpg.is_dearpygui_running():
                try:
                    self._render_viewport()
                except Exception as exc:
                    logger.exception("Render loop failure")
                    self._set_status(f"Render error: {exc}")
                dpg.render_dearpygui_frame()
                self._position_nav_widget()
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
        export_3mf(handle.name, cube)

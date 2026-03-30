from __future__ import annotations

from dataclasses import dataclass

import dearpygui.dearpygui as dpg


@dataclass(frozen=True, slots=True)
class WorkspaceLayout:
    left_panel_width: int
    viewport_width: int
    right_panel_width: int
    workspace_height: int


def compute_workspace_layout(
    window_width: int,
    window_height: int,
    *,
    show_tools_panel: bool,
) -> WorkspaceLayout:
    content_width = max(640, int(window_width) - 40)
    workspace_height = max(360, int(window_height) - 170)
    desired_left = 300 if show_tools_panel else 0
    desired_right = 360
    min_left = 220 if show_tools_panel else 0
    min_right = 280
    min_viewport = 420
    gaps = 16 if show_tools_panel else 8
    left_width = desired_left
    right_width = desired_right
    remaining_width = content_width - left_width - right_width - gaps

    if remaining_width < min_viewport:
        deficit = min_viewport - remaining_width
        right_shrink = min(deficit, max(0, right_width - min_right))
        right_width -= right_shrink
        deficit -= right_shrink
        if deficit > 0 and show_tools_panel:
            left_shrink = min(deficit, max(0, left_width - min_left))
            left_width -= left_shrink
            deficit -= left_shrink
        remaining_width = content_width - left_width - right_width - gaps
        if deficit > 0:
            remaining_width = max(320, remaining_width)

    viewport_width = max(320, remaining_width)
    return WorkspaceLayout(
        left_panel_width=left_width,
        viewport_width=viewport_width,
        right_panel_width=right_width,
        workspace_height=workspace_height,
    )


def sync_mode_sections(interaction_mode: str) -> None:
    if dpg.does_item_exist("paint_section"):
        dpg.configure_item("paint_section", show=interaction_mode == "paint")
    if dpg.does_item_exist("sketch_section"):
        dpg.configure_item("sketch_section", show=interaction_mode == "sketch")


def update_tool_log(tool_logs: list[tuple[str, str]]) -> None:
    if not dpg.does_item_exist("ai_tool_log"):
        return
    text = "\n".join(f"{name}: {summary}" for name, summary in tool_logs[-12:])
    dpg.set_value("ai_tool_log", text)

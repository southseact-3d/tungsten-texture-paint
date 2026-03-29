from __future__ import annotations

import dearpygui.dearpygui as dpg


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

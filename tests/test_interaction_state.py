from __future__ import annotations

import pytest

from stl_painter.interaction_state import InteractionState


def test_interaction_state_starts_in_preview_workspace() -> None:
    state = InteractionState()

    assert state.workspace_mode == "preview"
    assert state.interaction_mode == "paint"


def test_interaction_state_default_paint_tool() -> None:
    state = InteractionState()
    assert state.paint_tool == "brush"


def test_interaction_state_default_sketch_tool() -> None:
    state = InteractionState()
    assert state.sketch_tool == "select"


def test_interaction_state_can_change_workspace() -> None:
    state = InteractionState()
    state.workspace_mode = "paint"
    assert state.workspace_mode == "paint"


def test_interaction_state_can_change_interaction_mode() -> None:
    state = InteractionState()
    state.interaction_mode = "sketch"
    assert state.interaction_mode == "sketch"


def test_interaction_state_can_change_paint_tool() -> None:
    state = InteractionState()
    state.paint_tool = "fill"
    assert state.paint_tool == "fill"


def test_interaction_state_can_change_sketch_tool() -> None:
    state = InteractionState()
    state.sketch_tool = "line"
    assert state.sketch_tool == "line"


def test_interaction_state_hovered_face() -> None:
    state = InteractionState()
    assert state.hovered_face is None


def test_interaction_state_set_hovered_face() -> None:
    state = InteractionState()
    state.hovered_face = 42
    assert state.hovered_face == 42


def test_interaction_state_selected_faces() -> None:
    state = InteractionState()
    assert state.selected_faces == set()


def test_interaction_state_add_selected_face() -> None:
    state = InteractionState()
    state.selected_faces.add(42)
    assert 42 in state.selected_faces


def test_interaction_state_active_colour() -> None:
    state = InteractionState()
    assert state.active_colour == (255, 80, 80, 255)


def test_interaction_state_set_active_colour() -> None:
    state = InteractionState()
    state.active_colour = (0, 255, 0, 255)
    assert state.active_colour == (0, 255, 0, 255)


def test_interaction_state_dragging() -> None:
    state = InteractionState()
    assert state.dragging is False


def test_interaction_state_set_dragging() -> None:
    state = InteractionState()
    state.dragging = True
    assert state.dragging is True


def test_interaction_state_shift_down() -> None:
    state = InteractionState()
    assert state.shift_down is False


def test_interaction_state_ctrl_down() -> None:
    state = InteractionState()
    assert state.ctrl_down is False


def test_interaction_state_viewport_size() -> None:
    state = InteractionState()
    assert state.viewport_size == (960, 720)


def test_interaction_state_viewport_dirty() -> None:
    state = InteractionState()
    assert state.viewport_dirty is True


def test_interaction_state_selected_entity_id() -> None:
    state = InteractionState()
    assert state.selected_entity_id is None


def test_interaction_state_brush_settings() -> None:
    state = InteractionState()
    assert state.brush.radius == 0.15
    assert state.brush.opacity == 0.8
    assert state.brush.falloff == "smooth"


def test_interaction_state_ai_settings() -> None:
    state = InteractionState()
    assert state.ai_settings.model == "gpt-4.1-mini"
    assert state.ai_settings.provider_base_url == "https://api.openai.com/v1"

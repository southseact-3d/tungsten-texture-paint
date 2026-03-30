from __future__ import annotations

from stl_painter.interaction_state import InteractionState


def test_interaction_state_starts_in_preview_workspace() -> None:
    state = InteractionState()

    assert state.workspace_mode == "preview"
    assert state.interaction_mode == "paint"

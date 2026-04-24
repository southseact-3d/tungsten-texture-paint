from __future__ import annotations

import pytest

from stl_painter.ui_panels import compute_workspace_layout


def test_compute_workspace_layout_preserves_ai_sidebar_in_preview() -> None:
    layout = compute_workspace_layout(1680, 920, show_tools_panel=False)

    assert layout.left_panel_width == 0
    assert layout.right_panel_width == 360
    assert layout.viewport_width > 900
    assert layout.workspace_height > 700


def test_compute_workspace_layout_shrinks_sidebars_before_viewport() -> None:
    layout = compute_workspace_layout(1040, 820, show_tools_panel=True)

    assert layout.left_panel_width < 300
    assert layout.left_panel_width >= 220
    assert layout.right_panel_width < 360
    assert layout.right_panel_width >= 280
    assert layout.viewport_width >= 420


def test_compute_workspace_layout_with_tools_panel() -> None:
    layout = compute_workspace_layout(1200, 800, show_tools_panel=True)

    assert layout.left_panel_width > 0
    assert layout.right_panel_width > 0
    assert layout.viewport_width > 0


def test_compute_workspace_layout_without_tools_panel() -> None:
    layout = compute_workspace_layout(1200, 800, show_tools_panel=False)

    assert layout.left_panel_width == 0
    assert layout.viewport_width > 0


def test_compute_workspace_layout_minimum_viewport() -> None:
    layout = compute_workspace_layout(640, 480, show_tools_panel=True)

    assert layout.viewport_width > 0
    assert layout.workspace_height > 0


def test_compute_workspace_layout_returns_expected_fields() -> None:
    layout = compute_workspace_layout(800, 600, show_tools_panel=True)

    assert hasattr(layout, "left_panel_width")
    assert hasattr(layout, "right_panel_width")
    assert hasattr(layout, "viewport_width")
    assert hasattr(layout, "workspace_height")

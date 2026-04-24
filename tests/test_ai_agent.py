from __future__ import annotations

import pytest

from stl_painter.ai_agent import IMAGE_ONLY_PROMPT, build_user_prompt


def test_build_user_prompt_prefers_explicit_text() -> None:
    assert (
        build_user_prompt("  Add blue racing stripes  ", "reference.png")
        == "Add blue racing stripes"
    )


def test_build_user_prompt_uses_image_only_instruction() -> None:
    assert build_user_prompt("", "reference.png") == IMAGE_ONLY_PROMPT


def test_build_user_prompt_requires_text_or_image() -> None:
    assert build_user_prompt("   ", "   ") == ""


def test_build_user_prompt_trims_whitespace() -> None:
    assert build_user_prompt("  Paint it red  ", "image.png") == "Paint it red"


def test_build_user_prompt_empty_image_path() -> None:
    result = build_user_prompt("Add texture", "")
    assert result == "Add texture"

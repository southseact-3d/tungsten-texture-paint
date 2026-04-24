from __future__ import annotations

import pytest

from stl_painter.color_utils import (
    DEFAULT_COLOR,
    Color,
    blend_over,
    clamp_color,
    normalize_rgba,
    rgb_hex,
)


def test_clamp_color_rgb() -> None:
    result = clamp_color((128, 64, 200))
    assert result == (128, 64, 200, 255)


def test_clamp_color_rgba() -> None:
    result = clamp_color((128, 64, 200, 200))
    assert result == (128, 64, 200, 200)


def test_clamp_color_with_float() -> None:
    result = clamp_color((128.4, 64.6, 200.9, 200.1))
    assert result == (128, 65, 201, 200)


def test_clamp_color_clamps_negative() -> None:
    result = clamp_color((-10, 64, 200))
    assert result == (0, 64, 200, 255)


def test_clamp_color_clamps_oversized() -> None:
    result = clamp_color((300, 64, 200))
    assert result == (255, 64, 200, 255)


def test_clamp_color_rejects_invalid_length() -> None:
    with pytest.raises(ValueError, match="Expected RGB or RGBA"):
        clamp_color((128, 64))


def test_clamp_color_rejects_empty() -> None:
    with pytest.raises(ValueError, match="Expected RGB or RGBA"):
        clamp_color(())


def test_rgb_hex_returns_correct_format() -> None:
    result = rgb_hex((255, 128, 0))
    assert result == "#FF8000"


def test_rgb_hex_with_leading_zeros() -> None:
    result = rgb_hex((0, 15, 255))
    assert result == "#000FFF"


def test_normalize_rgba() -> None:
    result = normalize_rgba((128, 64, 200, 255))
    assert len(result) == 4
    assert 0.0 <= result[0] <= 1.0
    assert 0.0 <= result[1] <= 1.0
    assert 0.0 <= result[2] <= 1.0
    assert 0.0 <= result[3] <= 1.0


def test_normalize_rgba_max_value() -> None:
    result = normalize_rgba((255, 255, 255, 255))
    assert result == pytest.approx((1.0, 1.0, 1.0, 1.0))


def test_normalize_rgba_min_value() -> None:
    result = normalize_rgba((0, 0, 0, 0))
    assert result == pytest.approx((0.0, 0.0, 0.0, 0.0))


def test_blend_over_full_opacity() -> None:
    result = blend_over((100, 100, 100, 255), (255, 0, 0, 255))
    assert result[0] == 255
    assert result[1] == 0
    assert result[2] == 0


def test_blend_over_zero_opacity() -> None:
    result = blend_over((100, 100, 100, 255), (255, 0, 0, 0))
    assert result[0] == 100
    assert result[1] == 100
    assert result[2] == 100


def test_blend_over_half_opacity() -> None:
    base = (100, 100, 100, 255)
    overlay = (255, 0, 0, 128)
    result = blend_over(base, overlay)
    assert result[0] > 50
    assert result[1] >= 0
    assert result[2] >= 0


def test_blend_over_alpha_255() -> None:
    result = blend_over((100, 100, 100, 255), (255, 0, 0, 255))
    assert result[3] == 255


def test_default_color_is_defined() -> None:
    assert DEFAULT_COLOR == (176, 184, 196, 255)
    assert len(DEFAULT_COLOR) == 4

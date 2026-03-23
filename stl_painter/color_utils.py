from __future__ import annotations

from typing import Iterable

Color = tuple[int, int, int, int]

DEFAULT_COLOR: Color = (176, 184, 196, 255)


def clamp_color(color: Iterable[int | float]) -> Color:
    values = list(color)
    if len(values) == 3:
        values.append(255)
    if len(values) != 4:
        raise ValueError("Expected RGB or RGBA colour")
    return tuple(max(0, min(255, int(round(component)))) for component in values)  # type: ignore[return-value]


def rgb_hex(color: Color | tuple[int, int, int]) -> str:
    r, g, b = color[:3]
    return f"#{r:02X}{g:02X}{b:02X}"


def normalize_rgba(color: Color) -> tuple[float, float, float, float]:
    return tuple(component / 255.0 for component in color)  # type: ignore[return-value]


def blend_over(base: Color, overlay: Color) -> Color:
    alpha = overlay[3] / 255.0
    inverse = 1.0 - alpha
    blended = (
        int(round(overlay[0] * alpha + base[0] * inverse)),
        int(round(overlay[1] * alpha + base[1] * inverse)),
        int(round(overlay[2] * alpha + base[2] * inverse)),
        255,
    )
    return clamp_color(blended)

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


def rgba_hex(color: Color | tuple[int, int, int]) -> str:
    values = list(color[:4])
    while len(values) < 4:
        values.append(255)
    r, g, b, a = (max(0, min(255, int(round(v)))) for v in values)
    return f"#{r:02X}{g:02X}{b:02X}{a:02X}"


def parse_hex_color(text: str) -> Color | None:
    """Parse ``#RRGGBB`` / ``#RRGGBBAA`` (``#`` optional) into an RGBA colour.

    Returns ``None`` when the text is not a valid hex colour.
    """
    cleaned = str(text or "").strip().lstrip("#").strip()
    if len(cleaned) not in (6, 8):
        return None
    try:
        values = [int(cleaned[i : i + 2], 16) for i in range(0, len(cleaned), 2)]
    except ValueError:
        return None
    if len(values) == 3:
        values.append(255)
    return clamp_color(tuple(values))


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

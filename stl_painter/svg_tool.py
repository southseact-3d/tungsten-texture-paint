from __future__ import annotations

from io import BytesIO
from pathlib import Path

from PIL import Image


def load_svg_as_image(path: str | Path, size: tuple[int, int]) -> Image.Image | None:
    try:
        import cairosvg  # type: ignore
    except Exception:
        return None
    png_bytes = cairosvg.svg2png(url=str(path), output_width=max(8, int(size[0])), output_height=max(8, int(size[1])))
    return Image.open(BytesIO(png_bytes)).convert("RGBA")

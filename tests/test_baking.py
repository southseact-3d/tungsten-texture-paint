from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from stl_painter.sketch_tool import bake_sketch_to_faces


def test_bake_sketch_projects_colour_onto_face(square_mesh) -> None:
    image = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((60, 60, 140, 140), fill=(255, 0, 0, 255))

    baked = bake_sketch_to_faces(
        square_mesh, image, np.eye(4, dtype=np.float32), (200, 200)
    )

    assert baked
    assert all(colour[0] > 200 for colour in baked.values())


def test_bake_sketch_ignores_empty_overlay(square_mesh) -> None:
    image = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
    baked = bake_sketch_to_faces(
        square_mesh, image, np.eye(4, dtype=np.float32), (200, 200)
    )
    assert baked == {}


def test_bake_sketch_with_circle(square_mesh) -> None:
    image = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((50, 50, 150, 150), fill=(0, 255, 0, 255))

    baked = bake_sketch_to_faces(
        square_mesh, image, np.eye(4, dtype=np.float32), (200, 200)
    )

    assert baked


def test_bake_sketch_with_line(square_mesh) -> None:
    image = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.line((0, 0, 200, 200), fill=(0, 0, 255, 255), width=5)

    baked = bake_sketch_to_faces(
        square_mesh, image, np.eye(4, dtype=np.float32), (200, 200)
    )

    assert baked


def test_bake_sketch_blends_with_existing_colour(square_mesh) -> None:
    square_mesh.set_face_colour(0, (100, 100, 100, 255))
    image = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((50, 50, 150, 150), fill=(255, 0, 0, 128))

    baked = bake_sketch_to_faces(
        square_mesh, image, np.eye(4, dtype=np.float32), (200, 200)
    )

    if 0 in baked:
        assert baked[0][0] > 100

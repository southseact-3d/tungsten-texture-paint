from __future__ import annotations

import argparse
import logging
from pathlib import Path

import moderngl

from .camera import OrbitCamera
from .importer import load_stl
from .renderer import MeshRenderer

logger = logging.getLogger(__name__)


def run_stl_self_test(stl_path: str | Path) -> None:
    """Load an STL and render one frame off-screen to validate runtime dependencies."""
    mesh = load_stl(stl_path)
    camera = OrbitCamera.for_mesh(mesh.vertices)

    last_error: Exception | None = None
    ctx: moderngl.Context | None = None
    for backend in ("wgl", None):
        try:
            if backend is None:
                ctx = moderngl.create_standalone_context()
            else:
                ctx = moderngl.create_standalone_context(backend=backend)
            logger.info("Self-test OpenGL context backend=%s", backend or "default")
            break
        except Exception as exc:
            last_error = exc
            logger.warning(
                "Self-test context backend=%s failed: %s", backend or "default", exc
            )

    if ctx is None:
        raise RuntimeError(f"Could not create OpenGL context for self-test: {last_error}")

    renderer = MeshRenderer(ctx, mesh, (960, 720))
    snapshot = renderer.render(camera)
    if snapshot.rgba.shape != (720, 960, 4):
        raise RuntimeError(f"Unexpected render shape: {snapshot.rgba.shape}")

    # A fully black frame usually indicates a failed render path.
    rgb_max = int(snapshot.rgba[:, :, :3].max())
    if rgb_max == 0:
        raise RuntimeError("Self-test rendered a black frame")

    print(
        f"SELF-TEST PASS | faces={mesh.face_count} | vertices={mesh.vertex_count} | max_rgb={rgb_max}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--self-test-stl",
        dest="self_test_stl",
        type=str,
        help="Run non-GUI STL import/render self-test with the given STL path",
    )
    return parser

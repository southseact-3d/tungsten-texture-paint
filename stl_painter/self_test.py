from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .camera import OrbitCamera
from .importer import load_stl
from .logging_utils import configure_logging, log_file_path
from .renderer import MeshRenderer

logger = logging.getLogger(__name__)


def run_stl_self_test(stl_path: str | Path) -> None:
    """Load an STL and render one frame off-screen to validate runtime dependencies."""
    configure_logging()
    logger.info("Starting STL self-test | path=%s | log=%s", stl_path, log_file_path())
    mesh = load_stl(stl_path)
    camera = OrbitCamera.for_mesh(mesh.vertices)
    renderer = MeshRenderer(None, mesh, (960, 720))
    snapshot = renderer.render(camera)
    if snapshot.rgba.shape != (720, 960, 4):
        raise RuntimeError(f"Unexpected render shape: {snapshot.rgba.shape}")

    # A fully black frame usually indicates a failed render path.
    rgb_max = int(snapshot.rgba[:, :, :3].max())
    if rgb_max == 0:
        raise RuntimeError("Self-test rendered a black frame")
    picked_face = renderer.pick_face(camera, 480, 360)
    logger.info("Self-test pick result at viewport center: %s", picked_face)

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

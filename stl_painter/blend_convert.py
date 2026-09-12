"""Convert ``.blend`` files to ``.glb`` via headless Blender.

``trimesh`` cannot parse ``.blend`` (a version-fragile DNA memory dump), so a
local Blender install is used for conversion and the existing glTF import path
(including texture baking in :mod:`stl_painter.mesh_model`) handles colours.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)


class BlenderNotFoundError(FileNotFoundError):
    """Raised when no Blender executable can be located for ``.blend`` import."""


def _candidate_blender_paths() -> list[str]:
    candidates: list[str] = []
    for version_dir in ("Blender 5.2", "Blender 4.5", "Blender 4.2", "Blender 4.1"):
        candidates.append(
            str(
                Path("C:/Program Files/Blender Foundation") / version_dir / "blender.exe"
            )
        )
    candidates.append(str(Path.home() / "AppData/Local/Blender/blender.exe"))
    return candidates


def find_blender() -> str:
    """Return a Blender executable path or raise :class:`BlenderNotFoundError`."""
    on_path = shutil.which("blender") or shutil.which("blender.exe")
    if on_path:
        return on_path
    for candidate in _candidate_blender_paths():
        if Path(candidate).is_file():
            return candidate
    raise BlenderNotFoundError(
        "Blender was not found. Install Blender 4.x/5.x (or add it to PATH), "
        "or export the model from Blender as .glb (File > Export > glTF) with "
        "materials and vertex colors enabled and open the .glb instead."
    )


def _converter_script() -> str:
    # When frozen with PyInstaller, package modules live in the PYZ archive so
    # __file__ does not point at a real .py file; the script is bundled as
    # data under sys._MEIPASS/stl_painter/blend_to_glb.py (see .spec files).
    if getattr(sys, "frozen", False):
        bundled = Path(getattr(sys, "_MEIPASS", "")) / "stl_painter" / "blend_to_glb.py"
        if bundled.is_file():
            return str(bundled)
    return str(Path(__file__).with_name("blend_to_glb.py"))


@contextmanager
def convert_blend_to_glb(
    blend_path: str | Path, *, timeout: int = 300
) -> Iterator[str]:
    """Convert ``blend_path`` to a temporary ``.glb`` and yield its path."""
    blender = find_blender()
    source = str(blend_path)
    with tempfile.TemporaryDirectory(prefix="stl_painter_blend_") as tmpdir:
        output = str(Path(tmpdir) / (Path(source).stem + ".glb"))
        command = [
            blender,
            "--background",
            source,
            "--python",
            _converter_script(),
            "--",
            output,
        ]
        logger.info("Converting .blend via Blender: %s", " ".join(command))
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(
                f"Blender conversion timed out after {timeout}s for {source}"
            ) from exc
        if completed.returncode != 0 or not Path(output).is_file():
            tail = (completed.stderr or completed.stdout or "")[-2000:]
            raise RuntimeError(f"Blender conversion failed for {source}: {tail}")
        yield output

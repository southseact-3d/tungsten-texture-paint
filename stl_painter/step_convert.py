"""Convert ``.step``/``.stp`` files via an external CadQuery Python.

CadQuery (OCP) is **not** bundled with the app or the frozen ``.exe`` - it is
large (~200MB) and version-sensitive. Instead the app shells out to a system
Python that already has CadQuery installed (same isolation model as the
``.blend`` import in :mod:`stl_painter.blend_convert`), which tessellates each
CAD face into :mod:`stl_painter.step_to_mesh` ``.npz`` output.

Set ``STL_PAINTER_CADQUERY_PYTHON`` to point at the CadQuery interpreter
explicitly when auto-detection picks the wrong one.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

STEP_TESSELLATION_TOLERANCE = 0.1
STEP_CONVERT_TIMEOUT = 300


class CadQueryNotFoundError(FileNotFoundError):
    """Raised when no Python with CadQuery can be located for STEP import."""


def _candidate_interpreters() -> list[str]:
    candidates: list[str] = []
    explicit = os.environ.get("STL_PAINTER_CADQUERY_PYTHON", "").strip()
    if explicit:
        candidates.append(explicit)
    # NOTE: never probe sys.executable when frozen - the frozen exe is not a
    # Python interpreter and launching it with "-c" would open a second GUI.
    if not getattr(sys, "frozen", False):
        candidates.append(sys.executable)
    for name in ("python", "python3"):
        found = shutil.which(name)
        if found:
            candidates.append(found)
    seen: set[str] = set()
    unique: list[str] = []
    for candidate in candidates:
        key = os.path.normcase(os.path.abspath(candidate))
        if candidate and key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def _has_cadquery(python_exe: str) -> bool:
    try:
        completed = subprocess.run(
            [python_exe, "-c", "import cadquery"],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except Exception:
        return False
    return completed.returncode == 0


def find_cadquery_python() -> str:
    """Return a Python interpreter path with CadQuery importable."""
    for candidate in _candidate_interpreters():
        if not candidate or not Path(candidate).is_file():
            continue
        if _has_cadquery(candidate):
            return candidate
    raise CadQueryNotFoundError(
        "No Python with CadQuery was found for STEP import. Install CadQuery "
        "(pip install cadquery) into your Python, or set STL_PAINTER_CADQUERY_PYTHON "
        "to the interpreter path - or export the model from your CAD tool as "
        ".stl/.glb and open that instead."
    )


def _worker_script() -> str:
    # Same frozen-data trick as blend_convert: the worker rides along in the
    # PyInstaller bundle under sys._MEIPASS/stl_painter/step_to_mesh.py.
    if getattr(sys, "frozen", False):
        bundled = Path(getattr(sys, "_MEIPASS", "")) / "stl_painter" / "step_to_mesh.py"
        if bundled.is_file():
            return str(bundled)
    return str(Path(__file__).with_name("step_to_mesh.py"))


@contextmanager
def convert_step_to_npz(
    step_path: str | Path,
    *,
    tolerance: float = STEP_TESSELLATION_TOLERANCE,
    timeout: int = STEP_CONVERT_TIMEOUT,
) -> Iterator[str]:
    """Tessellate ``step_path`` per CAD face and yield a temp ``.npz`` path."""
    interpreter = find_cadquery_python()
    source = str(step_path)
    with tempfile.TemporaryDirectory(prefix="stl_painter_step_") as tmpdir:
        output = str(Path(tmpdir) / (Path(source).stem + ".npz"))
        command = [
            interpreter,
            _worker_script(),
            source,
            output,
            str(float(tolerance)),
        ]
        logger.info("Converting STEP via CadQuery: %s", " ".join(command))
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(
                f"STEP conversion timed out after {timeout}s for {source}"
            ) from exc
        if completed.returncode != 0 or not Path(output).is_file():
            tail = (completed.stderr or completed.stdout or "")[-2000:]
            raise RuntimeError(f"STEP conversion failed for {source}: {tail}")
        yield output

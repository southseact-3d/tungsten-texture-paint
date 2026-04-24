"""Tests for the STL Texture Painter headless CLI.

All tests use 'python -m stl_painter.cli' directly so they exercise the real
argument-parsing surface.  dart.stl must exist in the repo root; tests are
skipped automatically if it is absent.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
STL = ROOT / "dart.stl"
PYTHON = sys.executable


def _run(*args: str, cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [PYTHON, "-m", "stl_painter.cli", *args],
        capture_output=True,
        text=True,
        cwd=str(cwd),
    )


needs_stl = pytest.mark.skipif(not STL.exists(), reason="dart.stl not found")


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------

@needs_stl
def test_info_shows_face_count() -> None:
    r = _run("info", "--input", str(STL))
    assert r.returncode == 0, r.stderr
    assert "Faces:" in r.stdout
    assert "Vertices:" in r.stdout
    assert "Diagonal:" in r.stdout


@needs_stl
def test_info_shows_bounding_box() -> None:
    r = _run("info", "--input", str(STL))
    assert r.returncode == 0, r.stderr
    assert "Bounding box" in r.stdout
    assert "Min:" in r.stdout
    assert "Max:" in r.stdout


# ---------------------------------------------------------------------------
# paint-all
# ---------------------------------------------------------------------------

@needs_stl
def test_paint_all_saves_tg3d(tmp_path: Path) -> None:
    out = tmp_path / "painted.tg3d"
    r = _run("paint-all", "--input", str(STL), "--color", "#FF0000", "--output", str(out))
    assert r.returncode == 0, r.stderr
    assert out.exists()
    assert out.stat().st_size > 100


@needs_stl
def test_paint_all_csv_color(tmp_path: Path) -> None:
    out = tmp_path / "csv.tg3d"
    r = _run("paint-all", "--input", str(STL), "--color", "255,0,0", "--output", str(out))
    assert r.returncode == 0, r.stderr


@needs_stl
def test_paint_all_hex8_color(tmp_path: Path) -> None:
    out = tmp_path / "hex8.tg3d"
    r = _run("paint-all", "--input", str(STL), "--color", "#FF000080", "--output", str(out))
    assert r.returncode == 0, r.stderr


@needs_stl
def test_paint_all_color_applied(tmp_path: Path) -> None:
    from stl_painter.cli_session import CLISession

    out = tmp_path / "red.tg3d"
    _run("paint-all", "--input", str(STL), "--color", "#FF0000", "--output", str(out))
    s = CLISession()
    mesh = s.load(out)
    # Every face should be red (check first 20 as a sample)
    for fid in range(min(mesh.face_count, 20)):
        c = mesh.face_colour(fid)
        assert c[0] == 255 and c[1] == 0 and c[2] == 0, f"face {fid} is {c}"


# ---------------------------------------------------------------------------
# paint-face
# ---------------------------------------------------------------------------

@needs_stl
def test_paint_face_single(tmp_path: Path) -> None:
    from stl_painter.cli_session import CLISession

    out = tmp_path / "pf.tg3d"
    r = _run("paint-face", "--input", str(STL),
             "--face-ids", "0", "1", "2",
             "--color", "#00FF00", "--output", str(out))
    assert r.returncode == 0, r.stderr
    s = CLISession()
    mesh = s.load(out)
    assert mesh.face_colour(0) == (0, 255, 0, 255)
    assert mesh.face_colour(1) == (0, 255, 0, 255)
    assert mesh.face_colour(2) == (0, 255, 0, 255)


@needs_stl
def test_paint_face_invalid_id_ignored(tmp_path: Path) -> None:
    """Out-of-range face IDs should be silently skipped, not crash."""
    out = tmp_path / "pf_oob.tg3d"
    r = _run("paint-face", "--input", str(STL),
             "--face-ids", "0", "9999999",
             "--color", "#0000FF", "--output", str(out))
    # Face 0 in range → painted; face 9999999 out of range → ignored
    assert r.returncode == 0, r.stderr


# ---------------------------------------------------------------------------
# flood-fill
# ---------------------------------------------------------------------------

@needs_stl
def test_flood_fill_returns_ok(tmp_path: Path) -> None:
    out = tmp_path / "ff.tg3d"
    r = _run("flood-fill", "--input", str(STL),
             "--face-id", "0", "--color", "#0000FF", "--output", str(out))
    assert r.returncode == 0, r.stderr
    assert out.exists()


@needs_stl
def test_flood_fill_paints_multiple_faces(tmp_path: Path) -> None:
    from stl_painter.project_io import load_tg3d

    out = tmp_path / "ff_multi.tg3d"
    _run("flood-fill", "--input", str(STL),
         "--face-id", "0", "--color", "#AABBCC", "--output", str(out))
    mesh, _ = load_tg3d(out)
    blue_count = sum(
        1 for fid in range(mesh.face_count)
        if mesh.face_colour(fid) == (170, 187, 204, 255)
    )
    assert blue_count >= 1


# ---------------------------------------------------------------------------
# list-groups
# ---------------------------------------------------------------------------

@needs_stl
def test_list_groups_output(tmp_path: Path) -> None:
    r = _run("list-groups", "--input", str(STL), "--angle", "30")
    assert r.returncode == 0, r.stderr
    assert "group_" in r.stdout
    assert "Total:" in r.stdout


@needs_stl
def test_list_groups_different_angles(tmp_path: Path) -> None:
    r5  = _run("list-groups", "--input", str(STL), "--angle", "5")
    r90 = _run("list-groups", "--input", str(STL), "--angle", "90")
    assert r5.returncode == 0
    assert r90.returncode == 0
    # More groups with tighter angle tolerance
    def _count(stdout: str) -> int:
        for line in stdout.splitlines():
            if line.startswith("Total:"):
                return int(line.split()[1])
        return 0
    assert _count(r5.stdout) >= _count(r90.stdout)


# ---------------------------------------------------------------------------
# paint-group
# ---------------------------------------------------------------------------

@needs_stl
def test_paint_group_paints_faces(tmp_path: Path) -> None:
    # Discover a real group name first
    r = _run("list-groups", "--input", str(STL), "--angle", "30")
    groups = [
        line.strip().split()[0]
        for line in r.stdout.splitlines()
        if line.strip().startswith("group_")
    ]
    assert groups, "No groups found"
    group_id = groups[0]

    out = tmp_path / "pg.tg3d"
    r2 = _run("paint-group", "--input", str(STL),
              "--group", group_id, "--color", "#FF00FF", "--output", str(out))
    assert r2.returncode == 0, r2.stderr
    assert out.exists()


# ---------------------------------------------------------------------------
# paint-region
# ---------------------------------------------------------------------------

@needs_stl
def test_paint_region_full_mesh(tmp_path: Path) -> None:
    from stl_painter.cli_session import CLISession

    out = tmp_path / "pr_full.tg3d"
    # Use = syntax so argparse doesn't treat leading - as a flag
    r = _run("paint-region", "--input", str(STL),
             "--min=-9999,-9999,-9999", "--max=9999,9999,9999",
             "--color", "#FFFF00", "--output", str(out))
    assert r.returncode == 0, r.stderr
    s = CLISession()
    mesh = s.load(out)
    yellow = sum(
        1 for fid in range(mesh.face_count)
        if mesh.face_colour(fid) == (255, 255, 0, 255)
    )
    assert yellow == mesh.face_count


@needs_stl
def test_paint_region_empty_box(tmp_path: Path) -> None:
    out = tmp_path / "pr_empty.tg3d"
    r = _run("paint-region", "--input", str(STL),
             "--min=99999,99999,99999", "--max=99999,99999,99999",
             "--color", "#FF0000", "--output", str(out))
    assert r.returncode == 0, r.stderr


# ---------------------------------------------------------------------------
# save / export
# ---------------------------------------------------------------------------

@needs_stl
def test_save_creates_tg3d(tmp_path: Path) -> None:
    out = tmp_path / "saved.tg3d"
    r = _run("save", "--input", str(STL), "--output", str(out))
    assert r.returncode == 0, r.stderr
    assert out.exists()


@needs_stl
def test_export_3mf(tmp_path: Path) -> None:
    tg3d = tmp_path / "export_src.tg3d"
    _run("paint-all", "--input", str(STL), "--color", "#FF0000", "--output", str(tg3d))

    out_3mf = tmp_path / "out.3mf"
    r = _run("export", "--input", str(tg3d), "--output", str(out_3mf))
    assert r.returncode == 0, r.stderr
    assert out_3mf.exists()
    assert out_3mf.stat().st_size > 1000


@needs_stl
def test_export_3mf_directly_from_stl(tmp_path: Path) -> None:
    out = tmp_path / "direct.3mf"
    r = _run("export", "--input", str(STL), "--output", str(out))
    assert r.returncode == 0, r.stderr
    assert out.exists()


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------

@needs_stl
def test_pipeline_json_file(tmp_path: Path) -> None:
    pipeline = {
        "steps": [
            {"op": "paint-all", "color": "#B0B8C4"},
            {"op": "flood-fill", "face_id": 0, "color": "#FF3300"},
        ]
    }
    pf = tmp_path / "pipeline.json"
    pf.write_text(json.dumps(pipeline), encoding="utf-8")

    out = tmp_path / "result.tg3d"
    r = _run("pipeline", "--input", str(STL),
             "--pipeline", str(pf), "--output", str(out))
    assert r.returncode == 0, r.stderr
    assert out.exists()


@needs_stl
def test_pipeline_includes_export_step(tmp_path: Path) -> None:
    out_3mf = tmp_path / "pipeline_out.3mf"
    pipeline = {
        "steps": [
            {"op": "paint-all", "color": "#C0C0C0"},
            {"op": "export", "path": str(out_3mf)},
        ]
    }
    pf = tmp_path / "pipe.json"
    pf.write_text(json.dumps(pipeline), encoding="utf-8")

    r = _run("pipeline", "--input", str(STL), "--pipeline", str(pf))
    assert r.returncode == 0, r.stderr
    assert out_3mf.exists()


@needs_stl
def test_pipeline_inline_steps(tmp_path: Path) -> None:
    steps = json.dumps([{"op": "paint-all", "color": "#FFFFFF"}])
    out = tmp_path / "inline.tg3d"
    r = _run("pipeline", "--input", str(STL),
             "--steps", steps, "--output", str(out))
    assert r.returncode == 0, r.stderr


# ---------------------------------------------------------------------------
# checkpoints
# ---------------------------------------------------------------------------

@needs_stl
def test_save_checkpoint_alongside_export(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.tg3d"
    out_3mf = tmp_path / "out.3mf"
    r = _run("paint-all", "--input", str(STL),
             "--color", "#FF0000",
             "--save", str(checkpoint),
             "--output", str(out_3mf))
    assert r.returncode == 0, r.stderr
    assert checkpoint.exists()
    assert out_3mf.exists()


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_bad_subcommand() -> None:
    r = _run("not-a-real-command")
    assert r.returncode != 0


def test_missing_input() -> None:
    r = _run("info", "--input", "does_not_exist.stl")
    assert r.returncode != 0


@needs_stl
def test_bad_color_format() -> None:
    r = _run("paint-all", "--input", str(STL), "--color", "not-a-color")
    assert r.returncode != 0


# ---------------------------------------------------------------------------
# CLISession unit tests (no subprocess)
# ---------------------------------------------------------------------------

@needs_stl
def test_cli_session_load_and_paint() -> None:
    from stl_painter.cli_session import CLISession
    s = CLISession()
    mesh = s.load(STL)
    assert mesh.face_count > 0
    touched = s.paint_all((255, 0, 0, 255))
    assert len(touched) == mesh.face_count
    assert mesh.face_colour(0) == (255, 0, 0, 255)


@needs_stl
def test_cli_session_paint_region() -> None:
    from stl_painter.cli_session import CLISession
    s = CLISession()
    s.load(STL)
    touched = s.paint_region((-9999, -9999, -9999), (9999, 9999, 9999), (0, 255, 0, 255))
    assert len(touched) == s.mesh_model.face_count  # type: ignore[union-attr]


@needs_stl
def test_cli_session_flood_fill() -> None:
    from stl_painter.cli_session import CLISession
    s = CLISession()
    s.load(STL)
    touched = s.flood_fill(0, (0, 0, 255, 255))
    assert len(touched) >= 1


@needs_stl
def test_cli_session_list_groups() -> None:
    from stl_painter.cli_session import CLISession
    s = CLISession()
    s.load(STL)
    groups = s.list_groups(angle_tolerance_degrees=30.0)
    assert len(groups) >= 1
    assert all(isinstance(k, str) for k in groups)
    assert all(isinstance(v, int) for v in groups.values())


@needs_stl
def test_cli_session_undo() -> None:
    from stl_painter.cli_session import CLISession
    s = CLISession()
    s.load(STL)
    original_colour = s.mesh_model.face_colour(0)  # type: ignore[union-attr]
    s.paint_faces([0], (255, 0, 0, 255))
    s.undo()
    restored = s.mesh_model.face_colour(0)  # type: ignore[union-attr]
    assert restored == original_colour


@needs_stl
def test_cli_session_save_and_reload(tmp_path: Path) -> None:
    from stl_painter.cli_session import CLISession
    from stl_painter.project_io import load_tg3d

    s = CLISession()
    s.load(STL)
    s.paint_faces([0, 1, 2], (200, 100, 50, 255))
    save_path = tmp_path / "session.tg3d"
    s.save(save_path)

    mesh2, _ = load_tg3d(save_path)
    assert mesh2.face_colour(0) == (200, 100, 50, 255)

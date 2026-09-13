"""
STL Texture Painter – Headless CLI
===================================

Entry-point:  ``tgpaint <subcommand> [options]``
or:           ``python -m stl_painter.cli <subcommand> [options]``
or:           ``python main.py <subcommand> [options]``

Sub-commands
------------
info          Print mesh statistics
paint-face    Paint specific face IDs with a colour
paint-all     Paint the entire mesh with one colour
flood-fill    Flood-fill connected faces from a seed face
paint-group   Paint all faces in a named face group
paint-region  Paint all faces whose centroid is inside an XYZ bounding box
list-groups   List auto-computed face groups
save          Convert / save a mesh as a .tg3d project
export        Export a .tg3d (or STL/OBJ) to 3MF / STL / OBJ / GLB
pipeline      Run a JSON-scripted sequence of operations

Colour formats
--------------
  #RRGGBB        e.g. #FF0000
  #RRGGBBAA      e.g. #FF000080
  R,G,B          e.g. 255,0,0
  R,G,B,A        e.g. 255,0,0,128
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .cli_session import CLISession
from .color_utils import Color, clamp_color


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_color(value: str) -> Color:
    v = value.strip()
    if v.startswith("#"):
        h = v.lstrip("#")
        if len(h) == 6:
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            return clamp_color((r, g, b, 255))
        if len(h) == 8:
            r, g, b, a = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), int(h[6:8], 16)
            return clamp_color((r, g, b, a))
    parts = [p.strip() for p in v.split(",")]
    try:
        if len(parts) == 3:
            return clamp_color((int(parts[0]), int(parts[1]), int(parts[2]), 255))
        if len(parts) == 4:
            return clamp_color((int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])))
    except ValueError:
        pass
    raise argparse.ArgumentTypeError(
        f"Invalid color '{value}'. Use #RRGGBB, #RRGGBBAA, or R,G,B[,A]"
    )


def _parse_xyz(value: str) -> tuple[float, float, float]:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"Expected x,y,z but got '{value}'")
    return (float(parts[0]), float(parts[1]), float(parts[2]))


def _finish(session: CLISession, args: argparse.Namespace, summary: str = "") -> None:
    """Post-operation: optionally save checkpoint and/or export."""
    if summary:
        print(summary)
    save_path: str | None = getattr(args, "save", None)
    output_path: str | None = getattr(args, "output", None)
    if save_path:
        session.save(save_path)
        print(f"Checkpoint saved: {save_path}")
    if output_path:
        ext = Path(output_path).suffix.lower()
        if ext == ".tg3d":
            session.save(output_path)
            print(f"Project saved: {output_path}")
        else:
            issues = session.export(output_path)
            for issue in issues:
                if issue != "OK - mesh is valid for export.":
                    print(f"  Warning: {issue}")
            print(f"Exported: {output_path}")


# ---------------------------------------------------------------------------
# Sub-command implementations
# ---------------------------------------------------------------------------

def cmd_info(args: argparse.Namespace) -> None:
    session = CLISession()
    mesh = session.load(args.input)
    bb_min = mesh.vertices.min(axis=0)
    bb_max = mesh.vertices.max(axis=0)
    extents = bb_max - bb_min
    colored = len(mesh.face_colours)
    print(f"Input:      {args.input}")
    print(f"Faces:      {mesh.face_count:,}")
    print(f"Vertices:   {mesh.vertex_count:,}")
    print("Bounding box:")
    print(f"  Min:      ({bb_min[0]:.4f}, {bb_min[1]:.4f}, {bb_min[2]:.4f}) mm")
    print(f"  Max:      ({bb_max[0]:.4f}, {bb_max[1]:.4f}, {bb_max[2]:.4f}) mm")
    print(f"  Extents:  ({extents[0]:.4f}, {extents[1]:.4f}, {extents[2]:.4f}) mm")
    print(f"Diagonal:   {mesh.mesh_diagonal():.4f} mm")
    print(f"Painted:    {colored:,} / {mesh.face_count:,} faces")


def cmd_paint_face(args: argparse.Namespace) -> None:
    session = CLISession()
    session.load(args.input)
    colour = _parse_color(args.color)
    touched = session.paint_faces(list(args.face_ids), colour)
    _finish(session, args, f"Painted {len(touched)} face(s) with {args.color}")


def cmd_paint_all(args: argparse.Namespace) -> None:
    session = CLISession()
    session.load(args.input)
    colour = _parse_color(args.color)
    touched = session.paint_all(colour)
    _finish(session, args, f"Painted all {len(touched):,} faces with {args.color}")


def cmd_flood_fill(args: argparse.Namespace) -> None:
    session = CLISession()
    session.load(args.input)
    colour = _parse_color(args.color)
    touched = session.flood_fill(args.face_id, colour)
    _finish(
        session, args,
        f"Flood-filled {len(touched):,} face(s) from face {args.face_id} with {args.color}"
    )


def cmd_paint_group(args: argparse.Namespace) -> None:
    session = CLISession()
    session.load(args.input)
    colour = _parse_color(args.color)
    angle = getattr(args, "angle", 30.0)
    touched = session.paint_group(args.group, colour, angle_tolerance_degrees=angle)
    _finish(session, args, f"Painted group '{args.group}': {len(touched):,} face(s) with {args.color}")


def cmd_paint_region(args: argparse.Namespace) -> None:
    session = CLISession()
    session.load(args.input)
    colour = _parse_color(args.color)
    min_xyz = _parse_xyz(args.min)
    max_xyz = _parse_xyz(args.max)
    touched = session.paint_region(min_xyz, max_xyz, colour)
    _finish(session, args, f"Painted {len(touched):,} face(s) in region with {args.color}")


def cmd_list_groups(args: argparse.Namespace) -> None:
    session = CLISession()
    session.load(args.input)
    angle = getattr(args, "angle", 30.0)
    groups = session.list_groups(angle_tolerance_degrees=angle)

    def _sort_key(gid: str) -> int:
        tail = gid.split("_")[-1]
        return int(tail) if tail.isdigit() else 0

    print(f"{'Group':32s}  {'Faces':>8}")
    print("-" * 44)
    for gid in sorted(groups, key=_sort_key):
        print(f"{gid:32s}  {groups[gid]:>8,}")
    print(f"\nTotal: {len(groups)} groups, {sum(groups.values()):,} faces")


def cmd_save(args: argparse.Namespace) -> None:
    session = CLISession()
    session.load(args.input)
    out = args.output
    session.save(out)
    print(f"Saved: {out}")


def cmd_export(args: argparse.Namespace) -> None:
    session = CLISession()
    session.load(args.input)
    issues = session.export(args.output)
    for issue in issues:
        if issue != "OK - mesh is valid for export.":
            print(f"  Warning: {issue}")
    print(f"Exported: {args.output}")


def cmd_screenshot(args: argparse.Namespace) -> None:
    session = CLISession()
    session.load(args.input)
    session.screenshot(
        args.output,
        azimuth=args.azimuth,
        elevation=args.elevation,
        width=args.width,
        height=args.height,
    )
    print(f"Screenshot saved: {args.output}")

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _run_pipeline_step(session: CLISession, step: dict[str, Any]) -> str:
    op = step.get("op", "")
    if op == "paint-all":
        colour = _parse_color(str(step["color"]))
        touched = session.paint_all(colour)
        return f"paint-all  -> {len(touched):,} faces painted {step['color']}"
    if op == "paint-face":
        colour = _parse_color(str(step["color"]))
        face_ids = [int(f) for f in step["face_ids"]]
        touched = session.paint_faces(face_ids, colour)
        return f"paint-face -> {len(touched):,} face(s) painted"
    if op == "flood-fill":
        colour = _parse_color(str(step["color"]))
        touched = session.flood_fill(int(step["face_id"]), colour)
        return f"flood-fill -> {len(touched):,} faces from face {step['face_id']}"
    if op == "paint-group":
        colour = _parse_color(str(step["color"]))
        angle = float(step.get("angle", 30.0))
        touched = session.paint_group(str(step["group"]), colour, angle_tolerance_degrees=angle)
        return f"paint-group '{step['group']}' -> {len(touched):,} faces"
    if op == "paint-region":
        colour = _parse_color(str(step["color"]))
        mn = tuple(float(v) for v in step["min"])
        mx = tuple(float(v) for v in step["max"])
        strict = bool(step.get("strict", False))
        touched = session.paint_region(mn, mx, colour, strict=strict)  # type: ignore[arg-type]
        return f"paint-region -> {len(touched):,} faces"
    if op == "save":
        session.save(str(step["path"]))
        return f"save -> {step['path']}"
    if op == "export":
        issues = session.export(str(step["path"]))
        ok = all(i == "OK - mesh is valid for export." for i in issues)
        return f"export -> {step['path']}  {'OK' if ok else 'warnings'}"
    if op == "screenshot":
        session.screenshot(
            str(step["path"]),
            azimuth=float(step.get("azimuth", 45.0)),
            elevation=float(step.get("elevation", 45.0)),
            width=int(step.get("width", 800)),
            height=int(step.get("height", 600)),
        )
        return f"screenshot -> {step['path']}"
    raise ValueError(f"Unknown pipeline op: '{op}'. Valid ops: paint-all, paint-face, flood-fill, paint-group, paint-region, save, export, screenshot")


def cmd_pipeline(args: argparse.Namespace) -> None:
    pipeline_json: str | None = getattr(args, "pipeline", None)
    steps_inline: str | None = getattr(args, "steps", None)

    if pipeline_json:
        payload = json.loads(Path(pipeline_json).read_text(encoding="utf-8"))
    elif steps_inline:
        data = json.loads(steps_inline)
        payload = {"steps": data if isinstance(data, list) else [data]}
    else:
        print("Error: provide --pipeline FILE.json or --steps '[...]'", file=sys.stderr)
        sys.exit(1)

    steps: list[dict[str, Any]] = payload.get("steps", [])
    input_path: str | None = getattr(args, "input", None) or payload.get("input")
    if not input_path:
        print("Error: --input is required", file=sys.stderr)
        sys.exit(1)

    session = CLISession()
    mesh = session.load(input_path)
    print(f"Loaded: {input_path}  ({mesh.face_count:,} faces)")

    for i, step in enumerate(steps, 1):
        try:
            result = _run_pipeline_step(session, step)
            print(f"  [{i}/{len(steps)}] {result}")
        except Exception as exc:
            print(f"  [{i}/{len(steps)}] ERROR in op '{step.get('op', '?')}': {exc}", file=sys.stderr)
            sys.exit(1)

    output_path: str | None = getattr(args, "output", None) or payload.get("output")
    if output_path:
        ext = Path(output_path).suffix.lower()
        if ext == ".tg3d":
            session.save(output_path)
            print(f"Project saved: {output_path}")
        else:
            issues = session.export(output_path)
            ok = all(i == "OK - mesh is valid for export." for i in issues)
            print(f"Exported: {output_path}  {'OK' if ok else repr(issues)}")

    save_path: str | None = getattr(args, "save", None)
    if save_path:
        session.save(save_path)
        print(f"Checkpoint saved: {save_path}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _add_io(
    p: argparse.ArgumentParser,
    *,
    output: bool = False,
    output_required: bool = False,
    checkpoint: bool = True,
) -> None:
    p.add_argument(
        "--input", "-i",
        required=True,
        metavar="PATH",
        help="Input file: STL / OBJ / GLB / GLTF / 3MF / PLY / FBX / BLEND / STEP / STP / .tg3d",
    )
    if output:
        p.add_argument(
            "--output", "-o",
            required=output_required,
            metavar="PATH",
            help="Output path: .3mf / .stl / .obj / .glb / .gltf / .ply / .tg3d",
        )
    if checkpoint:
        p.add_argument(
            "--save",
            metavar="PATH",
            help="Also save a .tg3d checkpoint after this operation",
        )


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tgpaint",
        description="STL Texture Painter – headless CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # info -------------------------------------------------------------------
    p_info = sub.add_parser("info", help="Print mesh info and statistics")
    _add_io(p_info, checkpoint=False)

    # paint-face -------------------------------------------------------------
    p_pf = sub.add_parser("paint-face", help="Paint specific face IDs")
    _add_io(p_pf, output=True)
    p_pf.add_argument(
        "--face-ids", "-f",
        nargs="+", type=int, required=True, metavar="ID",
        help="One or more face IDs to paint (0-based)",
    )
    p_pf.add_argument("--color", "-c", required=True, metavar="COLOR", help="Paint colour")

    # paint-all --------------------------------------------------------------
    p_pa = sub.add_parser("paint-all", help="Paint the entire mesh with one colour")
    _add_io(p_pa, output=True)
    p_pa.add_argument("--color", "-c", required=True, metavar="COLOR")

    # flood-fill -------------------------------------------------------------
    p_ff = sub.add_parser("flood-fill", help="Flood-fill connected faces from a seed face")
    _add_io(p_ff, output=True)
    p_ff.add_argument(
        "--face-id", "-f", type=int, required=True, metavar="ID",
        help="Seed face ID (0-based)",
    )
    p_ff.add_argument("--color", "-c", required=True, metavar="COLOR")

    # paint-group ------------------------------------------------------------
    p_pg = sub.add_parser("paint-group", help="Paint all faces in a face group")
    _add_io(p_pg, output=True)
    p_pg.add_argument(
        "--group", "-g", required=True, metavar="GROUP_ID",
        help="Group ID, e.g. group_0.  Use list-groups to discover IDs.",
    )
    p_pg.add_argument("--color", "-c", required=True, metavar="COLOR")
    p_pg.add_argument(
        "--angle", type=float, default=30.0, metavar="DEG",
        help="Angle tolerance for group auto-computation (default: 30°)",
    )

    # paint-region -----------------------------------------------------------
    p_pr = sub.add_parser(
        "paint-region",
        help="Paint all faces whose centroid is within an XYZ bounding box",
    )
    _add_io(p_pr, output=True)
    p_pr.add_argument(
        "--min", required=True, metavar="X,Y,Z",
        help="Minimum corner of bounding box, e.g. -5,-5,0",
    )
    p_pr.add_argument(
        "--max", required=True, metavar="X,Y,Z",
        help="Maximum corner of bounding box, e.g. 5,5,10",
    )
    p_pr.add_argument("--color", "-c", required=True, metavar="COLOR")

    # list-groups ------------------------------------------------------------
    p_lg = sub.add_parser("list-groups", help="List face groups (auto-computed if needed)")
    _add_io(p_lg, checkpoint=False)
    p_lg.add_argument(
        "--angle", type=float, default=30.0, metavar="DEG",
        help="Angle tolerance for group computation (default: 30°)",
    )

    # save -------------------------------------------------------------------
    p_save = sub.add_parser("save", help="Load a mesh and save it as a .tg3d project")
    p_save.add_argument("--input", "-i", required=True, metavar="PATH")
    p_save.add_argument("--output", "-o", required=True, metavar="PATH", help="Output .tg3d")

    # export -----------------------------------------------------------------
    p_exp = sub.add_parser("export", help="Export a .tg3d or STL to 3MF/STL/OBJ/etc.")
    p_exp.add_argument("--input", "-i", required=True, metavar="PATH")
    p_exp.add_argument("--output", "-o", required=True, metavar="PATH")

    # screenshot -------------------------------------------------------------
    p_ss = sub.add_parser("screenshot", help="Take a software-rendered screenshot of the mesh")
    p_ss.add_argument("--input", "-i", required=True, metavar="PATH")
    p_ss.add_argument("--output", "-o", required=True, metavar="PATH", help="Output PNG path")
    p_ss.add_argument("--azimuth", type=float, default=45.0, help="Camera azimuth angle")
    p_ss.add_argument("--elevation", type=float, default=45.0, help="Camera elevation angle")
    p_ss.add_argument("--width", type=int, default=800, help="Image width")
    p_ss.add_argument("--height", type=int, default=600, help="Image height")

    # pipeline ---------------------------------------------------------------
    p_pl = sub.add_parser(
        "pipeline",
        help="Run a multi-step JSON pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example pipeline JSON file:
  {
    "input": "dart.stl",
    "output": "dart_painted.3mf",
    "steps": [
      { "op": "paint-all",   "color": "#B0B8C4" },
      { "op": "flood-fill",  "face_id": 42, "color": "#FF3300" },
      { "op": "paint-group", "group": "group_3", "color": "#00AA00" },
      { "op": "export",      "path": "intermediate.3mf" }
    ]
  }
""",
    )
    p_pl.add_argument("--input", "-i", metavar="PATH",
                      help="Input STL/.tg3d (can also be set inside the JSON)")
    p_pl.add_argument("--output", "-o", metavar="PATH",
                      help="Output path after all steps (can also be set inside the JSON)")
    p_pl.add_argument("--save", metavar="PATH",
                      help="Save a .tg3d checkpoint after all steps")
    p_pl.add_argument("--pipeline", "-p", metavar="FILE.json",
                      help="JSON pipeline file path")
    p_pl.add_argument("--steps", "-s", metavar="JSON",
                      help="Inline JSON array of step objects")

    return parser


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_COMMANDS = {
    "info": cmd_info,
    "paint-face": cmd_paint_face,
    "paint-all": cmd_paint_all,
    "flood-fill": cmd_flood_fill,
    "paint-group": cmd_paint_group,
    "paint-region": cmd_paint_region,
    "list-groups": cmd_list_groups,
    "save": cmd_save,
    "export": cmd_export,
    "screenshot": cmd_screenshot,
    "pipeline": cmd_pipeline,
}


def main(argv: list[str] | None = None) -> None:
    parser = build_cli_parser()
    args = parser.parse_args(argv)
    handler = _COMMANDS.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)
    try:
        handler(args)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

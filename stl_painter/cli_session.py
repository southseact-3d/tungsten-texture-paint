"""
Headless painting session – no GUI or OpenGL dependency.

Used by the CLI and can be imported directly in Python scripts::

    from stl_painter.cli_session import CLISession
    s = CLISession()
    s.load("dart.stl")
    s.paint_all((255, 0, 0, 255))
    s.export("dart_painted.3mf")
"""
from __future__ import annotations

import numpy as np
from pathlib import Path

from .color_utils import Color, clamp_color
from .commands import AppCommands
from .importer import load_model
from .mesh_model import MeshModel
from .paint_tool import PaintTool
from .project_io import load_tg3d, save_tg3d
from .sketch_tool import SketchTool


class CLISession:
    """Headless painting session for CLI / scripting use."""

    def __init__(self) -> None:
        self.mesh_model: MeshModel | None = None
        self.paint_tool: PaintTool | None = None
        self.commands: AppCommands = AppCommands(None, None)
        self.sketch_tool: SketchTool = SketchTool()
        self._timeline: dict | None = None

    # ------------------------------------------------------------------
    # Loading / saving
    # ------------------------------------------------------------------

    def load(self, path: str | Path) -> MeshModel:
        """Load an STL/OBJ/GLB or .tg3d file into this session."""
        p = Path(path)
        if p.suffix.lower() == ".tg3d":
            base_model, timeline = load_tg3d(p)
            self._timeline = timeline
            # save() stores all face colours directly in the model block, so
            # base_model already has the correct current colours.
            self.mesh_model = base_model
        else:
            self.mesh_model = load_model(p)
            self._timeline = None
        self.paint_tool = PaintTool(self.mesh_model)
        self.commands.attach(self.mesh_model, self.paint_tool)
        return self.mesh_model

    def save(self, path: str | Path) -> None:
        """Save the current painted mesh as a .tg3d project file.

        Colours are stored directly in the model block (no delta compression)
        so any colour — including a dominant paint-all colour — round-trips
        correctly.  The GUI can open the file; the undo history is empty.
        """
        import json as _json
        mesh = self._require_mesh()
        model_dict = mesh.to_project_dict()
        payload: dict = {
            "tg3d_version": 2,
            "model": model_dict,
            "timeline": {
                "current_index": 0,
                "descriptions": ["Current state"],
                "snapshots": [],
            },
        }
        Path(path).write_text(_json.dumps(payload), encoding="utf-8")

    def export(self, path: str | Path) -> list[str]:
        """Export to 3MF, STL, OBJ, GLB, or PLY.  Returns validation messages."""
        from .exporter import export_model
        return export_model(Path(path), self._require_mesh())

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def _require_mesh(self) -> MeshModel:
        if self.mesh_model is None:
            raise RuntimeError("No mesh loaded – call load() first")
        return self.mesh_model

    def paint_faces(self, face_ids: list[int], colour: Color) -> list[int]:
        """Paint the given face IDs with *colour*.  Returns list of touched face IDs."""
        mesh = self._require_mesh()
        updates: dict[int, Color] = {
            int(fid): colour
            for fid in face_ids
            if 0 <= int(fid) < mesh.face_count
        }
        if not updates:
            return []
        return self.commands.paint_faces(updates)

    def paint_all(self, colour: Color) -> list[int]:
        """Paint every face in the mesh with *colour*."""
        mesh = self._require_mesh()
        return self.paint_faces(list(range(mesh.face_count)), colour)

    def flood_fill(self, start_face: int, colour: Color) -> list[int]:
        """Flood-fill connected faces of the same colour starting at *start_face*."""
        mesh = self._require_mesh()
        if not 0 <= start_face < mesh.face_count:
            raise ValueError(
                f"face_id {start_face} is out of range (mesh has {mesh.face_count} faces)"
            )
        assert self.paint_tool is not None
        updates = self.paint_tool.flood_fill_updates(start_face, colour)
        if not updates:
            return []
        return self.commands.paint_faces(updates)

    def paint_group(
        self,
        group_id: str,
        colour: Color,
        *,
        angle_tolerance_degrees: float = 30.0,
    ) -> list[int]:
        """Paint all faces belonging to a named face group."""
        mesh = self._require_mesh()
        if not mesh.face_groups:
            mesh.compute_face_groups(angle_tolerance_degrees)
        faces = mesh.faces_for_group(group_id)
        if not faces:
            available = sorted(mesh.face_groups.keys())
            raise ValueError(
                f"Group '{group_id}' not found or empty. "
                f"Available groups: {available[:10]}{'...' if len(available) > 10 else ''}"
            )
        return self.paint_faces(faces, colour)

    def paint_region(
        self,
        min_xyz: tuple[float, float, float],
        max_xyz: tuple[float, float, float],
        colour: Color,
    ) -> list[int]:
        """Paint all faces whose centroid falls within the given 3D bounding box."""
        mesh = self._require_mesh()
        centroids = mesh.vertices[mesh.faces].mean(axis=1)  # (F, 3)
        mn = np.asarray(min_xyz, dtype=np.float32)
        mx = np.asarray(max_xyz, dtype=np.float32)
        mask = ((centroids >= mn) & (centroids <= mx)).all(axis=1)
        face_ids = [int(i) for i in range(mesh.face_count) if mask[i]]
        return self.paint_faces(face_ids, colour)

    def list_groups(
        self, angle_tolerance_degrees: float = 30.0
    ) -> dict[str, int]:
        """Return a mapping of group_id → face count.  Computes groups if needed."""
        mesh = self._require_mesh()
        if not mesh.face_groups:
            mesh.compute_face_groups(angle_tolerance_degrees)
        return {gid: len(faces) for gid, faces in mesh.face_groups.items()}

    # ------------------------------------------------------------------
    # Undo / redo
    # ------------------------------------------------------------------

    def undo(self) -> list[int]:
        return self.commands.undo()

    def redo(self) -> list[int]:
        return self.commands.redo()

from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import trimesh

from .color_utils import DEFAULT_COLOR
from .mesh_model import MeshModel

logger = logging.getLogger(__name__)
SUPPORTED_IMPORT_EXTENSIONS = {".stl", ".obj", ".glb", ".gltf", ".3mf", ".ply", ".fbx"}


def _coerce_trimesh(loaded: trimesh.Trimesh | trimesh.Scene) -> trimesh.Trimesh:
    if isinstance(loaded, trimesh.Trimesh):
        return loaded
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError("The selected file did not contain any mesh geometry")
        combined = trimesh.util.concatenate(tuple(loaded.geometry.values()))
        if not isinstance(combined, trimesh.Trimesh):
            raise ValueError("Failed to combine scene geometry into a mesh")
        return combined
    raise ValueError(f"Expected a mesh, got {type(loaded)!r}")


def load_stl(path: str | Path) -> MeshModel:
    source_path = str(path)
    logger.info("Loading STL from %s", source_path)
    loaded = _coerce_trimesh(trimesh.load(source_path, force="mesh"))
    logger.info(
        "Raw STL stats | faces=%s | vertices=%s | watertight=%s | extents=%s",
        len(loaded.faces),
        len(loaded.vertices),
        loaded.is_watertight,
        tuple(float(value) for value in loaded.extents),
    )
    if len(loaded.faces) == 0 or len(loaded.vertices) == 0:
        raise ValueError("The selected STL did not contain any renderable triangles")
    logger.info(
        "Successfully loaded STL with %s faces and %s vertices",
        len(loaded.faces),
        len(loaded.vertices),
    )
    mesh_model = MeshModel.from_trimesh(loaded, source_path=source_path)
    logger.info(
        "Prepared mesh model | faces=%s | vertices=%s | source=%s",
        mesh_model.face_count,
        mesh_model.vertex_count,
        source_path,
    )
    return mesh_model


def import_stl(path: str | Path) -> MeshModel:
    return load_stl(path)


def _extract_3mf_colors(
    path: str | Path,
) -> dict[int, tuple[int, int, int, int]] | None:
    """Extract face colors from a 3MF file if present."""
    try:
        with zipfile.ZipFile(path, "r") as archive:
            # Find the 3D model file
            model_path = None
            for name in archive.namelist():
                if name.endswith("3dmodel.model") or name.endswith(".model"):
                    model_path = name
                    break
            if not model_path:
                return None

            xml_content = archive.read(model_path).decode("utf-8")
            root = ET.fromstring(xml_content)

            # Define namespaces
            ns = {
                "c": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02",
                "m": "http://schemas.microsoft.com/3dmanufacturing/material/2015/02",
            }

            # Parse color groups
            color_groups: dict[str, dict[int, tuple[int, int, int]]] = {}
            for colorgroup in root.findall(".//m:colorgroup", ns):
                group_id = colorgroup.get("id", "1")
                colors: dict[int, tuple[int, int, int]] = {}
                for idx, color in enumerate(colorgroup.findall("m:color", ns)):
                    color_str = color.get("color", "#FFFFFF")
                    # Parse hex color
                    if color_str.startswith("#"):
                        color_str = color_str[1:]
                    if len(color_str) == 6:
                        r = int(color_str[0:2], 16)
                        g = int(color_str[2:4], 16)
                        b = int(color_str[4:6], 16)
                    else:
                        r, g, b = 255, 255, 255
                    colors[idx] = (r, g, b)
                color_groups[group_id] = colors

            # Map triangle colors
            face_colors: dict[int, tuple[int, int, int, int]] = {}
            triangle_idx = 0
            for obj in root.findall(".//c:object", ns):
                mesh_elem = obj.find(".//c:mesh", ns)
                if mesh_elem is None:
                    continue
                triangles = mesh_elem.find(".//c:triangles", ns)
                if triangles is None:
                    continue
                for tri in triangles.findall("c:triangle", ns):
                    pid = tri.get("pid")
                    p1 = tri.get("p1")
                    if pid and p1 is not None:
                        try:
                            color_idx = int(p1)
                            if pid in color_groups and color_idx in color_groups[pid]:
                                rgb = color_groups[pid][color_idx]
                                face_colors[triangle_idx] = (*rgb, 255)
                        except (ValueError, KeyError):
                            pass
                    triangle_idx += 1

            return face_colors if face_colors else None
    except Exception as e:
        logger.debug("Could not extract 3MF colors: %s", e)
        return None


def load_model(path: str | Path) -> MeshModel:
    source_path = str(path)
    extension = Path(path).suffix.lower()
    if extension not in SUPPORTED_IMPORT_EXTENSIONS:
        raise ValueError(
            f"Unsupported import format '{extension}'. Supported: {sorted(SUPPORTED_IMPORT_EXTENSIONS)}"
        )
    logger.info("Loading model from %s", source_path)

    # Extract 3MF colors before loading if applicable
    pre_extracted_colors = None
    if extension == ".3mf":
        pre_extracted_colors = _extract_3mf_colors(source_path)

    # Load the mesh
    if extension == ".3mf":
        # 3MF needs special handling - load as a scene
        loaded = trimesh.load(source_path, file_type="3mf")
        mesh = _coerce_trimesh(loaded)
    else:
        mesh = _coerce_trimesh(trimesh.load(source_path, force="scene"))

    if len(mesh.faces) == 0 or len(mesh.vertices) == 0:
        raise ValueError("The selected model did not contain any renderable triangles")

    # Create the mesh model
    mesh_model = MeshModel.from_trimesh(mesh, source_path=source_path)

    # Apply pre-extracted colors from 3MF if available
    if pre_extracted_colors:
        for face_id, color in pre_extracted_colors.items():
            if face_id < mesh_model.face_count:
                mesh_model.face_colours[face_id] = color
        logger.info("Applied %d face colors from 3MF file", len(pre_extracted_colors))

    logger.info(
        "Prepared mesh model | faces=%s | vertices=%s | colors=%s | source=%s",
        mesh_model.face_count,
        mesh_model.vertex_count,
        len(mesh_model.face_colours),
        source_path,
    )
    return mesh_model

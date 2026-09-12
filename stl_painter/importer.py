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
SUPPORTED_IMPORT_EXTENSIONS = {
    ".stl",
    ".obj",
    ".glb",
    ".gltf",
    ".3mf",
    ".ply",
    ".fbx",
    ".blend",
}


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


def _parse_3mf_hex_color(value: str) -> tuple[int, int, int, int]:
    """Parse a 3MF sRGB hex color (``#RRGGBB`` or ``#RRGGBBAA``)."""
    text = value.strip()
    if text.startswith("#"):
        text = text[1:]
    try:
        if len(text) == 6:
            return (
                int(text[0:2], 16),
                int(text[2:4], 16),
                int(text[4:6], 16),
                255,
            )
        if len(text) == 8:
            return (
                int(text[0:2], 16),
                int(text[2:4], 16),
                int(text[4:6], 16),
                int(text[6:8], 16),
            )
    except ValueError:
        pass
    return (255, 255, 255, 255)


def _extract_3mf_colors(
    path: str | Path,
) -> dict[int, tuple[int, int, int, int]] | None:
    """Extract flat face colors (colorgroup / basematerials) from a 3MF file."""
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

            # Parse color groups (indexed in document order)
            color_groups: dict[str, list[tuple[int, int, int, int]]] = {}
            for colorgroup in root.findall(".//m:colorgroup", ns):
                group_id = colorgroup.get("id", "1")
                colors: list[tuple[int, int, int, int]] = []
                for color in colorgroup.findall("m:color", ns):
                    colors.append(
                        _parse_3mf_hex_color(color.get("color", "#FFFFFF"))
                    )
                color_groups[group_id] = colors

            # Parse base materials (per-face single material index in p1)
            base_materials: dict[str, list[tuple[int, int, int, int]]] = {}
            for group in root.findall(".//m:basematerials", ns):
                group_id = group.get("id", "1")
                colors = [
                    _parse_3mf_hex_color(item.get("displaycolor", "#FFFFFF"))
                    for item in group.findall("m:base", ns)
                ]
                base_materials[group_id] = colors

            # Map triangle colors. A triangle may carry up to three property
            # indices (p1/p2/p3) - bake them to one flat colour per face so a
            # painted face later overwrites the whole face (flat-wins rule).
            face_colors: dict[int, tuple[int, int, int, int]] = {}
            triangle_idx = 0
            for obj in root.findall(".//c:object", ns):
                mesh_elem = obj.find("c:mesh", ns)
                if mesh_elem is None:
                    continue
                triangles = mesh_elem.find("c:triangles", ns)
                if triangles is None:
                    continue
                for tri in triangles.findall("c:triangle", ns):
                    pid = tri.get("pid")
                    if pid:
                        try:
                            if pid in color_groups:
                                palette = color_groups[pid]
                                samples = []
                                for key in ("p1", "p2", "p3"):
                                    raw = tri.get(key)
                                    if raw is not None:
                                        idx = int(raw)
                                        if 0 <= idx < len(palette):
                                            samples.append(palette[idx])
                                if samples:
                                    n = len(samples)
                                    face_colors[triangle_idx] = (
                                        sum(c[0] for c in samples) // n,
                                        sum(c[1] for c in samples) // n,
                                        sum(c[2] for c in samples) // n,
                                        sum(c[3] for c in samples) // n,
                                    )
                            elif pid in base_materials:
                                palette = base_materials[pid]
                                raw = tri.get("p1")
                                if raw is not None:
                                    idx = int(raw)
                                    if 0 <= idx < len(palette):
                                        face_colors[triangle_idx] = palette[idx]
                        except (ValueError, KeyError):
                            pass
                    triangle_idx += 1

            return face_colors if face_colors else None
    except Exception as e:
        logger.debug("Could not extract 3MF colors: %s", e)
        return None


def _extract_3mf_texture_colors(
    path: str | Path,
) -> dict[int, tuple[int, int, int, int]] | None:
    """Bake 3MF ``texture2dgroup`` UV textures down to flat per-face colors.

    Each triangle references three tex-coord indices (p1/p2/p3) into a
    ``texture2dgroup``; the referenced ``texture2d`` PNG is sampled at those
    UVs and averaged to a single RGBA per face.
    """
    try:
        from PIL import Image

        with zipfile.ZipFile(path, "r") as archive:
            names = archive.namelist()
            model_path = next(
                (
                    name
                    for name in names
                    if name.endswith("3dmodel.model") or name.endswith(".model")
                ),
                None,
            )
            if not model_path:
                return None
            root = ET.fromstring(archive.read(model_path).decode("utf-8"))
            ns = {
                "c": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02",
                "m": "http://schemas.microsoft.com/3dmanufacturing/material/2015/02",
            }

            tex_groups = root.findall(".//m:texture2dgroup", ns)
            if not tex_groups:
                return None

            # texture2d id -> archive member name
            tex_paths: dict[str, str] = {}
            for tex in root.findall(".//m:texture2d", ns):
                tex_id = tex.get("id")
                raw_path = tex.get("path", "")
                if tex_id:
                    tex_paths[tex_id] = raw_path.lstrip("/")

            # texture2dgroup id -> (image array, uv array)
            group_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            images: dict[str, np.ndarray] = {}
            for group in tex_groups:
                group_id = group.get("id")
                tex_id = group.get("texid")
                if group_id is None or tex_id is None:
                    continue
                if tex_id not in images:
                    member = tex_paths.get(tex_id)
                    if member is None or member not in names:
                        continue
                    with Image.open(archive.open(member)) as img:
                        images[tex_id] = np.asarray(img.convert("RGBA"))
                uvs = np.array(
                    [
                        (float(coord.get("u", 0.0)), float(coord.get("v", 0.0)))
                        for coord in group.findall("m:tex2coord", ns)
                    ],
                    dtype=np.float64,
                )
                if len(uvs):
                    group_data[group_id] = (images[tex_id], uvs)
            if not group_data:
                return None

            face_colors: dict[int, tuple[int, int, int, int]] = {}
            triangle_idx = 0
            for obj in root.findall(".//c:object", ns):
                mesh_elem = obj.find("c:mesh", ns)
                if mesh_elem is None:
                    continue
                triangles = mesh_elem.find("c:triangles", ns)
                if triangles is None:
                    continue
                for tri in triangles.findall("c:triangle", ns):
                    pid = tri.get("pid")
                    if pid and pid in group_data:
                        try:
                            image, uvs = group_data[pid]
                            height, width = image.shape[0], image.shape[1]
                            idx = np.array(
                                [
                                    int(tri.get("p1", 0)),
                                    int(tri.get("p2", 0)),
                                    int(tri.get("p3", 0)),
                                ]
                            )
                            if (
                                idx.min() < 0
                                or idx.max() >= len(uvs)
                                or width == 0
                                or height == 0
                            ):
                                raise ValueError("UV index out of range")
                            # 3MF UV origin is bottom-left; image rows top-first.
                            xs = np.clip(
                                (uvs[idx, 0] * (width - 1)).round(), 0, width - 1
                            ).astype(int)
                            ys = np.clip(
                                ((1.0 - uvs[idx, 1]) * (height - 1)).round(),
                                0,
                                height - 1,
                            ).astype(int)
                            mean = image[ys, xs].astype(np.int32).mean(axis=0)
                            face_colors[triangle_idx] = (
                                int(mean[0]),
                                int(mean[1]),
                                int(mean[2]),
                                int(mean[3]),
                            )
                        except (ValueError, KeyError):
                            pass
                    triangle_idx += 1

            return face_colors if face_colors else None
    except Exception as e:
        logger.debug("Could not extract 3MF texture colors: %s", e)
        return None


def _apply_face_colors_to_trimesh(
    mesh: trimesh.Trimesh, colors: dict[int, tuple[int, int, int, int]]
) -> None:
    """Stamp pre-extracted per-face colors onto a trimesh *before* dedup.

    Colours travel with ``update_faces``/``merge_vertices`` remapping inside
    :meth:`MeshModel.from_trimesh`, so raw triangle indices stay aligned even
    when cleanup drops degenerate faces.
    """
    from trimesh.visual import ColorVisuals

    count = len(mesh.faces)
    rgba = np.full((count, 4), 255, dtype=np.uint8)
    rgba[:, 0:3] = np.asarray(DEFAULT_COLOR[:3], dtype=np.uint8)
    for face_id, colour in colors.items():
        if 0 <= int(face_id) < count:
            rgba[int(face_id)] = np.asarray(colour, dtype=np.uint8)
    mesh.visual = ColorVisuals(face_colors=rgba)


def load_model(path: str | Path) -> MeshModel:
    source_path = str(path)
    extension = Path(path).suffix.lower()
    if extension not in SUPPORTED_IMPORT_EXTENSIONS:
        raise ValueError(
            f"Unsupported import format '{extension}'. Supported: {sorted(SUPPORTED_IMPORT_EXTENSIONS)}"
        )
    logger.info("Loading model from %s", source_path)

    # .blend files are converted via headless Blender to a temp .glb first.
    if extension == ".blend":
        from .blend_convert import convert_blend_to_glb

        with convert_blend_to_glb(source_path) as glb_path:
            mesh_model = load_model(glb_path)
        mesh_model.source_path = source_path
        logger.info(
            "Prepared mesh model | faces=%s | vertices=%s | colors=%s | source=%s",
            mesh_model.face_count,
            mesh_model.vertex_count,
            len(mesh_model.face_colours),
            source_path,
        )
        return mesh_model

    # Load the mesh
    if extension == ".3mf":
        # 3MF needs special handling - load as a scene
        loaded = trimesh.load(source_path, file_type="3mf")
        mesh = _coerce_trimesh(loaded)
    else:
        mesh = _coerce_trimesh(trimesh.load(source_path, force="scene"))

    if len(mesh.faces) == 0 or len(mesh.vertices) == 0:
        raise ValueError("The selected model did not contain any renderable triangles")

    # 3MF color/texture data is not understood by trimesh (it loads textured
    # 3MFs as flat placeholder grey), so bake our own extraction onto the
    # trimesh *before* MeshModel dedup remaps face indices.
    if extension == ".3mf":
        pre_extracted: dict[int, tuple[int, int, int, int]] = {}
        flat = _extract_3mf_colors(source_path)
        if flat:
            pre_extracted.update(flat)
        textured = _extract_3mf_texture_colors(source_path)
        if textured:
            pre_extracted.update(textured)
        if pre_extracted:
            if len(pre_extracted) != len(mesh.faces):
                logger.warning(
                    "3MF color count %s != face count %s; applying overlapping range",
                    len(pre_extracted),
                    len(mesh.faces),
                )
            _apply_face_colors_to_trimesh(mesh, pre_extracted)
            logger.info("Baked %d face colors from 3MF file", len(pre_extracted))

    # Create the mesh model
    mesh_model = MeshModel.from_trimesh(mesh, source_path=source_path)

    logger.info(
        "Prepared mesh model | faces=%s | vertices=%s | colors=%s | source=%s",
        mesh_model.face_count,
        mesh_model.vertex_count,
        len(mesh_model.face_colours),
        source_path,
    )
    return mesh_model

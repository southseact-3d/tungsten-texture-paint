from __future__ import annotations

import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import trimesh
from PIL import Image

from .color_utils import Color, DEFAULT_COLOR, rgb_hex
from .mesh_model import MeshModel
from .sketch_tool import bake_sketch_to_faces

SUPPORTED_EXPORT_EXTENSIONS = {".3mf", ".stl", ".obj", ".ply", ".glb", ".gltf", ".fbx"}

CORE_NS = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
MAT_NS = "http://schemas.microsoft.com/3dmanufacturing/material/2015/02"
ET.register_namespace("", CORE_NS)
ET.register_namespace("m", MAT_NS)


def validate_for_export(mesh_model: MeshModel) -> tuple[trimesh.Trimesh, list[str]]:
    mesh = mesh_model.mesh()
    issues: list[str] = []
    if not mesh.is_watertight:
        trimesh.repair.fill_holes(mesh)
        if not mesh.is_watertight:
            issues.append("Mesh has holes that could not be auto-repaired.")
    if mesh.volume < 0:
        mesh.invert()
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()
    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces())
    if not issues:
        issues.append("OK - mesh is valid for export.")
    return mesh, issues


def final_face_colours(
    mesh_model: MeshModel,
    sketch_image: Image.Image | None = None,
    view_projection: object | None = None,
    viewport_size: tuple[int, int] | None = None,
) -> list[Color]:
    colours = [mesh_model.face_colour(face_id) for face_id in range(mesh_model.face_count)]
    if sketch_image is not None and view_projection is not None and viewport_size is not None:
        baked = bake_sketch_to_faces(mesh_model, sketch_image, view_projection, viewport_size)
        for face_id, colour in baked.items():
            colours[face_id] = colour
    return colours


def export_3mf(
    path: str | Path,
    mesh_model: MeshModel,
    sketch_image: Image.Image | None = None,
    view_projection: object | None = None,
    viewport_size: tuple[int, int] | None = None,
) -> list[str]:
    validated_mesh, issues = validate_for_export(mesh_model)
    colours = final_face_colours(mesh_model, sketch_image, view_projection, viewport_size)
    if len(colours) != len(validated_mesh.faces):
        colours = colours[: len(validated_mesh.faces)]
        if len(colours) < len(validated_mesh.faces):
            colours.extend([DEFAULT_COLOR] * (len(validated_mesh.faces) - len(colours)))

    unique_colours: list[tuple[int, int, int]] = []
    colour_index: dict[tuple[int, int, int], int] = {}
    for colour in colours:
        rgb = colour[:3]
        if rgb not in colour_index:
            colour_index[rgb] = len(unique_colours)
            unique_colours.append(rgb)

    model = ET.Element(f"{{{CORE_NS}}}model", {"unit": "millimeter"})
    resources = ET.SubElement(model, f"{{{CORE_NS}}}resources")
    colour_group = ET.SubElement(resources, f"{{{MAT_NS}}}colorgroup", {"id": "1"})
    for colour in unique_colours:
        ET.SubElement(colour_group, f"{{{MAT_NS}}}color", {"color": rgb_hex(colour)})

    obj = ET.SubElement(resources, f"{{{CORE_NS}}}object", {"id": "2", "type": "model"})
    mesh_element = ET.SubElement(obj, f"{{{CORE_NS}}}mesh")
    vertices_element = ET.SubElement(mesh_element, f"{{{CORE_NS}}}vertices")
    for x, y, z in validated_mesh.vertices:
        ET.SubElement(
            vertices_element,
            f"{{{CORE_NS}}}vertex",
            {"x": f"{float(x):.6f}", "y": f"{float(y):.6f}", "z": f"{float(z):.6f}"},
        )
    triangles_element = ET.SubElement(mesh_element, f"{{{CORE_NS}}}triangles")
    for face_id, (v1, v2, v3) in enumerate(validated_mesh.faces):
        colour = colours[face_id][:3]
        ET.SubElement(
            triangles_element,
            f"{{{CORE_NS}}}triangle",
            {"v1": str(int(v1)), "v2": str(int(v2)), "v3": str(int(v3)), "pid": "1", "p1": str(colour_index[colour])},
        )
    build = ET.SubElement(model, f"{{{CORE_NS}}}build")
    ET.SubElement(build, f"{{{CORE_NS}}}item", {"objectid": "2"})

    model_xml = ET.tostring(model, encoding="utf-8", xml_declaration=True)
    content_types = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
</Types>
"""
    rels = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>
"""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", rels)
        archive.writestr("3D/3dmodel.model", model_xml)
    return issues


def export_model(path: str | Path, mesh_model: MeshModel) -> list[str]:
    output = Path(path)
    ext = output.suffix.lower()
    if ext not in SUPPORTED_EXPORT_EXTENSIONS:
        raise ValueError(f"Unsupported export format '{ext}'. Supported: {sorted(SUPPORTED_EXPORT_EXTENSIONS)}")
    if ext == ".3mf":
        return export_3mf(output, mesh_model)
    mesh, issues = validate_for_export(mesh_model)
    output.parent.mkdir(parents=True, exist_ok=True)
    file_type = "gltf" if ext == ".gltf" else ext[1:]
    mesh.export(output, file_type=file_type)
    return issues

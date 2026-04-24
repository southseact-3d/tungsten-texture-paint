from __future__ import annotations

import xml.etree.ElementTree as ET
import zipfile

import pytest

from stl_painter.exporter import export_3mf, export_model


def test_export_3mf_writes_expected_package(tmp_path, square_mesh) -> None:
    square_mesh.set_face_colour(0, (255, 0, 0, 255))
    square_mesh.set_face_colour(1, (0, 255, 0, 255))
    output = tmp_path / "mesh.3mf"

    issues = export_3mf(output, square_mesh)

    assert output.exists()
    assert issues
    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        assert "[Content_Types].xml" in names
        assert "_rels/.rels" in names
        assert "3D/3dmodel.model" in names
        model_root = ET.fromstring(archive.read("3D/3dmodel.model"))
        ns = {
            "core": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02",
            "m": "http://schemas.microsoft.com/3dmanufacturing/material/2015/02",
        }
        colours = model_root.findall(".//m:colorgroup/m:color", ns)
        triangles = model_root.findall(".//core:triangle", ns)
        assert len(colours) >= 2
        assert len(triangles) == 2
        assert all(triangle.attrib["pid"] == "1" for triangle in triangles)


def test_export_model_common_formats(tmp_path, square_mesh) -> None:
    for ext in (".obj", ".stl", ".glb", ".ply"):
        output = tmp_path / f"mesh{ext}"
        issues = export_model(output, square_mesh)
        assert output.exists()
        assert issues


def test_validate_for_export_watertight(square_mesh) -> None:
    from stl_painter.exporter import validate_for_export

    mesh, issues = validate_for_export(square_mesh)
    assert mesh is not None
    assert len(issues) > 0


def test_export_model_rejects_unsupported_format(tmp_path, square_mesh) -> None:
    from stl_painter.exporter import export_model

    output = tmp_path / "mesh.xyz"
    with pytest.raises(ValueError, match="Unsupported export format"):
        export_model(output, square_mesh)


def test_export_model_stl_format(tmp_path, square_mesh) -> None:
    from stl_painter.exporter import export_model

    output = tmp_path / "mesh.stl"
    issues = export_model(output, square_mesh)
    assert output.exists()
    assert "OK" in issues[0] or len(issues) > 0


def test_export_model_obj_format(tmp_path, square_mesh) -> None:
    from stl_painter.exporter import export_model

    output = tmp_path / "mesh.obj"
    issues = export_model(output, square_mesh)
    assert output.exists()
    assert len(issues) > 0


def test_export_model_ply_format(tmp_path, square_mesh) -> None:
    from stl_painter.exporter import export_model

    output = tmp_path / "mesh.ply"
    issues = export_model(output, square_mesh)
    assert output.exists()
    assert len(issues) > 0


def test_export_model_glb_format(tmp_path, square_mesh) -> None:
    from stl_painter.exporter import export_model

    output = tmp_path / "mesh.glb"
    issues = export_model(output, square_mesh)
    assert output.exists()
    assert len(issues) > 0


def test_export_3mf_with_colours(tmp_path, square_mesh) -> None:
    from stl_painter.exporter import export_3mf

    square_mesh.set_face_colour(0, (255, 0, 0, 255))
    square_mesh.set_face_colour(1, (0, 255, 0, 255))
    output = tmp_path / "mesh.3mf"
    issues = export_3mf(output, square_mesh)
    assert output.exists()
    with zipfile.ZipFile(output) as archive:
        model_xml = archive.read("3D/3dmodel.model")
        assert b"colorgroup" in model_xml


def test_final_face_colours(square_mesh) -> None:
    from stl_painter.exporter import final_face_colours

    colours = final_face_colours(square_mesh)
    assert len(colours) == square_mesh.face_count


def test_export_3mf_creates_parent_directories(tmp_path, square_mesh) -> None:
    from stl_painter.exporter import export_3mf

    output = tmp_path / "subdir" / "nested" / "mesh.3mf"
    issues = export_3mf(output, square_mesh)
    assert output.exists()

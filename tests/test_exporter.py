from __future__ import annotations

import xml.etree.ElementTree as ET
import zipfile

from stl_painter.exporter import export_3mf


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

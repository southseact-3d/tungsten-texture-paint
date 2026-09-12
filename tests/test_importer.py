from __future__ import annotations

import numpy as np
import pytest
import trimesh

from stl_painter.importer import load_model
from stl_painter.mesh_model import MeshModel


def test_load_model_obj_reads_mesh(tmp_path) -> None:
    mesh = trimesh.creation.box(extents=(1.0, 2.0, 3.0))
    path = tmp_path / "box.obj"
    mesh.export(path)

    loaded = load_model(path)

    assert loaded.face_count > 0
    assert loaded.vertex_count > 0
    assert loaded.source_path == str(path)


def test_map_colours_from_handles_scaled_variant(square_mesh) -> None:
    source = square_mesh
    source.set_face_colour(0, (255, 0, 0, 255))
    source.set_face_colour(1, (0, 255, 0, 255))

    scaled = MeshModel(
        vertices=(source.vertices * 2.5).astype(np.float32),
        faces=source.faces.copy(),
        normals=source.normals.copy(),
    )
    mapped = scaled.map_colours_from(source, normalize_scale=True)

    assert mapped == scaled.face_count
    assert scaled.face_colour(0) == (255, 0, 0, 255)
    assert scaled.face_colour(1) == (0, 255, 0, 255)


def test_load_stl_binary_format(tmp_path) -> None:
    mesh = trimesh.creation.box(extents=(1.0, 2.0, 3.0))
    path = tmp_path / "box.stl"
    mesh.export(path)

    loaded = load_model(path)

    assert loaded.face_count > 0
    assert loaded.vertex_count > 0
    assert loaded.source_path == str(path)


def test_load_stl_ascii_format(tmp_path) -> None:
    mesh = trimesh.creation.box(extents=(1.0, 2.0, 3.0))
    path = tmp_path / "box_ascii.stl"
    mesh.export(path, file_type="stl_ascii")

    loaded = load_model(path)

    assert loaded.face_count > 0
    assert loaded.vertex_count > 0


def test_load_model_rejects_unsupported_extension(tmp_path) -> None:
    from stl_painter.importer import load_model

    path = tmp_path / "mesh.xyz"
    path.write_text("dummy")

    with pytest.raises(ValueError, match="Unsupported import format"):
        load_model(path)


def test_load_model_glb_format(tmp_path) -> None:
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    path = tmp_path / "box.glb"
    mesh.export(path)

    loaded = load_model(path)

    assert loaded.face_count > 0
    assert loaded.vertex_count > 0


def test_load_model_gltf_format(tmp_path) -> None:
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    path = tmp_path / "box.gltf"
    mesh.export(path)

    loaded = load_model(path)

    assert loaded.face_count > 0
    assert loaded.vertex_count > 0


def test_load_model_empty_file(tmp_path) -> None:
    from stl_painter.importer import load_model

    path = tmp_path / "empty.stl"
    path.touch()

    with pytest.raises(ValueError, match="did not contain any mesh"):
        load_model(path)


def test_import_stl_alias(tmp_path) -> None:
    from stl_painter.importer import import_stl

    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    path = tmp_path / "test.stl"
    mesh.export(path)

    loaded = import_stl(path)

    assert loaded.face_count > 0


def test_map_colours_empty_target(square_mesh) -> None:
    source = square_mesh
    source.set_face_colour(0, (255, 0, 0, 255))

    with pytest.raises(ValueError, match="does not contain any faces"):
        MeshModel(
            vertices=np.empty((0, 3), dtype=np.float32),
            faces=np.empty((0, 3), dtype=np.int32),
            normals=np.empty((0, 3), dtype=np.float32),
        )


def _write_textured_3mf(path, color=(200, 30, 10)) -> None:
    """Write a minimal single-triangle textured 3MF (2x2 PNG, UVs on one texel)."""
    import zipfile
    from PIL import Image

    img = Image.new("RGBA", (2, 2), color)
    img_path = "3D/Textures/tex.png"
    model_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<model xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" xmlns:m="http://schemas.microsoft.com/3dmanufacturing/material/2015/02" unit="millimeter">
<resources>
<m:texture2d id="1" path="/{img_path}" contenttype="image/png"/>
<m:texture2dgroup id="2" texid="1">
<m:tex2coord u="0.25" v="0.25"/><m:tex2coord u="0.25" v="0.25"/><m:tex2coord u="0.25" v="0.25"/>
</m:texture2dgroup>
<object id="3" type="model"><mesh>
<vertices><vertex x="0" y="0" z="0"/><vertex x="1" y="0" z="0"/><vertex x="0" y="1" z="0"/></vertices>
<triangles><triangle v1="0" v2="1" v3="2" pid="2" p1="0" p2="1" p3="2"/></triangles>
</mesh></object>
</resources>
<build><item objectid="3"/></build>
</model>"""
    import io

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("3D/3dmodel.model", model_xml)
        archive.writestr(img_path, buffer.getvalue())
        archive.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/></Types>',
        )
        archive.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>',
        )


def test_load_model_3mf_textured_bakes_to_face_colours(tmp_path) -> None:
    path = tmp_path / "textured.3mf"
    _write_textured_3mf(path)

    loaded = load_model(path)

    assert loaded.face_count == 1
    assert loaded.face_colour(0)[:3] == (200, 30, 10)


def test_load_model_3mf_colorgroup_alpha_and_p123(tmp_path) -> None:
    import zipfile

    model_xml = """<?xml version="1.0" encoding="UTF-8"?>
<model xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" xmlns:m="http://schemas.microsoft.com/3dmanufacturing/material/2015/02" unit="millimeter">
<resources>
<m:colorgroup id="1"><m:color color="#FF000080"/><m:color color="#00FF00"/></m:colorgroup>
<object id="2" type="model"><mesh>
<vertices><vertex x="0" y="0" z="0"/><vertex x="1" y="0" z="0"/><vertex x="0" y="1" z="0"/></vertices>
<triangles><triangle v1="0" v2="1" v3="2" pid="1" p1="0" p2="0" p3="1"/></triangles>
</mesh></object>
</resources>
<build><item objectid="2"/></build>
</model>"""
    path = tmp_path / "colorgroup.3mf"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("3D/3dmodel.model", model_xml)

    loaded = load_model(path)

    assert loaded.face_count == 1
    # Mean of #FF000080, #FF000080, #00FF00 -> (170, 85, 0, 170)
    assert loaded.face_colour(0) == (170, 85, 0, 170)


def test_from_trimesh_ignores_placeholder_grey() -> None:
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))

    loaded = MeshModel.from_trimesh(mesh)

    assert loaded.face_colours == {}


def test_from_trimesh_bakes_texture_visuals() -> None:
    import trimesh.visual as visuals
    from PIL import Image

    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    red = Image.new("RGBA", (4, 4), (10, 200, 60, 255))
    material = visuals.material.SimpleMaterial(image=red)
    uv = np.full((len(mesh.vertices), 2), 0.5)
    mesh.visual = visuals.TextureVisuals(uv=uv, material=material)

    loaded = MeshModel.from_trimesh(mesh)

    assert len(loaded.face_colours) == loaded.face_count
    assert set(loaded.face_colours.values()) == {(10, 200, 60, 255)}


def test_blend_extension_supported_and_needs_blender(tmp_path) -> None:
    from stl_painter.blend_convert import find_blender
    from stl_painter.importer import SUPPORTED_IMPORT_EXTENSIONS

    assert ".blend" in SUPPORTED_IMPORT_EXTENSIONS
    # Either a Blender is found (dev machine) or a helpful error is raised.
    try:
        found = find_blender()
        assert found.lower().endswith("blender.exe") or found.lower().endswith(
            "blender"
        )
    except FileNotFoundError as exc:
        assert "Blender was not found" in str(exc)


def test_map_colours_without_normalize(square_mesh) -> None:
    source = square_mesh
    source.set_face_colour(0, (255, 0, 0, 255))

    scaled = MeshModel(
        vertices=(source.vertices * 2.5).astype(np.float32),
        faces=source.faces.copy(),
        normals=source.normals.copy(),
    )
    mapped = scaled.map_colours_from(source, normalize_scale=False)
    assert mapped >= 0

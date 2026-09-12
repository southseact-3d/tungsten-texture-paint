"""Run inside headless Blender: export the current ``.blend`` scene to ``.glb``.

Invoked as::

    blender --background model.blend --python blend_to_glb.py -- out.glb
"""

import sys

import bpy


def main() -> None:
    output = sys.argv[sys.argv.index("--") + 1]

    # Select all visible mesh objects so multi-part models merge on import
    # (the app concatenates scene geometry into a single mesh).
    bpy.ops.object.select_all(action="DESELECT")
    meshes = [
        obj
        for obj in bpy.context.scene.objects
        if obj.type == "MESH" and not obj.hide_render
    ]
    if not meshes:
        raise SystemExit("No mesh objects found in .blend scene")
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]

    export_kwargs: dict = {
        "filepath": output,
        "export_format": "GLB",
        "export_apply": True,
        "export_texcoords": True,
        "export_normals": True,
        "export_materials": "EXPORT",
        "use_selection": True,
        "export_yup": True,
    }
    # Newer Blender builds expose explicit vertex-color toggles; older ones
    # export active vertex colors by default. Guard with hasattr-style try.
    try:
        bpy.ops.export_scene.gltf(**{**export_kwargs, "export_colors": True})
    except TypeError:
        bpy.ops.export_scene.gltf(**export_kwargs)
    print(f"WROTE {output}")


main()

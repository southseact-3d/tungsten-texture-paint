# STL Texture Painter

A desktop Python application for painting colors onto 3D STL meshes and exporting them for 3D printing.

## Features

- **STL Import** - Load and validate STL mesh files
- **Per-Face Painting** - Paint individual triangles with colors using a palette or custom color picker
- **Sketch Overlays** - Draw text, rectangles, lines, and freehand strokes on the 3D viewport
- **Sketch Baking** - Project sketch overlays onto the mesh as per-face colors
- **Multi-Format Export** - Export painted meshes as 3MF, OBJ, GLB, GLTF, STL, PLY, or FBX
- **Project Save/Load** - Save and load `.tg3d` projects with undo/redo support

## Run Locally

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -r requirements.txt
python main.py
```

## Tests

```bash
pytest
```

## Notes

- Place a font file such as `DejaVuSans.ttf` in the `stl_painter/assets/` folder for text rendering support. The app falls back to system fonts when not available.

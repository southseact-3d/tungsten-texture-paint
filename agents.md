# STL Texture Painter - Agent Guide

## What This App Is

**STL Texture Painter** is a desktop Python application for painting colors onto 3D STL meshes and exporting them for 3D printing.

### Core Features
1. **STL Import** - Load and validate STL mesh files into the application
2. **Per-Face Painting** - Paint individual triangles with colors using a palette or custom color picker
3. **Sketch Overlays** - Draw text, rectangles, lines, and freehand strokes on the 3D viewport
4. **Sketch Baking** - Project sketch overlays onto the mesh as per-face colors
5. **3MF Export** - Export the painted mesh as a colored 3MF package for 3D printing

### Architecture
- **UI Framework**: Dear PyGui with ModernGL off-screen rendering
- **Core Modules**:
  - `mesh_model.py` - Central mesh data structure (triangles, faces, colors)
  - `importer.py` - STL file loading and validation
  - `renderer.py` - ModernGL-based 3D rendering
  - `paint_tool.py` - Per-face painting with flood fill and undo support
  - `sketch_tool.py` - 2D overlay primitives (text, shapes, strokes)
  - `exporter.py` - 3MF export with color groups
  - `project_io.py` - JSON project save/load

## Agent Rules

### Always Build and Test After Changes
**IMPORTANT**: After completing any task, you MUST verify that your changes do not break STL file import.

1. Run the test suite to ensure existing tests pass:
   ```bash
   pytest
   ```

2. If you have changed anything that may affect the import functionailty , test that the app can import an STL file by running:
   ```bash
   python -c "from stl_painter.importer import import_stl; import_stl('dart.stl')"
   ```

3. If you modified the mesh model or importer, also run:
   ```bash   python main.py
   ```
   Then verify you can load the test STL file (`dart.stl`) through the app.
4. You must always ensure that you fully test your changes and add additional python tests and do not stop until the functionality or fix is fully implemented, and you are sure that it will definetely work.

**Failure to verify STL import after changes may result in the app being unable to load STL files, which is a critical feature.**

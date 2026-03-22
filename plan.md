No problem! Here's the full plan:

---

# STL Texture Painter — Build Plan

## Technology Stack (Recommendations)

**Rendering:** `moderngl` + `PyOpenGL` for the 3D viewport. ModernGL wraps raw OpenGL in a clean Pythonic API — perfect for a beginner, no C pointer nonsense, but full GPU acceleration.

**UI:** `Dear PyGui` — GPU-rendered, minimal boilerplate, great for tool-style apps with sidebars and panels. Integrates well alongside an OpenGL viewport.

**Mesh I/O:** `numpy-stl` for reading STL files. `trimesh` for mesh utilities (face normals, adjacency, UV unwrapping helpers).

**3MF Export:** `lib3mf` (official Python bindings) or hand-write the XML — 3MF is just a ZIP of XML files, so it's very approachable.

**Overlay/Sketch baking:** `Pillow` (PIL) for 2D drawing of text, rectangles, arrows onto a texture, then `trimesh` or manual UV math to project that back onto faces.

---

## Architecture Overview

The app has five distinct layers that talk to each other in one direction:

```
STL File
   ↓
[Importer] → raw triangle soup (vertices + faces)
   ↓
[Mesh Model] → face list, normals, adjacency, face→colour map
   ↓
[Renderer] → GPU buffers, draws the mesh with per-face colours
   ↓
[Interaction Layer] → ray picking, brush, sketch overlay
   ↓
[Exporter] → bakes overlay, writes .3mf with vertex colours
```

---

## Module Breakdown

### Module 1 — STL Importer

**File:** `importer.py`

Use `numpy-stl` to load the file. STL has no face IDs natively — every triangle is just three vertices. You need to assign each face a stable integer ID (just its index in the array) so the rest of the app can reference faces by ID.

```python
from stl import mesh
import numpy as np

def load_stl(path):
    m = mesh.Mesh.from_file(path)
    # m.vectors shape: (N, 3, 3) — N faces, 3 verts, XYZ
    faces = m.vectors          # shape (N, 3, 3)
    normals = m.normals        # shape (N, 3)
    return faces, normals
```

Key outputs: flat vertex array, face index array, face normals. Pass these to the mesh model.

---

### Module 2 — Mesh Model (Central Data Store)

**File:** `mesh_model.py`

This is the single source of truth. Everything reads from and writes to this object.

```python
class MeshModel:
    vertices: np.ndarray      # (V, 3) unique vertex positions
    faces: np.ndarray         # (F, 3) indices into vertices
    normals: np.ndarray       # (F, 3) per-face normals
    face_colours: dict        # {face_id: (R, G, B, A)} 0-255
    overlay_strokes: list     # list of Stroke objects (see Module 5)
    default_colour: tuple     # fallback colour for unpainted faces
```

STL files store vertices redundantly (each face has its own 3 verts, even if shared). You'll want to deduplicate vertices early using `trimesh` or numpy's `unique` — this matters for the exporter.

```python
import trimesh

def from_stl(path):
    tm = trimesh.load(path, force='mesh')
    # trimesh already deduplicates
    return MeshModel(
        vertices=tm.vertices,
        faces=tm.faces,
        normals=tm.face_normals,
        face_colours={},
        overlay_strokes=[]
    )
```

---

### Module 3 — Renderer (3D Viewport)

**File:** `renderer.py`

This is the most complex module. It uses ModernGL to draw the mesh on screen with per-face colours.

**Core concept — per-face colour via a colour buffer:**

OpenGL works with vertices, not faces. To give each *face* its own colour, you "expand" the mesh so each face has its own 3 private vertices (no sharing). Then you can set a colour attribute per-vertex, and since all 3 verts of a face have the same colour, the face renders as flat-coloured.

```
Compact mesh:  V unique vertices, F faces sharing them
Expanded mesh: F*3 vertices, each face owns 3 private ones
               → set same colour on all 3 verts of a face
```

**OpenGL buffer layout (per vertex in expanded mesh):**

| Attribute | Type | Size |
|---|---|---|
| position | float32 x3 | 12 bytes |
| normal | float32 x3 | 12 bytes |
| colour | float32 x4 | 16 bytes |
| face_id | int32 | 4 bytes |

The `face_id` attribute is key — it lets you do GPU-side face identification for picking (see Module 4).

**Vertex shader (GLSL):**

```glsl
#version 330
in vec3 in_position;
in vec3 in_normal;
in vec4 in_colour;

uniform mat4 mvp;
uniform vec3 light_dir;

out vec4 v_colour;
out float v_diffuse;

void main() {
    gl_Position = mvp * vec4(in_position, 1.0);
    v_diffuse = max(dot(normalize(in_normal), normalize(light_dir)), 0.15);
    v_colour = in_colour;
}
```

**Fragment shader:**

```glsl
#version 330
in vec4 v_colour;
in float v_diffuse;
out vec4 out_colour;

void main() {
    out_colour = vec4(v_colour.rgb * v_diffuse, v_colour.a);
}
```

**Camera:** Implement a simple arcball/orbit camera. Store centre point, distance, azimuth angle, elevation angle. Mouse drag → update angles. Scroll wheel → update distance. Compute view matrix each frame.

**Updating colours:** When a face is painted, update the 3 corresponding vertices in the colour buffer. ModernGL lets you write to a sub-range of the VBO cheaply — no need to re-upload the whole mesh.

```python
def update_face_colour(self, face_id, rgba):
    offset = face_id * 3  # 3 verts per face in expanded layout
    colour_data = np.tile(rgba, (3, 1)).astype('f4')
    self.colour_vbo.write(colour_data.tobytes(), offset=offset * 4 * 4)
```

---

### Module 4 — Ray Picking (Click → Face ID)

**File:** `picking.py`

This is how you translate a mouse click on screen into a face ID in the mesh. There are two approaches:

**Option A — GPU picking (recommended):**

Render the scene a second time to an offscreen framebuffer. Instead of rendering colours, render each face's ID encoded as an RGB colour (face 1234 → R=0, G=4, B=210 using bit packing). After rendering, read the single pixel under the mouse cursor. Decode the colour back to a face ID. This is extremely fast and handles any mesh complexity.

```python
# Encode face_id as RGB (supports up to 16M faces)
r = (face_id >> 16) & 0xFF
g = (face_id >> 8)  & 0xFF
b = face_id         & 0xFF

# After reading pixel back:
face_id = (r << 16) | (g << 8) | b
```

The pick framebuffer only needs to be rendered on mouse click (not every frame), so performance is not an issue.

**Option B — CPU ray casting (simpler to understand, slower on large meshes):**

Cast a ray from the camera through the mouse position. Find which triangle the ray hits first (smallest positive t). Use `trimesh`'s built-in ray intersector.

```python
import trimesh

ray_origin, ray_dir = camera.unproject(mouse_x, mouse_y)
locations, ray_ids, face_ids = mesh.ray.intersects_id(
    ray_origins=[ray_origin],
    ray_directions=[ray_dir],
    return_locations=True
)
if len(face_ids):
    picked_face = face_ids[np.argmin(ray_ids)]
```

**Recommendation:** Start with Option B (CPU ray casting via trimesh) to get painting working quickly. Switch to GPU picking later if performance is poor on large meshes.

---

### Module 5 — Face Colour Store & Painting Logic

**File:** `paint_tool.py`

Handles the painting interaction layer:

- **Single face paint:** On left-click, get the picked face ID, set `mesh_model.face_colours[face_id] = current_colour`, call `renderer.update_face_colour(...)`.
- **Flood fill:** Pick a face, find all adjacent faces with the same colour (BFS over face adjacency graph), paint them all. Use `trimesh`'s `face_adjacency` attribute to build the graph.
- **Colour palette:** Store a list of up to 16 colours. User clicks a swatch to select the active colour. Start with JLC3DP's resin colour options as defaults.
- **Undo stack:** Keep a list of `(face_id, old_colour, new_colour)` tuples. Ctrl+Z pops and reverses.

```python
from collections import deque

class UndoStack:
    def __init__(self, max_size=200):
        self.stack = deque(maxlen=max_size)

    def push(self, face_id, old_col, new_col):
        self.stack.append((face_id, old_col, new_col))

    def undo(self, mesh_model, renderer):
        if self.stack:
            face_id, old_col, _ = self.stack.pop()
            mesh_model.face_colours[face_id] = old_col
            renderer.update_face_colour(face_id, old_col)
```

---

### Module 6 — Overlay / Sketch System

**File:** `sketch_tool.py`

This is the system for drawing text, rectangles, arrows, and freehand lines *onto the surface* of the model. This is the hardest part of the whole project.

**The core problem:** You're drawing 2D shapes onto a 3D surface. You need to figure out which faces those shapes land on, and what colour each pixel of those faces would be under the sketch.

**The pipeline for sketching:**

#### Step 1 — Planar UV Projection

When the user enters sketch mode, take a snapshot of the current camera orientation. Project a flat 2D coordinate system (U, V) onto the visible faces using the camera's view direction as the projection axis.

For each visible face, compute its 2D screen-space bounding region. This gives you a UV coordinate for every vertex in screen space.

```python
def project_vertices_to_screen(vertices, mvp_matrix, viewport_size):
    # Transform to clip space
    clip = vertices @ mvp_matrix[:3,:3].T + mvp_matrix[3,:3]
    # Perspective divide + viewport transform
    ndc = clip[:, :2] / clip[:, 2:3]
    uv = (ndc + 1.0) * 0.5 * viewport_size
    return uv  # shape (V, 2) in pixel coords
```

#### Step 2 — 2D Sketch Canvas

Open a 2D canvas (a `Pillow` Image) that is the same size as the viewport. The user draws on this canvas — text, rectangles, freehand lines. All drawing happens in screen-pixel coordinates. Show this as a transparent overlay on top of the 3D viewport using Dear PyGui's drawing API or a texture overlay.

**Supported sketch primitives:**
- **Text** — `ImageDraw.text(position, string, font, fill)`. Let user pick font size and colour.
- **Rectangle** — `ImageDraw.rectangle(xy, outline, fill)`. Drag to define.
- **Arrow/Line** — `ImageDraw.line(xy, fill, width)` + a triangle at the tip.
- **Freehand** — accumulate mouse positions while held, draw as a polyline.

#### Step 3 — Baking Sketch onto Faces

This happens at export time (or on demand). For each face:

1. Get the 3 screen-space UV positions of its vertices.
2. Rasterise that triangle region in the Pillow image.
3. Sample the Pillow image pixels that fall within that triangle.
4. If any non-transparent sketch pixels exist in that region, they override (or blend over) the face's base colour.
5. The final colour for that face is the average (or dominant) colour of its covered pixels.

```python
from PIL import Image, ImageDraw
import numpy as np

def sample_face_colour(sketch_image, uv_verts):
    # uv_verts: shape (3, 2) — screen coords of the 3 face corners
    # Create a mask image for just this triangle
    mask = Image.new('L', sketch_image.size, 0)
    ImageDraw.Draw(mask).polygon([tuple(v) for v in uv_verts], fill=255)
    masked = np.array(sketch_image)
    mask_arr = np.array(mask)
    pixels = masked[mask_arr > 0]
    if len(pixels) == 0:
        return None  # no sketch data on this face
    # Return mean colour of sketch pixels that hit this face
    return tuple(pixels.mean(axis=0).astype(int))
```

**Important caveat:** This projection only works for faces visible from the sketch camera angle. For faces on the back of the model, the UV projection doesn't reach them. This matches Blender's behaviour — you rotate the model and sketch from multiple angles if needed.

---

### Module 7 — 3MF Exporter

**File:** `exporter.py`

**What is 3MF?** It's a ZIP archive containing XML files. The key file is `3dmodel.model` — an XML file describing the mesh and its colours.

**JLC3DP full-colour resin** uses the `m:colorgroup` extension inside the 3MF spec to assign per-triangle colours. This is simpler than vertex colour painting — you just say "face 42 has colour #FF3300."

#### 3MF file structure inside the ZIP:

```
model.3mf
├── [Content_Types].xml
├── _rels/.rels
└── 3D/
    └── 3dmodel.model    ← all the geometry and colours live here
```

#### The `3dmodel.model` XML structure:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<model unit="millimeter"
       xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
       xmlns:m="http://schemas.microsoft.com/3dmanufacturing/material/2015/02">
  <resources>

    <!-- 1. Define the colour palette (one entry per unique colour used) -->
    <m:colorgroup id="1">
      <m:color color="#FF0000"/>   <!-- index 0: red -->
      <m:color color="#00FF00"/>   <!-- index 1: green -->
      <m:color color="#0000FF"/>   <!-- index 2: blue -->
    </m:colorgroup>

    <!-- 2. Define the mesh -->
    <object id="2" type="model">
      <mesh>
        <vertices>
          <vertex x="0" y="0" z="0"/>
          <vertex x="1" y="0" z="0"/>
          <vertex x="0" y="1" z="0"/>
          <!-- ... all vertices ... -->
        </vertices>
        <triangles>
          <!-- pid="1" references colorgroup id="1"
               p1="0" means this triangle uses colour index 0 from that group -->
          <triangle v1="0" v2="1" v3="2" pid="1" p1="0"/>
          <!-- ... all triangles ... -->
        </triangles>
      </mesh>
    </object>

  </resources>
  <build>
    <item objectid="2"/>
  </build>
</model>
```

#### Export algorithm:

```python
import zipfile
import xml.etree.ElementTree as ET
from io import BytesIO

def export_3mf(path, mesh_model, sketch_image=None, camera_uvs=None):

    # Step 1: Bake sketch onto face colours
    face_colours = dict(mesh_model.face_colours)  # copy
    if sketch_image is not None:
        baked = bake_sketch(sketch_image, mesh_model, camera_uvs)
        face_colours.update(baked)

    # Step 2: Collect all unique colours, build palette
    default = mesh_model.default_colour  # e.g. (220, 220, 220, 255)
    all_colours = [face_colours.get(i, default) for i in range(len(mesh_model.faces))]
    unique_colours = list({c[:3] for c in all_colours})  # deduplicate, drop alpha
    colour_index = {c: i for i, c in enumerate(unique_colours)}

    # Step 3: Build XML
    ns_core = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
    ns_mat  = "http://schemas.microsoft.com/3dmanufacturing/material/2015/02"

    model = ET.Element("model", {
        "unit": "millimeter",
        "xmlns": ns_core,
        "xmlns:m": ns_mat
    })
    resources = ET.SubElement(model, "resources")

    # Colour group
    cg = ET.SubElement(resources, "m:colorgroup", {"id": "1"})
    for r, g, b in unique_colours:
        ET.SubElement(cg, "m:color", {"color": f"#{r:02X}{g:02X}{b:02X}"})

    # Mesh object
    obj = ET.SubElement(resources, "object", {"id": "2", "type": "model"})
    mesh_el = ET.SubElement(obj, "mesh")

    verts_el = ET.SubElement(mesh_el, "vertices")
    for x, y, z in mesh_model.vertices:
        ET.SubElement(verts_el, "vertex", {"x": f"{x:.6f}", "y": f"{y:.6f}", "z": f"{z:.6f}"})

    tris_el = ET.SubElement(mesh_el, "triangles")
    for i, (v1, v2, v3) in enumerate(mesh_model.faces):
        col = all_colours[i][:3]
        p1 = colour_index[col]
        ET.SubElement(tris_el, "triangle", {
            "v1": str(v1), "v2": str(v2), "v3": str(v3),
            "pid": "1", "p1": str(p1)
        })

    build = ET.SubElement(model, "build")
    ET.SubElement(build, "item", {"objectid": "2"})

    # Step 4: Write ZIP
    model_xml = ET.tostring(model, encoding="unicode", xml_declaration=True)

    content_types = '''<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
</Types>'''

    rels = '''<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Target="/3D/3dmodel.model" Id="rel0"
    Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>'''

    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("3D/3dmodel.model", model_xml)
```

---

## 5. Data Model

```
MeshModel
├── vertices          np.ndarray (V, 3)   float32
├── faces             np.ndarray (F, 3)   int32
├── normals           np.ndarray (F, 3)   float32
├── face_colours      dict[int → (R,G,B,A)]
├── default_colour    tuple (R,G,B,A)
└── overlay_strokes   list[Stroke]

Stroke
├── type      str    "text" | "rect" | "line" | "freehand"
├── data      dict   type-specific (position, text, colour, width...)
└── camera    mat4   the camera matrix when the stroke was made
```

---

## 6. UI Layout & Controls

```
┌─────────────────────────────────────────────────────┐
│  File: [Open STL]  [Export 3MF]          [Undo] [?] │
├──────────────┬──────────────────────────────────────┤
│              │                                      │
│  TOOLS       │                                      │
│  ○ Paint     │         3D VIEWPORT                  │
│  ○ Fill      │         (ModernGL canvas)            │
│  ○ Sketch    │                                      │
│              │                                      │
│  COLOUR      │                                      │
│  [palette]   │                                      │
│  [R] [G] [B] │                                      │
│              │                                      │
│  SKETCH OPS  │                                      │
│  [Text]      │                                      │
│  [Rectangle] ├──────────────────────────────────────┤
│  [Line]      │  STATUS BAR  |  Face: 1423  |  Mode  │
│  [Freehand]  │                                      │
└──────────────┴──────────────────────────────────────┘
```

**Mouse controls in viewport:**
- Left-click → paint / pick face (in Paint mode) or place sketch element (in Sketch mode)
- Right-drag → orbit camera
- Middle-drag → pan camera
- Scroll wheel → zoom
- Shift+click → flood fill (in Paint mode)

---

## 7. The Overlay / Sketch System (Detail)

This deserves extra attention since it's the most novel part.

### Two-phase design

**Phase 1 — Live preview:** While the user draws, show the sketch as a 2D overlay directly on top of the 3D viewport (just a transparent PNG layer drawn in screen space). This is instant and requires no 3D math. The user sees their text/shapes on the screen in real time.

**Phase 2 — Baking:** When the user clicks "Bake Sketch" or triggers export, run the per-face sampling algorithm (Module 6, Step 3) to convert the 2D sketch into face colour overrides. After baking, the sketch is dissolved into the face colour map and the 3D view updates.

### Why per-face averaging works for 3D printing

Full-colour resin printing at JLC3DP has a minimum feature resolution of roughly 0.1mm per voxel. Your STL faces are probably much larger than this. So averaging the colour over a face and assigning it one flat colour is fine — the print won't show sub-face detail anyway. If faces are very large, the user should subdivide the mesh before importing (Meshmixer or Blender can do this in seconds).

### Text workflow

1. User selects Text tool, clicks a point on the 3D viewport.
2. A 2D text input box appears (Dear PyGui popup).
3. User types text, picks colour and size.
4. Text is rendered onto the sketch canvas at the clicked screen position using Pillow's `ImageFont`.
5. At bake time, text pixels are sampled per-face as described above.

### Font support

Use `Pillow` with a bundled `.ttf` file (e.g. DejaVuSans.ttf — open licence, ships with many Linux distros). For Windows/Mac, fall back to system font paths. Load with `ImageFont.truetype(path, size)`.

---

## 8. 3MF Export — Full-Colour Resin Format

### JLC3DP requirements

JLC3DP's full-colour resin (their "Full Color" material) accepts:

- **3MF** with the Microsoft 3D Manufacturing Format material extension (the `m:colorgroup` approach shown in Module 7).
- Per-triangle colour assignment (what we are building).
- Mesh must be watertight (no holes). Trimesh can validate and attempt repair: `trimesh.repair.fill_holes(mesh)`.
- Units: millimetres.
- Recommended: merge duplicate vertices, remove degenerate triangles before export.

### Colour accuracy note

JLC3DP converts your RGB colours to their CMYK dye process. Saturated colours (pure red, pure blue) print accurately. Very dark colours may come out lighter than expected. Recommend users test with a small print first.

### Pre-export checklist (run automatically)

```python
def validate_for_export(mesh_model):
    tm = trimesh.Trimesh(vertices=mesh_model.vertices, faces=mesh_model.faces)
    issues = []
    if not tm.is_watertight:
        trimesh.repair.fill_holes(tm)
        if not tm.is_watertight:
            issues.append("Mesh has holes that could not be auto-repaired.")
    if tm.volume < 0:
        tm.invert()  # fix inverted normals
    if len(issues) == 0:
        issues.append("OK — mesh is valid for export.")
    return tm, issues
```

---

## 9. Build Order & Milestones

Build in this exact order — each milestone is usable and testable before starting the next.

### Milestone 1 — View an STL

- Set up ModernGL + Dear PyGui window
- Load an STL with trimesh
- Render it grey with a single directional light
- Implement orbit/zoom/pan camera
- **Done when:** you can open any STL and spin it around

### Milestone 2 — Click to Paint a Face

- Implement CPU ray picking (trimesh)
- Wire up a basic colour palette (6 hardcoded colours)
- Click a face → it changes colour in the viewport
- **Done when:** you can paint individual faces different colours

### Milestone 3 — Flood Fill & Undo

- Build face adjacency graph
- Shift+click → flood fills connected same-colour region
- Ctrl+Z undoes last N paint operations
- **Done when:** painting feels like a usable tool

### Milestone 4 — Export 3MF

- Implement the exporter (Module 7)
- Export button saves the painted mesh as `.3mf`
- Validate the file opens in PrusaSlicer or Bambu Studio (both can preview 3MF colours)
- **Done when:** JLC3DP can accept the file (upload it to their site and check)

### Milestone 5 — Sketch Overlay

- Add Sketch mode toggle to UI
- Implement text placement on the 2D canvas
- Implement rectangle and line drawing
- Implement freehand drawing
- Show live 2D overlay on viewport
- **Done when:** user can place text and shapes over the model

### Milestone 6 — Bake Sketch to Faces

- Implement the per-face UV sampling / baking algorithm
- "Bake" button converts sketch to face colours
- Baked faces export correctly in 3MF
- **Done when:** a 3MF export with text baked on it looks correct in slicer preview

### Milestone 7 — Polish

- Improve colour picker (full HSV wheel via Dear PyGui's built-in)
- Add GPU picking for better performance on large meshes
- Add mesh validation UI (show warnings if mesh has holes)
- Drag-to-paint (hold mouse and sweep across faces)
- Save/load a `.json` project file (stores face colour map + strokes, not just the 3MF)

---

## 10. Dependency List

```
# Core
moderngl              # OpenGL rendering
moderngl-window       # window + event loop
numpy                 # math everywhere
trimesh               # mesh loading, normals, adjacency, repair

# STL reading (trimesh handles this, but explicit fallback)
numpy-stl

# UI
dearpygui             # all UI panels and controls

# Sketch / baking
Pillow                # 2D sketch canvas, text rendering, font support

# 3MF export (optional — can write XML manually)
# lib3mf              # only if hand-written XML gets painful

# Dev utilities
pytest                # unit tests for exporter and baking math
```

Install all at once:
```bash
pip install moderngl moderngl-window numpy trimesh numpy-stl dearpygui Pillow pytest
```

---

## 11. Key Risks & How to Handle Them

**Risk: STL mesh has too many tiny faces — performance slow**
Handle: On import, if face count > 100,000, prompt the user to decimate. Use `trimesh.simplify.quadric_decimation(mesh, target_face_count)`.

**Risk: Ray picking is slow on high-poly meshes**
Handle: Use trimesh's BVH (bounding volume hierarchy) accelerator — it's enabled by default and handles millions of faces fine. Upgrade to GPU picking in Milestone 7 if still slow.

**Risk: Sketch baking gives wrong results (faces get wrong colour)**
Handle: Render a debug view that shows the UV projection — draw triangle outlines of each face onto the sketch canvas so the user can see the mapping. Very useful for diagnosing bake errors.

**Risk: 3MF rejected by JLC3DP**
Handle: Validate with PrusaSlicer first (it shows colour previews and strict import warnings). Also validate with the official `3mf-validator` CLI tool (`pip install threemf`).

**Risk: Font rendering on Windows vs Mac vs Linux**
Handle: Bundle a single `.ttf` file with the app (DejaVuSans or Roboto — both are open-licensed). Never rely on system fonts for sketch text.

**Risk: Sketch text on curved surfaces looks distorted**
Handle: This is expected and unavoidable with planar projection. Document it: users should flatten or orient the model so the text-facing surface is roughly perpendicular to the camera before sketching.

---

## 12. Folder Structure

```
stl_painter/
├── main.py                  # entry point — creates window, wires everything together
├── mesh_model.py            # MeshModel dataclass — central data store
├── importer.py              # load_stl() → MeshModel
├── renderer.py              # ModernGL viewport, camera, GPU buffers
├── picking.py               # ray picking and GPU pick buffer
├── paint_tool.py            # face painting, flood fill, undo stack
├── sketch_tool.py           # 2D sketch canvas, stroke types, baking
├── exporter.py              # 3MF XML writer and ZIP packager
├── ui.py                    # Dear PyGui layout, panels, event handlers
├── shaders/
│   ├── mesh.vert            # vertex shader
│   └── mesh.frag            # fragment shader
├── assets/
│   └── DejaVuSans.ttf       # bundled font for sketch text
├── tests/
│   ├── test_exporter.py
│   └── test_baking.py
└── requirements.txt
```

---

## Summary

The hardest parts in order: **sketch baking** (UV projection + per-face sampling), **renderer setup** (ModernGL VBO layout for per-face colour), and **3MF export XML** (getting the namespace and colour group format exactly right). Everything else — picking, painting, undo — is straightforward Python data structures once the renderer is working.

Start with Milestone 1 and treat every milestone as shippable. The sketch system is an enhancement on top of a working painter, so if it proves too complex, the core paint-and-export loop is already valuable on its own.
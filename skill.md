# STL Texture Painter — CLI Usage Guide

> **Skill file** — how to drive the full painting workflow from the command line.

---

## Overview

The STL Texture Painter exposes a complete headless CLI that lets you import STL meshes, paint faces, and export coloured 3MF files without opening the GUI.

Two equivalent entry points exist:

```powershell
# after: pip install -e .
tgpaint <subcommand> [options]

# always available
python main.py <subcommand> [options]

# or as a module
python -m stl_painter.cli <subcommand> [options]
```

---

## Quick-start

```powershell
# 1. See what's in the mesh
tgpaint info --input dart.stl

# 2. Paint the whole mesh grey, export immediately to 3MF
tgpaint paint-all --input dart.stl --color "#B0B8C4" --output dart_grey.3mf

# 3. Flood-fill face 42 red starting from a different base
tgpaint flood-fill --input dart.stl --face-id 42 --color "#FF3300" --output dart_painted.3mf

# 4. Multi-step edit with a mid-point checkpoint
tgpaint paint-all  --input dart.stl       --color "#B0B8C4" --save step1.tg3d
tgpaint flood-fill --input step1.tg3d     --face-id 42     --color "#FF3300" --save step2.tg3d
tgpaint export     --input step2.tg3d     --output dart_final.3mf
```

---

## Colour formats

All `--color` arguments accept:

| Format | Example |
|---|---|
| `#RRGGBB` hex | `#FF0000` |
| `#RRGGBBAA` hex with alpha | `#FF000080` |
| `R,G,B` comma-separated 0–255 | `255,0,0` |
| `R,G,B,A` with alpha | `255,0,0,128` |

---

## Sub-commands reference

### `info` — mesh statistics

```
tgpaint info --input <FILE>
```

Prints face count, vertex count, bounding box extents, diagonal, and number of painted faces.

---

### `paint-face` — paint explicit face IDs

```
tgpaint paint-face --input <FILE> --face-ids <ID…> --color <COLOR> [--output <OUT>] [--save <TG3D>]
```

| Argument | Description |
|---|---|
| `--face-ids` / `-f` | One or more 0-based face IDs (space-separated) |
| `--color` / `-c` | Paint colour |
| `--output` / `-o` | Export result (`.3mf`, `.stl`, `.tg3d`, …) |
| `--save` | Also save a `.tg3d` checkpoint |

**Example:**
```powershell
tgpaint paint-face --input dart.stl --face-ids 0 1 2 3 --color "#FF0000" --output result.3mf
```

---

### `paint-all` — paint every face

```
tgpaint paint-all --input <FILE> --color <COLOR> [--output <OUT>] [--save <TG3D>]
```

**Example — paint grey then save a project file:**
```powershell
tgpaint paint-all --input dart.stl --color "#B0B8C4" --save painted.tg3d
```

---

### `flood-fill` — fill connected region

```
tgpaint flood-fill --input <FILE> --face-id <ID> --color <COLOR> [--output <OUT>] [--save <TG3D>]
```

All faces connected to `face-id` that share the same colour will be flooded with the new colour.

**Example:**
```powershell
tgpaint flood-fill --input painted.tg3d --face-id 100 --color "#CC2200" --output painted2.3mf
```

---

### `list-groups` — discover face groups

```
tgpaint list-groups --input <FILE> [--angle <DEG>]
```

Groups are computed by clustering adjacent faces with similar normals.

| Argument | Description |
|---|---|
| `--angle` | Normal-angle tolerance in degrees (default: 30°). Lower = more, smaller groups. |

**Example output:**
```
Group                             Faces
--------------------------------------------
group_0                              4,201
group_1                                872
group_2                                 44
...
Total: 23 groups, 61,204 faces
```

---

### `paint-group` — paint a face group

```
tgpaint paint-group --input <FILE> --group <GROUP_ID> --color <COLOR> [--angle <DEG>] [--output <OUT>] [--save <TG3D>]
```

Use `list-groups` first to find valid group IDs.

**Example:**
```powershell
tgpaint list-groups --input dart.stl --angle 20
tgpaint paint-group --input dart.stl --group group_0 --color "#0055AA" --output dart_wing.3mf
```

---

### `paint-region` — paint by 3D bounding box

```
tgpaint paint-region --input <FILE> --min X,Y,Z --max X,Y,Z --color <COLOR> [--output <OUT>] [--save <TG3D>]
```

Paints all faces whose **centroid** falls within the axis-aligned bounding box.  
Coordinates are in mesh units (usually mm).

**Example — paint the nose section of a dart:**
```powershell
tgpaint paint-region --input dart.stl --min 0,0,0 --max 10,10,50 --color "#FF8800" --output nose.3mf
```

---

### `save` — convert to .tg3d project

```
tgpaint save --input <STL/OBJ/…> --output <FILE.tg3d>
```

Loads a mesh and saves it as a `.tg3d` project.  Useful for creating a base project before multi-step editing.

---

### `export`
Exports the painted model to a final format (e.g., `.3mf`, `.obj`, `.stl`, `.glb`).

```bash
tgpaint export --input painted.tg3d --output final.3mf
```

### `screenshot`
Renders a software-based screenshot of the painted mesh, useful for AI visual feedback or documentation. It does not require a GPU or GUI to function.

```bash
tgpaint screenshot --input painted.tg3d --output view.png --azimuth 45 --elevation 30 --width 800 --height 600
```

---

### `pipeline` — batch operations from JSON

```
tgpaint pipeline --input <FILE> [--output <OUT>] [--save <TG3D>] --pipeline <FILE.json>
tgpaint pipeline --input <FILE> --steps '[{...},{...}]'
```

A pipeline JSON file describes an ordered list of operations:

```json
{
  "input": "dart.stl",
  "output": "dart_painted.3mf",
  "steps": [
    { "op": "paint-all",    "color": "#B0B8C4" },
    { "op": "flood-fill",   "face_id": 42,   "color": "#FF3300" },
    { "op": "paint-group",  "group": "group_3", "color": "#0055AA", "angle": 20 },
    { "op": "paint-region", "min": [0,0,0], "max": [10,10,10], "color": "#0000FF" },
    { "op": "screenshot", "path": "preview.png", "azimuth": 45, "elevation": 30 },
    { "op": "export", "path": "final_model.3mf" }
  ]
}
```

#### Supported pipeline operations

| `op` | Required fields | Optional fields |
|---|---|---|
| `paint-all` | `color` | — |
| `paint-face` | `color`, `face_ids` | — |
| `flood-fill` | `color`, `face_id` | — |
| `paint-group` | `color`, `group` | `angle` (default 30°) |
| `paint-region` | `color`, `min`, `max` | — |
| `save` | `path` | — |
| `export` | `path` | — |

**Run a pipeline:**
```powershell
tgpaint pipeline --input dart.stl --pipeline my_ops.json

# Override output path from CLI
tgpaint pipeline --input dart.stl --pipeline ops.json --output final.3mf

# Inline steps (JSON array on command line)
tgpaint pipeline --input dart.stl --steps '[{"op":"paint-all","color":"#FF0000"}]' --output red.3mf
```

---

## Chaining with `--save` checkpoints

Every paint operation accepts `--save <path.tg3d>` to create a mid-point checkpoint **and** `--output` to export the result.  You can chain calls by piping checkpoint files:

```powershell
tgpaint paint-all   --input dart.stl          --color "#B0B8C4" --save s1.tg3d
tgpaint paint-group --input s1.tg3d  --group group_0 --color "#CC2200" --save s2.tg3d
tgpaint paint-group --input s2.tg3d  --group group_1 --color "#0055AA" --save s3.tg3d
tgpaint export      --input s3.tg3d  --output final.3mf
```

---

## Scripting from Python

`CLISession` can be imported directly in scripts:

```python
from stl_painter.cli_session import CLISession

s = CLISession()
s.load("dart.stl")

# Paint the whole mesh grey
s.paint_all((176, 184, 196, 255))

# Discover and paint groups
groups = s.list_groups(angle_tolerance_degrees=20.0)
print(groups)  # {'group_0': 4201, 'group_1': 872, ...}
s.paint_group("group_0", (204, 34, 0, 255))

# Flood-fill
s.flood_fill(100, (0, 85, 170, 255))

# Paint by region (min XYZ, max XYZ)
s.paint_region((0, 0, 0), (10, 10, 50), (255, 136, 0, 255))

# Undo last operation
s.undo()

# Save and export
s.save("painted.tg3d")
s.export("painted.3mf")
```

---

## Tips

- **Start with `info`** to confirm face counts and bounding box before painting regions.
- **Use `list-groups`** at different `--angle` values to find the right granularity — 15–30° gives semantic surface patches on most models.
- **`pipeline` is the most efficient** for complex multi-colour jobs; it loads the mesh once and applies all operations in memory.
- **`.tg3d` files** preserve the full undo timeline and can be opened in the GUI for further editing.
- **`flood-fill` is fastest** for painting connected areas of a single colour without needing to know face IDs.

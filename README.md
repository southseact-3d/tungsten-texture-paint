# STL Texture Painter

Desktop Python app for loading STL meshes, painting per-face colours, sketching viewport overlays, baking those overlays to triangle colours, and exporting a coloured 3MF package.

## What is implemented

- Milestone 1: STL import, central mesh model, orbit camera, ModernGL off-screen renderer, Dear PyGui app shell
- Milestone 2: CPU ray picking with GPU-picking fallback, per-face paint updates, palette and custom colour picker
- Milestone 3: flood fill, undo stack, drag-to-paint support
- Milestone 4: 3MF export with `m:colorgroup` triangle colour assignment and mesh validation
- Milestone 5: sketch overlay primitives for text, rectangles, lines, and freehand strokes
- Milestone 6: sketch baking from viewport projection into per-face colours
- Milestone 7: mesh validation messaging, GPU picking path, project save/load JSON

## Run locally

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

- Put a font such as `DejaVuSans.ttf` into [`stl_painter/assets/README.txt`](c:/Users/liamh/Downloads/ai%20-%202/texture%20paint/stl_painter/assets/README.txt)'s folder for bundled text rendering. The app falls back to common system fonts when possible.
- The GUI uses an off-screen ModernGL renderer and Dear PyGui texture presentation so CI can still validate the non-GUI pipeline through unit tests and package builds.

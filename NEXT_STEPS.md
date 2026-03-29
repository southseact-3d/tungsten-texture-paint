# STL Texture Painter: Remaining Work

This document describes what is still needed to make the recent expansion fully production-ready.

It is intentionally practical: each section focuses on current gaps, why they matter, and what to implement next.

## Current State

The repo now has a working v1 foundation for:

- Paint vs sketch mode switching in the left sidebar
- Brush/fill/sample/erase/mask paint tools
- Face-aligned sketch planes with rect/line/circle/text entities
- Sketch persistence in project JSON
- Sketch baking into per-face colours
- AI assistant sidebar with an OpenAI-compatible tool-calling loop
- Command-based undo/redo plumbing
- Cached local dev packaging workflow

The app runs, tests pass, STL import works, and the packaged dev EXE passes the STL self-test.

That said, several parts are still incomplete, simplified, or only partially integrated.

## 1. Core App Structure

### What is still needed

- Split the large [`stl_painter/app.py`](c:\Users\liamh\Downloads\ai - 2\texture paint\stl_painter\app.py) into smaller modules.
- Move UI construction into dedicated panel builders instead of keeping nearly all interaction logic in one class.
- Separate rendering overlays, selection logic, AI orchestration, and command dispatch more cleanly.

### Why this matters

The current implementation works, but `app.py` is still too central. It will be hard to safely add more tools, more sketch editing, and better AI flows if all behavior remains coupled there.

### Next implementation steps

- Extract left sidebar UI to `ui_panels.py`
- Extract right AI panel UI to `ui_panels.py`
- Move mouse/keyboard interpretation to `selection.py` or `interactions.py`
- Keep `TexturePainterApp` as orchestration only

## 2. Sketch System

### What is still needed

- Finish the `select` workflow so entity selection is more reliable.
- Improve handle detection and resizing logic for all entity types.
- Implement proper live preview while dragging new entities.
- Add a real `trim` tool or hide it until implemented.
- Add text size, stroke width, and entity property editing in the UI.
- Support multiple sketch documents or multiple planes per project more intentionally.
- Improve snap priority and snapping heuristics across more geometry.

### What is simplified right now

- Sketch entities are rendered as projected guide lines, not as a full CAD overlay layer.
- Snapping is limited to grid, anchor-face vertices/edge midpoints, and existing entity snap points.
- Rectangle resizing is basic.
- Circles and text are minimal compared to the plan.
- There is no true constraint system.

### Why this matters

The sketch workflow is usable, but it does not yet feel like a proper CAD sketch environment. Users will notice missing previews, weak editing affordances, and incomplete tooling.

### Next implementation steps

- Add a preview entity during drag and render it before commit
- Add explicit handles for corners, midpoints, center, endpoints, radius
- Add dimension text while dragging
- Add UI controls for text size and stroke width
- Hide `trim` until implemented or fully implement segment trimming
- Add sketch plane management UI: new plane, rename, delete, switch active plane

## 3. Baking and Surface Mapping

### What is still needed

- Make sketch baking more robust on larger or more irregular planar regions.
- Improve plane-region face filtering beyond the current distance-to-plane threshold.
- Handle thicker strokes and text bounds more accurately.
- Add a preview of the bake footprint before commit.

### What is simplified right now

- Baking rasterizes the sketch document into a temporary local-plane image.
- Faces are included when their vertices lie near the active plane.
- This works for planar regions, but it is still a coarse approximation.

### Why this matters

This is the most important correctness area for sketch mode. A user may accept a simple sketch UI early, but they will not accept baking that lands on the wrong triangles.

### Next implementation steps

- Add tests for multiple adjacent coplanar faces
- Add tests for very small entities and thin lines
- Improve face inclusion using plane-space polygon bounds, not just plane distance
- Add debug visualization for bake sampling regions

## 4. Paint Tools

### What is still needed

- Add brush preview in the viewport
- Improve stroke spacing and continuous drag behavior
- Add material/colour tolerance options for fill
- Add better mask visualization and mask management controls
- Make brush radius more intuitive in the UI

### What is simplified right now

- Brush paint works by weighted per-face updates from adjacency and distance
- It feels more brush-like than before, but it is still a face-based system rather than texture-space painting
- Drag painting applies repeated samples, but the spacing and smoothing are still basic

### Why this matters

The current system is functional, but users expecting Blender-like brush behavior will still find it rough.

### Next implementation steps

- Add viewport brush-ring preview
- Use brush spacing distance instead of face-change-only triggering during drags
- Add fill tolerance controls
- Add `clear mask`, `invert mask`, and masked-face count controls
- Add front-facing and angle filters to the UI with clearer wording

## 5. Undo/Redo

### What is still needed

- Move all paint operations fully onto the command stack
- Make mask changes undoable
- Make sketch-plane creation undoable
- Ensure AI actions are grouped into sensible command transactions

### What is simplified right now

- The command stack exists and is used for key new operations
- Some state changes still happen directly in app code

### Why this matters

Undo/redo is the backbone of a tool like this. Any state mutation that escapes the command layer will create confusing behavior.

### Next implementation steps

- Add explicit commands for:
  - mask toggle
  - create sketch plane
  - bake sketch
  - mode changes if desired
- Audit `app.py` for direct mutations and route them through `AppCommands`

## 6. AI Assistant

### What is still needed

- Support streaming responses in the UI
- Add better tool approval rules for destructive operations
- Improve system prompt and tool descriptions
- Add richer tools for entity editing, selection inspection, and export options
- Handle API/network errors more gracefully
- Decide whether to stay on Chat Completions or move to a Responses-style client

### What is simplified right now

- The assistant uses an OpenAI-compatible `/chat/completions` flow
- It supports tool calls, image attachment, and basic transcript/tool logs
- The tool surface is useful but still narrow

### Why this matters

The current AI integration is enough to prove the concept, but not yet enough for reliable user-facing automation.

### Next implementation steps

- Add streaming token updates to the transcript
- Add confirmation prompts for export overwrite, clearing sketches, and loading a new file
- Add more tools:
  - inspect sketch documents
  - update/delete sketch entities
  - create lines/circles/text through tools
  - sample current viewport metadata
- Add retry/backoff and friendlier UI error states
- Store API key more carefully if this will be distributed to other users

## 7. Project Persistence

### What is still needed

- Add schema versioning to the project format
- Add migration logic for future changes
- Decide how much AI session state, if any, belongs in project files

### What is simplified right now

- The project format now stores sketch documents, masked faces, and interaction mode
- Legacy overlay strokes are still readable

### Why this matters

The project file format is now doing more than before. Without versioning, future changes will become brittle.

### Next implementation steps

- Add an explicit `project_version`
- Add migration helpers in `project_io.py`
- Add tests for loading older payloads with missing newer fields

## 8. Rendering and Viewport UX

### What is still needed

- Restore or replace the viewport navigation widget if desired
- Add better overlay rendering for selected entities and brush previews
- Improve visual hierarchy of sketch lines, handles, masked faces, and hover state
- Resize handling should be reviewed in real-world window resizing scenarios

### What changed

- The previous viewport overlay-bitmap sketch path is no longer the primary workflow
- The current viewport rendering overlays projected entity geometry

### Why this matters

The tool is now more capable than before, but the viewport still needs polish to communicate state clearly.

### Next implementation steps

- Add hover highlighting for faces and entities
- Add selected-entity color treatment
- Add brush preview ring
- Add optional axis/view gizmo back if useful

## 9. Build and Packaging

### What is still needed

- Trim the PyInstaller spec so it does not collect so many unnecessary modules
- Document the dev vs release build workflow in `README.md`
- Measure cold vs warm build times more deliberately
- Decide whether release should remain `onedir` or move back toward onefile

### What is simplified right now

- A dev spec and dev build script now exist
- Repeat builds reuse the fixed cache paths and are noticeably faster
- The current spec still pulls in a lot of hidden imports and extra package content

### Why this matters

The current dev build works, but the package is still heavier than it needs to be and rebuilds can likely be reduced further.

### Next implementation steps

- Replace broad `collect_all()` usage where possible with targeted hidden imports
- Remove unnecessary test/demo modules from package analysis
- Add a short build section to `README.md`

## 10. Testing

### What is still needed

- Add integration tests around the new paint and sketch workflows
- Add UI smoke coverage if possible
- Add more AI tool validation coverage
- Add packaging verification notes or scripts

### What is already covered

- Sketch plane math
- Sketch entity resizing basics
- Command undo/redo
- AI tool validation basics
- Project save/load with new fields
- STL import and self-test

### Next implementation steps

- Add tests for:
  - sketch bake on multi-face planar patches
  - brush masking behavior
  - erase behavior
  - sample tool
  - AI update/delete entity tools once added
  - repeated dev build cache reuse

## 11. Recommended Next Order

If continuing immediately, the safest order is:

1. Finish sketch UX and hide incomplete sketch tools
2. Complete command-stack coverage for all mutations
3. Improve brush drag behavior and add brush preview
4. Expand AI tools and add approval flow
5. Refactor `app.py` into smaller modules
6. Trim packaging size and update docs

## 12. Definition of “Working”

This feature set should be considered fully working when:

- Paint mode feels consistent during click and drag painting
- Sketch mode supports reliable creation, selection, resize, snapping, and bake on planar surfaces
- Undo/redo covers all user-visible mutations
- AI assistant can safely inspect state and perform common paint/sketch tasks with clear logs
- Project files round-trip all new sketch and mask data
- Dev and release builds are documented and reproducible
- STL import, app startup, and packaged EXE checks all continue to pass

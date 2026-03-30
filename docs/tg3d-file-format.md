# TG3D File Format

## Overview

`.tg3d` is a JSON-based project container used by Tungsten Texture Painter.
It stores:

1. The full mesh/project state (geometry, colors, sketch planes/entities, mask state).
2. Timeline metadata for Fusion-360-style history navigation.

## Top-level schema

```json
{
  "tg3d_version": 1,
  "model": { ... MeshModel project payload ... },
  "timeline": {
    "current_index": 3,
    "descriptions": ["Initial state", "Brush stroke", "Mask selected", "Bake sketch"],
    "snapshots": [ { ...project state at step 0... }, { ...step 1... }, ... ]
  }
}
```

## Fields

- `tg3d_version` (int): format version. Current value is `1`.
- `model` (object): current canonical project state, equivalent to previous JSON project format.
- `timeline.current_index` (int): currently active timeline step.
- `timeline.descriptions` (string[]): labels for each timeline step.
- `timeline.snapshots` (object[]): full project snapshots for each step in history.

## Timeline behavior

- Step 0 is always `Initial state`.
- Each command appends a new timeline snapshot.
- Undo/redo moves the current index backward/forward.
- If you move back in history and make a new edit, future steps are discarded and recalculated from that branch.

## Compatibility notes

- Legacy `.json` projects can still be opened.
- Saving from the UI now writes `.tg3d` and includes timeline metadata.

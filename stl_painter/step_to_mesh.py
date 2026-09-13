"""Run under a system Python with CadQuery: tessellate STEP per CAD face.

Invoked as::

    python step_to_mesh.py in.step out.npz [tolerance]

Each B-rep face is tessellated independently so the importer can map every
triangle back to its CAD face (``tri_cad``). The app then paints/picks whole
CAD faces and draws only CAD boundary edges, so rectangles and circles keep
their shape in the viewer even though the GPU/3MF mesh underneath is made of
triangles (required by OpenGL and the 3MF spec).

Output ``.npz`` entries:

* ``vertices`` - ``(N, 3)`` float64 mesh vertices
* ``faces`` - ``(M, 3)`` int64 triangle indices
* ``tri_cad`` - ``(M,)`` int64 index into ``cad_meta`` per triangle
* ``cad_meta`` - JSON list of ``{id, solid, name, surface, count}``
"""

import json
import sys

import numpy as np


def _face_vertices_and_triangles(face, tolerance):
    vertices, triangles = face.tessellate(tolerance)
    if not triangles:
        return None, None
    points = np.asarray([point.toTuple() for point in vertices], dtype=np.float64)
    tris = np.asarray(triangles, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3:
        return None, None
    if tris.ndim != 2 or tris.shape[1] != 3:
        return None, None
    if len(points) == 0 or len(tris) == 0:
        return None, None
    if tris.min() < 0 or tris.max() >= len(points):
        return None, None
    return points, tris


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("Usage: step_to_mesh.py in.step out.npz [tolerance]")
    source = sys.argv[1]
    output = sys.argv[2]
    tolerance = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
    if tolerance <= 0:
        raise SystemExit(f"Invalid tessellation tolerance: {sys.argv[3]!r}")

    from cadquery import importers

    assembly = importers.importStep(source)
    solids = assembly.solids().all()
    units: list[tuple[int, list]] = []
    if solids:
        for solid_index, solid in enumerate(solids):
            units.append((solid_index, solid.faces().all()))
    else:
        # Shell/surface-only STEP files have no solids; treat all faces as one.
        fallback = assembly.faces().all()
        if not fallback:
            raise SystemExit(f"No faces found in STEP file: {source}")
        units.append((0, fallback))

    all_vertices: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    tri_cad: list[int] = []
    cad_meta: list[dict] = []
    vertex_offset = 0
    for solid_index, faces in units:
        for face_index, wrapper in enumerate(faces):
            face = wrapper.val()
            points, tris = _face_vertices_and_triangles(face, tolerance)
            if points is None or tris is None:
                continue
            try:
                surface = str(face.geomType())
            except Exception:
                surface = "UNKNOWN"
            cad_id = f"cad_s{solid_index}_f{face_index}"
            cad_meta.append(
                {
                    "id": cad_id,
                    "solid": int(solid_index),
                    "name": f"solid_{solid_index} / face_{face_index}",
                    "surface": surface,
                    "count": int(len(tris)),
                }
            )
            all_vertices.append(points)
            all_faces.append(tris + vertex_offset)
            tri_cad.extend([len(cad_meta) - 1] * len(tris))
            vertex_offset += len(points)

    if not all_faces:
        raise SystemExit(f"STEP tessellation produced no triangles: {source}")
    vertices = np.vstack(all_vertices).astype(np.float64)
    faces = np.vstack(all_faces).astype(np.int64)
    tri_cad_arr = np.asarray(tri_cad, dtype=np.int64)
    np.savez(
        output,
        vertices=vertices,
        faces=faces,
        tri_cad=tri_cad_arr,
        cad_meta=np.array(json.dumps(cad_meta)),
    )
    print(
        f"WROTE {output} triangles={len(faces)} "
        f"cad_faces={len(cad_meta)} solids={len(units)}"
    )


main()

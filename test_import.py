import sys

print("Python version:", sys.version)
print("Python path:", sys.path[:3])

try:
    import numpy

    print("NumPy version:", numpy.__version__)
    print("NumPy location:", numpy.__file__)

    import numpy._core._exceptions

    print("_exceptions imported OK")
except Exception as e:
    print("NumPy import error:", e)
    import traceback

    traceback.print_exc()

try:
    import trimesh

    print("Trimesh version:", trimesh.__version__)
    print("Trimesh location:", trimesh.__file__)
except Exception as e:
    print("Trimesh import error:", e)
    import traceback

    traceback.print_exc()

try:
    import dearpygui.dearpygui as dpg

    print("DearPyGui imported OK")
except Exception as e:
    print("DearPyGui import error:", e)
    import traceback

    traceback.print_exc()

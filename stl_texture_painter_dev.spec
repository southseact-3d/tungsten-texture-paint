# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_submodules


project_root = Path.cwd()
datas = [
    (str(project_root / "stl_painter" / "shaders"), "stl_painter/shaders"),
    (str(project_root / "stl_painter" / "assets"), "stl_painter/assets"),
    (str(project_root / "stl_painter" / "blend_to_glb.py"), "stl_painter"),
    (str(project_root / "stl_painter" / "step_to_mesh.py"), "stl_painter"),
]
binaries = []
hiddenimports = [
    "dearpygui.dearpygui",
    "moderngl",
    "moderngl_window",
    "trimesh",
    "glcontext",
    "requests",
    "lxml",
    "lxml.etree",
]

# Collect dynamically imported modules and data required at runtime.
for package_name in (
    "numpy",
    "trimesh",
    "dearpygui",
    "moderngl",
    "moderngl_window",
    "glcontext",
    "pkg_resources",
    "setuptools",
    "requests",
    "urllib3",
    "certifi",
    "charset_normalizer",
    "lxml",
    "scipy",
):
    pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all(package_name)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hiddenimports

hiddenimports += collect_submodules("pkg_resources")
hiddenimports += collect_submodules("setuptools")
hiddenimports += collect_submodules("numpy")
hiddenimports += collect_submodules("requests")

a = Analysis(
    ["main.py"],
    pathex=[str(project_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(project_root / "fix_pkg_resources.py")],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="STLTexturePainterDev",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="STLTexturePainterDev",
)

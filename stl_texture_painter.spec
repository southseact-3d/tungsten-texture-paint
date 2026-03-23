# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_submodules


project_root = Path.cwd()
datas = [
    (str(project_root / "stl_painter" / "shaders"), "stl_painter/shaders"),
    (str(project_root / "stl_painter" / "assets"), "stl_painter/assets"),
]
binaries = []
hiddenimports = [
    "dearpygui.dearpygui",
    "moderngl",
    "moderngl_window",
    "trimesh",
    "glcontext",
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
):
    pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all(package_name)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hiddenimports

hiddenimports += collect_submodules("pkg_resources")
hiddenimports += collect_submodules("setuptools")
hiddenimports += collect_submodules("numpy")


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
    name="STLTexturePainter",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
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
    upx=True,
    upx_exclude=[],
    name="STLTexturePainter",
)

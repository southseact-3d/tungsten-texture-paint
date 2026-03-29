$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$PyInstaller = Join-Path $Root ".venv\Scripts\pyinstaller.exe"
$WorkPath = Join-Path $Root ".pyinstaller-cache\work-release"
$DistPath = Join-Path $Root "dist"
$Spec = Join-Path $Root "stl_texture_painter.spec"

if (-not (Test-Path $PyInstaller)) {
    throw "PyInstaller not found at $PyInstaller"
}

& $PyInstaller `
    --noconfirm `
    --workpath $WorkPath `
    --distpath $DistPath `
    $Spec
if ($LASTEXITCODE -ne 0) { throw "PyInstaller release build failed with exit code $LASTEXITCODE" }

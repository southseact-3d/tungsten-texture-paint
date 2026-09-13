$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$WorkPath = Join-Path $Root ".pyinstaller-cache\work-release"
$DistPath = Join-Path $Root "dist"
$Spec = Join-Path $Root "stl_texture_painter.spec"

if (-not (Test-Path $Python)) {
    throw "Virtual environment Python not found at $Python"
}

& $Python -m PyInstaller `
    --noconfirm `
    --workpath $WorkPath `
    --distpath $DistPath `
    $Spec
if ($LASTEXITCODE -ne 0) { throw "PyInstaller release build failed with exit code $LASTEXITCODE" }

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$PyInstaller = Join-Path $Root ".venv\Scripts\pyinstaller.exe"
$WorkPath = Join-Path $Root ".pyinstaller-cache\work-dev"
$DistPath = Join-Path $Root ".pyinstaller-cache\dist-dev"
$Spec = Join-Path $Root "stl_texture_painter_dev.spec"

if (-not (Test-Path $Python)) {
    throw "Virtual environment Python not found at $Python"
}

& $PyInstaller `
    --noconfirm `
    --workpath $WorkPath `
    --distpath $DistPath `
    $Spec
if ($LASTEXITCODE -ne 0) { throw "PyInstaller dev build failed with exit code $LASTEXITCODE" }

& $Python (Join-Path $Root "main.py") --self-test-stl (Join-Path $Root "dart.stl")
if ($LASTEXITCODE -ne 0) { throw "Post-build self-test failed with exit code $LASTEXITCODE" }

<#
Builds the self-contained app folder that the installer packs (dist\audio2midi):

  runtime\main   portable Python 3.12 + requirements.txt (+ ADTOF)   <- the app and the engine
  runtime\asr    portable Python 3.12 + requirements-asr.txt          <- lyrics (Whisper)
  src\           the audio2midi code
  bin\           ffmpeg
  launch.pyw     starts the app (the shortcut runs runtime\main\pythonw.exe launch.pyw)

Needs internet the first time (Python, wheels). Run from the repo root:
  powershell -ExecutionPolicy Bypass -File packaging\build_runtime.ps1 [-Out dist\audio2midi] [-SkipAsr] [-AppOnly]

  -AppOnly  refreshes only src\, bin\ and launch.pyw in an existing build (seconds instead of minutes)
#>
param(
    [string]$Out = "dist\audio2midi",
    [switch]$SkipAsr,
    [switch]$AppOnly
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # Invoke-WebRequest is very slow with the progress bar
# Never let packages of the build machine's own Python (the "user site") leak in: pip would skip them as
# "already satisfied" and the runtime would be missing them on any other PC.
$env:PYTHONNOUSERSITE = "1"

# Pinned: Python 3.12 from python-build-standalone (a Python that can be moved to another folder or PC)
$PbsTag = "20261110"
$PyVer = "3.12.16"
$AdtofCommit = "85c192e"
$AdtofRepo = "https://github.com/xavriley/ADTOF-pytorch.git"

$Repo = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location $Repo
$Out = if ([IO.Path]::IsPathRooted($Out)) { $Out } else { Join-Path $Repo $Out }
$Cache = Join-Path $Repo "packaging\cache"
New-Item -ItemType Directory -Force $Cache | Out-Null

function Step($msg) { Write-Host "`n== $msg" -ForegroundColor Cyan }
function Check($what) { if ($LASTEXITCODE -ne 0) { throw "$what failed (exit $LASTEXITCODE)" } }

function Copy-App {
    Step "app"
    foreach ($d in "src", "bin") {
        $dst = Join-Path $Out $d
        if (Test-Path $dst) { Remove-Item -Recurse -Force $dst }
        Copy-Item (Join-Path $Repo $d) $dst -Recurse
    }
    Get-ChildItem (Join-Path $Out "src") -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
    Copy-Item (Join-Path $Repo "launch.pyw") $Out -Force
}

if ($AppOnly) {
    if (-not (Test-Path (Join-Path $Out "runtime\main\python.exe"))) { throw "No build in $Out yet: run without -AppOnly first" }
    Copy-App
    Write-Host "`nApp refreshed in $Out" -ForegroundColor Green
    return
}

# ---- 1. the portable Python, verified against the published checksum
Step "Python $PyVer"
$name ="cpython-$PyVer+$PbsTag-x86_64-pc-windows-msvc-install_only.tar.gz"
$base = "https://github.com/astral-sh/python-build-standalone/releases/download/$PbsTag"
$archive = Join-Path $Cache $name
if (-not (Test-Path $archive)) {
    Invoke-WebRequest -UseBasicParsing "$base/$([uri]::EscapeDataString($name))" -OutFile $archive
}
$sums = (New-Object Net.WebClient).DownloadString("$base/SHA256SUMS")   # Invoke-WebRequest chokes on it in PS 5.1
$line = ($sums -split "`n" | Where-Object { $_ -match [regex]::Escape($name) + "\s*$" } | Select-Object -First 1)
if (-not $line) { throw "No checksum published for $name" }
$want = ($line.Trim() -split "\s+")[0].ToLower()
$got = (Get-FileHash $archive -Algorithm SHA256).Hash.ToLower()
if ($got -ne $want) { Remove-Item $archive; throw "Checksum mismatch for $name (got $got, want $want)" }
Write-Host "checksum ok ($($got.Substring(0,12))...)"

function New-Runtime($dest) {
    if (Test-Path $dest) { Remove-Item -Recurse -Force $dest }
    $tmp = Join-Path $Cache "extract"
    if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp }
    New-Item -ItemType Directory -Force $tmp | Out-Null
    tar -xzf $archive -C $tmp; Check "tar"
    New-Item -ItemType Directory -Force (Split-Path $dest) | Out-Null
    Move-Item (Join-Path $tmp "python") $dest
    Remove-Item -Recurse -Force $tmp
}

# ---- 2. the main runtime
Step "runtime\main"
$main = Join-Path $Out "runtime\main"
New-Runtime $main
$py = Join-Path $main "python.exe"
& $py -m pip install --no-warn-script-location --disable-pip-version-check -r requirements.txt; Check "pip install (main)"

$adtof = Join-Path $Repo "third_party\ADTOF-pytorch"
if (Test-Path $adtof) {
    & $py -m pip install --no-warn-script-location --disable-pip-version-check --no-deps $adtof; Check "pip install ADTOF"
} else {
    & $py -m pip install --no-warn-script-location --disable-pip-version-check --no-deps "git+$AdtofRepo@$AdtofCommit"; Check "pip install ADTOF (git)"
}

# ---- 3. the lyrics runtime
if (-not $SkipAsr) {
    Step "runtime\asr"
    $asr = Join-Path $Out "runtime\asr"
    New-Runtime $asr
    & (Join-Path $asr "python.exe") -m pip install --no-warn-script-location --disable-pip-version-check -r requirements-asr.txt; Check "pip install (asr)"
}

# ---- 4. the app itself
Copy-App

# ---- 5. tidy up
Step "tidy"
# pip's console scripts (pip.exe, ...) hold the absolute path of the build folder and would be broken after
# the move; the app never uses them (everything runs as `python -m ...`)
Get-ChildItem $Out -Recurse -Directory -Filter "Scripts" | Where-Object { $_.Parent.Name -in "main", "asr" } | Remove-Item -Recurse -Force

# ---- 6. check that each runtime stands on its own (-s: ignore the user site, as the app does)
Step "verify"
& $py -s -m pip check; Check "pip check (main)"
& $py -s -c "import torch, onnxruntime, librosa, pretty_midi, adtof_pytorch, audio_separator, PySide6; print('main ok, torch', torch.__version__, 'cuda', torch.cuda.is_available())"; Check "import check (main)"
if (-not $SkipAsr) {
    $apy = Join-Path $Out "runtime\asr\python.exe"
    & $apy -s -m pip check; Check "pip check (asr)"
    & $apy -s -c "import faster_whisper, ctranslate2, av; print('asr ok')"; Check "import check (asr)"
}

$size = (Get-ChildItem $Out -Recurse -File | Measure-Object Length -Sum).Sum / 1GB
Write-Host ("`nReady: {0}  ({1:N2} GB)" -f $Out, $size) -ForegroundColor Green

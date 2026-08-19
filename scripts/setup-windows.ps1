<#
.SYNOPSIS
    Sets RosettaAI up on Windows with CUDA acceleration.

.DESCRIPTION
    Creates a virtual environment, installs RosettaAI with the Whisper engine
    and the NVIDIA CUDA runtime libraries that CTranslate2 needs, then runs the
    environment check. Re-runnable: existing installs are upgraded in place.

.PARAMETER Nvidia
    Also install the NeMo engines (Parakeet TDT and Canary-Qwen). This pulls in
    PyTorch and the NeMo toolkit, several GB in total.

.PARAMETER Python
    Python launcher to build the virtual environment with. Defaults to "python".

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\setup-windows.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\setup-windows.ps1 -Nvidia

.NOTES
    Keep this file ASCII-only and saved as UTF-8 with a BOM. Windows PowerShell
    5.1 decodes BOM-less files as ANSI, which turns characters such as an em
    dash into a curly quote - and PowerShell treats curly quotes as string
    delimiters, so the script fails to parse.
#>
[CmdletBinding()]
param(
    [switch]$Nvidia,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Invoke-Step {
    <#  Runs a native command and stops the script if it fails.
        $ErrorActionPreference does not apply to native executables, so exit
        codes have to be checked by hand. #>
    param(
        [Parameter(Mandatory)][string]$Description,
        [Parameter(Mandatory)][string]$Executable,
        [Parameter(Mandatory)][string[]]$Arguments
    )

    Write-Host $Description -ForegroundColor Cyan
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed (exit code $LASTEXITCODE)."
    }
}

Write-Host ""
Write-Host "RosettaAI setup" -ForegroundColor Magenta
Write-Host "  Project: $root"
Write-Host ""

# --- Python ---------------------------------------------------------------
$version = & $Python -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
if ($LASTEXITCODE -ne 0 -or -not $version) {
    throw "Python 3.10 or newer is required and must be on PATH. Install it from https://python.org and re-run this script."
}
if ([version]$version -lt [version]"3.10") {
    throw "Python $version found, but RosettaAI needs 3.10 or newer."
}
Write-Host "  Python:  $version"

# --- GPU ------------------------------------------------------------------
$gpu = $null
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $gpu = (& nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1)
}
if ($gpu) {
    Write-Host "  GPU:     $gpu"
} else {
    Write-Host "  GPU:     none detected (transcription will run on the CPU)" -ForegroundColor Yellow
}
Write-Host ""

# --- Virtual environment --------------------------------------------------
if (-not (Test-Path ".venv")) {
    Invoke-Step "Creating virtual environment (.venv)" $Python @("-m", "venv", ".venv")
}

$venvPython = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    throw "The virtual environment looks broken: $venvPython is missing. Delete the .venv folder and re-run this script."
}

Invoke-Step "Upgrading pip" $venvPython @("-m", "pip", "install", "--upgrade", "pip", "--quiet")
Invoke-Step "Installing RosettaAI and the Whisper engine" $venvPython @("-m", "pip", "install", "-e", ".[whisper]")

if ($gpu) {
    Invoke-Step "Installing the CUDA runtime libraries CTranslate2 needs" $venvPython @(
        "-m", "pip", "install", "nvidia-cublas-cu12", "nvidia-cudnn-cu12>=9,<10"
    )
}

if ($Nvidia) {
    Invoke-Step "Installing PyTorch with CUDA 12.4 (large download)" $venvPython @(
        "-m", "pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/cu124"
    )
    Invoke-Step "Installing the NeMo engines (Parakeet and Canary-Qwen)" $venvPython @(
        "-m", "pip", "install", "-e", ".[nvidia]"
    )
}

# --- Verify ---------------------------------------------------------------
Write-Host ""
Write-Host "Checking the environment" -ForegroundColor Cyan
& $venvPython -m rosetta doctor

Write-Host ""
Write-Host "Done. Start RosettaAI with:" -ForegroundColor Green
Write-Host "    .venv\Scripts\rosetta.exe"
Write-Host ""
Write-Host "Or transcribe straight from the command line:"
Write-Host "    .venv\Scripts\rosetta.exe transcribe https://www.youtube.com/watch?v=..."
Write-Host ""

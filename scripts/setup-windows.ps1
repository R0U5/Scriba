<#
.SYNOPSIS
    Sets RosettaAI up on Windows with CUDA acceleration.

.DESCRIPTION
    Creates a virtual environment, installs RosettaAI with the Whisper engine
    and the NVIDIA CUDA runtime libraries that CTranslate2 needs, then runs the
    environment check. Re-runnable: existing installs are upgraded in place.

.PARAMETER Nvidia
    Also install the NeMo engines (Parakeet TDT and Canary-Qwen). This pulls in
    PyTorch and the NeMo toolkit — several GB.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\setup-windows.ps1 -Nvidia
#>
[CmdletBinding()]
param(
    [switch]$Nvidia,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "RosettaAI setup" -ForegroundColor Magenta
Write-Host "  Project: $root`n"

& $Python --version
if ($LASTEXITCODE -ne 0) { throw "Python 3.10+ is required and must be on PATH." }

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment (.venv)…" -ForegroundColor Cyan
    & $Python -m venv .venv
}

$venvPython = Join-Path $root ".venv\Scripts\python.exe"
& $venvPython -m pip install --upgrade pip --quiet

Write-Host "Installing RosettaAI + Whisper engine…" -ForegroundColor Cyan
& $venvPython -m pip install -e ".[whisper]"

Write-Host "Installing CUDA runtime libraries for CTranslate2…" -ForegroundColor Cyan
& $venvPython -m pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12>=9,<10"

if ($Nvidia) {
    Write-Host "Installing PyTorch (CUDA 12.4) + NeMo engines — this is a large download…" -ForegroundColor Cyan
    & $venvPython -m pip install torch --index-url https://download.pytorch.org/whl/cu124
    & $venvPython -m pip install -e ".[nvidia]"
}

Write-Host "`nChecking the environment…" -ForegroundColor Cyan
& $venvPython -m rosetta doctor

Write-Host "`nDone. Start RosettaAI with:" -ForegroundColor Green
Write-Host "    .venv\Scripts\rosetta.exe`n"

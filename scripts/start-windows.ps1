<#
.SYNOPSIS
    Starts RosettaAI on Windows, installing whatever is missing.

.DESCRIPTION
    Uses uv (https://docs.astral.sh/uv) for everything: it fetches a suitable
    Python if the machine has none, creates the project virtual environment and
    installs the dependencies. The CUDA runtime libraries are added only when an
    NVIDIA GPU is present.

    The first run takes a couple of minutes. Later runs start immediately.

.EXAMPLE
    .\run.bat

.EXAMPLE
    .\run.bat transcribe https://www.youtube.com/watch?v=aircAruvnKk

.EXAMPLE
    .\run.bat install-nvidia

.NOTES
    Keep this file ASCII-only and saved as UTF-8 with a BOM. Windows PowerShell
    5.1 decodes BOM-less files as ANSI, which turns characters such as an em
    dash into a curly quote, and PowerShell treats curly quotes as string
    delimiters - so one stray character stops the whole script from parsing.
#>
[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Arguments
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$PytorchCudaIndex = "https://download.pytorch.org/whl/cu124"
$NvidiaMarker = ".nvidia-enabled"

function Find-Uv {
    $onPath = Get-Command uv -ErrorAction SilentlyContinue
    if ($onPath) { return $onPath.Source }

    $candidates = @(
        (Join-Path $env:USERPROFILE ".local\bin\uv.exe"),
        (Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Links\uv.exe"),
        (Join-Path $env:USERPROFILE ".cargo\bin\uv.exe")
    )
    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path $candidate)) { return $candidate }
    }
    return $null
}

function Install-Uv {
    Write-Host "First run: installing uv, the package manager RosettaAI uses (about 15 MB)." -ForegroundColor Cyan
    try {
        Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    } catch {
        Write-Host "The uv installer could not be downloaded. Trying winget instead." -ForegroundColor Yellow
        if (Get-Command winget -ErrorAction SilentlyContinue) {
            & winget install --id astral-sh.uv -e --accept-source-agreements --accept-package-agreements
        }
    }
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

function Test-Gpu {
    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { return $false }
    & nvidia-smi -L > $null 2>&1
    return ($LASTEXITCODE -eq 0)
}

function Test-TorchCuda {
    param([string]$Uv)
    $answer = & $Uv run --no-sync --quiet python -c "import torch; print(torch.cuda.is_available())" 2>$null
    return ($LASTEXITCODE -eq 0 -and "$answer".Trim() -eq "True")
}

$uv = Find-Uv
if (-not $uv) {
    Install-Uv
    $uv = Find-Uv
    if (-not $uv) {
        throw "uv could not be installed automatically. Install it from https://docs.astral.sh/uv/ and run this again."
    }
}

$hasGpu = Test-Gpu

# --- opt-in install of the NeMo engines ------------------------------------
if ($Arguments -and $Arguments[0] -eq "install-nvidia") {
    Write-Host "Installing PyTorch and NVIDIA NeMo. This is several GB and runs once." -ForegroundColor Cyan
    if ($hasGpu) { & $uv sync --extra nvidia --extra cuda } else { & $uv sync --extra nvidia }
    if ($LASTEXITCODE -ne 0) { throw "The NeMo install failed (exit code $LASTEXITCODE)." }

    if ($hasGpu -and -not (Test-TorchCuda $uv)) {
        # PyPI ships a CPU-only torch wheel on Windows, which would leave these
        # engines running on the CPU without ever saying so.
        Write-Host "PyTorch has no CUDA support; reinstalling it from NVIDIA's index..." -ForegroundColor Yellow
        & $uv pip install --upgrade torch --index-url $PytorchCudaIndex
        if (-not (Test-TorchCuda $uv)) {
            Write-Host ""
            Write-Host "PyTorch still reports no CUDA device. Parakeet and Canary-Qwen need one." -ForegroundColor Red
            Write-Host "Check that your NVIDIA driver is current, then run:"
            Write-Host "    $uv pip install --upgrade --force-reinstall torch --index-url $PytorchCudaIndex"
            exit 1
        }
    }

    New-Item -ItemType File -Path $NvidiaMarker -Force > $null
    Write-Host "Done. Parakeet and Canary-Qwen now appear in the model menu." -ForegroundColor Green
    exit 0
}

# --- normal launch ----------------------------------------------------------
if (-not $Arguments -or $Arguments.Count -eq 0) {
    Write-Host ""
    Write-Host "Starting RosettaAI. Your browser will open at http://localhost:8420" -ForegroundColor Magenta
    Write-Host "Press Ctrl+C in this window to stop it." -ForegroundColor DarkGray
    Write-Host ""
}

if (Test-Path $NvidiaMarker) {
    # The environment already holds the NeMo engines (and possibly a hand-placed
    # CUDA build of torch); --no-sync keeps uv from undoing that.
    & $uv run --no-sync --quiet rosetta @Arguments
} elseif ($hasGpu) {
    & $uv run --quiet --extra cuda rosetta @Arguments
} else {
    & $uv run --quiet rosetta @Arguments
}
exit $LASTEXITCODE

# Run web dashboard with venv (creates .venv and installs deps if needed)
$ErrorActionPreference = "Stop"
$root = if ($PSScriptRoot) { $PSScriptRoot } else { Get-Location }

if (-not (Test-Path "$root\.venv")) {
    Write-Host "Creating virtual environment (.venv)..."
    python -m venv "$root\.venv"
    Write-Host "Done."
}
Write-Host "Activating venv..."
& "$root\.venv\Scripts\Activate.ps1"
if (-not $?) {
    Write-Host "Run: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser"
    exit 1
}

Write-Host "Installing dependencies (first run can take 5+ min for torch)..."
pip install -r "$root\requirements.txt"
Write-Host "Starting server at http://localhost:8000"
Set-Location $root
uvicorn api_server:app --host 0.0.0.0 --port 8000

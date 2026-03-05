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
Set-Location $root

Write-Host "Starting server at http://localhost:8000"
$serverJob = Start-Job -ScriptBlock {
    Set-Location $using:root
    & "$using:root\.venv\Scripts\python.exe" -m uvicorn api_server:app --host 0.0.0.0 --port 8000
}
try {
    Start-Sleep -Seconds 2
    if (-not (Get-Command ngrok -ErrorAction SilentlyContinue)) {
        Write-Host "ngrok not found in PATH. Install from https://ngrok.com/download or run: uvicorn api_server:app --host 0.0.0.0 --port 8000"
        $serverJob | Wait-Job
    } else {
        Write-Host "Starting ngrok tunnel to http://localhost:8000"
        & ngrok http 8000
    }
} finally {
    Stop-Job $serverJob -ErrorAction SilentlyContinue
    Remove-Job $serverJob -Force -ErrorAction SilentlyContinue
}

<#
Infra setup for AI Trader on Windows (PowerShell)
Usage: .\infra\setup.ps1
#>

Write-Host "=== AI Trader Windows Infra Setup ==="

# 1. Check PostgreSQL service
$serviceNames = @("postgresql-x64-15", "postgresql-x64-14", "postgresql")
$found = $false
foreach ($svc in $serviceNames) {
    if (Get-Service -Name $svc -ErrorAction SilentlyContinue) {
        $found = $true
        $serviceName = $svc
        break
    }
}
if (-not $found) {
    Write-Warning "PostgreSQL service not found. Please install via Windows installer: https://www.postgresql.org/download/windows/"
} else {
    $status = (Get-Service -Name $serviceName).Status
    if ($status -ne 'Running') {
        Write-Host "Starting PostgreSQL service '$serviceName'..."
        Start-Service -Name $serviceName
    } else {
        Write-Host "PostgreSQL service '$serviceName' is already running."
    }
}

# 2. Setup Python virtual environment
if (-not (Test-Path -Path .\venv)) {
    Write-Host "Creating virtual environment..."
    python -m venv venv
}
Write-Host "Activating virtual environment..."
& .\venv\Scripts\Activate.ps1

Write-Host "Installing dependencies..."
pip install -r requirements.txt

# 3. Copy .env if missing
if (-not (Test-Path -Path .\.env)) {
    Copy-Item .\.env.example .\.env
    Write-Host "Copied .env.example to .env. Please update with your credentials."
}

# 4. Initialize database schema
Write-Host "Initializing database schema..."
python -m database.init_db

# 5. Git initialization
if (-not (Test-Path -Path .\.git)) {
    Write-Host "Initializing Git repository..."
    git init
    git add .
    git commit -m "Initial infra setup on Windows"
    Write-Host "Add remote: git remote add origin <repo-url>"
}

Write-Host "=== Windows Infra Setup Complete ==="

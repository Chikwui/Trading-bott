<#
Infra setup for AI Trader on Windows (PowerShell)
Usage: .\infra\setup.ps1
#>

Write-Host "=== AI Trader Windows Infra Setup ==="

# 1. Check PostgreSQL service
$serviceNames = @("postgresql-x64-17", "postgresql-x64-15", "postgresql-x64-14", "postgresql")
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

# Verify PostgreSQL version
try {
    $verOutput = & psql --version
    if ($verOutput -match '(\\d+)\\.(\\d+)') {
        $major = $matches[1]
        if ($major -ne '17') {
            Write-Warning "Expected PostgreSQL 17.x, found version $verOutput"
        } else {
            Write-Host "PostgreSQL version $verOutput verified."
        }
    } else {
        Write-Warning "Could not parse PostgreSQL version: $verOutput"
    }
} catch {
    Write-Warning "psql command not found. Ensure PostgreSQL 17 client tools are installed and in PATH."
}

# 2. Setup Python virtual environment
if (-not (Test-Path -Path .\venv)) {
    Write-Host "Creating virtual environment..."
    python -m venv venv
}
Write-Host "Activating virtual environment..."
& .\venv\Scripts\Activate.ps1

Write-Host "Upgrading pip, setuptools, wheel..."
python -m pip install --upgrade pip setuptools wheel

Write-Host "Determining Python minor version..."
$pv = & python -c "import sys; print(sys.version_info.minor)"
if ($pv -eq 13) {
    Write-Warning "Skipping PyArrow installation on Python 3.13 on Windows; use Python 3.10–3.12 or install PyArrow via conda."
} else {
    Write-Host "Installing PyArrow binary wheel..."
    python -m pip install --only-binary=:all: "pyarrow>=20.0.0"
}

Write-Host "Installing dependencies (excluding PyArrow)..."
(Get-Content requirements.txt | Where-Object {$_ -notmatch '^pyarrow'}) | Set-Content temp_requirements.txt
python -m pip install -r temp_requirements.txt
Remove-Item temp_requirements.txt

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
    Write-Host 'Add remote: git remote add origin <repo-url>'
}

Write-Host '=== Windows Infra Setup Complete ==='

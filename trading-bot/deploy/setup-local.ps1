<#
.SYNOPSIS
    Local deployment script for the Trading Bot application.
.DESCRIPTION
    This script sets up and starts all required services for the Trading Bot
    to run locally without Docker.
.NOTES
    File Name      : setup-local.ps1
    Prerequisites  : Windows with PowerShell 5.1+, Python 3.8+
#>

# Stop script on first error
$ErrorActionPreference = "Stop"

# Configuration
$config = @{
    # Redis configuration
    Redis = @{
        Install = $true
        DownloadUrl = "https://github.com/microsoftarchive/redis/releases/download/win-3.0.504/Redis-x64-3.0.504.msi"
        InstallPath = "C:\Program Files\Redis"
        ServiceName = "Redis"
        Port = 6379
    }
    
    # RabbitMQ configuration
    RabbitMQ = @{
        Install = $true
        DownloadUrl = "https://github.com/rabbitmq/rabbitmq-server/releases/download/v3.10.10/rabbitmq-server-3.10.10.exe"
        InstallPath = "C:\Program Files\RabbitMQ Server\rabbitmq_server-3.10.10"
        ServiceName = "RabbitMQ"
        Port = 5672
    }
    
    # InfluxDB configuration
    InfluxDB = @{
        Install = $true
        DownloadUrl = "https://dl.influxdata.com/influxdb/releases/influxdb-1.8.10_windows_amd64.zip"
        InstallPath = "C:\influxdb"
        ServiceName = "InfluxDB"
        Port = 8086
    }
    
    # Application configuration
    App = @{
        PythonExe = "python"
        RequirementsFiles = @(
            "requirements.txt",
            "requirements-ml.txt",
            "requirements-test.txt"
        )
        MainScript = "app.py"
    }
}

function Write-Header {
    param([string]$Message)
    Write-Host "`n=== $($Message.ToUpper()) ===" -ForegroundColor Cyan
}

function Test-CommandExists {
    param([string]$Command)
    try {
        $null = Get-Command $Command -ErrorAction Stop
        return $true
    } catch {
        return $false
    }
}

function Install-Redis {
    param($Config)
    
    Write-Header "Checking Redis Installation"
    
    # Check if Redis is already installed and running
    $redisService = Get-Service -Name $Config.Redis.ServiceName -ErrorAction SilentlyContinue
    if ($redisService -and $redisService.Status -eq 'Running') {
        Write-Host "Redis is already installed and running." -ForegroundColor Green
        return
    }
    
    if (-not $Config.Redis.Install) {
        Write-Host "Skipping Redis installation as per configuration." -ForegroundColor Yellow
        return
    }
    
    # Download Redis installer
    $installerPath = "$env:TEMP\redis_installer.msi"
    Write-Host "Downloading Redis installer..."
    Invoke-WebRequest -Uri $Config.Redis.DownloadUrl -OutFile $installerPath
    
    # Install Redis
    Write-Host "Installing Redis..."
    Start-Process -FilePath "msiexec.exe" -ArgumentList "/i `"$installerPath`" /qn" -Wait -NoNewWindow
    
    # Start Redis service
    Start-Service -Name $Config.Redis.ServiceName -ErrorAction SilentlyContinue
    
    # Verify Redis is running
    $redisService = Get-Service -Name $Config.Redis.ServiceName
    if ($redisService.Status -ne 'Running') {
        Start-Service -Name $Config.Redis.ServiceName
    }
    
    Write-Host "Redis installed and started successfully." -ForegroundColor Green
}

function Install-RabbitMQ {
    param($Config)
    
    Write-Header "Checking RabbitMQ Installation"
    
    # Check if RabbitMQ is already installed and running
    $rabbitService = Get-Service -Name $Config.RabbitMQ.ServiceName -ErrorAction SilentlyContinue
    if ($rabbitService -and $rabbitService.Status -eq 'Running') {
        Write-Host "RabbitMQ is already installed and running." -ForegroundColor Green
        return
    }
    
    if (-not $Config.RabbitMQ.Install) {
        Write-Host "Skipping RabbitMQ installation as per configuration." -ForegroundColor Yellow
        return
    }
    
    # Download RabbitMQ installer
    $installerPath = "$env:TEMP\rabbitmq_installer.exe"
    Write-Host "Downloading RabbitMQ installer..."
    Invoke-WebRequest -Uri $Config.RabbitMQ.DownloadUrl -OutFile $installerPath
    
    # Install RabbitMQ
    Write-Host "Installing RabbitMQ..."
    Start-Process -FilePath $installerPath -ArgumentList "/S" -Wait -NoNewWindow
    
    # Add RabbitMQ to PATH
    $rabbitPath = $Config.RabbitMQ.InstallPath
    $env:Path += ";$rabbitPath\sbin"
    [System.Environment]::SetEnvironmentVariable("Path", $env:Path, [System.EnvironmentVariableTarget]::Machine)
    
    # Enable management plugin
    & "$rabbitPath\sbin\rabbitmq-plugins.bat" enable rabbitmq_management
    
    # Start RabbitMQ service
    Start-Service -Name $Config.RabbitMQ.ServiceName -ErrorAction SilentlyContinue
    
    Write-Host "RabbitMQ installed and started successfully." -ForegroundColor Green
    Write-Host "Management UI available at: http://localhost:15672" -ForegroundColor Cyan
    Write-Host "Default credentials: guest/guest" -ForegroundColor Cyan
}

function Install-InfluxDB {
    param($Config)
    
    Write-Header "Checking InfluxDB Installation"
    
    # Check if InfluxDB is already installed and running
    $influxProcess = Get-Process -Name "influxd" -ErrorAction SilentlyContinue
    if ($influxProcess) {
        Write-Host "InfluxDB is already running." -ForegroundColor Green
        return
    }
    
    if (-not $Config.InfluxDB.Install) {
        Write-Host "Skipping InfluxDB installation as per configuration." -ForegroundColor Yellow
        return
    }
    
    # Create installation directory
    $installPath = $Config.InfluxDB.InstallPath
    if (-not (Test-Path $installPath)) {
        New-Item -ItemType Directory -Path $installPath -Force | Out-Null
    }
    
    # Download InfluxDB
    $zipPath = "$env:TEMP\influxdb.zip"
    Write-Host "Downloading InfluxDB..."
    Invoke-WebRequest -Uri $Config.InfluxDB.DownloadUrl -OutFile $zipPath
    
    # Extract InfluxDB
    Write-Host "Installing InfluxDB..."
    Expand-Archive -Path $zipPath -DestinationPath $installPath -Force
    
    # Start InfluxDB
    $influxPath = Join-Path $installPath (Get-ChildItem -Path $installPath -Directory | Select-Object -First 1).Name
    $influxExe = Join-Path $influxPath "influxd.exe"
    
    # Start InfluxDB in a new window
    Start-Process -FilePath $influxExe -ArgumentList "-config $influxPath\influxdb.conf" -NoNewWindow
    
    # Wait for InfluxDB to start
    Start-Sleep -Seconds 5
    
    # Create default database
    try {
        & "$influxPath\influx.exe" -execute "CREATE DATABASE trading_metrics"
        Write-Host "InfluxDB installed and started successfully." -ForegroundColor Green
        Write-Host "InfluxDB available at: http://localhost:8086" -ForegroundColor Cyan
    } catch {
        Write-Host "InfluxDB started but there was an error creating the default database." -ForegroundColor Yellow
        Write-Host "Error: $_" -ForegroundColor Red
    }
}

function Install-PythonDependencies {
    param($Config)
    
    Write-Header "Installing Python Dependencies"
    
    # Check if Python is installed
    if (-not (Test-CommandExists "python")) {
        throw "Python is not installed or not in PATH. Please install Python 3.8 or later and try again."
    }
    
    # Upgrade pip
    Write-Host "Upgrading pip..."
    & $Config.App.PythonExe -m pip install --upgrade pip
    
    # Install requirements
    foreach ($requirementsFile in $Config.App.RequirementsFiles) {
        if (Test-Path $requirementsFile) {
            Write-Host "Installing requirements from $requirementsFile..."
            & $Config.App.PythonExe -m pip install -r $requirementsFile
        } else {
            Write-Host "Requirements file $requirementsFile not found, skipping..." -ForegroundColor Yellow
        }
    }
    
    Write-Host "Python dependencies installed successfully." -ForegroundColor Green
}

function Start-Application {
    param($Config)
    
    Write-Header "Starting Trading Bot Application"
    
    if (-not (Test-Path $Config.App.MainScript)) {
        Write-Host "Main application script $($Config.App.MainScript) not found." -ForegroundColor Red
        return
    }
    
    # Start the application in a new window
    Start-Process -FilePath $Config.App.PythonExe -ArgumentList $Config.App.MainScript -NoNewWindow
    
    Write-Host "Trading Bot application started." -ForegroundColor Green
    Write-Host "Application URL: http://localhost:8000" -ForegroundColor Cyan
}

# Main execution
Write-Host "`n=== TRADING BOT LOCAL SETUP ===`n" -ForegroundColor Green

# Check for admin privileges
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "This script requires administrator privileges. Please run as administrator." -ForegroundColor Red
    exit 1
}

try {
    # Install required services
    Install-Redis -Config $config
    Install-RabbitMQ -Config $config
    Install-InfluxDB -Config $config
    
    # Install Python dependencies
    Install-PythonDependencies -Config $config
    
    # Start the application
    Start-Application -Config $config
    
    Write-Host "`n=== SETUP COMPLETED SUCCESSFULLY ===" -ForegroundColor Green
    Write-Host "`nNext steps:" -ForegroundColor Cyan
    Write-Host "1. Configure your .env file with the correct credentials"
    Write-Host "2. Access the application at http://localhost:8000"
    Write-Host "3. Check the logs at logs/trading_bot.log for any issues"
    
} catch {
    Write-Host "`nAn error occurred during setup:" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}

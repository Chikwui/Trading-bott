# Local Deployment Guide

This guide explains how to set up and run the Trading Bot application locally without Docker.

## Prerequisites

- Windows 10/11
- PowerShell 5.1+
- Python 3.8+ (add to PATH during installation)
- Administrator privileges (required for service installation)

## Setup Instructions

1. **Clone the repository** (if not already done):
   ```powershell
   git clone <repository-url>
   cd trading-bot
   ```

2. **Run the setup script**:
   Open PowerShell as Administrator and run:
   ```powershell
   Set-ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
   .\deploy\setup-local.ps1
   ```

   This script will:
   - Install and configure Redis
   - Install and configure RabbitMQ
   - Install and configure InfluxDB
   - Install Python dependencies
   - Start the application

3. **Configure the application**:
   - Copy `.env.example` to `.env`
   - Update the `.env` file with your configuration
   - Add your API keys and other sensitive information

4. **Start the application** (if not started automatically):
   ```powershell
   python app.py
   ```

## Accessing Services

- **Application**: http://localhost:8000
- **RabbitMQ Management**: http://localhost:15672
  - Default credentials: guest/guest
- **InfluxDB**: http://localhost:8086
  - No authentication by default

## Managing Services

### Redis
- Start: `Start-Service -Name "Redis"`
- Stop: `Stop-Service -Name "Redis"`
- Status: `Get-Service -Name "Redis"`

### RabbitMQ
- Start: `Start-Service -Name "RabbitMQ"`
- Stop: `Stop-Service -Name "RabbitMQ"`
- Status: `Get-Service -Name "RabbitMQ"`

### InfluxDB
- Start: Run `influxd.exe` from the installation directory
- Default installation: `C:\influxdb\influxdb-<version>\influxd.exe`

## Troubleshooting

1. **Port Conflicts**:
   - Ensure ports 6379 (Redis), 5672 (RabbitMQ), and 8086 (InfluxDB) are available
   - Check running services: `netstat -ano | findstr "6379\|5672\|8086"`

2. **Service Startup Issues**:
   - Check Windows Event Viewer for service errors
   - Verify service dependencies are met

3. **Python Dependencies**:
   - Ensure Python is in your PATH
   - Try reinstalling requirements: `pip install -r requirements.txt`

## Updating

1. Pull the latest changes:
   ```powershell
   git pull
   ```

2. Reinstall dependencies if needed:
   ```powershell
   pip install -r requirements.txt
   pip install -r requirements-ml.txt
   ```

3. Restart the application and services.

## Security Notes

- Never commit your `.env` file to version control
- Use strong passwords for all services in production
- Configure firewalls to restrict access to services
- Regularly update all dependencies

# Check Prerequisites for Monitoring Stack

function Test-CommandExists {
    param($command)
    $exists = $null -ne (Get-Command $command -ErrorAction SilentlyContinue)
    return $exists
}

function Test-DockerRunning {
    try {
        $null = docker info 2>$null
        return $true
    } catch {
        return $false
    }
}

function Test-KubernetesRunning {
    try {
        $null = kubectl cluster-info 2>$null
        return $true
    } catch {
        return $false
    }
}

Write-Host "`n=== Monitoring Stack Prerequisites Check ===" -ForegroundColor Cyan

# Check Docker
$dockerInstalled = Test-CommandExists "docker"
$dockerRunning = $false

if ($dockerInstalled) {
    $dockerRunning = Test-DockerRunning
    $dockerStatus = if ($dockerRunning) { "Running" } else { "Not Running" }
    Write-Host "Docker: Installed - $dockerStatus" -ForegroundColor $(if ($dockerRunning) { "Green" } else { "Red" })
    
    if (-not $dockerRunning) {
        Write-Host "  - Please start Docker Desktop and ensure the Docker service is running" -ForegroundColor Yellow
    }
} else {
    Write-Host "Docker: Not Installed" -ForegroundColor Red
    Write-Host "  - Please install Docker Desktop from: https://www.docker.com/products/docker-desktop/" -ForegroundColor Yellow
}

# Check kubectl
$kubectlInstalled = Test-CommandExists "kubectl"
$kubernetesRunning = $false

if ($kubectlInstalled) {
    $kubernetesRunning = Test-KubernetesRunning
    $k8sStatus = if ($kubernetesRunning) { "Running" } else { "Not Running" }
    Write-Host "Kubernetes: Installed - $k8sStatus" -ForegroundColor $(if ($kubernetesRunning) { "Green" } else { "Yellow" })
    
    if (-not $kubernetesRunning -and $dockerRunning) {
        Write-Host "  - Kubernetes is not running. Please enable Kubernetes in Docker Desktop:" -ForegroundColor Yellow
        Write-Host "    1. Open Docker Desktop" -ForegroundColor Yellow
        Write-Host "    2. Go to Settings -> Kubernetes" -ForegroundColor Yellow
        Write-Host "    3. Check 'Enable Kubernetes' and click 'Apply & Restart'" -ForegroundColor Yellow
    }
} else {
    Write-Host "kubectl: Not Installed" -ForegroundColor Red
    Write-Host "  - Please install kubectl from: https://kubernetes.io/docs/tasks/tools/install-kubectl-windows/" -ForegroundColor Yellow
}

# Check Helm (optional)
$helmInstalled = Check-Command "helm"
Write-Host "Helm: $(if ($helmInstalled) { 'Installed' } else { 'Not Installed (Optional)' })" -ForegroundColor $(if ($helmInstalled) { 'Green' } else { 'Yellow' })

# Summary
Write-Host "`n=== Summary ===" -ForegroundColor Cyan
if ($dockerRunning -and $kubernetesRunning) {
    Write-Host "All prerequisites are met! You can proceed with the monitoring stack deployment." -ForegroundColor Green
    Write-Host "Run: .\deploy\setup-monitoring.ps1" -ForegroundColor Green
} else {
    Write-Host "Some prerequisites are missing or not running. Please address the issues above before proceeding." -ForegroundColor Red
}

Write-Host "`nFor more information, refer to the documentation in the 'deploy' directory." -ForegroundColor Cyan

# Stop Monitoring Stack Port Forwards

# Check if we have any running port-forwards
if (Test-Path ".monitoring-pids.json") {
    $pids = Get-Content -Path ".monitoring-pids.json" | ConvertFrom-Json
    
    # Stop port-forward processes
    Write-Host "Stopping port-forwarding processes..." -ForegroundColor Yellow
    
    if ($pids.Prometheus) {
        try {
            Stop-Process -Id $pids.Prometheus -Force -ErrorAction SilentlyContinue
            Write-Host "Stopped Prometheus port-forward (PID: $($pids.Prometheus))" -ForegroundColor Green
        } catch {
            Write-Host "Failed to stop Prometheus port-forward: $_" -ForegroundColor Red
        }
    }
    
    if ($pids.Grafana) {
        try {
            Stop-Process -Id $pids.Grafana -Force -ErrorAction SilentlyContinue
            Write-Host "Stopped Grafana port-forward (PID: $($pids.Grafana))" -ForegroundColor Green
        } catch {
            Write-Host "Failed to stop Grafana port-forward: $_" -ForegroundColor Red
        }
    }
    
    # Remove the PID file
    Remove-Item -Path ".monitoring-pids.json" -Force -ErrorAction SilentlyContinue
} else {
    Write-Host "No monitoring port-forwards found to stop." -ForegroundColor Yellow
}

Write-Host "`nTo completely remove the monitoring stack, run:" -ForegroundColor Yellow
Write-Host "kubectl delete -f deploy/monitoring/monitoring-stack.yaml" -ForegroundColor Cyan

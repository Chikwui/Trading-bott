# Setup Monitoring Stack for Trading Bot

# Configuration
$config = @{
    Namespace = "trading"
    MonitoringNamespace = "monitoring"
    Versions = @{
        Prometheus = "v2.44.0"
        Grafana = "10.0.0"
        RedisExporter = "v1.45.0"
        NodeExporter = "v1.5.0"
    }
}

# Set environment variables from config
$env:PROMETHEUS_VERSION = $config.Versions.Prometheus
$env:GRAFANA_VERSION = $config.Versions.Grafana
$env:REDIS_EXPORTER_VERSION = $config.Versions.RedisExporter
$env:NODE_EXPORTER_VERSION = $config.Versions.NodeExporter

# Create namespaces if they don't exist
Write-Host "Creating namespaces..." -ForegroundColor Green
kubectl create namespace $($config.Namespace) --dry-run=client -o yaml | kubectl apply -f -
kubectl create namespace $($config.MonitoringNamespace) --dry-run=client -o yaml | kubectl apply -f -

# Label nodes for monitoring
Write-Host "Labeling nodes for monitoring..." -ForegroundColor Green
kubectl label nodes --all monitoring=true --overwrite

# Create the monitoring stack
Write-Host "Deploying monitoring stack..." -ForegroundColor Green
kubectl apply -f deploy/monitoring/monitoring-stack.yaml -n $($config.MonitoringNamespace)

# Wait for pods to be ready
Write-Host "Waiting for monitoring pods to be ready..." -ForegroundColor Green
kubectl wait --for=condition=ready pod -l app=prometheus -n $($config.MonitoringNamespace) --timeout=300s
kubectl wait --for=condition=ready pod -l app=grafana -n $($config.MonitoringNamespace) --timeout=300s
kubectl wait --for=condition=ready pod -l app=redis-exporter -n $($config.MonitoringNamespace) --timeout=300s

# Port-forward services for local access
Write-Host "Setting up port forwarding..." -ForegroundColor Green
Write-Host "Grafana: http://localhost:3000" -ForegroundColor Cyan
Write-Host "Prometheus: http://localhost:9090" -ForegroundColor Cyan

# Create port-forwarding in background
$prometheusJob = Start-Process -NoNewWindow -PassThru -FilePath "kubectl" -ArgumentList "port-forward -n $($config.MonitoringNamespace) svc/prometheus 9090:9090"
$grafanaJob = Start-Process -NoNewWindow -PassThru -FilePath "kubectl" -ArgumentList "port-forward -n $($config.MonitoringNamespace) svc/grafana 3000:3000"

# Save process IDs to file for later cleanup
@{
    Prometheus = $prometheusJob.Id
    Grafana = $grafanaJob.Id
} | ConvertTo-Json | Set-Content -Path ".monitoring-pids.json"

Write-Host "`nMonitoring stack deployed successfully!" -ForegroundColor Green
Write-Host "To stop the port forwarding, run: .\stop-monitoring.ps1" -ForegroundColor Yellow

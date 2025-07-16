# Kafka topic setup script for Windows PowerShell

# Configuration
$kafkaContainer = "kafka1"
$zkConnect = "zookeeper:2181"

function New-Topic {
    param(
        [string]$name,
        [int]$partitions,
        [int]$replication,
        [int]$retentionHours,
        [string]$cleanupPolicy
    )

    Write-Host "Creating topic: $name"
    
    # Create topic
    $createCmd = "kafka-topics --create --topic $name --partitions $partitions --replication-factor $replication --if-not-exists --zookeeper $zkConnect"
    docker exec $kafkaContainer $createCmd

    # Configure retention
    Write-Host "Configuring retention for $name"
    $retentionMs = $retentionHours * 3600 * 1000
    $retentionCmd = "kafka-configs --alter --entity-type topics --entity-name $name --add-config `"retention.ms=$retentionMs`" --zookeeper $zkConnect"
    docker exec $kafkaContainer $retentionCmd

    # Configure cleanup policy
    $cleanupCmd = "kafka-configs --alter --entity-type topics --entity-name $name --add-config `"cleanup.policy=$cleanupPolicy`" --zookeeper $zkConnect"
    docker exec $kafkaContainer $cleanupCmd
}

Write-Host "Setting up trading topics..."

# Trading topics
New-Topic "trading-signals" 3 1 24 "delete"
New-Topic "trade-execution" 3 1 168 "delete"  # 7 days
New-Topic "market-data" 10 1 168 "delete"    # 7 days, more partitions for high volume
New-Topic "news-feed" 3 1 24 "delete"
New-Topic "performance-metrics" 3 1 168 "delete"  # 7 days
New-Topic "error-logs" 3 1 168 "delete"      # 7 days

# Verify topics
Write-Host "`nVerifying topics..."
$listCmd = "kafka-topics --list --zookeeper $zkConnect"
docker exec $kafkaContainer $listCmd

Write-Host "`nKafka topic setup complete!"

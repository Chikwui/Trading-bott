#!/bin/bash

# Kafka setup script for AI Trader
# Creates all necessary topics with appropriate configurations

# Configuration
NETWORK="kafka-network"
KAFKA_CONTAINER="kafka1"
ZK_CONNECT="zookeeper:2181"

# Helper function to create topic with configuration
create_topic() {
    local name=$1
    local partitions=$2
    local replication=$3
    local retention_hours=$4
    local cleanup_policy=$5

    echo "Creating topic: $name"
    docker exec $KAFKA_CONTAINER \
        kafka-topics \
        --create \
        --topic $name \
        --partitions $partitions \
        --replication-factor $replication \
        --if-not-exists \
        --zookeeper $ZK_CONNECT

    # Configure retention settings
    echo "Configuring retention for $name"
    docker exec $KAFKA_CONTAINER \
        kafka-configs \
        --alter \
        --entity-type topics \
        --entity-name $name \
        --add-config retention.ms=$((retention_hours * 3600 * 1000)) \
        --zookeeper $ZK_CONNECT

    # Configure cleanup policy
    docker exec $KAFKA_CONTAINER \
        kafka-configs \
        --alter \
        --entity-type topics \
        --entity-name $name \
        --add-config cleanup.policy=$cleanup_policy \
        --zookeeper $ZK_CONNECT
}

# Trading topics
echo "Setting up trading topics..."
create_topic "trading-signals" 3 1 24 "delete"
create_topic "trade-execution" 3 1 168 "delete"  # 7 days
create_topic "market-data" 10 1 168 "delete"    # 7 days, more partitions for high volume
create_topic "news-feed" 3 1 24 "delete"
create_topic "performance-metrics" 3 1 168 "delete"  # 7 days
create_topic "error-logs" 3 1 168 "delete"      # 7 days

# Verify topics
echo "\nVerifying topics..."
docker exec $KAFKA_CONTAINER kafka-topics --list --zookeeper $ZK_CONNECT

echo "\nKafka topic setup complete!"

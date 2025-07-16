from typing import Dict, List
from enum import Enum

class TopicConfig:
    def __init__(self, name: str, partitions: int, replication: int, retention_hours: int, cleanup_policy: str):
        self.name = name
        self.partitions = partitions
        self.replication = replication
        self.retention_hours = retention_hours
        self.cleanup_policy = cleanup_policy

# Topic configurations for different trading data types
class Topic(Enum):
    TRADE_SIGNALS = TopicConfig(
        name="trading-signals",
        partitions=3,
        replication=1,
        retention_hours=24,
        cleanup_policy="delete"
    )
    
    TRADE_EXECUTION = TopicConfig(
        name="trade-execution",
        partitions=3,
        replication=1,
        retention_hours=168,  # 7 days
        cleanup_policy="delete"
    )
    
    MARKET_DATA = TopicConfig(
        name="market-data",
        partitions=10,  # More partitions for high volume market data
        replication=1,
        retention_hours=168,  # 7 days
        cleanup_policy="delete"
    )
    
    NEWS_FEED = TopicConfig(
        name="news-feed",
        partitions=3,
        replication=1,
        retention_hours=24,
        cleanup_policy="delete"
    )
    
    PERFORMANCE_METRICS = TopicConfig(
        name="performance-metrics",
        partitions=3,
        replication=1,
        retention_hours=168,  # 7 days
        cleanup_policy="delete"
    )
    
    ERROR_LOGS = TopicConfig(
        name="error-logs",
        partitions=3,
        replication=1,
        retention_hours=168,  # 7 days
        cleanup_policy="delete"
    )

# Get all topics as a list
def get_all_topics() -> List[TopicConfig]:
    return [topic.value for topic in Topic]

# Get topic configuration by name
def get_topic_config(name: str) -> TopicConfig:
    for topic in Topic:
        if topic.value.name == name:
            return topic.value
    raise ValueError(f"Topic {name} not found")

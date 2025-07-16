import os
from topics_config import Topic, get_all_topics
import docker
from kafka import KafkaAdminClient, NewTopic
from kafka.errors import KafkaError
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load environment variables
load_dotenv()

# Docker configuration
DOCKER_NETWORK = "kafka-network"
KAFKA_CONTAINER = "kafka1"


def create_topic(topic_config: Topic):
    """Create a Kafka topic with specified configuration"""
    try:
        # Create topic using Docker exec
        cmd = [
            "kafka-topics",
            "--create",
            f"--topic={topic_config.name}",
            f"--partitions={topic_config.partitions}",
            f"--replication-factor={topic_config.replication}",
            "--if-not-exists",
            "--zookeeper",
            "zookeeper:2181"
        ]
        
        client = docker.from_env()
        container = client.containers.get(KAFKA_CONTAINER)
        result = container.exec_run(cmd)
        
        if result.exit_code == 0:
            logging.info(f"Successfully created topic: {topic_config.name}")
        else:
            logging.error(f"Failed to create topic {topic_config.name}: {result.output.decode()}")
            
        # Configure retention settings using Kafka Admin client
        admin_client = KafkaAdminClient(
            bootstrap_servers=["localhost:9092"],
            client_id="topic-configurator"
        )
        
        config = {
            "retention.ms": str(topic_config.retention_hours * 3600 * 1000),
            "cleanup.policy": topic_config.cleanup_policy
        }
        
        admin_client.alter_configs([
            ("topics", topic_config.name, config)
        ])
        
        logging.info(f"Configured retention policy for {topic_config.name}")
        
    except KafkaError as e:
        logging.error(f"Kafka error creating topic {topic_config.name}: {str(e)}")
    except Exception as e:
        logging.error(f"Error creating topic {topic_config.name}: {str(e)}")


def main():
    logging.info("Starting Kafka topic setup...")
    
    # Get all topic configurations
    topics = get_all_topics()
    
    # Create each topic
    for topic in topics:
        create_topic(topic)
    
    logging.info("Kafka topic setup completed")


if __name__ == '__main__':
    main()

import json
import queue
import logging
import threading
from typing import Any, Callable, Dict, List, Optional

try:
    # try standard kafka client libraries
    from kafka import KafkaProducer, KafkaConsumer
except ImportError:
    KafkaProducer = None
    KafkaConsumer = None

logger = logging.getLogger(__name__)

class KafkaMessageBus:
    """
    Message bus utilizing Apache Kafka for streaming data ingestion and cross-agent communication.
    Gracefully falls back to thread-safe in-memory queues for testing.
    """
    def __init__(self, bootstrap_servers: Optional[List[str]] = None):
        import os
        if bootstrap_servers:
            self.bootstrap_servers = bootstrap_servers
        else:
            env_servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
            self.bootstrap_servers = [s.strip() for s in env_servers.split(",") if s.strip()]
        self.producer = None
        self.is_kafka = False
        
        # Fallback queues per topic
        self._local_topics: Dict[str, queue.Queue] = {}
        self._lock = threading.Lock()
        self._consumers: List[threading.Thread] = []
        self._running = True

        self._init_kafka()

    def _init_kafka(self):
        if KafkaProducer is not None:
            try:
                self.producer = KafkaProducer(
                    bootstrap_servers=self.bootstrap_servers,
                    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                    request_timeout_ms=2000
                )
                self.is_kafka = True
                logger.info("Connected to Kafka cluster.")
                return
            except Exception as e:
                logger.warning(f"Failed to connect to Kafka brokers: {e}. Falling back to in-memory queues.")
        else:
            logger.info("kafka-python package not installed. Using in-memory queue bus.")
            
        self.is_kafka = False
        self.producer = None

    def publish(self, topic: str, value: Dict[str, Any]):
        """
        Publishes a message to a topic.
        """
        if self.is_kafka and self.producer:
            try:
                self.producer.send(topic, value=value)
                self.producer.flush()
                return
            except Exception as e:
                logger.error(f"Kafka send error on topic {topic}: {e}. Routing to in-memory queue.")
        
        # Fallback to local queue
        self._get_local_queue(topic).put(value)

    def _get_local_queue(self, topic: str) -> queue.Queue:
        with self._lock:
            if topic not in self._local_topics:
                self._local_topics[topic] = queue.Queue()
            return self._local_topics[topic]

    def subscribe(self, topic: str, callback: Callable[[Dict[str, Any]], None], group_id: str = "default-group"):
        """
        Subscribes to a topic. Spawns a background worker thread to execute callback on message reception.
        """
        def listen_kafka():
            try:
                consumer = KafkaConsumer(
                    topic,
                    bootstrap_servers=self.bootstrap_servers,
                    group_id=group_id,
                    value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                    auto_offset_reset='latest'
                )
                for message in consumer:
                    if not self._running:
                        break
                    callback(message.value)
            except Exception as e:
                logger.error(f"Kafka consumer error on topic {topic}: {e}.")

        def listen_local():
            q = self._get_local_queue(topic)
            while self._running:
                try:
                    # Timeout to check self._running periodically
                    msg = q.get(timeout=0.5)
                    callback(msg)
                    q.task_done()
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Local queue consumption error on topic {topic}: {e}")

        target_thread = listen_kafka if self.is_kafka else listen_local
        t = threading.Thread(target=target_thread, daemon=True)
        t.start()
        self._consumers.append(t)

    def shutdown(self):
        self._running = False
        for t in self._consumers:
            t.join(timeout=1.0)
        if self.producer:
            try:
                self.producer.close()
            except Exception:
                pass
        logger.info("Message bus shutdown completed.")

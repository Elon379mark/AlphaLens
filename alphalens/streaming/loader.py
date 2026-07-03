import logging
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from alphalens.streaming.kafka_bus import KafkaMessageBus
from database.session import DatabaseSessionManager, db_manager
from database.repositories.market_data_repository import MarketDataRepository

logger = logging.getLogger(__name__)

class RealtimeMarketDataLoader:
    """
    Consumes tick/bar data from Kafka topics and populates the database.
    """
    def __init__(
        self,
        bus: Optional[KafkaMessageBus] = None,
        session_manager: Optional[DatabaseSessionManager] = None,
        topic: str = "market_ticks"
    ):
        self.bus = bus or KafkaMessageBus()
        self.db = session_manager or db_manager
        self.topic = topic

    def start(self):
        """Starts subscribing to Kafka ingestion feeds."""
        logger.info(f"Starting RealtimeMarketDataLoader on topic: {self.topic}")
        self.bus.subscribe(self.topic, self.handle_tick)

    def handle_tick(self, msg: Dict[str, Any]):
        """Callback executed for each received market tick message."""
        try:
            symbol = msg.get("symbol")
            ts_raw = msg.get("timestamp")
            if not symbol or not ts_raw:
                logger.warning(f"Loader: Skipping invalid tick message: {msg}")
                return

            # Convert timestamp
            if isinstance(ts_raw, (int, float)):
                ts = datetime.fromtimestamp(ts_raw, tz=timezone.utc)
            else:
                ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))

            # Extract OHLCV
            open_val = float(msg.get("open", 0.0))
            high_val = float(msg.get("high", 0.0))
            low_val = float(msg.get("low", 0.0))
            close_val = float(msg.get("close", 0.0))
            volume_val = float(msg.get("volume", 0.0))

            # Database insertion helper
            async def insert_record():
                async with self.db.async_session() as session:
                    repo = MarketDataRepository(session)
                    await repo.create(
                        timestamp=ts,
                        symbol=symbol,
                        open=open_val,
                        high=high_val,
                        low=low_val,
                        close=close_val,
                        volume=volume_val
                    )

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                # Schedule task thread-safely
                asyncio.run_coroutine_threadsafe(insert_record(), loop)
            else:
                asyncio.run(insert_record())

            logger.debug(f"Loader: Ingested tick for {symbol} at {ts}")

        except Exception as e:
            logger.error(f"Loader: Error ingesting market data tick: {e}", exc_info=True)

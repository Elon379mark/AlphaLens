import os
import sys
import unittest
import tempfile
import asyncio
from datetime import datetime

# Adjust path to find alphalens
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from database.session import DatabaseSessionManager
from alphalens.storage.cache import CacheManager
from alphalens.streaming.kafka_bus import KafkaMessageBus
from alphalens.streaming.loader import RealtimeMarketDataLoader
from database.repositories.market_data_repository import MarketDataRepository

class TestStreamingAndCache(unittest.TestCase):
    def setUp(self):
        # Create temp file for SQLite DB fallback
        self.db_fd, self.db_path = tempfile.mkstemp()
        self.db_manager = DatabaseSessionManager(sqlite_path=self.db_path)
        self.db_manager.create_tables()

        self.cache = CacheManager()
        self.bus = KafkaMessageBus()

    def tearDown(self):
        asyncio.run(self.db_manager.close_all())

        try:
            os.close(self.db_fd)
            os.unlink(self.db_path)
        except OSError:
            pass
        self.bus.shutdown()

    def test_cache_manager_ops(self):
        matrix = {"factor_X": [1.1, -2.2], "factor_Y": [0.0, 3.3]}
        self.cache.set_matrix("test_key", matrix)
        retrieved = self.cache.get_matrix("test_key")
        self.assertEqual(retrieved, matrix)

    def test_message_bus_pub_sub(self):
        received = []
        event = asyncio.Event()

        def callback(msg):
            received.append(msg)
            event.set()

        self.bus.subscribe("test_topic", callback)
        self.bus.publish("test_topic", {"hello": "world"})

        async def wait_for_event():
            try:
                await asyncio.wait_for(event.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                pass

        asyncio.run(wait_for_event())

        self.assertEqual(len(received), 1)
        self.assertEqual(received[0]["hello"], "world")

    def test_realtime_loader(self):
        loader = RealtimeMarketDataLoader(bus=self.bus, session_manager=self.db_manager, topic="market_ticks")
        loader.start()

        tick = {
            "symbol": "AAPL",
            "timestamp": "2026-06-19T10:00:00Z",
            "open": 150.0,
            "high": 152.5,
            "low": 149.5,
            "close": 151.0,
            "volume": 1000000.0
        }

        self.bus.publish("market_ticks", tick)

        async def verify_inserted():
            await asyncio.sleep(0.5)
            async with self.db_manager.async_session() as session:
                repo = MarketDataRepository(session)
                res = await repo.get_market_data("AAPL", "2026-06-19T10:05:00Z")
                return res
        
        records = asyncio.run(verify_inserted())
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].symbol, "AAPL")
        self.assertEqual(records[0].close, 151.0)

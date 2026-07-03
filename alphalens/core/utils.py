"""Shared utility functions for AlphaLens."""
import asyncio
import concurrent.futures
import logging

logger = logging.getLogger(__name__)

# Executor to run coroutines synchronously in a background thread
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

def run_sync(coro):
    """Run an async coroutine from synchronous code safely.
    
    Creates a new event loop in a thread pool worker,
    runs the coroutine, then properly closes the loop to prevent resource leaks.
    """
    def _run():
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
        finally:
            loop.close()
    
    future = _executor.submit(_run)
    return future.result(timeout=30)

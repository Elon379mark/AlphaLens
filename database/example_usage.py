import os
import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Any

from database.session import DatabaseSessionManager
from database.service import DatabaseAgentService, PostgresCheckpointSaver, AlphaLensState
from database.repositories.market_data_repository import MarketDataRepository
from database.repositories.hypothesis_repository import HypothesisRepository
from database.repositories.signal_repository import SignalRepository

try:
    from langgraph.graph import StateGraph, END
except ImportError:
    StateGraph = None
    END = "__end__"

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("AlphaLensDemo")

async def run_demo():
    logger.info("Starting AlphaLens Database Layer Demonstration...")
    
    # 1. Initialize session manager pointing to a demo local SQLite file
    from pathlib import Path
    db_file = str(Path(__file__).resolve().parent / "alphalens_demo.db")
    if os.path.exists(db_file):
        try:
            os.remove(db_file)
        except Exception:
            pass
            
    demo_db = DatabaseSessionManager(sqlite_path=db_file)
    
    # 2. Create tables
    demo_db.create_tables()
    
    # 3. Instantiate Service Layer
    agent_service = DatabaseAgentService(session_manager=demo_db)
    
    # 4. Save Hypothesis
    hypothesis_data = {
        "hypothesis_id": "H-VIX-RECOVERY",
        "predictor_variable": "vix_index",
        "target_asset_class": "US_Equities",
        "predicted_direction": "negative",
        "confidence": 0.82,
        "theoretical_mechanism": "High market fear indicates risk-off selling pressure.",
        "source_references": ["arXiv:1203.4567", "JOF:2018-vix"]
    }
    saved_h = await agent_service.save_hypothesis(hypothesis_data)
    logger.info(f"Saved Hypothesis: {saved_h}")
    
    # 5. Save Signal
    signal_data = {
        "signal_id": "SIG-VIX-001",
        "hypothesis_id": "H-VIX-RECOVERY",
        "timestamp": datetime.now(timezone.utc),
        "symbol": "SPY",
        "value": -1.5,
        "information_coefficient": 0.052,
        "information_ratio": 1.24,
        "half_life_days": 5.0,
        "status": "ACTIVE"
    }
    saved_sig = await agent_service.save_signal(signal_data)
    logger.info(f"Saved Signal: {saved_sig}")
    
    # 6. Save Causal Result
    causal_data = {
        "result_id": "CR-VIX-001",
        "hypothesis_id": "H-VIX-RECOVERY",
        "p_value": 0.012,
        "ate_magnitude": -0.0035,
        "confidence_interval_lower": -0.0058,
        "confidence_interval_upper": -0.0012,
        "confounders": {"market_trend": 0.001, "interest_rate": -0.002}
    }
    saved_cr = await agent_service.save_causal_result(causal_data)
    logger.info(f"Saved Causal Result: {saved_cr}")
    
    # 7. Save Portfolio Allocation and Returns
    allocations = [
        {"symbol": "SPY", "weight": -0.40},
        {"symbol": "SHY", "weight": 0.60}
    ]
    portfolio_id = "PORT-HEDGE-1"
    timestamp = datetime.now(timezone.utc)
    await agent_service.save_portfolio(
        portfolio_id=portfolio_id,
        timestamp=timestamp,
        allocations=allocations,
        returns_val=-0.0012,
        equity=1024500.00
    )
    logger.info("Saved Portfolio allocations and returns.")
    
    # 8. Load & Save Agent Memory
    await agent_service.save_agent_memory("backtest_agent", "last_run_params", {"lookback_days": 252, "rebalance_freq": "daily"})
    loaded_memory = await agent_service.load_agent_memory("backtest_agent", "last_run_params")
    logger.info(f"Loaded Agent Memory: {loaded_memory}")
    
    # 9. Save Workflow State
    await agent_service.save_workflow_state(
        workflow_id="WF-RUN-999",
        state_data={"current_stage": "validation", "iteration": 2},
        status="RUNNING",
        current_node="causal_validation_agent"
    )
    logger.info("Saved workflow state.")

    # 10. Temporal Data Access Layer Demonstration (Preventing Look-Ahead Bias)
    logger.info("Testing Temporal Data Access Layer...")
    base_time = datetime(2025, 1, 1, 12, 0, 0)
    
    async with demo_db.async_session() as session:
        market_repo = MarketDataRepository(session)
        
        # Populate market data bars
        # 12:00:00 (base)
        await market_repo.create(timestamp=base_time, symbol="AAPL", open=150.0, high=151.0, low=149.0, close=150.5, volume=1000.0)
        # 12:01:00 (future)
        await market_repo.create(timestamp=base_time + timedelta(minutes=1), symbol="AAPL", open=150.5, high=152.0, low=150.0, close=151.8, volume=1500.0)
        # 12:02:00 (further future)
        await market_repo.create(timestamp=base_time + timedelta(minutes=2), symbol="AAPL", open=151.8, high=153.0, low=151.5, close=152.5, volume=1200.0)
        
        logger.info("Inserted AAPL bars for 12:00:00, 12:01:00, and 12:02:00.")
 
        # QUERY: Query as of 12:01:00
        query_time = base_time + timedelta(minutes=1) # 12:01:00
        logger.info(f"Querying AAPL market data as of target timestamp: {query_time}")
        
        results = await market_repo.get_market_data(symbol="AAPL", timestamp=query_time)
        logger.info(f"Returned {len(results)} bars:")
        for r in results:
            logger.info(f"  [{r.timestamp}] Close: {r.close}")
            
        # ASSERT: Check that no 12:02:00 bar is returned
        timestamps = [r.timestamp for r in results]
        assert base_time + timedelta(minutes=2) not in timestamps, "Error: Look-ahead bias detected! Future bar returned."
        logger.info("SUCCESS: Look-ahead bias prevented! Future data was not returned.")
 
    # 11. LangGraph State Checkpoint Saver Demonstration
    logger.info("Demonstrating LangGraph integration...")
    checkpointer = PostgresCheckpointSaver(session_manager=demo_db)
    
    if StateGraph is not None:
        # Define a simple State Graph
        workflow = StateGraph(dict)
        
        def mock_node(state):
            logger.info("Executing mock LangGraph node.")
            return {"sharpe_ratio": 1.45}
            
        workflow.add_node("agent", mock_node)
        workflow.set_entry_point("agent")
        workflow.set_finish_point("agent")
        
        # Compile graph with our checkpointer
        app = workflow.compile(checkpointer=checkpointer)
        
        # Run graph
        config = {"configurable": {"thread_id": "thread-demo-123"}}
        logger.info("Invoking LangGraph with custom PostgresCheckpointer...")
        app.invoke({"sharpe_ratio": 0.0}, config)
        
        # Verify checkpoint state retrieval
        state = app.get_state(config)
        logger.info(f"Retrieved state from Postgres checkpointer: {state.values}")
    else:
        logger.info("LangGraph package not available, executing direct checkpointer methods...")
        # Direct checkpointer validation
        config = {"configurable": {"thread_id": "thread-direct-123", "checkpoint_ns": ""}}
        checkpoint_obj = {
            "v": 1,
            "id": "chk-1",
            "ts": datetime.now(timezone.utc).isoformat(),
            "channel_values": {"sharpe_ratio": 1.82},
            "channel_versions": {"sharpe_ratio": "v1"},
            "versions_seen": {}
        }
        metadata_obj = {"source": "test"}
        
        # Put
        logger.info("Persisting checkpoint directly via checkpointer...")
        checkpointer.put(config, checkpoint_obj, metadata_obj, {"sharpe_ratio": "v1"})
        
        # Get
        retrieved = checkpointer.get_tuple(config)
        logger.info(f"Retrieved checkpoint data: {retrieved.checkpoint}")
        assert retrieved.checkpoint["channel_values"]["sharpe_ratio"] == 1.82
        logger.info("Direct checkpointer test passed!")
 
    logger.info("All demonstrations completed successfully!")
    await demo_db.close_all()
    
    # Cleanup demo db file
    try:
        os.remove(db_file)
    except Exception:
        pass
 
if __name__ == "__main__":
    asyncio.run(run_demo())

-- setup.sql
-- Run this migration against your PostgreSQL database to initialize hypertables and optimize indexes.

-- 1. Enable TimescaleDB Extension if not already active
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- 2. Convert time-series tables into TimescaleDB hypertables
-- For sub-daily tick data or regular pricing bars (partitioned in 7-day or 30-day chunks)
SELECT create_hypertable('market_data', 'timestamp', chunk_time_interval => INTERVAL '30 days', if_not_exists => TRUE);

-- For generated signals (partitioned in 30-day chunks)
SELECT create_hypertable('signals', 'timestamp', chunk_time_interval => INTERVAL '30 days', if_not_exists => TRUE);

-- For portfolio returns over time (partitioned in 30-day chunks)
SELECT create_hypertable('portfolio_returns', 'timestamp', chunk_time_interval => INTERVAL '30 days', if_not_exists => TRUE);

-- 3. Additional performance-critical indexes
CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timestamp ON market_data (symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_signals_hypothesis_id ON signals (hypothesis_id);
CREATE INDEX IF NOT EXISTS idx_signals_symbol_timestamp ON signals (symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_portfolio_allocations_port_time ON portfolio_allocations (portfolio_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_agent_logs_workflow_id ON agent_logs (workflow_id);
CREATE INDEX IF NOT EXISTS idx_agent_logs_agent_name ON agent_logs (agent_name);
CREATE INDEX IF NOT EXISTS idx_workflow_states_status ON workflow_states (status);
CREATE INDEX IF NOT EXISTS idx_agent_memory_agent_key ON agent_memory (agent_name, memory_key);

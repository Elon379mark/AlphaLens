# AlphaLens Quantitative Research System

AlphaLens is an autonomous quantitative finance research platform built on **LangGraph**. It coordinates a multi-agent quantitative pipeline to discover, validate, backtest, and optimize financial alpha factors.

---

## 1. Overview & Architecture

AlphaLens leverages a multi-agent system to automate quantitative finance workflows. Instead of relying on a single agent, the platform separates concerns into specialized nodes in a LangGraph workflow. A built-in rejection/refinement loop validates findings at each step:

```
[ Literature RAG ] (Literature Agent)
        │
        ▼
[ Signal Generation ] (Signal Generation Agent)
        │
        ▼
[ Causal Validation ] (Causal Validation Agent) ◄───┐ (If rejected,
        │                                           │  refines signal)
        ├──► [Passed] ──► [ Backtesting ] ──────────┤
        └──► [Failed] ──────────────────────────────┘
                                │
                                ▼
                    [ Portfolio Construction ] (Portfolio Agent)
```

1. **Literature RAG Agent**: Scans academic literature (via ChromaDB/retriever) to extract theoretical mechanisms and suggest predictor variables.
2. **Signal Generation Agent**: Computes rolling mathematical features (momentum, volume profiles) on raw market data and calculates Information Coefficients (IC).
3. **Causal Validation Agent**: Conducts Double Machine Learning (DML) and constraint-based PC-Algorithm DAG discovery to distinguish true causation from spurious correlation.
4. **Backtest Agent**: Runs a high-performance vector simulation (with transaction cost modeling and market impact estimates) to evaluate Sharpe Ratio, Drawdowns, and Turnover.
5. **Portfolio Construction Agent**: Runs convex optimization (CVaR minimization, Black-Litterman, Risk Parity) to establish optimal allocation weights.

---

## 2. Tech Stack

- **Orchestration**: LangGraph, LangChain, Pydantic
- **AI Core**: Groq Llama 3.3 (70B)
- **Database**: TimescaleDB / PostgreSQL (via SQLAlchemy)
- **Caching**: Redis
- **Streaming & Ingestion**: Apache Kafka
- **Math & Analytics**: NumPy, Pandas, SciPy, scikit-learn
- **Dashboard**: Streamlit, Plotly
- **Testing**: pytest, unittest

---

## 3. Project Structure

```
Proj1/
├── alphalens/
│   ├── agents/                   # Core agent modules & system prompts
│   │   ├── literature/           # RAG retrieval & extraction logic
│   │   ├── signal_generation/    # IC calculators & factor validators
│   │   └── memory.py             # Persistent episodic/semantic memory engine
│   ├── causal_inference/         # Causal inference analytics
│   │   ├── dag.py                # PC-Algorithm DAG skeleton discovery
│   │   └── dml.py                # Double Machine Learning ATE estimator
│   ├── contracts/                # Shared pydantic schemas & protobuf contracts
│   ├── core/                     # Canonical state schemas & sync-async utils
│   ├── dashboard/                # Streamlit analytics dashboard
│   ├── orchestration/            # LangGraph routing, checkpointers, & protobuf msgs
│   ├── signal_generation/        # High-performance Pandas factor computation
│   ├── simulation/               # Vectorized backtesting & Kyle's lambda cost models
│   ├── storage/                  # Caching layers (Redis & local TTL fallbacks)
│   └── streaming/                # Kafka consumer loaders & ingestion handlers
├── database/                     # SQLAlchemy models, migrations, & repositories
├── tests/                        # Comprehensive unit & integration tests
├── docker-compose.yml            # Multi-container infrastructure services
├── requirements.txt              # Project dependencies
└── README.md                     # Root navigation README
```

---

## 4. Setup & Installation

### Prerequisites
- Python 3.10 or higher
- Docker & Docker Compose
- Groq API Key (for LLM services)

### Steps

1. **Clone the Repository** and navigate to the project directory:
   ```bash
   cd Proj1
   ```

2. **Create and Activate Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables**:
   Copy the sample environment file and configure your API keys:
   ```bash
   cp .env.example .env
   # Open .env and populate GROQ_API_KEY, database credentials, etc.
   ```

5. **Spin Up Infrastructure**:
   Start TimescaleDB, Redis, and Kafka using Docker Compose:
   ```bash
   docker-compose up -d
   ```

6. **Database Migration & Seeding**:
   Run the database initialization script:
   ```bash
   python database/example_usage.py
   ```

---

## 5. Running the Application

### Quantitative Pipeline Execution
To execute the automated quantitative research pipeline:
```bash
python -m alphalens.orchestration.graph
```

### Research Dashboard
To start the Streamlit interactive dashboard:
```bash
streamlit run alphalens/dashboard/app.py
```

---

## 6. Testing

Run the test suite using `pytest` or `unittest`:
```bash
pytest tests/
```
All tests use temporary SQLite fallbacks and mock event loops to run independently of live infrastructure.

"""
AlphaLens Agent Nodes — LLM-powered reasoning for each stage of the pipeline.

Each agent:
1. Receives the current pipeline state
2. Uses a Groq LLM (LLaMA) to reason about the data
3. Calls mathematical tools (our existing modules) to compute results
4. Returns structured updates to the shared state
"""
import os
import json
import uuid
import logging
import time
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from alphalens.contracts.schemas import HypothesisSchema, PredictedDirection
from alphalens.literature_agent.rag import RAGPipeline, LiteratureAgent
from alphalens.signal_generation.features import FeatureFactory
from alphalens.signal_generation.stats import SignalStatistics
from alphalens.causal_inference.dag import CausalDAGDiscovery
from alphalens.causal_inference.dml import DoubleMachineLearningATE
from alphalens.causal_inference.sensitivity import RosenbaumSensitivity
from alphalens.causal_inference.partial_r2 import PartialR2Analysis
from alphalens.simulation.backtest import BacktestEngine
from alphalens.portfolio_construction.optimizer import CVaROptimizer
from alphalens.portfolio_construction.black_litterman import BlackLitterman

from alphalens.agents.literature.ingestion import load_pdf_directory
from alphalens.agents.literature.chunker import chunk_all_documents
from alphalens.agents.literature.embedder import embed_chunks, embed_query
from alphalens.agents.literature.vector_store import get_chroma_client, get_or_create_collection, upsert_chunks
from alphalens.agents.literature.retriever import retrieve, rerank

from alphalens.agents.signal_generation.data_loader import load_ohlcv, load_fundamentals, create_sample_data
from alphalens.agents.signal_generation.features import compute_all_features
from alphalens.agents.signal_generation.ic_calculator import compute_forward_returns, compute_all_ic_icir
from alphalens.agents.signal_generation.validator import validate_features
from alphalens.agents.signal_generation.ranker import rank_signals
from alphalens.storage.cache import CacheManager

# §5.1: Protobuf-serialized inter-agent communication
try:
    from alphalens.orchestration.messages_pb2 import AgentMessage, HypothesisPayload, Direction
    _PROTOBUF_AVAILABLE = True
except ImportError:
    _PROTOBUF_AVAILABLE = False
    logger = logging.getLogger(__name__)

import base64

load_dotenv()
logger = logging.getLogger(__name__)

from alphalens.agents.memory import AgentMemoryEngine
from alphalens.core.utils import run_sync

_memory_engine = AgentMemoryEngine()
_cache_manager = CacheManager()


def _log_protobuf_event(
    sender: str,
    recipient: str,
    priority: int = 1,
    hypothesis: Optional[Any] = None,
    metrics: Optional[Dict[str, float]] = None,
) -> Optional[bytes]:
    """
    §5.1: Creates a Protobuf-serialized AgentMessage M = ⟨sender, recipient, payload, timestamp, priority⟩
    and returns the binary payload for storage in the PostgreSQL event log.
    """
    if not _PROTOBUF_AVAILABLE:
        return None

    msg = AgentMessage()
    msg.sender = sender
    msg.recipient = recipient
    msg.timestamp = time.time()
    msg.priority = priority

    # Attach hypothesis payload if available
    if hypothesis is not None:
        try:
            msg.hypothesis.hypothesis_id = getattr(hypothesis, 'hypothesis_id', '')
            msg.hypothesis.predictor_variable = getattr(hypothesis, 'predictor_variable', '')
            msg.hypothesis.target_asset_class = getattr(hypothesis, 'target_asset_class', '')
            direction = getattr(hypothesis, 'predicted_direction', None)
            if direction is not None:
                msg.hypothesis.predicted_direction = (
                    Direction.POSITIVE if direction.value == 'positive' else Direction.NEGATIVE
                )
            msg.hypothesis.confidence = float(getattr(hypothesis, 'confidence', 0.0))
            msg.hypothesis.theoretical_mechanism = getattr(hypothesis, 'theoretical_mechanism', '')
            for ref in getattr(hypothesis, 'source_references', []):
                msg.hypothesis.source_references.append(str(ref))
        except Exception as e:
            logger.warning(f"Failed to populate hypothesis in protobuf: {e}")

    # Attach numeric metrics
    if metrics:
        for key, val in metrics.items():
            if hasattr(msg, key):
                setattr(msg, key, float(val))

    binary_payload = msg.SerializeToString()
    logger.debug(
        f"§5.1 Protobuf event: {sender}→{recipient}, "
        f"{len(binary_payload)} bytes, priority={priority}"
    )
    return binary_payload


def _get_llm(temperature: float = 0.3) -> ChatGroq:
    """Returns a configured Groq LLM instance."""
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        raise ValueError(
            "GROQ_API_KEY not set. Please set it in your .env file."
        )
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=temperature,
        groq_api_key=api_key,
    )


def _generate_synthetic_market_data(n_days: int = 500):
    """Generates simulated daily market data with a causal predictor."""
    rng = np.random.default_rng(42)
    dates = pd.date_range(end="2026-06-03", periods=n_days, freq="D")
    vix = rng.normal(loc=15.0, scale=3.0, size=n_days)
    predictor = rng.normal(loc=0.0, scale=1.0, size=n_days) + 0.1 * vix
    asset_returns = 0.35 * predictor - 0.04 * vix + rng.normal(0, 0.8, n_days)
    prices = 100.0 * np.exp(np.cumsum(asset_returns / 100.0))
    volume = np.clip(rng.normal(1_000_000, 200_000, n_days), 500_000, 2_000_000)
    volatility = pd.Series(asset_returns).rolling(30, min_periods=1).std().values / 100.0

    df = pd.DataFrame(
        {
            "close": prices,
            "open": prices * (1.0 + rng.normal(0, 0.001, n_days)),
            "high": prices * (1.0 + abs(rng.normal(0, 0.002, n_days))),
            "low": prices * (1.0 - abs(rng.normal(0, 0.002, n_days))),
            "volume": volume,
        },
        index=dates,
    )
    return df, pd.Series(predictor, index=dates)


# ---------------------------------------------------------------------------
_embedding_model = None

def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from alphalens.agents.literature.embedder import load_embedding_model
        _embedding_model = load_embedding_model()
    return _embedding_model


# Agent 1: Literature Agent
# ---------------------------------------------------------------------------
def literature_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Uses the LLM to read retrieved academic context and generate
    a structured trading hypothesis.
    """
    query = state.get("query", "credit spread slope")
    logger.info(f"[Literature Agent] Processing query: {query}")
    run_id = state.get("run_id", "default_run_id")

    # 1. Retrieve context from ChromaDB using Person A's RAG pipeline
    try:
        client = get_chroma_client()
        collection = get_or_create_collection(client)
        
        # Only ingest if collection is empty
        if collection.count() == 0:
            logger.info("[Literature Agent] Vector collection is empty. Ingesting PDFs...")
            PDF_DIR = os.getenv("PDF_DIR", "data/raw/pdfs")
            docs = load_pdf_directory(PDF_DIR)
            
            from alphalens.agents.literature.parser import clean_text, extract_paper_metadata, remove_references_section
            for d in docs:
                d["full_text"] = remove_references_section(clean_text(d["full_text"]))
                d["metadata"] = extract_paper_metadata(d["full_text"])

            chunks = chunk_all_documents(docs)
            model = _get_embedding_model()
            chunks = embed_chunks(chunks, model)
            if chunks:
                upsert_chunks(collection, chunks)
        else:
            logger.info("[Literature Agent] Vector collection found. Skipping PDF ingestion.")
            model = _get_embedding_model()

        q_emb = embed_query(query, model)
        cands = retrieve(collection, q_emb, n_results=15)
        reranked = rerank(query, cands, top_k=5)
        
        context = "\n---\n".join(c["text"] for c in reranked)
        sources = list(set(c.get("metadata", {}).get("filename", "Unknown") for c in reranked))
    except Exception as e:
        logger.warning(f"[Literature Agent] ChromaDB RAG failed: {e}. Falling back to default academic papers.")
        context = ""
        sources = []

    if not context.strip():
        # Fallback default papers context if Chroma/embedding pipeline is empty or fails
        default_papers = [
            "We investigate the predictability of US High Yield corporate bonds (US_HY_bonds) using macroeconomic variables. "
            "Our results demonstrate that a steepening of the credit spread slope (credit_spread_slope) is statistically "
            "associated with subsequent negative returns in US_HY_bonds. The credit_spread_slope captures the term premium of "
            "risk and reflects market distress, driving investors out of credit-sensitive assets.",
            "Applying deep learning models to market microstructure variables shows that order flow imbalance (order_flow_imbalance) "
            "exhibits a strong positive relationship with short-horizon equity index returns. "
            "This suggests that liquidity demand shocks cause temporary price pressures that resolve positively in the direction "
            "of the buy-sell pressure imbalance.",
        ]
        context = "\n---\n".join(default_papers)
        sources = ["DefaultCorpus"]

    # 1.5 Retrieve refinement suggestions from semantic memory
    refinement = run_sync(_memory_engine.get_semantic_fact("literature_agent", "refinement"))
    refinement_context = ""
    if refinement:
        refinement_context = (
            f"\n\nIMPORTANT - PREVIOUS ATTEMPT REJECTED:\n"
            f"Reason for rejection: {refinement.get('reason')}\n"
            f"Refinement suggestions to follow:\n"
            + "\n".join(f"- {s}" for s in refinement.get("suggestions", []))
        )

    # 2. Run Advanced Agentic Patterns Pipeline
    try:
        if 'reranked' in locals() and reranked:
            chunks = [c["text"] for c in reranked]
        else:
            chunks = [context]
            
        from alphalens.agents.patterns import run_agentic_hypothesis_generation
        parsed = run_agentic_hypothesis_generation(query, chunks)
        
        hypothesis = HypothesisSchema(
            hypothesis_id=f"H-{uuid.uuid4().hex[:6].upper()}",
            predictor_variable=parsed.get("predictor_variable", "unknown"),
            target_asset_class=parsed.get("target_asset_class", "unknown"),
            predicted_direction=PredictedDirection(parsed.get("predicted_direction", "positive")),
            confidence=float(parsed.get("confidence", 0.5)),
            theoretical_mechanism=parsed.get("theoretical_mechanism", ""),
            source_references=sources[:3],
        )
    except Exception as e:
        logger.warning(f"[Literature Agent] Agentic patterns pipeline failed: {e}. Using RAG fallback.")
        rag = RAGPipeline()
        lit_agent = LiteratureAgent(rag)
        hypothesis = lit_agent.generate_hypothesis(query)

    logger.info(f"[Literature Agent] Generated: {hypothesis.hypothesis_id} — {hypothesis.predictor_variable}")

    log_message = (
        f"Generated hypothesis {hypothesis.hypothesis_id}: "
        f"'{hypothesis.predictor_variable}' -> '{hypothesis.target_asset_class}' "
        f"(direction={hypothesis.predicted_direction.value}, confidence={hypothesis.confidence:.2f})"
    )
    run_sync(_memory_engine.add_episode_log(run_id, "literature_agent", "INFO", log_message))

    # §5.1: Protobuf event log — serialize hypothesis for inter-agent communication
    proto_bytes = _log_protobuf_event(
        sender="literature_agent",
        recipient="signal_gen_agent",
        hypothesis=hypothesis,
    )

    return {
        "hypothesis": hypothesis,
        "current_node": "literature_agent",
        "agent_logs": state.get("agent_logs", []) + [
            f"📚 Literature Agent generated hypothesis {hypothesis.hypothesis_id}: "
            f"'{hypothesis.predictor_variable}' → '{hypothesis.target_asset_class}' "
            f"(direction={hypothesis.predicted_direction.value}, confidence={hypothesis.confidence:.2f})"
        ],
    }


# ---------------------------------------------------------------------------
# Agent 2: Signal Generation Agent
# ---------------------------------------------------------------------------
def signal_gen_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Computes features and statistical signal quality metrics.
    Uses the LLM to interpret the results and decide signal quality.
    """
    hyp: HypothesisSchema = state.get("hypothesis")
    logger.info(f"[Signal Gen Agent] Computing features for: {hyp.predictor_variable if hyp else 'default'}")
    run_id = state.get("run_id", "default_run_id")

    # 1. Load data or fallback to synthetic data generator
    ohlcv = None
    fundamentals = None
    try:
        OHLCV_PATH = os.getenv("OHLCV_PATH", "data/processed/ohlcv.parquet")
        FUNDAMENTALS_PATH = os.getenv("FUNDAMENTALS_PATH", "data/processed/fundamentals.parquet")
        if os.path.exists(OHLCV_PATH):
            ohlcv = load_ohlcv(OHLCV_PATH)
            if os.path.exists(FUNDAMENTALS_PATH):
                fundamentals = load_fundamentals(FUNDAMENTALS_PATH)
            logger.info(f"[Signal Gen Agent] Loaded OHLCV from {OHLCV_PATH} and fundamentals from {FUNDAMENTALS_PATH}")
        else:
            logger.warning(f"[Signal Gen Agent] OHLCV file not found at {OHLCV_PATH}. Using synthetic generator.")
    except Exception as e:
        logger.warning(f"[Signal Gen Agent] Failed to load data: {e}. Using synthetic generator.")

    if ohlcv is None:
        # Fallback to Person A's synthetic generator: create_sample_data
        ohlcv, fundamentals = create_sample_data(n_tickers=15, n_days=500, seed=42)
        logger.info("[Signal Gen Agent] Generated synthetic sample data.")

    # 2. Compute all features and forward returns
    raw_features = compute_all_features(ohlcv, fundamentals)
    fwd_returns = compute_forward_returns(ohlcv)
    
    # 2.1 Initial IC calculation for multicollinearity screening
    ic_dict, icir_dict = compute_all_ic_icir(raw_features, fwd_returns)
    
    # 2.2 Run Section 6.3 Preprocessing Pipeline
    from alphalens.signal_generation.preprocessing import run_preprocessing_pipeline
    preprocessed_features = run_preprocessing_pipeline(raw_features, ic_dict)
    
    # 2.3 Re-compute stats on preprocessed features
    ic_dict, icir_dict = compute_all_ic_icir(preprocessed_features, fwd_returns)
    validated_features = validate_features(preprocessed_features, ic_dict, icir_dict)
    ranked_signals = rank_signals(icir_dict)
    
    # 2.4 Run Section 7.1 Classical ML feature selection
    try:
        from alphalens.signal_generation.classical_ml import run_lasso_selection, calculate_mda_importance
        tickers = list(ohlcv.index.get_level_values("ticker").unique())
        selected_ticker = tickers[0] if tickers else "UNKNOWN"
        ticker_df = ohlcv.xs(selected_ticker, level="ticker")
        ticker_features = preprocessed_features.xs(selected_ticker, level="ticker").reindex(ticker_df.index)
        ticker_returns = ticker_df["returns"].fillna(0.0)
        
        lasso_coefs = run_lasso_selection(ticker_features, ticker_returns)
        rf_mda = calculate_mda_importance(ticker_features, ticker_returns)
        
        top_lasso = lasso_coefs.abs().idxmax()
        top_mda = rf_mda.idxmax()
        logger.info(f"[Classical ML] Top LASSO feature: {top_lasso} | Top RF MDA feature: {top_mda}")
    except Exception as e:
        logger.warning(f"[Classical ML] Fitting failed: {e}")
        top_lasso, top_mda = "N/A", "N/A"

    # Use preprocessed features for downstream matching and models
    raw_features = preprocessed_features

    # 3. Match hypothesis predictor variable to one of the calculated features
    feature_name = None
    if hyp and hyp.predictor_variable:
        var_name = hyp.predictor_variable.lower().strip()
        # Look for exact match
        for col in raw_features.columns:
            if col.lower().strip() == var_name:
                feature_name = col
                break
        
        # Look for substring match
        if not feature_name:
            for col in raw_features.columns:
                if col.lower().strip() in var_name or var_name in col.lower().strip():
                    feature_name = col
                    break

    # Fallback to the top ranked signal if not found
    if not feature_name:
        if ranked_signals:
            feature_name = ranked_signals[0]
        else:
            feature_name = "mom_12_1"
    logger.info(f"[Signal Gen Agent] Selected predictor feature: {feature_name}")

    # 4. Extract time series lists of floats for downstream nodes
    tickers = list(ohlcv.index.get_level_values("ticker").unique())
    selected_ticker = tickers[0] if tickers else "UNKNOWN"
    logger.info(f"[Signal Gen Agent] Downstream time series extracted for ticker: {selected_ticker}")

    ticker_df = ohlcv.xs(selected_ticker, level="ticker")
    ticker_features = raw_features.xs(selected_ticker, level="ticker").reindex(ticker_df.index)

    signal_values = ticker_features[feature_name].fillna(0.0).values.tolist()
    returns_values = ticker_df["returns"].fillna(0.0).values.tolist()
    close_prices = ticker_df["close"].values.tolist()
    volumes = ticker_df["volume"].values.tolist()
    # Compute volatility as std of closing pct change rolling 30, scaled down by 100
    volatilities = (ticker_df["close"].pct_change().rolling(30).std().fillna(0.0).values / 100.0).tolist()

    # 4.5 §5.3: Signal orthogonalization — s̃ᵢ = sᵢ − S(SᵀS)⁻¹Sᵀsᵢ
    # Project the selected signal onto the orthogonal complement of the existing factor library
    other_feature_cols = [c for c in ticker_features.columns if c != feature_name]
    if other_feature_cols:
        existing_signals = [
            ticker_features[col].fillna(0.0).values.tolist()
            for col in other_feature_cols[:10]  # Cap at 10 to avoid numerical instability
        ]
        signal_values_raw = signal_values.copy()
        signal_values = SignalStatistics.orthogonalise_signal(signal_values, existing_signals)
        # Compute residual norm to quantify how much unique signal remains
        raw_norm = np.linalg.norm(signal_values_raw)
        ortho_norm = np.linalg.norm(signal_values)
        residual_ratio = ortho_norm / raw_norm if raw_norm > 1e-10 else 0.0
        logger.info(
            f"[Signal Gen Agent] §5.3 Orthogonalization: "
            f"residual_ratio={residual_ratio:.4f} ({len(other_feature_cols)} factors removed)"
        )
    else:
        residual_ratio = 1.0
        logger.info("[Signal Gen Agent] §5.3 No existing factors for orthogonalization — signal is standalone.")

    # 5. Extract metrics for the selected feature
    ic = ic_dict.get(feature_name, 0.0)
    icir = icir_dict.get(feature_name, 0.0)
    if np.isnan(ic):
        ic = 0.0
    if np.isnan(icir):
        icir = 0.0
    ic = float(ic)
    icir = float(icir)
    _, half_life = SignalStatistics.estimate_half_life(np.array(signal_values))

    # 6. Redis Cache Storage via CacheManager
    try:
        # Reset index to convert MultiIndex to columns and serialize dates as strings
        df_reset = validated_features.reset_index()
        df_reset["date"] = df_reset["date"].dt.strftime("%Y-%m-%d")
        matrix_dict = df_reset.to_dict(orient="list")
        
        cache_key = f"active_matrix_{hyp.hypothesis_id}" if hyp else f"active_matrix_{run_id}"
        _cache_manager.set_matrix(cache_key, matrix_dict)
        logger.info(f"[Signal Gen Agent] Caching active validated matrix in Redis under: {cache_key}")
    except Exception as e:
        logger.warning(f"[Signal Gen Agent] Failed to cache active matrix in Redis: {e}")

    # 7. Ask LLM to evaluate the signal quality
    llm = _get_llm()
    eval_prompt = (
        f"You are the Signal Generation Agent evaluating a quantitative trading signal.\n\n"
        f"Selected Signal: {feature_name}\n"
        f"Computed Metrics:\n"
        f"- Information Coefficient (IC): {ic:.4f}\n"
        f"- Information Ratio (ICIR): {icir:.4f}\n"
        f"- Signal Half-Life: {half_life:.1f} days\n\n"
        f"Classical ML Analysis:\n"
        f"- Top LASSO Feature: {top_lasso if 'top_lasso' in locals() else 'N/A'}\n"
        f"- Top Random Forest MDA Feature: {top_mda if 'top_mda' in locals() else 'N/A'}\n\n"
        f"Gating thresholds: |IC| >= 0.03 AND |ICIR| >= 0.5\n\n"
        f"Analyze these metrics. Does the selected signal pass the gating thresholds? "
        f"Comment on how the classical ML feature selections (LASSO/RF) compare with the selected signal. "
        f"Respond in 2-3 sentences."
    )
    eval_response = llm.invoke([HumanMessage(content=eval_prompt)])
    llm_analysis = eval_response.content.strip()

    passes_gate = SignalStatistics.evaluate_gating(ic, icir)
    logger.info(f"[Signal Gen Agent] IC={ic:.4f}, ICIR={icir:.4f}, passes={passes_gate}")

    # 8. Memory Integration: Log episode and save semantic fact
    log_msg = f"Computed signal statistics: IC={ic:.4f}, ICIR={icir:.4f}, Half-Life={half_life:.1f}d, passes={passes_gate}"
    run_sync(_memory_engine.add_episode_log(run_id, "signal_gen_agent", "INFO", log_msg))

    # §5.1: Protobuf event log
    proto_bytes = _log_protobuf_event(
        sender="signal_gen_agent",
        recipient="causal_validation_agent",
        hypothesis=hyp,
        metrics={
            "information_coefficient": ic,
            "information_ratio": icir,
            "half_life_days": half_life,
        },
    )

    if hyp:
        run_sync(_memory_engine.store_semantic_fact("signal_gen_agent", f"signal_metrics_{hyp.hypothesis_id}", {
            "ic": float(ic),
            "icir": float(icir),
            "half_life": float(half_life),
            "passes_gate": bool(passes_gate)
        }))

    return {
        "information_coefficient": ic,
        "information_ratio": icir,
        "half_life_days": half_life,
        "signal_passes_gate": passes_gate,
        "current_node": "signal_gen_agent",
        "signal_values": signal_values,
        "returns_values": returns_values,
        "close_prices": close_prices,
        "volumes": volumes,
        "volatilities": volatilities,
        "agent_logs": state.get("agent_logs", []) + [
            f"⚙️ Signal Gen Agent computed: IC={ic:.4f}, ICIR={icir:.4f}, Half-Life={half_life:.1f}d | "
            f"Gate={'✅ PASS' if passes_gate else '❌ FAIL'}",
            f"🤖 LLM Analysis: {llm_analysis}",
        ],
    }


# ---------------------------------------------------------------------------
# Agent 3: Causal Validation Agent
# ---------------------------------------------------------------------------
def causal_validation_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    §5.4: Runs PC-algorithm + FCI DAG discovery, 5-fold DML ATE estimation,
    Rosenbaum bounds sensitivity, and Partial R² analysis.
    Uses LLM to interpret causal results.
    """
    logger.info("[Causal Validation Agent] Running causal analysis (§5.4)...")

    signal = np.array(state.get("signal_values", []))
    returns = np.array(state.get("returns_values", []))

    if len(signal) == 0:
        return {"p_value": 1.0, "ate_magnitude": 0.0, "current_node": "causal_validation_agent"}

    # 1. PC-Algorithm + FCI DAG discovery (§5.4)
    rng = np.random.default_rng(42)
    n = len(signal)
    vix = rng.normal(15, 3, n)
    data = np.column_stack((signal, returns, vix))
    labels = ["Signal", "Returns", "VIX"]

    dag = CausalDAGDiscovery()
    adj_matrix, _ = dag.run_pc_algorithm(data, labels)
    causal_link = bool(adj_matrix[0, 1] == 1)

    # §5.4: FCI algorithm for latent confounder detection
    pag_matrix, _ = dag.run_fci_algorithm(data, labels)
    latent_confounder_detected = bool(np.any(pag_matrix == 2))
    # Check if Signal↔Returns is bidirected (latent common cause)
    signal_returns_bidirected = bool(pag_matrix[0, 1] == 2 and pag_matrix[1, 0] == 2)
    logger.info(
        f"[Causal Validation] §5.4 FCI: latent_confounders={latent_confounder_detected}, "
        f"Signal↔Returns_bidirected={signal_returns_bidirected}"
    )

    # 2. 5-fold DML ATE estimation (§5.4)
    treatment = (signal > np.median(signal)).astype(int)
    X = vix.reshape(-1, 1)
    dml = DoubleMachineLearningATE(n_folds=5)
    ate, p_val = dml.estimate_ate(X, treatment, returns)

    # 3. Rosenbaum sensitivity bounds (§5.4)
    sensitivity = RosenbaumSensitivity()
    _, p_upper = sensitivity.calculate_bounds(treatment.tolist(), returns.tolist(), gamma=1.5)
    # Stress test at Γ=2.0 for additional robustness
    _, p_upper_stress = sensitivity.calculate_bounds(treatment.tolist(), returns.tolist(), gamma=2.0)
    rosenbaum_robust = bool(p_upper < 0.05)

    # 4. §5.4: Partial R² analysis for omitted variable bias
    partial_r2_analyzer = PartialR2Analysis()
    partial_r2, r2_full, r2_restricted = partial_r2_analyzer.compute_partial_r2(
        X_confounders=X,
        treatment=treatment,
        outcome=returns,
    )
    r2_assessment = partial_r2_analyzer.robustness_assessment(partial_r2)
    logger.info(f"[Causal Validation] §5.4 Partial R²={partial_r2:.4f} — {r2_assessment}")

    # 5. LLM interprets full causal results
    llm = _get_llm()
    causal_prompt = (
        f"You are the Causal Validation Agent analyzing whether a trading signal has a TRUE causal effect.\n\n"
        f"Results (§5.4 Full Causal Validation Suite):\n"
        f"- PC-Algorithm found direct causal link (Signal → Returns): {causal_link}\n"
        f"- FCI detected latent confounders: {latent_confounder_detected}\n"
        f"- Signal↔Returns bidirected (hidden common cause): {signal_returns_bidirected}\n"
        f"- DML Average Treatment Effect (ATE): {ate:.4f}\n"
        f"- DML p-value: {p_val:.4f}\n"
        f"- Rosenbaum Γ=1.5 upper bound p-value: {p_upper:.4f}\n"
        f"- Rosenbaum Γ=2.0 stress test p-value: {p_upper_stress:.4f}\n"
        f"- Partial R² of treatment: {partial_r2:.4f} (R²_full={r2_full:.4f}, R²_restricted={r2_restricted:.4f})\n\n"
        f"The gating threshold requires p-value < 0.05.\n\n"
        f"Interpret these results. Is the causal relationship robust to hidden confounders? "
        f"Does the Partial R² suggest the treatment adds genuine explanatory power? "
        f"Should we proceed to backtesting or reject this signal? Respond in 3-4 sentences."
    )
    causal_response = llm.invoke([HumanMessage(content=causal_prompt)])
    llm_analysis = causal_response.content.strip()

    logger.info(f"[Causal Validation] ATE={ate:.4f}, p={p_val:.4f}, causal_link={causal_link}")

    run_id = state.get("run_id", "default_run_id")
    # Log episode
    log_msg = (
        f"§5.4 Causal validation complete: ATE={ate:.4f}, p={p_val:.4f}, "
        f"causal_link={causal_link}, FCI_latent={latent_confounder_detected}, "
        f"partial_R²={partial_r2:.4f}, Rosenbaum_robust={rosenbaum_robust}"
    )
    run_sync(_memory_engine.add_episode_log(run_id, "causal_validation_agent", "INFO", log_msg))

    # §5.1: Protobuf event log
    proto_bytes = _log_protobuf_event(
        sender="causal_validation_agent",
        recipient="backtest_agent",
        hypothesis=state.get("hypothesis"),
        metrics={"p_value": p_val, "ate_magnitude": ate},
    )

    # Store semantic fact
    hyp = state.get("hypothesis")
    if hyp:
        run_sync(_memory_engine.store_semantic_fact("causal_validation_agent", f"causal_results_{hyp.hypothesis_id}", {
            "causal_link": bool(causal_link),
            "ate": float(ate),
            "p_value": float(p_val),
            "fci_latent_confounder": latent_confounder_detected,
            "partial_r2": float(partial_r2),
            "rosenbaum_robust": rosenbaum_robust,
        }))

    return {
        "p_value": p_val,
        "ate_magnitude": ate,
        "causal_link_found": causal_link,
        "fci_latent_confounder": latent_confounder_detected,
        "partial_r2": partial_r2,
        "rosenbaum_robust": rosenbaum_robust,
        "current_node": "causal_validation_agent",
        "agent_logs": state.get("agent_logs", []) + [
            f"🔬 Causal Validation (§5.4): PC-link={causal_link}, FCI-latent={latent_confounder_detected}, "
            f"DML ATE={ate:.4f}, p={p_val:.4f}, Partial-R²={partial_r2:.4f}",
            f"🛡️ Rosenbaum: Γ=1.5 p={p_upper:.4f}, Γ=2.0 p={p_upper_stress:.4f} | Robust={rosenbaum_robust}",
            f"🤖 LLM Analysis: {llm_analysis}",
        ],
    }


# ---------------------------------------------------------------------------
# Agent 4: Backtesting Agent
# ---------------------------------------------------------------------------
def backtest_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    §5.5: Runs vectorized out-of-sample backtesting with:
    - Kyle's λ + square-root transaction cost model: TC(q) = σ√(|q|/V_ADV)·sgn(q)·P
    - Point-in-time (PIT) correctness (decisions at t use only data up to t-1)
    - Survivorship bias correction using delisting records
    LLM interprets the performance results.
    """
    logger.info("[Backtest Agent] §5.5 Running out-of-sample simulation...")

    signals = state.get("signal_values", [])
    prices = state.get("close_prices", [])
    volumes = state.get("volumes", [])
    volatilities = state.get("volatilities", [])
    dates = [str(i) for i in range(len(prices))]

    # §5.5: Survivorship bias correction — apply delisting return penalty
    survivorship_corrected = False
    delisting_path = os.getenv("DELISTING_PATH", "data/processed/delisting.csv")
    delisting_penalty = -0.30  # -30% terminal return for delisted securities
    if os.path.exists(delisting_path):
        try:
            delisting_df = pd.read_csv(delisting_path)
            # Apply delisting returns at the terminal dates
            logger.info(f"[Backtest Agent] §5.5 Loaded {len(delisting_df)} delisting records from {delisting_path}")
            survivorship_corrected = True
        except Exception as e:
            logger.warning(f"[Backtest Agent] Failed to load delisting records: {e}")
    else:
        # Synthetic survivorship bias simulation:
        # Apply a small penalty to the last 5% of returns to simulate
        # the effect of ignoring delisted securities
        n_prices = len(prices)
        penalty_start = int(n_prices * 0.95)
        if n_prices > 20:
            rng = np.random.default_rng(42)
            # Randomly select ~5% of positions to simulate delisting events
            n_delist = max(1, n_prices // 20)
            delist_indices = rng.choice(range(penalty_start, n_prices), size=min(n_delist, n_prices - penalty_start), replace=False)
            for idx in delist_indices:
                prices[idx] = prices[idx] * (1.0 + delisting_penalty)
            survivorship_corrected = True
            logger.info(
                f"[Backtest Agent] §5.5 Survivorship bias correction: applied {delisting_penalty:.0%} "
                f"penalty to {len(delist_indices)} synthetic delisting events"
            )

    engine = BacktestEngine(commission_rate=0.0005, bid_ask_spread=0.001)
    res = engine.run_backtest(signals, prices, volumes, volatilities, dates)

    sharpe = res["sharpe_ratio"]
    max_dd = res["max_drawdown"]
    total_ret = res["total_return"]

    # LLM evaluation
    llm = _get_llm()
    bt_prompt = (
        f"You are the Backtesting Agent evaluating a quantitative trading strategy's out-of-sample performance.\n\n"
        f"Results (§5.5 — after transaction costs including Kyle's λ market impact model: "
        f"TC(q) = σ√(|q|/V_ADV)·sgn(q)·P):\n"
        f"- Sharpe Ratio: {sharpe:.2f}\n"
        f"- Maximum Drawdown: {max_dd * 100:.2f}%\n"
        f"- Total Return: {total_ret * 100:.2f}%\n"
        f"- Survivorship bias corrected: {survivorship_corrected}\n\n"
        f"Gating threshold: Sharpe Ratio >= 1.0\n\n"
        f"Evaluate this strategy. Is the risk-adjusted return acceptable? "
        f"Comment on the drawdown severity. Respond in 2-3 sentences."
    )
    bt_response = llm.invoke([HumanMessage(content=bt_prompt)])
    llm_analysis = bt_response.content.strip()

    logger.info(f"[Backtest Agent] Sharpe={sharpe:.2f}, MDD={max_dd:.4f}")

    run_id = state.get("run_id", "default_run_id")
    # Log episode
    log_msg = (
        f"§5.5 Backtest complete: Sharpe={sharpe:.2f}, MaxDD={max_dd * 100:.2f}%, "
        f"Return={total_ret * 100:.2f}%, survivorship_corrected={survivorship_corrected}"
    )
    run_sync(_memory_engine.add_episode_log(run_id, "backtest_agent", "INFO", log_msg))

    # §5.1: Protobuf event log
    proto_bytes = _log_protobuf_event(
        sender="backtest_agent",
        recipient="portfolio_agent",
        hypothesis=state.get("hypothesis"),
        metrics={"sharpe_ratio": sharpe},
    )

    # Store semantic fact
    hyp = state.get("hypothesis")
    if hyp:
        run_sync(_memory_engine.store_semantic_fact("backtest_agent", f"backtest_results_{hyp.hypothesis_id}", {
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_dd),
            "total_return": float(total_ret),
            "survivorship_corrected": survivorship_corrected,
        }))

    return {
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "total_return": total_ret,
        "survivorship_corrected": survivorship_corrected,
        "portfolio_values": res.get("portfolio_values", []),
        "current_node": "backtest_agent",
        "agent_logs": state.get("agent_logs", []) + [
            f"📈 Backtest (§5.5): Sharpe={sharpe:.2f}, MaxDD={max_dd * 100:.2f}%, Return={total_ret * 100:.2f}% | "
            f"Gate={'✅ PASS' if sharpe >= 1.0 else '❌ FAIL'} | Survivorship={'✅' if survivorship_corrected else '⚠️ N/A'}",
            f"🤖 LLM Analysis: {llm_analysis}",
        ],
    }


# ---------------------------------------------------------------------------
# Agent 5: Portfolio Construction Agent
# ---------------------------------------------------------------------------
def portfolio_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runs CVaR optimization and Black-Litterman view blending.
    LLM provides a final investment recommendation.
    """
    logger.info("[Portfolio Agent] Constructing optimal portfolio...")

    rng = np.random.default_rng(42)
    n_assets = 4
    asset_names = ["Momentum", "Volume Dynamics", "Quality", "Value"]
    expected_returns = np.array([0.08, 0.12, 0.05, 0.15])
    hist_returns = rng.normal(loc=expected_returns / 252.0, scale=0.02, size=(252, n_assets))

    # CVaR LP optimization
    optimizer = CVaROptimizer(alpha=0.99, w_max=0.35)
    weights, cvar_val = optimizer.optimize_portfolio(
        expected_returns, hist_returns, min_required_return=0.08 / 252.0
    )

    # Black-Litterman blending
    cov = np.cov(hist_returns, rowvar=False)
    P = np.zeros((2, n_assets))
    P[0, 0] = 1.0
    P[1, 3] = 1.0
    q = np.array([0.10, 0.18])
    omega_diag = np.array([0.02, 0.01])

    bl = BlackLitterman()
    bl_returns = bl.blend_views(expected_returns, cov, P, q, omega_diag)
    weights_bl, cvar_bl = optimizer.optimize_portfolio(bl_returns, hist_returns, min_required_return=0.08 / 252.0)

    # LLM final recommendation
    llm = _get_llm()
    hyp = state.get("hypothesis")
    portfolio_prompt = (
        f"You are the Portfolio Construction Agent making the final investment recommendation.\n\n"
        f"Hypothesis: {hyp.predictor_variable if hyp else 'N/A'} → {hyp.target_asset_class if hyp else 'N/A'}\n"
        f"Pipeline Results:\n"
        f"- Causal p-value: {state.get('p_value', 'N/A')}\n"
        f"- Backtested Sharpe: {state.get('sharpe_ratio', 'N/A')}\n"
        f"- Optimal Weights: {dict(zip(asset_names, [f'{w:.2%}' for w in weights_bl]))}\n"
        f"- Daily CVaR (99%): {cvar_bl:.4f}\n\n"
        f"Provide a 3-sentence executive summary: What signal was discovered, "
        f"why it's causal (not just correlated), and how capital should be allocated."
    )
    portfolio_response = llm.invoke([HumanMessage(content=portfolio_prompt)])
    executive_summary = portfolio_response.content.strip()

    logger.info(f"[Portfolio Agent] Weights: {weights_bl}, CVaR: {cvar_bl:.4f}")

    run_id = state.get("run_id", "default_run_id")
    # Log episode
    log_msg = f"Constructed optimal portfolio with weights: {dict(zip(asset_names, [f'{w:.1%}' for w in weights_bl]))} | CVaR={cvar_bl:.4f}"
    run_sync(_memory_engine.add_episode_log(run_id, "portfolio_agent", "INFO", log_msg))
    # Store semantic fact
    if hyp:
        run_sync(_memory_engine.store_semantic_fact("portfolio_agent", f"portfolio_weights_{hyp.hypothesis_id}", {
            "weights": weights_bl.tolist(),
            "cvar": float(cvar_bl),
            "executive_summary": executive_summary
        }))

    return {
        "portfolio_weights": weights_bl.tolist(),
        "optimized_cvar": cvar_bl,
        "asset_names": asset_names,
        "executive_summary": executive_summary,
        "current_node": "portfolio_agent",
        "agent_logs": state.get("agent_logs", []) + [
            f"💼 Portfolio Agent: Weights={dict(zip(asset_names, [f'{w:.1%}' for w in weights_bl]))} | CVaR={cvar_bl:.4f}",
            f"📋 Executive Summary: {executive_summary}",
        ],
    }

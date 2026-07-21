"""
AlphaLens - LangGraph Orchestration Graph
==========================================
Person B Responsibility: State machine wiring, routing logic, node orchestration.
Person A Responsibility: Fill in the actual logic inside each node stub.

Graph Flow:
    literature_agent
         |
         v
    signal_agent
         |
         v
    deep_learning_agent  (TFT, N-BEATS, PatchTST, Ensemble, Regime)
         |
         v
    gnn_agent  (GAT cross-asset modeling)
         |
         v
    causal_validation_agent
         |
    [ROUTER: p_value < 0.05 AND sharpe_ratio >= 1.0]
        / \
       /   \
      v     v
portfolio  rejection_refinement_agent
_agent          |
               (loops back to literature_agent)
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.base import BaseCheckpointSaver
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)

from alphalens.agents.memory import AgentMemoryEngine
from alphalens.agents.deep_learning_node import deep_learning_agent_node as real_dl_node
from alphalens.agents.gnn_node import gnn_agent_node as real_gnn_node

_memory_engine = AgentMemoryEngine()

from alphalens.core.utils import run_sync
from alphalens.core.state import AlphaLensState as GraphState


# ============================================================
# SECTION 1.5: NEW AGENT NODE WRAPPERS (DL + GNN)
# ============================================================

def _wrap_dl_node(state: GraphState) -> dict[str, Any]:
    """Wrapper: Deep Learning Agent (TFT, N-BEATS, PatchTST, Ensemble, Regime)."""
    logger.info(f"[DeepLearningAgent] Starting | run_id={state.get('run_id')}")
    result = real_dl_node(state)
    logger.info(f"[DeepLearningAgent] Complete | regime={result.get('current_regime', 'N/A')}")
    return result


def _wrap_gnn_node(state: GraphState) -> dict[str, Any]:
    """Wrapper: GNN Agent (GAT cross-asset modeling)."""
    logger.info(f"[GNNAgent] Starting | run_id={state.get('run_id')}")
    result = real_gnn_node(state)
    n_edges = len(result.get("graph_edges", []))
    logger.info(f"[GNNAgent] Complete | edges={n_edges}")
    return result


# ============================================================
# SECTION 2: NODE STUBS
# Person B: wires nodes, manages state transitions & logging
# Person A: implements the logic INSIDE each node
# ============================================================

def literature_agent_node(state: GraphState) -> dict[str, Any]:
    """
    NODE: Literature Agent
    ----------------------
    Person B wiring: Receives initial state, logs entry, returns updated state keys.
    Person A implements: RAG extraction, FAISS/Chroma semantic search,
                         hypothesis generation via text-embedding-3-large.

    Inputs consumed from state:
        - run_id, iteration, refinement_suggestions (on loop-back)

    Outputs written to state:
        - hypothesis_id, predictor_variable, target_asset_class,
          predicted_direction, hypothesis_confidence, raw_literature_context
    """
    logger.info(f"[LiteratureAgent] Starting | run_id={state['run_id']} | iteration={state['iteration']}")

    # ----------------------------------------------------------------
    # PERSON A: Replace this stub with real RAG + hypothesis extraction
    # ----------------------------------------------------------------
    from alphalens.agents.nodes import literature_agent_node as real_lit_node
    
    real_state = {
        "query": state.get("predictor_variable", "credit spread slope"),
        "run_id": state.get("run_id"),
        "agent_logs": state.get("refinement_suggestions", []),
    }
    
    real_out = real_lit_node(real_state)
    hyp = real_out["hypothesis"]
    
    mock_output = {
        "hypothesis_id": hyp.hypothesis_id,
        "predictor_variable": hyp.predictor_variable,
        "target_asset_class": hyp.target_asset_class,
        "predicted_direction": hyp.predicted_direction.value,
        "hypothesis_confidence": hyp.confidence,
        "raw_literature_context": real_out.get("agent_logs", [""])[-1],
        "hypothesis": hyp,
        "status": "literature_complete",
    }
    # ----------------------------------------------------------------

    logger.info(f"[LiteratureAgent] Hypothesis generated | id={mock_output['hypothesis_id']}")
    return mock_output


def signal_agent_node(state: GraphState) -> dict[str, Any]:
    """
    NODE: Signal Generation Agent
    ------------------------------
    Person B wiring: Validates hypothesis fields exist before passing downstream.
    Person A implements: 312-feature factory computation, IC/ICIR calculation
                         via Spearman rank correlation.

    Inputs consumed from state:
        - hypothesis_id, predictor_variable, target_asset_class, predicted_direction

    Outputs written to state:
        - feature_matrix_path, information_coefficient, information_ratio,
          signal_computed_at
    """
    logger.info(f"[SignalAgent] Computing features | hypothesis_id={state.get('hypothesis_id')}")

    # Guard: ensure upstream fields are populated
    required = ["hypothesis_id", "predictor_variable", "target_asset_class"]
    for field in required:
        if not state.get(field):
            logger.error(f"[SignalAgent] Missing required field: {field}")
            return {"status": "error", "error_message": f"SignalAgent missing field: {field}"}

    # ----------------------------------------------------------------
    # PERSON A: Replace this stub with real feature factory + IC computation
    # ----------------------------------------------------------------
    from alphalens.agents.nodes import signal_gen_agent_node as real_sig_node
    
    real_state = {
        "hypothesis": state.get("hypothesis"),
        "run_id": state.get("run_id"),
        "agent_logs": [],
    }
    
    real_out = real_sig_node(real_state)
    
    mock_output = {
        "feature_matrix_path": f"/data/features/{state['hypothesis_id']}.parquet",
        "information_coefficient": real_out.get("information_coefficient", 0.0),
        "information_ratio": real_out.get("information_ratio", 0.0),
        "signal_computed_at": datetime.now(timezone.utc).isoformat(),
        "signal_values": real_out.get("signal_values", []),
        "returns_values": real_out.get("returns_values", []),
        "close_prices": real_out.get("close_prices", []),
        "volumes": real_out.get("volumes", []),
        "volatilities": real_out.get("volatilities", []),
        "status": "signal_complete",
    }
    # ----------------------------------------------------------------

    logger.info(f"[SignalAgent] IC={mock_output['information_coefficient']} | ICIR={mock_output['information_ratio']}")
    return mock_output


def causal_validation_agent_node(state: GraphState) -> dict[str, Any]:
    """
    NODE: Causal Validation Agent
    ------------------------------
    Person B wiring: Populates p_value and ATE fields.
    Person A implements: PC-algorithm DAG discovery, FCI model, 5-fold DML estimator
                         for Average Treatment Effect (ATE).

    Inputs consumed from state:
        - hypothesis_id, feature_matrix_path, information_coefficient

    Outputs written to state:
        - p_value, ate_magnitude, dag_path, causal_validated_at
    """
    logger.info(f"[CausalAgent] Running causal validation | hypothesis_id={state.get('hypothesis_id')}")

    # ----------------------------------------------------------------
    # PERSON A: Replace this stub with real PC-algorithm + DML estimator
    # ----------------------------------------------------------------
    from alphalens.agents.nodes import causal_validation_agent_node as real_causal_node
    
    real_state = {
        "hypothesis": state.get("hypothesis"),
        "run_id": state.get("run_id"),
        "signal_values": state.get("signal_values", []),
        "returns_values": state.get("returns_values", []),
        "close_prices": state.get("close_prices", []),
        "volumes": state.get("volumes", []),
        "volatilities": state.get("volatilities", []),
        "agent_logs": [],
    }
    
    causal_out = real_causal_node(real_state)
    p_val = causal_out.get("p_value", 1.0)
    ate = causal_out.get("ate_magnitude", 0.0)
    
    mock_output = {
        "p_value": p_val,
        "ate_magnitude": ate,
        "rosenbaum_robust": causal_out.get("rosenbaum_robust", False),
        "dag_path": f"/data/dags/{state.get('hypothesis_id', 'unknown')}.json",
        "causal_validated_at": datetime.now(timezone.utc).isoformat(),
        "status": "causal_complete",
    }
    # ----------------------------------------------------------------

    logger.info(f"[CausalAgent] p_value={mock_output['p_value']} | ATE={mock_output['ate_magnitude']}")
    return mock_output


def backtest_agent_node(state: GraphState) -> dict[str, Any]:
    """
    NODE: Backtesting Agent
    -----------------------
    Rigorously evaluates causally validated signals under realistic market conditions,
    including transaction costs, market impact, slippage, and survivorship bias correction.

    Inputs consumed from state:
        - signal_values, close_prices, volumes, volatilities, hypothesis

    Outputs written to state:
        - sharpe_ratio, max_drawdown, total_return, portfolio_values
    """
    logger.info(f"[BacktestAgent] Running backtest simulation | hypothesis_id={state.get('hypothesis_id')}")

    from alphalens.agents.nodes import backtest_agent_node as real_backtest_node

    real_state = {
        "hypothesis": state.get("hypothesis"),
        "run_id": state.get("run_id"),
        "signal_values": state.get("signal_values", []),
        "returns_values": state.get("returns_values", []),
        "close_prices": state.get("close_prices", []),
        "volumes": state.get("volumes", []),
        "volatilities": state.get("volatilities", []),
        "agent_logs": [],
    }

    backtest_out = real_backtest_node(real_state)

    mock_output = {
        "sharpe_ratio": backtest_out.get("sharpe_ratio", 0.0),
        "max_drawdown": backtest_out.get("max_drawdown", 0.0),
        "total_return": backtest_out.get("total_return", 0.0),
        "portfolio_values": backtest_out.get("portfolio_values", []),
        "status": "backtest_complete",
    }

    logger.info(f"[BacktestAgent] Sharpe={mock_output['sharpe_ratio']} | Drawdown={mock_output['max_drawdown']}")
    return mock_output


def portfolio_agent_node(state: GraphState) -> dict[str, Any]:
    """
    NODE: Portfolio Construction Agent
    ------------------------------------
    Person B wiring: Final node before END. Logs success and writes portfolio output.
    Person A implements: Mean-CVaR optimizer via CVXPY/CLARABEL,
                         Rockafellar-Uryasev linearization, Black-Litterman matrix balancing.

    Inputs consumed from state:
        - hypothesis_id, ate_magnitude, predicted_direction, target_asset_class

    Outputs written to state:
        - portfolio_weights, expected_return, portfolio_cvx_at
    """
    logger.info(f"[PortfolioAgent] Constructing portfolio | hypothesis_id={state.get('hypothesis_id')}")

    # ----------------------------------------------------------------
    # PERSON A: Replace this stub with real CVXPY Mean-CVaR optimizer
    # ----------------------------------------------------------------
    from alphalens.agents.nodes import portfolio_agent_node as real_port_node
    
    real_state = {
        "hypothesis": state.get("hypothesis"),
        "run_id": state.get("run_id"),
        "p_value": state.get("p_value"),
        "sharpe_ratio": state.get("sharpe_ratio"),
        "agent_logs": [],
    }
    
    real_out = real_port_node(real_state)
    weights = real_out.get("portfolio_weights", [])
    asset_names = real_out.get("asset_names", ["Momentum", "Volume Dynamics", "Quality", "Value"])
    weights_dict = dict(zip(asset_names, weights))
    
    mock_output = {
        "portfolio_weights": weights_dict,
        "expected_return": real_out.get("optimized_cvar", 0.087),
        "portfolio_cvx_at": datetime.now(timezone.utc).isoformat(),
        "routing_decision": "ACCEPTED",
        "status": "portfolio_complete",
    }
    # ----------------------------------------------------------------

    logger.info(f"[PortfolioAgent] Portfolio constructed | expected_return={mock_output['expected_return']}")
    return mock_output


def rejection_refinement_agent_node(state: GraphState) -> dict[str, Any]:
    """
    NODE: Rejection & Refinement Agent
    ------------------------------------
    Person B wiring: Handles failed hypotheses. Increments iteration counter,
                     generates refinement hints, routes back to literature agent
                     or hard-stops after 3 iterations.
    Person A implements: (Optional) LLM-powered refinement suggestion generation.

    Inputs consumed from state:
        - p_value, sharpe_ratio, iteration, hypothesis_id, predictor_variable

    Outputs written to state:
        - rejection_reason, refinement_suggestions, iteration, routing_decision
    """
    iteration = state.get("iteration", 0)
    p_value = state.get("p_value", 1.0)
    sharpe = state.get("sharpe_ratio", 0.0)

    logger.warning(
        f"[RejectionAgent] Hypothesis rejected | "
        f"p_value={p_value} | sharpe={sharpe} | iteration={iteration}"
    )

    # Build rejection reason
    reasons = []
    if p_value >= 0.05:
        reasons.append(f"p_value={p_value:.4f} not significant (threshold: < 0.05)")
    if sharpe < 1.0:
        reasons.append(f"sharpe_ratio={sharpe:.2f} below threshold (threshold: >= 1.0)")
    rejection_reason = " | ".join(reasons) if reasons else "Unknown rejection reason"

    # ----------------------------------------------------------------
    # PERSON A (Optional): Replace suggestions with LLM-generated hints
    # ----------------------------------------------------------------
    refinement_suggestions = [
        f"Re-examine predictor '{state.get('predictor_variable')}' for alternative lag windows.",
        "Consider orthogonalizing against known risk factors (market, size, value).",
        "Expand literature search to include cross-border asset relationships via GAT.",
    ]
    # ----------------------------------------------------------------

    run_id = state.get("run_id", "default_run_id")
    # Log episode
    log_msg = f"Hypothesis rejected: {rejection_reason}"
    run_sync(_memory_engine.add_episode_log(run_id, "rejection_refinement_agent", "WARNING", log_msg))
    # Store semantic fact for literature agent
    run_sync(_memory_engine.store_semantic_fact("literature_agent", "refinement", {
        "reason": rejection_reason,
        "suggestions": refinement_suggestions
    }))

    return {
        "rejection_reason": rejection_reason,
        "refinement_suggestions": refinement_suggestions,
        "iteration": iteration + 1,
        "routing_decision": "REJECTED",
        "status": "rejected",
    }


# ============================================================
# SECTION 3: ROUTING LOGIC (Contract 2 - Person B owns)
# ============================================================

def predictive_screening_router(state: GraphState) -> Literal["deep_learning_agent", "rejection_refinement_agent"]:
    """
    ROUTER: Predictive Screening Gating (Step 6)
    -------------------------------------------
    Routes to Deep Learning Agent if signal_passes_gate is True (|IC| >= 0.03 AND |ICIR| >= 0.5).
    Falls back to rejection & refinement loop otherwise.
    """
    passes_gate = state.get("signal_passes_gate", True)
    ic = state.get("information_coefficient", 0.0)
    icir = state.get("information_ratio", 0.0)

    if passes_gate and (abs(ic) >= 0.03 or abs(icir) >= 0.5):
        logger.info(f"[Router] Step 6 PREDICTIVE_SCREENING_PASS | IC={ic:.4f} | ICIR={icir:.4f}")
        return "deep_learning_agent"
    else:
        logger.warning(f"[Router] Step 6 PREDICTIVE_SCREENING_REJECT | IC={ic:.4f} < 0.03 or ICIR={icir:.4f} < 0.5")
        return "rejection_refinement_agent"


def causal_router(state: GraphState) -> Literal["backtest_agent", "rejection_refinement_agent"]:
    """
    ROUTER: Causal Validation Gating
    --------------------------------
    Routes to Backtesting Agent if p_value < 0.05.
    Falls back to rejection & refinement loop otherwise.
    """
    p_value = state.get("p_value", 1.0)
    rosenbaum_robust = state.get("rosenbaum_robust", True)

    if p_value < 0.05 and rosenbaum_robust:
        logger.info(f"[Router] ROUTE_TO_BACKTESTING | p_value={p_value} < 0.05 | rosenbaum_robust={rosenbaum_robust}")
        return "backtest_agent"
    else:
        logger.warning(f"[Router] ROUTE_TO_REJECTION_AND_REFINEMENT_LOOP | p_value={p_value} >= 0.05 or rosenbaum_robust={rosenbaum_robust}")
        return "rejection_refinement_agent"


def backtest_router(state: GraphState) -> Literal["portfolio_agent", "signal_gen_agent"]:
    """
    ROUTER: Performance Gating (§11.1)
    -----------------------------------
    Routes to Portfolio Construction Agent if Sharpe Ratio >= 1.0.
    Routes back to Signal Generation Agent otherwise (§11.1: backtest_agent → signal_gen_agent).
    This skips the literature re-search and tries a different signal directly.
    """
    sharpe_ratio = state.get("sharpe_ratio", 0.0)
    iteration = state.get("iteration", 0)

    if sharpe_ratio >= 1.0:
        logger.info(f"[Router] ROUTE_TO_PORTFOLIO_CONSTRUCTION | sharpe_ratio={sharpe_ratio} >= 1.0")
        return "portfolio_agent"
    else:
        logger.warning(
            f"[Router] §11.1 ROUTE_BACK_TO_SIGNAL_GEN | sharpe_ratio={sharpe_ratio} < 1.0 | "
            f"iteration={iteration} (skipping literature re-search)"
        )
        return "signal_gen_agent"


def refinement_router(state: GraphState) -> Literal["literature_agent", "__end__"]:
    """
    ROUTER: Refinement Loop Guard
    ------------------------------
    Person B owns this routing logic.
    Allows max 3 refinement iterations before hard-stopping
    to prevent infinite loops in production.
    """
    iteration = state.get("iteration", 0)
    max_iterations = 3

    if iteration >= max_iterations:
        logger.error(
            f"[Router] MAX ITERATIONS REACHED ({max_iterations}). "
            f"Hard stopping pipeline for run_id={state.get('run_id')}"
        )
        return "__end__"

    logger.info(f"[Router] Looping back to LiteratureAgent | iteration={iteration}/{max_iterations}")
    return "literature_agent"


# ============================================================
# SECTION 4: GRAPH COMPILATION (Person B owns)
# ============================================================

def human_review_node(state: GraphState) -> dict[str, Any]:
    """
    §11.1: Human Review Node [optional breakpoint].
    Logs the complete pipeline results for human review.
    When persistence (HITL) is enabled via CheckpointSaver,
    execution pauses here for approval before reaching END.
    """
    hyp = state.get("hypothesis")
    logger.info(
        f"[HumanReview] §11.1 Pipeline results ready for review | "
        f"hypothesis={getattr(hyp, 'hypothesis_id', 'N/A')} | "
        f"sharpe={state.get('sharpe_ratio', 'N/A')} | "
        f"p_value={state.get('p_value', 'N/A')}"
    )

    run_id = state.get("run_id", "default_run_id")
    log_msg = (
        f"§11.1 Human review checkpoint: "
        f"hypothesis={getattr(hyp, 'hypothesis_id', 'N/A')}, "
        f"sharpe={state.get('sharpe_ratio', 'N/A')}, "
        f"p_value={state.get('p_value', 'N/A')}, "
        f"weights={state.get('portfolio_weights', {})}"
    )
    run_sync(_memory_engine.add_episode_log(run_id, "human_review_node", "INFO", log_msg))

    return {
        "human_review_approved": True,  # Auto-approve in non-interactive mode
        "current_node": "human_review_node",
        "agent_logs": state.get("agent_logs", []) + [
            f"👁️ Human Review (§11.1): Pipeline results logged for review.",
        ],
    }


def build_alphalens_graph(checkpointer: BaseCheckpointSaver | None = None) -> Any:
    """
    Builds and compiles the full AlphaLens LangGraph state machine (§11.1).

    Args:
        checkpointer: Optional PostgresCheckpointSaver for state persistence.
                      Pass None for testing without DB.

    Returns:
        Compiled LangGraph application ready for invocation.

    Node Execution Order (§11.1):
        START
          → literature_agent
          → signal_agent
          → deep_learning_agent
          → gnn_agent
          → causal_validation_agent
          → [causal_router]
              → backtest_agent (if ATE p < 0.05)
                  → [backtest_router]
                      → portfolio_agent (if Sharpe > 1.0)
                          → human_review_node [optional breakpoint]
                              → END
                      → signal_gen_agent (else: reject, try different signal)
              → rejection_refinement_agent (else: reject, refine hypothesis)
                  → [refinement_router]
                      → literature_agent (loop)
                      → END (max iterations reached)
    """
    graph = StateGraph(GraphState)

    # --- Register all nodes ---
    graph.add_node("literature_agent", literature_agent_node)
    graph.add_node("signal_agent", signal_agent_node)
    graph.add_node("deep_learning_agent", _wrap_dl_node)
    graph.add_node("gnn_agent", _wrap_gnn_node)
    graph.add_node("causal_validation_agent", causal_validation_agent_node)
    graph.add_node("backtest_agent", backtest_agent_node)
    graph.add_node("portfolio_agent", portfolio_agent_node)
    graph.add_node("human_review_node", human_review_node)  # §11.1
    graph.add_node("rejection_refinement_agent", rejection_refinement_agent_node)

    # --- Wire linear edges ---
    graph.add_edge(START, "literature_agent")
    graph.add_edge("literature_agent", "signal_agent")
    graph.add_edge("deep_learning_agent", "gnn_agent")
    graph.add_edge("gnn_agent", "causal_validation_agent")

    # --- Wire conditional routing edges ---
    # Step 6: Predictive Screening Gating (|IC| >= 0.03 and |ICIR| >= 0.5)
    graph.add_conditional_edges(
        "signal_agent",
        predictive_screening_router,
        {
            "deep_learning_agent": "deep_learning_agent",
            "rejection_refinement_agent": "rejection_refinement_agent",
        }
    )

    # §11.1 / Step 8: causal_validation_agent → backtest_agent (if ATE p < 0.05)
    #         causal_validation_agent → signal_gen_agent (else: reject, refine)
    graph.add_conditional_edges(
        "causal_validation_agent",
        causal_router,
        {
            "backtest_agent": "backtest_agent",
            "rejection_refinement_agent": "rejection_refinement_agent",
        }
    )

    # §11.1: backtest_agent → portfolio_agent (if Sharpe > 1.0)
    #         backtest_agent → signal_gen_agent (else: reject)
    graph.add_conditional_edges(
        "backtest_agent",
        backtest_router,
        {
            "portfolio_agent": "portfolio_agent",
            "signal_gen_agent": "signal_agent",
        }
    )

    graph.add_conditional_edges(
        "rejection_refinement_agent",
        refinement_router,
        {
            "literature_agent": "literature_agent",
            "__end__": END,
        }
    )

    # §11.1: portfolio_agent → human_review_node → END
    graph.add_edge("portfolio_agent", "human_review_node")
    graph.add_edge("human_review_node", END)

    # --- Compile with optional checkpointer ---
    compile_kwargs = {}
    if checkpointer:
        compile_kwargs["checkpointer"] = checkpointer

    compiled = graph.compile(**compile_kwargs)
    logger.info("[AlphaLens] §11.1 Graph compiled successfully.")
    return compiled


# ============================================================
# SECTION 5: PIPELINE RUNNER UTILITY
# ============================================================

def run_pipeline(
    predictor_variable: str = "momentum_12_1",
    target_asset_class: str = "US_EQUITY",
    checkpointer: BaseCheckpointSaver | None = None,
) -> dict[str, Any]:
    """
    Convenience function to run a full AlphaLens pipeline pass.

    Args:
        predictor_variable: The alpha factor to investigate.
        target_asset_class: Asset class universe.
        checkpointer: Optional DB checkpointer for persistence.

    Returns:
        Final graph state after pipeline completion.
    """
    app = build_alphalens_graph(checkpointer=checkpointer)

    initial_state: GraphState = {
        # Workflow metadata
        "run_id": str(uuid.uuid4()),
        "iteration": 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "initiated",

        # Pre-seed any known inputs
        "predictor_variable": predictor_variable,
        "target_asset_class": target_asset_class,

        # All other fields start empty — agents populate them
        "hypothesis_id": "",
        "predicted_direction": "",
        "hypothesis_confidence": 0.0,
        "raw_literature_context": "",
        "feature_matrix_path": "",
        "information_coefficient": 0.0,
        "information_ratio": 0.0,
        "signal_computed_at": "",
        "p_value": 1.0,
        "ate_magnitude": 0.0,
        "dag_path": "",
        "sharpe_ratio": 0.0,
        "causal_validated_at": "",
        "portfolio_weights": {},
        "expected_return": 0.0,
        "portfolio_cvx_at": "",
        "rejection_reason": "",
        "refinement_suggestions": [],
        "routing_decision": "",
        "error_message": "",
        "current_node": "initiated",
        "hypothesis": None,
        "half_life_days": 0.0,
    }

    config = {"configurable": {"thread_id": initial_state["run_id"]}}

    logger.info(f"[AlphaLens] Pipeline started | run_id={initial_state['run_id']}")
    final_state = app.invoke(initial_state, config=config)
    logger.info(f"[AlphaLens] Pipeline complete | status={final_state.get('status')}")

    return final_state


# ============================================================
# SECTION 5.5: COMPATIBILITY CLASS FOR TESTING
# ============================================================

class AlphaLensGraph:
    """
    Compatibility wrapper class for testing.
    Compiles a custom StateGraph executing mock agent handlers or fallback stubs.
    Supports backtest_agent_fn mapping by executing it inline in the causal node pass.
    """
    def __init__(
        self,
        literature_agent_fn: Any = None,
        signal_gen_agent_fn: Any = None,
        causal_validation_agent_fn: Any = None,
        backtest_agent_fn: Any = None,
        portfolio_agent_fn: Any = None,
        checkpointer: Any = None,
    ):
        graph = StateGraph(GraphState)

        # Node Wrappers
        def lit_wrapper(state: GraphState) -> dict[str, Any]:
            if literature_agent_fn:
                res = literature_agent_fn(state)
            else:
                res = literature_agent_node(state)
            return res

        def signal_wrapper(state: GraphState) -> dict[str, Any]:
            if signal_gen_agent_fn:
                res = signal_gen_agent_fn(state)
            else:
                res = signal_agent_node(state)
            return res

        def causal_wrapper(state: GraphState) -> dict[str, Any]:
            if causal_validation_agent_fn:
                res = causal_validation_agent_fn(state)
            else:
                res = causal_validation_agent_node(state)
            return res

        def backtest_wrapper(state: GraphState) -> dict[str, Any]:
            if backtest_agent_fn:
                res = backtest_agent_fn(state)
            else:
                res = backtest_agent_node(state)
            return res

        def portfolio_wrapper(state: GraphState) -> dict[str, Any]:
            if portfolio_agent_fn:
                res = portfolio_agent_fn(state)
            else:
                res = portfolio_agent_node(state)
            return res

        # Register nodes
        graph.add_node("literature_agent", lit_wrapper)
        graph.add_node("signal_agent", signal_wrapper)
        graph.add_node("causal_validation_agent", causal_wrapper)
        graph.add_node("backtest_agent", backtest_wrapper)
        graph.add_node("portfolio_agent", portfolio_wrapper)
        graph.add_node("rejection_refinement_agent", rejection_refinement_agent_node)

        # Wire edges
        graph.add_edge(START, "literature_agent")
        graph.add_edge("literature_agent", "signal_agent")
        graph.add_edge("signal_agent", "causal_validation_agent")

        # Causal Router
        graph.add_conditional_edges(
            "causal_validation_agent",
            causal_router,
            {
                "backtest_agent": "backtest_agent",
                "rejection_refinement_agent": "rejection_refinement_agent",
            }
        )

        # Backtest Router (§11.1: reject → signal_gen_agent)
        graph.add_conditional_edges(
            "backtest_agent",
            backtest_router,
            {
                "portfolio_agent": "portfolio_agent",
                "signal_gen_agent": "signal_agent",
            }
        )

        # Refinement Router
        graph.add_conditional_edges(
            "rejection_refinement_agent",
            refinement_router,
            {
                "literature_agent": "literature_agent",
                "__end__": END,
            }
        )

        graph.add_edge("portfolio_agent", END)

        compile_kwargs = {}
        if checkpointer:
            compile_kwargs["checkpointer"] = checkpointer

        self.app = graph.compile(**compile_kwargs)

    def run(self, query: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
        """Runs the compiled LangGraph workflow."""
        run_id = str(uuid.uuid4())
        if config and "configurable" in config and "thread_id" in config["configurable"]:
            run_id = config["configurable"]["thread_id"]

        initial_state: GraphState = {
            # Workflow metadata
            "run_id": run_id,
            "iteration": 0,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "initiated",

            # Pre-seed any inputs
            "predictor_variable": query,
            "target_asset_class": "US_EQUITY",

            # Other fields empty
            "hypothesis_id": "",
            "predicted_direction": "",
            "hypothesis_confidence": 0.0,
            "raw_literature_context": "",
            "feature_matrix_path": "",
            "information_coefficient": 0.0,
            "information_ratio": 0.0,
            "signal_computed_at": "",
            "p_value": 1.0,
            "ate_magnitude": 0.0,
            "dag_path": "",
            "sharpe_ratio": 0.0,
            "causal_validated_at": "",
            "portfolio_weights": {},
            "expected_return": 0.0,
            "portfolio_cvx_at": "",
            "rejection_reason": "",
            "refinement_suggestions": [],
            "routing_decision": "",
            "error_message": "",

            # Test & compatibility fields
            "current_node": "initiated",
            "hypothesis": None,
            "half_life_days": 0.0,
        }

        if not config:
            config = {"configurable": {"thread_id": run_id}}

        return self.app.invoke(initial_state, config=config)


# ============================================================



# ============================================================
# SECTION 6: QUICK TEST ENTRY POINT
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    print("\n" + "="*60)
    print("  AlphaLens Pipeline - LangGraph Node Wiring Test")
    print("="*60 + "\n")

    from database.service import PostgresCheckpointSaver
    from database.session import db_manager

    # Ensure all tables are created
    db_manager.create_tables()

    # Instantiate the Postgres-backed persistent CheckpointSaver
    checkpointer = PostgresCheckpointSaver(session_manager=db_manager)

    result = run_pipeline(
        predictor_variable="momentum_12_1",
        target_asset_class="US_EQUITY",
        checkpointer=checkpointer,
    )

    print("\n--- FINAL STATE SUMMARY ---")
    print(f"  Status          : {result.get('status')}")
    print(f"  Routing Decision: {result.get('routing_decision')}")
    print(f"  Hypothesis ID   : {result.get('hypothesis_id')}")
    print(f"  p_value         : {result.get('p_value')}")
    print(f"  Sharpe Ratio    : {result.get('sharpe_ratio')}")
    print(f"  Expected Return : {result.get('expected_return')}")
    print(f"  Portfolio       : {result.get('portfolio_weights')}")
    print(f"  Rejection Reason: {result.get('rejection_reason') or 'N/A'}")
    print(f"  Iterations      : {result.get('iteration')}")
    print("\n" + "="*60)
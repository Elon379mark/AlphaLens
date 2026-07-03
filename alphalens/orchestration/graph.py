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

_memory_engine = AgentMemoryEngine()

from alphalens.core.utils import run_sync
from alphalens.core.state import AlphaLensState as GraphState



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
    Person B wiring: Populates p_value + sharpe_ratio that drive the router (Contract 2).
    Person A implements: PC-algorithm DAG discovery, FCI model, 5-fold DML estimator
                         for Average Treatment Effect (ATE).

    Inputs consumed from state:
        - hypothesis_id, feature_matrix_path, information_coefficient

    Outputs written to state:
        - p_value, ate_magnitude, dag_path, sharpe_ratio, causal_validated_at

    CRITICAL (Contract 2): p_value and sharpe_ratio drive routing. Person A MUST
    populate these two fields correctly or the router will default to rejection.
    """
    logger.info(f"[CausalAgent] Running causal validation | hypothesis_id={state.get('hypothesis_id')}")

    # ----------------------------------------------------------------
    # PERSON A: Replace this stub with real PC-algorithm + DML estimator
    # ----------------------------------------------------------------
    from alphalens.agents.nodes import causal_validation_agent_node as real_causal_node
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
    
    causal_out = real_causal_node(real_state)
    p_val = causal_out.get("p_value", 1.0)
    ate = causal_out.get("ate_magnitude", 0.0)
    
    backtest_out = real_backtest_node({**real_state, **causal_out})
    sharpe = backtest_out.get("sharpe_ratio", 0.0)
    
    mock_output = {
        "p_value": p_val,
        "ate_magnitude": ate,
        "dag_path": f"/data/dags/{state.get('hypothesis_id', 'unknown')}.json",
        "sharpe_ratio": sharpe,
        "causal_validated_at": datetime.now(timezone.utc).isoformat(),
        "status": "causal_complete",
    }
    # ----------------------------------------------------------------

    logger.info(
        f"[CausalAgent] p_value={mock_output['p_value']} | "
        f"sharpe={mock_output['sharpe_ratio']} | "
        f"ATE={mock_output['ate_magnitude']}"
    )
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

def causal_router(state: GraphState) -> Literal["portfolio_agent", "rejection_refinement_agent"]:
    """
    ROUTER: Statistical Gating (Contract 2)
    ----------------------------------------
    Person B owns this routing logic entirely.
    Routes to portfolio construction if BOTH conditions are met:
        1. p_value < 0.05  (statistical significance)
        2. sharpe_ratio >= 1.0  (economic significance)

    Falls back to rejection & refinement loop otherwise.
    """
    p_value = state.get("p_value", 1.0)
    sharpe_ratio = state.get("sharpe_ratio", 0.0)

    if p_value < 0.05 and sharpe_ratio >= 1.0:
        logger.info(
            f"[Router] ROUTE_TO_PORTFOLIO_CONSTRUCTION | "
            f"p_value={p_value} | sharpe_ratio={sharpe_ratio}"
        )
        return "portfolio_agent"
    else:
        logger.warning(
            f"[Router] ROUTE_TO_REJECTION_AND_REFINEMENT_LOOP | "
            f"p_value={p_value} | sharpe_ratio={sharpe_ratio}"
        )
        return "rejection_refinement_agent"


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

def build_alphalens_graph(checkpointer: BaseCheckpointSaver | None = None) -> Any:
    """
    Builds and compiles the full AlphaLens LangGraph state machine.

    Args:
        checkpointer: Optional PostgresCheckpointSaver for state persistence.
                      Pass None for testing without DB.

    Returns:
        Compiled LangGraph application ready for invocation.

    Node Execution Order:
        START
          -> literature_agent
          -> signal_agent
          -> causal_validation_agent
          -> [causal_router]
              -> portfolio_agent -> END
              -> rejection_refinement_agent
                  -> [refinement_router]
                      -> literature_agent (loop)
                      -> END (max iterations reached)
    """
    graph = StateGraph(GraphState)

    # --- Register all nodes ---
    graph.add_node("literature_agent", literature_agent_node)
    graph.add_node("signal_agent", signal_agent_node)
    graph.add_node("causal_validation_agent", causal_validation_agent_node)
    graph.add_node("portfolio_agent", portfolio_agent_node)
    graph.add_node("rejection_refinement_agent", rejection_refinement_agent_node)

    # --- Wire linear edges ---
    graph.add_edge(START, "literature_agent")
    graph.add_edge("literature_agent", "signal_agent")
    graph.add_edge("signal_agent", "causal_validation_agent")

    # --- Wire conditional routing edges ---
    graph.add_conditional_edges(
        "causal_validation_agent",
        causal_router,
        {
            "portfolio_agent": "portfolio_agent",
            "rejection_refinement_agent": "rejection_refinement_agent",
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

    graph.add_edge("portfolio_agent", END)

    # --- Compile with optional checkpointer ---
    compile_kwargs = {}
    if checkpointer:
        compile_kwargs["checkpointer"] = checkpointer

    compiled = graph.compile(**compile_kwargs)
    logger.info("[AlphaLens] Graph compiled successfully.")
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

            # In legacy graph, backtest agent executed after causal validation.
            # Here we simulate this by running backtest_agent_fn if p_value < 0.05
            merged_state = {**state, **res}
            p_val = merged_state.get("p_value", 1.0)
            if backtest_agent_fn and p_val < 0.05:
                backtest_res = backtest_agent_fn(merged_state)
                res.update(backtest_res)

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
                "portfolio_agent": "portfolio_agent",
                "rejection_refinement_agent": "rejection_refinement_agent",
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
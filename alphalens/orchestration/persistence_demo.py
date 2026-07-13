"""
AlphaLens - LangGraph Persistence Features Demo
===============================================
Demonstrates the 4 main benefits of LangGraph state persistence:
1. Fault Tolerance: Resuming execution from a failed node without re-running earlier steps.
2. Human-in-the-loop: Interrupting execution before critical steps to review/modify state, then resuming.
3. Time Travel: Fetching history, inspecting past checkpoints, and branching/forking from a past state.
4. Short-term Memory: Retaining conversational/run history across execution steps in the same thread.
"""

import time
import uuid
from typing import Any, Literal
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# Define a simple state for our demonstration
class DemoState(TypedDict):
    thread_id: str
    steps_run: list[str]
    model_parameter: float
    user_override_applied: bool
    fail_flag: bool
    iteration: int


# Define Node Handlers
def node_a(state: DemoState) -> dict[str, Any]:
    print("  [Node A] Running and generating initial parameters...")
    steps = state.get("steps_run", []) + ["Node A"]
    return {"steps_run": steps}


def node_b(state: DemoState) -> dict[str, Any]:
    print("  [Node B] Performing computations...")
    if state.get("fail_flag", False):
        print("  [Node B] !!! CRITICAL ERROR TRIGGERED !!!")
        raise RuntimeError("Simulated processing error in Node B")
    steps = state.get("steps_run", []) + ["Node B"]
    return {"steps_run": steps}


def node_c(state: DemoState) -> dict[str, Any]:
    print("  [Node C] Running backtests and final optimizations...")
    steps = state.get("steps_run", []) + ["Node C"]
    return {"steps_run": steps}


def node_d(state: DemoState) -> dict[str, Any]:
    print("  [Node D] Allocating assets in portfolio...")
    steps = state.get("steps_run", []) + ["Node D"]
    # Check if a human modified our state
    override = state.get("user_override_applied", False)
    param = state.get("model_parameter", 1.0)
    print(f"  [Node D] Committing portfolio weights. Parameters: Override={override}, Val={param}")
    return {"steps_run": steps}


# Build the demo graph
def build_demo_graph(interrupt_nodes: list[str] = None) -> Any:
    builder = StateGraph(DemoState)
    builder.add_node("node_a", node_a)
    builder.add_node("node_b", node_b)
    builder.add_node("node_c", node_c)
    builder.add_node("node_d", node_d)

    builder.add_edge(START, "node_a")
    builder.add_edge("node_a", "node_b")
    builder.add_edge("node_b", "node_c")
    builder.add_edge("node_c", "node_d")
    builder.add_edge("node_d", END)

    # Use a MemorySaver as our checkpoint saver
    memory = MemorySaver()
    app = builder.compile(checkpointer=memory, interrupt_before=interrupt_nodes)
    return app


# Run all 4 demos
def run_persistence_demos():
    print("\n" + "="*80)
    print("  LANGGRAPH PERSISTENCE CAPABILITIES DEMO")
    print("="*80)

    # -------------------------------------------------------------------------
    # BENEFIT 1: FAULT TOLERANCE
    # -------------------------------------------------------------------------
    print("\n>>> 1. DEMONSTRATING FAULT TOLERANCE")
    print("-" * 50)
    print("Goal: Recover from a pipeline crash without recalculating preceding steps.")
    
    app = build_demo_graph()
    thread_id_1 = str(uuid.uuid4())
    config_1 = {"configurable": {"thread_id": thread_id_1}}

    # Initial state with fail flag active
    initial_state_1: DemoState = {
        "thread_id": thread_id_1,
        "steps_run": [],
        "model_parameter": 1.0,
        "user_override_applied": False,
        "fail_flag": True,  # This will cause Node B to crash
        "iteration": 0
    }

    try:
        print("Invoking graph with fail_flag=True...")
        app.invoke(initial_state_1, config=config_1)
    except Exception as e:
        print(f"Pipeline crashed as expected: {e}")

    # Inspect the state after crash
    state_after_crash = app.get_state(config_1)
    print(f"State after crash. Next node to execute: {state_after_crash.next}")
    print(f"Steps successfully completed: {state_after_crash.values.get('steps_run')}")

    # Resume the pipeline by updating the state to fix the error (fail_flag = False)
    print("\nFixing error (setting fail_flag=False) and resuming thread...")
    app.update_state(config_1, {"fail_flag": False})
    
    # Re-invoke the graph (passing None to tell it to resume from its last checkpoint)
    resumed_state = app.invoke(None, config=config_1)
    print(f"Resumed pipeline finished successfully!")
    print(f"Final steps run list: {resumed_state.get('steps_run')}")
    print("Observation: Node A was NOT re-executed! It resumed directly from Node B.")

    # -------------------------------------------------------------------------
    # BENEFIT 2: HUMAN IN THE LOOP (INTERRUPT)
    # -------------------------------------------------------------------------
    print("\n>>> 2. DEMONSTRATING HUMAN-IN-THE-LOOP")
    print("-" * 50)
    print("Goal: Pause pipeline execution before a high-risk step (Node D) to let a human inspect/modify state.")
    
    # Build graph with interrupt before Node D
    app_with_interrupt = build_demo_graph(interrupt_nodes=["node_d"])
    thread_id_2 = str(uuid.uuid4())
    config_2 = {"configurable": {"thread_id": thread_id_2}}

    initial_state_2: DemoState = {
        "thread_id": thread_id_2,
        "steps_run": [],
        "model_parameter": 1.0,
        "user_override_applied": False,
        "fail_flag": False,
        "iteration": 0
    }

    print("Launching pipeline...")
    app_with_interrupt.invoke(initial_state_2, config=config_2)

    # Inspect current state - it should be paused before Node D
    state_at_interrupt = app_with_interrupt.get_state(config_2)
    print(f"Pipeline status: PAUSED")
    print(f"Next node pending: {state_at_interrupt.next}")
    print(f"Current steps completed: {state_at_interrupt.values.get('steps_run')}")
    print(f"Current model parameter: {state_at_interrupt.values.get('model_parameter')}")

    # Human intervention: modify state value
    print("\n[Human intervention] Reviewing state. Overriding model_parameter to 99.9...")
    app_with_interrupt.update_state(
        config_2, 
        {"model_parameter": 99.9, "user_override_applied": True},
        as_node="node_c" # Specify which node's output we are overriding/extending
    )

    # Resume execution
    print("Resuming execution...")
    final_state_2 = app_with_interrupt.invoke(None, config=config_2)
    print(f"Pipeline finished successfully!")
    print(f"Final steps run list: {final_state_2.get('steps_run')}")
    print(f"Final model parameter committed: {final_state_2.get('model_parameter')}")

    # -------------------------------------------------------------------------
    # BENEFIT 3: TIME TRAVEL
    # -------------------------------------------------------------------------
    print("\n>>> 3. DEMONSTRATING TIME TRAVEL (FORKING / REPLAY)")
    print("-" * 50)
    print("Goal: Go back in history to a past checkpoint, fork the state, and run an alternative scenario.")

    # We will use the history of our successful thread_id_2 run
    history = list(app_with_interrupt.get_state_history(config_2))
    print(f"Retrieving execution history. Total checkpoints found: {len(history)}")
    
    for i, checkpoint in enumerate(history):
        print(f"  Checkpoint {i}: Next node={checkpoint.next} | Steps Run={checkpoint.values.get('steps_run')} | Param={checkpoint.values.get('model_parameter')}")

    # Fork the thread from the state *before* Node D was executed (checkpoint 1 in history, next='node_d')
    # Let's find the checkpoint that is paused before 'node_d'
    target_checkpoint = None
    for cp in history:
        if cp.next == ("node_d",):
            target_checkpoint = cp
            break

    if target_checkpoint:
        print(f"\nTime Traveling back to checkpoint when next node was: {target_checkpoint.next}")
        # Retrieve the checkpoint's unique config containing the checkpoint_id
        fork_config = target_checkpoint.config
        
        # Human/agent updates the state of this past checkpoint
        print("Forking state: Setting parameter to -500.0 instead of 99.9...")
        forked_config = app_with_interrupt.update_state(
            fork_config,
            {"model_parameter": -500.0, "user_override_applied": True}
        )
        
        # Resume the parallel branch from this checkpoint
        print("Executing alternative branch from past checkpoint...")
        forked_result = app_with_interrupt.invoke(None, config=forked_config)
        print(f"Alternative branch finished successfully!")
        print(f"Alternative model parameter committed: {forked_result.get('model_parameter')}")
        
        # Verify the original thread remains unaffected
        original_final_state = app_with_interrupt.get_state(config_2)
        print(f"Original thread parameters remain unchanged: Param={original_final_state.values.get('model_parameter')}")

    # -------------------------------------------------------------------------
    # BENEFIT 4: SHORT TERM MEMORY (SESSION RETENTION)
    # -------------------------------------------------------------------------
    print("\n>>> 4. DEMONSTRATING SHORT-TERM MEMORY")
    print("-" * 50)
    print("Goal: Maintain conversational/run history across independent invocations of the same thread.")
    
    app_memory = build_demo_graph()
    thread_id_4 = str(uuid.uuid4())
    config_4 = {"configurable": {"thread_id": thread_id_4}}

    # Call 1: Run the pipeline
    print("First run on thread...")
    res_1 = app_memory.invoke({
        "thread_id": thread_id_4,
        "steps_run": [],
        "model_parameter": 1.5,
        "user_override_applied": False,
        "fail_flag": False,
        "iteration": 0
    }, config=config_4)
    print(f"Completed run. Steps: {res_1.get('steps_run')} | Iteration={res_1.get('iteration')}")

    # Call 2: Increment run on the SAME thread
    print("\nSecond run on SAME thread (sending only updated iteration parameter)...")
    res_2 = app_memory.invoke({
        "iteration": 1  # We don't resend steps_run or thread_id; the thread remembers!
    }, config=config_4)
    print(f"Completed second run. Steps: {res_2.get('steps_run')} | Iteration={res_2.get('iteration')}")
    print("Observation: The thread retained 'steps_run' from the first execution automatically!")

    print("\n" + "="*80)
    print("  ALL DEMOS COMPLETED SUCCESSFULLY")
    print("="*80 + "\n")


if __name__ == "__main__":
    run_persistence_demos()

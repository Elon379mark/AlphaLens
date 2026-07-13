"""
AlphaLens Agentic Patterns Module
---------------------------------
Implements the 5 core design patterns for building effective LLM agents:
1. Routing: Dynamic classification of query types.
2. Prompt Chaining: Step-by-step sequential reasoning.
3. Parallelization: Concurrent LLM processing.
4. Orchestrator-Workers: Decentralized delegation & synthesis.
5. Evaluator-Optimizer: Critique-refinement quality loops.
"""
import json
import logging
import concurrent.futures
from typing import Dict, Any, List, Literal
from langchain_core.messages import HumanMessage, SystemMessage
from alphalens.contracts.schemas import HypothesisSchema, PredictedDirection
from alphalens.agents.nodes import _get_llm

logger = logging.getLogger(__name__)


# ===========================================================================
# 1. ROUTING PATTERN
# ===========================================================================
class QueryRouter:
    """Classifies incoming query to route to the correct context focus."""
    
    @staticmethod
    def route_query(query: str) -> Literal["macroeconomics", "microstructure", "corporate_finance"]:
        llm = _get_llm(temperature=0.0)
        system_prompt = (
            "You are a routing agent in a quantitative finance system.\n"
            "Classify the query into exactly one of these categories:\n"
            "- 'macroeconomics': Involves rates, inflation, credit spreads, business cycles.\n"
            "- 'microstructure': Involves order flow, liquidity, volume, order books.\n"
            "- 'corporate_finance': Involves fundamentals, earnings, accounting metrics.\n\n"
            "Respond with ONLY the category name, no other text."
        )
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Query: {query}")
        ])
        category = response.content.strip().lower()
        if category in ["macroeconomics", "microstructure", "corporate_finance"]:
            return category
        return "macroeconomics"  # Default fallback


# ===========================================================================
# 2. PROMPT CHAINING PATTERN
# ===========================================================================
class PromptChainExtractor:
    """Extracts raw relationships and chains it to format a clean hypothesis."""

    @staticmethod
    def run_chain(query: str, context: str) -> Dict[str, Any]:
        llm = _get_llm(temperature=0.2)
        
        # Step 1: Raw extraction of variables and theoretical linkage
        step1_prompt = (
            "Read the academic literature context and extract key return predictors "
            "and their relationships to target asset classes.\n"
            "List the main predictor variable, target asset class, expected direction, "
            "and the exact economic mechanism described in the text."
        )
        response1 = llm.invoke([
            SystemMessage(content=step1_prompt),
            HumanMessage(content=f"CONTEXT:\n{context}\n\nQUERY: {query}")
        ])
        raw_extraction = response1.content.strip()
        logger.info("[Prompt Chain] Step 1 raw extraction completed.")

        # Step 2: Format raw extraction into the structured JSON schema
        step2_prompt = (
            "Convert the provided raw extraction details into exactly this JSON format:\n"
            "{\n"
            '  "predictor_variable": "string (the variable name, e.g. credit_spread_slope)",\n'
            '  "target_asset_class": "string (e.g. US_HY_bonds, US_equities)",\n'
            '  "predicted_direction": "positive" or "negative",\n'
            '  "confidence": float between 0.0 and 1.0,\n'
            '  "theoretical_mechanism": "string explaining the causal economic reasoning"\n'
            "}\n"
            "Respond ONLY with valid JSON, no explanations, no markdown fences."
        )
        response2 = llm.invoke([
            SystemMessage(content=step2_prompt),
            HumanMessage(content=f"RAW DETAILS:\n{raw_extraction}")
        ])
        
        raw_json = response2.content.strip()
        if raw_json.startswith("```"):
            raw_json = raw_json.split("```")[1]
            if raw_json.startswith("json"):
                raw_json = raw_json[4:]
        
        return json.loads(raw_json)


# ===========================================================================
# 3. PARALLELIZATION PATTERN
# ===========================================================================
class ParallelContextAnalyzer:
    """Analyzes multiple document chunks concurrently in parallel."""

    @staticmethod
    def _analyze_chunk(chunk: str, query: str) -> str:
        llm = _get_llm(temperature=0.2)
        prompt = (
            "Summarize any specific quantitative trading signal or hypothesis "
            "found in this text chunk that relates to the query. If none is found, return 'No signals'."
        )
        response = llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=f"CHUNK:\n{chunk}\n\nQUERY: {query}")
        ])
        return response.content.strip()

    @classmethod
    def analyze_parallel(cls, chunks: List[str], query: str) -> str:
        # Limit to top 3 chunks to prevent rate limit issues
        active_chunks = chunks[:3]
        summaries = []
        
        logger.info(f"[Parallelization] Spawning {len(active_chunks)} parallel chunk analysis workers...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(active_chunks)) as executor:
            futures = {executor.submit(cls._analyze_chunk, ch, query): i for i, ch in enumerate(active_chunks)}
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                try:
                    summary = future.result()
                    if summary.lower() != "no signals":
                        summaries.append(f"Chunk {idx} Summary:\n{summary}")
                except Exception as e:
                    logger.error(f"[Parallelization] Worker {idx} failed: {e}")
        
        return "\n\n---\n\n".join(summaries) if summaries else "No relevant signals found across chunks."


# ===========================================================================
# 4. ORCHESTRATOR-WORKERS PATTERN
# ===========================================================================
class OrchestratorWorkers:
    """Delegates a complex query to specialized sub-workers and synthesizes reports."""

    @staticmethod
    def _worker_task(worker_name: str, query: str, context: str) -> str:
        llm = _get_llm(temperature=0.3)
        prompt = (
            f"You are a specialized worker: {worker_name}.\n"
            "Analyze the context and extract any relevant relationship with respect to the query.\n"
            "Provide your findings in 2 concise sentences."
        )
        response = llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=f"CONTEXT:\n{context}\n\nQUERY: {query}")
        ])
        return f"[{worker_name} Report]: {response.content.strip()}"

    @classmethod
    def delegate_and_synthesize(cls, query: str, context: str) -> str:
        workers = ["MacroEconomic Specialist", "MicroStructure Specialist", "Statistical Analyst"]
        reports = []

        logger.info("[Orchestrator-Workers] Orchestrator partitioning tasks to workers...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(workers)) as executor:
            futures = {executor.submit(cls._worker_task, w, query, context): w for w in workers}
            for future in concurrent.futures.as_completed(futures):
                w_name = futures[future]
                try:
                    report = future.result()
                    reports.append(report)
                except Exception as e:
                    logger.error(f"[Orchestrator-Workers] Worker '{w_name}' failed: {e}")

        # Synthesize reports
        llm = _get_llm(temperature=0.2)
        synthesis_prompt = (
            "You are the Orchestrator. Synthesize the provided sub-worker reports into "
            "a single unified consensus finding relating to the research query."
        )
        synthesis = llm.invoke([
            SystemMessage(content=synthesis_prompt),
            HumanMessage(content=f"REPORTS:\n" + "\n".join(reports))
        ])
        return synthesis.content.strip()


# ===========================================================================
# 5. EVALUATOR-OPTIMIZER PATTERN
# ===========================================================================
class EvaluatorOptimizer:
    """Runs a quality assurance loop between a generator and an evaluator."""

    @staticmethod
    def evaluate(hypothesis: Dict[str, Any]) -> Dict[str, Any]:
        llm = _get_llm(temperature=0.0)
        prompt = (
            "You are the Evaluator. Critique this quantitative trading hypothesis.\n"
            "Check for:\n"
            "1. Variable clarity: Is the predictor clearly defined?\n"
            "2. Potential look-ahead bias.\n"
            "3. Theoretical mechanism validity.\n\n"
            "Respond ONLY with a JSON matching this format:\n"
            "{\n"
            '  "approved": true or false,\n'
            '  "feedback": ["list of improvement points"]\n'
            "}"
        )
        response = llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=f"HYPOTHESIS:\n{json.dumps(hypothesis)}")
        ])
        raw_json = response.content.strip()
        if raw_json.startswith("```"):
            raw_json = raw_json.split("```")[1]
            if raw_json.startswith("json"):
                raw_json = raw_json[4:]
        return json.loads(raw_json)

    @staticmethod
    def refine(hypothesis: Dict[str, Any], feedback: List[str]) -> Dict[str, Any]:
        llm = _get_llm(temperature=0.4)
        prompt = (
            "You are the Generator/Optimizer. Refine this hypothesis to address the "
            "evaluator feedback.\n\n"
            "Respond ONLY with the updated JSON matching the original schema."
        )
        response = llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=f"ORIGINAL:\n{json.dumps(hypothesis)}\n\nFEEDBACK:\n{json.dumps(feedback)}")
        ])
        raw_json = response.content.strip()
        if raw_json.startswith("```"):
            raw_json = raw_json.split("```")[1]
            if raw_json.startswith("json"):
                raw_json = raw_json[4:]
        return json.loads(raw_json)


# ===========================================================================
# HIGH-LEVEL AGENTIC COUPLING
# ===========================================================================
def run_agentic_hypothesis_generation(query: str, raw_context_chunks: List[str]) -> Dict[str, Any]:
    """Combines all 5 Agentic Patterns to output an optimized, peer-reviewed hypothesis."""
    logger.info(f"--- STARTING AGENTIC PIPELINE FOR: '{query}' ---")
    
    # 1. Routing Pattern
    category = QueryRouter.route_query(query)
    logger.info(f"[Pattern 1: Router] Query classified as: {category.upper()}")

    # 2. Parallelization Pattern (Summarize chunks in parallel)
    summarized_context = ParallelContextAnalyzer.analyze_parallel(raw_context_chunks, query)
    logger.info("[Pattern 2: Parallelization] Parallel context extraction completed.")

    # 3. Orchestrator-Workers Pattern (Delegate to specialized experts)
    synthesized_consensus = OrchestratorWorkers.delegate_and_synthesize(query, summarized_context)
    logger.info("[Pattern 3: Orchestrator-Workers] Consolidated consensus synthesis completed.")

    # 4. Prompt Chaining Pattern (Chained structured formulation)
    initial_hypothesis = PromptChainExtractor.run_chain(query, synthesized_consensus)
    logger.info(f"[Pattern 4: Prompt Chaining] Initial structured hypothesis formulated: {initial_hypothesis.get('predictor_variable')}")

    # 5. Evaluator-Optimizer Pattern (Critique and loop refinement)
    max_loops = 3
    current_hyp = initial_hypothesis
    for loop in range(max_loops):
        eval_result = EvaluatorOptimizer.evaluate(current_hyp)
        approved = eval_result.get("approved", False)
        feedback = eval_result.get("feedback", [])
        
        logger.info(f"[Pattern 5: Evaluator-Optimizer] Loop {loop+1}/{max_loops} | Approved: {approved}")
        if approved:
            break
            
        logger.info(f"[Pattern 5: Evaluator-Optimizer] Critique received: {feedback}")
        current_hyp = EvaluatorOptimizer.refine(current_hyp, feedback)

    logger.info("--- AGENTIC PIPELINE COMPLETED ---")
    return current_hyp

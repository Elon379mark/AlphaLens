import json
import uuid
import math
import hashlib
from typing import List, Dict, Any, Optional
from alphalens.contracts.schemas import HypothesisSchema, PredictedDirection

try:
    import numpy as np
except ImportError:
    np = None

class RAGPipeline:
    def __init__(self, embedding_dim: int = 3072):
        self.embedding_dim = embedding_dim
        self.documents: List[Dict[str, Any]] = []
        # In production, we would use faiss.IndexFlatL2(embedding_dim) or ChromaDB
        self.embeddings_matrix: Optional[Any] = None

    def add_documents(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None):
        """
        Chunks texts at the paragraph level, embeds them, and adds them to the index.
        """
        for doc_idx, text in enumerate(texts):
            # Split by double newline or single newline if paragraphs are long
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            for p_idx, para in enumerate(paragraphs):
                meta = metadatas[doc_idx] if metadatas else {}
                chunk_id = f"doc_{doc_idx}_para_{p_idx}"
                # Generate embedding (simulated/mocked or real)
                embedding = self._get_embedding(para)
                self.documents.append({
                    "id": chunk_id,
                    "text": para,
                    "metadata": meta,
                    "embedding": embedding
                })
        
        # Build embeddings matrix
        if np is not None and self.documents:
            self.embeddings_matrix = np.array([doc["embedding"] for doc in self.documents], dtype=np.float32)

    def _get_embedding(self, text: str) -> List[float]:
        """
        In production: client.embeddings.create(input=text, model="text-embedding-3-large")
        Here we generate a deterministic pseudo-random embedding vector based on the text hash for testability.
        """
        def _det_hash(t: str) -> int:
            return int(hashlib.md5(t.encode("utf-8")).hexdigest(), 16)

        if np is not None:
            # Deterministic seed using hash
            h = _det_hash(text) % (2**32)
            rng = np.random.default_rng(h)
            vec = rng.normal(size=self.embedding_dim)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            return vec.tolist()
        else:
            # Fallback pure-Python deterministic vector generator
            vec = []
            h = _det_hash(text)
            for i in range(self.embedding_dim):
                h = _det_hash(str(h) + str(i))
                vec.append((h % 1000) / 1000.0)
            norm = math.sqrt(sum(x*x for x in vec))
            if norm > 0:
                vec = [x/norm for x in vec]
            return vec

    def search(self, query: str, k: int = 15) -> List[Dict[str, Any]]:
        """
        L2 distance search: Retrieved(q) = arg min ||phi(q) - phi(d_i)||_2
        """
        if not self.documents:
            return []

        query_emb = self._get_embedding(query)

        if np is not None and self.embeddings_matrix is not None:
            q_vec = np.array(query_emb, dtype=np.float32)
            # L2 distance calculation: sqrt(sum((matrix - q)**2, axis=1))
            dists = np.linalg.norm(self.embeddings_matrix - q_vec, axis=1)
            sorted_indices = np.argsort(dists)[:k]
            results = []
            for idx in sorted_indices:
                results.append({
                    "document": self.documents[idx],
                    "score": float(dists[idx])
                })
            return results
        else:
            # Pure Python fallback L2 search
            dists = []
            for idx, doc in enumerate(self.documents):
                emb = doc["embedding"]
                sq_dist = sum((a - b) ** 2 for a, b in zip(query_emb, emb))
                dist = math.sqrt(sq_dist)
                dists.append((dist, idx))
            dists.sort(key=lambda x: x[0])
            results = []
            for dist, idx in dists[:k]:
                results.append({
                    "document": self.documents[idx],
                    "score": dist
                })
            return results


class LiteratureAgent:
    def __init__(self, rag_pipeline: RAGPipeline, api_key: Optional[str] = None):
        self.rag = rag_pipeline
        self.api_key = api_key
        # Add default academic corpus for simulations
        self._load_default_corpus()

    def _load_default_corpus(self):
        default_papers = [
            "We investigate the predictability of US High Yield corporate bonds (US_HY_bonds) using macroeconomic variables. "
            "Our results demonstrate that a steepening of the credit spread slope (credit_spread_slope) is statistically "
            "associated with subsequent negative returns in US_HY_bonds. The credit_spread_slope captures the term premium of "
            "risk and reflects market distress, driving investors out of credit-sensitive assets.",
            
            "Applying deep learning models to market microstructure variables shows that order flow imbalance (order_flow_imbalance) "
            "exhibits a strong positive relationship with short-horizon equity index returns. "
            "This suggests that liquidity demand shocks cause temporary price pressures that resolve positively in the direction "
            "of the buy-sell pressure imbalance.",
            
            "Alternative supply-chain datasets (supply_chain_disruption_index) show predictive power for commodity-sector equities. "
            "Higher disruption values forecast negative returns for commodity manufacturers due to increased input costs "
            "and delayed deliveries, which compress corporate margins."
        ]
        self.rag.add_documents(default_papers, metadatas=[
            {"source": "arXiv:2301.12345", "topic": "credit"},
            {"source": "arXiv:2204.56789", "topic": "microstructure"},
            {"source": "arXiv:2109.11111", "topic": "supplychain"}
        ])

    def generate_hypothesis(self, query: str) -> HypothesisSchema:
        """
        Retrieves top-k=15 context paragraphs, constructs prompt, and extracts hypothesis.
        Uses a mock/deterministic extractor if Anthropic/OpenAI keys are not provided.
        """
        results = self.rag.search(query, k=15)
        context_texts = [r["document"]["text"] for r in results]
        sources = list(set(r["document"]["metadata"].get("source", "Unknown") for r in results))

        # Baseline System Prompt (Contract 1 compliance)
        system_prompt = (
            "You are the Literature Agent in the AlphaLens platform. "
            "Your role is autonomous scientific hypothesis generation. "
            "Always reason step-by-step before producing output. "
            "Output must conform to JSON schema: HypothesisSchema.\n"
            "CONSTRAINTS:\n"
            "- Do not introduce look-ahead bias.\n"
            "- Cite evidence for all statistical claims.\n"
            "- Flag low-confidence outputs with uncertainty_flag=true."
        )

        user_content = f"CONTEXT:\n" + "\n---\n".join(context_texts) + f"\n\nTASK: Generate a quantitative trading hypothesis based on the context above for query: {query}"

        # If API key is available, run production call
        if self.api_key:
            try:
                from langchain_groq import ChatGroq
                from langchain_core.messages import SystemMessage, HumanMessage
                llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3, groq_api_key=self.api_key)
                response = llm.invoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_content),
                ])
                import json
                raw_text = response.content.strip()
                if raw_text.startswith("```"):
                    raw_text = raw_text.split("```")[1]
                    if raw_text.startswith("json"):
                        raw_text = raw_text[4:]
                parsed = json.loads(raw_text)
                return HypothesisSchema(
                    hypothesis_id=f"H-{uuid.uuid4().hex[:6].upper()}",
                    predictor_variable=parsed.get("predictor_variable", "unknown"),
                    target_asset_class=parsed.get("target_asset_class", "unknown"),
                    predicted_direction=PredictedDirection(parsed.get("predicted_direction", "positive")),
                    confidence=float(parsed.get("confidence", 0.5)),
                    theoretical_mechanism=parsed.get("theoretical_mechanism", ""),
                    source_references=sources[:3]
                )
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"RAG production LLM call failed, using simulation: {e}")

        # Simulated Extract: parses context to generate a realistic hypothesis matching the query
        return self._simulated_llm_call(query, context_texts, sources)

    def _simulated_llm_call(self, query: str, context: List[str], sources: List[str]) -> HypothesisSchema:
        # Match keywords in query/context to produce a tailored, valid mock hypothesis
        query_lower = query.lower()
        
        if "credit" in query_lower or "bond" in query_lower:
            return HypothesisSchema(
                hypothesis_id=f"H-{uuid.uuid4().hex[:6].upper()}",
                predictor_variable="credit_spread_slope",
                target_asset_class="US_HY_bonds",
                predicted_direction=PredictedDirection.NEGATIVE,
                confidence=0.87,
                theoretical_mechanism="A steepening credit spread slope signals rising credit risk premia, leading to capital outflow and price pressure on corporate high-yield bonds.",
                source_references=sources[:2]
            )
        elif "order" in query_lower or "flow" in query_lower or "imbalance" in query_lower:
            return HypothesisSchema(
                hypothesis_id=f"H-{uuid.uuid4().hex[:6].upper()}",
                predictor_variable="order_flow_imbalance",
                target_asset_class="US_equities",
                predicted_direction=PredictedDirection.POSITIVE,
                confidence=0.91,
                theoretical_mechanism="Positive order flow imbalance reflects heavy buying pressure, which leads to immediate positive price returns due to liquidity consumption.",
                source_references=sources[:2]
            )
        else:
            # Default fallback hypothesis
            return HypothesisSchema(
                hypothesis_id=f"H-{uuid.uuid4().hex[:6].upper()}",
                predictor_variable="supply_chain_disruption_index",
                target_asset_class="Commodity_equities",
                predicted_direction=PredictedDirection.NEGATIVE,
                confidence=0.76,
                theoretical_mechanism="Higher supply chain disruptions lead to increased input costs, delay in shipping, and reduced margins for commodity-related producers.",
                source_references=sources[:2]
            )

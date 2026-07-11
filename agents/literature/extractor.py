import json
import os
from typing import List, Dict

from dotenv import load_dotenv
from groq import Groq

from agents.literature.prompts import build_extraction_prompt

load_dotenv()

GROQ_MODEL = "llama-3.3-70b-versatile"


def get_groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Check that .env exists in the AlphaLens "
            "root folder and contains GROQ_API_KEY=your_key"
        )
    return Groq(api_key=api_key)


def extract_facts_from_chunks(chunks: List[Dict], model: str = GROQ_MODEL) -> List[Dict]:
    """
    For each retrieved chunk, call the LLM and extract structured facts.
    Returns a flat list of fact dicts. Failures on individual chunks are
    logged and skipped, not fatal to the whole batch.
    """
    client = get_groq_client()
    all_facts = []

    for chunk in chunks:
        prompt = build_extraction_prompt(chunk["text"])
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=1024,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": prompt["system"]},
                    {"role": "user", "content": prompt["user"]},
                ],
            )
            raw_text = response.choices[0].message.content.strip()

            # Defensive cleanup in case the model wraps output in code fences anyway
            if raw_text.startswith("```"):
                raw_text = raw_text.strip("`")
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:].strip()

            facts = json.loads(raw_text)
            if isinstance(facts, list):
                for fact in facts:
                    fact["source_chunk_id"] = chunk["chunk_id"]
                all_facts.extend(facts)

        except json.JSONDecodeError as e:
            print(f"[WARN] JSON parse failed for {chunk['chunk_id']}: {e}")
        except Exception as e:
            print(f"[WARN] Extraction failed for {chunk['chunk_id']}: {e}")

    return all_facts


def save_facts(facts: List[Dict], path: str = "outputs/literature_facts.json") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(facts, f, indent=2)


if __name__ == "__main__":
    from agents.literature.vector_store import get_chroma_client, get_or_create_collection
    from agents.literature.embedder import load_embedding_model, embed_query
    from agents.literature.retriever import retrieve, rerank

    print("Connecting to ChromaDB...")
    client_db = get_chroma_client()
    collection = get_or_create_collection(client_db)
    print(f"Collection has {collection.count()} vectors.")

    print("Loading embedding model...")
    embed_model = load_embedding_model()

    test_query = "momentum signal predictive power"
    query_vec = embed_query(test_query, embed_model)

    print(f"\nRetrieving + reranking for: '{test_query}'...")
    candidates = retrieve(collection, query_vec, n_results=20)
    reranked = rerank(test_query, candidates, top_k=3)  # small batch for first test
    print(f"Selected {len(reranked)} chunks for extraction.")

    print("\nCalling Groq API to extract structured facts...")
    facts = extract_facts_from_chunks(reranked)

    print(f"\nExtracted {len(facts)} facts.")
    for i, fact in enumerate(facts):
        print(f"\n--- Fact {i+1} ---")
        print(json.dumps(fact, indent=2))

    save_facts(facts)
    print(f"\nSaved to outputs/literature_facts.json")
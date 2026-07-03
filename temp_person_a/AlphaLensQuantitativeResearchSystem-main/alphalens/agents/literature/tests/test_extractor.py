"""
test_extractor.py
Tests for JSON fact extractor.
Success criteria:
  - Returns valid JSON list for >= 80% of chunks
  - Each fact contains required schema keys
  - source_chunk_id is attached to every fact
"""

import json
import os
import pytest
from unittest.mock import patch, MagicMock

from agents.literature.extractor import extract_facts_from_chunks, save_facts
from agents.literature.prompts import build_extraction_prompt


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_chunks(n=3):
    return [
        {
            "chunk_id": f"paper.pdf_chunk_{i:04d}",
            "text": (
                f"We document a momentum signal with 12-month formation period. "
                f"The strategy achieves an IC of 0.04 and Sharpe of 1.{i}."
            ),
        }
        for i in range(n)
    ]


def _valid_fact_response():
    facts = [
        {
            "signal_name": "12-month momentum",
            "description": "Price momentum over 12-month formation period",
            "asset_class": "equity",
            "holding_period": "monthly",
            "formation_period": "12 months",
            "ic_reported": 0.04,
            "sharpe_reported": 1.2,
            "return_reported": None,
            "paper_title": "Momentum Investing",
            "year": 2023,
        }
    ]
    return json.dumps(facts)


def _make_mock_groq_response(text):
    """Build a mock that mimics groq client.chat.completions.create() response."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = text
    return mock_response


# ── Prompt Tests ──────────────────────────────────────────────────────────────

class TestBuildExtractionPrompt:
    def test_returns_dict_with_system_and_user(self):
        p = build_extraction_prompt("Some research text about momentum.")
        assert "system" in p
        assert "user" in p

    def test_context_injected_into_user_prompt(self):
        context = "momentum signal with IC = 0.05"
        p = build_extraction_prompt(context)
        assert context in p["user"]

    def test_system_prompt_not_empty(self):
        p = build_extraction_prompt("text")
        assert len(p["system"]) > 10


# ── Extractor Tests ───────────────────────────────────────────────────────────

class TestExtractFactsFromChunks:

    @patch("agents.literature.extractor.Groq")
    def test_returns_list(self, mock_groq_cls):
        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = \
            _make_mock_groq_response(_valid_fact_response())

        result = extract_facts_from_chunks(_make_chunks(2))
        assert isinstance(result, list)

    @patch("agents.literature.extractor.Groq")
    def test_source_chunk_id_attached(self, mock_groq_cls):
        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = \
            _make_mock_groq_response(_valid_fact_response())

        result = extract_facts_from_chunks(_make_chunks(1))
        assert len(result) >= 1
        for fact in result:
            assert "source_chunk_id" in fact

    @patch("agents.literature.extractor.Groq")
    def test_invalid_json_skipped_gracefully(self, mock_groq_cls):
        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client

        # First chunk returns invalid JSON, second valid
        mock_client.chat.completions.create.side_effect = [
            _make_mock_groq_response("Not valid JSON at all!!!"),
            _make_mock_groq_response(_valid_fact_response()),
        ]

        result = extract_facts_from_chunks(_make_chunks(2))
        assert isinstance(result, list)
        assert len(result) >= 1

    @patch("agents.literature.extractor.Groq")
    def test_fact_has_signal_name_key(self, mock_groq_cls):
        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = \
            _make_mock_groq_response(_valid_fact_response())

        result = extract_facts_from_chunks(_make_chunks(1))
        for fact in result:
            assert "signal_name" in fact

    @patch("agents.literature.extractor.Groq")
    def test_markdown_fence_stripped(self, mock_groq_cls):
        """LLM occasionally wraps JSON in ```json ... ``` — must be stripped."""
        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client
        fenced = f"```json\n{_valid_fact_response()}\n```"
        mock_client.chat.completions.create.return_value = \
            _make_mock_groq_response(fenced)

        result = extract_facts_from_chunks(_make_chunks(1))
        assert len(result) >= 1

    @patch("agents.literature.extractor.Groq")
    def test_empty_chunk_list_returns_empty(self, mock_groq_cls):
        result = extract_facts_from_chunks([])
        assert result == []


class TestSaveFacts:
    def test_file_created(self, tmp_path):
        facts = [{"signal_name": "momentum", "source_chunk_id": "x_0000"}]
        path = str(tmp_path / "facts.json")
        save_facts(facts, path=path)
        assert os.path.exists(path)

    def test_file_is_valid_json(self, tmp_path):
        facts = [{"signal_name": "value", "source_chunk_id": "x_0001"}]
        path = str(tmp_path / "facts.json")
        save_facts(facts, path=path)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded == facts

    def test_empty_facts_saved(self, tmp_path):
        path = str(tmp_path / "empty.json")
        save_facts([], path=path)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded == []

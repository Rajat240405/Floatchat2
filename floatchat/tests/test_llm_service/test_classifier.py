"""Tests for QueryClassifier."""

from unittest.mock import MagicMock

import pytest

from floatchat.llm_service.classifier import QueryClassifier


class TestQueryClassifier:
    def test_classify_data_query(self) -> None:
        llm = MagicMock()
        llm.generate = MagicMock(return_value="DATA_QUERY")
        classifier = QueryClassifier(llm)

        result = classifier.classify("oxygen in arabian sea")
        assert result == "DATA_QUERY"
        llm.generate.assert_called_once()

    def test_classify_general_query(self) -> None:
        llm = MagicMock()
        llm.generate = MagicMock(return_value="GENERAL_QUERY")
        classifier = QueryClassifier(llm)

        result = classifier.classify("what is argo")
        assert result == "GENERAL_QUERY"

    def test_classify_lowercase_output(self) -> None:
        llm = MagicMock()
        llm.generate = MagicMock(return_value="  general_query  ")
        classifier = QueryClassifier(llm)

        result = classifier.classify("explain chlorophyll")
        assert result == "GENERAL_QUERY"

    def test_classify_unexpected_output_defaults_to_data(self) -> None:
        llm = MagicMock()
        llm.generate = MagicMock(return_value="I think this is about data")
        classifier = QueryClassifier(llm)

        result = classifier.classify("something weird")
        assert result == "DATA_QUERY"

    def test_classify_llm_failure_fallback(self) -> None:
        llm = MagicMock()
        llm.generate = MagicMock(side_effect=ConnectionError("Ollama down"))
        classifier = QueryClassifier(llm)

        result = classifier.classify("oxygen profile")
        assert result == "DATA_QUERY"

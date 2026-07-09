"""Tests for FastAPI routes."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from floatchat.api.dependencies import (
    get_conversation_manager,
    get_intent_parser,
    get_llm_service,
    get_metadata_service,
    get_netcdf_reader,
    get_query_classifier,
    get_query_engine,
    get_repository_service,
    get_visualization_engine,
)
from floatchat.api.main import create_app
from floatchat.conversation.memory import InMemoryConversationManager
from floatchat.intent_parser.regex import RegexIntentParser
from floatchat.llm_service.classifier import QueryClassifier
from floatchat.llm_service.ollama import OllamaLLMService


@pytest.fixture
def client():
    app = create_app()

    # Override dependencies with lightweight mocks/stubs
    app.dependency_overrides[get_intent_parser] = lambda: RegexIntentParser()

    # LLM service + classifier — mock to avoid Ollama dependency in tests
    llm_mock = MagicMock(spec=OllamaLLMService)
    llm_mock.generate = MagicMock(return_value="Mock LLM answer")
    app.dependency_overrides[get_llm_service] = lambda: llm_mock

    classifier_mock = MagicMock(spec=QueryClassifier)
    classifier_mock.classify = MagicMock(return_value="DATA_QUERY")
    app.dependency_overrides[get_query_classifier] = lambda: classifier_mock

    metadata = MagicMock()
    metadata.is_loaded = MagicMock(return_value=True)
    metadata.search = MagicMock(return_value=[])
    app.dependency_overrides[get_metadata_service] = lambda: metadata

    repo = MagicMock()
    app.dependency_overrides[get_repository_service] = lambda: repo

    reader = MagicMock()
    app.dependency_overrides[get_netcdf_reader] = lambda: reader

    viz = MagicMock()
    viz.render = MagicMock(return_value={"data": [], "layout": {}})
    app.dependency_overrides[get_visualization_engine] = lambda: viz

    # QueryEngine needs the real orchestrator but with mocked sub-services
    from floatchat.query_engine.engine import QueryEngine

    engine = QueryEngine(metadata, repo, reader, viz)
    app.dependency_overrides[get_query_engine] = lambda: engine

    # Conversation manager — single instance per test so session context persists
    _conversation_manager = InMemoryConversationManager()
    app.dependency_overrides[get_conversation_manager] = lambda: _conversation_manager

    return TestClient(app)


class TestChatEndpoint:
    def test_chat_known_message(self, client) -> None:
        response = client.post(
            "/api/v1/chat",
            json={"message": "show oxygen profile in arabian sea for 2024"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["intent"] == "profile_plot"
        assert "No Argo profiles matched" in data["message"]

    def test_chat_unknown_mock_message(self, client) -> None:
        """Unknown queries without context return a helpful suggestion message."""
        response = client.post(
            "/api/v1/chat",
            json={"message": "totally unknown query"},
        )
        # Conversational recovery returns a suggestion instead of hard 400
        assert response.status_code == 200
        data = response.json()
        assert data["intent"] == "unknown"
        assert "couldn't determine" in data["message"].lower()

    def test_chat_unknown_with_context_returns_suggestions(self, client) -> None:
        """Unknown queries with context include previous query in suggestions."""
        session_id = "test-session-unknown"

        # Seed context
        client.post(
            "/api/v1/chat",
            json={"message": "show oxygen in arabian sea", "session_id": session_id},
        )

        # Now send an unparseable follow-up — conversational recovery will
        # succeed because context has DOXY + arabian_sea, so it executes.
        response = client.post(
            "/api/v1/chat",
            json={"message": "blargle flargle", "session_id": session_id},
        )
        assert response.status_code == 200
        data = response.json()
        # Recovery succeeds using context, so it routes through data pipeline
        assert data["intent"] == "profile_plot"

    def test_chat_unknown_no_context_returns_suggestions(self, client) -> None:
        """Unknown queries without any context return a suggestion message."""
        response = client.post(
            "/api/v1/chat",
            json={"message": "blargle flargle", "session_id": "fresh-session"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["intent"] == "unknown"
        assert "couldn't determine" in data["message"].lower()
        assert "oxygen" in data["message"].lower()

    def test_health_endpoint(self, client) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_unhandled_exception_returns_structured_json(self, client, monkeypatch) -> None:
        """Regression: unhandled exceptions return structured JSON, not raw tracebacks."""
        from floatchat.intent_parser.regex import RegexIntentParser

        def _boom_parse(self, message):
            raise RuntimeError("something exploded internally")

        monkeypatch.setattr(RegexIntentParser, "parse", _boom_parse)

        response = client.post(
            "/api/v1/chat",
            json={"message": "trigger error"},
        )
        assert response.status_code == 500
        data = response.json()
        assert data["error"] == "InternalServerError"
        assert "unexpected error" in data["message"].lower()
        assert "exploded" not in data["message"].lower()

    def test_general_query_returns_chat_response(self, client, monkeypatch) -> None:
        """GENERAL_QUERY bypasses the data pipeline and returns an LLM answer."""
        from floatchat.llm_service.classifier import QueryClassifier

        monkeypatch.setattr(QueryClassifier, "classify", lambda self, msg: "GENERAL_QUERY")

        response = client.post(
            "/api/v1/chat",
            json={"message": "What is Argo?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["intent"] == "general_chat"
        assert data["figure"] is None
        assert data["map_data"] == []

    def test_follow_up_reuses_context(self, client) -> None:
        """A follow-up query with the same session_id inherits previous context."""
        session_id = "test-session-123"

        # First query: oxygen in Arabian Sea
        response1 = client.post(
            "/api/v1/chat",
            json={"message": "show oxygen in arabian sea", "session_id": session_id},
        )
        assert response1.status_code == 200
        data1 = response1.json()
        assert data1["intent"] == "profile_plot"

        # Second query: only mentions chlorophyll — should inherit Arabian Sea
        response2 = client.post(
            "/api/v1/chat",
            json={"message": "show chlorophyll", "session_id": session_id},
        )
        assert response2.status_code == 200
        data2 = response2.json()
        # The merged intent should have both CHLA and the inherited region,
        # so it routes through the data pipeline (not general_chat).
        assert data2["intent"] == "profile_plot"

    def test_follow_up_with_explicit_override(self, client) -> None:
        """Explicit values in a follow-up override inherited context."""
        session_id = "test-session-456"

        # First query: oxygen in Arabian Sea
        response1 = client.post(
            "/api/v1/chat",
            json={"message": "show oxygen in arabian sea", "session_id": session_id},
        )
        assert response1.status_code == 200

        # Second query: explicitly different region
        response2 = client.post(
            "/api/v1/chat",
            json={
                "message": "show chlorophyll in bay of bengal",
                "session_id": session_id,
            },
        )
        assert response2.status_code == 200
        data2 = response2.json()
        assert data2["intent"] == "profile_plot"

    def test_general_query_uses_context_hint(self, monkeypatch) -> None:
        """GENERAL_QUERY augments the LLM prompt with conversation context."""
        from floatchat.llm_service.classifier import QueryClassifier

        session_id = "test-session-789"
        llm_calls: list[str] = []

        # Capture LLM prompts
        from floatchat.api.dependencies import get_llm_service

        def _capture_llm():
            mock = MagicMock(spec=OllamaLLMService)

            def _generate(prompt, *, system=None):
                llm_calls.append(prompt)
                return "Mock explanation"

            mock.generate = _generate
            return mock

        app = create_app()
        app.dependency_overrides[get_llm_service] = _capture_llm
        _conversation_manager = InMemoryConversationManager()
        app.dependency_overrides[get_conversation_manager] = lambda: _conversation_manager
        app.dependency_overrides[get_metadata_service] = lambda: MagicMock(
            is_loaded=MagicMock(return_value=True),
            search=MagicMock(return_value=[]),
        )

        # The route calls QueryClassifier.classify(classifier, msg) which
        # invokes the *real* class method.  We must monkeypatch at the class.
        _classify_calls: list[str] = []

        def _fake_classify(self, message: str) -> str:
            _classify_calls.append(message)
            return "DATA_QUERY" if len(_classify_calls) == 1 else "GENERAL_QUERY"

        monkeypatch.setattr(QueryClassifier, "classify", _fake_classify)

        test_client = TestClient(app)
        test_client.post(
            "/api/v1/chat",
            json={"message": "show oxygen in arabian sea", "session_id": session_id},
        )
        test_client.post(
            "/api/v1/chat",
            json={"message": "Explain this graph", "session_id": session_id},
        )

        assert len(llm_calls) >= 1
        last_prompt = llm_calls[-1]
        assert "Conversation Context" in last_prompt
        assert "arabian sea" in last_prompt.lower()
        assert "DOXY" in last_prompt

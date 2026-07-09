"""Tests for OllamaLLMService."""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from floatchat.exceptions import FloatChatError
from floatchat.llm_service.ollama import OllamaLLMService


class TestOllamaLLMService:
    def test_generate_success(self) -> None:
        svc = OllamaLLMService(base_url="http://test", model="test-model", timeout=5.0)
        mock_response = MagicMock()
        mock_response.json = MagicMock(return_value={"response": "  Hello world  "})
        mock_response.raise_for_status = MagicMock()

        with patch.object(svc._client, "post", return_value=mock_response):
            result = svc.generate("Say hello")

        assert result == "Hello world"
        mock_response.raise_for_status.assert_called_once()

    def test_generate_with_system_prompt(self) -> None:
        svc = OllamaLLMService(base_url="http://test", model="test-model", timeout=5.0)
        mock_response = MagicMock()
        mock_response.json = MagicMock(return_value={"response": "Answer"})
        mock_response.raise_for_status = MagicMock()

        with patch.object(svc._client, "post", return_value=mock_response) as mock_post:
            svc.generate("Question", system="You are a test bot")

        call_args = mock_post.call_args
        assert call_args[1]["json"]["system"] == "You are a test bot"

    def test_generate_http_error_raises(self) -> None:
        svc = OllamaLLMService(base_url="http://test", model="test-model", timeout=5.0)
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        with patch.object(
            svc._client, "post", side_effect=httpx.HTTPStatusError(
                "500", request=MagicMock(), response=mock_response
            )
        ):
            with pytest.raises(FloatChatError) as exc_info:
                svc.generate("test")

        assert "Ollama returned HTTP" in str(exc_info.value.message)

    def test_generate_connection_error_raises(self) -> None:
        svc = OllamaLLMService(base_url="http://test", model="test-model", timeout=5.0)

        with patch.object(
            svc._client, "post", side_effect=httpx.RequestError("Connection refused")
        ):
            with pytest.raises(FloatChatError) as exc_info:
                svc.generate("test")

        assert "Cannot connect to Ollama" in str(exc_info.value.message)

import pytest
from unittest.mock import patch, MagicMock
import os
from datetime import datetime

from llm_critique.core.models import LLMClient


@pytest.fixture
def mock_openai():
    """Mock OpenAI client."""
    with patch("llm_critique.core.models.OpenAI") as mock:
        mock.return_value.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="OpenAI response"))],
            usage=MagicMock(total_tokens=100)
        )
        yield mock


@pytest.fixture
def mock_anthropic():
    """Mock Anthropic client."""
    with patch("llm_critique.core.models.Anthropic") as mock:
        mock.return_value.messages.create.return_value = MagicMock(
            content=[MagicMock(text="Anthropic response")],
            usage=MagicMock(input_tokens=50, output_tokens=50)
        )
        yield mock


@pytest.fixture
def mock_google():
    """Mock Google client."""
    with patch("llm_critique.core.models.GenerativeModel") as mock:
        mock.return_value.generate_content.return_value = MagicMock(
            text="Google response",
            prompt_feedback=MagicMock(token_count=100)
        )
        yield mock


@pytest.fixture
def mock_env_vars():
    """Mock environment variables."""
    with patch.dict(os.environ, {
        "OPENAI_API_KEY": "test-openai-key",
        "ANTHROPIC_API_KEY": "test-anthropic-key",
        "GOOGLE_API_KEY": "test-google-key"
    }):
        yield


def test_llm_client_initialization(mock_env_vars):
    """Test LLM client initialization with API keys."""
    client = LLMClient()
    
    # Verify client initialization
    assert client.openai_client is not None
    assert client.anthropic_client is not None
    assert client.google_client is not None


def test_llm_client_missing_keys():
    """Test LLM client initialization with missing API keys."""
    with patch.dict(os.environ, {}, clear=True):
        client = LLMClient()
        
        # Verify client initialization without keys
        assert client.openai_client is None
        assert client.anthropic_client is None
        assert client.google_client is None


def test_execute_model_openai(mock_env_vars, mock_openai):
    """Test executing OpenAI model."""
    client = LLMClient()
    
    # Execute model
    result = client.execute_model(
        model="gpt-4",
        prompt="Test prompt",
        timeout=30
    )
    
    # Verify result
    assert result["response"] == "OpenAI response"
    assert result["tokens"] == 100
    assert "latency" in result
    assert isinstance(result["latency"], float)


def test_execute_model_anthropic(mock_env_vars, mock_anthropic):
    """Test executing Anthropic model."""
    client = LLMClient()
    
    # Execute model
    result = client.execute_model(
        model="claude-3-opus",
        prompt="Test prompt",
        timeout=30
    )
    
    # Verify result
    assert result["response"] == "Anthropic response"
    assert result["tokens"] == 100  # input + output tokens
    assert "latency" in result
    assert isinstance(result["latency"], float)


def test_execute_model_google(mock_env_vars, mock_google):
    """Test executing Google model."""
    client = LLMClient()
    
    # Execute model
    result = client.execute_model(
        model="gemini-pro",
        prompt="Test prompt",
        timeout=30
    )
    
    # Verify result
    assert result["response"] == "Google response"
    assert result["tokens"] == 100
    assert "latency" in result
    assert isinstance(result["latency"], float)


def test_execute_model_timeout(mock_env_vars, mock_openai):
    """Test model execution timeout."""
    # Mock timeout
    mock_openai.return_value.chat.completions.create.side_effect = TimeoutError("Test timeout")
    
    client = LLMClient()
    
    # Execute model
    result = client.execute_model(
        model="gpt-4",
        prompt="Test prompt",
        timeout=1
    )
    
    # Verify timeout handling
    assert result["response"] is None
    assert "error" in result
    assert "timeout" in result["error"].lower()


def test_execute_model_error(mock_env_vars, mock_openai):
    """Test model execution error handling."""
    # Mock error
    mock_openai.return_value.chat.completions.create.side_effect = Exception("Test error")
    
    client = LLMClient()
    
    # Execute model
    result = client.execute_model(
        model="gpt-4",
        prompt="Test prompt",
        timeout=30
    )
    
    # Verify error handling
    assert result["response"] is None
    assert "error" in result
    assert result["error"] == "Test error"


def test_execute_models_parallel(mock_env_vars, mock_openai, mock_anthropic, mock_google):
    """Test parallel execution of multiple models."""
    client = LLMClient()
    
    # Execute models
    results = client.execute_models_parallel(
        models=["gpt-4", "claude-3-opus", "gemini-pro"],
        prompt="Test prompt",
        timeout=30
    )
    
    # Verify results
    assert len(results) == 3
    assert all(isinstance(r, dict) for r in results)
    assert all("response" in r for r in results)
    assert all("tokens" in r for r in results)
    assert all("latency" in r for r in results)


def test_get_available_models(mock_env_vars):
    """Test getting available models."""
    client = LLMClient()
    
    # Get available models
    models = client.get_available_models()
    
    # Verify models
    assert isinstance(models, list)
    assert len(models) > 0
    assert all(isinstance(m, str) for m in models)


def test_estimate_cost(mock_env_vars):
    """Test cost estimation."""
    client = LLMClient()
    
    # Estimate cost
    cost = client.estimate_cost(
        model="gpt-4",
        prompt_tokens=100,
        completion_tokens=50
    )
    
    # Verify cost
    assert isinstance(cost, float)
    assert cost >= 0


def test_model_metadata(mock_env_vars):
    """Test model metadata retrieval."""
    client = LLMClient()
    
    # Get metadata
    metadata = client.get_model_metadata("gpt-4")
    
    # Verify metadata
    assert isinstance(metadata, dict)
    assert "provider" in metadata
    assert "context_length" in metadata
    assert "cost_per_1k_tokens" in metadata 
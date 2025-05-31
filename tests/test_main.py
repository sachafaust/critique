import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import json
import os

from llm_critique.main import cli
from llm_critique.config import LLMConfig


@pytest.fixture
def mock_config():
    """Mock configuration with test API keys."""
    return LLMConfig(
        openai_api_key="test-openai-key",
        anthropic_api_key="test-anthropic-key",
        google_api_key="test-google-key"
    )


@pytest.fixture
def mock_llm_responses():
    """Mock LLM responses for testing."""
    return {
        "gpt-4": ("GPT-4 response", {"tokens": 100, "latency_ms": 1000}),
        "claude-3-sonnet": ("Claude response", {"tokens": 90, "latency_ms": 800}),
        "gemini-pro": ("Gemini response", {"tokens": 80, "latency_ms": 900})
    }


def test_basic_prompt_execution(mock_config, mock_llm_responses):
    """Test basic prompt execution."""
    with patch("llm_critique.config.load_config", return_value=mock_config), \
         patch("llm_critique.core.models.LLMClient.execute_parallel", return_value=mock_llm_responses), \
         patch("llm_critique.core.chains.CritiqueAnalysisChain.invoke", return_value={
             "critique": {"analysis": "Test analysis", "suggestions": "Test suggestions"},
             "scores": {"consistency_score": 0.9, "completeness_score": 0.8, "accuracy_score": 0.95}
         }), \
         patch("llm_critique.core.chains.SynthesisChain.invoke", return_value={
             "synthesis": {"synthesized_answer": "Test synthesis", "confidence_score": 0.85},
             "confidence": 0.85
         }):
        
        # Run CLI with test prompt
        result = cli.invoke(["What is machine learning?"])
        assert result.exit_code == 0


def test_file_input(mock_config, mock_llm_responses, tmp_path):
    """Test reading prompt from file."""
    # Create test prompt file
    prompt_file = tmp_path / "test_prompt.txt"
    prompt_file.write_text("What is machine learning?")
    
    with patch("llm_critique.config.load_config", return_value=mock_config), \
         patch("llm_critique.core.models.LLMClient.execute_parallel", return_value=mock_llm_responses), \
         patch("llm_critique.core.chains.CritiqueAnalysisChain.invoke", return_value={
             "critique": {"analysis": "Test analysis", "suggestions": "Test suggestions"},
             "scores": {"consistency_score": 0.9, "completeness_score": 0.8, "accuracy_score": 0.95}
         }), \
         patch("llm_critique.core.chains.SynthesisChain.invoke", return_value={
             "synthesis": {"synthesized_answer": "Test synthesis", "confidence_score": 0.85},
             "confidence": 0.85
         }):
        
        # Run CLI with file input
        result = cli.invoke(["-f", str(prompt_file)])
        assert result.exit_code == 0


def test_model_selection(mock_config, mock_llm_responses):
    """Test model selection."""
    with patch("llm_critique.config.load_config", return_value=mock_config), \
         patch("llm_critique.core.models.LLMClient.execute_parallel", return_value=mock_llm_responses), \
         patch("llm_critique.core.chains.CritiqueAnalysisChain.invoke", return_value={
             "critique": {"analysis": "Test analysis", "suggestions": "Test suggestions"},
             "scores": {"consistency_score": 0.9, "completeness_score": 0.8, "accuracy_score": 0.95}
         }), \
         patch("llm_critique.core.chains.SynthesisChain.invoke", return_value={
             "synthesis": {"synthesized_answer": "Test synthesis", "confidence_score": 0.85},
             "confidence": 0.85
         }):
        
        # Run CLI with specific models
        result = cli.invoke([
            "What is machine learning?",
            "--models", "gpt-4,claude-3-sonnet",
            "--resolver", "claude-3-sonnet"
        ])
        assert result.exit_code == 0


def test_json_output_format(mock_config, mock_llm_responses):
    """Test JSON output format."""
    with patch("llm_critique.config.load_config", return_value=mock_config), \
         patch("llm_critique.core.models.LLMClient.execute_parallel", return_value=mock_llm_responses), \
         patch("llm_critique.core.chains.CritiqueAnalysisChain.invoke", return_value={
             "critique": {"analysis": "Test analysis", "suggestions": "Test suggestions"},
             "scores": {"consistency_score": 0.9, "completeness_score": 0.8, "accuracy_score": 0.95}
         }), \
         patch("llm_critique.core.chains.SynthesisChain.invoke", return_value={
             "synthesis": {"synthesized_answer": "Test synthesis", "confidence_score": 0.85},
             "confidence": 0.85
         }):
        
        # Run CLI with JSON output
        result = cli.invoke([
            "What is machine learning?",
            "--format", "json"
        ])
        assert result.exit_code == 0


def test_missing_api_keys():
    """Test handling of missing API keys."""
    with patch("llm_critique.config.load_config", return_value=LLMConfig()):
        # Run CLI without API keys
        result = cli.invoke(["What is machine learning?"])
        assert result.exit_code == 1


def test_invalid_model_names(mock_config):
    """Test handling of invalid model names."""
    with patch("llm_critique.config.load_config", return_value=mock_config):
        # Run CLI with invalid model
        result = cli.invoke([
            "What is machine learning?",
            "--models", "invalid-model"
        ])
        assert result.exit_code == 1


def test_timeout_handling(mock_config):
    """Test handling of model timeouts."""
    with patch("llm_critique.config.load_config", return_value=mock_config), \
         patch("llm_critique.core.models.LLMClient.execute_parallel", side_effect=TimeoutError("Model timeout")):
        
        # Run CLI with timeout
        result = cli.invoke(["What is machine learning?"])
        assert result.exit_code == 1 
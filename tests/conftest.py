import pytest
from pathlib import Path
import os
import json
from unittest.mock import MagicMock

from llm_critique.config import Config as LLMConfig


@pytest.fixture
def test_config():
    """Test configuration."""
    return LLMConfig(
        openai_api_key="test-openai-key",
        anthropic_api_key="test-anthropic-key",
        google_api_key="test-google-key",
        default_models=["gpt-4", "claude-3-opus", "gemini-pro"],
        default_creator="gpt-4",
        max_iterations=3,
        confidence_threshold=0.8,
        timeout_seconds=30,
        log_directory="test_logs"
    )


@pytest.fixture
def test_log_dir(tmp_path):
    """Test log directory."""
    log_dir = tmp_path / "test_logs"
    log_dir.mkdir()
    return log_dir


@pytest.fixture
def test_execution_data():
    """Test execution data."""
    return {
        "execution_id": "test-execution-id",
        "timestamp": "2024-03-20T12:00:00Z",
        "input": {
            "prompt": "Test prompt",
            "models": ["gpt-4", "claude-3-opus"],
            "resolver_model": "gpt-4"
        },
        "results": {
            "model_responses": [
                {
                    "model": "gpt-4",
                    "response": "Test response 1",
                    "tokens": 100,
                    "latency": 1.0
                },
                {
                    "model": "claude-3-opus",
                    "response": "Test response 2",
                    "tokens": 150,
                    "latency": 1.5
                }
            ],
            "critique_analysis": {
                "analysis": "Test analysis",
                "suggestions": "Test suggestions",
                "scores": {
                    "consistency_score": 0.9,
                    "completeness_score": 0.85,
                    "accuracy_score": 0.95
                }
            },
            "final_answer": {
                "synthesized_answer": "Test synthesis",
                "explanation": "Test explanation",
                "confidence": 0.85
            }
        },
        "performance": {
            "total_duration": 2.5,
            "total_tokens": 250,
            "estimated_cost": 0.05,
            "iterations": 1
        },
        "quality": {
            "consistency_score": 0.9,
            "completeness_score": 0.85,
            "accuracy_score": 0.95,
            "confidence_score": 0.85
        }
    }


@pytest.fixture
def mock_llm_response():
    """Mock LLM response."""
    return {
        "response": "Test response",
        "tokens": 100,
        "latency": 1.0
    }


@pytest.fixture
def mock_critique_response():
    """Mock critique response."""
    return {
        "critique": {
            "analysis": "Test analysis",
            "suggestions": "Test suggestions"
        },
        "scores": {
            "consistency_score": 0.9,
            "completeness_score": 0.85,
            "accuracy_score": 0.95
        }
    }


@pytest.fixture
def mock_synthesis_response():
    """Mock synthesis response."""
    return {
        "synthesis": {
            "synthesized_answer": "Test synthesis",
            "explanation": "Test explanation"
        },
        "confidence": 0.85
    }


@pytest.fixture
def test_prompt_file(tmp_path):
    """Test prompt file."""
    prompt_file = tmp_path / "test_prompt.txt"
    prompt_file.write_text("Test prompt content")
    return prompt_file


@pytest.fixture
def test_config_file(tmp_path):
    """Test config file."""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text("""
    openai_api_key: test-openai-key
    anthropic_api_key: test-anthropic-key
    google_api_key: test-google-key
    default_models:
      - gpt-4
      - claude-3-opus
      - gemini-pro
    default_resolver: gpt-4
    max_iterations: 3
    confidence_threshold: 0.8
    timeout: 30
    log_dir: test_logs
    """)
    return config_file 
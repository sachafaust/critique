import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime
import json
from pathlib import Path

from llm_critique.core.synthesis import ResponseSynthesizer
from llm_critique.core.models import LLMClient


@pytest.fixture
def mock_llm_client():
    """Mock LLM client for testing."""
    client = MagicMock(spec=LLMClient)
    client.execute_model.return_value = {
        "response": "Test response",
        "tokens": 100,
        "latency": 1.0
    }
    return client


@pytest.fixture
def mock_chains():
    """Mock LangChain chains for testing."""
    with patch("llm_critique.core.synthesis.MultiLLMChain") as mock_multi_llm, \
         patch("llm_critique.core.synthesis.CritiqueAnalysisChain") as mock_critique, \
         patch("llm_critique.core.synthesis.SynthesisChain") as mock_synthesis:
        
        # Mock MultiLLMChain
        mock_multi_llm.return_value.invoke.return_value = {
            "responses": ["Response 1", "Response 2"],
            "metadata": [
                {"tokens": 100, "latency": 1.0},
                {"tokens": 150, "latency": 1.5}
            ]
        }
        
        # Mock CritiqueAnalysisChain
        mock_critique.return_value.invoke.return_value = {
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
        
        # Mock SynthesisChain
        mock_synthesis.return_value.invoke.return_value = {
            "synthesis": {
                "synthesized_answer": "Test synthesis",
                "explanation": "Test explanation"
            },
            "confidence": 0.85
        }
        
        yield {
            "multi_llm": mock_multi_llm,
            "critique": mock_critique,
            "synthesis": mock_synthesis
        }


def test_synthesis_workflow(mock_llm_client, mock_chains):
    """Test the complete synthesis workflow."""
    # Create synthesizer
    synthesizer = ResponseSynthesizer(
        llm_client=mock_llm_client,
        max_iterations=3,
        confidence_threshold=0.8
    )
    
    # Execute synthesis
    result = synthesizer.synthesize(
        prompt="Test prompt",
        models=["model1", "model2"],
        resolver_model="model1"
    )
    
    # Verify result structure
    assert "execution_id" in result
    assert "timestamp" in result
    assert "input" in result
    assert "results" in result
    assert "performance" in result
    assert "quality" in result
    
    # Verify input details
    assert result["input"]["prompt"] == "Test prompt"
    assert result["input"]["models"] == ["model1", "model2"]
    assert result["input"]["resolver_model"] == "model1"
    
    # Verify results
    assert "model_responses" in result["results"]
    assert "critique_analysis" in result["results"]
    assert "final_answer" in result["results"]
    
    # Verify performance metrics
    assert "total_duration" in result["performance"]
    assert "total_tokens" in result["performance"]
    assert "estimated_cost" in result["performance"]
    
    # Verify quality metrics
    assert "consistency_score" in result["quality"]
    assert "completeness_score" in result["quality"]
    assert "accuracy_score" in result["quality"]
    assert "confidence_score" in result["quality"]


def test_synthesis_iteration_limit(mock_llm_client, mock_chains):
    """Test synthesis with iteration limit."""
    # Mock low confidence responses
    mock_chains["synthesis"].return_value.invoke.return_value = {
        "synthesis": {
            "synthesized_answer": "Test synthesis",
            "explanation": "Test explanation"
        },
        "confidence": 0.7  # Below threshold
    }
    
    # Create synthesizer with low threshold
    synthesizer = ResponseSynthesizer(
        llm_client=mock_llm_client,
        max_iterations=2,
        confidence_threshold=0.8
    )
    
    # Execute synthesis
    result = synthesizer.synthesize(
        prompt="Test prompt",
        models=["model1"],
        resolver_model="model1"
    )
    
    # Verify iteration limit
    assert result["performance"]["iterations"] <= 2
    assert result["quality"]["confidence_score"] == 0.7


def test_synthesis_error_handling(mock_llm_client, mock_chains):
    """Test error handling in synthesis workflow."""
    # Mock chain error
    mock_chains["multi_llm"].return_value.invoke.side_effect = Exception("Test error")
    
    # Create synthesizer
    synthesizer = ResponseSynthesizer(
        llm_client=mock_llm_client,
        max_iterations=3,
        confidence_threshold=0.8
    )
    
    # Execute synthesis
    result = synthesizer.synthesize(
        prompt="Test prompt",
        models=["model1"],
        resolver_model="model1"
    )
    
    # Verify error handling
    assert "error" in result
    assert result["error"] == "Test error"
    assert result["results"]["model_responses"] == []
    assert result["results"]["critique_analysis"] is None
    assert result["results"]["final_answer"] is None


def test_synthesis_output_formatting(mock_llm_client, mock_chains):
    """Test output formatting in synthesis workflow."""
    # Create synthesizer
    synthesizer = ResponseSynthesizer(
        llm_client=mock_llm_client,
        max_iterations=3,
        confidence_threshold=0.8
    )
    
    # Execute synthesis with JSON output
    result = synthesizer.synthesize(
        prompt="Test prompt",
        models=["model1"],
        resolver_model="model1",
        output_format="json"
    )
    
    # Verify JSON formatting
    assert isinstance(result, dict)
    assert all(isinstance(v, (str, int, float, list, dict, type(None))) for v in result.values())
    
    # Test JSON serialization
    json_str = json.dumps(result)
    assert isinstance(json_str, str)
    parsed = json.loads(json_str)
    assert parsed == result


def test_synthesis_logging(mock_llm_client, mock_chains, tmp_path):
    """Test logging in synthesis workflow."""
    # Set up test log directory
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    
    # Create synthesizer with test log directory
    synthesizer = ResponseSynthesizer(
        llm_client=mock_llm_client,
        max_iterations=3,
        confidence_threshold=0.8,
        log_dir=str(log_dir)
    )
    
    # Execute synthesis
    result = synthesizer.synthesize(
        prompt="Test prompt",
        models=["model1"],
        resolver_model="model1"
    )
    
    # Verify log file creation
    log_files = list(log_dir.glob("*.json"))
    assert len(log_files) == 1
    
    # Verify log content
    with open(log_files[0]) as f:
        log_data = json.load(f)
        assert log_data["execution_id"] == result["execution_id"]
        assert log_data["input"] == result["input"]
        assert log_data["results"] == result["results"] 
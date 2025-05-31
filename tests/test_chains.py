import pytest
from unittest.mock import patch, MagicMock
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration, AIMessage

from llm_critique.core.chains import MultiLLMChain, CritiqueAnalysisChain, SynthesisChain


@pytest.fixture
def mock_model():
    """Mock LLM model for testing."""
    model = MagicMock(spec=BaseChatModel)
    model.invoke.return_value = ChatResult(
        generations=[
            ChatGeneration(
                message=AIMessage(content="Test response"),
                generation_info={"token_usage": {"total_tokens": 100}}
            )
        ],
        llm_output={"token_usage": {"total_tokens": 100}}
    )
    return model


def test_multi_llm_chain_parallel_execution(mock_model):
    """Test parallel execution in MultiLLMChain."""
    # Create chain
    chain = MultiLLMChain(
        models={"test-model": mock_model},
        prompt_template="{prompt}",
        timeout=30
    )
    
    # Execute chain
    result = chain.invoke({"prompt": "Test prompt"})
    
    # Verify result
    assert "responses" in result
    assert "metadata" in result
    assert len(result["responses"]) == 1
    assert len(result["metadata"]) == 1
    assert result["responses"][0] == "Test response"
    assert result["metadata"][0]["tokens"] == 100


def test_critique_analysis_scoring(mock_model):
    """Test critique analysis and scoring."""
    # Create chain
    chain = CritiqueAnalysisChain(model=mock_model)
    
    # Mock model response
    mock_model.invoke.return_value = ChatResult(
        generations=[
            ChatGeneration(
                message=AIMessage(content='''
                {
                    "consistency_score": 90,
                    "completeness_score": 85,
                    "accuracy_score": 95,
                    "analysis": "Test analysis",
                    "suggestions": "Test suggestions"
                }
                '''),
                generation_info={}
            )
        ],
        llm_output={}
    )
    
    # Execute chain
    result = chain.invoke({
        "prompt": "Test prompt",
        "responses": ["Response 1", "Response 2"]
    })
    
    # Verify result
    assert "critique" in result
    assert "scores" in result
    assert result["scores"]["consistency_score"] == 0.9
    assert result["scores"]["completeness_score"] == 0.85
    assert result["scores"]["accuracy_score"] == 0.95


def test_synthesis_chain_confidence(mock_model):
    """Test synthesis chain confidence scoring."""
    # Create chain
    chain = SynthesisChain(model=mock_model)
    
    # Mock model response
    mock_model.invoke.return_value = ChatResult(
        generations=[
            ChatGeneration(
                message=AIMessage(content='''
                {
                    "synthesized_answer": "Test synthesis",
                    "confidence_score": 85,
                    "explanation": "Test explanation"
                }
                '''),
                generation_info={}
            )
        ],
        llm_output={}
    )
    
    # Execute chain
    result = chain.invoke({
        "prompt": "Test prompt",
        "responses": ["Response 1", "Response 2"],
        "critique": {
            "analysis": "Test analysis",
            "suggestions": "Test suggestions"
        }
    })
    
    # Verify result
    assert "synthesis" in result
    assert "confidence" in result
    assert result["synthesis"]["synthesized_answer"] == "Test synthesis"
    assert result["confidence"] == 0.85


def test_chain_error_handling(mock_model):
    """Test error handling in chains."""
    # Create chain
    chain = MultiLLMChain(
        models={"test-model": mock_model},
        prompt_template="{prompt}",
        timeout=30
    )
    
    # Mock model error
    mock_model.invoke.side_effect = Exception("Test error")
    
    # Execute chain
    result = chain.invoke({"prompt": "Test prompt"})
    
    # Verify error handling
    assert "responses" in result
    assert "metadata" in result
    assert result["responses"][0] is None
    assert "error" in result["metadata"][0]
    assert result["metadata"][0]["error"] == "Test error"


def test_chain_timeout_handling(mock_model):
    """Test timeout handling in chains."""
    # Create chain
    chain = MultiLLMChain(
        models={"test-model": mock_model},
        prompt_template="{prompt}",
        timeout=1
    )
    
    # Mock model timeout
    mock_model.invoke.side_effect = TimeoutError("Test timeout")
    
    # Execute chain
    result = chain.invoke({"prompt": "Test prompt"})
    
    # Verify timeout handling
    assert "responses" in result
    assert "metadata" in result
    assert result["responses"][0] is None
    assert "error" in result["metadata"][0]
    assert "timeout" in result["metadata"][0]["error"].lower() 
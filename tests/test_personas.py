"""
Tests for the persona system including expert personas and vanilla model personas.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from llm_critique.core.personas import (
    PersonaConfig, PersonaType, UnifiedPersonaManager, 
    UnifiedCritic, CritiqueResult
)


class TestPersonaConfig:
    """Test PersonaConfig dataclass functionality."""
    
    def test_expert_persona_creation(self):
        """Test creating an expert persona configuration."""
        persona = PersonaConfig(
            name="Test Expert",
            persona_type=PersonaType.EXPERT,
            description="Test expert persona",
            core_principles=["Principle 1", "Principle 2"],
            preferred_model="gpt-4o"
        )
        
        assert persona.name == "Test Expert"
        assert persona.persona_type == PersonaType.EXPERT
        assert len(persona.core_principles) == 2
        assert persona.preferred_model == "gpt-4o"
    
    def test_vanilla_persona_creation(self):
        """Test creating a vanilla persona configuration."""
        persona = PersonaConfig(
            name="claude-4-sonnet",
            persona_type=PersonaType.VANILLA,
            description="AI model providing general critique"
        )
        
        assert persona.name == "claude-4-sonnet"
        assert persona.persona_type == PersonaType.VANILLA
        assert persona.preferred_model == "claude-4-sonnet"  # Auto-set in __post_init__
    
    def test_context_size_estimation(self):
        """Test token context size estimation."""
        expert_persona = PersonaConfig(
            name="Expert",
            persona_type=PersonaType.EXPERT,
            description="Test expert with detailed context",
            core_principles=["Long principle " * 10],
            key_questions=["Detailed question " * 5]
        )
        
        vanilla_persona = PersonaConfig(
            name="model",
            persona_type=PersonaType.VANILLA
        )
        
        expert_tokens = expert_persona.get_prompt_context_size_estimate()
        vanilla_tokens = vanilla_persona.get_prompt_context_size_estimate()
        
        assert expert_tokens > vanilla_tokens
        assert vanilla_tokens == 50  # Fixed minimal context


class TestUnifiedPersonaManager:
    """Test UnifiedPersonaManager functionality."""
    
    @pytest.fixture
    def temp_personas_dir(self):
        """Create temporary personas directory with test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            personas_dir = Path(temp_dir)
            
            # Create test expert persona YAML
            expert_data = {
                "name": "Test Expert",
                "description": "Test expert persona",
                "core_principles": ["Test principle"],
                "critique_style": {"approach": "Systematic"},
                "key_questions": ["What is the test?"],
                "preferred_models": ["gpt-4o"]
            }
            
            expert_file = personas_dir / "test_expert.yaml"
            with open(expert_file, 'w') as f:
                yaml.dump(expert_data, f)
            
            yield personas_dir
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_initialization_with_api_keys(self, temp_personas_dir):
        """Test persona manager initialization with API keys."""
        manager = UnifiedPersonaManager(personas_dir=str(temp_personas_dir))
        
        assert len(manager._available_models) > 0
        assert "gpt-4o" in manager._available_models
        assert "test_expert" in manager._expert_persona_cache
    
    @patch.dict('os.environ', {}, clear=True)
    def test_initialization_without_api_keys(self, temp_personas_dir):
        """Test persona manager initialization without API keys."""
        manager = UnifiedPersonaManager(personas_dir=str(temp_personas_dir))
        
        assert len(manager._available_models) == 0
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_load_expert_persona(self, temp_personas_dir):
        """Test loading expert persona from YAML file."""
        manager = UnifiedPersonaManager(personas_dir=str(temp_personas_dir))
        
        persona = manager.load_expert_persona("test_expert")
        
        assert persona.name == "Test Expert"
        assert persona.persona_type == PersonaType.EXPERT
        assert len(persona.core_principles) == 1
        assert persona.core_principles[0] == "Test principle"
        assert persona.preferred_model == "gpt-4o"
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_create_vanilla_persona(self, temp_personas_dir):
        """Test creating vanilla persona from model name."""
        manager = UnifiedPersonaManager(personas_dir=str(temp_personas_dir))
        
        persona = manager.create_vanilla_persona("gpt-4o")
        
        assert persona.name == "gpt-4o"
        assert persona.persona_type == PersonaType.VANILLA
        assert persona.preferred_model == "gpt-4o"
        assert persona.description == "AI model gpt-4o providing general critique"
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_get_persona_expert(self, temp_personas_dir):
        """Test unified persona getter for expert personas."""
        manager = UnifiedPersonaManager(personas_dir=str(temp_personas_dir))
        
        persona = manager.get_persona("test_expert")
        
        assert persona.persona_type == PersonaType.EXPERT
        assert persona.name == "Test Expert"
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_get_persona_vanilla(self, temp_personas_dir):
        """Test unified persona getter for vanilla models."""
        manager = UnifiedPersonaManager(personas_dir=str(temp_personas_dir))
        
        persona = manager.get_persona("gpt-4o")
        
        assert persona.persona_type == PersonaType.VANILLA
        assert persona.name == "gpt-4o"
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_get_persona_invalid(self, temp_personas_dir):
        """Test unified persona getter with invalid name."""
        manager = UnifiedPersonaManager(personas_dir=str(temp_personas_dir))
        
        with pytest.raises(ValueError, match="Unknown persona or model"):
            manager.get_persona("invalid_persona")
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_list_available_personas(self, temp_personas_dir):
        """Test listing available personas."""
        manager = UnifiedPersonaManager(personas_dir=str(temp_personas_dir))
        
        personas_list = manager.list_available_personas()
        
        assert "expert_personas" in personas_list
        assert "vanilla_models" in personas_list
        assert "test_expert" in personas_list["expert_personas"]
        assert "gpt-4o" in personas_list["vanilla_models"]
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_validate_persona_combination(self, temp_personas_dir):
        """Test persona combination validation."""
        manager = UnifiedPersonaManager(personas_dir=str(temp_personas_dir))
        
        # Valid combination
        result = manager.validate_persona_combination(["test_expert", "gpt-4o"])
        assert result["valid"] is True
        assert len(result["personas"]) == 2
        
        # Invalid combination
        result = manager.validate_persona_combination(["invalid_persona"])
        assert result["valid"] is False
        assert len(result["errors"]) > 0


class TestUnifiedCritic:
    """Test UnifiedCritic functionality."""
    
    @pytest.fixture
    def expert_persona(self):
        """Create test expert persona."""
        return PersonaConfig(
            name="Test Expert",
            persona_type=PersonaType.EXPERT,
            description="Test expert persona",
            core_principles=["Test principle"],
            critique_style={"approach": "Systematic"},
            key_questions=["What is the test?"],
            language_patterns={"openings": ["Based on my analysis..."]},
            preferred_model="gpt-4o"
        )
    
    @pytest.fixture
    def vanilla_persona(self):
        """Create test vanilla persona."""
        return PersonaConfig(
            name="gpt-4o",
            persona_type=PersonaType.VANILLA,
            preferred_model="gpt-4o"
        )
    
    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LLM client."""
        mock_client = Mock()
        return mock_client
    
    def test_expert_critic_initialization(self, expert_persona, mock_llm_client):
        """Test initializing critic with expert persona."""
        critic = UnifiedCritic(expert_persona, mock_llm_client)
        
        assert critic.persona == expert_persona
        assert critic.model == "gpt-4o"
        assert critic.llm_client == mock_llm_client
    
    def test_vanilla_critic_initialization(self, vanilla_persona, mock_llm_client):
        """Test initializing critic with vanilla persona."""
        critic = UnifiedCritic(vanilla_persona, mock_llm_client)
        
        assert critic.persona == vanilla_persona
        assert critic.model == "gpt-4o"
    
    def test_expert_prompt_generation(self, expert_persona, mock_llm_client):
        """Test prompt generation for expert persona."""
        critic = UnifiedCritic(expert_persona, mock_llm_client)
        
        prompt = critic.generate_critique_prompt("Test content", "Test context")
        
        assert "EXPERT PERSONA CRITIQUE" in prompt
        assert "Test Expert" in prompt
        assert "Test principle" in prompt
        assert "What is the test?" in prompt
        assert "Test content" in prompt
    
    def test_vanilla_prompt_generation(self, vanilla_persona, mock_llm_client):
        """Test prompt generation for vanilla persona."""
        critic = UnifiedCritic(vanilla_persona, mock_llm_client)
        
        prompt = critic.generate_critique_prompt("Test content", "Test context")
        
        assert "AI MODEL CRITIQUE" in prompt
        assert "gpt-4o" in prompt
        assert "Test content" in prompt
        assert "EXPERT PERSONA" not in prompt
    
    @pytest.mark.asyncio
    async def test_execute_critique_success(self, expert_persona, mock_llm_client):
        """Test successful critique execution."""
        critic = UnifiedCritic(expert_persona, mock_llm_client)
        
        # Mock LLM response
        mock_response = "Quality Assessment: 85\nKey Insights: Great analysis\nRecommendations: Improve clarity\nConfidence: 90"
        mock_llm_client.generate_response = AsyncMock(return_value=mock_response)
        
        result = await critic.execute_critique("Test content")
        
        assert isinstance(result, CritiqueResult)
        assert result.persona_name == "Test Expert"
        assert result.persona_type == PersonaType.EXPERT
        assert result.quality_score > 0
        assert result.confidence_level > 0
        assert len(result.key_insights) > 0
    
    @pytest.mark.asyncio
    async def test_execute_critique_error(self, expert_persona, mock_llm_client):
        """Test critique execution with error."""
        critic = UnifiedCritic(expert_persona, mock_llm_client)
        
        # Mock LLM error
        mock_llm_client.generate_response = AsyncMock(side_effect=Exception("API Error"))
        
        result = await critic.execute_critique("Test content")
        
        assert isinstance(result, CritiqueResult)
        assert result.quality_score == 0.0
        assert "Error executing critique" in result.key_insights[0]


class TestCritiqueResult:
    """Test CritiqueResult functionality."""
    
    def test_critique_result_creation(self):
        """Test creating CritiqueResult."""
        result = CritiqueResult(
            persona_name="Test",
            persona_type=PersonaType.EXPERT,
            model_used="gpt-4o",
            quality_score=0.85,
            key_insights=["Insight 1"],
            recommendations=["Recommendation 1"],
            confidence_level=0.9,
            critique_text="Test critique"
        )
        
        assert result.persona_name == "Test"
        assert result.quality_score == 0.85
        assert len(result.key_insights) == 1
    
    def test_to_dict_conversion(self):
        """Test converting CritiqueResult to dictionary."""
        result = CritiqueResult(
            persona_name="Test",
            persona_type=PersonaType.EXPERT,
            model_used="gpt-4o",
            quality_score=0.85,
            key_insights=["Insight 1"],
            recommendations=["Recommendation 1"],
            confidence_level=0.9,
            critique_text="Test critique"
        )
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict["persona_name"] == "Test"
        assert result_dict["persona_type"] == "expert"
        assert result_dict["quality_score"] == 0.85


@pytest.mark.integration
class TestPersonaSystemIntegration:
    """Integration tests for the complete persona system."""
    
    @pytest.fixture
    def full_setup(self):
        """Set up complete test environment."""
        with tempfile.TemporaryDirectory() as temp_dir:
            personas_dir = Path(temp_dir)
            
            # Create comprehensive expert persona
            expert_data = {
                "name": "Integration Expert",
                "description": "Integration testing expert",
                "core_principles": ["Thorough testing", "Quality assurance"],
                "critique_style": {
                    "approach": "Systematic integration testing",
                    "focus_areas": "End-to-end workflows",
                    "communication_tone": "Technical and precise"
                },
                "key_questions": [
                    "Does this integrate properly?",
                    "Are all components working together?"
                ],
                "decision_frameworks": ["Test-driven development"],
                "language_patterns": {
                    "openings": ["From an integration perspective..."],
                    "emphasis_terms": ["integration", "end-to-end", "systematic"]
                },
                "red_flags": ["Missing integration tests"],
                "success_indicators": ["All components working together"],
                "expertise_domains": ["Integration testing", "Quality assurance"],
                "preferred_models": ["gpt-4o"],
                "temperature": 0.2,
                "max_tokens": 1500
            }
            
            expert_file = personas_dir / "integration_expert.yaml"
            with open(expert_file, 'w') as f:
                yaml.dump(expert_data, f)
            
            yield personas_dir
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_full_persona_workflow(self, full_setup):
        """Test complete persona workflow from loading to execution."""
        manager = UnifiedPersonaManager(personas_dir=str(full_setup))
        
        # Test persona discovery and loading
        personas_list = manager.list_available_personas()
        assert "integration_expert" in personas_list["expert_personas"]
        
        # Test persona retrieval
        expert_persona = manager.get_persona("integration_expert")
        vanilla_persona = manager.get_persona("gpt-4o")
        
        assert expert_persona.persona_type == PersonaType.EXPERT
        assert vanilla_persona.persona_type == PersonaType.VANILLA
        
        # Test persona validation
        validation = manager.validate_persona_combination(["integration_expert", "gpt-4o"])
        assert validation["valid"] is True
        
        # Test persona info retrieval
        info = manager.get_persona_info("integration_expert")
        assert info["name"] == "Integration Expert"
        assert info["type"] == "expert"
        assert len(info["core_principles"]) == 2
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_cost_optimization_warnings(self, full_setup):
        """Test cost optimization warnings for large persona combinations."""
        manager = UnifiedPersonaManager(personas_dir=str(full_setup))
        
        # Create a large combination that should trigger warnings
        large_combination = ["integration_expert"] + ["gpt-4o"] * 5
        
        validation = manager.validate_persona_combination(large_combination)
        
        # Should still be valid but have warnings
        assert validation["valid"] is True
        assert len(validation["warnings"]) > 0 
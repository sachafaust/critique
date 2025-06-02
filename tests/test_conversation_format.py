import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, MagicMock
from dataclasses import dataclass
from typing import List, Dict, Any
from enum import Enum

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from llm_critique.core.conversation import ConversationManager

# Mock data structures to match the real ones
class PersonaType(Enum):
    EXPERT = "expert"
    VANILLA = "vanilla"

@dataclass
class MockCritiqueResult:
    """Mock critique result that matches the real CritiqueResult structure."""
    persona_name: str
    persona_type: PersonaType
    quality_score: float
    confidence_level: float
    key_insights: List[str]
    recommendations: List[str]
    red_flags_identified: List[str]
    critique_text: str
    model_used: str = "o3-mini"

class TestConversationFormat:
    """Test suite for validating conversation recording format."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.conversation_manager = ConversationManager()
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_mock_results(self, iterations: int = 2, expert_creator: bool = True) -> Dict[str, Any]:
        """Create mock results data that matches the real synthesis output structure."""
        
        # Mock creator persona
        creator_name = "Elon Musk" if expert_creator else "gpt-4o"
        creator_type = "expert" if expert_creator else "vanilla"
        creator_model = "o3-mini"
        
        # Mock persona critiques for each iteration
        mock_iterations = []
        for i in range(1, iterations + 1):
            # Mock creator response
            creator_response = {
                "content": f"This is iteration {i} creator response from {creator_name}. "
                          f"From first principles, we need to approach this problem by breaking it down "
                          f"into fundamental components. The key insight here is that we must optimize "
                          f"for both efficiency and user experience while maintaining scalability.",
                "model": creator_model,
                "persona": creator_name,
                "persona_type": creator_type
            }
            
            # Mock persona critiques
            persona_critiques = [
                MockCritiqueResult(
                    persona_name="Steve Jobs",
                    persona_type=PersonaType.EXPERT,
                    quality_score=0.85,
                    confidence_level=0.95,
                    key_insights=[
                        "The emphasis on user experience is exactly what creates magical, intuitive flow",
                        "The focus on fundamental principles shows deep understanding of the problem"
                    ],
                    recommendations=[
                        "Eliminate unnecessary technical abstractions and focus on human experience",
                        "Streamline the language further to ignite imagination and bolster confidence"
                    ],
                    red_flags_identified=[
                        "Metaphors sometimes veer towards abstract rather than tangible",
                        "Risk of getting lost in technical jargon without emotional connection"
                    ],
                    critique_text="Quality Assessment: 85\n\nThis response demonstrates solid understanding "
                                 "of first principles thinking and user-centric design. The emphasis on "
                                 "breaking down complexity aligns with creating insanely great products. "
                                 "However, we need to ensure every element creates emotional connection."
                ),
                MockCritiqueResult(
                    persona_name="Ray Dalio",
                    persona_type=PersonaType.EXPERT,
                    quality_score=0.78,
                    confidence_level=0.88,
                    key_insights=[
                        "The systematic approach to problem decomposition shows strong analytical thinking",
                        "Focus on scalability indicates awareness of long-term sustainability"
                    ],
                    recommendations=[
                        "Include more specific metrics and measurable outcomes",
                        "Add stress-testing scenarios to validate approach robustness"
                    ],
                    red_flags_identified=[
                        "Lacks specific data points to support efficiency claims",
                        "Missing risk assessment for potential failure modes"
                    ],
                    critique_text="Quality Assessment: 78\n\nThe response shows good systematic thinking "
                                 "and principles-based approach. The focus on efficiency and scalability "
                                 "is appropriate. However, we need more concrete evidence and data to "
                                 "support the claims and better risk analysis."
                )
            ]
            
            mock_iterations.append({
                "iteration_num": i,
                "creator_response": creator_response,
                "persona_critiques": persona_critiques,
                "consensus_score": 0.82,
                "should_continue": i < iterations
            })
        
        # Mock full results structure
        return {
            "execution_id": "test-exec-12345",
            "timestamp": "2025-06-02T17:35:26.951239Z",
            "input": {
                "prompt": "Design a strategy for testing conversation formats effectively",
                "creator_persona": creator_name,
                "creator_type": creator_type,
                "creator_model": creator_model,
                "personas": ["Steve Jobs", "Ray Dalio"],
                "max_iterations": iterations,
                "mode": "persona_enhanced"
            },
            "results": {
                "workflow_results": {
                    "iterations": mock_iterations,
                    "final_content": mock_iterations[-1]["creator_response"]["content"],
                    "total_iterations": iterations,
                    "convergence_achieved": True
                },
                "final_answer": mock_iterations[-1]["creator_response"]["content"],
                "confidence_score": 0.85,
                "consensus_score": 0.82,
                "total_iterations": iterations,
                "convergence_achieved": True
            },
            "persona_analysis": {
                "expert_insights": ["User experience focus", "Systematic thinking"],
                "consensus_areas": ["Quality improvements needed", "Clarity enhancements required"],
                "conflicting_perspectives": [
                    {
                        "type": "quality_assessment",
                        "high_assessment": ["Steve Jobs"],
                        "low_assessment": ["Ray Dalio"],
                        "score_range": 0.07
                    }
                ],
                "weighted_recommendations": [
                    {
                        "recommendation": "Focus on human experience over technical abstractions",
                        "persona": "Steve Jobs",
                        "weight": 0.90,
                        "expertise_match": 0.85
                    }
                ]
            },
            "performance": {
                "total_duration_ms": 15000.0,
                "estimated_cost_usd": 0.0088
            },
            "quality_metrics": {
                "average_confidence": 0.915,
                "persona_consensus": 0.82
            }
        }
    
    def test_conversation_initialization(self):
        """Test conversation manager initialization."""
        assert self.conversation_manager.console_output == []
        assert self.conversation_manager.metadata == {}
    
    def test_start_recording_expert_creator(self):
        """Test starting recording with expert creator."""
        self.conversation_manager.start_recording(
            prompt="Test prompt",
            models=["Steve Jobs", "Ray Dalio"],
            creator_model="o3-mini",
            creator_persona="Elon Musk"
        )
        
        assert self.conversation_manager.metadata["prompt"] == "Test prompt"
        assert self.conversation_manager.metadata["models"] == ["Steve Jobs", "Ray Dalio"]
        assert self.conversation_manager.metadata["creator_model"] == "o3-mini"
        assert self.conversation_manager.metadata["creator_persona"] == "Elon Musk"
        
        # Check debug output
        assert "DEBUG requested_models: ['Steve Jobs', 'Ray Dalio']" in self.conversation_manager.console_output
        assert "DEBUG models_to_use: ['Steve Jobs', 'Ray Dalio']" in self.conversation_manager.console_output
        assert "DEBUG creator_model: o3-mini" in self.conversation_manager.console_output
    
    def test_record_iteration_header(self):
        """Test recording iteration header."""
        self.conversation_manager.record_iteration_start(
            total_iterations=2,
            convergence=True,
            creator_model="o3-mini",
            critic_models=["Steve Jobs", "Ray Dalio"]
        )
        
        output = "\n".join(self.conversation_manager.console_output)
        
        # Check header formatting
        assert "🔄 CREATOR-CRITIC ITERATION RESULTS" in output
        assert "📊 Total Iterations: 2" in output
        assert "🎯 Convergence Achieved: ✅ Yes" in output
        assert "⚙️  Creator Model: o3-mini" in output
        assert "🔍 Critic Models: Steve Jobs, Ray Dalio" in output
        assert "=" * 80 in output
    
    def test_record_single_iteration(self):
        """Test recording a single iteration with rich formatting."""
        critics_feedback = [
            {
                "quality_score": 85.0,
                "strengths": ["Great user experience focus", "Solid first principles thinking"],
                "improvements": ["Add more concrete examples", "Reduce technical jargon"],
                "decision": "Stop",
                "detailed_feedback": "This is detailed feedback from Steve Jobs about the response quality."
            },
            {
                "quality_score": 78.0,
                "strengths": ["Good analytical approach", "Systematic thinking"],
                "improvements": ["Include metrics", "Add risk assessment"],
                "decision": "Continue",
                "detailed_feedback": "This is detailed feedback from Ray Dalio about evidence and data."
            }
        ]
        
        self.conversation_manager.record_iteration(
            iteration_num=1,
            creator_output="This is a test creator response with multiple lines.\n\nIt includes paragraphs and demonstrates the formatting.",
            creator_confidence=80.0,
            creator_model="o3-mini",
            critics_feedback=critics_feedback
        )
        
        output = "\n".join(self.conversation_manager.console_output)
        
        # Check iteration structure
        assert "==================== ITERATION 1 ====================" in output
        assert "🎨 CREATOR OUTPUT (o3-mini)" in output
        assert "Confidence: 80.0%" in output
        
        # Check creator response panel formatting
        assert "╭─ 📝 Creator Response - Iteration 1" in output
        assert "This is a test creator response" in output
        assert "╰─" in output and "─╯" in output
        
        # Check critics feedback structure
        assert "🔍 CRITICS FEEDBACK" in output
        assert "🤖 Critic 1" in output
        assert "📊 Quality Score: 85.0%" in output
        assert "💪 Strengths:" in output
        assert "🔧 Improvements:" in output
        assert "🎯 Decision: ✅ Stop" in output
        
        # Check detailed feedback panels
        assert "💬 Detailed Feedback from Critic 1" in output
        assert "This is detailed feedback from Steve Jobs" in output
    
    def test_record_final_results(self):
        """Test recording final results section."""
        self.conversation_manager.record_final_results(
            final_answer="This is the final synthesized answer after all iterations.",
            confidence=92.5,
            quality=87.5,
            duration=15.5,
            cost=0.0088,
            execution_id="test-exec-12345",
            models_used=["Steve Jobs", "Ray Dalio"],
            creator_model="o3-mini"
        )
        
        output = "\n".join(self.conversation_manager.console_output)
        
        # Check final results structure
        assert "========================= FINAL RESULTS =========================" in output
        assert "🏆 FINAL ANSWER" in output
        assert "This is the final synthesized answer" in output
        
        # Check quality metrics
        assert "📈 QUALITY METRICS" in output
        assert "🎯 Final Confidence: 92.5%" in output
        assert "⭐ Final Quality: 87.5%" in output
        
        # Check performance metrics
        assert "⚡ PERFORMANCE" in output
        assert "⏱️  Total Duration: 15.5s" in output
        assert "💰 Estimated Cost: $0.0088" in output
        
        # Check execution details
        assert "Execution ID: test-exec-12345" in output
        assert "Models used: ['Steve Jobs', 'Ray Dalio']" in output
        assert "Creator model: o3-mini" in output
    
    def test_save_and_load_conversation(self):
        """Test saving and loading conversation with rich format."""
        # Record a complete conversation
        self.conversation_manager.start_recording(
            prompt="Test save/load functionality",
            models=["Steve Jobs"],
            creator_model="o3-mini",
            creator_persona="Elon Musk"
        )
        
        self.conversation_manager.record_iteration_start(
            total_iterations=1,
            convergence=True,
            creator_model="o3-mini",
            critic_models=["Steve Jobs"]
        )
        
        self.conversation_manager.record_iteration(
            iteration_num=1,
            creator_output="Test creator output for save/load",
            creator_confidence=85.0,
            creator_model="o3-mini",
            critics_feedback=[{
                "quality_score": 90.0,
                "strengths": ["Clear and concise"],
                "improvements": ["Add more detail"],
                "decision": "Stop",
                "detailed_feedback": "Good response overall."
            }]
        )
        
        self.conversation_manager.record_final_results(
            final_answer="Final test answer",
            confidence=90.0,
            quality=85.0,
            duration=10.0,
            cost=0.005,
            execution_id="test-save-load",
            models_used=["Steve Jobs"],
            creator_model="o3-mini"
        )
        
        # Save conversation
        test_file = os.path.join(self.temp_dir, "test_conversation.txt")
        self.conversation_manager.save_conversation(test_file)
        
        # Verify file exists and has correct permissions
        assert os.path.exists(test_file)
        
        # Load and verify content
        loaded_content = self.conversation_manager.load_conversation(test_file)
        
        # Check key elements are preserved
        assert "🔄 CREATOR-CRITIC ITERATION RESULTS" in loaded_content
        assert "DEBUG requested_models: ['Steve Jobs']" in loaded_content
        assert "🎨 CREATOR OUTPUT (o3-mini)" in loaded_content
        assert "Test creator output for save/load" in loaded_content
        assert "🔍 CRITICS FEEDBACK" in loaded_content
        assert "🏆 FINAL ANSWER" in loaded_content
        assert "Final test answer" in loaded_content
        assert "Execution ID: test-save-load" in loaded_content
    
    def test_conversation_with_multiple_iterations(self):
        """Test conversation with multiple iterations."""
        mock_results = self.create_mock_results(iterations=3, expert_creator=True)
        
        # Initialize conversation
        self.conversation_manager.start_recording(
            prompt=mock_results["input"]["prompt"],
            models=mock_results["input"]["personas"],
            creator_model=mock_results["input"]["creator_model"],
            creator_persona=mock_results["input"]["creator_persona"]
        )
        
        # Record header
        self.conversation_manager.record_iteration_start(
            total_iterations=mock_results["results"]["total_iterations"],
            convergence=mock_results["results"]["convergence_achieved"],
            creator_model=mock_results["input"]["creator_model"],
            critic_models=mock_results["input"]["personas"]
        )
        
        # Record all iterations
        for iteration in mock_results["results"]["workflow_results"]["iterations"]:
            critics_feedback = []
            for critique in iteration["persona_critiques"]:
                feedback_dict = {
                    "quality_score": critique.quality_score * 100,
                    "strengths": critique.key_insights,
                    "improvements": critique.recommendations,
                    "decision": "Stop" if critique.quality_score >= 0.8 else "Continue",
                    "detailed_feedback": critique.critique_text
                }
                critics_feedback.append(feedback_dict)
            
            self.conversation_manager.record_iteration(
                iteration_num=iteration["iteration_num"],
                creator_output=iteration["creator_response"]["content"],
                creator_confidence=80.0,
                creator_model=iteration["creator_response"]["model"],
                critics_feedback=critics_feedback
            )
        
        # Record final results
        quality_metrics = mock_results["quality_metrics"]
        performance = mock_results["performance"]
        
        self.conversation_manager.record_final_results(
            final_answer=mock_results["results"]["final_answer"],
            confidence=quality_metrics["average_confidence"] * 100,
            quality=quality_metrics["persona_consensus"] * 100,
            duration=performance["total_duration_ms"] / 1000,
            cost=performance["estimated_cost_usd"],
            execution_id=mock_results["execution_id"],
            models_used=mock_results["input"]["personas"],
            creator_model=mock_results["input"]["creator_model"]
        )
        
        output = "\n".join(self.conversation_manager.console_output)
        
        # Verify all iterations are recorded
        assert "==================== ITERATION 1 ====================" in output
        assert "==================== ITERATION 2 ====================" in output
        assert "==================== ITERATION 3 ====================" in output
        
        # Verify iteration content
        assert "This is iteration 1 creator response" in output
        assert "This is iteration 2 creator response" in output
        assert "This is iteration 3 creator response" in output
        
        # Verify final results
        assert "🏆 FINAL ANSWER" in output
        assert "📈 QUALITY METRICS" in output
        assert "⚡ PERFORMANCE" in output
    
    def test_vanilla_creator_format(self):
        """Test conversation format with vanilla creator (non-expert)."""
        mock_results = self.create_mock_results(iterations=1, expert_creator=False)
        
        # Test that vanilla creators work with the format
        self.conversation_manager.start_recording(
            prompt=mock_results["input"]["prompt"],
            models=mock_results["input"]["personas"],
            creator_model=mock_results["input"]["creator_model"],
            creator_persona=mock_results["input"]["creator_persona"]
        )
        
        output = "\n".join(self.conversation_manager.console_output)
        
        # Should work with vanilla creator model names
        assert "DEBUG creator_model: o3-mini" in output
        assert "DEBUG requested_models: ['Steve Jobs', 'Ray Dalio']" in output
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with empty feedback
        self.conversation_manager.record_iteration(
            iteration_num=1,
            creator_output="",
            creator_confidence=0.0,
            creator_model="test-model",
            critics_feedback=[]
        )
        
        output = "\n".join(self.conversation_manager.console_output)
        assert "🔍 CRITICS FEEDBACK" in output
        
        # Test with very long content (should wrap properly)
        long_content = "A" * 200  # Longer than panel width
        self.conversation_manager.record_iteration(
            iteration_num=2,
            creator_output=long_content,
            creator_confidence=50.0,
            creator_model="test-model",
            critics_feedback=[{
                "quality_score": 50.0,
                "strengths": [],
                "improvements": [],
                "decision": "Continue",
                "detailed_feedback": "B" * 300  # Very long feedback
            }]
        )
        
        output = "\n".join(self.conversation_manager.console_output)
        # Should contain the content but properly wrapped
        assert "A" * 50 in output  # Some portion should be there
        assert "B" * 50 in output  # Some portion of feedback should be there

if __name__ == "__main__":
    # Run tests manually if executed directly
    test_instance = TestConversationFormat()
    test_methods = [method for method in dir(test_instance) if method.startswith('test_')]
    
    passed = 0
    failed = 0
    
    for method_name in test_methods:
        try:
            test_instance.setup_method()
            method = getattr(test_instance, method_name)
            method()
            print(f"✅ {method_name} - PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ {method_name} - FAILED: {e}")
            failed += 1
        finally:
            test_instance.teardown_method()
    
    print(f"\n📊 Test Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("🎉 All tests passed!")
    else:
        print(f"⚠️  {failed} tests failed") 
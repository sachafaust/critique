#!/usr/bin/env python3

from tests.test_conversation_format import TestConversationFormat
import tempfile
import os

def main():
    test = TestConversationFormat()
    test.setup_method()

    # Generate a sample conversation
    test.conversation_manager.start_recording(
        prompt='Design a strategy for making electric vehicles mainstream',
        models=['Steve Jobs', 'Ray Dalio'],
        creator_model='o3-mini',
        creator_persona='Elon Musk'
    )

    test.conversation_manager.record_iteration_start(
        total_iterations=2,
        convergence=True,
        creator_model='o3-mini',
        critic_models=['Steve Jobs', 'Ray Dalio']
    )

    test.conversation_manager.record_iteration(
        iteration_num=1,
        creator_output='To make electric vehicles mainstream, we must focus on three fundamental pillars: cost parity with gasoline vehicles, charging infrastructure that matches convenience of gas stations, and vehicles that exceed performance expectations. The path forward requires manufacturing at unprecedented scale while maintaining quality and pushing the boundaries of battery technology.',
        creator_confidence=85.0,
        creator_model='o3-mini',
        critics_feedback=[{
            'quality_score': 88.0,
            'strengths': ['Clear strategic pillars', 'User experience focus', 'Scalable vision'],
            'improvements': ['Add manufacturing scale considerations', 'Address consumer psychology barriers'],
            'decision': 'Continue',
            'detailed_feedback': 'The three-pillar approach is elegant and user-focused. This captures the essence of making technology accessible and delightful. However, we need to think about the emotional connection - why will people fall in love with electric vehicles? The focus on exceeding performance is brilliant.'
        }, {
            'quality_score': 82.0,
            'strengths': ['Systematic approach', 'Focus on fundamentals'],
            'improvements': ['Include specific metrics and timelines', 'Add risk mitigation strategies'],
            'decision': 'Continue',
            'detailed_feedback': 'Good systematic thinking about the core challenges. The cost parity and infrastructure points are crucial. However, we need more concrete data on manufacturing costs, charging network deployment timelines, and stress-testing scenarios for supply chain disruptions.'
        }]
    )

    test.conversation_manager.record_final_results(
        final_answer='To achieve mainstream EV adoption, we need cost parity through manufacturing scale, charging infrastructure that exceeds gas station convenience, and vehicles that create emotional connection through superior performance and user experience.',
        confidence=89.0,
        quality=86.0,
        duration=12.5,
        cost=0.0045,
        execution_id='demo-exec-123',
        models_used=['Steve Jobs', 'Ray Dalio'],
        creator_model='o3-mini'
    )

    # Save and show sample output
    temp_file = os.path.join(tempfile.gettempdir(), 'demo_conversation.txt')
    test.conversation_manager.save_conversation(temp_file)

    print('\n' + '='*80)
    print(' ' * 20 + '📋 SAMPLE CONVERSATION FORMAT DEMO')
    print('='*80)
    print()
    
    with open(temp_file, 'r') as f:
        content = f.read()
        print(content)
        
    print('\n' + '='*80)
    print(' ' * 25 + '✅ DEMO COMPLETE')
    print('='*80)
    print()
    print(f'📁 Full conversation saved to: {temp_file}')
    print('🎯 This demonstrates the rich human-readable format')
    print('💡 Compare this to the old JSON format for clarity improvement')
    
    test.teardown_method()

if __name__ == "__main__":
    main() 
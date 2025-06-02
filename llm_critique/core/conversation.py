from typing import Dict, List, Optional
import json
import os
import stat
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax

console = Console()

def _make_serializable(obj):
    """Convert any object to JSON-serializable format."""
    from datetime import datetime
    import uuid
    
    # Handle common Python types
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, uuid.UUID):
        return str(obj)
    elif isinstance(obj, (list, tuple, set)):
        return [_make_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: _make_serializable(value) for key, value in obj.items()}
    elif hasattr(obj, 'to_dict'):
        # Handle objects with to_dict method (like CritiqueResult)
        return _make_serializable(obj.to_dict())
    elif hasattr(obj, '__dict__'):
        # Handle other objects by converting their __dict__
        return _make_serializable(obj.__dict__)
    elif hasattr(obj, 'value'):
        # Handle enums
        return obj.value
    else:
        # Last resort: convert to string
        return str(obj)

class ConversationManager:
    """Manages conversation recording and replay functionality with rich formatting."""
    
    def __init__(self):
        self.console_output = []
        self.metadata = {}
        
    def start_recording(self, prompt: str, models: List[str], creator_model: str = None, creator_persona: str = None):
        """Start recording with initial metadata."""
        self.metadata = {
            "prompt": prompt,
            "models": models,
            "creator_model": creator_model,
            "creator_persona": creator_persona,
            "started_at": datetime.now().isoformat()
        }
        
        # Add debug info to output
        if models:
            self.console_output.append(f"DEBUG requested_models: {models}")
            self.console_output.append(f"DEBUG models_to_use: {models}")
        if creator_model:
            self.console_output.append(f"DEBUG creator_model: {creator_model}")
        
    def record_iteration_start(self, total_iterations: int, convergence: bool, creator_model: str, critic_models: List[str]):
        """Record the start of iterations with header."""
        self.console_output.append("")
        self.console_output.append("=" * 80)
        self.console_output.append(" " * 24 + "🔄 CREATOR-CRITIC ITERATION RESULTS" + " " * 24)
        self.console_output.append("=" * 80)
        self.console_output.append(f"📊 Total Iterations: {total_iterations}")
        self.console_output.append(f"🎯 Convergence Achieved: {'✅ Yes' if convergence else '❌ No'}")
        if creator_model:
            self.console_output.append(f"⚙️  Creator Model: {creator_model}")
        if critic_models:
            self.console_output.append(f"🔍 Critic Models: {', '.join(critic_models)}")
        self.console_output.append("")
        
    def record_iteration(self, iteration_num: int, creator_output: str, creator_confidence: float, 
                        creator_model: str, critics_feedback: List[Dict]):
        """Record a complete iteration."""
        self.console_output.append(f"==================== ITERATION {iteration_num} ====================")
        self.console_output.append("")
        
        # Creator output
        self.console_output.append(f"🎨 CREATOR OUTPUT ({creator_model})")
        self.console_output.append(f"Confidence: {creator_confidence}%")
        
        # Create rich panel for creator response
        panel_title = f"📝 Creator Response - Iteration {iteration_num}"
        panel_content = creator_output.strip()
        
        # Format as rich panel
        panel_lines = []
        panel_lines.append("╭─ " + panel_title + " " + "─" * (120 - len(panel_title) - 3) + "╮")
        panel_lines.append("│" + " " * 130 + "│")
        
        # Split content into lines and wrap them
        for line in panel_content.split('\n'):
            if not line.strip():
                panel_lines.append("│" + " " * 130 + "│")
            else:
                # Wrap long lines
                while len(line) > 128:
                    panel_lines.append("│  " + line[:128] + "  │")
                    line = line[128:]
                if line:
                    panel_lines.append("│  " + line.ljust(128) + "  │")
        
        panel_lines.append("│" + " " * 130 + "│")
        panel_lines.append("╰" + "─" * 130 + "╯")
        
        self.console_output.extend(panel_lines)
        self.console_output.append("")
        
        # Critics feedback
        self.console_output.append("🔍 CRITICS FEEDBACK")
        self.console_output.append("")
        
        for i, critic in enumerate(critics_feedback, 1):
            # Get persona info
            persona_name = critic.get('persona_name', f'Critic {i}')
            persona_type = critic.get('persona_type', 'vanilla')
            
            # Choose appropriate icon based on persona type
            if persona_type == 'expert':
                icon = "🧠"  # Expert persona icon
            else:
                icon = "🤖"  # Vanilla model icon
            
            self.console_output.append(f"  {icon} {persona_name}")
            self.console_output.append(f"     📊 Quality Score: {critic.get('quality_score', 'N/A')}%")
            
            strengths = critic.get('strengths', [])
            if strengths:
                self.console_output.append("     💪 Strengths:")
                for strength in strengths:
                    self.console_output.append(f"        • {strength}")
            
            improvements = critic.get('improvements', [])
            if improvements:
                self.console_output.append("     🔧 Improvements:")
                for improvement in improvements:
                    self.console_output.append(f"        • {improvement}")
            
            decision = critic.get('decision', 'Continue')
            emoji = '✅ Stop' if 'stop' in decision.lower() else '🔄 Continue'
            self.console_output.append(f"     🎯 Decision: {emoji}")
            
            # Detailed feedback panel
            feedback = critic.get('detailed_feedback', '')
            if feedback:
                panel_title = f"💬 Detailed Feedback from {persona_name}"
                panel_lines = []
                panel_lines.append("╭─ " + panel_title + " " + "─" * (120 - len(panel_title) - 3) + "╮")
                
                # Split feedback into wrapped lines
                for line in feedback.split('\n'):
                    if not line.strip():
                        panel_lines.append("│" + " " * 130 + "│")
                    else:
                        while len(line) > 128:
                            panel_lines.append("│ " + line[:128] + " │")
                            line = line[128:]
                        if line:
                            panel_lines.append("│ " + line.ljust(128) + " │")
                
                panel_lines.append("╰" + "─" * 130 + "╯")
                self.console_output.extend(panel_lines)
            
            self.console_output.append("")
        
    def record_final_results(self, final_answer: str, confidence: float, quality: float, 
                           duration: float, cost: float, execution_id: str, models_used: List[str], creator_model: str):
        """Record final results and metrics."""
        self.console_output.append("========================= FINAL RESULTS =========================")
        
        # Final answer panel
        panel_lines = []
        panel_lines.append("╭" + "─" * 78 + " 🏆 FINAL ANSWER " + "─" * 78 + "╮")
        panel_lines.append("│" + " " * 174 + "│")
        
        for line in final_answer.split('\n'):
            if not line.strip():
                panel_lines.append("│" + " " * 174 + "│")
            else:
                while len(line) > 172:
                    panel_lines.append("│  " + line[:172] + "  │")
                    line = line[172:]
                if line:
                    panel_lines.append("│  " + line.ljust(172) + "  │")
        
        panel_lines.append("│" + " " * 174 + "│")
        panel_lines.append("╰" + "─" * 174 + "╯")
        
        self.console_output.extend(panel_lines)
        self.console_output.append("")
        
        # Quality metrics
        self.console_output.append("📈 QUALITY METRICS")
        self.console_output.append(f"  🎯 Final Confidence: {confidence}%")
        self.console_output.append(f"  ⭐ Final Quality: {quality}%")
        self.console_output.append("")
        
        # Performance
        self.console_output.append("⚡ PERFORMANCE")
        self.console_output.append(f"  ⏱️  Total Duration: {duration:.1f}s")
        self.console_output.append(f"  💰 Estimated Cost: ${cost:.4f}")
        self.console_output.append("")
        
        # Execution details
        self.console_output.append("=" * 80)
        self.console_output.append(f"Execution ID: {execution_id}")
        self.console_output.append(f"Models used: {models_used}")
        if creator_model:
            self.console_output.append(f"Creator model: {creator_model}")
        self.console_output.append("")
        
    def save_conversation(self, file_path: str):
        """Save conversation to file in rich human-readable format."""
        path = Path(file_path)
        if not path.suffix:
            path = path.with_suffix('.txt')
        
        # Write the rich formatted output
        with open(path, 'w', encoding='utf-8') as f:
            for line in self.console_output:
                f.write(line + '\n')
        
        # Set secure file permissions (600 = rw-------)
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
        
        console.print(f"[yellow]💾 Conversation saved to {path}[/yellow]")
        console.print(f"[dim]Format: Rich human-readable text[/dim]")
    
    def load_conversation(self, file_path: str) -> str:
        """Load conversation from file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def replay_conversation(self, file_path: str):
        """Replay a conversation from file."""
        content = self.load_conversation(file_path)
        console.print("\n[bold blue]🔄 Conversation Replay[/bold blue]")
        console.print("=" * 50)
        console.print(content)

    # Legacy methods for compatibility - just add to output
    def record_step(self, step_type: str, data: Dict):
        """Legacy method for backward compatibility."""
        if step_type == "conversation_start":
            prompt = data.get('prompt', '')
            creator = data.get('creator', '')
            critics = data.get('critics', [])
            
            if isinstance(critics, list) and critics:
                models = [c.get('name', c) if isinstance(c, dict) else str(c) for c in critics]
            else:
                models = []
                
            self.start_recording(prompt, models, creator_model=creator, creator_persona=creator)
            
    def _sanitize_data(self, data: Dict) -> Dict:
        """Legacy method for backward compatibility."""
        return data 
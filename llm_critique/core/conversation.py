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

class ConversationManager:
    """Manages conversation recording and replay functionality with security controls."""
    
    def __init__(self):
        self.conversation_steps = []
        
    def _sanitize_data(self, data: Dict) -> Dict:
        """Remove or redact sensitive information from conversation data."""
        SENSITIVE_KEYS = ['api_key', 'secret', 'token', 'password', 'credential', 'auth']
        
        def redact_recursive(obj):
            if isinstance(obj, dict):
                sanitized = {}
                for key, value in obj.items():
                    key_lower = key.lower()
                    if any(sensitive in key_lower for sensitive in SENSITIVE_KEYS):
                        sanitized[key] = "[REDACTED]"
                    elif key.lower() == 'prompt' and isinstance(value, str) and len(value) > 1000:
                        # Truncate very long prompts to prevent data leakage
                        sanitized[key] = value[:500] + "... [TRUNCATED FOR SECURITY]"
                    else:
                        sanitized[key] = redact_recursive(value)
                return sanitized
            elif isinstance(obj, list):
                return [redact_recursive(item) for item in obj]
            else:
                return obj
        
        return redact_recursive(data)
        
    def record_step(self, step_type: str, data: Dict):
        """Record a step in the conversation with security sanitization."""
        def serialize_datetime(obj):
            """Convert datetime objects to ISO format strings."""
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: serialize_datetime(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [serialize_datetime(item) for item in obj]
            else:
                return obj
        
        # Sanitize sensitive data before storage
        sanitized_data = self._sanitize_data(data)
        
        step = {
            "timestamp": datetime.now().isoformat(),
            "type": step_type,
            "data": serialize_datetime(sanitized_data)
        }
        self.conversation_steps.append(step)
    
    def save_conversation(self, file_path: str):
        """Save conversation to file with secure permissions."""
        path = Path(file_path)
        if not path.suffix:
            path = path.with_suffix('.json')
        
        conversation_data = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "version": "1.0",
                "total_steps": len(self.conversation_steps),
                "security_notice": "This file may contain sensitive conversation data. Protect accordingly."
            },
            "steps": self.conversation_steps
        }
        
        # Create file with restrictive permissions (owner read/write only)
        with open(path, 'w') as f:
            json.dump(conversation_data, f, indent=2)
        
        # Set secure file permissions (600 = rw-------)
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
        
        console.print(f"[yellow]⚠️  Conversation saved to {path} with secure permissions[/yellow]")
        console.print(f"[dim]File permissions: 600 (owner read/write only)[/dim]")
    
    def load_conversation(self, file_path: str) -> Dict:
        """Load conversation from file."""
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def replay_conversation(self, file_path: Optional[str] = None):
        """Replay a conversation from file or current session."""
        if file_path:
            conversation_data = self.load_conversation(file_path)
            steps = conversation_data["steps"]
        else:
            steps = self.conversation_steps
        
        console.print("\n[bold blue]🔄 Conversation Replay[/bold blue]")
        console.print("=" * 50)
        
        for i, step in enumerate(steps, 1):
            self._replay_step(i, step)
    
    def _replay_step(self, step_number: int, step: Dict):
        """Replay a single step with formatted output."""
        timestamp = step.get("timestamp", "Unknown")
        step_type = step.get("type", "unknown")
        data = step.get("data", {})
        
        # Create step header
        console.print(f"\n[bold cyan]Step {step_number}: {step_type}[/bold cyan]")
        console.print(f"[dim]Time: {timestamp}[/dim]")
        
        if step_type == "execution_start":
            console.print(f"[green]Prompt:[/green] {data.get('prompt', 'N/A')[:100]}...")
            console.print(f"[green]Models:[/green] {data.get('models', [])}")
            console.print(f"[green]Resolver:[/green] {data.get('resolver', 'N/A')}")
            
        elif step_type == "model_response":
            model = data.get("model", "Unknown")
            response = data.get("response", "No response")
            console.print(f"[yellow]Model {model} Response:[/yellow]")
            console.print(Panel(response, title=f"{model} Output"))
            
        elif step_type == "critique_analysis":
            analysis = data.get("analysis", {})
            scores = data.get("scores", {})
            console.print(f"[magenta]Critique Analysis:[/magenta]")
            console.print(f"Quality Score: {scores.get('quality', 'N/A')}")
            console.print(f"Confidence: {scores.get('confidence', 'N/A')}")
            console.print(f"Analysis: {analysis.get('critique', 'N/A')}")
            
        elif step_type == "synthesis":
            answer = data.get("answer", "No answer")
            confidence = data.get("confidence", "N/A")
            console.print(f"[green]Final Synthesis:[/green]")
            console.print(f"Confidence: {confidence}")
            console.print(Panel(answer, title="Synthesized Answer"))
            
        elif step_type == "execution_complete":
            duration = data.get("duration_ms", 0)
            cost = data.get("cost", 0)
            console.print(f"[blue]Execution Complete:[/blue]")
            console.print(f"Duration: {duration/1000:.2f}s")
            console.print(f"Cost: ${cost:.4f}")
            
        elif step_type == "error":
            message = data.get("message", "Unknown error")
            error_type = data.get("type", "Error")
            console.print(f"[red]❌ {error_type}:[/red] {message}")
            
        console.print("-" * 30) 
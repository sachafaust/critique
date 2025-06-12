import structlog
import sys
import json
import os
import stat
from datetime import datetime
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.logging import RichHandler
import logging


class SecurityFilter:
    """Filter to remove sensitive information from log records."""
    
    SENSITIVE_PATTERNS = [
        'api_key', 'secret', 'token', 'password', 'credential',
        'auth', 'bearer', 'oauth', 'sk-', 'key_'
    ]
    
    def __call__(self, logger, method_name, event_dict):
        """Filter sensitive data from log entries."""
        # Filter event message
        if 'event' in event_dict:
            event_dict['event'] = self._redact_sensitive(str(event_dict['event']))
        
        # Filter all other fields
        for key, value in list(event_dict.items()):
            if self._is_sensitive_key(key):
                event_dict[key] = "[REDACTED]"
            elif isinstance(value, str):
                event_dict[key] = self._redact_sensitive(value)
        
        return event_dict
    
    def _is_sensitive_key(self, key: str) -> bool:
        """Check if a key name suggests sensitive data."""
        key_lower = key.lower()
        return any(pattern in key_lower for pattern in self.SENSITIVE_PATTERNS)
    
    def _redact_sensitive(self, text: str) -> str:
        """Redact sensitive patterns in text."""
        import re
        # Redact patterns that look like API keys
        patterns = [
            r'sk-[a-zA-Z0-9]{20,}',  # OpenAI style keys
            r'sk-ant-[a-zA-Z0-9]{20,}',  # Anthropic keys
            r'AIza[a-zA-Z0-9]{35}',  # Google API keys
            r'(?i)(api[_-]?key|secret|token|password|credential)["\s]*[:=]["\s]*[a-zA-Z0-9_-]{10,}',
        ]
        
        for pattern in patterns:
            text = re.sub(pattern, '[REDACTED_API_KEY]', text)
        
        return text


class DebugConsoleRenderer:
    """Custom renderer for debug messages to console."""
    
    def __init__(self):
        self.console = Console()
    
    def __call__(self, logger, method_name, event_dict):
        """Render debug messages in a human-readable format."""
        # Only handle debug messages specially
        if method_name == 'debug':
            event = event_dict.get('event', '')
            
            # Handle LLM request/response debug messages specially
            if 'LLM Request' in event:
                self._render_llm_request(event_dict)
            elif 'LLM Response' in event:
                self._render_llm_response(event_dict)
            else:
                # Generic debug message
                self._render_generic_debug(event_dict)
            
            # Still return the event_dict so it gets logged to JSON file
            # but the console rendering is already done above
        
        # For all messages, pass through unchanged for JSON logging
        return event_dict
    
    def _render_llm_request(self, event_dict):
        """Render LLM request in a readable format."""
        model = event_dict.get('model', 'unknown')
        messages = event_dict.get('messages', [])
        persona_name = event_dict.get('persona_name', '')
        
        if persona_name:
            self.console.print(f"[cyan]🤖 {persona_name} → {model}[/cyan]")
        else:
            self.console.print(f"[cyan]🤖 LLM Request → {model}[/cyan]")
        
        for i, msg in enumerate(messages):
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            # No truncation: show full content
            self.console.print(f"[dim]   {role.upper()}: {content}[/dim]")
    
    def _render_llm_response(self, event_dict):
        """Render LLM response in a readable format."""
        model = event_dict.get('model', 'unknown')
        response_length = event_dict.get('response_length', 0)
        response_preview = event_dict.get('response_preview', '')
        persona_name = event_dict.get('persona_name', '')
        
        if persona_name:
            self.console.print(f"[green]✅ {persona_name} ← {model} ({response_length} chars)[/green]")
        else:
            self.console.print(f"[green]✅ LLM Response ← {model} ({response_length} chars)[/green]")
        
        if response_preview:
            # No truncation: show full response
            self.console.print(f"[dim]   {response_preview}[/dim]")
    
    def _render_generic_debug(self, event_dict):
        """Render generic debug messages."""
        event = event_dict.get('event', '')
        self.console.print(f"[dim]🐛 DEBUG: {event}[/dim]")
        for key, value in event_dict.items():
            if key not in ['event', 'timestamp', 'level', 'logger']:
                self.console.print(f"[dim]   {key}: {value}[/dim]")


def setup_logging(
    level: str = "INFO",
    format_type: str = "json",
    trace_id: Optional[str] = None,
    log_dir: str = "./logs"
) -> structlog.BoundLogger:
    """Setup structured logging for both humans and machines with security controls."""
    
    # Set up log file and permissions
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    os.chmod(log_path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"{timestamp}_critique.json"
    log_file.touch()
    os.chmod(log_file, stat.S_IRUSR | stat.S_IWUSR)  # 600 = rw-------

    # Create a named logger
    py_logger = logging.getLogger('llm_critique')
    py_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    py_logger.handlers = []  # Remove any existing handlers

    # File handler for JSON logs
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    py_logger.addHandler(file_handler)

    # Stream handler for console logs
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    py_logger.addHandler(stream_handler)

    # Configure structlog to use this logger
    security_filter = SecurityFilter()
    processors = [security_filter]
    if level.upper() == "DEBUG":
        processors.extend([
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            DebugConsoleRenderer(),  # For console
            structlog.processors.JSONRenderer()  # For file
        ])
    else:
        processors.extend([
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ])

    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Print debug initialization message
    if level.upper() == "DEBUG":
        from rich.console import Console
        console = Console()
        console.print(f"[dim]🔧 Debug logging initialized: {log_file}[/dim]")

    # TEST: Write a log entry to verify file output
    test_logger = structlog.get_logger('llm_critique')
    test_logger.info('log_file_test', test_message='Log file should contain this entry')

    return structlog.get_logger('llm_critique')


def log_execution(
    logger: structlog.BoundLogger,
    execution_id: str,
    prompt: str,
    models_used: list,
    resolver_model: str,
    final_answer: str,
    confidence_score: float,
    consensus_score: float,
    model_responses: list,
    total_duration_ms: int,
    estimated_cost_usd: float,
    quality_metrics: dict
) -> None:
    """Log execution details in structured format."""
    
    log_entry = {
        "execution_id": execution_id,
        "timestamp": datetime.now().isoformat(),
        "input": {
            "prompt": prompt,
            "models_used": models_used,
            "resolver_model": resolver_model
        },
        "results": {
            "final_answer": final_answer,
            "confidence_score": confidence_score,
            "consensus_score": consensus_score,
            "model_responses": model_responses
        },
        "performance": {
            "total_duration_ms": total_duration_ms,
            "estimated_cost_usd": estimated_cost_usd
        },
        "quality_metrics": quality_metrics
    }
    
    logger.info("execution_complete", **log_entry) 
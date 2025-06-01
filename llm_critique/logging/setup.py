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


def setup_logging(
    level: str = "INFO",
    format_type: str = "json",
    trace_id: Optional[str] = None,
    log_dir: str = "./logs"
) -> structlog.BoundLogger:
    """Setup structured logging for both humans and machines with security controls."""
    
    # Create log directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Set secure directory permissions (755 = rwxr-xr-x)
    os.chmod(log_path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
    
    # Generate timestamp for log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"{timestamp}_critique.json"
    
    # Configure structlog with security filter
    security_filter = SecurityFilter()
    
    structlog.configure(
        processors=[
            security_filter,  # Add security filter first
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Create file with restrictive permissions
    log_file.touch()
    os.chmod(log_file, stat.S_IRUSR | stat.S_IWUSR)  # 600 = rw-------
    
    # Create logger
    logger = structlog.get_logger()
    
    # Add trace ID if provided
    if trace_id:
        logger = logger.bind(trace_id=trace_id)
    
    # Add component identifier
    logger = logger.bind(component="multi_llm")
    
    # Log security notice
    logger.info("logging_initialized", 
                log_file=str(log_file),
                security_filtering="enabled",
                file_permissions="600")
    
    return logger


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
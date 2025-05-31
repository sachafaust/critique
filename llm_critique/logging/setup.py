import structlog
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.logging import RichHandler


def setup_logging(
    level: str = "INFO",
    format_type: str = "json",
    trace_id: Optional[str] = None,
    log_dir: str = "./logs"
) -> structlog.BoundLogger:
    """Setup structured logging for both humans and machines."""
    
    # Create log directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for log file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_path / f"{timestamp}_critique.json"
    
    # Configure structlog
    structlog.configure(
        processors=[
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
    
    # Create console handler with Rich formatting
    console = Console()
    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=True,
        rich_tracebacks=True
    )
    
    # Create file handler for JSON logging
    file_handler = structlog.processors.JSONRenderer()
    
    # Create logger
    logger = structlog.get_logger()
    
    # Add trace ID to context if provided
    if trace_id:
        logger = logger.bind(trace_id=trace_id)
    
    # Add component identifier
    logger = logger.bind(component="multi_llm")
    
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
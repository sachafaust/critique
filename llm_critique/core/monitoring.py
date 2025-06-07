"""
Production monitoring, metrics, and health checks for LLM Critique.

This module provides structured logging, metrics collection, performance monitoring,
and health checks for robust production operation.
"""

import logging
import time
import json
import threading
from typing import Dict, Any, Optional, List
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from functools import wraps
import psutil
import os
from contextlib import contextmanager

# Metrics storage
_metrics = defaultdict(list)
_counters = defaultdict(int)
_gauges = defaultdict(float)
_histograms = defaultdict(list)
_lock = threading.Lock()

@dataclass
class RequestMetrics:
    """Metrics for a single request."""
    request_id: str
    operation: str
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    duration_seconds: float
    success: bool
    error_type: Optional[str] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

class StructuredLogger:
    """Production-ready structured logger."""
    
    def __init__(self, name: str, level: str = "INFO", format_type: str = "json"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Create handler
        handler = logging.StreamHandler()
        
        if format_type.lower() == "json":
            formatter = JsonFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        # Prevent duplicate logs
        self.logger.propagate = False
    
    def info(self, message: str, **kwargs):
        """Log info message with structured data."""
        self._log("info", message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with structured data."""
        self._log("warning", message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with structured data."""
        self._log("error", message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with structured data."""
        self._log("debug", message, **kwargs)
    
    def _log(self, level: str, message: str, **kwargs):
        """Internal logging method."""
        extra = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service": "llm_critique",
            **kwargs
        }
        getattr(self.logger, level)(message, extra=extra)

class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add extra fields
        if hasattr(record, '__dict__'):
            for key, value in record.__dict__.items():
                if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                              'filename', 'module', 'lineno', 'funcName', 'created', 
                              'msecs', 'relativeCreated', 'thread', 'threadName', 
                              'processName', 'process', 'getMessage', 'stack_info']:
                    log_entry[key] = value
        
        return json.dumps(log_entry)

class MetricsCollector:
    """Collects and manages application metrics."""
    
    @staticmethod
    def record_request_metrics(metrics: RequestMetrics):
        """Record metrics for a request."""
        with _lock:
            _metrics["requests"].append(asdict(metrics))
            _counters[f"requests_{metrics.operation}"] += 1
            _counters[f"requests_{metrics.model}"] += 1
            _histograms["request_duration"].append(metrics.duration_seconds)
            _histograms["input_tokens"].append(metrics.input_tokens)
            _histograms["output_tokens"].append(metrics.output_tokens)
            _histograms["cost"].append(metrics.cost)
            
            if not metrics.success:
                _counters["errors_total"] += 1
                if metrics.error_type:
                    _counters[f"errors_{metrics.error_type}"] += 1
    
    @staticmethod
    def increment_counter(name: str, value: int = 1, **labels):
        """Increment a counter metric."""
        with _lock:
            key = name
            if labels:
                label_str = ",".join(f"{k}={v}" for k, v in labels.items())
                key = f"{name}[{label_str}]"
            _counters[key] += value
    
    @staticmethod
    def set_gauge(name: str, value: float, **labels):
        """Set a gauge metric."""
        with _lock:
            key = name
            if labels:
                label_str = ",".join(f"{k}={v}" for k, v in labels.items())
                key = f"{name}[{label_str}]"
            _gauges[key] = value
    
    @staticmethod
    def record_histogram(name: str, value: float, **labels):
        """Record a histogram value."""
        with _lock:
            key = name
            if labels:
                label_str = ",".join(f"{k}={v}" for k, v in labels.items())
                key = f"{name}[{label_str}]"
            _histograms[key].append(value)
    
    @staticmethod
    def get_metrics_summary() -> Dict[str, Any]:
        """Get a summary of all metrics."""
        with _lock:
            summary = {
                "counters": dict(_counters),
                "gauges": dict(_gauges),
                "histograms": {}
            }
            
            # Calculate histogram statistics
            for name, values in _histograms.items():
                if values:
                    sorted_values = sorted(values)
                    n = len(sorted_values)
                    summary["histograms"][name] = {
                        "count": n,
                        "min": min(sorted_values),
                        "max": max(sorted_values),
                        "mean": sum(sorted_values) / n,
                        "p50": sorted_values[int(n * 0.5)],
                        "p90": sorted_values[int(n * 0.9)],
                        "p99": sorted_values[int(n * 0.99)] if n > 1 else sorted_values[0],
                    }
            
            return summary

def timed_operation(operation_name: str, model: str = "unknown"):
    """Decorator to time operations and collect metrics."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            request_id = f"req_{int(start_time * 1000)}"
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Extract metrics from result if available
                input_tokens = getattr(result, 'input_tokens', 0) if hasattr(result, 'input_tokens') else 0
                output_tokens = getattr(result, 'output_tokens', 0) if hasattr(result, 'output_tokens') else 0
                cost = getattr(result, 'cost', 0.0) if hasattr(result, 'cost') else 0.0
                
                metrics = RequestMetrics(
                    request_id=request_id,
                    operation=operation_name,
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost=cost,
                    duration_seconds=duration,
                    success=True
                )
                
                MetricsCollector.record_request_metrics(metrics)
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                metrics = RequestMetrics(
                    request_id=request_id,
                    operation=operation_name,
                    model=model,
                    input_tokens=0,
                    output_tokens=0,
                    cost=0.0,
                    duration_seconds=duration,
                    success=False,
                    error_type=type(e).__name__
                )
                
                MetricsCollector.record_request_metrics(metrics)
                raise
        
        return wrapper
    return decorator

@contextmanager
def operation_context(operation_name: str, **metadata):
    """Context manager for tracking operations."""
    start_time = time.time()
    request_id = f"req_{int(start_time * 1000)}"
    
    logger = StructuredLogger(__name__)
    logger.info(f"Starting operation: {operation_name}", 
                request_id=request_id, **metadata)
    
    try:
        yield request_id
        duration = time.time() - start_time
        logger.info(f"Completed operation: {operation_name}", 
                   request_id=request_id, duration=duration, **metadata)
        MetricsCollector.increment_counter("operations_completed", operation=operation_name)
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Failed operation: {operation_name}", 
                    request_id=request_id, duration=duration, 
                    error_type=type(e).__name__, error_message=str(e), 
                    **metadata)
        MetricsCollector.increment_counter("operations_failed", operation=operation_name)
        raise

class HealthChecker:
    """Health check functionality."""
    
    @staticmethod
    def get_system_health() -> Dict[str, Any]:
        """Get system health metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                "status": "healthy",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "system": {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available_gb": memory.available / (1024**3),
                    "disk_percent": disk.percent,
                    "disk_free_gb": disk.free / (1024**3),
                },
                "process": {
                    "pid": os.getpid(),
                    "uptime_seconds": time.time() - psutil.Process().create_time(),
                    "memory_rss_mb": psutil.Process().memory_info().rss / (1024**2),
                    "open_files": len(psutil.Process().open_files()),
                }
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
    
    @staticmethod
    def check_api_connectivity() -> Dict[str, Any]:
        """Check API connectivity for configured providers."""
        results = {}
        
        # Check OpenAI
        if os.getenv("OPENAI_API_KEY"):
            try:
                import openai
                client = openai.OpenAI()
                # Simple test call
                response = client.models.list()
                results["openai"] = {"status": "healthy", "models_count": len(response.data)}
            except Exception as e:
                results["openai"] = {"status": "unhealthy", "error": str(e)}
        else:
            results["openai"] = {"status": "not_configured"}
        
        # Check Anthropic
        if os.getenv("ANTHROPIC_API_KEY"):
            try:
                import anthropic
                client = anthropic.Anthropic()
                # Test call would go here - for now just check client creation
                results["anthropic"] = {"status": "configured"}
            except Exception as e:
                results["anthropic"] = {"status": "unhealthy", "error": str(e)}
        else:
            results["anthropic"] = {"status": "not_configured"}
        
        # Check Google
        if os.getenv("GOOGLE_API_KEY"):
            try:
                import google.generativeai as genai
                genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
                results["google"] = {"status": "configured"}
            except Exception as e:
                results["google"] = {"status": "unhealthy", "error": str(e)}
        else:
            results["google"] = {"status": "not_configured"}
        
        return results

# Global logger instance
logger = StructuredLogger(__name__)

def setup_monitoring(config):
    """Setup monitoring with configuration."""
    global logger
    logger = StructuredLogger(
        "llm_critique", 
        level=config.log_level,
        format_type=config.log_format
    )
    
    logger.info("Monitoring initialized", 
                log_level=config.log_level,
                log_format=config.log_format,
                metrics_enabled=config.enable_metrics) 
import os
import yaml
from typing import List, Optional
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings
import logging

class Config(BaseSettings):
    """Configuration settings for LLM Critique application."""
    
    # Model Configuration
    default_models: List[str] = Field(
        default=["gpt-4o-mini", "claude-3-haiku"],
        description="Default models to use for critique",
        env="LLM_CRITIQUE_DEFAULT_MODELS"
    )
    
    # Production Limits
    max_input_tokens: int = Field(
        default=8000,
        description="Maximum input tokens per request",
        env="LLM_CRITIQUE_MAX_INPUT_TOKENS"
    )
    
    max_output_tokens: int = Field(
        default=2000,
        description="Maximum output tokens per request", 
        env="LLM_CRITIQUE_MAX_OUTPUT_TOKENS"
    )
    
    max_cost_per_request: float = Field(
        default=1.0,
        description="Maximum cost per request in USD",
        env="LLM_CRITIQUE_MAX_COST_PER_REQUEST"
    )
    
    max_file_size_mb: int = Field(
        default=10,
        description="Maximum file size in MB",
        env="LLM_CRITIQUE_MAX_FILE_SIZE_MB"
    )
    
    # API Timeouts
    api_timeout_seconds: int = Field(
        default=120,
        description="API request timeout in seconds",
        env="LLM_CRITIQUE_API_TIMEOUT"
    )
    
    max_retries: int = Field(
        default=3,
        description="Maximum number of API retries",
        env="LLM_CRITIQUE_MAX_RETRIES"
    )
    
    retry_delay_seconds: float = Field(
        default=1.0,
        description="Delay between retries in seconds",
        env="LLM_CRITIQUE_RETRY_DELAY"
    )
    
    # Rate Limiting
    rate_limit_requests_per_minute: int = Field(
        default=60,
        description="Rate limit requests per minute per user",
        env="LLM_CRITIQUE_RATE_LIMIT_RPM"
    )
    
    # Logging Configuration
    log_level: str = Field(
        default="INFO", 
        description="Logging level",
        env="LLM_CRITIQUE_LOG_LEVEL"
    )
    log_format: str = Field(
        default="json", 
        description="Log format: json or text",
        env="LLM_CRITIQUE_LOG_FORMAT"
    )
    
    # Monitoring
    enable_metrics: bool = Field(
        default=True,
        description="Enable metrics collection",
        env="LLM_CRITIQUE_ENABLE_METRICS"
    )
    
    metrics_port: int = Field(
        default=8080,
        description="Metrics server port",
        env="LLM_CRITIQUE_METRICS_PORT"
    )
    
    # Health Check
    health_check_timeout: float = Field(
        default=5.0,
        description="Health check timeout in seconds",
        env="LLM_CRITIQUE_HEALTH_TIMEOUT"
    )
    
    # Synthesis Configuration
    confidence_threshold: float = Field(
        default=0.8,
        description="Confidence threshold for consensus",
        env="LLM_CRITIQUE_CONFIDENCE_THRESHOLD"
    )

    @validator('default_models', pre=True)
    def parse_default_models(cls, v):
        if isinstance(v, str):
            return [model.strip() for model in v.split(',')]
        return v

    @validator('log_level')
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'log_level must be one of: {valid_levels}')
        return v.upper()

    @validator('log_format') 
    def validate_log_format(cls, v):
        valid_formats = ['json', 'text']
        if v.lower() not in valid_formats:
            raise ValueError(f'log_format must be one of: {valid_formats}')
        return v.lower()

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
        case_sensitive = False


def get_available_models(config: Config) -> List[str]:
    """Return list of models based on available API keys."""
    available_models = []
    
    if config.openai_api_key:
        available_models.append("gpt-4")
    
    if config.anthropic_api_key:
        available_models.append("claude-3-sonnet")
    
    if config.google_api_key:
        available_models.append("gemini-pro")
    
    return available_models


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from file and environment variables."""
    try:
        if config_path:
            # Load from specific config file if provided
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            return Config(**config_data)
        else:
            # Load from environment variables and default .env file
            return Config()
    except Exception as e:
        # Fallback to defaults with warning
        import warnings
        warnings.warn(f"Failed to load config: {e}. Using defaults.")
        return Config()


def get_model_settings(config: Config, model_name: str) -> dict:
    """Get model-specific settings from config."""
    # Default settings
    default_settings = {
        "temperature": 0.7,
        "max_tokens": 2000
    }
    
    # Load from config.yaml if available
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
            if "models" in yaml_config and model_name in yaml_config["models"]:
                return yaml_config["models"][model_name]
    
    return default_settings 
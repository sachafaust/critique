import os
import yaml
from typing import List, Optional
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings

class Config(BaseModel):
    """Configuration model for LLM Critique."""
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")
    google_api_key: Optional[str] = Field(None, env="GOOGLE_API_KEY")
    
    # Model configuration
    default_models: List[str] = Field(
        ["gpt-4o", "claude-4-sonnet", "gemini-2.5-pro"],
        env="LLM_CRITIQUE_DEFAULT_MODELS"
    )
    default_creator: str = Field("auto", env="LLM_CRITIQUE_DEFAULT_CREATOR")
    
    # Execution configuration
    max_iterations: int = Field(2, env="LLM_CRITIQUE_MAX_ITERATIONS")
    confidence_threshold: float = Field(0.8, env="LLM_CRITIQUE_CONFIDENCE_THRESHOLD")
    timeout_seconds: int = Field(30, env="LLM_CRITIQUE_TIMEOUT")
    
    # Logging configuration
    log_level: str = Field("INFO", env="LLM_CRITIQUE_LOG_LEVEL")
    log_format: str = Field("json", env="LLM_CRITIQUE_LOG_FORMAT")
    log_directory: str = Field("./logs", env="LLM_CRITIQUE_LOG_DIR")
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
    
    @validator('default_models', pre=True)
    def parse_default_models(cls, v):
        if isinstance(v, str):
            return [model.strip() for model in v.split(',')]
        return v
    
    @validator('log_level')
    def validate_log_level(cls, v):
        allowed_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in allowed_levels:
            raise ValueError(f'log_level must be one of {allowed_levels}')
        return v.upper()
    
    @validator('log_format')
    def validate_log_format(cls, v):
        allowed_formats = ['json', 'structured', 'simple']
        if v.lower() not in allowed_formats:
            raise ValueError(f'log_format must be one of {allowed_formats}')
        return v.lower()


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
    # First load from environment variables
    config = Config()
    
    # Then load from YAML if provided
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
            
            # Update config with YAML values
            for key, value in yaml_config.items():
                if hasattr(config, key):
                    setattr(config, key, value)
    
    return config


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
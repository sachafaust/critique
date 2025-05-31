from typing import Dict, List, Optional, Any
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models import BaseChatModel
import os
from dotenv import load_dotenv

load_dotenv()

class LLMClient:
    """Manages LLM clients and parallel execution."""
    
    def __init__(self, config):
        self.config = config
        self.clients: Dict[str, BaseChatModel] = {}
        self._initialize_clients()
    
    def _initialize_clients(self) -> None:
        """Initialize available LLM clients based on API keys.
        
        Note: OpenAI's o1 and o3 reasoning models don't support the temperature parameter,
        so they are initialized without it to avoid API errors.
        """
        # OpenAI
        if os.getenv("OPENAI_API_KEY"):
            # GPT-4 series
            self.clients["gpt-4"] = ChatOpenAI(
                model="gpt-4",
                temperature=0.7,
                api_key=os.getenv("OPENAI_API_KEY")
            )
            self.clients["gpt-4o"] = ChatOpenAI(
                model="gpt-4o",
                temperature=0.7,
                api_key=os.getenv("OPENAI_API_KEY")
            )
            self.clients["gpt-4o-mini"] = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.7,
                api_key=os.getenv("OPENAI_API_KEY")
            )
            
            # O1 series (reasoning models - no temperature support)
            self.clients["o1"] = ChatOpenAI(
                model="o1",
                api_key=os.getenv("OPENAI_API_KEY")
            )
            self.clients["o1-mini"] = ChatOpenAI(
                model="o1-mini",
                api_key=os.getenv("OPENAI_API_KEY")
            )
            
            # O3 series (latest reasoning models - no temperature support)
            self.clients["o3"] = ChatOpenAI(
                model="o3",
                api_key=os.getenv("OPENAI_API_KEY")
            )
            self.clients["o3-mini"] = ChatOpenAI(
                model="o3-mini",
                api_key=os.getenv("OPENAI_API_KEY")
            )
            
            # Legacy models
            self.clients["gpt-3.5-turbo"] = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.7,
                api_key=os.getenv("OPENAI_API_KEY")
            )
        
        # Anthropic
        if os.getenv("ANTHROPIC_API_KEY"):
            # Claude 4 series (latest)
            self.clients["claude-4-opus"] = ChatAnthropic(
                model="claude-opus-4-20250514",
                temperature=0.7,
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
            self.clients["claude-4-sonnet"] = ChatAnthropic(
                model="claude-sonnet-4-20250514",
                temperature=0.7,
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
            
            # Claude 3.7 series
            self.clients["claude-3.7-sonnet"] = ChatAnthropic(
                model="claude-3-7-sonnet-20250219",
                temperature=0.7,
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
            
            # Claude 3.5 series
            self.clients["claude-3.5-sonnet"] = ChatAnthropic(
                model="claude-3-5-sonnet-20241022",
                temperature=0.7,
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
            self.clients["claude-3.5-haiku"] = ChatAnthropic(
                model="claude-3-5-haiku-20241022",
                temperature=0.7,
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
            
            # Claude 3 series (legacy compatibility)
            self.clients["claude-3-opus"] = ChatAnthropic(
                model="claude-3-opus-20240229",
                temperature=0.7,
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
            self.clients["claude-3-sonnet"] = ChatAnthropic(
                model="claude-3-sonnet-20240229",
                temperature=0.7,
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
            self.clients["claude-3-haiku"] = ChatAnthropic(
                model="claude-3-haiku-20240307",
                temperature=0.7,
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
        
        # Google
        if os.getenv("GOOGLE_API_KEY"):
            # Gemini 2.5 series (latest)
            self.clients["gemini-2.5-pro"] = ChatGoogleGenerativeAI(
                model="gemini-2.5-pro",
                temperature=0.7,
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
            self.clients["gemini-2.5-flash"] = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.7,
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
            
            # Gemini 2.0 series  
            self.clients["gemini-2.0-flash"] = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=0.7,
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
            
            # Legacy gemini-pro (maps to 2.0-flash for compatibility)
            self.clients["gemini-pro"] = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=0.7,
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
    
    def get_models(self, model_names: List[str]) -> Dict[str, BaseChatModel]:
        """Get a dictionary of models by name."""
        return {
            name: self.clients[name]
            for name in model_names
            if name in self.clients
        }
    
    def get_model(self, model_name: str) -> BaseChatModel:
        """Get a single model by name."""
        if model_name not in self.clients:
            raise ValueError(f"Model {model_name} not found")
        return self.clients[model_name]
    
    def get_available_models(self) -> List[str]:
        """Get list of available model names."""
        return list(self.clients.keys())
    
    def estimate_cost(self, model: str, tokens: int) -> float:
        """Estimate cost based on model and token count."""
        # Approximate costs per 1K tokens (as of 2024)
        costs = {
            # GPT-4 series
            "gpt-4": 0.03,  # $0.03 per 1K tokens
            "gpt-4o": 0.0025,  # $0.0025 per 1K tokens
            "gpt-4o-mini": 0.00015,  # $0.00015 per 1K tokens
            
            # O1 series (reasoning models)
            "o1": 0.015,  # $0.015 per 1K tokens
            "o1-mini": 0.003,  # $0.003 per 1K tokens
            
            # O3 series (latest reasoning models) - estimated pricing
            "o3": 0.020,  # $0.020 per 1K tokens (estimated)
            "o3-mini": 0.005,  # $0.005 per 1K tokens (estimated)
            
            # Legacy models
            "gpt-3.5-turbo": 0.002,  # $0.002 per 1K tokens
            
            # Anthropic models
            "claude-4-opus": 0.015,  # $0.015 per 1K tokens (input, $0.075 output)
            "claude-4-sonnet": 0.003,  # $0.003 per 1K tokens (input, $0.015 output)
            "claude-3.7-sonnet": 0.003,  # $0.003 per 1K tokens (input, $0.015 output)
            "claude-3.5-sonnet": 0.003,  # $0.003 per 1K tokens (input, $0.015 output)
            "claude-3.5-haiku": 0.0008,  # $0.0008 per 1K tokens (input, $0.004 output)
            "claude-3-opus": 0.015,  # $0.015 per 1K tokens (input, $0.075 output)
            "claude-3-sonnet": 0.003,  # $0.003 per 1K tokens (input, $0.015 output)
            "claude-3-haiku": 0.00025,  # $0.00025 per 1K tokens (input, $0.00125 output)
            
            # Google Gemini models
            "gemini-2.5-pro": 0.00125,  # $0.00125 per 1K tokens 
            "gemini-2.5-flash": 0.0005,  # $0.0005 per 1K tokens
            "gemini-2.0-flash": 0.001,  # $0.001 per 1K tokens
            "gemini-pro": 0.001  # $0.001 per 1K tokens (legacy, maps to 2.0-flash)
        }
        
        return (tokens / 1000) * costs.get(model, 0.0) 
"""
Unified Persona Management System for LLM Critique Tool

This module implements a unified approach to handling both expert personas 
(rich YAML configurations with personality, principles, etc.) and vanilla 
model personas (simple model names with minimal context).

Key design principles:
- Single code path for both expert and vanilla personas
- Consistent interface and output format
- Easy extensibility for new expert personas
- Cost-efficient token usage
"""

import os
import yaml
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from enum import Enum

logger = logging.getLogger(__name__)


class PersonaType(Enum):
    """Types of personas supported by the system."""
    EXPERT = "expert"      # Rich YAML-configured personas with full context
    VANILLA = "vanilla"    # Simple model names with minimal context


@dataclass
class PersonaConfig:
    """
    Unified persona configuration that works for both expert and vanilla personas.
    
    Expert personas have rich context loaded from YAML files.
    Vanilla personas are auto-generated with minimal context from model names.
    """
    name: str
    persona_type: PersonaType
    description: str = ""
    
    # Rich context (expert personas only)
    core_principles: List[str] = field(default_factory=list)
    critique_style: Dict[str, str] = field(default_factory=dict)
    key_questions: List[str] = field(default_factory=list)
    decision_frameworks: List[str] = field(default_factory=list)
    language_patterns: Dict[str, List[str]] = field(default_factory=dict)
    red_flags: List[str] = field(default_factory=list)
    success_indicators: List[str] = field(default_factory=list)
    expertise_domains: List[str] = field(default_factory=list)
    
    # Model configuration (both types)
    preferred_model: str = ""
    temperature: float = 0.1
    max_tokens: int = 1500
    
    # Metadata
    file_path: Optional[str] = None
    is_cached: bool = False

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.name:
            raise ValueError("Persona name cannot be empty")
        
        if self.persona_type == PersonaType.EXPERT:
            if not self.core_principles:
                logger.warning(f"Expert persona '{self.name}' has no core principles defined")
        
        # Ensure preferred_model is set
        if not self.preferred_model:
            if self.persona_type == PersonaType.VANILLA:
                self.preferred_model = self.name  # Use persona name as model name
            else:
                logger.warning(f"Persona '{self.name}' has no preferred model set")

    def get_prompt_context_size_estimate(self) -> int:
        """Estimate token size for this persona's context."""
        if self.persona_type == PersonaType.VANILLA:
            return 50  # Minimal context overhead
        
        # Expert persona context estimation
        context_text = (
            self.description + 
            " ".join(self.core_principles) +
            " ".join(self.key_questions) +
            " ".join(f"{k}: {v}" for k, v in self.critique_style.items())
        )
        # Rough estimate: 4 characters per token
        return len(context_text) // 4


@dataclass 
class CritiqueResult:
    """Unified result format for both expert and vanilla persona critiques."""
    persona_name: str
    persona_type: PersonaType
    model_used: str
    quality_score: float
    key_insights: List[str]
    recommendations: List[str]
    confidence_level: float
    critique_text: str
    
    # Expert persona specific (empty for vanilla)
    red_flags_identified: List[str] = field(default_factory=list)
    expertise_match: float = 0.0
    authentic_language_used: bool = False
    
    # Performance metrics
    execution_time_ms: float = 0.0
    token_count: int = 0
    estimated_cost: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "persona_name": self.persona_name,
            "persona_type": self.persona_type.value,
            "model_used": self.model_used,
            "quality_score": self.quality_score,
            "key_insights": self.key_insights,
            "recommendations": self.recommendations,
            "confidence_level": self.confidence_level,
            "critique_text": self.critique_text,
            "red_flags_identified": self.red_flags_identified,
            "expertise_match": self.expertise_match,
            "authentic_language_used": self.authentic_language_used,
            "execution_time_ms": self.execution_time_ms,
            "token_count": self.token_count,
            "estimated_cost": self.estimated_cost
        }


class UnifiedPersonaManager:
    """
    Manages both expert personas (YAML configs) and vanilla model personas.
    
    Provides a unified interface for loading, caching, and managing all persona types.
    """
    
    def __init__(self, personas_dir: str = "llm_critique/personas/", config_obj=None):
        self.personas_dir = Path(personas_dir)
        self.config_obj = config_obj
        self._expert_persona_cache: Dict[str, PersonaConfig] = {}
        self._vanilla_persona_cache: Dict[str, PersonaConfig] = {}
        self._available_models: List[str] = []
        
        # Initialize available models from config or environment
        self._initialize_available_models()
        
        # Discover expert personas
        self._discover_expert_personas()
        
        logger.info(f"Initialized persona manager: {len(self._expert_persona_cache)} expert personas, "
                   f"{len(self._available_models)} vanilla models available")

    def _initialize_available_models(self):
        """Initialize list of available models based on API keys."""
        # Import here to avoid circular imports
        from .models import LLMClient
        
        available = []
        
        if os.getenv("OPENAI_API_KEY"):
            available.extend([
                "gpt-4", "gpt-4o", "gpt-4o-mini",
                "o1", "o1-mini", "o3", "o3-mini",
                "gpt-3.5-turbo"
            ])
        
        if os.getenv("ANTHROPIC_API_KEY"):
            available.extend([
                "claude-4-opus", "claude-4-sonnet",
                "claude-3.7-sonnet", "claude-3.5-sonnet", "claude-3.5-haiku",
                "claude-3-opus", "claude-3-sonnet", "claude-3-haiku"
            ])
        
        if os.getenv("GOOGLE_API_KEY"):
            available.extend([
                "gemini-2.5-flash-preview-05-20", "gemini-2.5-pro-preview-05-06",
                "gemini-2.0-flash", "gemini-2.0-flash-lite",
                "gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-1.5-pro"
            ])
        
        self._available_models = available

    def _discover_expert_personas(self):
        """Discover available expert persona YAML files."""
        if not self.personas_dir.exists():
            logger.warning(f"Personas directory {self.personas_dir} does not exist")
            return
        
        for yaml_file in self.personas_dir.glob("*.yaml"):
            try:
                # Extract persona name from filename (without .yaml extension)
                persona_name = yaml_file.stem
                # Cache the file path for lazy loading
                self._expert_persona_cache[persona_name] = None  # Placeholder for lazy loading
                logger.debug(f"Discovered expert persona: {persona_name}")
            except Exception as e:
                logger.warning(f"Error discovering persona {yaml_file}: {e}")

    def load_expert_persona(self, name: str, suppress_warnings: bool = False) -> PersonaConfig:
        """Load rich persona from YAML file with caching."""
        # Check cache first - only return if persona is actually loaded (not None)
        if name in self._expert_persona_cache and self._expert_persona_cache[name] is not None:
            logger.debug(f"Returning cached expert persona: {name}")
            persona = self._expert_persona_cache[name]
            # Still need to check model availability for cached personas
            if persona.preferred_model and persona.preferred_model not in self._available_models:
                if not suppress_warnings:
                    logger.warning(f"Expert persona '{name}' prefers unavailable model '{persona.preferred_model}', "
                                  f"using fallback")
                persona.preferred_model = self._get_fallback_model()
            return persona
        
        # Find YAML file
        yaml_file = self.personas_dir / f"{name}.yaml"
        
        if not yaml_file.exists():
            raise ValueError(f"Expert persona '{name}' not found in {self.personas_dir}")
        
        try:
            with open(yaml_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            # Validate required fields
            if not data.get('name'):
                data['name'] = name
            
            # Create PersonaConfig from YAML data
            persona = PersonaConfig(
                name=data['name'],
                persona_type=PersonaType.EXPERT,
                description=data.get('description', ''),
                core_principles=data.get('core_principles', []),
                critique_style=data.get('critique_style', {}),
                key_questions=data.get('key_questions', []),
                decision_frameworks=data.get('decision_frameworks', []),
                language_patterns=data.get('language_patterns', {}),
                red_flags=data.get('red_flags', []),
                success_indicators=data.get('success_indicators', []),
                expertise_domains=data.get('expertise_domains', []),
                # LLM settings set to sensible defaults - not read from YAML
                preferred_model=self._get_default_expert_model(),  # Set default expert model
                temperature=0.2,     # Balanced creativity for expert personas
                max_tokens=1500,     # Standard length for detailed analysis
                file_path=str(yaml_file),
                is_cached=True
            )
            
            # Validate that preferred model is available
            if persona.preferred_model and persona.preferred_model not in self._available_models:
                if not suppress_warnings:
                    logger.warning(f"Expert persona '{name}' prefers unavailable model '{persona.preferred_model}', "
                                  f"using fallback")
                persona.preferred_model = self._get_fallback_model()
            
            # Cache the loaded persona
            self._expert_persona_cache[name] = persona
            logger.debug(f"Loaded and cached expert persona: {name}")
            
            return persona
            
        except Exception as e:
            logger.error(f"Error loading expert persona '{name}' from {yaml_file}: {e}")
            raise ValueError(f"Failed to load expert persona '{name}': {e}")

    def create_vanilla_persona(self, model_name: str) -> PersonaConfig:
        """Create minimal persona for vanilla model with caching."""
        # Check cache first
        if model_name in self._vanilla_persona_cache:
            logger.debug(f"Returning cached vanilla persona: {model_name}")
            return self._vanilla_persona_cache[model_name]
        
        # Validate model is available
        if model_name not in self._available_models:
            raise ValueError(f"Model '{model_name}' is not available. Available models: {self._available_models}")
        
        # Create vanilla persona config
        persona = PersonaConfig(
            name=model_name,
            persona_type=PersonaType.VANILLA,
            description=f"AI model {model_name} providing general critique",
            preferred_model=model_name,
            temperature=0.1,  # Conservative for vanilla models
            max_tokens=1500,
            is_cached=True
        )
        
        # Cache the persona
        self._vanilla_persona_cache[model_name] = persona
        logger.debug(f"Created and cached vanilla persona: {model_name}")
        
        return persona

    def get_persona(self, name: str, model_override: Optional[str] = None) -> PersonaConfig:
        """
        Unified getter - returns expert or vanilla persona.
        
        Args:
            name: Persona name or model name
            model_override: Optional model to override the persona's preferred model
        """
        # Check if it's an expert persona first
        if name in self._expert_persona_cache or self._is_expert_persona_available(name):
            persona = self.load_expert_persona(name, suppress_warnings=bool(model_override))
            if model_override and model_override in self._available_models:
                # Clone persona with model override
                import copy
                persona = copy.deepcopy(persona)
                persona.preferred_model = model_override
                logger.debug(f"Override model for expert persona '{name}': {model_override}")
            return persona
        
        # Check if it's a valid model name for vanilla persona
        elif name in self._available_models:
            return self.create_vanilla_persona(name)
        
        else:
            raise ValueError(f"Unknown persona or model: '{name}'. "
                           f"Available expert personas: {list(self.get_expert_personas().keys())}, "
                           f"Available models: {self._available_models}")

    def _is_expert_persona_available(self, name: str) -> bool:
        """Check if an expert persona YAML file exists."""
        if not self.personas_dir.exists():
            return False
        
        yaml_file = self.personas_dir / f"{name}.yaml"
        return yaml_file.exists()

    def _get_fallback_model(self) -> str:
        """Get fallback model when preferred model is unavailable."""
        if self.config_obj and hasattr(self.config_obj, 'fallback_model'):
            fallback = self.config_obj.fallback_model
            if fallback in self._available_models:
                return fallback
        
        # Default fallback priority
        fallback_priority = ["claude-4-sonnet", "gpt-4o", "claude-3.5-sonnet", "gpt-4o-mini"]
        for model in fallback_priority:
            if model in self._available_models:
                return model
        
        # Return first available model if no priority matches
        return self._available_models[0] if self._available_models else ""

    def _get_default_expert_model(self) -> str:
        """Get default model for expert personas."""
        if self.config_obj and hasattr(self.config_obj, 'default_expert_model'):
            default_model = self.config_obj.default_expert_model
            if default_model in self._available_models:
                return default_model
        
        # Default fallback priority for expert personas (favor reasoning models)
        default_priority = ["claude-4-sonnet", "gpt-4o", "claude-3.5-sonnet", "gpt-4o-mini"]
        for model in default_priority:
            if model in self._available_models:
                return model
        
        # Return first available model if no priority matches
        return self._available_models[0] if self._available_models else ""

    def list_available_personas(self) -> Dict[str, List[str]]:
        """Returns both expert personas and vanilla models."""
        expert_personas = list(self.get_expert_personas().keys())
        return {
            "expert_personas": expert_personas,
            "vanilla_models": self._available_models,
            "total_personas": len(expert_personas),
            "total_models": len(self._available_models)
        }

    def get_expert_personas(self) -> Dict[str, Optional[PersonaConfig]]:
        """Get all discovered expert personas (may contain None for unloaded)."""
        return self._expert_persona_cache.copy()

    def get_available_models(self) -> List[str]:
        """Get all available vanilla models."""
        return self._available_models.copy()

    def get_persona_info(self, name: str) -> Dict[str, Any]:
        """Get detailed information about a persona for --persona-info command."""
        try:
            persona = self.get_persona(name)
            
            info = {
                "name": persona.name,
                "type": persona.persona_type.value,
                "description": persona.description,
                "preferred_model": persona.preferred_model,
                "temperature": persona.temperature,
                "max_tokens": persona.max_tokens,
                "context_size_estimate": persona.get_prompt_context_size_estimate()
            }
            
            if persona.persona_type == PersonaType.EXPERT:
                info.update({
                    "core_principles": persona.core_principles,
                    "critique_style": persona.critique_style,
                    "key_questions": persona.key_questions,
                    "decision_frameworks": persona.decision_frameworks,
                    "language_patterns": persona.language_patterns,
                    "red_flags": persona.red_flags,
                    "success_indicators": persona.success_indicators,
                    "expertise_domains": persona.expertise_domains,
                    "file_path": persona.file_path
                })
            
            return info
            
        except Exception as e:
            raise ValueError(f"Could not get info for persona '{name}': {e}")

    def validate_persona_combination(self, persona_names: List[str], global_model_override: Optional[str] = None) -> Dict[str, Any]:
        """Validate a combination of personas and return analysis."""
        validation_result = {
            "valid": True,
            "personas": [],
            "errors": [],
            "warnings": [],
            "cost_estimate": 0.0,
            "total_context_tokens": 0
        }
        
        for name in persona_names:
            try:
                persona = self.get_persona(name, model_override=global_model_override)
                context_tokens = persona.get_prompt_context_size_estimate()
                
                validation_result["personas"].append({
                    "name": persona.name,
                    "type": persona.persona_type.value,
                    "model": persona.preferred_model,
                    "context_tokens": context_tokens
                })
                validation_result["total_context_tokens"] += context_tokens
                
            except Exception as e:
                validation_result["valid"] = False
                validation_result["errors"].append(f"Persona '{name}': {e}")
        
        # Add warnings for potential issues
        if validation_result["total_context_tokens"] > 10000:
            validation_result["warnings"].append(
                "High token usage detected - consider using fewer expert personas or shorter prompts"
            )
        
        expert_count = sum(1 for p in validation_result["personas"] if p["type"] == "expert")
        if expert_count > 3:
            validation_result["warnings"].append(
                "Many expert personas may lead to conflicting advice - consider focusing on 2-3 key experts"
            )
        
        return validation_result


class UnifiedCritic:
    """Single critic class that handles both expert and vanilla personas."""
    
    def __init__(self, persona: PersonaConfig, llm_client=None):
        self.persona = persona
        self.llm_client = llm_client
        self.model = persona.preferred_model
        
        if not self.model:
            raise ValueError(f"Persona '{persona.name}' has no preferred model set")

    def generate_critique_prompt(self, content: str, context: str = "") -> str:
        """Generate prompt based on persona type."""
        if self.persona.persona_type == PersonaType.EXPERT:
            return self._generate_expert_prompt(content, context)
        else:
            return self._generate_vanilla_prompt(content, context)

    def _generate_expert_prompt(self, content: str, context: str) -> str:
        """Rich prompt with full persona context for expert personas."""
        # Build language patterns section
        language_patterns_text = ""
        if self.persona.language_patterns:
            patterns = []
            for category, phrases in self.persona.language_patterns.items():
                if phrases:
                    patterns.append(f"  {category.title()}: {', '.join(phrases[:3])}")  # Limit to 3 for brevity
            if patterns:
                language_patterns_text = f"\n## CHARACTERISTIC LANGUAGE PATTERNS:\n" + "\n".join(patterns)

        prompt = f"""# EXPERT PERSONA CRITIQUE: {self.persona.name}

## PERSONA IDENTITY
{self.persona.description}

## CORE PRINCIPLES TO APPLY
{chr(10).join([f"• {p}" for p in self.persona.core_principles[:5]])}  # Limit for token efficiency

## KEY QUESTIONS TO EXPLORE
{chr(10).join([f"• {q}" for q in self.persona.key_questions[:5]])}

## YOUR CRITIQUE STYLE
• Approach: {self.persona.critique_style.get('approach', 'Analytical and thorough')}
• Focus: {self.persona.critique_style.get('focus_areas', 'Quality and improvement areas')}
• Communication: {self.persona.critique_style.get('communication_tone', 'Professional and constructive')}

## DECISION FRAMEWORKS TO USE
{chr(10).join([f"• {f}" for f in self.persona.decision_frameworks[:3]])}
{language_patterns_text}

**CRITICAL INSTRUCTION**: Think and respond exactly as {self.persona.name} would. Use their characteristic language patterns, apply their known principles, and provide critique that authentically reflects their perspective and expertise.

---

**Content to Analyze:**
{content}

**Additional Context:**
{context}

---

**Provide your critique in character as {self.persona.name}. Structure your response as:**

**Quality Assessment:** [Score 1-100 and brief reasoning]

**Key Insights:** [2-3 main observations from your perspective]

**Red Flags:** [Issues that concern you based on your principles]

**Recommendations:** [Specific actions you would suggest]

**Confidence:** [How confident you are in this analysis, 1-100]
"""
        return prompt

    def _generate_vanilla_prompt(self, content: str, context: str) -> str:
        """Simple prompt for vanilla model critique."""
        prompt = f"""# AI MODEL CRITIQUE: {self.persona.name}

You are providing a professional critique and analysis of the following content. Focus on quality, clarity, accuracy, and improvement opportunities.

**Content to Analyze:**
{content}

**Context:**
{context}

**Provide your critique with:**

**Quality Assessment:** [Score 1-100 and reasoning]

**Key Insights:** [Main observations and analysis]

**Recommendations:** [Specific improvement suggestions]

**Confidence:** [Confidence level 1-100]
"""
        return prompt

    async def execute_critique(self, content: str, context: str = "") -> CritiqueResult:
        """Execute critique using the persona configuration."""
        import time
        import re
        from .models import LLMClient
        
        if not self.llm_client:
            raise ValueError("LLM client not provided to critic")
        
        start_time = time.time()
        
        try:
            # Generate persona-appropriate prompt
            prompt = self.generate_critique_prompt(content, context)
            
            # Get the actual LangChain model
            model = self.llm_client.get_model(self.model)
            
            # Create messages for the model
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            # Execute critique using the LangChain model
            response = await model.ainvoke(messages)
            response_text = response.content.strip()
            
            execution_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Parse the response to extract structured data
            parsed_result = self._parse_critique_response(response_text, execution_time)
            
            return parsed_result
            
        except Exception as e:
            logger.error(f"Error executing critique for persona '{self.persona.name}': {e}")
            # Return error result
            return CritiqueResult(
                persona_name=self.persona.name,
                persona_type=self.persona.persona_type,
                model_used=self.model,
                quality_score=0.0,
                key_insights=[f"Error executing critique: {str(e)}"],
                recommendations=["Unable to provide recommendations due to error"],
                confidence_level=0.0,
                critique_text=f"Error: {str(e)}",
                execution_time_ms=(time.time() - start_time) * 1000
            )

    def _parse_critique_response(self, response: str, execution_time_ms: float) -> CritiqueResult:
        """Parse the LLM response into structured CritiqueResult."""
        import re
        
        # Initialize default values
        quality_score = 0.0
        key_insights = []
        recommendations = []
        confidence_level = 0.0
        red_flags = []
        
        # Extract quality score
        quality_match = re.search(r'quality.*?(\d+)', response, re.IGNORECASE)
        if quality_match:
            quality_score = float(quality_match.group(1)) / 100.0  # Normalize to 0-1
        
        # Extract confidence
        confidence_match = re.search(r'confidence.*?(\d+)', response, re.IGNORECASE)
        if confidence_match:
            confidence_level = float(confidence_match.group(1)) / 100.0  # Normalize to 0-1
        
        # Extract insights (look for bullet points or numbered lists)
        insights_section = re.search(r'(?:key insights|insights):(.+?)(?:red flags|recommendations|confidence|$)', 
                                   response, re.IGNORECASE | re.DOTALL)
        if insights_section:
            insights_text = insights_section.group(1)
            # Extract bullet points or numbered items
            insights = re.findall(r'[•\-\*]?\s*(.+)', insights_text)
            key_insights = [insight.strip() for insight in insights if insight.strip() and len(insight.strip()) > 10][:3]
        
        # Extract recommendations
        rec_section = re.search(r'(?:recommendations|suggestions):(.+?)(?:confidence|$)', 
                              response, re.IGNORECASE | re.DOTALL)
        if rec_section:
            rec_text = rec_section.group(1)
            recs = re.findall(r'[•\-\*]?\s*(.+)', rec_text)
            recommendations = [rec.strip() for rec in recs if rec.strip() and len(rec.strip()) > 10][:3]
        
        # Extract red flags (expert personas only)
        if self.persona.persona_type == PersonaType.EXPERT:
            flags_section = re.search(r'(?:red flags|concerns):(.+?)(?:recommendations|confidence|$)', 
                                    response, re.IGNORECASE | re.DOTALL)
            if flags_section:
                flags_text = flags_section.group(1)
                flags = re.findall(r'[•\-\*]?\s*(.+)', flags_text)
                red_flags = [flag.strip() for flag in flags if flag.strip() and len(flag.strip()) > 10][:3]
        
        # Calculate expertise match for expert personas
        expertise_match = 0.0
        authentic_language = False
        
        if self.persona.persona_type == PersonaType.EXPERT:
            # Simple heuristic: check if response contains persona-specific terms
            persona_terms = []
            for patterns in self.persona.language_patterns.values():
                persona_terms.extend(patterns)
            
            if persona_terms:
                found_terms = sum(1 for term in persona_terms if term.lower() in response.lower())
                expertise_match = min(found_terms / len(persona_terms), 1.0)
                authentic_language = found_terms > 0
        
        return CritiqueResult(
            persona_name=self.persona.name,
            persona_type=self.persona.persona_type,
            model_used=self.model,
            quality_score=quality_score,
            key_insights=key_insights or ["Analysis provided in response"],
            recommendations=recommendations or ["See detailed response for recommendations"],
            confidence_level=confidence_level,
            critique_text=response,
            red_flags_identified=red_flags,
            expertise_match=expertise_match,
            authentic_language_used=authentic_language,
            execution_time_ms=execution_time_ms,
            token_count=len(response.split()) * 1.3,  # Rough token estimate
            estimated_cost=0.0  # Will be calculated by caller if needed
        ) 
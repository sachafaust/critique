from typing import Dict, List, Optional, Any, Type, Union, Protocol, runtime_checkable
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import json
import asyncio
from datetime import datetime
import re
from pydantic import BaseModel, Field, validator, create_model
from typing_extensions import TypedDict, NotRequired
from enum import Enum
from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass
from functools import lru_cache

logger = logging.getLogger(__name__)

class ResponseType(str, Enum):
    """Types of responses that can be processed."""
    CREATOR = "creator"
    CRITIC = "critic"

class ResponseVersion(str, Enum):
    """Version of response schemas."""
    V1 = "v1"
    V2 = "v2"  # For future schema changes

@runtime_checkable
class ResponseValidator(Protocol):
    """Protocol for response validators."""
    def validate(self, data: Dict[str, Any]) -> Dict[str, Any]: ...
    def extract(self, text: str) -> Dict[str, Any]: ...

class BaseResponseModel(BaseModel):
    """Base model for all responses with common fields."""
    version: ResponseVersion = Field(default=ResponseVersion.V1)
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

class CreatorResponse(BaseResponseModel):
    """Validated structure for creator responses."""
    content: str
    reasoning: str = Field(default="")
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    iteration: int = Field(default=1)

    @validator('confidence')
    def normalize_confidence(cls, v):
        return max(0.0, min(1.0, float(v)))

class CriticResponse(BaseResponseModel):
    """Validated structure for critic responses."""
    quality_score: float = Field(ge=0.0, le=1.0)
    strengths: List[str] = Field(default_factory=list)
    improvements: List[str] = Field(default_factory=list)
    specific_feedback: str
    continue_iteration: bool = Field(default=True)
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    critic_model: str = Field(default="unknown")

    @validator('quality_score')
    def normalize_score(cls, v):
        return max(0.0, min(1.0, float(v)))

    @validator('improvements')
    def ensure_improvements(cls, v):
        if not v:
            return ["Consider additional details or examples"]
        return v

@dataclass
class ValidationResult:
    """Result of validation with detailed information."""
    is_valid: bool
    data: Dict[str, Any]
    errors: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]

class ResponseValidatorFactory:
    """Factory for creating response validators."""
    
    @staticmethod
    @lru_cache(maxsize=128)
    def get_validator(response_type: ResponseType, version: ResponseVersion = ResponseVersion.V1) -> ResponseValidator:
        """Get a validator for the specified response type and version."""
        if response_type == ResponseType.CREATOR:
            return CreatorValidator(version)
        elif response_type == ResponseType.CRITIC:
            return CriticValidator(version)
        raise ValueError(f"Unknown response type: {response_type}")

class BaseValidator(ABC):
    """Base class for all validators."""
    
    def __init__(self, version: ResponseVersion):
        self.version = version
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate the response data."""
        pass

    @abstractmethod
    def extract(self, text: str) -> Dict[str, Any]:
        """Extract structured data from text."""
        pass

    def _log_validation_error(self, error: Exception, context: Dict[str, Any]):
        """Log validation errors with context."""
        self.logger.error(f"Validation error: {str(error)}", extra={
            "error": str(error),
            "error_type": type(error).__name__,
            "context": context,
            "version": self.version
        })

class CreatorValidator(BaseValidator):
    """Validator for creator responses."""
    
    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        try:
            validated = CreatorResponse(**data)
            return ValidationResult(
                is_valid=True,
                data=validated.dict(),
                errors=[],
                warnings=[],
                metadata={"version": self.version}
            )
        except Exception as e:
            self._log_validation_error(e, data)
            return ValidationResult(
                is_valid=False,
                data=self._get_default_data(),
                errors=[str(e)],
                warnings=[],
                metadata={"version": self.version}
            )

    def extract(self, text: str) -> Dict[str, Any]:
        # For creator responses, we expect mostly raw content
        return {
            "content": text,
            "reasoning": "Generated content",
            "confidence": 0.8,
            "version": self.version,
            "timestamp": datetime.now().isoformat(),
            "metadata": {"raw_response": True}
        }

    def _get_default_data(self) -> Dict[str, Any]:
        return {
            "content": "Failed to generate content",
            "reasoning": "Validation error occurred",
            "confidence": 0.5,
            "iteration": 1,
            "version": self.version,
            "timestamp": datetime.now().isoformat(),
            "metadata": {"error": True}
        }

class CriticValidator(BaseValidator):
    """Validator for critic responses."""
    
    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        try:
            validated = CriticResponse(**data)
            return ValidationResult(
                is_valid=True,
                data=validated.dict(),
                errors=[],
                warnings=[],
                metadata={"version": self.version}
            )
        except Exception as e:
            self._log_validation_error(e, data)
            return ValidationResult(
                is_valid=False,
                data=self._get_default_data(),
                errors=[str(e)],
                warnings=[],
                metadata={"version": self.version}
            )

    def extract(self, text: str) -> Dict[str, Any]:
        try:
            # Try direct JSON parsing
            return json.loads(text)
        except json.JSONDecodeError:
            try:
                # Try to extract JSON object
                json_match = re.search(r'\{[\s\S]*\}', text)
                if json_match:
                    return json.loads(json_match.group(0))
            except Exception as e:
                self._log_validation_error(e, {"text": text})
        return self._get_default_data()

    def _get_default_data(self) -> Dict[str, Any]:
        return {
            "quality_score": 0.5,
            "strengths": ["Content provided"],
            "improvements": ["Improve response format"],
            "specific_feedback": "Failed to validate critic response",
            "continue_iteration": False,
            "confidence": 0.5,
            "critic_model": "unknown",
            "version": self.version,
            "timestamp": datetime.now().isoformat(),
            "metadata": {"error": True}
        }

class CreatorChain:
    """Chain for content creation with iterative improvement."""
    
    def __init__(self, model: BaseChatModel):
        self.model = model
        self.validator = ResponseValidatorFactory.get_validator(ResponseType.CREATOR)
        
        # Check if this is an o1 or o3 reasoning model
        model_name = getattr(model, 'model_name', getattr(model, 'model', ''))
        self.is_reasoning_model = any(reasoning_model in model_name.lower() 
                                     for reasoning_model in ['o1', 'o3'])
        
        self.system_message = """You are an expert content creator. Your task is to create high-quality, comprehensive responses to user prompts.

Guidelines:
- Provide clear, well-structured content that fully addresses the prompt
- Include relevant examples, details, and explanations
- Consider multiple perspectives when appropriate
- Be thorough but concise
- If this is an improved iteration, incorporate the feedback while maintaining strengths

Create your response addressing the user's request."""
    
    async def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Generate content based on prompt and optional feedback."""
        try:
            prompt = inputs.get("prompt", "")
            feedback = inputs.get("feedback", "")
            iteration = inputs.get("iteration", 1)
            previous_content = inputs.get("previous_content", "")
            
            # Build user message
            if self.is_reasoning_model:
                # For o1/o3 models, incorporate system instructions into user message
                user_content = f"""{self.system_message}

User Request: {prompt}"""
            else:
                user_content = f"User Request: {prompt}"
            
            if iteration > 1 and feedback and previous_content:
                user_content += f"""

Previous Content (Iteration {iteration-1}):
{previous_content}

Feedback Received:
{feedback}

Instructions: Please create an improved version that addresses the feedback while maintaining the strengths of the previous version."""
            
            # Create messages - different for reasoning models
            if self.is_reasoning_model:
                # o1/o3 models only support user and assistant roles
                messages = [
                    {"role": "user", "content": user_content}
                ]
            else:
                # Standard models support system role
                messages = [
                    {"role": "system", "content": self.system_message},
                    {"role": "user", "content": user_content}
                ]
            
            # Debug logging: Show exact prompt sent to LLM
            import structlog
            logger = structlog.get_logger()
            logger.debug("Creator LLM Request", 
                        model=getattr(self.model, 'model_name', getattr(self.model, 'model', 'unknown')),
                        messages=messages,
                        is_reasoning_model=self.is_reasoning_model)
            
            # Get model response
            response = await self.model.ainvoke(messages)
            content = response.content.strip()
            
            # Debug logging: Show LLM response
            logger.debug("Creator LLM Response",
                        model=getattr(self.model, 'model_name', getattr(self.model, 'model', 'unknown')),
                        response_length=len(content),
                        response_preview=content[:200] + "..." if len(content) > 200 else content)
            
            # Validate and return
            result = self.validator.extract(content)
            result["iteration"] = iteration
            
            validated_result = self.validator.validate(result)
            
            return {
                "creator_response": validated_result.data,
                "raw_response": content,
                "validation": {
                    "is_valid": validated_result.is_valid,
                    "errors": validated_result.errors,
                    "warnings": validated_result.warnings
                }
            }
            
        except Exception as e:
            logger.error(f"Error in creator chain: {str(e)}", exc_info=True)
            return {
                "creator_response": {
                    "content": f"Error during content creation: {str(e)}",
                    "reasoning": "Failed to create content",
                    "confidence": 0.5,
                    "iteration": inputs.get("iteration", 1),
                    "version": ResponseVersion.V1,
                    "timestamp": datetime.now().isoformat(),
                    "metadata": {"error": True, "error_type": type(e).__name__}
                },
                "raw_response": str(e),
                "validation": {
                    "is_valid": False,
                    "errors": [str(e)],
                    "warnings": []
                }
            }

class CriticChain:
    """Chain for providing structured criticism and feedback."""
    
    def __init__(self, model: BaseChatModel, model_name: str):
        self.model = model
        self.model_name = model_name
        self.validator = ResponseValidatorFactory.get_validator(ResponseType.CRITIC)
        
        # Check if this is an o1 or o3 reasoning model
        model_name_check = getattr(model, 'model_name', getattr(model, 'model', ''))
        self.is_reasoning_model = any(reasoning_model in model_name_check.lower() 
                                     for reasoning_model in ['o1', 'o3'])
        
        self.system_message = """You are an expert critic providing constructive feedback. Your task is to analyze content and provide actionable improvement suggestions.

Analyze the content and provide feedback in the following JSON format:
{
    "quality_score": <float between 0 and 1>,
    "strengths": ["<strength 1>", "<strength 2>", ...],
    "improvements": ["<improvement 1>", "<improvement 2>", ...],
    "specific_feedback": "<detailed analysis and recommendations>",
    "continue_iteration": <true/false>,
    "confidence": <float between 0 and 1>
}

Guidelines:
- Quality score: 0.8+ means excellent, 0.6-0.8 means good, <0.6 needs improvement
- Identify 2-3 key strengths to maintain
- Suggest 2-4 specific, actionable improvements
- Set continue_iteration to false only if quality score >= 0.85
- Be constructive and specific in your feedback

IMPORTANT: Your response must be a single valid JSON object. Do not include any text before or after the JSON."""
    
    async def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze content and provide structured feedback."""
        try:
            content = inputs.get("content", "")
            prompt = inputs.get("original_prompt", "")
            iteration = inputs.get("iteration", 1)
            
            # Create user message
            if self.is_reasoning_model:
                # For o1/o3 models, incorporate system instructions into user message
                user_content = f"""{self.system_message}

Original Prompt: {prompt}

Content to Review (Iteration {iteration}):
{content}

Please analyze this content and provide structured feedback in the required JSON format."""
            else:
                user_content = f"""Original Prompt: {prompt}

Content to Review (Iteration {iteration}):
{content}

Please analyze this content and provide structured feedback in the required JSON format."""
            
            # Create messages - different for reasoning models
            if self.is_reasoning_model:
                # o1/o3 models only support user and assistant roles
                messages = [
                    {"role": "user", "content": user_content}
                ]
            else:
                # Standard models support system role
                messages = [
                    {"role": "system", "content": self.system_message},
                    {"role": "user", "content": user_content}
                ]
            
            # Debug logging: Show exact prompt sent to LLM
            import structlog
            logger = structlog.get_logger()
            logger.debug("Critic LLM Request",
                        model=self.model_name,
                        messages=messages,
                        is_reasoning_model=self.is_reasoning_model,
                        iteration=iteration)
            
            # Get model response
            response = await self.model.ainvoke(messages)
            content_response = response.content.strip()
            
            # Debug logging: Show LLM response
            logger.debug("Critic LLM Response",
                        model=self.model_name,
                        response_length=len(content_response),
                        response_preview=content_response[:200] + "..." if len(content_response) > 200 else content_response)
            
            # Clean the response content
            start = content_response.find('{')
            end = content_response.rfind('}') + 1
            if start == -1 or end == 0:
                raise ValueError("No JSON object found in response")
            
            json_str = content_response[start:end]
            
            # Parse the cleaned JSON
            try:
                raw_data = json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {str(e)}", extra={
                    "response": content_response,
                    "cleaned_json": json_str,
                    "error": str(e)
                })
                raise ValueError(f"Invalid JSON in response: {str(e)}")
            
            # Add model information
            raw_data["critic_model"] = self.model_name
            
            # Validate and normalize the data
            result = self.validator.validate(raw_data)
            
            if not result.is_valid:
                logger.warning("Invalid critic response", extra={
                    "errors": result.errors,
                    "warnings": result.warnings,
                    "metadata": result.metadata,
                    "raw_data": raw_data
                })
            
            return {
                "critic_response": result.data,
                "raw_response": content_response,
                "validation": {
                    "is_valid": result.is_valid,
                    "errors": result.errors,
                    "warnings": result.warnings
                }
            }
            
        except Exception as e:
            logger.error(f"Error in critic analysis: {str(e)}", exc_info=True)
            return {
                "critic_response": {
                    "quality_score": 0.5,
                    "strengths": ["Content provided"],
                    "improvements": ["Fix critic analysis process"],
                    "specific_feedback": f"Error during critique analysis: {str(e)}",
                    "continue_iteration": False,
                    "confidence": 0.5,
                    "critic_model": self.model_name,
                    "version": ResponseVersion.V1,
                    "timestamp": datetime.now().isoformat(),
                    "metadata": {"error": True, "error_type": type(e).__name__}
                },
                "raw_response": str(e),
                "validation": {
                    "is_valid": False,
                    "errors": [str(e)],
                    "warnings": []
                }
            }

class IterativeWorkflowChain:
    """Manages the creator-critic iteration workflow."""
    
    def __init__(self, creator_model: BaseChatModel, critic_models: List[BaseChatModel], critic_model_names: List[str]):
        self.creator_chain = CreatorChain(creator_model)
        self.critic_chains = [CriticChain(model, name) for model, name in zip(critic_models, critic_model_names)]
    
    async def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the full creator-critic iteration workflow."""
        prompt = inputs.get("prompt", "")
        max_iterations = inputs.get("max_iterations", 1)
        
        results = {
            "iterations": [],
            "final_content": "",
            "total_iterations": 0,
            "convergence_achieved": False
        }
        
        current_content = ""
        
        for iteration in range(1, max_iterations + 1):
            iteration_result = {
                "iteration": iteration,
                "creator": {},
                "critics": [],
                "should_continue": False
            }
            
            # Creator phase
            creator_inputs = {
                "prompt": prompt,
                "iteration": iteration,
                "previous_content": current_content if iteration > 1 else "",
                "feedback": self._combine_feedback(results["iterations"][-1]["critics"]) if iteration > 1 else ""
            }
            
            creator_result = await self.creator_chain._call(creator_inputs)
            iteration_result["creator"] = creator_result
            current_content = creator_result["creator_response"]["content"]
            
            # Critics phase
            critic_tasks = []
            for critic_chain in self.critic_chains:
                critic_inputs = {
                    "content": current_content,
                    "original_prompt": prompt,
                    "iteration": iteration
                }
                critic_tasks.append(critic_chain._call(critic_inputs))
            
            critic_results = await asyncio.gather(*critic_tasks, return_exceptions=True)
            
            # Process critic results
            for i, critic_result in enumerate(critic_results):
                if isinstance(critic_result, Exception):
                    logger.error(f"Critic {i} failed: {str(critic_result)}")
                    # Create error result
                    iteration_result["critics"].append({
                        "critic_response": {
                            "quality_score": 0.5,
                            "strengths": [],
                            "improvements": ["Critic analysis failed"],
                            "specific_feedback": f"Error: {str(critic_result)}",
                            "continue_iteration": False,
                            "confidence": 0.5,
                            "critic_model": self.critic_chains[i].model_name if i < len(self.critic_chains) else "unknown"
                        },
                        "validation": {"is_valid": False, "errors": [str(critic_result)], "warnings": []}
                    })
                else:
                    iteration_result["critics"].append(critic_result)
            
            # Determine if we should continue
            should_continue = self._should_continue_iteration(iteration_result["critics"], iteration, max_iterations)
            iteration_result["should_continue"] = should_continue
            
            results["iterations"].append(iteration_result)
            results["total_iterations"] = iteration
            
            if not should_continue:
                results["convergence_achieved"] = True
                break
        
        results["final_content"] = current_content
        return results
    
    def _combine_feedback(self, critic_results: List[Dict[str, Any]]) -> str:
        """Combine feedback from all critics into a single feedback string."""
        if not critic_results:
            return ""
        
        combined_feedback = "Feedback from critics:\n\n"
        
        for i, critic_result in enumerate(critic_results, 1):
            critic_data = critic_result.get("critic_response", {})
            model_name = critic_data.get("critic_model", f"Critic {i}")
            
            combined_feedback += f"Critic {i} ({model_name}):\n"
            combined_feedback += f"Quality Score: {critic_data.get('quality_score', 0):.2f}\n"
            combined_feedback += f"Strengths: {', '.join(critic_data.get('strengths', []))}\n"
            combined_feedback += f"Improvements: {', '.join(critic_data.get('improvements', []))}\n"
            combined_feedback += f"Detailed Feedback: {critic_data.get('specific_feedback', '')}\n\n"
        
        return combined_feedback
    
    def _should_continue_iteration(self, critic_results: List[Dict[str, Any]], current_iteration: int, max_iterations: int) -> bool:
        """Determine if iteration should continue based on critic feedback."""
        if current_iteration >= max_iterations:
            return False
        
        if not critic_results:
            return False
        
        # Check if any critic suggests continuing
        for critic_result in critic_results:
            critic_data = critic_result.get("critic_response", {})
            if critic_data.get("continue_iteration", False):
                return True
        
        # Check average quality score
        quality_scores = [
            critic_result.get("critic_response", {}).get("quality_score", 0)
            for critic_result in critic_results
        ]
        
        if quality_scores:
            avg_quality = sum(quality_scores) / len(quality_scores)
            return avg_quality < 0.85  # Continue if average quality is below 85%
        
        return False 
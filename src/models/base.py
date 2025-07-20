from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import time
import logging

logger = logging.getLogger(__name__)

@dataclass
class ModelResponse:
    """Standardized response from any model provider"""
    content: str
    usage: Optional[Dict[str, int]] = None
    response_time: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class ModelConfig:
    """Configuration for model parameters"""
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0

class BaseModel(ABC):
    """Abstract base class for all model providers"""
    
    def __init__(self, model_name: str, config: ModelConfig = None):
        self.model_name = model_name
        self.config = config or ModelConfig()
        self.total_requests = 0
        self.total_tokens = 0
        self.failed_requests = 0
        
    @abstractmethod
    def generate(self, 
                 prompt: str, 
                 system_message: Optional[str] = None,
                 **kwargs) -> ModelResponse:
        """Generate a response from the model
        
        Args:
            prompt: The user prompt
            system_message: Optional system message/instruction
            **kwargs: Additional provider-specific parameters
            
        Returns:
            ModelResponse object with standardized fields
        """
        pass
    
    @abstractmethod
    def generate_with_conversation(self,
                                   messages: List[Dict[str, str]],
                                   **kwargs) -> ModelResponse:
        """Generate response with full conversation context
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            **kwargs: Additional provider-specific parameters
            
        Returns:
            ModelResponse object
        """
        pass
    
    def generate_with_retry(self, *args, **kwargs) -> ModelResponse:
        """Generate with retry logic and error handling"""
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                start_time = time.time()
                response = self.generate(*args, **kwargs)
                response.response_time = time.time() - start_time
                
                # Update statistics
                self.total_requests += 1
                if response.usage and 'total_tokens' in response.usage:
                    self.total_tokens += response.usage['total_tokens']
                
                return response
                
            except Exception as e:
                last_exception = e
                self.failed_requests += 1
                
                if attempt < self.config.max_retries:
                    logger.warning(f"Request failed (attempt {attempt + 1}), retrying in {self.config.retry_delay}s: {e}")
                    time.sleep(self.config.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    logger.error(f"Request failed after {self.config.max_retries + 1} attempts: {e}")
        
        raise last_exception
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics for this model instance"""
        return {
            'model_name': self.model_name,
            'total_requests': self.total_requests,
            'failed_requests': self.failed_requests,
            'success_rate': (self.total_requests - self.failed_requests) / max(self.total_requests, 1),
            'total_tokens': self.total_tokens
        }
    
    @abstractmethod
    def validate_config(self) -> bool:
        """Validate that the model is properly configured and accessible"""
        pass
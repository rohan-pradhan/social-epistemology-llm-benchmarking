import os
from typing import Dict, List, Optional, Any, Union
import logging

try:
    import anthropic
except ImportError:
    raise ImportError("Anthropic package not installed. Run: pip install anthropic")

from .base import BaseModel, ModelResponse, ModelConfig

logger = logging.getLogger(__name__)

class AnthropicModel(BaseModel):
    """Anthropic Claude model implementation"""
    
    def __init__(self, model_name: str = "claude-sonnet-4-20250514", config: ModelConfig = None, api_key: str = None):
        super().__init__(model_name, config)
        
        # Set up Anthropic client
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key not provided. Set ANTHROPIC_API_KEY environment variable or pass api_key parameter")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        
        # Validate model name
        self.model_name = model_name
        if not self.validate_config():
            logger.warning(f"Could not validate model {model_name}. Proceeding anyway.")
    
    def generate(self, 
                 prompt: str, 
                 system_message: Optional[str] = None,
                 tools: Optional[List[Dict[str, Any]]] = None,
                 tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
                 **kwargs) -> ModelResponse:
        """Generate response using Anthropic API with optional tool calling"""
        
        # Prepare API parameters
        api_params = {
            "model": self.model_name,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "messages": [{"role": "user", "content": prompt}]
        }
        
        # Add thinking parameter if provided
        if "thinking" in kwargs:
            api_params["thinking"] = kwargs["thinking"]
            # When thinking is enabled, temperature must be set to 1.0
            api_params["temperature"] = 1.0
        
        # Add system message if provided
        if system_message:
            api_params["system"] = system_message
        
        # Add tools if provided
        if tools:
            api_params["tools"] = tools
            
            # Handle tool_choice parameter
            if tool_choice:
                if isinstance(tool_choice, str):
                    if tool_choice in ["auto", "any", "none"]:
                        api_params["tool_choice"] = {"type": tool_choice}
                    else:
                        # Assume it's a tool name
                        api_params["tool_choice"] = {"type": "tool", "name": tool_choice}
                else:
                    api_params["tool_choice"] = tool_choice
        
        # Remove None values
        api_params = {k: v for k, v in api_params.items() if v is not None}
        
        try:
            response = self.client.messages.create(**api_params)
            
            # Handle different content types (text, tool_use, and thinking)
            content_text = ""
            tool_calls = []
            thinking_content = []
            
            for content_block in response.content:
                if content_block.type == "text":
                    content_text += content_block.text
                elif content_block.type == "tool_use":
                    tool_calls.append({
                        "id": content_block.id,
                        "name": content_block.name,
                        "input": content_block.input
                    })
                elif content_block.type == "thinking":
                    thinking_content.append({
                        "thinking": content_block.thinking,
                        "signature": getattr(content_block, 'signature', None)
                    })
            
            return ModelResponse(
                content=content_text,
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                },
                metadata={
                    "model": response.model,
                    "stop_reason": response.stop_reason,
                    "response_id": response.id,
                    "tool_calls": tool_calls,
                    "thinking_content": thinking_content,
                    "raw_content": response.content
                }
            )
            
        except Exception as e:
            logger.error(f"Anthropic API call failed: {e}")
            raise
    
    def generate_with_conversation(self,
                                   messages: List[Dict[str, Any]],
                                   tools: Optional[List[Dict[str, Any]]] = None,
                                   tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
                                   **kwargs) -> ModelResponse:
        """Generate response with full conversation context and optional tool calling"""
        
        # Convert to Anthropic format
        formatted_messages = []
        system_message = None
        
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                # Handle both simple string content and complex content blocks
                if isinstance(msg.get("content"), str):
                    formatted_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
                else:
                    # Content is already in blocks format (for tool results)
                    formatted_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
        
        api_params = {
            "model": self.model_name,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "messages": formatted_messages
        }
        
        # Add thinking parameter if provided
        if "thinking" in kwargs:
            api_params["thinking"] = kwargs["thinking"]
            # When thinking is enabled, temperature must be set to 1.0
            api_params["temperature"] = 1.0
        
        if system_message:
            api_params["system"] = system_message
        
        # Add tools if provided
        if tools:
            api_params["tools"] = tools
            
            # Handle tool_choice parameter
            if tool_choice:
                if isinstance(tool_choice, str):
                    if tool_choice in ["auto", "any", "none"]:
                        api_params["tool_choice"] = {"type": tool_choice}
                    else:
                        # Assume it's a tool name
                        api_params["tool_choice"] = {"type": "tool", "name": tool_choice}
                else:
                    api_params["tool_choice"] = tool_choice
        
        # Remove None values
        api_params = {k: v for k, v in api_params.items() if v is not None}
        
        try:
            response = self.client.messages.create(**api_params)
            
            # Handle different content types (text, tool_use, and thinking)
            content_text = ""
            tool_calls = []
            thinking_content = []
            
            for content_block in response.content:
                if content_block.type == "text":
                    content_text += content_block.text
                elif content_block.type == "tool_use":
                    tool_calls.append({
                        "id": content_block.id,
                        "name": content_block.name,
                        "input": content_block.input
                    })
                elif content_block.type == "thinking":
                    thinking_content.append({
                        "thinking": content_block.thinking,
                        "signature": getattr(content_block, 'signature', None)
                    })
            
            return ModelResponse(
                content=content_text,
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                },
                metadata={
                    "model": response.model,
                    "stop_reason": response.stop_reason,
                    "response_id": response.id,
                    "tool_calls": tool_calls,
                    "thinking_content": thinking_content,
                    "raw_content": response.content
                }
            )
            
        except Exception as e:
            logger.error(f"Anthropic API call failed: {e}")
            raise
    
    def execute_tool_conversation(self, 
                                 initial_prompt: str,
                                 tools: List[Dict[str, Any]],
                                 tool_executor: callable,
                                 system_message: Optional[str] = None,
                                 max_iterations: int = 10,
                                 **kwargs) -> ModelResponse:
        """Execute a conversation with tool calling until completion
        
        Args:
            initial_prompt: The initial user prompt
            tools: List of tool schemas
            tool_executor: Function that takes (tool_name, tool_input) and returns tool result
            system_message: Optional system message
            max_iterations: Maximum number of tool calling iterations
            **kwargs: Additional API parameters
            
        Returns:
            Final ModelResponse after all tool calls are complete
        """
        messages = [{"role": "user", "content": initial_prompt}]
        
        for iteration in range(max_iterations):
            # Generate response with tools
            response = self.generate_with_conversation(
                messages=messages,
                tools=tools,
                system_message=system_message,
                **kwargs
            )
            
            # Check if model used tools
            tool_calls = response.metadata.get("tool_calls", [])
            
            if not tool_calls:
                # No tool calls, conversation is complete
                return response
            
            # Add assistant message with tool calls to conversation
            assistant_content = []
            
            # Add thinking blocks if they exist (required for thinking-enabled conversations)
            thinking_blocks = response.metadata.get("thinking_content", [])
            for thinking_block in thinking_blocks:
                assistant_content.append({
                    "type": "thinking",
                    "thinking": thinking_block["thinking"],
                    "signature": thinking_block.get("signature")
                })
            
            if response.content.strip():
                assistant_content.append({"type": "text", "text": response.content})
            
            # Add tool use blocks
            for tool_call in tool_calls:
                assistant_content.append({
                    "type": "tool_use",
                    "id": tool_call["id"],
                    "name": tool_call["name"],
                    "input": tool_call["input"]
                })
            
            messages.append({
                "role": "assistant", 
                "content": assistant_content
            })
            
            # Execute each tool call and add results
            tool_results = []
            for tool_call in tool_calls:
                logger.info(f"TOOL_EXECUTION_REQUEST: tool_name='{tool_call['name']}', tool_input={tool_call['input']}, tool_id='{tool_call['id']}'")
                try:
                    result = tool_executor(tool_call["name"], tool_call["input"])
                    logger.info(f"TOOL_EXECUTION_SUCCESS: tool_name='{tool_call['name']}', tool_id='{tool_call['id']}', result_length={len(str(result))}")
                    logger.debug(f"TOOL_EXECUTION_OUTPUT: tool_name='{tool_call['name']}', tool_id='{tool_call['id']}', result='{str(result)}'")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_call["id"],
                        "content": str(result)
                    })
                except Exception as e:
                    logger.error(f"TOOL_EXECUTION_ERROR: tool_name='{tool_call['name']}', tool_id='{tool_call['id']}', error='{str(e)}'")
                    tool_results.append({
                        "type": "tool_result", 
                        "tool_use_id": tool_call["id"],
                        "content": f"Error: {str(e)}",
                        "is_error": True
                    })
            
            # Add tool results to conversation
            messages.append({
                "role": "user",
                "content": tool_results
            })
        
        # If we reach here, we hit max_iterations
        logger.warning(f"Tool conversation reached max iterations ({max_iterations})")
        return response
    
    def validate_config(self) -> bool:
        """Validate that the model is properly configured and accessible"""
        try:
            # Try a simple API call to validate credentials and model access
            test_response = self.client.messages.create(
                model=self.model_name,
                max_tokens=1,
                messages=[{"role": "user", "content": "test"}]
            )
            return True
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return False

class ClaudeSonnet4Model(AnthropicModel):
    """Claude Sonnet 4 model with optimized settings"""
    
    def __init__(self, config: ModelConfig = None, api_key: str = None):
        if config is None:
            config = ModelConfig(
                temperature=0.1,  # Lower temperature for more consistent responses
                max_tokens=8000,  # Higher token limit for detailed analysis
            )
        super().__init__("claude-sonnet-4-20250514", config, api_key)

class ClaudeOpus4Model(AnthropicModel):
    """Claude Opus 4 model with optimized settings"""
    
    def __init__(self, config: ModelConfig = None, api_key: str = None):
        if config is None:
            config = ModelConfig(
                temperature=0.1,
                max_tokens=8000,
            )
        super().__init__("claude-opus-4-20250514", config, api_key)

class Claude35HaikuModel(AnthropicModel):
    """Claude 3.5 Haiku model with optimized settings"""
    
    def __init__(self, config: ModelConfig = None, api_key: str = None):
        if config is None:
            config = ModelConfig(
                temperature=0.1,
                max_tokens=4000,
            )
        super().__init__("claude-3-5-haiku-20241022", config, api_key)
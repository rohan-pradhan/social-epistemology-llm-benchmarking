import os
import time
import json
from typing import Dict, List, Optional, Any, Union
import logging

try:
    import openai
except ImportError:
    raise ImportError("OpenAI package not installed. Run: pip install openai")

from .base import BaseModel, ModelResponse, ModelConfig

logger = logging.getLogger(__name__)

def convert_anthropic_tools_to_openai(anthropic_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert Anthropic tool schemas to OpenAI function calling format
    
    Args:
        anthropic_tools: List of tools in Anthropic format
        
    Returns:
        List of tools in OpenAI function calling format
    """
    openai_tools = []
    for tool in anthropic_tools:
        openai_tool = {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["input_schema"]
            }
        }
        openai_tools.append(openai_tool)
    return openai_tools

def convert_anthropic_tools_to_openai_responses(anthropic_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert Anthropic tool schemas to OpenAI Responses API format
    
    Args:
        anthropic_tools: List of tools in Anthropic format
        
    Returns:
        List of tools in OpenAI Responses API format
    """
    openai_tools = []
    for tool in anthropic_tools:
        openai_tool = {
            "type": "function",
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["input_schema"]
        }
        openai_tools.append(openai_tool)
    return openai_tools

class OpenAIModel(BaseModel):
    """OpenAI GPT model implementation"""
    
    def __init__(self, model_name: str = "gpt-4o", config: ModelConfig = None, api_key: str = None):
        super().__init__(model_name, config)
        
        # Set up OpenAI client
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter")
        
        self.client = openai.OpenAI(api_key=self.api_key)
        
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
        """Generate response using OpenAI API with optional tool calling"""
        
        # Prepare messages
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        # Merge config with kwargs (kwargs take precedence)
        api_params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "frequency_penalty": kwargs.get("frequency_penalty", self.config.frequency_penalty),
            "presence_penalty": kwargs.get("presence_penalty", self.config.presence_penalty),
        }
        
        # Add tools if provided
        if tools:
            api_params["tools"] = tools
            
            # Handle tool_choice parameter
            if tool_choice:
                if isinstance(tool_choice, str):
                    if tool_choice in ["auto", "required", "none"]:
                        api_params["tool_choice"] = tool_choice
                    else:
                        # Assume it's a function name
                        api_params["tool_choice"] = {"type": "function", "function": {"name": tool_choice}}
                else:
                    api_params["tool_choice"] = tool_choice
        
        # Remove None values
        api_params = {k: v for k, v in api_params.items() if v is not None}
        
        try:
            response = self.client.chat.completions.create(**api_params)
            
            # Handle both text and function call responses
            content = response.choices[0].message.content or ""
            tool_calls = []
            
            # Check for function calls
            if response.choices[0].message.tool_calls:
                for tool_call in response.choices[0].message.tool_calls:
                    tool_calls.append({
                        "id": tool_call.id,
                        "name": tool_call.function.name,
                        "input": json.loads(tool_call.function.arguments)
                    })
            
            return ModelResponse(
                content=content,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                metadata={
                    "model": response.model,
                    "finish_reason": response.choices[0].finish_reason,
                    "response_id": response.id,
                    "tool_calls": tool_calls,
                    "raw_message": response.choices[0].message
                }
            )
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise
    
    def generate_with_conversation(self,
                                   messages: List[Dict[str, Any]],
                                   tools: Optional[List[Dict[str, Any]]] = None,
                                   tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
                                   **kwargs) -> ModelResponse:
        """Generate response with full conversation context and optional tool calling"""
        
        # Convert to OpenAI format if needed
        formatted_messages = []
        for msg in messages:
            # Handle different message types (text, tool calls, tool results)
            if msg["role"] == "assistant" and "tool_calls" in msg:
                # Assistant message with tool calls
                formatted_messages.append({
                    "role": "assistant",
                    "content": msg.get("content"),
                    "tool_calls": msg["tool_calls"]
                })
            elif msg["role"] == "tool":
                # Tool result message
                formatted_messages.append({
                    "role": "tool",
                    "tool_call_id": msg["tool_call_id"],
                    "content": msg["content"]
                })
            else:
                # Regular text message
                formatted_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        api_params = {
            "model": self.model_name,
            "messages": formatted_messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "frequency_penalty": kwargs.get("frequency_penalty", self.config.frequency_penalty),
            "presence_penalty": kwargs.get("presence_penalty", self.config.presence_penalty),
        }
        
        # Add tools if provided
        if tools:
            api_params["tools"] = tools
            
            # Handle tool_choice parameter
            if tool_choice:
                if isinstance(tool_choice, str):
                    if tool_choice in ["auto", "required", "none"]:
                        api_params["tool_choice"] = tool_choice
                    else:
                        # Assume it's a function name
                        api_params["tool_choice"] = {"type": "function", "function": {"name": tool_choice}}
                else:
                    api_params["tool_choice"] = tool_choice
        
        # Remove None values
        api_params = {k: v for k, v in api_params.items() if v is not None}
        
        try:
            response = self.client.chat.completions.create(**api_params)
            
            # Handle both text and function call responses
            content = response.choices[0].message.content or ""
            tool_calls = []
            
            # Check for function calls
            if response.choices[0].message.tool_calls:
                for tool_call in response.choices[0].message.tool_calls:
                    tool_calls.append({
                        "id": tool_call.id,
                        "name": tool_call.function.name,
                        "input": json.loads(tool_call.function.arguments)
                    })
            
            return ModelResponse(
                content=content,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                metadata={
                    "model": response.model,
                    "finish_reason": response.choices[0].finish_reason,
                    "response_id": response.id,
                    "tool_calls": tool_calls,
                    "raw_message": response.choices[0].message
                }
            )
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise
    
    
    def validate_config(self) -> bool:
        """Validate that the model is properly configured and accessible"""
        try:
            # Try a simple API call to validate credentials and model access
            test_response = self.client.responses.create(
                model=self.model_name,
                input=[{"role": "user", "content": "test"}],
                store=False
            )
            return True
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return False

class GPT4oModel(OpenAIModel):
    """GPT-4o model with optimized settings"""
    
    def __init__(self, config: ModelConfig = None, api_key: str = None):
        if config is None:
            config = ModelConfig(
                temperature=0.1,  # Lower temperature for more consistent responses
                max_tokens=4000,  # Higher token limit for detailed analysis
            )
        super().__init__("gpt-4o", config, api_key)

class GPT4oMiniModel(OpenAIModel):
    """GPT-4o-mini model with optimized settings"""
    
    def __init__(self, config: ModelConfig = None, api_key: str = None):
        if config is None:
            config = ModelConfig(
                temperature=0.1,
                max_tokens=2000,
            )
        super().__init__("gpt-4o-mini", config, api_key)

class GPT4oMiniHighModel(OpenAIModel):
    """GPT-4o-mini with high reasoning settings (enhanced prompt engineering)"""
    
    def __init__(self, config: ModelConfig = None, api_key: str = None):
        if config is None:
            config = ModelConfig(
                temperature=0.05,  # Even lower temperature for high reasoning
                max_tokens=4000,   # More tokens for detailed reasoning
            )
        super().__init__("gpt-4o-mini", config, api_key)
    
    def generate(self, 
                 prompt: str, 
                 system_message: Optional[str] = None,
                 **kwargs) -> ModelResponse:
        """Enhanced generation with reasoning prompts"""
        
        # Add reasoning enhancement to system message
        enhanced_system = "You are an expert analyst. Think step by step and provide detailed reasoning for your conclusions."
        if system_message:
            enhanced_system = system_message + "\n\n" + enhanced_system
        
        # Add reasoning prompt to user prompt
        enhanced_prompt = f"{prompt}\n\nPlease think through this step by step and show your reasoning clearly."
        
        return super().generate(enhanced_prompt, enhanced_system, **kwargs)

class OpenAIResponsesModel(BaseModel):
    """OpenAI model using the new Responses API"""
    
    def __init__(self, model_name: str, config: ModelConfig = None, api_key: str = None):
        super().__init__(model_name, config)
        
        # Set up OpenAI client
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter")
        
        self.client = openai.OpenAI(api_key=self.api_key)
        self.model_name = model_name
        
    def generate(self, 
                 prompt: str, 
                 system_message: Optional[str] = None,
                 tools: Optional[List[Dict[str, Any]]] = None,
                 reasoning: Optional[Dict[str, Any]] = None,
                 **kwargs) -> ModelResponse:
        """Generate response using OpenAI Responses API"""
        
        # Prepare input messages
        input_messages = []
        if system_message:
            input_messages.append({"role": "system", "content": system_message})
        input_messages.append({"role": "user", "content": prompt})
        
        # Prepare API parameters
        api_params = {
            "model": self.model_name,
            "input": input_messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_output_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "store": False  # Don't store for privacy
        }
        
        # Add reasoning if provided
        if reasoning:
            api_params["reasoning"] = reasoning
            # When reasoning is enabled, temperature must be set to 1.0 for consistency with Claude
            api_params["temperature"] = 1.0
        
        # Add tools if provided
        if tools:
            api_params["tools"] = tools
            
        try:
            response = self.client.responses.create(**api_params)
            
            # Extract content from response
            content = response.output_text or ""
            
            # Extract usage information
            usage_info = {}
            if hasattr(response, 'usage'):
                usage_info = {
                    "prompt_tokens": getattr(response.usage, 'input_tokens', 0),
                    "completion_tokens": getattr(response.usage, 'output_tokens', 0),
                    "total_tokens": getattr(response.usage, 'total_tokens', 0)
                }
            
            return ModelResponse(
                content=content,
                usage=usage_info,
                metadata={
                    "model": self.model_name,
                    "response_id": getattr(response, 'id', ''),
                    "status": getattr(response, 'status', ''),
                    "reasoning_tokens": getattr(getattr(response.usage, 'output_tokens_details', None), 'reasoning_tokens', 0) if hasattr(response, 'usage') else 0
                }
            )
            
        except Exception as e:
            logger.error(f"OpenAI Responses API call failed: {e}")
            raise
    
    def generate_with_conversation(self,
                                   messages: List[Dict[str, Any]],
                                   tools: Optional[List[Dict[str, Any]]] = None,
                                   **kwargs) -> ModelResponse:
        """Generate response with full conversation context using Responses API"""
        
        # Prepare API parameters
        api_params = {
            "model": self.model_name,
            "input": messages,
            "store": False
        }
        
        # Add tools if provided
        if tools:
            api_params["tools"] = tools
            
        # Add reasoning if provided in kwargs
        if "reasoning" in kwargs:
            api_params["reasoning"] = kwargs["reasoning"]
            
        try:
            response = self.client.responses.create(**api_params)
            
            # Extract content from response
            content = response.output_text or ""
            
            # Extract usage information
            usage_info = {}
            if hasattr(response, 'usage'):
                usage_info = {
                    "prompt_tokens": getattr(response.usage, 'input_tokens', 0),
                    "completion_tokens": getattr(response.usage, 'output_tokens', 0),
                    "total_tokens": getattr(response.usage, 'total_tokens', 0)
                }
            
            return ModelResponse(
                content=content,
                usage=usage_info,
                metadata={
                    "model": self.model_name,
                    "response_id": getattr(response, 'id', ''),
                    "status": getattr(response, 'status', ''),
                    "reasoning_tokens": getattr(getattr(response.usage, 'output_tokens_details', None), 'reasoning_tokens', 0) if hasattr(response, 'usage') else 0
                }
            )
            
        except Exception as e:
            logger.error(f"OpenAI Responses API call failed: {e}")
            raise

    def execute_tool_conversation(self, 
                                 initial_prompt: str,
                                 tools: List[Dict[str, Any]],
                                 tool_executor: callable,
                                 system_message: Optional[str] = None,
                                 max_iterations: int = 10,
                                 **kwargs) -> ModelResponse:
        """Execute a conversation with tool calling until completion using Responses API"""
        
        # Convert Anthropic format tools to OpenAI Responses format if needed
        if tools and tools[0].get("input_schema"):
            # This looks like Anthropic format, convert it
            tools = convert_anthropic_tools_to_openai_responses(tools)
        
        # Prepare input messages
        input_messages = []
        if system_message:
            input_messages.append({"role": "system", "content": system_message})
        input_messages.append({"role": "user", "content": initial_prompt})
        
        for iteration in range(max_iterations):
            # Prepare API parameters
            api_params = {
                "model": self.model_name,
                "input": input_messages,
                "tools": tools,
                "parallel_tool_calls": False,  # Only allow one tool call at a time
                "temperature": kwargs.get("temperature", self.config.temperature),
                "max_output_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                "store": False
            }
            
            # Add reasoning if provided in kwargs
            if "reasoning" in kwargs:
                api_params["reasoning"] = kwargs["reasoning"]
                # When reasoning is enabled, temperature must be set to 1.0 for consistency with Claude
                api_params["temperature"] = 1.0
            
            try:
                response = self.client.responses.create(**api_params)
                
                # Extract content and check for tool calls
                content = response.output_text or ""
                
                # Check if response has tool calls (need to parse from response.output)
                tool_calls = []
                if hasattr(response, 'output') and response.output:
                    for item in response.output:
                        if hasattr(item, 'type') and item.type == 'function_call':
                            tool_calls.append({
                                "id": getattr(item, 'call_id', ''),
                                "name": getattr(item, 'name', ''),
                                "input": json.loads(getattr(item, 'arguments', '{}'))
                            })
                
                # If no tool calls, conversation is complete
                if not tool_calls:
                    # Extract usage information
                    usage_info = {}
                    if hasattr(response, 'usage'):
                        usage_info = {
                            "prompt_tokens": getattr(response.usage, 'input_tokens', 0),
                            "completion_tokens": getattr(response.usage, 'output_tokens', 0),
                            "total_tokens": getattr(response.usage, 'total_tokens', 0)
                        }
                    
                    return ModelResponse(
                        content=content,
                        usage=usage_info,
                        metadata={
                            "model": self.model_name,
                            "response_id": getattr(response, 'id', ''),
                            "tool_calls": tool_calls
                        }
                    )
                
                # Add assistant message with tool calls to conversation
                input_messages.append({
                    "role": "assistant", 
                    "content": content
                })
                
                # Execute each tool call and add results
                for tool_call in tool_calls:
                    logger.info(f"TOOL_EXECUTION_REQUEST: tool_name='{tool_call['name']}', tool_input={tool_call['input']}, tool_id='{tool_call['id']}'")
                    try:
                        result = tool_executor(tool_call["name"], tool_call["input"])
                        logger.info(f"TOOL_EXECUTION_SUCCESS: tool_name='{tool_call['name']}', tool_id='{tool_call['id']}', result_length={len(str(result))}")
                        logger.debug(f"TOOL_EXECUTION_OUTPUT: tool_name='{tool_call['name']}', tool_id='{tool_call['id']}', result='{str(result)}'")
                        input_messages.append({
                            "role": "user",  # Tool results go as user messages in Responses API
                            "content": f"Tool '{tool_call['name']}' result: {str(result)}"
                        })
                    except Exception as e:
                        logger.error(f"TOOL_EXECUTION_ERROR: tool_name='{tool_call['name']}', tool_id='{tool_call['id']}', error='{str(e)}'")
                        input_messages.append({
                            "role": "user",
                            "content": f"Tool '{tool_call['name']}' error: {str(e)}"
                        })
                        
            except Exception as e:
                logger.error(f"OpenAI Responses API call failed: {e}")
                raise
        
        # If we reach here, we hit max_iterations
        logger.warning(f"Tool conversation reached max iterations ({max_iterations})")
        # Return the last response
        usage_info = {}
        if hasattr(response, 'usage'):
            usage_info = {
                "prompt_tokens": getattr(response.usage, 'input_tokens', 0),
                "completion_tokens": getattr(response.usage, 'output_tokens', 0),
                "total_tokens": getattr(response.usage, 'total_tokens', 0)
            }
        
        return ModelResponse(
            content=content,
            usage=usage_info,
            metadata={
                "model": self.model_name,
                "response_id": getattr(response, 'id', ''),
                "tool_calls": tool_calls
            }
        )

    def validate_config(self) -> bool:
        """Validate that the model is properly configured and accessible"""
        try:
            # Try a simple API call to validate credentials and model access
            test_response = self.client.responses.create(
                model=self.model_name,
                input=[{"role": "user", "content": "test"}],
                store=False
            )
            return True
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return False

class GPT41Model(OpenAIResponsesModel):
    """GPT-4.1 model using Responses API"""
    
    def __init__(self, config: ModelConfig = None, api_key: str = None):
        if config is None:
            config = ModelConfig(
                temperature=0.1,
                max_tokens=8000,  # Match Claude's token limit
            )
        super().__init__("gpt-4.1", config, api_key)

class O4MiniModel(OpenAIResponsesModel):
    """O4-mini model with reasoning capabilities using Responses API"""
    
    def __init__(self, config: ModelConfig = None, api_key: str = None):
        if config is None:
            config = ModelConfig(
                temperature=0.1,
                max_tokens=8000,  # Match Claude's token limit
            )
        super().__init__("o4-mini", config, api_key)
    
    def generate_with_reasoning(self, 
                               prompt: str, 
                               system_message: Optional[str] = None,
                               effort: str = "medium",
                               **kwargs) -> ModelResponse:
        """Generate response with reasoning enabled"""
        reasoning_config = {"effort": effort}
        return self.generate(
            prompt=prompt,
            system_message=system_message,
            reasoning=reasoning_config,
            **kwargs
        )
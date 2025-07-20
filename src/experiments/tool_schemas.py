"""
Tool schemas for converting BaseCoinTool classes to Anthropic function calling format.
This module provides the schemas and execution logic for native tool calling.
"""

from typing import Dict, List, Any, Optional
import logging
from .tools import BaseCoinTool, ToolManager, ToolResult

logger = logging.getLogger(__name__)

def create_coin_tool_schemas(allow_think_tool: bool = True) -> List[Dict[str, Any]]:
    """Create simplified Anthropic function schemas for coin tools
    
    Args:
        allow_think_tool: Whether to include the think tool in the schema
    
    Returns:
        List of tool schemas compatible with Anthropic's function calling API
        Includes flip_coins (costs budget) and optionally think (free) functions
    """
    
    schemas = [
        {
            "name": "flip_coins",
            "description": """Flip coins using a specified tool and return the results. This tool allows you to collect empirical data about coin fairness by performing actual coin flips. 
            
            The tool will:
            - Perform a fixed number of coin flips using the specified tool
            - Return the results in a readable format 
            - Track usage against your total budget (each call costs 1 budget unit)
            - Show your remaining budget after each call
            
            Use this when you need data to evaluate coin fairness. Different tools may have different methodologies and reliability levels.""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "tool_name": {
                        "type": "string",
                        "description": "The name of the coin tool to use (e.g., 'Tool A', 'Tool B', 'Tool C')"
                    }
                },
                "required": ["tool_name"]
            }
        }
    ]
    
    # Conditionally add think tool if enabled
    if allow_think_tool:
        schemas.append({
            "name": "think",
            "description": """Use this tool to pause and think through a problem step by step. This tool does not cost any budget and is intended for reasoning and analysis. 
            
            The tool will:
            - Accept your thinking process as input
            - Not consume any budget units (free to use)
            - Return a simple acknowledgment
            
            Use this when you need to reason through data, plan your approach, or analyze results before making tool calls.""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Your thinking process, analysis, or reasoning"
                    }
                },
                "required": ["content"]
            }
        })
    
    return schemas

class ToolSchemaExecutor:
    """Executor class for handling tool calls via function schemas"""
    
    def __init__(self, tool_manager: ToolManager, flips_per_call: int = 10):
        self.tool_manager = tool_manager
        self.tool_manager.flips_per_call = flips_per_call
        self.native_tool_traces = []
    
    def execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        """Execute a tool call and return formatted result
        
        Args:
            tool_name: Name of the function being called
            tool_input: Input parameters for the function
            
        Returns:
            Formatted string result for the LLM
        """
        logger.info(f"TOOL_SCHEMA_EXECUTOR_REQUEST: tool_name='{tool_name}', tool_input={tool_input}")
        
        try:
            if tool_name == "flip_coins":
                result = self._execute_flip_coins(tool_input)
            elif tool_name == "think":
                result = self._execute_think(tool_input)
            else:
                error_msg = f"Unknown tool: {tool_name}"
                logger.error(f"TOOL_SCHEMA_EXECUTOR_ERROR: {error_msg}")
                raise ValueError(error_msg)
            
            # Capture native tool call trace
            trace_entry = {
                "schema_function": tool_name,
                "input_params": tool_input,
                "output": result,
                "timestamp": len(self.native_tool_traces),
                "success": True
            }
            self.native_tool_traces.append(trace_entry)
            
            logger.info(f"TOOL_SCHEMA_EXECUTOR_SUCCESS: tool_name='{tool_name}', result_length={len(result)}")
            logger.debug(f"TOOL_SCHEMA_EXECUTOR_OUTPUT: tool_name='{tool_name}', result='{result}'")
            return result
            
        except Exception as e:
            # Capture failed native tool call trace
            trace_entry = {
                "schema_function": tool_name,
                "input_params": tool_input,
                "output": None,
                "error": str(e),
                "timestamp": len(self.native_tool_traces),
                "success": False
            }
            self.native_tool_traces.append(trace_entry)
            
            logger.error(f"TOOL_SCHEMA_EXECUTOR_EXCEPTION: tool_name='{tool_name}', error='{str(e)}'")
            raise
    
    def _execute_flip_coins(self, tool_input: Dict[str, Any]) -> str:
        """Execute coin flip tool call"""
        tool_name = tool_input["tool_name"]
        
        try:
            # Get flips_per_call from tool_manager or use default
            flips_per_call = getattr(self.tool_manager, 'flips_per_call', 10)
            result = self.tool_manager.query_tool(tool_name, "flip_coins", flips_per_call=flips_per_call)
            
            # Format result for LLM with remaining budget
            response = f"{result.result}\n"
            response += f"Remaining budget: {self.tool_manager.remaining_budget}"
            
            return response
            
        except Exception as e:
            return f"Error using {tool_name}: {str(e)}\nRemaining budget: {self.tool_manager.remaining_budget}"
    
    def _execute_think(self, tool_input: Dict[str, Any]) -> str:
        """Execute think tool call - no budget cost, just acknowledgment"""
        content = tool_input["content"]
        
        # Log the thinking content for debugging/analysis but don't affect budget
        logger.info(f"THINK_TOOL_USED: content_length={len(content)}")
        logger.debug(f"THINK_TOOL_CONTENT: {content[:200]}...")  # Log first 200 chars
        
        # Return simple acknowledgment
        return "Thinking process recorded. This did not consume any budget."
        
    def get_native_tool_traces(self) -> List[Dict[str, Any]]:
        """Get traces of native tool schema calls"""
        return self.native_tool_traces.copy()

def create_tool_executor(tool_manager: ToolManager, flips_per_call: int = 10) -> callable:
    """Create a tool executor function for use with AnthropicModel.execute_tool_conversation
    
    Args:
        tool_manager: The ToolManager instance to use for execution
        flips_per_call: Number of flips to perform per tool call
        
    Returns:
        Callable that takes (tool_name, tool_input) and returns result string
    """
    executor = ToolSchemaExecutor(tool_manager, flips_per_call)
    return executor.execute_tool
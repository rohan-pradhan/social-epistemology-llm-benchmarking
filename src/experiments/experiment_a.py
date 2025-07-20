import re
import json
from typing import Dict, Any, List, Optional, Union
import numpy as np

from .base import BaseExperiment, TrialResult, ExperimentConfig
from .tools import create_tool_set, ToolManager
from .tool_schemas import create_coin_tool_schemas, create_tool_executor
from ..models.base import BaseModel
from ..models.anthropic_model import AnthropicModel
from ..models.openai_model import OpenAIResponsesModel
from ..evaluation.metrics import FAIRNESS_THRESHOLD, is_fair_coin
from dataclasses import dataclass

@dataclass
class ExperimentAConfig(ExperimentConfig):
    """Configuration specific to Experiment A"""
    allow_profile_queries: bool = True
    persona_mode: bool = False
    max_tool_iterations: int = 10
    flips_per_call: int = 10
    show_underlying_data: bool = False
    fairness_threshold: float = FAIRNESS_THRESHOLD
    display_noise_prob: float = 0.1
    allow_think_tool: Union[bool, str] = True

class ExperimentA(BaseExperiment):
    """Meta-Epistemic Delegation Experiment
    
    Tests LLM's ability to:
    1. Allocate queries across different information sources
    2. Identify reliable vs unreliable sources
    3. Integrate information from multiple noisy sources
    4. Make calibrated probability judgments
    """
    
    def __init__(self, config: ExperimentAConfig = None):
        self.config = config or ExperimentAConfig()
        super().__init__(self.config)
        
    def run_single_trial(self, 
                        model: BaseModel, 
                        trial_id: str, 
                        condition: str = "default") -> TrialResult:
        """Run single trial of Experiment A"""
        
        # Generate trial parameters
        trial_params = self._generate_trial_params(condition)
        ground_truth = self.get_ground_truth(trial_params)
        
        # Create tools for this trial
        tools = create_tool_set(
            fair_coin_bias=trial_params['coin_bias'],
            biased_coin_bias=trial_params['coin_bias'],  # Same coin for all tools
            persona_mode=self.config.persona_mode,
            show_underlying_data=self.config.show_underlying_data,
            display_noise_prob=self.config.display_noise_prob
        )
        
        tool_manager = ToolManager(tools, self.config.total_budget)
        
        try:
            # Use native tool calling if model supports it (AnthropicModel or OpenAIResponsesModel)
            if isinstance(model, (AnthropicModel, OpenAIResponsesModel)):
                parsed_response = self._run_with_native_tools(model, tool_manager, condition)
            else:
                # Fallback to original manual parsing approach
                parsed_response = self._run_with_manual_parsing(model, tool_manager, condition)
            
            # Calculate metrics
            metrics = self._calculate_trial_metrics(
                parsed_response, ground_truth, tool_manager
            )
            
            return TrialResult(
                trial_id=trial_id,
                experiment_type="ExperimentA",
                model_name=model.model_name,
                condition=condition,
                ground_truth=ground_truth,
                model_response={
                    'allocation_response': parsed_response.get('allocation_response', ''),
                    'final_response': parsed_response.get('raw_response', ''),
                    'parsed_probability': parsed_response.get('probability'),
                    'query_results': parsed_response.get('query_results', []),
                    'model_actions': parsed_response.get('model_actions', {}),
                    'raw_model_response': parsed_response.get('raw_model_response', ''),
                },
                metrics=metrics,
                metadata={
                    'trial_params': trial_params,
                    'tool_usage': tool_manager.query_history,
                    'budget_status': tool_manager.get_budget_status(),
                    'tool_call_traces': tool_manager.get_tool_traces(),
                    'native_tool_traces': parsed_response.get('native_tool_traces', [])
                }
            )
            
        except Exception as e:
            raise Exception(f"Trial {trial_id} failed: {str(e)}")
    
    def _generate_trial_params(self, condition: str) -> Dict[str, Any]:
        """Generate parameters for a single trial"""
        # Generate coin bias (same for all tools in this trial)
        coin_bias = self.generate_coin_bias()
        
        return {
            'coin_bias': coin_bias,
            'condition': condition,
            'is_fair': is_fair_coin(coin_bias, self.config.fairness_threshold)
        }
    
    def get_ground_truth(self, trial_params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate ground truth for this trial"""
        return {
            'coin_bias': trial_params['coin_bias'],
            'is_fair': trial_params['is_fair'],
            'fair_probability': 1.0 if trial_params['is_fair'] else 0.0
        }
    
    def _run_with_native_tools(self, model, tool_manager: ToolManager, condition: str) -> Dict[str, Any]:
        """Run experiment using native tool calling"""
        # Create simplified tool schemas and executor
        # If allow_think_tool is "extended_reasoning", disable think tool in schemas
        enable_think_tool = self.config.allow_think_tool is True
        tool_schemas = create_coin_tool_schemas(allow_think_tool=enable_think_tool)
        tool_executor = create_tool_executor(tool_manager, self.config.flips_per_call)
        
        # Generate prompt and system message
        prompt = self._create_native_trial_prompt(tool_manager, condition)
        system_message = self._create_system_message(condition)
        
        # Prepare additional kwargs based on model type
        additional_kwargs = {}
        
        if isinstance(model, AnthropicModel):
            # Anthropic-specific kwargs for extended reasoning
            if self.config.allow_think_tool == "extended_reasoning":
                additional_kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": 1024
                }
        elif isinstance(model, OpenAIResponsesModel):
            # OpenAI-specific kwargs for extended reasoning
            if self.config.allow_think_tool == "extended_reasoning":
                additional_kwargs["reasoning"] = {"effort": "medium"}
        
        # Execute tool conversation
        response = model.execute_tool_conversation(
            initial_prompt=prompt,
            tools=tool_schemas,
            tool_executor=tool_executor,
            system_message=system_message,
            max_iterations=self.config.max_tool_iterations,
            **additional_kwargs
        )
        
        # Parse final probability from response and add tracing info
        parsed_response = self.parse_model_response(response.content)
        
        # Add raw response for debugging
        parsed_response['raw_model_response'] = response.content
        
        # Get tool call traces
        from .tool_schemas import ToolSchemaExecutor
        if hasattr(tool_executor, '__self__') and isinstance(tool_executor.__self__, ToolSchemaExecutor):
            parsed_response['native_tool_traces'] = tool_executor.__self__.get_native_tool_traces()
        
        return parsed_response
    
    def _run_with_manual_parsing(self, model: BaseModel, tool_manager: ToolManager, condition: str) -> Dict[str, Any]:
        """Run experiment using original manual parsing approach"""
        # Generate initial prompt
        prompt = self._create_trial_prompt(tool_manager, condition)
        
        # Get model response
        response = model.generate_with_retry(prompt)
        
        # Parse model's allocation and queries
        model_actions = self._parse_model_actions(response.content, tool_manager)
        
        # Execute the model's queries
        query_results = self._execute_queries(model_actions, tool_manager)
        
        # Get final probability judgment
        final_prompt = self._create_final_judgment_prompt(query_results)
        final_response = model.generate_with_retry(final_prompt)
        
        # Parse final probability and add additional data
        parsed_response = self.parse_model_response(final_response.content)
        parsed_response['allocation_response'] = response.content
        parsed_response['query_results'] = query_results
        parsed_response['model_actions'] = model_actions
        parsed_response['raw_model_response'] = final_response.content
        
        return parsed_response
    
    def _create_system_message(self, condition: str) -> str:
        """Create system message for native tool calling"""
        system_msg = """You are an expert statistician tasked with determining whether a coin is fair (50% heads, 50% tails) or biased. You have access to several tools that can provide information about the coin.

Your approach should be:
1. First, understand what tools are available and their methodologies
2. Strategically allocate your limited budget across tools based on their reliability
3. Gather empirical evidence through coin flips
4. Integrate the evidence to form a calibrated probability judgment

Be methodical and scientific in your approach. Consider the reliability and potential biases of different information sources when making your final assessment."""
        
        if condition == "incentivized":
            system_msg += "\n\nIMPORTANT: Be aware that one or more tools may be incentivized to provide biased information. Consider this when evaluating tool reliability and weighting their results."
        
        return system_msg
    
    def _create_native_trial_prompt(self, tool_manager: ToolManager, condition: str) -> str:
        """Create trial prompt for native tool calling"""
        budget_status = tool_manager.get_budget_status()
        available_tools = tool_manager.get_available_tools()
        
        # Base prompt with tool specifications
        tool_info = "\n\nAvailable tools:\n"
        for i, tool_name in enumerate(available_tools, 1):
            tool_info += f"\n{i}. **{tool_name}**: A tool that can flip coins and provide results\n"
        
        # Add personas if configured
        if self.config.persona_mode:
            tool_info += "\n\nTool personas:\n"
            for tool_name in available_tools:
                tool = tool_manager.tools[tool_name]
                # Extract persona from the tool name (since tools use persona names in persona mode)
                if "Professor" in tool_name:
                    tool_info += f"- **{tool_name}**: Academic statistician\n"
                elif "Lucky" in tool_name:
                    tool_info += f"- **{tool_name}**: Gambling enthusiast\n"
                elif "Casino" in tool_name:
                    tool_info += f"- **{tool_name}**: Casino analytics bot\n"
        
        # Add tool profiles if configured
        if self.config.allow_profile_queries:
            tool_info += "\n\nTool methodologies:\n"
            for tool_name in available_tools:
                tool = tool_manager.tools[tool_name]
                profile_result = tool.get_profile()
                tool_info += f"\n**{tool_name}:**\n{profile_result.result}\n"
        
        prompt = f"""You need to determine whether a coin is fair or biased. You have a budget of {budget_status['total_budget']} tool calls to allocate across your available tools.{tool_info}

Your task:
1. Strategically allocate your budget across the available tools
2. Gather empirical evidence through coin flips
3. Make a final assessment of the probability that the coin is fair

IMPORTANT: Each call to a flip tool uses 1 unit of your budget. Each tool call will return results and show your remaining budget.

Use the flip_coins function to gather evidence from each tool. At the end, provide your conclusion in this format:

FINAL ASSESSMENT:
Based on my investigation, I estimate the probability that this coin is fair to be:
<final_probability>0.65</final_probability>

REASONING: [Explain your reasoning, including which tools you trusted and why]

Begin your investigation now."""

        return prompt
    
    def _create_trial_prompt(self, tool_manager: ToolManager, condition: str) -> str:
        """Create the initial trial prompt"""
        
        tool_names = tool_manager.get_available_tools()
        budget_status = tool_manager.get_budget_status()
        
        prompt = f"""You are participating in a coin fairness evaluation task. Your goal is to determine whether a coin is fair (50% heads, 50% tails) or biased.

You have access to {len(tool_names)} different tools that can provide information about the coin:
{chr(10).join(f"- {name}" for name in tool_names)}

You have a total budget of {budget_status['total_budget']} tool calls to allocate across these tools.

IMPORTANT: Each call to a flip tool uses 1 unit of your budget, regardless of how many coin flips that tool performs internally. The number of flips per call is predetermined - you control how many times to call each tool, not how many flips each call performs.

INSTRUCTIONS:
1. You may optionally query any tool's profile to understand its methodology (free, doesn't cost budget)
2. Allocate your budget across the tools to gather coin flip data
3. Based on the information gathered, estimate the probability that the coin is fair

Please start by deciding how to allocate your queries. Format your allocation as:
ALLOCATION:
- [Tool Name]: [Number of calls] (reason: [brief explanation])

If you want to query a tool's profile first, format as:
PROFILE_QUERY: [Tool Name]

You may query multiple profiles or proceed directly to allocation."""

        if condition == "incentivized":
            prompt += f"\n\nNOTE: One of the tools may be incentivized to provide biased information. Consider this when making your allocation decisions."
        
        return prompt
    
    def _parse_model_actions(self, response_text: str, tool_manager: ToolManager) -> Dict[str, Any]:
        """Parse model's requested actions from response text"""
        actions = {
            'profile_queries': [],
            'allocations': {},
            'reasoning': ""
        }
        
        # Extract profile queries
        profile_pattern = r'PROFILE_QUERY:\s*([^\n]+)'
        profile_matches = re.findall(profile_pattern, response_text, re.IGNORECASE)
        for match in profile_matches:
            tool_name = match.strip()
            if tool_name in tool_manager.get_available_tools():
                actions['profile_queries'].append(tool_name)
        
        # Extract allocations
        allocation_pattern = r'ALLOCATION:(.*?)(?=\n\n|\Z)'
        allocation_match = re.search(allocation_pattern, response_text, re.IGNORECASE | re.DOTALL)
        
        if allocation_match:
            allocation_text = allocation_match.group(1)
            # Parse individual allocations
            tool_pattern = r'-\s*([^:]+):\s*(\d+)'
            tool_matches = re.findall(tool_pattern, allocation_text)
            
            for tool_name, n_calls in tool_matches:
                tool_name = tool_name.strip()
                if tool_name in tool_manager.get_available_tools():
                    actions['allocations'][tool_name] = int(n_calls)
        
        # Extract reasoning
        actions['reasoning'] = response_text
        
        return actions
    
    def _execute_queries(self, actions: Dict[str, Any], tool_manager: ToolManager) -> List[Dict[str, Any]]:
        """Execute the model's requested queries"""
        results = []
        
        # Execute profile queries first (they don't cost budget)
        for tool_name in actions['profile_queries']:
            if self.config.allow_profile_queries:
                try:
                    result = tool_manager.query_tool(tool_name, "profile")
                    results.append({
                        'query_type': 'profile',
                        'tool_name': tool_name,
                        'result': result.result,
                        'success': True
                    })
                except Exception as e:
                    results.append({
                        'query_type': 'profile',
                        'tool_name': tool_name,
                        'result': f"Error: {str(e)}",
                        'success': False
                    })
        
        # Execute flip allocations (now allocation means number of calls, not flips)
        for tool_name, n_calls in actions['allocations'].items():
            if n_calls > 0:
                for _ in range(n_calls):
                    try:
                        result = tool_manager.query_tool(tool_name, "flip_coins", flips_per_call=self.config.flips_per_call)
                        results.append({
                            'query_type': 'flip_coins',
                            'tool_name': tool_name,
                            'n_calls': 1,
                            'n_flips': result.metadata.get('n_flips', self.config.flips_per_call),
                            'result': result.result,
                            'metadata': result.metadata,
                            'success': True
                        })
                    except Exception as e:
                        results.append({
                            'query_type': 'flip_coins',
                            'tool_name': tool_name,
                            'n_calls': 1,
                            'n_flips': 0,
                            'result': f"Error: {str(e)}",
                            'success': False
                        })
                        break  # Stop trying more calls for this tool if one fails
        
        return results
    
    def _create_final_judgment_prompt(self, query_results: List[Dict[str, Any]]) -> str:
        """Create prompt for final probability judgment"""
        
        prompt = "Based on the information you gathered from the tools, please provide your final assessment.\n\n"
        prompt += "QUERY RESULTS:\n"
        
        for i, result in enumerate(query_results, 1):
            if result['success']:
                if result['query_type'] == 'flip_coins':
                    n_calls = result.get('n_calls', 1)
                    n_flips = result.get('n_flips', self.config.flips_per_call)
                    prompt += f"{i}. {result['tool_name']} ({result['query_type']}, {n_calls} call{'s' if n_calls != 1 else ''}, {n_flips} flips):\n"
                else:
                    prompt += f"{i}. {result['tool_name']} ({result['query_type']}):\n"
                prompt += f"   {result['result']}\n\n"
            else:
                prompt += f"{i}. {result['tool_name']} ({result['query_type']}): FAILED\n"
                prompt += f"   {result['result']}\n\n"
        
        prompt += """Please provide your final judgment in this exact format:

ANALYSIS: [Your reasoning about the evidence and tool reliability]

<final_probability>0.65</final_probability>

This means you think there's a 65% chance the coin is fair."""
        
        return prompt
    
    def parse_model_response(self, response_text: str) -> Dict[str, Any]:
        """Parse model's final probability response"""
        result = {
            'probability': None,
            'reasoning': "",
            'raw_response': response_text
        }
        
        # Look for FINAL ASSESSMENT format first (native tool calling)
        assessment_pattern = r'FINAL ASSESSMENT:(.*?)(?=REASONING:|$)'
        assessment_match = re.search(assessment_pattern, response_text, re.IGNORECASE | re.DOTALL)
        
        # Look for REASONING section
        reasoning_pattern = r'REASONING:\s*(.*?)$'
        reasoning_match = re.search(reasoning_pattern, response_text, re.IGNORECASE | re.DOTALL)
        if reasoning_match:
            result['reasoning'] = reasoning_match.group(1).strip()
        
        # Fallback: look for ANALYSIS section (manual parsing)
        if not result['reasoning']:
            analysis_pattern = r'ANALYSIS:\s*(.*?)(?=PROBABILITY:|$)'
            analysis_match = re.search(analysis_pattern, response_text, re.IGNORECASE | re.DOTALL)
            if analysis_match:
                result['reasoning'] = analysis_match.group(1).strip()
        
        # Extract probability from XML tags (primary method)
        xml_pattern = r'<final_probability>\s*([0-9]*\.?[0-9]+)\s*</final_probability>'
        xml_match = re.search(xml_pattern, response_text, re.IGNORECASE)
        
        if xml_match:
            try:
                prob = float(xml_match.group(1))
                if 0 <= prob <= 1:
                    result['probability'] = prob
                elif 1 < prob <= 100:
                    result['probability'] = prob / 100
            except ValueError:
                pass
        
        # Fallback: original regex patterns for backward compatibility
        if result['probability'] is None:
            prob_patterns = [
                r'probability.*?(?:is|to be).*?([0-9]*\.?[0-9]+)',  # "probability that this coin is fair to be: 0.65"
                r'PROBABILITY:\s*([0-9]*\.?[0-9]+)',                # "PROBABILITY: 0.65"
                r'([0-9]*\.?[0-9]+).*?(?:probability|chance)',      # "0.65 probability" or "65% chance"
                r'(?:estimate|assess).*?([0-9]*\.?[0-9]+)',         # "I estimate 0.65"
            ]
            
            for pattern in prob_patterns:
                prob_match = re.search(pattern, response_text, re.IGNORECASE)
                if prob_match:
                    try:
                        prob = float(prob_match.group(1))
                        if 0 <= prob <= 1:
                            result['probability'] = prob
                            break
                        elif 1 < prob <= 100:
                            result['probability'] = prob / 100
                            break
                    except ValueError:
                        continue
        
        return result
    
    def _calculate_trial_metrics(self, 
                               parsed_response: Dict[str, Any],
                               ground_truth: Dict[str, Any],
                               tool_manager: ToolManager) -> Dict[str, float]:
        """Calculate metrics for this trial"""
        from ..evaluation.metrics import brier_score, log_loss
        
        metrics = {}
        
        predicted_prob = parsed_response.get('probability')
        if predicted_prob is not None:
            true_outcome = 1 if ground_truth['is_fair'] else 0
            
            metrics['brier_score'] = brier_score(predicted_prob, true_outcome)
            metrics['log_loss'] = log_loss(predicted_prob, true_outcome)
            metrics['accuracy'] = 1.0 if (predicted_prob >= 0.5) == ground_truth['is_fair'] else 0.0
        else:
            metrics['failed_parse'] = 1.0
            metrics['brier_score'] = 0.25  # Worst possible for uninformed guess
            metrics['log_loss'] = -np.log(0.5)  # Log loss for 50% prediction
            metrics['accuracy'] = 0.0
        
        # Tool usage efficiency metrics
        total_queries = len(tool_manager.query_history)
        metrics['budget_utilization'] = (tool_manager.total_budget - tool_manager.remaining_budget) / tool_manager.total_budget
        metrics['n_queries_made'] = total_queries
        
        # Tool diversity metrics
        tools_used = set(query['tool_name'] for query in tool_manager.query_history)
        metrics['n_tools_used'] = len(tools_used)
        metrics['tool_diversity'] = len(tools_used) / len(tool_manager.tools) if tool_manager.tools else 0
        
        return metrics
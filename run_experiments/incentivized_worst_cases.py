#!/usr/bin/env python3
"""
Incentivized Condition - Testing Worst Performing Configurations

This experiment tests whether providing explicit warnings about potentially biased tools
(the "incentivized" condition) improves model performance on the worst-performing
configurations identified from the main parameter sweep.

Key features:
- Tests 5 worst OpenAI configurations and 5 worst Anthropic configurations
- Uses the "incentivized" condition which warns models about potential tool biases
- Maintains exact parameter matching with original experiments for valid comparison
- Configurations selected based on highest Brier scores (worst calibration)

The incentivized condition adds explicit warnings that:
- The coin flipper might have biases
- Tools might not always provide accurate information
- The model should think critically about the evidence

Usage:
    python incentivized_worst_cases.py
    python incentivized_worst_cases.py --output-dir my_results

Environment Variables:
    ANTHROPIC_API_KEY: Required for Claude model access
    OPENAI_API_KEY: Required for OpenAI model access

Note: The worst configurations are hardcoded based on analysis of the main parameter
sweep results. These represent the parameter combinations where models showed the
poorest calibration in probability estimates.
"""

import os
import json
import logging
import time
import argparse
from typing import List, Dict, Any
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import experiments and models - EXACTLY as in the original experiments
from src.experiments.experiment_a import ExperimentA, ExperimentAConfig
from src.models.anthropic_model import ClaudeSonnet4Model
from src.models.openai_model import GPT41Model, O4MiniModel
from src.evaluation.metrics import compute_experiment_metrics
from src.utils.random import derive_trial_seed, initialize_trial_randomness

def create_experiment_config(persona_mode: bool, 
                           allow_profile_queries: bool, 
                           total_budget: int, 
                           coin_bias: float,
                           allow_think_tool) -> ExperimentAConfig:
    """Create ExperimentAConfig with specified parameters - EXACTLY matching original experiments"""
    return ExperimentAConfig(
        # Fixed parameters - EXACTLY as in parameter sweep
        flips_per_call=10,
        display_noise_prob=0.15,  # NOT 0.1 - this was 0.15 in original
        show_underlying_data=False,
        max_tool_iterations=30,
        num_trials=20,  # num_trials_per_config (though we override to 10 for this experiment)
        
        # Swept parameters
        persona_mode=persona_mode,
        allow_profile_queries=allow_profile_queries,
        total_budget=total_budget,
        allow_think_tool=allow_think_tool,
        
        # Override coin bias generation
        fair_prob=coin_bias,
        bias_range=(coin_bias, coin_bias),  # Force specific bias
        
        # Other settings
        save_results=False,  # We'll handle saving ourselves
        verbose=True
    )

def run_single_config_trials(config: ExperimentAConfig, 
                           model, 
                           config_id: str, 
                           num_trials: int = 10,
                           global_trial_start_index: int = 0,
                           condition: str = "incentivized") -> List[Dict[str, Any]]:
    """Run multiple trials for a single configuration - matching original structure"""
    experiment = ExperimentA(config)
    
    trial_results = []
    
    for trial_idx in range(num_trials):
        trial_id = f"{config_id}_trial_{trial_idx:02d}"
        global_trial_index = global_trial_start_index + trial_idx
        
        try:
            # Derive unique seed for this trial and initialize randomness
            trial_seed = derive_trial_seed(config.random_seed, global_trial_index)
            initialize_trial_randomness(trial_seed)
            
            # Override the coin bias generation to use our fixed value
            original_generate_bias = experiment.generate_coin_bias
            experiment.generate_coin_bias = lambda: config.fair_prob
            
            # Run with incentivized condition - THE ONLY DIFFERENCE FROM ORIGINAL
            result = experiment.run_single_trial(model, trial_id, condition)
            
            # Restore original method
            experiment.generate_coin_bias = original_generate_bias
            
            # Convert TrialResult to dict for easier handling
            trial_data = {
                'trial_id': result.trial_id,
                'experiment_type': result.experiment_type,
                'model_name': result.model_name,
                'condition': result.condition,
                'ground_truth': result.ground_truth,
                'model_response': result.model_response,
                'metrics': result.metrics,
                'metadata': result.metadata,
                'timestamp': result.timestamp,
                'execution_time': result.execution_time
            }
            
            trial_results.append(trial_data)
            
            logger.info(f"Completed trial {trial_idx + 1}/{num_trials} for {config_id}")
            
        except Exception as e:
            logger.error(f"Error in trial {trial_id}: {str(e)}")
            # Continue with next trial
            continue
    
    return trial_results

def get_worst_configurations():
    """
    Returns the worst performing configurations for both models.
    
    These configurations were identified from comprehensive parameter sweep analysis,
    selecting those with the highest Brier scores (indicating poor probability calibration).
    The Brier score measures the mean squared difference between predicted probabilities
    and actual outcomes, where higher scores indicate worse performance.
    
    Returns:
        tuple: (openai_worst_configs, anthropic_worst_configs)
    """
    # 5 worst OpenAI configurations
    openai_worst = [
        {"persona_mode": True, "allow_profile_queries": True, "total_budget": 5, "coin_bias": 0.5, "allow_think_tool": False},  # p1_q1_b5_c0.5_t0 - Brier: 0.562
        {"persona_mode": False, "allow_profile_queries": True, "total_budget": 5, "coin_bias": 0.5, "allow_think_tool": False},  # p0_q1_b5_c0.5_t0 - Brier: 0.486
        {"persona_mode": True, "allow_profile_queries": False, "total_budget": 5, "coin_bias": 0.6, "allow_think_tool": "extended_reasoning"},  # p1_q0_b5_c0.6_tE - Brier: 0.460
        {"persona_mode": False, "allow_profile_queries": True, "total_budget": 10, "coin_bias": 0.5, "allow_think_tool": True},  # p0_q1_b10_c0.5_t1 - Brier: 0.441
        {"persona_mode": True, "allow_profile_queries": False, "total_budget": 5, "coin_bias": 0.5, "allow_think_tool": True},  # p1_q0_b5_c0.5_t1 - Brier: 0.435
    ]
    
    # 5 worst Anthropic configurations
    anthropic_worst = [
        {"persona_mode": True, "allow_profile_queries": False, "total_budget": 10, "coin_bias": 0.5, "allow_think_tool": True},  # p1_q0_b10_c0.5_t1 - Brier: 0.419
        {"persona_mode": False, "allow_profile_queries": True, "total_budget": 5, "coin_bias": 0.5, "allow_think_tool": True},  # p0_q1_b5_c0.5_t1 - Brier: 0.412
        {"persona_mode": False, "allow_profile_queries": True, "total_budget": 5, "coin_bias": 0.5, "allow_think_tool": "extended_reasoning"},  # p0_q1_b5_c0.5_tE - Brier: 0.392
        {"persona_mode": False, "allow_profile_queries": False, "total_budget": 5, "coin_bias": 0.25, "allow_think_tool": False},  # p0_q0_b5_c0.25_t0 - Brier: 0.387
        {"persona_mode": False, "allow_profile_queries": True, "total_budget": 10, "coin_bias": 0.5, "allow_think_tool": "extended_reasoning"},  # p0_q1_b10_c0.5_tE - Brier: 0.366
    ]
    
    return openai_worst, anthropic_worst

def run_anthropic_configurations(configurations: List[Dict[str, Any]], start_index: int = 0):
    """Run all configurations for Anthropic model - EXACTLY as in parameter sweep"""
    all_results = []
    global_trial_index = start_index * 10  # 10 trials per config
    
    # Create model once for all Anthropic runs - EXACTLY as in original
    model = ClaudeSonnet4Model()
    logger.info("Using Claude Sonnet 4 model")
    
    for config_idx, config_params in enumerate(configurations):
        config_number = start_index + config_idx + 1
        
        # Generate config ID - EXACTLY as in original
        p = "1" if config_params["persona_mode"] else "0"
        q = "1" if config_params["allow_profile_queries"] else "0"
        b = str(config_params["total_budget"])
        c = str(config_params["coin_bias"])
        
        # Handle think tool parameter - EXACTLY as in original
        if config_params["allow_think_tool"] == "extended_reasoning":
            t = "E"
        elif config_params["allow_think_tool"]:
            t = "1"
        else:
            t = "0"
        
        config_id = f"config_{config_number:02d}_p{p}_q{q}_b{b}_c{c}_t{t}_incentivized"
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Running configuration {config_number}/{start_index + len(configurations)}: {config_id}")
        logger.info(f"Parameters: {config_params}")
        logger.info(f"{'='*60}")
        
        # Create configuration
        config = create_experiment_config(**config_params)
        
        # Run trials for this configuration
        start_time = time.time()
        trial_results = run_single_config_trials(
            config, 
            model, 
            config_id, 
            num_trials=10,
            global_trial_start_index=global_trial_index,
            condition="incentivized"  # THE ONLY ADDITION
        )
        elapsed_time = time.time() - start_time
        
        # Aggregate metrics using proper statistical methods
        aggregated_metrics = compute_experiment_metrics(trial_results)
        
        # Store configuration summary
        config_result = {
            "config_id": config_id,
            "model_name": model.model_name,  # Use actual model name from instance
            "parameters": config_params,
            "condition": "incentivized",
            "num_trials": len(trial_results),
            "aggregated_metrics": aggregated_metrics,
            "elapsed_time": elapsed_time,
            "trials": trial_results
        }
        
        all_results.append(config_result)
        
        logger.info(f"Completed {config_id} in {elapsed_time:.2f}s")
        logger.info(f"Mean Brier score: {aggregated_metrics['mean_brier_score']:.4f}")
        
        global_trial_index += 10
    
    return all_results

def run_openai_configurations(configurations: List[Dict[str, Any]], start_index: int = 0):
    """Run all configurations for OpenAI models - EXACTLY as in parameter sweep"""
    all_results = []
    global_trial_index = start_index * 10  # 10 trials per config
    
    for config_idx, config_params in enumerate(configurations):
        config_number = start_index + config_idx + 1
        
        # Create model based on think tool setting - EXACTLY as in original
        if config_params["allow_think_tool"] == "extended_reasoning":
            # Use o4-mini with reasoning for extended reasoning
            model = O4MiniModel()
        else:
            # Use gpt-4.1 for other experiments
            model = GPT41Model()
        
        logger.info(f"Using model: {model.model_name}")
        
        # Handle allow_think_tool config ID formatting - EXACTLY as in original
        if config_params["allow_think_tool"] == "extended_reasoning":
            think_tool_str = "E"
        else:
            think_tool_str = str(int(config_params["allow_think_tool"]))
        
        # Generate config ID
        config_id = f"config_{config_number:02d}_p{int(config_params['persona_mode'])}_q{int(config_params['allow_profile_queries'])}_b{config_params['total_budget']}_c{config_params['coin_bias']:.1f}_t{think_tool_str}_incentivized"
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Running configuration {config_number}/{start_index + len(configurations)}: {config_id}")
        logger.info(f"  persona_mode={config_params['persona_mode']}, allow_profile_queries={config_params['allow_profile_queries']}")
        logger.info(f"  total_budget={config_params['total_budget']}, coin_bias={config_params['coin_bias']}, allow_think_tool={config_params['allow_think_tool']}")
        logger.info(f"{'='*60}")
        
        # Create configuration
        config = create_experiment_config(**config_params)
        
        # Run trials for this configuration
        start_time = time.time()
        trial_results = run_single_config_trials(
            config, 
            model, 
            config_id, 
            num_trials=10,
            global_trial_start_index=global_trial_index,
            condition="incentivized"  # THE ONLY ADDITION
        )
        elapsed_time = time.time() - start_time
        
        # Aggregate metrics using proper statistical methods
        aggregated_metrics = compute_experiment_metrics(trial_results)
        
        # Store configuration summary
        config_result = {
            "config_id": config_id,
            "model_name": model.model_name,  # Use actual model name from instance
            "parameters": config_params,
            "condition": "incentivized",
            "num_trials": len(trial_results),
            "aggregated_metrics": aggregated_metrics,
            "elapsed_time": elapsed_time,
            "trials": trial_results
        }
        
        all_results.append(config_result)
        
        logger.info(f"Completed {config_id} in {elapsed_time:.2f}s")
        logger.info(f"Mean Brier score: {aggregated_metrics['mean_brier_score']:.4f}")
        
        global_trial_index += 10
    
    return all_results

def main(output_dir: str = "results"):
    """
    Main function to run the incentivized experiment.
    
    Args:
        output_dir: Directory to save results (default: "results")
    """
    logger.info("=" * 80)
    logger.info("Starting Incentivized Condition - Worst Performing Configurations")
    logger.info("=" * 80)
    
    # Check for API keys - EXACTLY as in originals
    if not os.getenv("ANTHROPIC_API_KEY"):
        logger.error("ANTHROPIC_API_KEY not found. Please set your Anthropic API key.")
        return
        
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not found. Please set your OpenAI API key.")
        return
    
    # Get worst configurations
    openai_worst, anthropic_worst = get_worst_configurations()
    
    logger.info(f"Testing {len(openai_worst)} worst OpenAI configurations")
    logger.info(f"Testing {len(anthropic_worst)} worst Anthropic configurations")
    logger.info("Each configuration will run 10 trials with incentivized condition")
    logger.info("Using display_noise_prob=0.15 (matching original experiments)")
    
    all_results = []
    
    # Run OpenAI configurations
    logger.info("\n" + "=" * 80)
    logger.info("Running OpenAI configurations...")
    logger.info("=" * 80)
    
    openai_results = run_openai_configurations(openai_worst, start_index=0)
    all_results.extend(openai_results)
    
    # Run Anthropic configurations
    logger.info("\n" + "=" * 80)
    logger.info("Running Anthropic configurations...")
    logger.info("=" * 80)
    
    anthropic_results = run_anthropic_configurations(anthropic_worst, start_index=len(openai_worst))
    all_results.extend(anthropic_results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/tier1_incentivized_worst_configs_{timestamp}.json"
    
    output_data = {
        "experiment": "tier1_incentivized_worst_configs",
        "timestamp": timestamp,
        "description": "Testing 5 worst OpenAI and 5 worst Anthropic configurations with incentivized condition",
        "experimental_parameters": {
            "flips_per_call": 10,
            "display_noise_prob": 0.15,
            "show_underlying_data": False,
            "max_tool_iterations": 30,
            "num_trials_per_config": 10,
            "condition": "incentivized"
        },
        "models": {
            "openai": "gpt-4.1 / o4-mini (for extended reasoning)",
            "anthropic": "claude-sonnet-4"
        },
        "total_configurations": len(all_results),
        "configuration_results": all_results
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"\nResults saved to: {output_file}")
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 80)
    
    total_time = sum(r["elapsed_time"] for r in all_results)
    total_trials = sum(r["num_trials"] for r in all_results)
    
    logger.info(f"Total configurations run: {len(all_results)}")
    logger.info(f"Total trials run: {total_trials}")
    logger.info(f"Total experiment time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    logger.info(f"Average time per configuration: {total_time/len(all_results):.2f}s")
    
    # Performance summary by model type
    model_names = set(r["model_name"] for r in all_results)
    for model_name in model_names:
        model_results = [r for r in all_results if r["model_name"] == model_name]
        if model_results:
            model_time = sum(r["elapsed_time"] for r in model_results)
            avg_brier = sum(r["aggregated_metrics"]["mean_brier_score"] for r in model_results) / len(model_results)
            logger.info(f"\n{model_name}:")
            logger.info(f"  Configurations: {len(model_results)}")
            logger.info(f"  Total time: {model_time:.2f}s")
            logger.info(f"  Average time: {model_time/len(model_results):.2f}s")
            logger.info(f"  Average Brier score: {avg_brier:.4f}")

def parse_args():
    """Parse command line arguments for the experiment."""
    parser = argparse.ArgumentParser(
        description="Run incentivized experiment on worst performing configurations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""This experiment tests whether explicit warnings about biased tools
improve model performance on configurations that showed poor calibration
in the main parameter sweep.
        """
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save results (default: results)"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(output_dir=args.output_dir)
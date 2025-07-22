#!/usr/bin/env python3
"""
Parameter Sweep for Social Epistemology Benchmark (OpenAI Models)

This script conducts a comprehensive parameter sweep for the benchmark using OpenAI models,
testing their ability to assess the fairness of a coin based on noisy observations.

Key features:
- Uses GPT-4.1 for standard experiments
- Uses O4-mini with reasoning capabilities for extended reasoning experiments
- Varies multiple parameters to understand their impact on model performance:
  * persona_mode: Whether the model adopts a specific persona
  * allow_profile_queries: Whether the model can query information about the coin flipper
  * total_budget: Number of coin flip observations (5 or 10)
  * coin_bias: The actual bias of the coin (0.25, 0.5, or 0.6)
  * allow_think_tool: Thinking tool access (False, True, or "extended_reasoning")

The script supports resuming from partial results if interrupted, with progress saved
after each configuration. Results include both detailed trial data and aggregated metrics.

Usage:
    python parameter_sweep_openai.py
    python parameter_sweep_openai.py --resume-detailed <path> --resume-summary <path>

Environment Variables:
    OPENAI_API_KEY: Required for OpenAI model access
"""

import os
import json
import logging
import itertools
import argparse
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import experiments, metrics, and models
from src.experiments.experiment_a import ExperimentA, ExperimentAConfig
from src.evaluation.metrics import compute_experiment_metrics
from src.utils.random import derive_trial_seed, initialize_trial_randomness
from src.models.openai_model import GPT41Model, O4MiniModel

def create_openai_model(allow_think_tool):
    """
    Create appropriate OpenAI model based on think tool setting.
    
    Args:
        allow_think_tool: Think tool setting (False, True, or "extended_reasoning")
    
    Returns:
        Model instance (O4MiniModel for extended reasoning, GPT41Model otherwise)
    """
    if allow_think_tool == "extended_reasoning":
        # Use o4-mini with reasoning for extended reasoning
        return O4MiniModel()
    else:
        # Use gpt-4.1 for other experiments
        return GPT41Model()

def create_experiment_config(persona_mode: bool, 
                           allow_profile_queries: bool, 
                           total_budget: int, 
                           coin_bias: float,
                           allow_think_tool) -> ExperimentAConfig:
    """Create ExperimentAConfig with specified parameters"""
    return ExperimentAConfig(
        # Fixed parameters
        flips_per_call=10,
        display_noise_prob=0.15,
        show_underlying_data=False,
        max_tool_iterations=30,
        num_trials=10,  # num_trials_per_config
        
        # Swept parameters
        persona_mode=persona_mode,
        allow_profile_queries=allow_profile_queries,
        total_budget=total_budget,
        allow_think_tool=allow_think_tool,
        
        # Set the coin bias
        coin_bias=coin_bias,
        
        # Other settings
        save_results=False,  # We'll handle saving ourselves
        verbose=True
    )

def run_single_config_trials(config: ExperimentAConfig, 
                           model, 
                           config_id: str, 
                           num_trials: int = 10,
                           global_trial_start_index: int = 0) -> List[Dict[str, Any]]:
    """Run multiple trials for a single configuration"""
    experiment = ExperimentA(config)
    
    trial_results = []
    
    for trial_idx in range(num_trials):
        trial_id = f"{config_id}_trial_{trial_idx:02d}"
        global_trial_index = global_trial_start_index + trial_idx
        
        try:
            # Derive unique seed for this trial and initialize randomness
            trial_seed = derive_trial_seed(config.random_seed, global_trial_index)
            initialize_trial_randomness(trial_seed)
            
            result = experiment.run_single_trial(model, trial_id, "default")
            
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
            
            # Add trial seed to metadata for reproducibility
            trial_data['metadata']['trial_seed'] = trial_seed
            trial_data['metadata']['global_trial_index'] = global_trial_index
            
            trial_results.append(trial_data)
            
            if (trial_idx + 1) % 5 == 0:
                logger.info(f"  Completed {trial_idx + 1}/{num_trials} trials for {config_id}")
                
        except Exception as e:
            logger.error(f"Trial {trial_id} failed: {str(e)}")
            # Add failed trial
            trial_results.append({
                'trial_id': trial_id,
                'experiment_type': 'ExperimentA',
                'model_name': model.model_name,
                'condition': 'default',
                'ground_truth': {},
                'model_response': {'error': str(e)},
                'metrics': {'failed': 1.0},
                'metadata': {
                    'error': str(e),
                    'trial_seed': trial_seed,
                    'global_trial_index': global_trial_index
                },
                'timestamp': datetime.now().isoformat(),
                'execution_time': None
            })
    
    return trial_results


def save_intermediate_results(all_results, summary_results, sweep_params, model, timestamp, config_count, total_configs, output_dir="results"):
    """
    Save intermediate results to files for experiment resumption.
    
    Args:
        all_results: List of all trial results
        summary_results: List of configuration summaries
        sweep_params: Dictionary of sweep parameters
        model: Model instance being used
        timestamp: Timestamp string for file naming
        config_count: Number of completed configurations
        total_configs: Total number of configurations
        output_dir: Directory to save results
    
    Returns:
        Tuple of (detailed_filename, summary_filename)
    """
    # Save partial detailed results
    detailed_filename = f"{output_dir}/tier1_parameter_sweep_openai_detailed_{timestamp}_partial.json"
    with open(detailed_filename, 'w') as f:
        json.dump({
            'experiment_info': {
                'experiment_type': 'ParameterSweep_OpenAI',
                'model_name': model.model_name,
                'timestamp': timestamp,
                'completed_configurations': config_count,
                'total_configurations': total_configs,
                'total_trials': len(all_results),
                'status': 'in_progress'
            },
            'sweep_parameters': sweep_params,
            'detailed_results': all_results
        }, f, indent=2)
    
    # Save partial summary results
    summary_filename = f"{output_dir}/tier1_parameter_sweep_openai_summary_{timestamp}_partial.json"
    with open(summary_filename, 'w') as f:
        json.dump({
            'experiment_info': {
                'experiment_type': 'ParameterSweep_OpenAI',
                'model_name': model.model_name,
                'timestamp': timestamp,
                'completed_configurations': config_count,
                'total_configurations': total_configs,
                'status': 'in_progress'
            },
            'sweep_parameters': sweep_params,
            'configuration_results': summary_results
        }, f, indent=2)
    
    return detailed_filename, summary_filename

def load_partial_results(detailed_path: str, summary_path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any], str, int]:
    """Load partial results from files and return the data needed to resume"""
    logger.info(f"Loading partial results from {detailed_path} and {summary_path}")
    
    # Load detailed results
    with open(detailed_path, 'r') as f:
        detailed_data = json.load(f)
    
    # Load summary results
    with open(summary_path, 'r') as f:
        summary_data = json.load(f)
    
    all_results = detailed_data['detailed_results']
    summary_results = summary_data['configuration_results']
    sweep_params = detailed_data['sweep_parameters']
    timestamp = detailed_data['experiment_info']['timestamp']
    completed_configs = detailed_data['experiment_info']['completed_configurations']
    
    logger.info(f"Loaded {len(all_results)} trial results from {completed_configs} completed configurations")
    
    return all_results, summary_results, sweep_params, timestamp, completed_configs

def run_parameter_sweep(resume_detailed_path: Optional[str] = None, 
                       resume_summary_path: Optional[str] = None,
                       output_dir: str = "results"):
    """
    Run the complete parameter sweep experiment with OpenAI models.
    
    Args:
        resume_detailed_path: Path to partial detailed results file to resume from
        resume_summary_path: Path to partial summary results file to resume from  
        output_dir: Directory to save results (default: "results")
    
    Returns:
        None
    """
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not found. Please set your OpenAI API key in environment variables.")
        return
    
    # Define parameter sweep
    sweep_params = {
        'persona_mode': [False, True],
        'allow_profile_queries': [False, True],
        'total_budget': [5, 10],
        'coin_bias': [0.25,0.5,0.6],
        'allow_think_tool': [False, True, "extended_reasoning"]
    }
    
    # Generate all combinations
    param_combinations = list(itertools.product(
        sweep_params['persona_mode'],
        sweep_params['allow_profile_queries'],
        sweep_params['total_budget'],
        sweep_params['coin_bias'],
        sweep_params['allow_think_tool']
    ))
    
    # Create results directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Handle resume mode
    if resume_detailed_path and resume_summary_path:
        logger.info("Resuming from partial results")
        all_results, summary_results, loaded_sweep_params, timestamp, completed_configs = load_partial_results(
            resume_detailed_path, resume_summary_path
        )
        
        # Verify sweep parameters match
        if loaded_sweep_params != sweep_params:
            logger.warning("Loaded sweep parameters don't match current parameters. Proceeding with loaded parameters.")
            sweep_params = loaded_sweep_params
        
        # Calculate global trial counter from existing results
        global_trial_counter = len(all_results)
        
        # Start from the next configuration
        start_config_index = completed_configs
        
        logger.info(f"Resuming from configuration {start_config_index + 1}/{len(param_combinations)}")
        logger.info(f"Already completed {len(all_results)} trials")
    else:
        logger.info(f"Starting new parameter sweep with {len(param_combinations)} configurations")
        logger.info(f"Total expected trials: {len(param_combinations) * 10}")
        
        # Generate timestamp for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Results storage
        all_results = []
        summary_results = []
        global_trial_counter = 0  # Track global trial index across all configs
        start_config_index = 0
    
    # Run each configuration
    for i, (persona_mode, allow_profile_queries, total_budget, coin_bias, allow_think_tool) in enumerate(param_combinations):
        # Skip configurations that were already completed
        if i < start_config_index:
            continue
            
        # Create model for this configuration (model selection based on allow_think_tool)
        model = create_openai_model(allow_think_tool)
        logger.info(f"Using model: {model.model_name}")
        
        # Handle allow_think_tool config ID formatting
        if allow_think_tool == "extended_reasoning":
            think_tool_str = "E"
        else:
            think_tool_str = str(int(allow_think_tool))
        
        config_id = f"config_{i:02d}_p{int(persona_mode)}_q{int(allow_profile_queries)}_b{total_budget}_c{coin_bias:.1f}_t{think_tool_str}"
        
        logger.info(f"Running configuration {i+1}/{len(param_combinations)}: {config_id}")
        logger.info(f"  persona_mode={persona_mode}, allow_profile_queries={allow_profile_queries}")
        logger.info(f"  total_budget={total_budget}, coin_bias={coin_bias}, allow_think_tool={allow_think_tool}")
        
        # Create configuration
        config = create_experiment_config(persona_mode, allow_profile_queries, total_budget, coin_bias, allow_think_tool)
        
        # Run trials for this configuration
        trial_results = run_single_config_trials(config, model, config_id, num_trials=10, global_trial_start_index=global_trial_counter)
        
        # Update global trial counter
        global_trial_counter += 10
        
        # Aggregate metrics using proper statistical methods
        aggregated_metrics = compute_experiment_metrics(trial_results)
        
        # Store configuration summary
        config_summary = {
            'config_id': config_id,
            'config_index': i,
            'parameters': {
                'persona_mode': persona_mode,
                'allow_profile_queries': allow_profile_queries,
                'total_budget': total_budget,
                'coin_bias': coin_bias,
                'allow_think_tool': allow_think_tool
            },
            'model_info': {
                'model_name': model.model_name,
                'use_reasoning': allow_think_tool == "extended_reasoning"
            },
            'aggregated_metrics': aggregated_metrics,
            'trial_seeds': [trial['metadata']['trial_seed'] for trial in trial_results],
            'global_trial_indices': [trial['metadata']['global_trial_index'] for trial in trial_results],
            'timestamp': datetime.now().isoformat()
        }
        
        summary_results.append(config_summary)
        
        # Store all trial data
        for trial in trial_results:
            trial['config_id'] = config_id
            trial['config_parameters'] = config_summary['parameters']
            trial['model_info'] = config_summary['model_info']
            all_results.append(trial)
        
        logger.info(f"  Configuration completed. Success rate: {aggregated_metrics.get('success_rate', 0):.2f}")
        logger.info(f"  Mean Brier score: {aggregated_metrics.get('mean_brier_score', 'N/A')}")
        logger.info(f"  Accuracy: {aggregated_metrics.get('accuracy', 'N/A')}")
        
        # Save intermediate results after each configuration
        detailed_partial, summary_partial = save_intermediate_results(
            all_results, summary_results, sweep_params, model, timestamp, i+1, len(param_combinations), output_dir
        )
        logger.info(f"  Intermediate results saved: {detailed_partial}")
        print()
    
    # Save final results (without _partial suffix)
    # Save detailed results
    detailed_filename = f"{output_dir}/tier1_parameter_sweep_openai_detailed_{timestamp}.json"
    
    with open(detailed_filename, 'w') as f:
        json.dump({
            'experiment_info': {
                'experiment_type': 'ParameterSweep_OpenAI',
                'timestamp': timestamp,
                'total_configurations': len(param_combinations),
                'total_trials': len(all_results),
                'status': 'completed'
            },
            'sweep_parameters': sweep_params,
            'detailed_results': all_results
        }, f, indent=2)
    
    # Save summary results
    summary_filename = f"{output_dir}/tier1_parameter_sweep_openai_summary_{timestamp}.json"
    
    with open(summary_filename, 'w') as f:
        json.dump({
            'experiment_info': {
                'experiment_type': 'ParameterSweep_OpenAI',
                'timestamp': timestamp,
                'total_configurations': len(param_combinations),
                'status': 'completed'
            },
            'sweep_parameters': sweep_params,
            'configuration_results': summary_results
        }, f, indent=2)
    
    # Print summary table
    print("\n" + "="*100)
    print("PARAMETER SWEEP SUMMARY (OPENAI)")
    print("="*100)
    print(f"Total Configurations: {len(param_combinations)}")
    print(f"Total Trials: {len(all_results)}")
    print()
    
    print(f"{'Config':<8} {'Persona':<7} {'Profile':<7} {'Budget':<6} {'Bias':<4} {'Think':<7} {'Model':<12} {'Success':<7} {'Brier':<6} {'Accuracy':<8}")
    print("-" * 92)
    
    for config in summary_results:
        params = config['parameters']
        metrics = config['aggregated_metrics']
        model_info = config['model_info']
        
        # Handle both old and new metric names for backward compatibility
        success_rate = metrics.get('success_rate', 0)
        brier_score = metrics.get('mean_brier_score', 'N/A')
        accuracy = metrics.get('accuracy', 'N/A')
        
        print(f"{config['config_index']:>6}   "
              f"{'Yes' if params['persona_mode'] else 'No':<7} "
              f"{'Yes' if params['allow_profile_queries'] else 'No':<7} "
              f"{params['total_budget']:>6} "
              f"{params['coin_bias']:<4.1f} "
              f"{('ExtReas' if params['allow_think_tool'] == 'extended_reasoning' else ('Yes' if params['allow_think_tool'] else 'No')):<7} "
              f"{model_info['model_name']:<12} "
              f"{success_rate:<7.2f} "
              f"{brier_score if isinstance(brier_score, str) else f'{brier_score:<6.3f}'} "
              f"{accuracy if isinstance(accuracy, str) else f'{accuracy:<8.2f}'}")
    
    print("\nFiles saved:")
    print(f"  Detailed results: {detailed_filename}")
    print(f"  Summary results: {summary_filename}")

def parse_args():
    """Parse command line arguments for the experiment."""
    parser = argparse.ArgumentParser(
        description="Run Experiment A parameter sweep with OpenAI models and optional resume functionality.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Run new experiment
  python parameter_sweep_openai.py
  
  # Resume from partial results
  python parameter_sweep_openai.py --resume-detailed results/partial_detailed.json --resume-summary results/partial_summary.json
  
  # Save to custom directory
  python parameter_sweep_openai.py --output-dir my_results
        """
    )
    parser.add_argument(
        "--resume-detailed",
        type=str,
        help="Path to partial detailed results file to resume from"
    )
    parser.add_argument(
        "--resume-summary", 
        type=str,
        help="Path to partial summary results file to resume from"
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
    
    if args.resume_detailed and args.resume_summary:
        # Validate that both files exist
        if not os.path.exists(args.resume_detailed):
            logger.error(f"Detailed results file not found: {args.resume_detailed}")
            exit(1)
        if not os.path.exists(args.resume_summary):
            logger.error(f"Summary results file not found: {args.resume_summary}")
            exit(1)
        
        run_parameter_sweep(args.resume_detailed, args.resume_summary, args.output_dir)
    elif args.resume_detailed or args.resume_summary:
        logger.error("Both --resume-detailed and --resume-summary must be provided together")
        exit(1)
    else:
        run_parameter_sweep(output_dir=args.output_dir)
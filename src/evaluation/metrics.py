import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from scipy import stats
import logging

logger = logging.getLogger(__name__)

# Global constant for fairness threshold
FAIRNESS_THRESHOLD = 0.01

def brier_score(predicted_prob: float, actual_outcome: int) -> float:
    """Calculate Brier score for binary prediction
    
    Args:
        predicted_prob: Predicted probability of positive outcome (0-1)
        actual_outcome: Actual outcome (0 or 1)
        
    Returns:
        Brier score (lower is better, 0 is perfect)
    """
    if not 0 <= predicted_prob <= 1:
        logger.warning(f"Predicted probability outside [0,1]: {predicted_prob}")
        predicted_prob = max(0, min(1, predicted_prob))
    
    return (predicted_prob - actual_outcome) ** 2

def log_loss(predicted_prob: float, actual_outcome: int) -> float:
    """Calculate log loss for binary prediction
    
    Args:
        predicted_prob: Predicted probability of positive outcome (0-1)
        actual_outcome: Actual outcome (0 or 1)
        
    Returns:
        Log loss (lower is better)
    """
    # Clip probabilities to avoid log(0)
    epsilon = 1e-15
    predicted_prob = max(epsilon, min(1 - epsilon, predicted_prob))
    
    if actual_outcome == 1:
        return -np.log(predicted_prob)
    else:
        return -np.log(1 - predicted_prob)

def calibration_error(predictions: List[float], outcomes: List[int], n_bins: int = 10) -> Tuple[float, Dict[str, Any]]:
    """Calculate calibration error (Expected Calibration Error)
    
    Args:
        predictions: List of predicted probabilities
        outcomes: List of actual outcomes (0 or 1)
        n_bins: Number of bins for calibration analysis
        
    Returns:
        Tuple of (calibration_error, calibration_details)
    """
    if len(predictions) != len(outcomes):
        raise ValueError("Predictions and outcomes must have same length")
    
    predictions = np.array(predictions)
    outcomes = np.array(outcomes)
    
    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    bin_details = []
    
    for i, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
        # Find predictions in this bin
        # Use half-open intervals [lower, upper) except for the last bin which is [lower, upper]
        if i == len(bin_lowers) - 1:  # Last bin includes upper boundary
            in_bin = (predictions >= bin_lower) & (predictions <= bin_upper)
        else:  # All other bins exclude upper boundary
            in_bin = (predictions >= bin_lower) & (predictions < bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = outcomes[in_bin].mean()
            avg_confidence_in_bin = predictions[in_bin].mean()
            
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            bin_details.append({
                'bin_lower': bin_lower,
                'bin_upper': bin_upper,
                'prop_in_bin': prop_in_bin,
                'accuracy_in_bin': accuracy_in_bin,
                'avg_confidence_in_bin': avg_confidence_in_bin,
                'calibration_error': abs(avg_confidence_in_bin - accuracy_in_bin)
            })
        else:
            bin_details.append({
                'bin_lower': bin_lower,
                'bin_upper': bin_upper,
                'prop_in_bin': 0,
                'accuracy_in_bin': None,
                'avg_confidence_in_bin': None,
                'calibration_error': 0
            })
    
    return ece, {'bins': bin_details, 'n_samples': len(predictions)}

def accuracy_score(predictions: List[float], outcomes: List[int], threshold: float = 0.5) -> float:
    """Calculate binary accuracy score
    
    Args:
        predictions: List of predicted probabilities
        outcomes: List of actual outcomes (0 or 1)
        threshold: Threshold for converting probabilities to binary predictions
        
    Returns:
        Accuracy score (0-1, higher is better)
    """
    if len(outcomes) == 0:
        return 0.0
    
    binary_predictions = [1 if p >= threshold else 0 for p in predictions]
    return sum(bp == outcome for bp, outcome in zip(binary_predictions, outcomes)) / len(outcomes)

def mean_brier_score(predictions: List[float], outcomes: List[int]) -> float:
    """Calculate mean Brier score across multiple predictions"""
    if len(predictions) == 0:
        return np.nan
    return np.mean([brier_score(pred, outcome) for pred, outcome in zip(predictions, outcomes)])

def mean_log_loss(predictions: List[float], outcomes: List[int]) -> float:
    """Calculate mean log loss across multiple predictions"""
    if len(predictions) == 0:
        return np.nan
    return np.mean([log_loss(pred, outcome) for pred, outcome in zip(predictions, outcomes)])

def confidence_interval(values: List[float], confidence_level: float = 0.95) -> Tuple[float, float]:
    """Calculate confidence interval for a list of values
    
    Args:
        values: List of numeric values
        confidence_level: Confidence level (0-1)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if len(values) == 0:
        return (0, 0)
    
    alpha = 1 - confidence_level
    sorted_values = sorted(values)
    n = len(sorted_values)
    
    lower_idx = int(alpha / 2 * n)
    upper_idx = int((1 - alpha / 2) * n)
    upper_idx = min(upper_idx, n - 1)
    
    return (sorted_values[lower_idx], sorted_values[upper_idx])

def binomial_confidence_interval(successes: int, trials: int, confidence_level: float = 0.95) -> Tuple[float, float]:
    """Calculate binomial confidence interval for accuracy using Wilson score interval
    
    Args:
        successes: Number of successful predictions
        trials: Total number of trials
        confidence_level: Confidence level (0-1)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if trials == 0:
        return (0, 0)
    
    alpha = 1 - confidence_level
    p_hat = successes / trials
    z = stats.norm.ppf(1 - alpha/2)
    
    # Wilson score interval (more accurate than normal approximation)
    denominator = 1 + z**2 / trials
    center = (p_hat + z**2 / (2 * trials)) / denominator
    margin = z * np.sqrt(p_hat * (1 - p_hat) / trials + z**2 / (4 * trials**2)) / denominator
    
    return (max(0, center - margin), min(1, center + margin))

def belief_shift_metrics(initial_beliefs: List[float], final_beliefs: List[float]) -> Dict[str, float]:
    """Calculate metrics for belief updating/shift
    
    Args:
        initial_beliefs: List of initial belief probabilities
        final_beliefs: List of final belief probabilities
        
    Returns:
        Dictionary of belief shift metrics
    """
    if len(initial_beliefs) != len(final_beliefs):
        raise ValueError("Initial and final beliefs must have same length")
    
    shifts = [abs(final - initial) for initial, final in zip(initial_beliefs, final_beliefs)]
    
    return {
        'mean_absolute_shift': np.mean(shifts),
        'max_shift': max(shifts),
        'min_shift': min(shifts),
        'std_shift': np.std(shifts),
        'proportion_shifted': sum(1 for s in shifts if s > 0.01) / len(shifts)  # Threshold for meaningful shift
    }

def convergence_metrics(agent_beliefs: List[List[float]]) -> Dict[str, float]:
    """Calculate convergence metrics for multi-agent scenarios
    
    Args:
        agent_beliefs: List of belief lists, one per agent
        
    Returns:
        Dictionary of convergence metrics
    """
    if not agent_beliefs or not agent_beliefs[0]:
        return {'variance': 0, 'range': 0, 'std': 0}
    
    # Calculate variance across agents for each time point
    n_agents = len(agent_beliefs)
    n_timepoints = len(agent_beliefs[0])
    
    variances = []
    ranges = []
    
    for t in range(n_timepoints):
        beliefs_at_t = [agent_beliefs[agent][t] for agent in range(n_agents)]
        variances.append(np.var(beliefs_at_t))
        ranges.append(max(beliefs_at_t) - min(beliefs_at_t))
    
    return {
        'final_variance': variances[-1] if variances else 0,
        'final_range': ranges[-1] if ranges else 0,
        'mean_variance': np.mean(variances) if variances else 0,
        'variance_reduction': (variances[0] - variances[-1]) if len(variances) > 1 else 0
    }

def allocation_efficiency(allocations: Dict[str, int], tool_reliabilities: Dict[str, float]) -> float:
    """Calculate efficiency of query allocation across tools
    
    Args:
        allocations: Dictionary mapping tool names to number of queries allocated
        tool_reliabilities: Dictionary mapping tool names to reliability scores (0-1)
        
    Returns:
        Allocation efficiency score (higher is better)
    """
    if not allocations or not tool_reliabilities:
        return 0.0
    
    total_queries = sum(allocations.values())
    if total_queries == 0:
        return 0.0
    
    # Calculate weighted reliability based on allocation
    weighted_reliability = 0
    for tool_name, n_queries in allocations.items():
        if tool_name in tool_reliabilities:
            weight = n_queries / total_queries
            weighted_reliability += weight * tool_reliabilities[tool_name]
    
    return weighted_reliability

def is_fair_coin(bias: float, threshold: float = FAIRNESS_THRESHOLD) -> bool:
    """Determine if a coin is considered fair based on bias threshold
    
    Args:
        bias: Coin bias (0.5 = perfectly fair)
        threshold: Maximum deviation from 0.5 to be considered fair
        
    Returns:
        True if coin is considered fair, False otherwise
    """
    return abs(bias - 0.5) < threshold

def compute_experiment_metrics(trial_results: List[Dict[str, Any]], 
                             experiment_type: str = None) -> Dict[str, Any]:
    """Compute comprehensive metrics for an experiment by aggregating pre-computed trial metrics
    
    Args:
        trial_results: List of trial result dictionaries with pre-computed metrics
        experiment_type: Type of experiment (A, B, or C) - kept for backward compatibility
        
    Returns:
        Dictionary of computed metrics
    """
    if not trial_results:
        return {}
    
    # Extract pre-computed metrics from trials
    brier_scores = []
    log_losses = []
    accuracies = []
    predictions = []
    failed_trials = 0
    
    for result in trial_results:
        metrics = result.get('metrics', {})
        
        if 'failed_parse' in metrics or 'failed' in metrics:
            failed_trials += 1
            continue
        
        # Use pre-computed trial metrics
        if 'brier_score' in metrics:
            brier_scores.append(metrics['brier_score'])
        if 'log_loss' in metrics:
            log_losses.append(metrics['log_loss'])
        if 'accuracy' in metrics:
            accuracies.append(metrics['accuracy'])
        
        # Extract predictions for calibration analysis
        pred_prob = result.get('model_response', {}).get('probability')
        if pred_prob is not None:
            predictions.append(pred_prob)
    
    if not brier_scores and not accuracies:
        return {'error': 'No valid metrics found in trial results', 'failed_trials': failed_trials}
    
    # Aggregate pre-computed metrics
    metrics = {
        'n_trials': len(trial_results),
        'n_valid_trials': len(brier_scores),
        'failed_trials': failed_trials,
        'success_rate': len(brier_scores) / len(trial_results) if trial_results else 0,
    }
    
    # Primary metrics from pre-computed values
    if brier_scores:
        metrics['mean_brier_score'] = np.mean(brier_scores)
        metrics['brier_score_ci'] = confidence_interval(brier_scores)
    
    if log_losses:
        metrics['mean_log_loss'] = np.mean(log_losses)
    
    if accuracies:
        metrics['accuracy'] = np.mean(accuracies)
        # Proper binomial confidence interval for accuracy
        n_correct = sum(accuracies)
        metrics['accuracy_ci'] = binomial_confidence_interval(n_correct, len(accuracies))
    
    # Calibration analysis using original ground truth from trials
    if len(predictions) >= 10:
        # Extract ground truth outcomes from pre-computed trial results
        outcomes = []
        for result in trial_results:
            if result.get('metrics', {}).get('accuracy') is not None:
                # Use the ground truth that was used for trial-level accuracy calculation
                pred_prob = result.get('model_response', {}).get('probability')
                trial_accuracy = result.get('metrics', {}).get('accuracy', 0)
                
                # Reconstruct outcome from trial accuracy and prediction
                if pred_prob is not None:
                    predicted_fair = pred_prob >= 0.5
                    actual_fair = (trial_accuracy == 1.0) == predicted_fair
                    outcomes.append(1 if actual_fair else 0)
        
        if len(outcomes) == len(predictions):
            ece, calibration_details = calibration_error(predictions, outcomes)
            metrics['expected_calibration_error'] = ece
            metrics['calibration_details'] = calibration_details
    
    return metrics
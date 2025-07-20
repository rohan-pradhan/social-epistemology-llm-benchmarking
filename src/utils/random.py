import numpy as np
import random
from typing import Optional, List
import hashlib

def set_seed(seed: int):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    random.seed(seed)

def get_random_state() -> dict:
    """Get current random state for saving/restoring"""
    return {
        'numpy_state': np.random.get_state(),
        'python_state': random.getstate()
    }

def set_random_state(state: dict):
    """Restore random state"""
    if 'numpy_state' in state:
        np.random.set_state(state['numpy_state'])
    if 'python_state' in state:
        random.setstate(state['python_state'])

def generate_master_seed_sequence(master_seed: int, total_trials: int) -> List[int]:
    """Generate a sequence of unique child seeds from a master seed.
    
    Args:
        master_seed: The master seed to derive child seeds from
        total_trials: Total number of trials (and thus child seeds) needed
        
    Returns:
        List of unique child seeds, one for each trial
    """
    # Use a deterministic approach to generate child seeds
    # This ensures reproducibility while providing independence between trials
    child_seeds = []
    
    for trial_idx in range(total_trials):
        # Combine master seed with trial index to create unique seed
        seed_string = f"{master_seed}_{trial_idx}"
        # Use SHA-256 to create a high-quality hash
        hash_digest = hashlib.sha256(seed_string.encode()).hexdigest()
        # Convert first 8 hex chars to integer (32-bit seed)
        child_seed = int(hash_digest[:8], 16)
        child_seeds.append(child_seed)
    
    return child_seeds

def derive_trial_seed(master_seed: int, trial_index: int) -> int:
    """Derive a unique seed for a specific trial from the master seed.
    
    Args:
        master_seed: The master seed
        trial_index: Zero-based index of the trial
        
    Returns:
        Unique seed for this trial
    """
    seed_string = f"{master_seed}_{trial_index}"
    hash_digest = hashlib.sha256(seed_string.encode()).hexdigest()
    return int(hash_digest[:8], 16)

def initialize_trial_randomness(trial_seed: int):
    """Initialize both NumPy and Python random generators with the trial seed.
    
    Args:
        trial_seed: The seed to use for this trial
    """
    np.random.seed(trial_seed)
    random.seed(trial_seed)
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import json
import time
import logging
from datetime import datetime
import numpy as np

from ..models.base import BaseModel
from ..utils.random import set_seed, derive_trial_seed, initialize_trial_randomness

logger = logging.getLogger(__name__)

@dataclass
class TrialResult:
    """Result from a single experimental trial"""
    trial_id: str
    experiment_type: str
    model_name: str
    condition: str
    ground_truth: Dict[str, Any]
    model_response: Dict[str, Any]
    metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    execution_time: Optional[float] = None

@dataclass
class ExperimentConfig:
    """Configuration for experiments"""
    num_trials: int = 100
    random_seed: int = 42
    save_results: bool = True
    verbose: bool = True
    output_dir: str = "results"
    
    # Coin parameters
    coin_bias: float = 0.5  # The actual bias of the coin to use
    fairness_threshold: float = 0.01
    
    # Budget constraints  
    total_budget: int = 20
    
    # Noise parameters
    display_noise_prob: float = 0.1
    

class BaseExperiment(ABC):
    """Abstract base class for all experiments"""
    
    def __init__(self, config: ExperimentConfig = None):
        self.config = config or ExperimentConfig()
        self.results: List[TrialResult] = []
        self.experiment_id = f"{self.__class__.__name__}_{int(time.time())}"
        
        # Set random seed for reproducibility
        set_seed(self.config.random_seed)
        
        # Track trial counter for seed derivation
        self._global_trial_counter = 0
        
    @abstractmethod
    def run_single_trial(self, 
                        model: BaseModel, 
                        trial_id: str, 
                        condition: str = "default") -> TrialResult:
        """Run a single experimental trial
        
        Args:
            model: The language model to test
            trial_id: Unique identifier for this trial
            condition: Experimental condition (if applicable)
            
        Returns:
            TrialResult containing all trial data
        """
        pass
    
    @abstractmethod
    def get_ground_truth(self, trial_params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate ground truth for scoring
        
        Args:
            trial_params: Parameters for this trial
            
        Returns:
            Dictionary containing ground truth values
        """
        pass
    
    @abstractmethod
    def parse_model_response(self, response_text: str) -> Dict[str, Any]:
        """Parse model response into structured format
        
        Args:
            response_text: Raw text response from model
            
        Returns:
            Dictionary containing parsed response data
        """
        pass
    
    def run_experiment(self, 
                      models: List[BaseModel], 
                      conditions: List[str] = None) -> List[TrialResult]:
        """Run the full experiment across models and conditions
        
        Args:
            models: List of models to test
            conditions: List of experimental conditions to test
            
        Returns:
            List of all trial results
        """
        conditions = conditions or ["default"]
        all_results = []
        
        total_trials = len(models) * len(conditions) * self.config.num_trials
        trial_count = 0
        
        logger.info(f"Starting {self.__class__.__name__} with {total_trials} total trials")
        
        for model in models:
            for condition in conditions:
                logger.info(f"Running {self.config.num_trials} trials for {model.model_name} in condition '{condition}'")
                
                for trial_idx in range(self.config.num_trials):
                    trial_id = f"{self.experiment_id}_{model.model_name}_{condition}_{trial_idx:04d}"
                    
                    # Derive unique seed for this trial and initialize randomness
                    trial_seed = derive_trial_seed(self.config.random_seed, self._global_trial_counter)
                    initialize_trial_randomness(trial_seed)
                    
                    try:
                        start_time = time.time()
                        result = self.run_single_trial(model, trial_id, condition)
                        result.execution_time = time.time() - start_time
                        
                        # Record the trial seed in metadata for reproducibility
                        result.metadata['trial_seed'] = trial_seed
                        result.metadata['global_trial_index'] = self._global_trial_counter
                        
                        all_results.append(result)
                        self.results.append(result)
                        
                        trial_count += 1
                        self._global_trial_counter += 1
                        if self.config.verbose and trial_count % 10 == 0:
                            logger.info(f"Completed {trial_count}/{total_trials} trials")
                            
                    except Exception as e:
                        logger.error(f"Trial {trial_id} failed: {str(e)}")
                        # Create failed trial result
                        failed_result = TrialResult(
                            trial_id=trial_id,
                            experiment_type=self.__class__.__name__,
                            model_name=model.model_name,
                            condition=condition,
                            ground_truth={},
                            model_response={"error": str(e)},
                            metrics={"failed": 1.0},
                            metadata={
                                "error": str(e),
                                "trial_seed": trial_seed,
                                "global_trial_index": self._global_trial_counter
                            }
                        )
                        self._global_trial_counter += 1
                        all_results.append(failed_result)
        
        logger.info(f"Experiment completed. {len(all_results)} total trials.")
        
        if self.config.save_results:
            self.save_results(all_results)
        
        return all_results
    
    def save_results(self, results: List[TrialResult]):
        """Save results to JSON file"""
        import os
        
        os.makedirs(self.config.output_dir, exist_ok=True)
        filename = f"{self.config.output_dir}/{self.experiment_id}_results.json"
        
        # Convert results to serializable format
        serializable_results = []
        for result in results:
            result_dict = {
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
            serializable_results.append(result_dict)
        
        with open(filename, 'w') as f:
            json.dump({
                'experiment_config': self.config.__dict__,
                'results': serializable_results
            }, f, indent=2)
        
        logger.info(f"Results saved to {filename}")
    
    def flip_coin(self, bias: float, n_flips: int) -> List[int]:
        """Generate coin flips with given bias
        
        Args:
            bias: Probability of heads (1)
            n_flips: Number of flips
            
        Returns:
            List of coin flips (0 for tails, 1 for heads)
        """
        return np.random.binomial(1, bias, n_flips).tolist()
    
    def apply_display_noise(self, flips: List[int]) -> List[int]:
        """Apply display noise to coin flips"""
        if self.config.display_noise_prob <= 0:
            return flips
        
        noisy_flips = []
        for flip in flips:
            if np.random.random() < self.config.display_noise_prob:
                noisy_flips.append(1 - flip)  # Flip the bit
            else:
                noisy_flips.append(flip)
        
        return noisy_flips
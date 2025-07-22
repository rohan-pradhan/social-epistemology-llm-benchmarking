# Social Epistemology Benchmark

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A benchmark for evaluating meta-epistemic reasoning capabilities in large language models. This benchmark tests how AI systems gather information from multiple sources, evaluate source reliability, and make calibrated probability judgments under uncertainty.

## Overview

This benchmark focuses on **Meta-Epistemic Delegation** - evaluating how well models can allocate limited information-gathering resources across sources with varying reliability to make accurate probabilistic judgments.

The benchmark simulates real-world scenarios where agents must determine whether a coin is fair or biased by querying different information sources, each with different reliability characteristics and observation noise.

## The Experiment: Meta-Epistemic Delegation

Tests an LLM's ability to efficiently allocate a limited budget of queries across multiple information sources to determine if a coin is fair or biased. The model must:
- Query different tools that provide noisy observations of coin flips
- Identify which sources are more reliable through profile queries
- Integrate information from multiple sources to make calibrated probability estimates
- Work within budget constraints to maximize information gain

**Key Features:**
- Configurable coin bias (0.25, 0.5, 0.6)
- Query budget limits (5, 10 queries)
- Optional persona mode with named sources
- Profile query capability to learn about source reliability
- Think tool for explicit reasoning
- Display noise simulation (15% default)

**Key Metrics:**
- **Brier Score**: Measures probability calibration (lower is better)
- **Log Loss**: Alternative accuracy measure for probabilistic predictions
- **Allocation Efficiency**: How well budget is allocated to reliable sources
- **Budget Utilization**: Percentage of available budget used
- **Tool Diversity**: Distribution of queries across different sources

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd social-epistemology-benchmark

# Install dependencies
pip install -e .

# Or install from requirements
pip install -r requirements.txt
```

## Setup

Set your API keys:
```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key" 
```

## Quick Start

```python
from src.experiments.experiment_a import ExperimentA, ExperimentAConfig
from src.models.openai_model import GPT4oMiniModel

# Configure experiment
config = ExperimentAConfig(num_trials=10, total_budget=20)
experiment = ExperimentA(config)

# Setup model
model = GPT4oMiniModel()

# Run experiment
results = experiment.run_experiment([model], ["default"])

# Analyze results
from src.evaluation.metrics import compute_experiment_metrics
metrics = compute_experiment_metrics([r.__dict__ for r in results], "A")
print(metrics)
```

## Running Parameter Sweeps

The `run_experiments/` directory contains scripts for comprehensive parameter sweeps:

### Anthropic Models
```bash
python run_experiments/parameter_sweep_anthropic.py
```

### OpenAI Models
```bash
python run_experiments/parameter_sweep_openai.py
```

### Parameters Tested
- **persona_mode**: [False, True] - Use named sources vs generic tools
- **allow_profile_queries**: [False, True] - Allow querying source reliability
- **total_budget**: [5, 10] - Number of allowed queries
- **coin_bias**: [0.25, 0.5, 0.6] - Actual coin bias
- **allow_think_tool**: [False, True, "extended_reasoning"] - Thinking capabilities

See `run_experiments/README.md` for detailed documentation on running experiments.

## Supported Models

The benchmark supports various models from OpenAI and Anthropic. While all models listed below can be used, the default models used in the parameter sweeps are:

**Default Models in Parameter Sweeps:**
- **OpenAI**: `GPT41Model` (GPT-4.1) and `O4MiniModel` (O4-Mini) for extended reasoning experiments
- **Anthropic**: `ClaudeSonnet4Model` (Claude Sonnet 4)

### OpenAI
- `GPT4oModel` - GPT-4o
- `GPT4oMiniModel` - GPT-4o-mini  
- `GPT4oMiniHighModel` - GPT-4o-mini with enhanced reasoning
- `GPT41Model` - GPT-4.1
- `O4MiniModel` - O4-Mini (used for extended reasoning in sweeps)

### Anthropic
- `ClaudeSonnet4Model` - Claude Sonnet 4 (claude-sonnet-4-20250514)
- `ClaudeOpus4Model` - Claude Opus 4 (claude-opus-4-20250514)  
- `Claude35HaikuModel` - Claude 3.5 Haiku (claude-3-5-haiku-20241022)

## Example Usage

See `example_usage.py` for complete examples of running each experiment.

```bash
python example_usage.py
```

## Project Structure

```
social-epistemology-benchmark/
├── src/                          # Core source code
│   ├── experiments/              # Experiment implementations
│   │   ├── base.py              # Base experiment class with common functionality
│   │   ├── experiment_a.py      # Meta-epistemic delegation experiment
│   │   ├── tools.py             # Information source tools (coin flip simulators)
│   │   └── tool_schemas.py      # Tool definitions for model interaction
│   ├── models/                  # Model provider implementations
│   │   ├── base.py              # Abstract base model interface
│   │   ├── openai_model.py      # OpenAI model implementations (GPT-4, etc.)
│   │   └── anthropic_model.py   # Anthropic model implementations (Claude)
│   ├── evaluation/              # Metrics and evaluation utilities
│   │   └── metrics.py           # Comprehensive metrics (Brier score, calibration, etc.)
│   └── utils/                   # Utility functions
│       └── random.py            # Random number generation utilities
│
├── run_experiments/             # Parameter sweep and batch experiment scripts
│   ├── README.md                # Detailed documentation for running experiments
│   ├── parameter_sweep_anthropic.py # Anthropic models parameter sweep
│   ├── parameter_sweep_openai.py    # OpenAI models parameter sweep
│   └── incentivized_worst_cases.py  # Worst-case scenario testing
│
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup configuration
└── README.md                    # This file
```

## Metrics

The benchmark uses several key metrics:

- **Brier Score**: Measures probability calibration (lower is better)
- **Log Loss**: Measures prediction confidence (lower is better)  
- **Accuracy**: Binary classification accuracy
- **Calibration Error**: Expected calibration error across predictions
- **Allocation Efficiency**: How well budget is allocated across reliable sources
- **Belief Convergence**: How well agents reach consensus (Experiment B)
- **Bias Resistance**: Ability to resist persona biases (Experiment C)

## Results

Results are automatically saved to the `results/` directory as JSON files containing:
- Trial-level data for each model and condition
- Aggregated metrics across trials
- Detailed metadata for analysis

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
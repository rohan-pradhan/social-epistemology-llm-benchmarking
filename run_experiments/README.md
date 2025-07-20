# Running Experiments

This directory contains scripts for running parameter sweeps and specific configurations of the Social Epistemology Benchmark.

## Available Scripts

### 1. parameter_sweep_anthropic.py
Runs a comprehensive parameter sweep using Anthropic's Claude models.

**Parameters swept:**
- `persona_mode`: [False, True]
- `allow_profile_queries`: [False, True]
- `total_budget`: [5, 10] - Number of coin flip observations allowed
- `coin_bias`: [0.25, 0.5, 0.6] - The actual bias of the coin
- `allow_think_tool`: [False, True, "extended_reasoning"] - Three options for thinking capability

**Usage:**
```bash
# Run full sweep
python parameter_sweep_anthropic.py

# Specify custom output directory
python parameter_sweep_anthropic.py --output-dir custom_results/

# Resume from partial results
python parameter_sweep_anthropic.py --resume-detailed <path> --resume-summary <path>
```

### 2. parameter_sweep_openai.py
Runs the same parameter sweep but using OpenAI's models.

**Model selection:**
- When `allow_think_tool="extended_reasoning"`: Uses O4MiniModel
- When `allow_think_tool=True` or `False`: Uses GPT-4 (GPT41Model)

**Usage:**
```bash
# Run with automatic model selection based on allow_think_tool parameter
python parameter_sweep_openai.py

# Specify output directory
python parameter_sweep_openai.py --output-dir openai_results/

# Resume from partial results
python parameter_sweep_openai.py --resume-detailed <path> --resume-summary <path>
```

### 3. incentivized_worst_cases.py
Tests specific "worst performing" configurations with an incentivized condition where the agent is rewarded for accuracy.

**Configurations tested:**
- Specific combinations identified as challenging from previous sweeps
- Includes persona mode and profile query variations
- Tests with different budget and bias settings

**Usage:**
```bash
# Run worst case analysis
python incentivized_worst_cases.py

# Custom output directory
python incentivized_worst_cases.py --output-dir incentivized_results/
```

## Environment Setup

Before running any experiments, ensure you have:

1. **API Keys**: Set environment variables for the model providers you plan to use:
   ```bash
   export ANTHROPIC_API_KEY="your-anthropic-api-key"
   export OPENAI_API_KEY="your-openai-api-key"
   ```

2. **Dependencies**: Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Output Format

All experiments generate:
- **Summary results**: JSON file with aggregated metrics per configuration
- **Detailed results**: JSON file with complete trial data including trajectories

Results are saved with timestamps in the specified output directory (default: `results/`).

## Notes

- Each configuration is tested with 10 trials
- The `--resume` flags allow continuing interrupted sweeps
- Total runtime depends on the number of configurations and API response times
- Monitor API usage as these experiments make many model calls
- The parameter sweep tests 72 different configurations (2×2×2×3×3)
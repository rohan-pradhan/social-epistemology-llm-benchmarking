# setup.py
from setuptools import setup, find_packages

setup(
    name="social-epistemic-benchmark",
    version="0.1.0",
    description="Benchmark suite for evaluating social-epistemic reasoning in large language models",
    packages=find_packages(include=['src', 'src.*', 'run_experiments']),
    install_requires=[
        "openai>=1.0.0",
        "anthropic>=0.3.0",
        "requests>=2.28.0",
        "pyyaml>=6.0",
        "numpy>=1.21.0",
        "scipy>=1.9.0",
        "pandas>=1.5.0",
        "python-dotenv>=0.19.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0.0", "black", "flake8"],
    },
    python_requires=">=3.8",
    entry_points={
        'console_scripts': [
            'seb-incentivized-worst=run_experiments.incentivized_worst_cases:main',
            'seb-sweep-anthropic=run_experiments.parameter_sweep_anthropic:main',
            'seb-sweep-openai=run_experiments.parameter_sweep_openai:main',
        ],
    },
)
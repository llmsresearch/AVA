# Anytime Verified Agents (AVA)

[![Paper](https://img.shields.io/badge/Paper-TMLR%20(Submitted)-blue)](https://www.jmlr.org/tmlr/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black)](https://github.com/llmsresearch/AVA)
[![Status](https://img.shields.io/badge/Status-Research-orange)](https://github.com/llmsresearch/AVA)

> **Anytime Verified Agents: Adaptive Compute Allocation for Reliable LLM Reasoning under Budget Constraints**

**Draft submitted to [Transactions on Machine Learning Research (TMLR)](https://www.jmlr.org/tmlr/)**

## Overview

Anytime Verified Agents (AVA) is a framework for building reliable LLM-based reasoning systems that adaptively allocate computational resources under budget constraints. AVA combines adaptive search strategies, uncertainty estimation, and verification cascades to maximize reliability while respecting token, tool-call, and verification budgets.

### Key Features

- **Adaptive Compute Allocation**: Dynamically allocates tokens, tool calls, and verification steps based on uncertainty and value-of-information
- **Uncertainty Estimation**: Multiple uncertainty estimators (token entropy, consistency, trajectory features) for informed decision-making
- **Verification Cascade**: Multi-level verification with early exits for efficient correctness checking
- **Budget-Aware**: Respects token, tool-call, and verification budgets while maximizing reliability
- **Anytime Algorithm**: Returns best solution found so far when budget is exhausted

## Installation

```bash
# Clone the repository
git clone https://github.com/llmsresearch/AVA.git
cd AVA

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- Python 3.8+
- numpy
- scikit-learn
- typer (for CLI)
- azure-openai (for model provider)

## Quick Start

```python
from ava.agents.ava_agent import AVAAgent
from ava.core.interfaces import Budget
from ava.models.azure_openai import AzureOpenAIModel

# Initialize model and agent
model = AzureOpenAIModel()
agent = AVAAgent(model, target_reliability=0.9)

# Define budget
budget = Budget(token_limit=1000, tool_calls_limit=5, verify_calls_limit=10)

# Solve a problem
prompt = "Question: What is 2+2?\nAnswer:"
result = agent.solve(prompt, budget)

print(f"Solution: {result.text}")
print(f"Reliability: {result.metadata.get('reliability', 'N/A')}")
print(f"Tokens used: {budget.tokens_used}/{budget.token_limit}")
```

## Project Structure

```
ava/
├── agents/              # AVA agent implementation
│   └── ava_agent.py
├── baselines/           # Baseline algorithms for comparison
│   ├── fixed_depth_search.py
│   └── self_consistency.py
├── benchmarks/          # Benchmark dataset loaders
│   ├── gsm8k.py
│   ├── hotpotqa.py
│   └── humaneval.py
├── controllers/         # Adaptive controller for compute allocation
│   └── ava_controller.py
├── core/                # Core interfaces and protocols
│   └── interfaces.py
├── models/              # Model provider implementations
│   ├── azure_openai.py
│   └── rate_limiter.py
├── search/              # Adaptive search strategies
│   └── adaptive.py
├── uncertainty/         # Uncertainty estimation and calibration
│   ├── calibration.py
│   └── estimators.py
├── utils/               # Utility functions
│   ├── logging.py
│   └── metrics.py
└── verification/        # Verification cascade implementation
    ├── base.py
    └── cascade.py
```

## Core Components

### AVA Agent

The main agent class that integrates all components:

- **Adaptive Controller**: Decides how to allocate compute based on current state
- **Adaptive Search**: Tree search with value-of-information heuristics
- **Verification Cascade**: Multi-level verification with early exits
- **Uncertainty Estimation**: Combines multiple uncertainty signals

### Baselines

- **Self-Consistency**: Majority voting over multiple samples
- **Fixed-Depth Search**: Exhaustive search up to fixed depth

### Benchmarks

Support for multiple reasoning benchmarks:

- **GSM8K**: Math word problems
- **HotpotQA**: Multi-hop question answering
- **HumanEval**: Code generation

## Usage Examples

### Basic Usage

```python
from ava.agents.ava_agent import AVAAgent
from ava.core.interfaces import Budget
from ava.models.azure_openai import AzureOpenAIModel

model = AzureOpenAIModel()
agent = AVAAgent(model, target_reliability=0.9)
budget = Budget(token_limit=2000)

result = agent.solve("Your prompt here", budget)
```

### Custom Configuration

```python
from ava.controllers.ava_controller import AVAController
from ava.verification.cascade import create_default_cascade

# Custom controller
controller = AVAController(target_reliability=0.95)

# Custom verification cascade
verifier_cascade = create_default_cascade()

# Create agent with custom components
agent = AVAAgent(
    model=model,
    controller=controller,
    verifier_cascade=verifier_cascade,
    target_reliability=0.95
)
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{ava2025,
  title={Anytime Verified Agents: Adaptive Compute Allocation for Reliable LLM Reasoning under Budget Constraints},
  author={[Dipkumar Patel]},
  journal={Transactions on Machine Learning Research},
  year={2025},
  note={Submitted}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or issues, please open an issue on [GitHub](https://github.com/llmsresearch/AVA/issues) or contact the authors.

---

**Note**: This is research code accompanying a paper submitted to TMLR. The codebase is actively maintained and may be updated based on reviewer feedback.
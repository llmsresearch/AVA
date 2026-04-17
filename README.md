# Anytime Verified Agents (AVA)

[![Paper](https://img.shields.io/badge/Paper-TMLR%202026-blue)](https://openreview.net/forum?id=JMDCMf7mlF)
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![SymPy](https://img.shields.io/badge/SymPy-3B5526?logo=sympy&logoColor=white)](https://www.sympy.org/)
[![OpenAI](https://img.shields.io/badge/Azure%20OpenAI-0078D4?logo=microsoft-azure&logoColor=white)](https://azure.microsoft.com/en-us/products/ai-services/openai-service)
[![Ollama](https://img.shields.io/badge/Ollama-000000?logo=ollama&logoColor=white)](https://ollama.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Anytime Verified Agents: Adaptive Compute Allocation for Reliable LLM Reasoning under Budget Constraints**

Published in [Transactions on Machine Learning Research (TMLR)](https://www.jmlr.org/tmlr/), 2026.

## Overview

AVA is a framework that dynamically allocates compute across search, sampling, and verification within a user-specified budget. It combines calibrated uncertainty estimation, value-of-information-guided search expansion, and selective verification cascades with early exits. The controller allocates compute based on uncertainty and estimated marginal reliability gains.

Evaluated on GSM8K, MATH, HotpotQA, and HumanEval with GPT-5 and GPT-4o.

## Installation

```bash
git clone https://github.com/llmsresearch/AVA.git
cd AVA
pip install -r requirements.txt
```

### Model backends

AVA supports two model backends:

**Ollama (local inference):** Install [Ollama](https://ollama.com/) and pull a model:
```bash
ollama pull qwen2.5:7b
```

**Azure OpenAI:** Create a `.env` file:
```
AZURE_MODEL_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_MODEL_KEY=your-key
AZURE_MODEL_DEPLOYMENT=gpt-4o
AZURE_MODEL_API_VERSION=2024-12-01-preview
```

## Quick start

```python
from ava.agents.ava_agent import AVAAgent
from ava.core.interfaces import Budget
from ava.models.ollama_model import OllamaModel

model = OllamaModel(model_name="qwen2.5:7b")
agent = AVAAgent(model, target_reliability=0.9)

budget = Budget(token_limit=800, tool_calls_limit=20, verify_calls_limit=20)
result = agent.solve("Question: What is 17 * 23?\nAnswer:", budget)

print(result.text)
print(f"Tokens used: {budget.tokens_used}/{budget.token_limit}")
```

See `examples/quick_start.py` for a full working example comparing AVA against the self-consistency baseline.

## Project structure

```
ava/
├── agents/ava_agent.py          # AVA agent (bootstrap, adaptive sampling, verify-and-vote)
├── controllers/ava_controller.py # Budget-aware controller (Algorithm 1 in paper)
├── core/interfaces.py           # Protocols: ModelProvider, Budget, SearchStrategy
├── search/adaptive.py           # VoI-guided adaptive tree search
├── uncertainty/
│   ├── estimators.py            # Token entropy, consistency, trajectory estimators
│   └── calibration.py           # Isotonic regression calibrator
├── verification/
│   ├── base.py                  # Verifier protocol, heuristic verifier
│   ├── cascade.py               # Multi-level verification cascade with early exits
│   └── llm_verifier.py          # Re-solve-and-vote verifier
├── baselines/
│   ├── self_consistency.py      # Majority voting over k samples
│   ├── fixed_depth_search.py    # Tree-of-Thoughts style BFS
│   ├── confidence_early_exit.py # Confidence-threshold early stopping
│   └── difficulty_bin.py        # Fixed allocation by difficulty tier
├── benchmarks/
│   ├── gsm8k.py                 # GSM8K loader
│   ├── hotpotqa.py              # HotpotQA loader
│   ├── humaneval.py             # HumanEval loader
│   ├── math.py                  # MATH dataset loader
│   └── math_grading.py          # SymPy-based MATH answer equivalence checker
├── models/
│   ├── azure_openai.py          # Azure OpenAI provider
│   ├── ollama_model.py          # Local Ollama provider
│   └── rate_limiter.py          # API rate limiter
└── utils/
    ├── metrics.py               # Reliability metrics, ECE, Brier score
    └── logging.py               # Result table formatting

experiments/
├── run_full_evaluation.py       # Main evaluation runner
├── sensitivity_analysis.py      # Controller threshold sensitivity
└── calibration_transfer.py      # Cross-dataset calibration transfer
```

## Running experiments

```bash
# Evaluate AVA and baselines on GSM8K with Ollama
python experiments/run_full_evaluation.py \
  --model ollama --model-name qwen2.5:7b \
  --benchmarks gsm8k \
  --methods ava,self_consistency,fixed_depth,always_verify \
  --budgets 400,600,800,1000

# Evaluate on multiple benchmarks with Azure OpenAI
python experiments/run_full_evaluation.py \
  --model azure \
  --benchmarks gsm8k,hotpotqa,humaneval \
  --methods ava,self_consistency \
  --budgets 400,600,800,1000 \
  --output results/my_run
```

Benchmark datasets should be placed in `data/`:
```
data/
├── gsm8k/gsm8k_test.jsonl
├── hotpotqa/hotpotqa_dev.json
├── humaneval/HumanEval.jsonl
└── math/test.json
```

## Citation

```bibtex
@article{patel2026ava,
  title={Anytime Verified Agents: Adaptive Compute Allocation for Reliable LLM Reasoning under Budget Constraints},
  author={Patel, Dipkumar},
  journal={Transactions on Machine Learning Research},
  year={2026},
  url={https://openreview.net/forum?id=JMDCMf7mlF}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

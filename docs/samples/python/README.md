# Python documentation samples

These files are the canonical sources included in the documentation.

| Sample | Documentation | Purpose | Verification |
| --- | --- | --- | --- |
| `matmul.py` | Getting Started / Quick Start | Build, plan, execute, and numerically check BF16 matmul | End-to-end on Blackwell |
| `custom_engine.py` | How-to Guides / Engine Selection | Inspect, select, and execute an automatically discovered plan | End-to-end on Blackwell |

Run from the repository root after activating the environment that provides the
frontend and its dependencies:

```bash
python docs/samples/python/matmul.py
python docs/samples/python/custom_engine.py
```

Requirements: Python 3.10+, CUDA-enabled PyTorch, a built
`cudnn._compiled_module`, cuDNN, and a supported NVIDIA GPU. Success prints the
result shape and workspace size; failure raises during graph construction,
plan build, execution, or numerical comparison.

See the generated [sample index](/verified-samples/catalog) for the complete inventory and the
runner command for each sample.

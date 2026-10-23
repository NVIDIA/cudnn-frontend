#!/usr/bin/env bash
set -euo pipefail

# Released core package.
python -m pip install nvidia-cudnn-frontend

# Optional CUTLASS DSL kernels and their pinned dependencies.
python -m pip install 'nvidia-cudnn-frontend[cutedsl]'

# Source checkout for development and documentation verification.
python -m pip install -v .

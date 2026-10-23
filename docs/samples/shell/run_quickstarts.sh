#!/usr/bin/env bash
set -euo pipefail

# --8<-- [start:python]
python docs/samples/python/matmul.py
# --8<-- [end:python]

# --8<-- [start:cpp]
nvcc -std=c++17 -Iinclude docs/samples/cpp/matmul.cc -lcudnn -lnvrtc -o /tmp/cudnn_fe_matmul
/tmp/cudnn_fe_matmul
# --8<-- [end:cpp]

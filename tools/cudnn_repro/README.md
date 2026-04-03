# cudnn-repro

A command-line tool to generate pytest repro commands from cuDNN Frontend SDPA logs.

## What it does

When a cuDNN SDPA test runs with logging enabled, it captures the exact graph configuration as JSON. This tool:
1. Extracts that JSON from log files
2. Translates it into a `test_repro()` function call
3. Outputs a pytest command you can run to reproduce the exact same test

This is useful for debugging failures, reproducing CI issues locally, or creating minimal test cases.

## Installation

```bash
# Install globally (recommended)
uv tool install --editable tools/cudnn_repro

# Or install in venv
pip install -e tools/cudnn_repro
```

## Quick Start

```bash
# 1. Run a test with logging
export CUDNN_FRONTEND_LOG_INFO=1
export CUDNN_FRONTEND_LOG_FILE=/tmp/sdpa.log
pytest test/python/test_mhas_v2.py::test_sdpa_random_fwd_L0[test1]

# 2. Generate repro command
cudnn-repro /tmp/sdpa.log

# 3. Run the repro
# (copy-paste the output command)
```

## Usage

```bash
# Process a log file
cudnn-repro /path/to/log

# Read from stdin
cat log | cudnn-repro -

# Process all entries (not just last)
cudnn-repro --all log

# Debug mode - saves intermediate stages
CUDNN_DEBUG_REPRO=1 cudnn-repro log
```

## How it works

**3-stage pipeline:**

1. **Stage 0**: Extract JSON context entries from log lines
2. **Stage 1**: Annotate with test config (shape, stride, dtype, etc.)
3. **Stage 2**: Build pytest command

The tool auto-detects SDPA operation tags and routes to the appropriate handler:
- `SDPA_FWD`
- `SDPA_BWD`
- `SDPA_FP8_FWD`
- `SDPA_FP8_BWD`

Non-MXFP8 FP8 forward and backward repro are supported. MXFP8 repro is not yet implemented.

**Debug mode** (`CUDNN_DEBUG_REPRO=1`) writes:
- `cudnn_repro_stage0.txt` - Raw log
- `cudnn_repro_stage1.json` - Extracted config
- `cudnn_repro_stage2.txt` - Final command

## Testing

```bash
pytest tools/cudnn_repro/tests/ -vv

# Control test targets
CUDNN_REPRO_TARGETS="test1,test2" pytest tools/cudnn_repro/tests/
```

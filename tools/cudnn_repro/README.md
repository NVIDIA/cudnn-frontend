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

# Debug mode - saves parsed payload and command output
CUDNN_DEBUG_REPRO=1 cudnn-repro log
```

## How it works

Pipeline:

1. Parse log context JSON
2. Select operation handler
3. Extract repro config
4. Render pytest command

The tool auto-detects SDPA operation tags and routes to the appropriate handler:
- `SDPA_FWD`
- `SDPA_BWD`
- `SDPA_FP8_FWD`
- `SDPA_FP8_BWD`
- `SDPA_MXFP8_FWD`
- `SDPA_MXFP8_BWD`

**Debug mode** (`CUDNN_DEBUG_REPRO=1`) writes:
- `cudnn_repro_log.txt` - Raw log
- `cudnn_repro_payload.json` - Annotated payload
- `cudnn_repro_command.txt` - Final command

## Testing

```bash
pytest tools/cudnn_repro/tests/ -vv

# Control test targets
CUDNN_REPRO_TARGETS="test1,test2" pytest tools/cudnn_repro/tests/
```

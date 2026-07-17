# samples — Agent Guide

Usage examples for the C++ graph API and the Python API. Build commands: [../AGENTS.md](../AGENTS.md).

## Layout

| Path | What |
|---|---|
| `cpp/` | Current C++ graph-API samples, grouped by topic (`convolution/`, `matmul/`, `sdpa/`, `norm/`, `moe_grouped_matmul/`, ...) — Catch2 cases in binary `build/bin/samples` |
| `legacy_samples/` | Frozen samples for the legacy flat API — binary `build/bin/legacy_samples`. Don't add here. |
| `python/` | Numbered Jupyter notebooks (`00_introduction.ipynb` ...) — the Python tutorial sequence |
| `llama/`, `llm_coverage/` | End-to-end LLaMA tie-out scripts and per-op coverage scripts (plain `.py`, see their READMEs) |

## Running

```bash
./build/bin/samples --list-tests
./build/bin/samples "Cached sdpa"        # run one Catch2 case by name
```

Samples need a GPU + cuDNN backend at runtime; individual cases `SKIP()` on unsupported arch or backend version.

## Conventions for new samples

- New C++ samples go under `samples/cpp/<topic>/` as a Catch2 `TEST_CASE` added to `samples/cpp/CMakeLists.txt`; reuse helpers from `samples/cpp/utils/`.
- Follow the existing pattern: build graph → `validate()` → `build_operation_graph()` → plans → `check_support()` (skip gracefully if unsupported) → execute → verify.
- Python notebooks are formatted by `black-jupyter` (`pre-commit run`); keep the numbered-prefix naming so the tutorial order stays obvious.
- A new public feature should come with a sample here and a doc page under `docs/`.

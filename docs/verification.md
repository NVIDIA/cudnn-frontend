# Verification record

Source target: [`0a07d4ee28f8ae2f2106bb766b1878cc732ee80c`](https://github.com/NVIDIA/cudnn-frontend/tree/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c) on `develop`.

Verification date: 2026-10-22.

| Component | Verified environment |
| --- | --- |
| cuDNN Frontend | 1.27.0 from this checkout's `python/` directory |
| Python | 3.13.2 in the `pt` pyenv environment |
| PyTorch | 2.10.0+cu130 |
| CUDA reported by PyTorch | 13.0 |
| CUDA compiler | 13.1.115 |
| cuDNN backend | 9.15.1 |
| GPU | Blackwell, compute capability 10.0 |
| CMake | 3.31.4 |
| GCC | 11.4.0 |

All recorded sample runs used this checkout's Python package and set
`CUDNN_FRONTEND_CUDART_LIB_NAME=libcudart.so.13` through the sample runner.

## Results

| Verification | Result |
| --- | --- |
| Documentation tooling tests | 47 passed |
| Fern configuration | `fern check --warnings` passed with Fern CLI 5.56.3 |
| Fern content bundle and links | Full preview bundle loaded; no broken links remained after route normalization |
| Canonical sample inclusions | 256 `Code` components and 42 `Markdown` component uses resolve; 18 generated regions are current |
| Published Python samples | 111 executed, 8 statically validated through declared requirement fallbacks, 0 skipped, 0 failed |
| Published C++ samples | 117 compiled and executed, 8 statically compiled through declared requirement fallbacks, 0 skipped, 0 failed |
| Shell samples | 1 executed, 1 syntax checked, 0 failed |
| CMake consumer | 1 configured and compiled, 0 failed |
| API ledger | 277 reference pages: 132 Python and 145 C++ |
| Coverage release gate | Passed: all 424 family-owned candidate APIs have source-backed index rows and generated API-local availability contracts |
| Historical-topic ledger | 340 Markdown topics have dispositions |
| Historical-topic reconciliation | 268 retained, 70 merged, 2 excluded, 0 review gaps |
| Candidate APIs with family learning ownership | 424; all link to a learning owner and state kind, purpose, stability, availability, and source evidence |

The complete 247-sample catalog ran in one six-worker invocation, with each C++
source using an isolated temporary build directory. Three integration failures
found by that pass were corrected and rerun. The final aggregate contains 229
executed samples and 18 declared static fallbacks, including the configured and
compiled CMake consumer. The compile-time-constant, MoE backward, and
standalone shape-operation samples require cuDNN 9.22 or later; standalone
RoPE requires cuDNN 9.24 or later. Unified SDPA block masks on SM10x require
cuDNN 9.26 or later. The reconciled OSS RMSNorm+SiLU sample matched a PyTorch
BF16 reference on SM100.

The current compiled extension does not contain the optional Causal Conv1d
bindings, and installed cuDNN 9.15.1 is below the Python MoE forward, MoE
backward, and standalone RoPE gates. Those samples passed their declared Python
syntax fallbacks.

## Verification limits

- The host's Snap/AppArmor configuration rejected its packaged Node runtime.
  Fern CLI 5.56.3 was instead run with the SHA256-verified official Node
  22.23.4 Linux binary.
- The local Fern check was unauthenticated. It skipped remote redirect
  validation, and the preview could not apply the private NVIDIA global theme
  without `FERN_TOKEN`; configuration, MDX, components, and content links were
  still validated.
- Candidate API contracts inherit their compatibility envelope from the
  authoritative family ledger. Symbol-specific shapes, types, layouts,
  ownership, and lifecycle rules remain on each linked learning owner rather
  than on an individual reference page.
- `Execution_plan_list::get_autotune_workspace()` dereferenced null plan slots
  after a partial all-plan build. The plan-inspection sample uses checked
  per-plan workspace queries instead.
- `error_object::get_message()` is not `const`, so a message cannot be read
  through a `const error_t`.
- Native CUDA Graph population returned
  `CUDNN_STATUS_NOT_SUPPORTED_CUDA_GRAPH_NATIVE_API` in the recorded
  environment.
- Execution coverage was limited to Linux and SM100. Windows, other GPU
  architectures, and older CUDA toolkits were not available.
- The environment contained CUTLASS DSL 4.5.2 while the project extra pins
  4.5.0, so the pinned optional-engine dependency combination was not verified.
- API source links are commit-pinned. Historical APIs absent from the current
  surface link to the class or fallback source that establishes that status.

Reproduction commands, after activating and configuring the intended Python
environment:

```bash
PYTHONPATH=. python -m pytest tools/doc/tests -q
python tools/doc/build_doc_manifests.py --check
python tools/doc/build_doc_indexes.py --check
CUDNN_FRONTEND_CUDART_LIB_NAME=libcudart.so.13 python tools/doc/run_doc_samples.py --require-execution --jobs 6
python tools/doc/check_doc_coverage.py --mode review
python tools/doc/check_doc_coverage.py --mode release
cd docs/fern && fern check --warnings
```

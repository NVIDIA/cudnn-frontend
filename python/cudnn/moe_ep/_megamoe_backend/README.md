# Vendored MegaMoE (CuTe DSL) device backend

Verbatim copy of the MegaMoE training package and the CuTe DSL kernel
sources that back `cudnn.MoeEp` (see `../_megamoe.py`), so the PR is
self-sufficient — no external checkout or `CUDNN_MEGAMOE_ROOT` needed.

Contents (import layout is preserved; `megamoe/repo_path.py` puts this
directory, `cutedsl_megamoe/`, and `cutedsl_megamoe/moe_nvfp4_swapab/` on
`sys.path`):

| Folder | What it is |
|---|---|
| `megamoe/` | Training-side package: MXFP8 forward wrapper (`forward.py`), NVFP4 forward wrapper (`forward_nvfp4.py`), hybrid training layer (`training.py`), FP8 mega backward (`bwd_kernel/`, `fp8_bwd.py`), pools/quant utilities. |
| `pt/` | Pure-PyTorch MoE+EP config/reference package (`EpConfig`, `QuantConfig`, references). |
| `cutedsl_megamoe/moe_mxfp8_glu/` | SM100 MXFP8 GLU mega-kernel (CuTe DSL): dispatch → grouped FC1/SwiGLU/FC2 → combine in one launch. |
| `cutedsl_megamoe/moe_nvfp4_swapab/` | SM100 NVFP4 (E2M1, swap-AB) mega-kernel — the FP4 forward; also provides shared runner/epilogue infrastructure imported by the MXFP8 kernel. |
| `cutedsl_megamoe/common/` | Host utilities and kernel constants shared by both kernels. |
| `cutedsl_megamoe/src/` | Dispatch/combine comm layer: NVSHMEM bootstrap, token comm, symmetric buffers, PTX helpers. |

Provenance (copied 2026-08-05, excluding only `__pycache__`/`.git`):

- `moe_ep_training` @ `f97c469ba98e9f264ba53d12edd4ed046a305663`
  (source of `megamoe/`, `pt/`)
- `cutedsl_megamoe` @ `8ff233d6040d522dba536a6179a5e67af950c52f`
  (https://gitlab-master.nvidia.com/bangyus/cutedsl_megamoe; only the four
  subpackages above plus `README.md` are vendored — `ci/`, `tester/`,
  `scripts/`, `tests/` are not needed at runtime)

`CUDNN_MEGAMOE_ROOT` still overrides this bundled copy when set (useful for
kernel development against a live checkout).

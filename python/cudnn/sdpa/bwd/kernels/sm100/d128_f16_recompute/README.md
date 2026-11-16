# Small-half recompute backward

This directory owns the exact D64/D128 FP16/BF16 dense MHA implementation.

| File | Role |
|---|---|
| `contract.py` | Public implementation identity and kernel location |
| `host.py` | Fixed recompute adapter over shared validation and binding |
| `kernel.py` | Q-owner and KV-owner recompute loops and local device helpers |
| `config.py` | Immutable tile, barrier and register configuration |
| `prepared_host.py` | One prepared launch and D64/D128 specialization |

Selection lives in `bwd/d128_selection.py`. The legacy standalone adapter
uses that policy; the explicit engine never selects another main body.
The shared `prepared_sm100_d128.py` owns metadata validation and binding.
Other accumulation and deterministic-pair implementations retain their own
code until their migration; they do not import this main body.

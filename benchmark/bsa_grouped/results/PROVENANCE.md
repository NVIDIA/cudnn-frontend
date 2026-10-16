# Measurement provenance

`b20z_131k.json` contains all 27 measured cases (9 topK values × 3 seeds),
all seven timing samples per implementation/case, correctness differences,
CPU preparation times, software versions and measured source hashes.
The source baseline is cuDNN Frontend develop
`6b2080aedf73ec1058f9c1d094a41d09bdc6370c` plus this grouped implementation.
The GPU was NVIDIA B20Z / SM100, 148 SMs, with the default 1000 W power limit
and unlocked clocks. CUDA Graph replay excludes CPU dispatch/allocation.

The recorded hashes describe the code **at measurement time**. Two final
review fixes followed the sweep:

- `api.py`: added the three grouped names to its module-level `__all__`.
  The family, top-level and Torch lazy exports were already present.
- `_grouped.py`: added rejection of token lengths >= `2**31` before CPU
  preprocessing. All measured lengths were 131072.

No GPU kernel, launch sequence, measured-shape behavior, or benchmark timing
code changed. Removing those three `__all__` entries and the two-line token
bound reproduces the measured source SHA256s exactly (verified). All other
source hashes in the JSON match the submitted files. Final tests cover both
fixes; the measured hashes have deliberately not been replaced with later hashes.

CAKE forward used the original CUDA implementation from
[FlashInfer PR #4593](https://github.com/flashinfer-ai/flashinfer/pull/4593),
checkout `c326d2c5c7cc2b96d58129371dc070a9b5af2205`, source
`csrc/cake_vsa/cake_vsa_blk64_persistent_sm_100a.cu`, SHA256
`c842e151ed70ed48674dd825aa68190c4b69824d8b1c37f0de0d0daf939f3a57`.
It produces O and natural-log LSE with full 64-token KV block lengths.

Maximum relative L2 differences from native64 over all measured cases:

| Variant | O | dQ | dK | dV | LSE max absolute |
| --- | ---: | ---: | ---: | ---: | ---: |
| Grouped, default buckets | 0.003155 | 0.001372 | 0.001414 | 0.000433 | 3.58e-5 |
| Grouped, one bucket | 0.003155 | 0.001372 | 0.001414 | 0.000639 | 3.58e-5 |
| CAKE CUDA | 0.003159 | — | — | — | 4.77e-6 |

These are implementation-to-implementation differences; independent FP64
autograd validation is provided by `test_BSA_grouped.py` at smaller shapes.

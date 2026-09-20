# SM100 D512 single-token boundary follow-up

Measured September 19, 2026, on a dedicated production B200 (148 SMs, 1000 W limit,
1965 MHz SM / 3996 MHz HBM), using public cuDNN 9.26.0.51, PyTorch
2.13.0+cu130 and CuTe DSL 4.8.0.dev0. FE source was
`e7a34fd2fdae2e250459922e4c84a6e7e93cc605`; the exact-develop native extension
was held constant. This records the old policy's independent follow-up measurements,
not a timing of the corrected policy or an end-to-end serving result.

The CSV preserves 16 BF16/FP16 settings, including shared and separate K/V.
Native-pinned and FROST-forced graphs use the same quantized inputs. Each setting
passed six route and numerical checks (automatic/native/FROST, Stats on/off):
full valid O/LSE finiteness, all O values compared across routes, and independent
CPU FP32 O/LSE checks at first/middle/last valid Q rows of every batch/head.
The independent reference is sampled, not an all-row mathematical check.

Kernel time is the sum of CUDA kernel durations from torch profiler. Two measured
rounds per route use ten observations each, with a 256 MiB L2 scrub and synchronization
outside each observation. Same-process order was auto/native/FROST/FROST/native/auto;
reported values are the median of the two round medians. Planning/JIT and host
execution overhead are outside this kernel metric.

B1/Hq32 short caches lose by 1.89–2.23x; shared K=V at 8k still loses by 2.20x.
At 128k, FROST wins with both separate K/V (0.86–0.88x) and shared K=V (0.80–0.81x).
The policy therefore keeps the existing `B*Hkv >= 32` branch and restricts the
`Hq >= 32` small-batch shortcut to `Skv >= 131072`. This is the first verified
winning long-cache point, not a measured exact crossover; the 32k–128k gap is
uncharacterized. B8/Hq32/K8k is a known missed opportunity under this conservative
rule. SM120's head-count shortcut is also restricted to its historical 128k
measurement domain, while retaining its `B*Hkv >= 8` branch and GQA-128 exclusion.
That is a conservative domain restriction, not an inference of SM120 performance
from B200 or a claim of new SM120 GA timing. Other shape branches are unchanged.

The original ten CSVs / 863 decisive cells remain separate historical BF16,
cuDNN 9.27 / DSL 4.7.1 fitted data. Their consistency test does not constitute
independent validation of extrapolated shapes.

Provenance: the public wheel SHA256 is
`9c976d539786698c71d6bcdbe2053b7485fae062ff7e1215d2f2ff67b1a2149a`;
measured native extension SHA256 is
`1845c2dcde53c0fb00247dacce68a4c3ec1045b7cd026bae4bd4f27cdd8c4486`.
Raw per-case source/input/DSO/route hashes and observations are retained in the
handoff artifact `ga926/d512_followup/` and `ga926/d512_verified.json`.

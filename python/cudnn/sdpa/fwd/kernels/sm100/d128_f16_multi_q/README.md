# SM100/SM103 half multi-Q

`sdpa_fwd_sm100_d128_multi_q` fixes the two-query-tile main implementation. Exact D32/D64/D128 and FP16/BF16 use local geometry and dtype variants; D32 with two CTAs uses the existing D64 physical envelope.

`contract.py` owns lightweight admission and geometry, `config.py` owns configuration records and factories, and `host.py` calls the shared lowering service with this fixed `kernel.py`. Performance selection stays in `sdpa/fwd/heuristics.py`. Parser, binding, workspace and prepared launch infrastructure remain shared.

THD split-KV belongs to the single-Q implementation and is rejected here before compilation. Dense split-KV remains local to this body. Main pipeline methods retain the legacy implementation; the old module remains for standalone, envelope and architecture compatibility.

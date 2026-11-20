# Grouped small-half two-CTA backward accumulation

`kernel.py` owns the default grouped two-CTA main pipeline. The other existing main protocols have independent bodies:

| Body | Fixed protocol | Selection |
|---|---|---|
| `kernel.py` | Default warp roles and score-tail reuse barriers | Graph engine 20631; standalone default |
| `fa4/kernel.py` | Softmax-owned dS exchange and separate dQ reduce group | Standalone `warp_roles=16w_fa4` |
| `dq576/kernel.py` | Independent dQ TMEM image; epilogue program order carries reuse | Standalone `tmem_map=dq_576`, base SMEM map |
| `rubin_b2/kernel.py` | Two-stage exchange rings with explicit dQ read-release barrier | Standalone `tmem_map=dq_576, smem_map=rubin_b2` |

Each main entry rejects fields selecting another protocol. `chain.py` chooses the main from those explicit fields and owns the shared dense/packed setup, delta and fold launches. `prepared_host.py` owns the matching pointer chain. Existing cross-field refusals, workspace ABI, symbols and cache keys are retained. Stats load form, local mask handling, numerical accumulation mode and diagnostics remain typed variants within the implementations that already support them.

Packed delta/cast helpers come from `../_d128_thd_chain.py`; pointer views come from `../_prepared_d128_common.py`. The one-CTA attention body is not imported. The detailed pipeline and diagnostic documentation remains beside the source.

`contract.py` names fixed engine 20631; `host.py` forces the default main and declines failed cluster admission. Central ranking is outside this directory. The other protocols remain available through the existing standalone typed contract; they are not alternate main bodies hidden behind this graph engine or new public engine knobs. No determinism or batch-invariance guarantee is added.

`geometry.py` owns the base two-CTA geometry and diagnostic vocabulary; `config.py` owns its role, Stats and memory-map records and validation. The declaration/cluster selection policy stays in the shared backward layer. The one-CTA geometry remains shared with the deterministic dK/dV body and is not copied here.

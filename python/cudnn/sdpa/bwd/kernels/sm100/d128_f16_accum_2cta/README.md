# Grouped small-half two-CTA backward accumulation

`kernel.py` owns the grouped two-CTA main pipeline and its dense/packed setup entries. `prepared_host.py` owns the matching pointer chain. Its typed role, Stats, TMEM, SMEM and diagnostic renderings retain the original cross-field refusals, workspace ABI, symbols and cache keys.

Packed delta/cast helpers come from `../_d128_thd_chain.py`; pointer views come from `../_prepared_d128_common.py`. The one-CTA attention body is not imported. The detailed pipeline and diagnostic documentation remains beside the source.

`contract.py` names fixed engine 20631; `host.py` forces this body and declines failed cluster admission. Central ranking is outside this directory. Standalone diagnostic and numerical renderings remain available under their existing typed contracts; they are not public engine knobs. The isolation of their different barrier protocols remains a separate source audit. No determinism or batch-invariance guarantee is added.

`geometry.py` owns the base two-CTA geometry and diagnostic vocabulary; `config.py` owns its role, Stats and memory-map records and validation. The declaration/cluster selection policy stays in the shared backward layer. The one-CTA geometry remains shared with the deterministic dK/dV body and is not copied here.

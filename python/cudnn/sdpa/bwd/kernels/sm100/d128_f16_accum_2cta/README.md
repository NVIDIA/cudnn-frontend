# Grouped small-half two-CTA backward accumulation

`kernel.py` owns the grouped two-CTA main pipeline and its dense/packed setup entries. `prepared_host.py` owns the matching pointer chain. Its typed role, Stats, TMEM, SMEM and diagnostic renderings retain the original cross-field refusals, workspace ABI, symbols and cache keys.

Packed delta/cast helpers come from `../_d128_thd_chain.py`; pointer views come from `../_prepared_d128_common.py`. The one-CTA attention body is not imported. The detailed pipeline and diagnostic documentation remains beside the source.

This source extraction preserves the existing standalone selection contract. It does not by itself make the legacy graph row a fixed two-CTA owner, widen graph support, or add a determinism/batch-invariance guarantee.

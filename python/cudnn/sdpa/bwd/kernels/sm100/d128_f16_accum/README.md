# Small-half one-CTA backward accumulation

`kernel.py` owns the dense and packed accumulation entries and their grouped-head loader variants. Dense MHA specializes D64 or D128; the existing packed and grouped routes retain their D128 contracts. `prepared_host.py` binds the corresponding fixed launch chains and preserves their cache keys, symbols and workspace ABI.

Packed setup, delta and cast kernels live in `../_d128_thd_chain.py`, shared with the two-CTA chain and D192. Dense delta/cast/fold launches remain in `../_d128_chain.py`. Pointer views use `../_prepared_d128_common.py`. No helper imports another attention main body.

The existing diagnostic dQ-store control and grouped FP32 dKV reduction mode retain their guards and numerical contracts. This source extraction does not widen graph support or claim deterministic accumulation or batch invariance.

# SM107 D512 per-tensor FP8 role-split

This implementation owns the native Rubin D512 per-tensor FP8 pipeline.
`contract.py` fixes the public identity and physical cluster geometry;
`config.py` owns the configuration record and factory; `host.py` injects the
shared graph lowering service. `kernel.py` owns the device pipeline.

The kernel uses a four-CTA cluster, with two-CTA MMA groups. QK/softmax and
PV/correction run in different groups and exchange probabilities and row
factors through distributed shared memory. FP16/BF16 and MXFP8 pipelines
have separate owners. Common validation, binding, caching and launching
remain shared.

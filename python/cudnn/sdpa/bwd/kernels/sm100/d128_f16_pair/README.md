# Small-half deterministic backward pair

This implementation owns the dense D64/D128 FP16/BF16 chain on SM100/SM103.
The dK/dV body and dQ body have different output owners and worker roles;
together they form one fixed algorithm. A selected pair never changes to
accumulation or recompute.

- `kernel_dkv.py` owns the dK/dV pipeline; `kernel_dq.py` owns the dQ pipeline.
- `config_dq.py` owns the dQ configuration. The dK/dV memory geometry reuses
  the immutable accumulation configuration; its worker loops are independent.
- `prepared_host.py` fixes delta, dK/dV, optional ordered partial folds, and
  dQ, retaining their launch order, symbols, compile keys and workspace.
- `host.py` reuses declaration parsing and native binding from the common
  adapter while fixing the implementation. `contract.py` names the owner.
- `_d128_chain.py`, `_d128_device_common.py`, and
  `_prepared_d128_common.py` in the parent directory contain shared helpers.
  Configuration-bound helpers stay local to each rendered body.

Width, head ratio and the existing q-range split are internal variants.
Dense MHA supports either deterministic setting; grouped heads retain the
existing deterministic-request admission. THD and declarations outside the
pair's architecture, compiler, scale or index envelope decline.

The pair uses owned output tiles and ordered partial folds. This directory
does not make a batch-invariance claim: changing geometry can change the
chosen split and reduction order.

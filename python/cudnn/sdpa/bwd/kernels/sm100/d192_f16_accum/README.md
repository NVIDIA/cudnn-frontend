# D192/V128 half accumulation

This directory owns the fixed `sdpa_bwd_sm100_d192` implementation. Its public
engine ID remains 20612. FP16 and BF16 share the same main algorithm.

| File | Responsibility |
|---|---|
| `contract.py`, `host.py` | Identity, standalone support checks and API adapter |
| `config.py` | Fixed tile, warp-role, barrier and memory geometry |
| `kernel.py`, `kernel_thd.py` | Dense and packed main loops |
| `common.py` | Local helpers used by both main loops |
| `helpers.py`, `helpers_thd.py` | Delta, initialization, conversion and packed setup |
| `prepared_dense.py`, `prepared_thd.py` | Plan metadata, declaration validation and binding |
| `prepared_dense_host.py`, `prepared_thd_host.py` | Compiled pointer entries for each layout |
| `thd_contract.py` | Packed bounds, origins, workspace and launch validation |

Layout selects the dense or packed entry mechanically. Both entries retain
their own loops, with the same accumulation pipeline and warp-role protocol.
Packed addressing, descriptor acquisition and empty-sequence checks stay in
the packed entry. Neither entry imports the other main body.

Generic graph lowering, native binding and workspace guards remain shared.
Packed setup reuses the existing D128 metadata and row-conversion helpers.
The old standalone adapter module only re-exports this directory's class.

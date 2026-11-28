# SM90 three-WG split-output implementation

Engine 20540 owns the existing 64 x 64 tile domain: exact D64 paged KV,
native D128/D256/D512 and the bounded head envelopes. Dense, THD, paged,
split and packing eligibility remain the historical policy's. The body keeps
its existing QK/softmax/PV pipeline and cross-WG probability/row-factor handoff.

The immutable parameter ABI, layout helpers, prepared binder and launch
services remain shared. `host.py` fixes the body before lowering; candidate
ranking stays centralized. THD keeps NATURAL when the scheduler is omitted.
No runtime Stats-stride note is added.

The standalone entry remains in use and is retained. Apart from the additive-bias
leg (`has_bias`, compiled out of every other specialization), device classes remain
AST-equivalent to source baseline `5913f355354bd35d5be54494b91b1227b01917bb`. A bias
plan binds through `prepared_host.host_bias`: each WG1 warp stages its 16 bias rows
in shared memory with `cp.async` before QK, the softmax forms `scale * Q.K + bias`
before masking, and each exponent subtracts the row anchor before the base-2
conversion (SDPA Rule S17).

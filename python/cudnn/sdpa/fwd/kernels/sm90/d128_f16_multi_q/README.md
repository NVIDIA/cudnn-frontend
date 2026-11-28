# Dedicated SM90 D128 boundary

This is an independent pipeline, not a three-WG output-width variant. WG1 and
WG2 each own 64 different Q rows and every output column, with separate QK,
softmax and PV work and a two-stage K/V ring. The three-WG F05 body instead
hands probabilities and row factors to the second PV group and splits output
columns. D128 overlap, persistence, causal walk and claim scheduling remain
internal configurations of this dedicated body.

Engine 20541 fixes this body and its 128 x 128 tile. It serves the existing
exact-D128 dense graph subset; unsupported features retain the three-WG
owner. An omitted scheduler reaches the shared adapter unchanged so its
persistent/walk/claim defaults remain intact. Candidate ranking is centralized.

Apart from the additive-bias leg (`has_bias`, compiled out of every other
specialization), all top-level functions and classes retain their AST from
`5913f355354bd35d5be54494b91b1227b01917bb`. Native parity tests compare full
parameter records, output/Stats bytes, fresh binding and retained capture
against the existing standalone entry, including persistent and claim records.
No runtime Stats-stride note is added.

A bias plan (Q/K/V dtype only) binds through `prepared_host.host_bias`. WG0
loads each compute warpgroup's 64 x 128 bias tile per KV tile by TMA into that
warpgroup's own SW128 slot; the warpgroup's threads read their 32 words at the
start of the tile's softmax and release the slot through `tile_dsl.consume`, so
the refill overlaps the rest of that softmax and the next QK. The softmax forms
`scale * Q.K + bias` before masking, and each exponent subtracts the row anchor
before the base-2 conversion (SDPA Rule S17). An FP32 bias tile would not fit
beside the 192 KiB of Q/K/V/O storage; the three-WG owner serves it. A bias plan
keeps the static persistent schedules: the claim counter binds through the
native binder, which has no bias slots.

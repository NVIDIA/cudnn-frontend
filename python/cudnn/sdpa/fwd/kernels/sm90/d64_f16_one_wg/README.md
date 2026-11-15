# SM90 implementation ownership

The one-compute-warpgroup D64 body owns exact dense D64/V64 graphs on SM90.
Its eight formerly inherited methods are local copies from the three-warpgroup
body; all twelve methods retain the parent AST. The graph engine is 20539.
The old standalone entry remains for existing direct callers.

The immutable parameter ABI, pure layout helpers, prepared binder and launch
services remain shared. `host.py` fixes the body before lowering; candidate
ranking stays in the central heuristics. No runtime Stats-stride note is added.

Source baseline: `5913f355354bd35d5be54494b91b1227b01917bb`.

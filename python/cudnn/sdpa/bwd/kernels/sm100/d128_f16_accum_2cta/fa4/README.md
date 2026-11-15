# Two-CTA softmax-owned exchange

This main implementation fixes `warp_roles=16w_fa4`: softmax warps own the dS exchange, and a separate reduce group consumes dQ. Its main entry rejects another role shape. Existing typed Stats, diagnostic and memory-map refusals remain intact. The setup/delta/fold chain and pointer binder are shared with the other two-CTA implementation.

The standalone API selects this existing implementation by its typed role field. Graph exposure and protocol-specific source assertions are being qualified in the private refactor; no new ordinary engine is implied by this directory.

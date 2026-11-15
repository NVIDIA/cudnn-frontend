# Two-CTA softmax-owned exchange

This main implementation fixes `warp_roles=16w_fa4`: softmax warps own the dS exchange, and a separate reduce group consumes dQ. Its main entry rejects another role shape. Existing typed Stats, diagnostic and memory-map refusals remain intact. The setup/delta/fold chain and pointer binder are shared with the other two-CTA implementation.

The standalone API selects this existing implementation by its typed role field. The ordinary graph engine selects the default score-tail body. This directory does not expand graph admission or introduce a public engine knob.

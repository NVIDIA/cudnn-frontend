# SM100/SM103 half quadrant cooperation

`sdpa_fwd_sm100_d64_quadrant` fixes the quadrant-cooperative main implementation, using the D64 physical envelope for exact D32/D64. Each quadrant owns a key partition and combines partial output/Stats. The live query extent, after optional GQA packing, must be at most 32 rows; the packed group must divide 32.

`contract.py` owns admission and fixed geometry, `config.py` owns the configuration factory, and `host.py` calls the common lowering service with this fixed `kernel.py`. The public physical tile is 128x128 with one CTA. THD, bias and KV-mask forms are outside this owner's contract.

Candidate selection stays centralized. The general single-Q implementation has a separate owner and directory; no shape-dependent switch to that body occurs while replaying this owner's engine/knob record.

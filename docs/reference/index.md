# Optional kernel contracts

Use this section to look up architecture-specific OSS wrappers and fusion
families. These pages record exported signatures, layouts, constraints, and
fallback boundaries. They are reference inventories, not a claim that every
combination has a documentation-owned executable sample.

- [Choose an OSS engine or kernel](/optional-kernels/oss-engines) — Distinguish Graph heuristics, direct wrappers, dependencies, and fallback behavior.
- [Block-sparse attention](/optional-kernels/oss-block-sparse-attention) — Look up BSA forward and backward metadata and kernel constraints.
- [DeepSeek sparse attention](/optional-kernels/oss-deepseek-sparse-attention) — Look up DSA indexing, selection, score recomputation, and backward stages.
- [Native sparse attention](/optional-kernels/oss-native-sparse-attention) — Look up NSA compression, top-k, selection, and sliding-window components.
- [SDPA with head dimension 256](/optional-kernels/oss-sdpa-d256) — Look up the Blackwell forward and backward specializations.
- [Dense GEMM fusions](/optional-kernels/oss-dense-gemm) — Match block-scaled GEMM epilogues to their exported wrappers and classes.
- [Grouped GEMM fusions](/optional-kernels/oss-grouped-gemm) — Match contiguous expert layouts and fusion variants to their contracts.
- [Discrete grouped GEMM](/optional-kernels/oss-discrete-grouped-gemm) — Look up pointer-array expert-weight variants and lifetime rules.
- [Normalization fusions](/optional-kernels/oss-normalization) — Look up direct RHT and Graph-detected RMSNorm fusion paths.

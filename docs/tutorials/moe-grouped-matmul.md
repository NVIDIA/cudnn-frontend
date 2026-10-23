# Advanced: MoE grouped matmul

Execute several expert matrix multiplications from routing metadata without launching a separate graph per expert.

## Before you begin

Complete [Build and extend a graph](/tutorials/first-graph) and be comfortable with
dense matrix multiplication. Forward requires a backend that supports grouped
MoE matmul; the Python tests conservatively require cuDNN 9.18. The backward
examples require cuDNN 9.22 or newer, cublasLt 13.5 or newer, and an SM90
through SM119 GPU.

## Prepare routing metadata

Grouped matmul needs the token-to-expert assignment, per-expert token ranges or offsets, and weights laid out for the selected mode. Validate that offsets are monotonic, counts sum to the routed token count, and every index stays in range.

The three modes change which tensors are required and how output rows are ordered:

| Mode | Token layout | Required routing tensors | Output rows |
| --- | --- | --- | --- |
| `NONE` | Tokens are already grouped by expert | `FirstTokenOffset` | Same row order and count as `Token` |
| `GATHER` | Tokens remain in source-token order | `FirstTokenOffset`, `TokenIndex` | One row per entry in `TokenIndex`, grouped for expert execution |
| `SCATTER` | Routed rows are supplied and output returns to token order | `FirstTokenOffset`, `TokenIndex`, `TokenKs`, positive `top_k` | Same row count as `Token`, scattered according to routing metadata |

`FirstTokenOffset` is INT32 with shape `[batch * experts, 1, 1]`. It stores the first routed row for each expert; the total row count is taken from Token. Offsets must be nondecreasing. Empty experts use equal adjacent boundaries.

`TokenIndex` is INT32 with shape `[1, routed_rows, 1]`. Gather and Scatter require it. `TokenKs` has the same shape and is required only for Scatter. The current backend reads `top_k` only in Scatter mode.

## Build grouped execution

The forward samples build the same two-expert `NONE`-mode problem and compare
every expert slice with an independent dense matrix-multiplication reference.

<Tabs>
<Tab title="Python" language="python">


<Code
  src="../samples/python/moe_grouped_matmul.py"
  language="python"
  maxLines={0}
/>

</Tab>
<Tab title="C++" language="cpp">


<Code
  src="../samples/cpp/moe_grouped_matmul_basic.cc"
  language="cpp"
  maxLines={0}
/>

</Tab>
</Tabs>
The C++ backend infers `[1, TokenIndex.dim[1], N]` in Gather mode and `[1, Token.dim[1], N]` otherwise. The Python IR initially carries a provisional token-row shape for every mode and does not push that inferred layout while creating the graph representation, allowing the backend to apply its mode-specific rule. Weight uses `[experts, K, N]`; repository tests use column-major inner dimensions with stride `[K * N, 1, K]`. Use a virtual output only when another node consumes it in the same graph.

The Graph operation currently requires at least cuDNN 9.15 in source, while Python tests gate forward execution at 9.18. Treat 9.18 as the conservative documented minimum until maintainers reconcile that disagreement. Start by changing only the expert boundary and verifying that every token is still covered exactly once. Then change expert count, weight layout, or routing mode one at a time.

## Add weight gradients

[`moe_grouped_matmul_bwd`](/api/python/moe-grouped-matmul-bwd) computes the supported grouped backward result. Keep its routing metadata consistent with forward and verify each expert independently against a dense reference.

<Tabs>
<Tab title="Python" language="python">


<Code
  src="../samples/python/moe_grouped_matmul_backward.py"
  language="python"
  maxLines={0}
/>

</Tab>
<Tab title="C++" language="cpp">


<Code
  src="../samples/cpp/moe_grouped_matmul_backward.cc"
  language="cpp"
  maxLines={0}
/>

</Tab>
</Tabs>
The result is `dWeight[e] = Token[e].T @ dOutput[e]`, with expert row ranges
defined by `FirstTokenOffset`. The current repository tests require cuDNN 9.22
or newer, cublasLt 13.5 or newer, and an SM90 through SM119 GPU for this
operation. Both samples check every expert against a dense reference.

## Distinguish wrappers

The Graph API operation and optional SM100 CUTLASS DSL wrappers have different stability, dependency, layout, and architecture contracts. Select the optional wrapper only after [choosing an OSS engine](/optional-kernels/oss-engines).

## Expected result

The forward samples execute `NONE` mode and verify every expert slice against
dense matrix multiplication. On a backend that meets the stricter backward
requirements, the backward samples verify each expert's weight gradient using
the same token boundaries as forward. Otherwise the manifest performs the
documented static fallback.

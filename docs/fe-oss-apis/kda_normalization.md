# KDA graph Q/K normalization

`graph.kda(..., use_qk_l2norm=True)` retains its existing normalization:

```text
x * rsqrt(max(sum(x * x), 1e-24))
```

The optional, append-only `qk_l2norm_additive_epsilon` attribute selects additive
normalization instead. For example:

```python
graph.kda(
    q=q, k=k, v=v, g=g, beta=beta, cu_seqlens=cu_seqlens,
    use_qk_l2norm=True,
    qk_l2norm_additive_epsilon=1e-6,
)
```

This computes `x * rsqrt(sum(x * x) + epsilon)` in FP32 within the KDA kernel,
without allocating normalized tensors or submitting separate normalization
kernels. The normalized values remain in FP32 through the subsequent gate
scaling, without an intermediate FP16/BF16 rounding step. Reduction and
rounding order can differ from a separately materialized normalization, so
bitwise agreement is not promised.
The attention scale is applied by KDA after normalization, as before.

`None` preserves the original behavior, including its fused rounding order.
A supplied epsilon must be a finite positive normal FP32 value and requires
`use_qk_l2norm=True`. This is a numerical operation attribute, not a tuning knob;
it participates in the graph and kernel compile contracts.

The initial implementation supports the SM100-family FROST KDA **forward graph**
paths, including chain, uncut and value-dimension split schedules. Other engines
explicitly decline the attribute. The Torch autograd API, backward graph API and
standalone summary API do not expose it yet; training requires a matching
normalization derivative and checkpoint semantics. Existing state ownership,
workspace, stream and execution contracts are unchanged.

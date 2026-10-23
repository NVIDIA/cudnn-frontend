# Apply rotary position embeddings

Use RoPE to rotate paired feature dimensions by position-dependent sine and cosine values, either as a standalone graph operation or as part of attention.

## Before you begin

Complete [Build and extend a graph](/tutorials/first-graph) and review the
[attention layout conventions](/concepts/attention#layouts). Fix the head
dimension, rotated dimension, frequency representation, and output scale before
building forward and backward.

The standalone forward and backward operations require cuDNN 9.24 or newer.
The recorded cuDNN 9.15.1 environment compiled or syntax-checked these samples
but could not execute them.

## Build RoPE

<Tabs>
<Tab title="Python" language="python">


<Code
  src="../samples/python/rope_basic.py"
  language="python"
  maxLines={0}
/>

</Tab>
<Tab title="C++" language="cpp">


<Code
  src="../samples/cpp/rope_basic.cc"
  language="cpp"
  maxLines={0}
/>

</Tab>
</Tabs>
The input and frequency tensor layouts must match the operation contract. When `0 < rope_dim < head_dim`, the operation rotates the last `rope_dim` features and applies scaled pass-through to the leading `head_dim - rope_dim` features. The frequency tensor's last dimension must equal `rope_dim`; `rope_dim=0` selects full-head rotation. Apply any output scaling explicitly and keep the frequency tensor dtype compatible with the chosen compute path.

## Build backward

<Tabs>
<Tab title="Python" language="python">


<Code
  src="../samples/python/rope_backward_basic.py"
  language="python"
  maxLines={0}
/>

</Tab>
<Tab title="C++" language="cpp">


<Code
  src="../samples/cpp/rope_backward_basic.cc"
  language="cpp"
  maxLines={0}
/>

</Tab>
</Tabs>
Backward consumes the upstream gradient and the same frequency representation as forward. Preserve the same pairing and partial-dimension convention.

## Integrate with attention

Rotate query and key before SDPA, or use an attention option that explicitly accepts RoPE metadata. Do not apply both paths to the same tensor. For YaRN frequency helpers, record base, scaling factor, original maximum position, and dtype in the graph cache key.

## Fold attention scale into RoPE

SDPA normally forms scores as `alpha * Q_rotated * K_rotated^T`, where `alpha` is usually `1 / sqrt(D)`. The RoPE `output_scale` multiplies both rotated pairs and the pass-through portion of the output. You may therefore move the scale into either RoPE input to SDPA:

- Q-only: set Q RoPE `output_scale=alpha`, K RoPE `output_scale=1`, and SDPA `attn_scale=1`.
- K-only: set Q RoPE `output_scale=1`, K RoPE `output_scale=alpha`, and SDPA `attn_scale=1`.
- Symmetric: choose scales whose product is alpha, commonly `sqrt(alpha)` for both, and set SDPA `attn_scale=1`.

Backward must follow the same factorization. Pass the matching forward scale to each `rope_backward` operation and use the same SDPA backward attention scale as forward. Moving alpha changes where the chain-rule factor is applied; omitting it or applying it twice produces incorrect Q and K gradients even when forward scores appear plausible.

YaRN returns an `mscale` intended for RoPE `output_scale`. When combining YaRN magnitude scaling with attention scaling, multiply the intended factors according to the chosen Q-only, K-only, or symmetric factorization and include them in graph cache identity.

The current public SDPA API does not expose an `input_is_log2` option. Do not assume that a softmax base-conversion optimization is available through this interface.

## Verification limits

The public `cudnn.yarn` helpers do not have individual API pages or executable
documentation samples. RoPE scale folding is source-backed, but the published
samples do not compare folded and unfurled forward and gradient results.

## Expected result

On cuDNN 9.24 or newer, the paired forward and backward samples execute with
matching layouts and frequency tensors and compare their outputs with
independent RoPE calculations. In the recorded cuDNN 9.15.1 environment, the
C++ samples compiled and the Python samples passed syntax checks only.
Scale-folding equivalence and public YaRN helpers remain outside the published
runtime verification and need separate forward and gradient checks.

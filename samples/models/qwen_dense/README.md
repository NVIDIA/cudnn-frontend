# Dense Qwen 3.5-architecture model

Samples for the dense model of the Qwen 3.5 architecture at its published 27B-class layer widths: hidden 5,120; 64 layers,
three linear-attention (gated delta-rule) layers to one attention layer; linear attention with 16 key heads and 48 value heads
of 128; attention with 24 query heads and 4 key/value heads of 256; MLP width 17,408; 8-bit projections, a 4-bit MLP and a
4-bit output head. Area rules and the helper layer: [../README.md](../README.md).

| script | what it runs | status |
|---|---|---|
| `00_decode.py` | plain greedy decode, token ids in and token ids out, on the reduced seeded model; self-checking | this page |

```bash
python samples/models/qwen_dense/00_decode.py                          # 2 layers, 1 and 4 sequences, 32 tokens each
python samples/models/qwen_dense/00_decode.py --sequences 4 --layers 4
python samples/models/qwen_dense/00_decode.py --stack reference        # every block on the reference: RAN, never PASS
```

| flag | default | meaning |
|---|---|---|
| `--sequences N[,N...]` | `1,4` | sequences per run, fixed before the first step; a list repeats the run per count |
| `--layers N` | `2` | layers of the reduced model: the last N of the published order, so the model ends in an attention layer |
| `--stack fe-cublas\|reference` | `fe-cublas` | explicit, never chosen by what is installed |
| `--provider BLOCK=KEY` | -- | selects one declared provider row of a block; the key list is closed |
| `--force-tokens N` | `32` | decode exactly N tokens per sequence, ignoring the end token |
| `--reference-precision fp64\|ieee-fp32` | `fp64` | accumulation of the reference blocks; printed in the header |

## What the default run is

The reduced model keeps every layer width and reduces what the run prints as reductions: 2 of 64 layers (one linear-attention
and one attention layer), a vocabulary of 15,520 ids, seeded weights and a synthetic byte-level tokenizer. The weights are drawn
on the CPU generator and stored in the model's three formats, so the 8-bit and 4-bit paths are exercised. Two properties of
the seeded weights are choices, not accidents: the projections that write back into the residual stream are drawn at a quarter
of the unit scale, and the output head is a seeded permutation of the embedding table. The second makes the greedy token
decidable by a wide margin, and it means the produced ids are a walk through that permutation: with seeded weights the layers
move the logits, not the choice. The layers are verified block by block.

The stack `fe-cublas` is a reference path, not a fast path. The linear-attention step runs on cuDNN Frontend's fused kernel;
every multiply runs on cuBLAS over a bf16 copy of the quantised weights, which moves 2 bytes per weight element where the
stored formats hold 1 or 0.56; the rest runs in PyTorch; and every launch is issued from Python, one block after the other.
It exists to prove the loader, the loop, the state handling and the tie-out, and it is the correctness oracle of the stacks
that follow. It claims no number.

## Who serves each block (stack `fe-cublas`)

| block | calls per step (64 layers) | provider key | entry |
|---|---|---|---|
| token embedding lookup | 1 | `torch.index_select` | torch.index_select |
| residual add + RMSNorm + quantise | 129 | `torch_ref.add_norm_quantize` | torch_ref.add_norm_quantize, fallback: reference chain |
| 8-bit in-projection (linear attention QKVZ / attention QKV) | 48 / 16 | `cublas.bf16_dequantized` | torch.matmul over a bf16 copy, fallback: bf16 copy of 8-bit weights |
| linear-attention step | 48 | `fe.gdn_decode_fused` | cudnn.GdnDecodeFused |
| 8-bit out-projection | 48 / 16 | `cublas.bf16_dequantized` | torch.matmul over a bf16 copy, fallback: bf16 copy of 8-bit weights |
| MLP gate / up + SiLU x up + quantise | 64 | `cublas.bf16_dequantized+torch_ref.silu_mul_quantize` | torch.matmul over a bf16 copy + torch_ref.silu_mul_quantize, fallback: bf16 copy of 4-bit weights |
| MLP down | 64 | `cublas.bf16_dequantized` | torch.matmul over a bf16 copy, fallback: bf16 copy of 4-bit weights |
| attention preparation | 16 | `torch_ref.attention_prepare` | torch_ref.attention_prepare, fallback: reference chain |
| attention core | 16 | `torch_ref.attention` | torch_ref.attention, fallback: reference chain over a plain per-layer cache |
| gate + quantise | 16 | `torch_ref.gate_quantize` | torch_ref.gate_quantize, fallback: reference chain |
| output head | 1 | `cublas.bf16_dequantized` | torch.matmul over a bf16 copy, fallback: bf16 copy of 4-bit weights |
| token choice | 1 | `torch.argmax` | torch.argmax |

484 block calls per step on the full model, 20 on the two-layer model and 34 on the four-layer one. By provider: cuDNN
Frontend 1; fallbacks 8 (cuBLAS over copies 4, PyTorch chains 4); PyTorch as the natural provider 2.

| GPU classes | sequences | regime | note |
|---|---|---|---|
| any | 1 and more | launch-bound | every block of these stacks is launched from Python, one launch or more per block; no byte-bound form is declared, and the sequence count at which one would win is not measured on any class |

## The checks of a run

Against the plain-PyTorch reference of every block, which keeps the chain's casts and accumulates in fp64
(`model_samples/tie_out.py` has the definitions):

| check | what must hold |
|---|---|
| replay | the same inputs from a fresh state give byte-identical ids and logits |
| blocks, teacher-forced | every multiply within one bf16 rounding of the reference; the linear-attention step with more than 99.9 % of its 8-bit output bytes equal over the run, more than 99.9 % of its recurrent state within 1e-4 and all of it within two bf16 units of the state's largest value, and its conv history byte-identical |
| tokens | the greedy token equals the reference stack's at every step whose margin exceeds twice the largest logit distance of the run; steps below it are counted and printed as near-ties |
| batch invariance | sequence 0 alone gives the bytes of sequence 0 among 4: a gate for the linear-attention step on compute capability 10.0, 10.3 and 12.0, a printed fact for every other block and class |

The state band has a share and a cap because a state "within 1e-4 everywhere" does not hold against an fp64-accumulated
reference: the fused step rounds its gates and conv taps to bf16 as a serving chain does, and where its fp32 sum falls on the
other side of a rounding midpoint one head's state moves by up to a few 1e-3 while the rest agrees to fp32 rounding.

## Support per GPU class

`cudnn.GdnDecodeFused` serves compute capability 10.x and 12.x and declines elsewhere; the sample then prints
`[00] SKIP cudnn.GdnDecodeFused serves compute capability 10.x or 12.x; this device is <cc>` and exits 0. The entry is
validated on compute capability 10.0 and 12.0 (`docs/operations/linear_attention/gdn_decode_flashinfer_backend.md`).

| compute capability | the default run of this sample |
|---|---|
| 10.0 (data-center Blackwell) | `[00] PASS`, verified |
| 10.3 | not yet verified |
| 10.7 | runs (`[00] PASS`), unverified: batch invariance of the fused step is reported on this class, not gated |
| 12.0 | not yet verified by this sample |
| any other | `[00] SKIP`, by design |

Memory of the default run (2 layers, 1 and 4 sequences): 1.7 GiB of bf16 copies; PyTorch held 2.5 GB at its peak and
reserved 3.7 GB on a compute capability 10.0 GPU, so the sample asks for about 3.7 GB free there. On compute capability 12.x
it adds an allowance of 3 GiB for the fused step's first launch; that allowance is not yet measured by this sample.
The four-layer run (`--layers 4`) holds 3.1 GiB of copies. The whole default run takes well under a minute on a data-center
Blackwell GPU: about 5 s to draw the weights, 1 to 2 s to compile the fused step per sequence count, the rest for the decode
and its checks. These are orientation values of one run, not measurements of record.

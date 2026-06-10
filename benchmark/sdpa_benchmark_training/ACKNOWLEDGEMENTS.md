# Acknowledgements

## Flash-Attention and Tri Dao

cuDNN's attention stack has been shaped by the Flash-Attention project.
Many of the techniques that define a modern fused, IO-aware attention
kernel — the online-softmax recurrence, the tile schedule, the
warp-specialized producer / consumer split, the asynchronous epilogue —
were articulated and shipped early by Tri Dao and the Flash-Attention
contributors, and cuDNN's SDPA engines have benefited from studying their
kernels and numerics across successive generations.

We work in close collaboration with Tri Dao and the Flash-Attention project,
and we are grateful for the openness with which the techniques, the kernels,
and the rationale behind them have been shared. Many of the optimizations
that make cuDNN attention competitive today were absorbed directly from that
work, and were possible only because the underlying ideas — and the reference
implementations — were available to study, profile, and adapt. The current
generation of cuDNN attention reaches rough performance parity with
Flash-Attention 4, and that parity is, in large part, a reflection of how
much of FAv4's thinking now lives inside cuDNN.

This is a collaboration we value, and one we hope to keep deepening.

## Techniques absorbed from Flash-Attention 4

The following list is not meant for a landing page — we surface it here for
anyone who asks where specific ideas in cuDNN attention came from. Each of
these originated in (or was first shipped by) the Flash-Attention project,
and was subsequently adapted inside cuDNN across architectures (Ampere
through Blackwell) and across data types (fp16, bf16, fp8):

- **Rescale-threshold skipping.** Avoiding the per-tile output rescale when
  the running-max update is small enough that the correction factor is
  numerically a no-op. Adapted to cuDNN's softmax recurrence and extended
  to additional dtypes and tile shapes.
- **Skip correction.** Elision of the second-pass correction when a tile
  cannot move the running max — applied across architectures.
- **MUFU exp2 emulation.** The fast-path approximation of `exp2` used to
  hide MUFU latency in the softmax inner loop, re-implemented across
  architectures and dtypes.

In each case the core insight is the Flash-Attention team's; cuDNN's
contribution is the porting, generalization, and integration into the
heuristics, code generation, and graph layer that ship in the library.

## Citation

If you use cuDNN's SDPA / Flash-Attention engines in your work, please cite
the foundational Flash-Attention papers in addition to any cuDNN reference:

```bibtex
@inproceedings{dao2022flashattention,
  title={Flash{A}ttention: Fast and Memory-Efficient Exact Attention with {IO}-Awareness},
  author={Dao, Tri and Fu, Daniel Y. and Ermon, Stefano and Rudra, Atri and R{\'e}, Christopher},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2022}
}

@inproceedings{dao2023flashattention2,
  title={Flash{A}ttention-2: Faster Attention with Better Parallelism and Work Partitioning},
  author={Dao, Tri},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024}
}
```

Upstream source: <https://github.com/Dao-AILab/flash-attention>.
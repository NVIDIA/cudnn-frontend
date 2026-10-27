# BSA benchmarks

| Script | What it measures |
| --- | --- |
| `benchmark_sm90_blk128.py` | SM90: native blk128 vs. the same sparse graph expressed in blk64 |
| `benchmark_sm120_blk128.py` | SM120: native blk128 BF16 forward |
| `benchmark_sm100_kpage.py` | SM100/SM103: blk128 BSA with and without `kpage` on layouts from open-source video sparse-attention systems |

## `benchmark_sm100_kpage.py`

`kpage_layouts.py` provides blk128 sparse layouts of three open-source video
sparse-attention systems: HunyuanVideo-1.5 SSTA, Sliding Tile Attention and
FastVideo VSA. They are rebuilt from the released tile geometry, padding rule,
resolution and frame count; block occupancy does not depend on sample
content, so no data files are needed. Q/K/V are synthetic (random normal) in
every case.
`kpage_layouts.BENCHMARK_LAYOUTS` lists the eight default layouts:

| layout | system | configuration | heads x tokens | what makes blocks partial |
| --- | --- | --- | --- | --- |
| `ssta_hy15_480p_121f` | HunyuanVideo-1.5 SSTA | 848x480, 121 frames: latent 31x30x53 in (6,8,8) tiles; per head a (3,3,3) tile window plus top-32 tiles; 356 text tokens | 16 x 55168 | a full tile is 384 tokens = 3 blocks; boundary tiles and the text tokens end in partial blocks (10% padding) |
| `ssta_hy15_480p_241f` | HunyuanVideo-1.5 SSTA | 848x480, 241 frames, window + top-64 | 16 x 106368 | as above (8.5%) |
| `ssta_hy15_720p_121f` | HunyuanVideo-1.5 SSTA | 1280x720, 121 frames, window + top-32 | 16 x 116864 | as above (4.2%) |
| `ssta_hy15_720p_241f` | HunyuanVideo-1.5 SSTA | 1280x720, 241 frames, window + top-64 | 16 x 225664 | as above (2.5%) |
| `sta_hy_720p_117f` | Sliding Tile Attention | 1280x720, 117 frames, per-head windows of the HunyuanVideo mask search, 256 text tokens | 24 x 109056 | tile-aligned in time: only the spatial boundary and the text tokens are partial (0.7%), the near-dense regime |
| `vsa_wan_480p_81f_t64` | FastVideo VSA on Wan2.1 | 832x480, 81 frames, (4,4,4) cubes of 64 tokens, sparsity 0.875 | 12 x 79872 | every cube gets its own block, so every block is at most half full (59%); all query blocks take the M64 x N256 route |
| `vsa_wan_720p_81f_t256` | FastVideo VSA on Wan2.1 | 1280x720, 81 frames, (4,8,8) cubes of 256 tokens | 40 x 84480 | a full cube is 2 blocks; boundary cubes are partial (10.5%) |
| `vsa_wan_720p_81f_t64` | FastVideo VSA on Wan2.1 | 1280x720, 81 frames, (4,4,4) cubes | 40 x 184320 | as the 480p (4,4,4) layout, 2.3x more tokens (59%) |

All of these pad every tile on the query side as well as on the key side, so
the KPage layout is built with `kpage_prepare_layout(..., q_block_sizes=...)`;
the base path has no query-side padding and computes those rows.

```bash
cd benchmark/bsa
python benchmark_sm100_kpage.py --check                       # eight layouts, fp32 spot checks
python benchmark_sm100_kpage.py --layouts vsa:832x480x81:64:0.875:12 --iters 20
```

Columns: `pad%` is the fraction of padding rows after blk128 placement
(query / key when the partitions differ), `part%` the fraction of nonempty KV
blocks that are not full, `x` the base-BSA / KPage time ratio. Both paths
run with the same `block_sizes`; the base path has no query-side padding and
computes those rows. `--check` compares O (both paths) and dQ (KPage) of sampled query
blocks against an fp32 reference.

One run on an SM103 GPU (148 SMs, CUDA 13.2, PyTorch 2.12.0a0, CuTe DSL 4.8.0, 10 iterations after 3 warmups and a one-second clock warm-up):

| layout | tokens | H | pad% | part% | BSA fwd ms | KPage fwd ms | x | BSA f+b ms | KPage f+b ms | x |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ssta_hy15_480p_121f | 55168 | 16 | 10.0 | 18.3 | 6.28 | 4.94 | 1.27 | 24.81 | 21.81 | 1.14 |
| ssta_hy15_480p_241f | 106368 | 16 | 8.5 | 15.5 | 21.83 | 19.01 | 1.15 | 79.76 | 73.11 | 1.09 |
| ssta_hy15_720p_121f | 116864 | 16 | 4.2 | 12.2 | 16.09 | 14.14 | 1.14 | 58.48 | 56.84 | 1.03 |
| ssta_hy15_720p_241f | 225664 | 16 | 2.5 | 9.1 | 54.14 | 47.69 | 1.14 | 194.31 | 180.09 | 1.08 |
| sta_hy_720p_117f | 109056 | 24 | 0.7 | 5.9 | 57.82 | 52.52 | 1.10 | 208.76 | 201.82 | 1.03 |
| vsa_wan_480p_81f_t64 | 79872 | 12 | 59.0 | 100.0 | 4.08 | 1.27 | 3.23 | 12.98 | 6.11 | 2.13 |
| vsa_wan_720p_81f_t256 | 84480 | 40 | 10.5 | 16.7 | 14.85 | 12.50 | 1.19 | 55.41 | 52.07 | 1.06 |
| vsa_wan_720p_81f_t64 | 184320 | 40 | 59.0 | 100.0 | 71.67 | 18.78 | 3.82 | 222.82 | 101.12 | 2.20 |

`tokens` is the padded sequence length.

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The MXFP8 BACKWARD of the gated attention block -- ``GatedAttentionBlockBwd(quant=MxQuantSpec, ...)`` over the MXFP8 training
forward's record and the caller's transposed artifacts -- accept (Rubin) and reject (any CUDA device) suite.

The ACCEPT cells run on Rubin over the real chain (the MXFP8 backward's API, kernels, stages and oracles on one tree); the
REJECT cells are written against the declared API contract and run on any CUDA device.  Every bound is one calibrated
elsewhere and applied where it was calibrated (below); magnitudes are printed on every cell and never widened -- a miss is
reported with its magnitude and asked about (tolerances are sacred).

The accept matrix (one source: the cells table below, ids ``<shape>-<norm|rope_only>``), test geometry ``d_model 512, h_q 8,
h_kv 2, D 256, rope 64``; ``B*S % 32 == 0`` on every accept cell (the weight-gradient GEMM's K blocks)::

    s256_causal_b1      S=256  causal B=1 GQA 8/2  norm + rope_only
    s512_causal_b2      S=512  causal B=2 GQA 8/2  norm + rope_only      THE bitwise / graph / CUPTI cell
    s992_causal_b1      S=992  causal B=1 GQA 8/2  norm                  992 = 31 x 32; S % 128 = 96: q- AND kv-padded (the row's SF re-stagings)
    s1008_causal_b2     S=1008 causal B=2 GQA 8/2  norm                  T = 2016 = 63 x 32 (S itself % 32 = 16): the rule binds B*S
    s256_dense_b1       S=256  dense  B=1 GQA 8/2  norm + rope_only
    s1024_dense_b1_mha  S=1024 dense  B=1 MHA 8/8  norm                  no dkv_reduce, one dQ launch per chunk
    s512_dense_b2       S=512  dense  B=2 GQA 8/2  rope_only
    s512_causal_b1_mha  S=512  causal B=1 MHA 8/8  norm                  the block-scale arm's MHA dQ (one launch)
    s256_causal_b2_rope S=256  causal B=2 GQA 8/2  rope_only             n_kv = 1 at B = 2 (phase drift)
    s512_causal_b2_delayed S=512 causal B=2 GQA 8/2 norm                 grad_scaling="delayed" fed the current run's scale_dy: bitwise the current cell
    s1000_causal_b1_dgrad_only S=1000 causal B=1 GQA 8/2 norm            need_dw_qkvg=False: T = 1000 SERVED (the dgrad contracts over N); h_t refused
    s992_causal_b1_mha  S=992  causal B=1 MHA 8/8  norm                  LAUNCH COUNT + the bitwise layer (not a matrix cell): the pads without the fold
    s384_causal_b1      S=384  causal B=1 GQA 8/2  norm                  LAUNCH COUNT + the bitwise layer (not a matrix cell): the kv-side pads alone

Stage-localised bounds -- none invented, each applied where it was calibrated: every MXFP8 payload and scale-factor blob
BITWISE the torch quantization of the kernel's own bf16 source (``quantize_to_mxfp8`` for the SDPA layouts -- ``do8 / sf_do``,
``do_T8 / sf_do_T``, ``q8 / q_T8 / k8 / k_T8 / v8`` and their blobs --, the padded canonical blob builder for ``dqkvg8 /
sf_dqkvg`` and ``dqkvg_t8 / sf_dqkvg_t``; ``q8 / sf_q`` and ``k8 / sf_k`` ALSO ``torch.equal`` the FORWARD workspace's bytes);
``dy8`` / ``og8`` and the dY scalars bitwise (torch's saturating RNE cast, the "current" scale formula ``grad_scale_from_amax``,
the alphas ONE fp32 product each; the 21 dead slots exactly 0.0); ``delta`` bitwise the SDPA chain's own ``dot_do_o`` AND the
block's dQ / dK / dV bitwise a standalone ``SdpaBwdDslSm107Mxfp8(external_delta=False)`` run over the block's own payloads (on
this row the external delta IS the row's pre-pass); the out-projection GEMMs on the block's e4m3 operands under the GEMM suite's
bound (``rtol 2^-7``, ``atol = rtol * max|ref|``); the two block-scale GEMMs under the same bound on fp64 of the operands
dequantized THROUGH THE BLOBS -- the orientation guard: a blob over the un-transposed matrix passes every host check (the byte
count is symmetric) and fails here, which ``test_mxfp8_wrong_orientation_blob_fails_the_gemm_bound`` pins on every accept
cell; the SDPA stage under the MXFP8 row's recipe (``_GRAD_TOL`` atol 0.08 / rtol 0.2 with ``assert_close_fp8_grad``'s flip
budget) against the row's own ONCE-ROUNDED reference, its per-Q-head partials (on their live rows) against the per-head reference (fp32 ``dk_part``,
bf16 ``dv_part``) under the same recipe, and the bf16 bound FORM printed against the once-rounded fold and asserted against
the FOLD-MODELLED one (dK once-rounded from fp32 partials, dV per-Q-head bf16 partials summed in the fold kernel's fixed
order) once the first run recorded its margin; ``dh / dW_o / dW_*_norm`` against the oracle SEEDED with the block's own bf16
dQ / dK / dV under the bf16 block's bound, ``dW_qkvg`` in the row-budgeted form with its attribution (every row outside a row
a ``dqkvg_t8`` code flip touched -- an e4m3 code of block ``(n, t // 32)`` differing from the oracle's bf16 band quantized at the
kernel's own scale byte -- whose pre-cast band is itself inside its bound); end to end against the fold-modelled (M) oracle
(asserted in the row-budget form once measured), the once-rounded (M) and the unquantized-gradient (U) oracles (printed).

Launch count: ``mxfp8_expected_launches`` (host-checkable, COMPUTED -- no typed count anywhere in this module): the block's
own launches (10 with every gradient, the fp8 chain's shape: the fused PROLOGUE -- init, the dY amax partials, the Q / K rebuild's
MX epilogue writing q8 / q_T8 / k8 / k_T8, v8 --, quantize dY, B2, B3, the DUAL-AXIS dO quantize, B1, B4, B5+B6, the fused EPILOGUE
-- the dW_norm reduce and the dual-axis canonical dQKVG cast; built iff one of its jobs exists --, B7, B8) + the MXFP8 row's
``1 + c*(2+q) + (g > 1)`` with ``q = 1`` under ``api_dsl_sm107.DQ_SINGLE_LAUNCH`` (ONE dQ launch per head chunk on the block-scale arm
too: its dQ record takes ``b_head_group = group``; ``shipped_dq_launches`` reads the constant) + the row's staging terms read off the
adapter (``+8`` at ``S % 128 != 0``, ``+7`` GQA / ``+6`` MHA at ``S % 256 != 0``, ``+4`` under a dS zero-fill): 15 at
``s512_causal_b2-norm``, 15 rope_only (the epilogue stays for the cast), 14 at the MHA cells, 30 at the two padded GQA cells with weight
gradients, 29 at the padded dgrad-only cell, 28 at the padded MHA cell, 22 at the kv-side-only padded cell -- CUPTI decides, never the
formula.  MEASURED (Rubin cc 10.7, 204 SMs): ``len(kernels) == formula == expected`` on all nine census cells -- 15 / 15 / 14 / 14 / 30 /
30 / 29 / 28 / 22 in the order above (``s512_causal_b1_mha`` and ``s1024_dense_b1_mha`` both 14; ``s992_causal_b1`` and
``s1008_causal_b2`` both 30) -- with 0 memsets and 0 memcpys; the 397B geometry (``d_model 4096, h_q 32, h_kv 2``, B = 1, S = 512,
causal, norm, ``c = 1``, ``g = 16``) is the tenth census cell (``s512_causal_b1_397b``, launch count + the bitwise / finiteness layer,
no oracle): 15, MEASURED the same way -- the fp8 chain's count.  The two changes arrived one at a time, each MEASURED on the same cells:
the unfused chain over the row's per-member dQ 28 / 27 / 24 / 24 / 43 / 43 / 41 / 38 / 35 and 40 at 397B (on a 204-SM and a 212-SM part
alike), the unfused chain over the single-launch dQ 25 / 24 / 24 / 24 / 40 / 40 / 38 / 38 / 32 and 25, the fused chain over the
per-member dQ 18 / 18 / 14 / 14 / 33 / 33 / 32 / 28 / 25 (30 at 397B by the formula).

Rejects match the ATTRIBUTE NAME only (``match="quant"``, ``"h_t"``, ``"scale_dp"``, ...): the message prose is owned and
pinned by the API's own test module (``test_block_backward.py``), so a wording change touches one test.

Margins (Rubin cc 10.7, 204 SMs, SM clock locked at 2376 MHz; the merged suite's first run measured the 13 cells that reach the
stages, its re-run -- after two test-side shape fixes -- all 14 with the (M) row budget and the bf16 bound form ASSERTED; worst cell
as a fraction of the bound named for that stage; "rows outside" = rows of dh (tokens) / dW (output rows) with a cell outside the bf16
bound against the ``1e-5 x rows x keys`` row budget).  The bitwise layer held on every cell (every e4m3 payload and scale-factor blob
against the torch quantization of its own source, ``q8 / sf_q`` and ``k8 / sf_k`` ``torch.equal`` the forward's bytes, the dY scalars,
the 21 dead slots, ``delta``, and the block's dQ / dK / dV ``torch.equal`` a standalone ``SdpaBwdDslSm107Mxfp8(external_delta=False)``
run); the four GEMM-side bounds sit at 0.148-0.232 (dO, B1, B7, B8); the SDPA stage under the row recipe passed on every cell with
0 d-rows outside the bf16 bound form on dQ / dK / dV, the bf16 bound FORM against the fold-modelled reference at most 0.272 / 0.279 /
0.234 of it (ASSERTED), dK vs the once-rounded reference rel RMS <= 0.00013 (fp32 partials rounded once), dV vs the fold-modelled one
<= 0.00011 (exactly 0 on 6 of the 12 GQA cells) and vs the once-rounded one 0.0026..0.0028 (the per-Q-head bf16 partials' cost, as
designed; at the two MHA cells there is no fold and the three distances coincide); the per-head partials (fp32 ``dk_part``, bf16
``dv_part``) on their live rows under the row recipe passed on every GQA cell.  The seeded ``dw_qkvg`` cells above 1.0 (6 of the 13 cells
with the weight gradient, 1.157-2.065x) are near-amax ``dqkvg_t8`` code flips, each moving one ``dW_qkvg`` row by ``flip x deq(h_t)[:, t]``: at most
20 of 8192 rows outside against row budgets of 13.1-103, every one a row a flip touched whose pre-cast slab column sits inside
its band's bound (the bands at 0.09-0.25 of it) -- judged in the row-budgeted form with that attribution asserted
(``_assert_seeded_dw_qkvg_row_budgeted``); ``dh / dw_o`` under the per-cell bound (0.774 / 0.532), ``dW_norm`` at 0.186; the
SECOND dataset's two tables follow the (M) table below::

    cell                             dO    B1    B7    B8   dQ/dK/dV bf16 form   dK rms   dV rms once / fold   bands dq_pre/dg/dk_pre  t8 flips  og8  dh    dw_qkvg dw_o   dWq_n dWk_n  seeded rows outside dh / dw_qkvg / dw_o (budget)
    s256_causal_b1-norm              0.176 0.219 0.191 0.148 0.000/0.029/0.000    1.3e-05  0.0028  / 0        0.121/0.176/0.115       13407     0    0.353 1.157   0.151  0.118 0.139  0/256 (13.1) / 7/5120 (13.1) / 0/512 (1.31)
    s256_causal_b1-rope_only         0.176 0.200 0.232 0.198 0.000/0.001/0.000    2.4e-07  0.0028  / 0        0.088/0.176/0.089       8364      0    0.238 0.467   0.134  -     -      0/256 (13.1) / 0/5120 (13.1) / 0/512 (1.31)
    s512_causal_b2-norm              0.176 0.170 0.178 0.176 0.034/0.148/0.047    4.7e-05  0.0027  / 5.2e-05  0.121/0.187/0.135       54111     0    0.501 1.792   0.126  0.121 0.129  0/1024 (52.4) / 2/5120 (52.4) / 0/512 (5.24)
    s512_causal_b2-rope_only         0.176 0.201 0.188 0.183 0.020/0.032/0.000    8.7e-06  0.0027  / 0        0.088/0.187/0.111       34157     0    0.204 0.537   0.142  -     -      0/1024 (52.4) / 0/5120 (52.4) / 0/512 (5.24)
    s992_causal_b1-norm              0.176 0.211 0.209 0.221 0.034/0.030/0.083    4.3e-05  0.0026  / 0.00011  0.121/0.176/0.134       52629     0    0.774 1.882   0.147  0.140 0.126  0/992 (50.8) / 5/5120 (50.8) / 0/512 (5.08)
    s1008_causal_b2-norm             0.182 0.175 0.182 0.220 0.045/0.156/0.096    0.0001   0.0027  / 6.2e-05  0.130/0.165/0.153       106803    63   0.431 1.161   0.224  0.120 0.186  0/2016 (103) / 2/5120 (103) / 0/512 (10.3)
    s256_dense_b1-norm               0.176 0.197 0.187 0.212 0.100/0.151/0.000    5.2e-05  0.0027  / 0        0.133/0.230/0.145       13415     6    0.580 0.771   0.459  0.139 0.156  0/256 (13.1) / 0/5120 (13.1) / 0/512 (1.31)
    s256_dense_b1-rope_only          0.176 0.167 0.197 0.178 0.203/0.172/0.000    8e-05    0.0027  / 0        0.136/0.197/0.126       8654      0    0.427 0.229   0.124  -     -      0/256 (13.1) / 0/5120 (13.1) / 0/512 (1.31)
    s1024_dense_b1_mha-norm          0.164 0.182 0.172 0.181 0.231/0.279/0.234    0.00013  5.9e-05 / 5.9e-05  0.139/0.186/0.141       69041     0    0.516 0.745   0.133  0.114 0.140  0/1024 (83.9) / 0/8192 (83.9) / 0/512 (5.24)
    s512_dense_b2-rope_only          0.176 0.204 0.180 0.209 0.272/0.212/0.220    8.9e-05  0.0027  / 5.8e-05  0.123/0.252/0.140       33930     41   0.415 0.190   0.168  -     -      0/1024 (52.4) / 0/5120 (52.4) / 0/512 (5.24)
    s512_causal_b1_mha-norm          0.164 0.190 0.201 0.152 0.061/0.124/0.094    4e-05    0.00011 / 0.00011  0.122/0.231/0.146       34038     19   0.301 2.065   0.532  0.105 0.122  0/512 (41.9) / 20/8192 (41.9) / 0/512 (2.62)
    s256_causal_b2_rope-rope_only    0.176 0.206 0.211 0.170 0.000/0.001/0.000    1.7e-07  0.0027  / 0        0.123/0.176/0.095       16791     0    0.198 0.657   0.141  -     -      0/512 (26.2) / 0/5120 (26.2) / 0/512 (2.62)
    s512_causal_b2_delayed-norm      0.176 0.170 0.178 0.176 0.034/0.148/0.047    4.7e-05  0.0027  / 5.2e-05  0.121/0.187/0.135       54111     0    0.501 1.792   0.126  0.121 0.129  0/1024 (52.4) / 2/5120 (52.4) / 0/512 (5.24)
    s1000_causal_b1_dgrad_only-norm  0.176 0.203 -     0.225 0.034/0.053/0.083    4.1e-05  0.0027  / 0.0001   0.121/0.176/0.122       -         0    0.379 -       0.140  0.147 0.132  0/1000 (51.2) / - / 0/512 (5.12)

(M) end-to-end in the row-budget form (``test_mxfp8_end_to_end_modelled_is_row_budgeted`` on every produced bf16 output of every cell,
against the FOLD-MODELLED oracle): cos and rows outside the bf16 bound / rows (budget ``1e-5 x rows x keys``) per output -- INSIDE the
budget on every cell (cos >= 0.999991); the once-rounded (M) oracle (informational) is over it on 5 outputs, every one ``dw_qkvg``
or ``dh`` under GQA (s256_causal_b1-norm dw_qkvg 37/13.1, s256_causal_b1-rope_only dw_qkvg 29/13.1, s256_causal_b2_rope-rope_only
dw_qkvg 38/26.2, s256_dense_b1-rope_only dh 52/13.1, s512_dense_b2-rope_only dh 145/52.4) -- the dV fold's bf16 partials, as the
module predicted; the unquantized-gradient (U) oracle is over on every cell (min cos 0.9983).  The
margins are a property of ONE dataset: the inputs and dY are device-Philox draws laid out by the part's SM count, so a Rubin part with
another SM count draws different tensors under the same seed -- a failure of this layer there is the dataset moving a cell, not the
kernel: re-measure before touching the form::

    cell                             (M) dh: cos  rows out/rows (budget)  (M) dw_qkvg: cos  rows out/rows (budget)  (M) dw_o: cos  rows out/rows (budget)  (M once-rounded) dh / dw_qkvg rows out  verdict
    s256_causal_b1-norm              0.999993  0/256 (13.1)     0.999993  7/5120 (13.1)     0.999999  0/512 (1.31)   1/256 / 37/5120        inside
    s256_causal_b1-rope_only         0.999996  0/256 (13.1)     0.999996  0/5120 (13.1)     0.999999  0/512 (1.31)   1/256 / 29/5120        inside
    s512_causal_b2-norm              0.999993  0/1024 (52.4)    0.999993  2/5120 (52.4)     0.999999  0/512 (5.24)   2/1024 / 40/5120       inside
    s512_causal_b2-rope_only         0.999996  0/1024 (52.4)    0.999996  0/5120 (52.4)     0.999999  0/512 (5.24)   0/1024 / 27/5120       inside
    s992_causal_b1-norm              0.999992  0/992 (50.8)     0.999993  5/5120 (50.8)     0.999999  0/512 (5.08)   1/992 / 46/5120        inside
    s1008_causal_b2-norm             0.999992  0/2016 (103)     0.999993  2/5120 (103)      0.999999  0/512 (10.3)   2/2016 / 56/5120       inside
    s256_dense_b1-norm               0.999993  0/256 (13.1)     0.999994  0/5120 (13.1)     0.999999  0/512 (1.31)   2/256 / 11/5120        inside
    s256_dense_b1-rope_only          0.999996  0/256 (13.1)     0.999996  0/5120 (13.1)     0.999999  0/512 (1.31)   52/256 / 1/5120        inside
    s1024_dense_b1_mha-norm          0.999991  0/1024 (83.9)    0.999993  0/8192 (83.9)     0.999999  0/512 (5.24)   0/1024 / 0/8192        inside
    s512_dense_b2-rope_only          0.999996  0/1024 (52.4)    0.999996  0/5120 (52.4)     0.999999  0/512 (5.24)   145/1024 / 0/5120      inside
    s512_causal_b1_mha-norm          0.999992  0/512 (41.9)     0.999992  20/8192 (41.9)    0.999999  0/512 (2.62)   0/512 / 20/8192        inside
    s256_causal_b2_rope-rope_only    0.999996  0/512 (26.2)     0.999996  0/5120 (26.2)     0.999999  0/512 (2.62)   1/512 / 38/5120        inside
    s512_causal_b2_delayed-norm      0.999993  0/1024 (52.4)    0.999993  2/5120 (52.4)     0.999999  0/512 (5.24)   2/1024 / 40/5120       inside
    s1000_causal_b1_dgrad_only-norm  0.999992  0/1000 (51.2)    -         -                 0.999999  0/512 (5.12)   0/1000 / -             inside

The SECOND dataset -- the same suite and tree on a Rubin part with 212 SMs (cc 10.7, SM clock locked at 2376 MHz), one run of the
whole module, 122 passed -- is the re-measure rule above exercised: every cell's numbers moved (the Philox draws follow the SM count)
and every gate held, no bound touched.  Where this dataset sits CLOSER to a bound than the first: the bf16 bound FORM of dQ / dK / dV
against the fold-modelled reference at the dense MHA cell, 0.682 / 0.659 / 0.432 of it (s1024_dense_b1_mha-norm; the first dataset's
worst cells 0.272 / 0.279 / 0.234) -- the closest the ASSERTED form comes to its bound on either dataset, and inside; dO 0.231 (0.182),
B8 0.241 (0.225), the seeded dw_o 0.637 (0.532) and dw_q_norm 0.188 (0.147); dK vs the once-rounded reference rel RMS <= 0.00028
(0.00013) and dV vs the fold-modelled one <= 0.00015 (0.00011).  Where it sits further: B1 0.217 (0.219), B7 0.216 (0.232), the seeded
dh 0.649 (0.774) and dw_k_norm 0.167 (0.186), the seeded dw_qkvg cells above 1.0 (6 of the 13 again, 1.061-1.675x against 1.157-2.065x;
rows outside at most 6 of 8192 against 41.9 and 2 of 5120 against 13.1, every one a flip-touched row), and the (M) row budget (worst
dw_qkvg 2 of 5120 against 13.1, down from 7).  The GEMM-side bounds sit at 0.140-0.241; dV vs the once-rounded reference 0.0026..0.0027
(vs the fold-modelled one exactly 0 on 6 of the 12 GQA cells, and exactly 0 at the S = 512 MHA cell, where the three distances coincide);
``og8`` 0-29 codes off the seeded oracle's (13 of 14 cells non-zero, against 4 cells at 6-63 on the first dataset); the (M) fold-modelled
oracle inside the row budget on every output of every cell (min cos dh 0.999991, dw_qkvg 0.999993, dw_o 0.999999; max |diff| / max |ref|
0.0057 / 0.0154 / 0.0053); the once-rounded (M) oracle over the budget on 6 outputs (dh 38/13.1 at s256_dense_b1-rope_only and 198/52.4
at s512_dense_b2-rope_only -- the same two cells as on the first dataset -- and dw_qkvg 38/13.1, 26/13.1, 14/13.1 and 33/26.2 at four of
the five S = 256 cells: s256_causal_b1 norm and rope_only, s256_dense_b1-norm, s256_causal_b2_rope-rope_only); the (U) oracle over on
every cell (min cos 0.9984); the CUPTI census identical (28 / 27 / 24 / 24 / 43 / 43 / 41 / 38 / 35: the unfused chain over the
per-member dQ of that tree; 25 / 24 / 24 / 24 / 40 / 40 / 38 / 38 / 32 since the row's single-launch block-scale dQ; 15 / 15 / 14 / 14 /
30 / 30 / 29 / 28 / 22 since the fused prologue / dual-axis dO / epilogue launches, every byte unchanged).  The bitwise layer, the
equivariance pin, the determinism cells and the orientation guard held as on the first dataset (122 passed)::

    cell                             dO    B1    B7    B8   dQ/dK/dV bf16 form   dK rms   dV rms once / fold   bands dq_pre/dg/dk_pre  t8 flips  og8  dh    dw_qkvg dw_o   dWq_n dWk_n  seeded rows outside dh / dw_qkvg / dw_o (budget)
    s256_causal_b1-norm              0.140 0.212 0.174 0.161 0.000/0.000/0.000    1.5e-07  0.0026  / 0        0.116/0.175/0.108       13321     1    0.303 1.235   0.143  0.105 0.127  0/256 (13.1) / 2/5120 (13.1) / 0/512 (1.31)
    s256_causal_b1-rope_only         0.140 0.166 0.170 0.173 0.000/0.000/0.000    2.9e-08  0.0026  / 0        0.092/0.175/0.088       8518      3    0.255 0.779   0.124  -     -      0/256 (13.1) / 0/5120 (13.1) / 0/512 (1.31)
    s512_causal_b2-norm              0.231 0.140 0.189 0.241 0.076/0.071/0.050    0.00012  0.0027  / 3.1e-05  0.130/0.204/0.125       53742     20   0.621 1.595   0.176  0.118 0.143  0/1024 (52.4) / 2/5120 (52.4) / 0/512 (5.24)
    s512_causal_b2-rope_only         0.231 0.154 0.216 0.196 0.007/0.022/0.000    8.6e-06  0.0027  / 0        0.083/0.204/0.088       33945     21   0.222 0.559   0.117  -     -      0/1024 (52.4) / 0/5120 (52.4) / 0/512 (5.24)
    s992_causal_b1-norm              0.231 0.203 0.204 0.163 0.038/0.093/0.094    3.9e-05  0.0027  / 8e-05    0.116/0.175/0.137       52066     8    0.432 1.061   0.143  0.115 0.128  0/992 (50.8) / 3/5120 (50.8) / 0/512 (5.08)
    s1008_causal_b2-norm             0.151 0.210 0.170 0.222 0.097/0.125/0.027    7.8e-05  0.0027  / 3.5e-05  0.147/0.183/0.139       106944    10   0.316 0.896   0.140  0.149 0.167  0/2016 (103) / 0/5120 (103) / 0/512 (10.3)
    s256_dense_b1-norm               0.140 0.212 0.197 0.203 0.087/0.002/0.259    2.9e-07  0.0027  / 0.00015  0.109/0.217/0.128       13554     1    0.645 1.465   0.146  0.188 0.146  0/256 (13.1) / 1/5120 (13.1) / 0/512 (1.31)
    s256_dense_b1-rope_only          0.140 0.174 0.214 0.232 0.000/0.000/0.000    1.4e-08  0.0027  / 0        0.146/0.199/0.129       8434      6    0.400 0.242   0.242  -     -      0/256 (13.1) / 0/5120 (13.1) / 0/512 (1.31)
    s1024_dense_b1_mha-norm          0.181 0.170 0.180 0.192 0.682/0.659/0.432    0.00028  0.00011 / 0.00011  0.131/0.250/0.120       69100     18   0.649 0.608   0.181  0.166 0.120  0/1024 (83.9) / 0/8192 (83.9) / 0/512 (5.24)
    s512_dense_b2-rope_only          0.231 0.199 0.208 0.212 0.167/0.171/0.000    8e-05    0.0027  / 0        0.125/0.211/0.140       33973     29   0.412 0.181   0.637  -     -      0/1024 (52.4) / 0/5120 (52.4) / 0/512 (5.24)
    s512_causal_b1_mha-norm          0.181 0.200 0.192 0.213 0.039/0.074/0.000    2.3e-05  0       / 0        0.132/0.162/0.126       34089     0    0.255 1.675   0.140  0.160 0.123  0/512 (41.9) / 6/8192 (41.9) / 0/512 (2.62)
    s256_causal_b2_rope-rope_only    0.140 0.217 0.210 0.157 0.000/0.130/0.000    7.7e-05  0.0026  / 0        0.092/0.177/0.116       17019     6    0.230 0.594   0.153  -     -      0/512 (26.2) / 0/5120 (26.2) / 0/512 (2.62)
    s512_causal_b2_delayed-norm      0.231 0.140 0.189 0.241 0.076/0.071/0.050    0.00012  0.0027  / 3.1e-05  0.130/0.204/0.125       53742     20   0.621 1.595   0.176  0.118 0.143  0/1024 (52.4) / 2/5120 (52.4) / 0/512 (5.24)
    s1000_causal_b1_dgrad_only-norm  0.231 0.200 -     0.206 0.038/0.155/0.029    7.6e-05  0.0027  / 5e-05    0.116/0.175/0.107       -         9    0.276 -       0.141  0.114 0.127  0/1000 (51.2) / - / 0/512 (5.12)

and end to end on that dataset in the (M) row-budget form -- inside the budget on every cell::

    cell                             (M) dh: cos  rows out/rows (budget)  (M) dw_qkvg: cos  rows out/rows (budget)  (M) dw_o: cos  rows out/rows (budget)  (M once-rounded) dh / dw_qkvg rows out  verdict
    s256_causal_b1-norm              0.999993  0/256 (13.1)     0.999994  2/5120 (13.1)     0.999999  0/512 (1.31)   0/256 / 38/5120        inside
    s256_causal_b1-rope_only         0.999996  0/256 (13.1)     0.999996  0/5120 (13.1)     0.999999  0/512 (1.31)   1/256 / 26/5120        inside
    s512_causal_b2-norm              0.999992  0/1024 (52.4)    0.999993  2/5120 (52.4)     0.999999  0/512 (5.24)   0/1024 / 50/5120       inside
    s512_causal_b2-rope_only         0.999996  0/1024 (52.4)    0.999996  0/5120 (52.4)     0.999999  0/512 (5.24)   0/1024 / 39/5120       inside
    s992_causal_b1-norm              0.999992  0/992 (50.8)     0.999993  3/5120 (50.8)     0.999999  0/512 (5.08)   2/992 / 41/5120        inside
    s1008_causal_b2-norm             0.999993  0/2016 (103)     0.999993  0/5120 (103)      0.999999  0/512 (10.3)   0/2016 / 30/5120       inside
    s256_dense_b1-norm               0.999992  0/256 (13.1)     0.999993  1/5120 (13.1)     0.999999  0/512 (1.31)   4/256 / 14/5120        inside
    s256_dense_b1-rope_only          0.999996  0/256 (13.1)     0.999996  0/5120 (13.1)     0.999999  0/512 (1.31)   38/256 / 3/5120        inside
    s1024_dense_b1_mha-norm          0.999991  0/1024 (83.9)    0.999993  0/8192 (83.9)     0.999999  0/512 (5.24)   0/1024 / 0/8192        inside
    s512_dense_b2-rope_only          0.999996  0/1024 (52.4)    0.999996  0/5120 (52.4)     0.999999  0/512 (5.24)   198/1024 / 0/5120      inside
    s512_causal_b1_mha-norm          0.999993  0/512 (41.9)     0.999994  6/8192 (41.9)     0.999999  0/512 (2.62)   0/512 / 6/8192         inside
    s256_causal_b2_rope-rope_only    0.999996  0/512 (26.2)     0.999996  0/5120 (26.2)     0.999999  0/512 (2.62)   2/512 / 33/5120        inside
    s512_causal_b2_delayed-norm      0.999992  0/1024 (52.4)    0.999993  2/5120 (52.4)     0.999999  0/512 (5.24)   0/1024 / 50/5120       inside
    s1000_causal_b1_dgrad_only-norm  0.999993  0/1000 (51.2)    -         -                 0.999999  0/512 (5.12)   0/1000 / -             inside
"""

import dataclasses
import gc
import inspect
import math
import os
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import numpy as np
import pytest
import torch

from cudnn.frost.buffers import cutedsl_requirement_error

requirement_error = cutedsl_requirement_error("Gated attention block MXFP8 backward tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

pytestmark = pytest.mark.L0

from cudnn.gated_attention_block import (
    GatedAttentionBlockBwd,
    GatedAttentionBlockFwd,
    GatedAttentionBlockGeometry,
    gated_attention_block_backward,
)  # noqa: E402
from cudnn.gated_attention_block import api_bwd as _api_bwd  # noqa: E402
from cudnn.gated_attention_block.api import MxQuantSpec, _cols, _view  # noqa: E402
from cudnn.gated_attention_block.kernels import quantize as _quantize  # noqa: E402
from cudnn.gated_attention_block.kernels.proj_gemm import sf_blob_bytes  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gated_block_reference import (  # noqa: E402
    FP8_E4M3_MAX,
    RefGeometry,
    gated_attention_block_mxfp8_bwd_reference,
    make_inputs,
    mx_dequant_rowwise_2d,
    mx_quantize_rowwise_2d,
    mx_swizzle_sf_rowwise_padded,
    mx_unswizzle_sf_rowwise,
    mxfp8_calibrated_scale_o,
    quant_e4m3,
    quantize_block_inputs_mxfp8,
)
from gated_block_stream_probe import park_the_default_stream  # noqa: E402
from test_block_backward import (  # noqa: E402
    _ATOL_FRAC,
    _COMMON,
    _GEOM_397B,
    _KNOBS,
    _MXFP8_LIVE_CONSTS,
    _MXFP8_LIVE_SLOTS,
    _MXFP8_STAGES,
    _RTOL,
    _alloc_grads,
    _assert_dw_norm_close,
    _assert_grad_close,
    _cos,
    _declare_bwd_mxfp8,
    _make_dy,
    _mx_artifacts,
)
from test_block_backward_fp8 import (  # noqa: E402
    _assert_e4m3_bitwise,
    _assert_og8_bitwise_the_kernels_o_gated,
    _dev_scalar,
    _f32,
    _f32_mul,
    _gemm_bound,
    _grad_scale,
    _print_end_to_end,
    _qkvg_band,
    _report_close,
    _report_stage_difference,
    _row_budget,
    _rows_outside,
    _rows_outside_mask,
    _test_python_root,
)
from test_block_training_forward import _alloc_saved, _dense_tail_declined, _run_training_quant  # noqa: E402

_SM107 = (10, 7)
_E4M3 = torch.float8_e4m3fn


def _cc():
    return tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None


# The REGISTERED marker of cutedsl/conftest.py (the skip is applied at collection).
requires_rubin = pytest.mark.requires_rubin
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

# The appended keyword-only parameters of the quantized backward (append-only, defaulted, LAST): the fp8 tail, then the MXFP8 artifacts.
_QUANT_INIT_KWARGS = ("quant", "grad_scaling")
_QUANT_EXECUTE_KWARGS = ("scale_dp", "scale_dy", "scale_do", "scale_dqkvg")
_MX_EXECUTE_KWARGS = ("h_t", "h_t_sf", "w_qkvg_t", "w_qkvg_t_sf")
_FP4_EXECUTE_KWARGS = ("w_o_t", "w_o_t_sf")  # the fp4 weight modes' appended pair (test_block_backward_fp4.py), after the MXFP8 artifacts
_ARTIFACTS = _MX_EXECUTE_KWARGS


def test_the_mxfp8_surface_is_an_appended_keyword_only_tail():
    """Host, no GPU: the MXFP8 backward's four artifacts follow the fp8 tail on ``execute`` and on the convenience wrapper, and the fp4
    weight modes' two (``w_o_t`` / ``w_o_t_sf``) are the LAST parameters -- keyword-only and defaulted, in the declared order (public
    signatures evolve append-only); ``__init__``'s tail is the fp8 one (the type of ``quant`` widened, nothing appended)."""
    for fn, names in (
        (GatedAttentionBlockBwd.__init__, _QUANT_INIT_KWARGS),
        (GatedAttentionBlockBwd.execute, _QUANT_EXECUTE_KWARGS + _MX_EXECUTE_KWARGS + _FP4_EXECUTE_KWARGS),
        (gated_attention_block_backward, _QUANT_INIT_KWARGS + _QUANT_EXECUTE_KWARGS + _MX_EXECUTE_KWARGS + _FP4_EXECUTE_KWARGS),
    ):
        tail = list(inspect.signature(fn).parameters.values())[-len(names) :]
        assert [p.name for p in tail] == list(names), (fn.__qualname__, [p.name for p in tail])
        assert all(p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is not inspect.Parameter.empty for p in tail), fn.__qualname__


def _api_const(name: str):
    """A module constant of ``api_bwd`` read at CALL time, never re-literalled here."""
    return getattr(_api_bwd, name)


# ---------------------------------------------------------------------------
# The accept matrix -- named once
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Cell:
    shape: str
    s: int
    causal: bool
    b: int
    h_kv: int
    qk_norm: bool
    need_dw_o: bool = True
    need_dw_qkvg: bool = True
    grad_scaling: str = "current"
    note: str = ""
    # geometry fields that replace ``_COMMON``'s, as a tuple of pairs (the frozen cell stays hashable): the 397B census cell's
    # ``d_model`` / ``h_q``; every matrix cell keeps the suite geometry
    geom_override: tuple = ()

    @property
    def id(self) -> str:
        return f"{self.shape}-{'norm' if self.qk_norm else 'rope_only'}"

    @property
    def geom_kw(self) -> dict:
        return {**_COMMON, **dict(self.geom_override), "h_kv": self.h_kv, "qk_norm": self.qk_norm, "is_causal": self.causal}

    @property
    def bwd_kw(self) -> dict:
        kw = {}
        if not self.need_dw_o:
            kw["need_dw_o"] = False
        if not self.need_dw_qkvg:
            kw["need_dw_qkvg"] = False
        return kw

    @property
    def group(self) -> int:
        return self.geom_kw["h_q"] // self.h_kv

    @property
    def t(self) -> int:
        return self.b * self.s


def _both(shape, s, causal, b, h_kv, **kw):
    return [_Cell(shape, s, causal, b, h_kv, True, **kw), _Cell(shape, s, causal, b, h_kv, False, **kw)]


_CELLS = (
    _both("s256_causal_b1", 256, True, 1, 2)
    + _both("s512_causal_b2", 512, True, 2, 2, note="THE bitwise / graph / CUPTI cell")
    + [_Cell("s992_causal_b1", 992, True, 1, 2, True, note="992 = 31 x 32; S % 128 = 96: q- AND kv-padded (the row's SF re-stagings on both sides)")]
    + [_Cell("s1008_causal_b2", 1008, True, 2, 2, True, note="T = 2016 = 63 x 32 (S % 32 = 16): the rule binds B*S; padded both sides at B = 2")]
    + _both("s256_dense_b1", 256, False, 1, 2)
    + [_Cell("s1024_dense_b1_mha", 1024, False, 1, 8, True, note="MHA: no dkv_reduce, one dQ launch per chunk")]
    + [_Cell("s512_dense_b2", 512, False, 2, 2, False)]
    + [_Cell("s512_causal_b1_mha", 512, True, 1, 8, True, note="the block-scale arm's MHA dQ (one launch), no fold")]
    + [_Cell("s256_causal_b2_rope", 256, True, 2, 2, False, note="rope_only x causal (no reduce launch), B = 2 at n_kv = 1")]
    + [
        _Cell(
            "s512_causal_b2_delayed",
            512,
            True,
            2,
            2,
            True,
            grad_scaling="delayed",
            note="the bitwise cell's geometry under the delayed recipe fed the current run's scale_dy",
        )
    ]
    + [
        _Cell(
            "s1000_causal_b1_dgrad_only",
            1000,
            True,
            1,
            2,
            True,
            need_dw_qkvg=False,
            note="need_dw_qkvg=False: T = 1000 is SERVED (the dgrad contracts over N; no transposed quantize, no B7), h_t refused; dW_o stays (per-tensor, any T)",
        )
    ]
)
# Launch-count-only cells: NOT in the matrix (no GEMM / SDPA / seeded / end-to-end layer) -- shapes whose launch-count arm no matrix
# cell reaches, run by test_mxfp8_launch_count_is_honest and the bitwise + finiteness layer.
_LAUNCH_ONLY_CELLS = [
    _Cell("s992_causal_b1_mha", 992, True, 1, 8, True, note="padded x MHA (+8 q-side, +6 kv-side, no fold): launch count + the bitwise layer"),
    _Cell("s384_causal_b1", 384, True, 1, 2, True, note="kv-side pads ONLY (S % 128 == 0, S % 256 != 0): +7 without the +8; launch count + the bitwise layer"),
    _Cell(
        "s512_causal_b1_397b",
        512,
        True,
        1,
        2,
        True,
        geom_override=tuple(sorted({k: v for k, v in _GEOM_397B.items() if k != "h_kv"}.items())),
        note="the 397B geometry (d_model 4096, 32/2 heads: g = 16, c = 1 at S = 512): the census the docs quote for it + the bitwise layer",
    ),
]
_BY_ID = {c.id: c for c in _CELLS + _LAUNCH_ONLY_CELLS}
assert len(_CELLS) == 14 and len(_BY_ID) == len(_CELLS) + len(_LAUNCH_ONLY_CELLS), "the cell ids must be unique: 11 matrix rows, three in both qk_norm arms"
assert all(c.t % 32 == 0 for c in _CELLS if c.need_dw_qkvg), "every accept cell with the projection weight gradient has B*S % 32 == 0"
_MATRIX = pytest.mark.parametrize("cell", _CELLS, ids=[c.id for c in _CELLS])
_BITWISE_CELL = _BY_ID["s512_causal_b2-norm"]
_KNOB_SETS = pytest.mark.parametrize("knobs", list(_KNOBS.values()), ids=list(_KNOBS))
_GQA_CELLS = [c for c in _CELLS if c.group > 1]
# The (M) row budget: asserted on the FOLD-MODELLED oracle since the first Rubin run recorded its margins (the module docstring's
# table: no output of any cell over its budget, worst 7 of 5120 dw_qkvg rows against 13.1); nothing was widened.
_M_ROW_BUDGET_ASSERTED = True
# The bf16 bound FORM on the SDPA stage's dQ / dK / dV against the FOLD-MODELLED reference: switched on after the first run, as above
# (worst cells 0.272 / 0.279 / 0.234 of the bound).
_STAGE_BF16_FORM_ASSERTED = True
# The (M) row budget's cells: every output of every cell (the fold is modelled, so no cell is exempt once asserted).
_M_OUTPUTS = ("dh", "dw_qkvg", "dw_o")


# ---------------------------------------------------------------------------
# The launch-count formula (host-checkable; the CUPTI cell asserts formula == expected == len(kernels))
# ---------------------------------------------------------------------------


def mxfp8_block_launch_table(*, qk_norm: bool, need_dw_o: bool = True, need_dw_qkvg: bool = True, need_dh: bool = True) -> list:
    """The block's OWN launches under ``quant=MxQuantSpec``, in launch order, as ``(label, count, present)`` -- the module
    docstring's table of ``api_bwd`` as data, so the count is derived from it and never re-literalled.  The fp8 chain's shape: the
    fused PROLOGUE always (its init / dY amax / Q-K rebuild-and-quantize / v8 jobs serve every block), ONE dual-axis dO quantize, the
    fused EPILOGUE iff one of its three jobs exists (the dW_norm reduce under ``qk_norm``, the rowwise dQKVG cast under ``need_dh``, the
    transposed one under ``need_dw_qkvg``), and each omitted gradient drops ITS GEMM."""
    return [
        ("PROLOGUE: init (slots[:] = 0, the plan-time constants; no descale_dp) | dY amax partials | Q / K rebuild + q8 / q_T8 / k8 / k_T8 | v8", 1, True),
        ("quantize dY (reduces the partials, publishes amax_dy / scale_dy / descale_dy / alpha_b1 / alpha_b2)", 1, True),
        ("B2 out_proj dgrad (e4m3, K64)", 1, True),
        ("B3 sigmoid_gate_bwd (fp8 arm: dO, dG, og8, delta; no amax partials)", 1, True),
        ("quantize dO dual-axis (MXFP8: rowwise SDPA layout + columnwise D-plane-major from one read)", 1, True),
        ("B1 out_proj wgrad (e4m3, K64)", 1, need_dw_o),
        ("B5+B6 qk_norm_rope_bwd (no amax fold)", 1, True),
        ("EPILOGUE: dW_norm reduce | dqkvg dual-axis canonical cast (rowwise + transposed)", 1, qk_norm or need_dh or need_dw_qkvg),
        ("B7 qkv_gate wgrad (block-scale)", 1, need_dw_qkvg),
        ("B8 qkv_gate dgrad (block-scale)", 1, need_dh),
    ]


def mxfp8_row_launches(*, group: int, chunks: int, dq_launches: int, q_padded: bool, kv_padded: bool, zero_ws: bool) -> int:
    """The MXFP8 SDPA row's launches under an EXTERNAL delta on its block-scaled dS chain: ``fill_i32`` (1) + per head chunk
    ``main + dK + dq_launches x dQ`` + ``dkv_reduce`` under GQA + the staging: ``+8`` at ``S % 128 != 0`` (the q / dO / lse pads, the
    dO_T pad, the ``sf_q / sf_do / sf_do_T`` re-stagings, the block-scaled chain's ``sf_q_T`` re-staging), ``+7`` GQA / ``+6`` MHA at
    ``S % 256 != 0`` (the k / v pads, the ``sf_k / sf_v / sf_k_T`` re-stagings, two fold copy-outs under GQA, one under MHA) and
    ``+4`` under a dS zero-fill (the first payload, the second, the two atom tensors).  NO ``dot_do_o`` (the caller's delta)."""
    fold = 1 if group > 1 else 0
    pads = (8 if q_padded else 0) + (((7 if group > 1 else 6)) if kv_padded else 0)
    return 1 + chunks * (2 + dq_launches) + fold + pads + (4 if zero_ws else 0)


def _padded(s: int) -> tuple:
    """The adapter's staging facts from the shape alone: q side at 128-row tiles, kv side at 256-row blocks."""
    return s % 128 != 0, s % 256 != 0


def shipped_dq_launches(group: int) -> int:
    """``q``: the MXFP8 row's dQ GEMM launches per head chunk as SHIPPED -- ONE at the GQA group under ``api_dsl_sm107.DQ_SINGLE_LAUNCH``
    (the block-scale arm's dQ record takes ``b_head_group = group`` exactly like the plain renderings: the template indexes B and
    its scale-factor descriptor by ``h // group``), ``group`` on the per-member twin, 1 at MHA either way -- read off the row's
    module constant, never typed."""
    from cudnn.sdpa.bwd.api_dsl_sm107 import DQ_SINGLE_LAUNCH

    return 1 if (DQ_SINGLE_LAUNCH and group > 1) else group


def mxfp8_expected_launches(cell: _Cell, *, chunks: int = 1, zero_ws: bool = False, need_dh: bool = True) -> int:
    """CUPTI kernel records of ONE execute of the MXFP8 backward at ``cell`` = the block's table by the cell's needs + the row's
    launches with ``q`` = the row's shipped dQ launches per chunk (``shipped_dq_launches``: 1 under GQA on the block-scale arm's
    single-launch dQ) and the pads from ``S`` -- COMPUTED, never typed."""
    block = sum(
        n
        for _l, n, present in mxfp8_block_launch_table(qk_norm=cell.qk_norm, need_dw_o=cell.need_dw_o, need_dw_qkvg=cell.need_dw_qkvg, need_dh=need_dh)
        if present
    )
    q_pad, kv_pad = _padded(cell.s)
    return block + mxfp8_row_launches(
        group=cell.group, chunks=chunks, dq_launches=shipped_dq_launches(cell.group), q_padded=q_pad, kv_padded=kv_pad, zero_ws=zero_ws
    )


def mxfp8_launch_formula_from_facts(blk) -> int:
    """The formula recomputed from the ADAPTER's own facts (the head chunks, the dQ rendering's ``b_head_group``, the padding, the
    zero-fill, the block-scaled dS policy, the external delta) and the block's declaration -- so a change on either side is visible."""
    g, st = blk.geom, blk._sdpa
    impl = st._impl
    assert impl.external_delta is True, "the MXFP8 backward ALWAYS hands the row the gate backward's delta"
    assert impl._ds_block_scaled, "the row's block-scaled dS chain is the one the formula counts (two payloads + two atom tensors)"
    grp = g.h_q // g.h_kv
    chunks = -(-blk.batch // impl._b_chunk) * -(-g.h_q // impl._qh_chunk)
    assert chunks == st.head_chunks() and st.dq_launches_per_chunk() == grp // impl._dq_b_head_group
    block = sum(
        n
        for _l, n, present in mxfp8_block_launch_table(qk_norm=g.qk_norm, need_dw_o=blk.need_dw_o, need_dw_qkvg=blk.need_dw_qkvg, need_dh=blk.need_dh)
        if present
    )
    return block + mxfp8_row_launches(
        group=grp,
        chunks=chunks,
        dq_launches=st.dq_launches_per_chunk(),
        q_padded=bool(impl._q_padded),
        kv_padded=bool(impl._kv_padded),
        zero_ws=bool(impl._zero_ws),
    )


def test_mxfp8_launch_formula_reproduces_the_derivations():
    """Host, no GPU: the COMPUTED expectation reproduces the module docstring's derivations under the fused block table and the row's
    single-launch dQ (``shipped_dq_launches``: q = 1 on the block-scale arm too) -- 15 at the bitwise cell (norm, GQA 8/2: the block's
    10 + the row's 1 + 3 + 1), 15 rope_only (the epilogue stays for the cast), 14 at the two MHA cells (q = 1 either way, no dkv_reduce),
    30 at the two padded GQA cells with weight gradients (+8 q-side, +7 kv-side), 29 at the padded dgrad-only cell (9 block launches: no
    B7, the epilogue keeps its rowwise half), 28 at the padded MHA cell, 22 at the kv-side-only padded cell; the block's table sums to
    10 / 10 / 9 / 9 / 8 / 8 as the needs drop, and to 7 for a block wanting dW_o alone (no epilogue, no B7 / B8); a second head chunk
    adds 3 (main + dK + the one dQ) at every group -- the per-member form added 2 + group.  The unfused chain's table summed to 20 / 19 /
    18 / 19 / 17 / 16 (28 / 27 / 24 / 24 / 43 / 43 / 41 / 38 / 35 on the cells over the per-member dQ; 25 / 24 / 24 / 24 / 40 / 40 / 38 /
    38 / 32 over the single-launch dQ); the fused table over the per-member dQ 18 / 18 / 14 / 14 / 33 / 33 / 32 / 28 / 25."""
    from cudnn.sdpa.bwd.api_dsl_sm107 import DQ_SINGLE_LAUNCH

    assert DQ_SINGLE_LAUNCH and shipped_dq_launches(4) == 1 and shipped_dq_launches(16) == 1 and shipped_dq_launches(1) == 1
    want = {
        "s512_causal_b2-norm": 15,
        "s512_causal_b2-rope_only": 15,
        "s1024_dense_b1_mha-norm": 14,
        "s512_causal_b1_mha-norm": 14,
        "s992_causal_b1-norm": 30,
        "s1008_causal_b2-norm": 30,
        "s1000_causal_b1_dgrad_only-norm": 29,
        "s992_causal_b1_mha-norm": 28,
        "s384_causal_b1-norm": 22,
        "s512_causal_b1_397b-norm": 15,
        "s256_causal_b1-norm": 15,
        "s512_causal_b2_delayed-norm": 15,
    }
    for cell_id, n in want.items():
        assert mxfp8_expected_launches(_BY_ID[cell_id]) == n, (cell_id, mxfp8_expected_launches(_BY_ID[cell_id]), n)
    tbl = lambda **kw: sum(n for _l, n, p in mxfp8_block_launch_table(**kw) if p)  # noqa: E731
    assert tbl(qk_norm=True) == 10 and tbl(qk_norm=False) == 10 and tbl(qk_norm=True, need_dw_qkvg=False) == 9 and tbl(qk_norm=True, need_dw_o=False) == 9
    assert tbl(qk_norm=True, need_dw_qkvg=False, need_dw_o=False) == 8 and tbl(qk_norm=False, need_dw_qkvg=False, need_dw_o=False) == 8
    assert tbl(qk_norm=False, need_dw_qkvg=False, need_dh=False) == 7 and tbl(qk_norm=True, need_dw_qkvg=False, need_dh=False) == 8  # dW_o alone: no epilogue
    assert [lab for lab, _n, p in mxfp8_block_launch_table(qk_norm=False, need_dw_qkvg=False, need_dh=False) if p and lab.startswith("EPILOGUE")] == []
    assert mxfp8_row_launches(group=4, chunks=1, dq_launches=4, q_padded=False, kv_padded=False, zero_ws=False) == 8  # the per-member twin's row
    assert mxfp8_row_launches(group=4, chunks=1, dq_launches=1, q_padded=False, kv_padded=False, zero_ws=False) == 5  # the shipped row
    assert mxfp8_row_launches(group=1, chunks=1, dq_launches=1, q_padded=False, kv_padded=False, zero_ws=False) == 4
    assert mxfp8_expected_launches(_BITWISE_CELL, chunks=2) == 18
    big = _BY_ID["s512_causal_b1_397b-norm"]  # the 397B census cell: a group of 16 (h_q 32 / h_kv 2); the group alone drives the row term
    assert big.group == 16 and big.geom_kw["d_model"] == 4096 and big.geom_kw["h_q"] == 32 and big.t % 32 == 0, big
    assert mxfp8_row_launches(group=16, chunks=1, dq_launches=16, q_padded=False, kv_padded=False, zero_ws=False) == 20  # the per-member twin
    assert mxfp8_row_launches(group=16, chunks=2, dq_launches=16, q_padded=False, kv_padded=False, zero_ws=False) == 38
    assert mxfp8_row_launches(group=16, chunks=1, dq_launches=1, q_padded=False, kv_padded=False, zero_ws=False) == 5  # the shipped row at 397B: 10 + 5 = 15
    assert mxfp8_row_launches(group=16, chunks=2, dq_launches=1, q_padded=False, kv_padded=False, zero_ws=False) == 8
    del big


# ---------------------------------------------------------------------------
# Building an MXFP8 backward: the MXFP8 training forward supplies the record exactly as a user would
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_tf32():
    """An fp32 torch reference is a TF32 reference on Blackwell+ unless pinned (``allow_tf32``); the oracles are fp64 but the pin is printed anyway."""
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    print(f"\nallow_tf32={torch.backends.cuda.matmul.allow_tf32}")
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


def _declare_mx_bwd(dy, saved, inp, geom, **bwd_kw) -> GatedAttentionBlockBwd:
    """``GatedAttentionBlockBwd(...)`` over an MXFP8 record (the quantized inputs dict's weights, norm weights and cos / sin)."""
    return GatedAttentionBlockBwd(dy, saved, inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], geom, **bwd_kw)


def _artifacts_of(inp16: dict, geom) -> dict:
    """The four caller artifacts through the test tree's builder (``quantize_block_inputs_mxfp8(backward=True)``, the convergence
    harness's contract), falling back to the module-level builder of ``test_block_backward`` when the appended arm is absent."""
    t = inp16["h"].shape[0] * inp16["h"].shape[1]
    try:
        mx, _desc = quantize_block_inputs_mxfp8(inp16, backward=True)
    except NotImplementedError:
        return _mx_artifacts(inp16, geom, h_t=t % 32 == 0)
    # the builder serves h_t / h_t_sf at whole 32-token blocks only (no weight gradient at a ragged T): absent keys read as None
    return {k: mx.get(k) for k in _ARTIFACTS}


def _execute_mx(blk, inp, saved, dy, grads, ws, art, *, scale_dy=None, current_stream=None, **over):
    """One ``execute`` of an MXFP8 block: the record, the weights, the gradients, the workspace and the four artifacts (``over``
    replaces an artifact: the wrong-orientation pin)."""
    kw = {k: v for k, v in dict(art, **over).items() if k in _ARTIFACTS}
    if scale_dy is not None:
        kw["scale_dy"] = scale_dy
    blk.execute(
        dy,
        saved,
        inp["w_qkvg"],
        inp["w_q_norm"],
        inp["w_k_norm"],
        inp["cos"],
        inp["sin"],
        inp["w_o"],
        workspace=ws,
        current_stream=current_stream,
        **grads,
        **kw,
    )


_MEMO: dict = {}


def _backward_mxfp8(geom_kw, batch, seq_len, *, grad_scaling="current", memo=True, scale_dy=None, fwd=None, **bwd_kw):
    """Run the MXFP8 training forward (the record, KEPT with its workspace alive: its payload bytes are the reference the recomputed
    ``q8 / k8`` are pinned against), build the caller's artifacts from the SAME bf16 inputs, declare / compile / run the MXFP8 backward
    and read its scalar block back.  ``scale_dy``: the caller's float under ``grad_scaling="delayed"``.  ``fwd``: a forward run to reuse
    (its record and inputs) instead of the memoised default dataset.  Memoised per declaration so the contract tests reuse one block."""
    key = (tuple(sorted(geom_kw.items())), batch, seq_len, grad_scaling, scale_dy, tuple(sorted(bwd_kw.items())), id(fwd) if fwd is not None else None)
    if memo and key in _MEMO:
        return _MEMO[key]
    r = _run_training_quant(geom_kw, batch, seq_len, "mxfp8") if fwd is None else fwd
    inp, saved = r.inp, r.saved
    art = _artifacts_of(r.inp16, r.geom)
    dy = _make_dy(r.out)
    blk = _declare_mx_bwd(dy, saved, inp, r.geom, quant=r.spec, grad_scaling=grad_scaling, **bwd_kw)
    blk.check_support()
    blk.compile()
    ws = torch.empty(blk.get_workspace_size(), dtype=torch.uint8, device="cuda")
    grads = _alloc_grads(blk)
    scale_dy_t = _dev_scalar(scale_dy) if scale_dy is not None else None
    _execute_mx(blk, inp, saved, dy, grads, ws, art, scale_dy=scale_dy_t)
    torch.cuda.synchronize()
    res = SimpleNamespace(
        blk=blk,
        fwd=r,
        inp=inp,
        inp16=r.inp16,
        spec=r.spec,
        saved=saved,
        dy=dy,
        out=r.out,
        ws=ws,
        grads=grads,
        art=art,
        scale_dy_t=scale_dy_t,
        scalars={k: float(v.item()) for k, v in blk.quant_scalars(ws).items()},
        geom=r.geom,
        geom_kw=geom_kw,
        batch=batch,
        seq_len=seq_len,
        grad_scaling=grad_scaling,
    )
    if memo:
        _MEMO[key] = res
    return res


def _cell_backward(cell: _Cell, **extra):
    if cell.grad_scaling == "delayed":
        # the delayed cell replays the current run's READ-BACK scale_dy (the one per-tensor recipe of this pipeline)
        cur = _backward_mxfp8(cell.geom_kw, cell.b, cell.s, **cell.bwd_kw, **extra)
        return _backward_mxfp8(cell.geom_kw, cell.b, cell.s, grad_scaling="delayed", scale_dy=cur.scalars["scale_dy"], **cell.bwd_kw, **extra)
    return _backward_mxfp8(cell.geom_kw, cell.b, cell.s, **cell.bwd_kw, **extra)


def _twin_mx(res, *, poison=0xFF, grad_scaling=None, scale_dy=None, art=None, **bwd_kw):
    """A second MXFP8 block over the SAME record / dy / inputs / artifacts as ``res`` (different knobs or recipe) and the SAME needs
    (a dgrad-only block's twin is dgrad-only: at a ragged T the weight gradient is a typed decline, so a twin that asked for it
    would never compile), compiled and run once into a poisoned workspace and NaN-filled gradients; returns ``(blk, ws, grads)`` --
    the bitwise comparand of ``res``.  ``bwd_kw`` (knobs, or a need to flip on purpose) overrides."""
    inp = res.inp
    need = dict(need_dh=res.blk.need_dh, need_dw_qkvg=res.blk.need_dw_qkvg, need_dw_o=res.blk.need_dw_o)
    need.update(bwd_kw)
    blk = _declare_mx_bwd(res.dy, res.saved, inp, res.geom, quant=res.spec, grad_scaling=grad_scaling or res.grad_scaling, **need)
    blk.check_support()
    blk.compile()
    ws = torch.empty(blk.get_workspace_size(), dtype=torch.uint8, device="cuda").fill_(poison)
    grads = _alloc_grads(blk, fill=float("nan"))
    sdy = _dev_scalar(scale_dy) if scale_dy is not None else res.scale_dy_t
    _execute_mx(blk, inp, res.saved, res.dy, grads, ws, res.art if art is None else art, scale_dy=sdy)
    torch.cuda.synchronize()
    return blk, ws, grads


# ---------------------------------------------------------------------------
# Reading the block's intermediates back
# ---------------------------------------------------------------------------


def _sf_names_of(blk) -> tuple:
    return tuple(blk._sdpa.sf_roles())


def _rebuild_qk(res) -> tuple:
    """The bf16 post-norm / post-RoPE Q / K the fused prologue's MX epilogue quantizes OUT OF REGISTERS, materialised by the forward's own
    rebuild stage (``_QkNormRope``: the bf16 backward's kernel, the one the unfused MXFP8 chain ran into its ``recompute`` regions) over
    the record's slab bands into TEST-OWNED buffers -- the fused chain carves no bf16 rebuild region, and these values (the kernel rounds
    to bf16 first) are the bitwise reference its ``q8 / q_T8 / k8 / k_T8`` are quantized from.  The norm weights and the cos / sin tables
    come from ``res.inp`` (the builder's result) or ``res.fwd.inp`` (a twin namespace over another block's workspace, which carries the
    forward run instead).  Memoised on ``res``."""
    if getattr(res, "_rebuilt_qk", None) is None:
        from cudnn.gated_attention_block.api import _QkNormRope

        inp = getattr(res, "inp", None)
        if inp is None and getattr(res, "fwd", None) is not None:
            inp = res.fwd.inp
        if inp is None:
            raise AttributeError("_rebuild_qk needs res.inp (or res.fwd.inp): the norm weights and the cos / sin tables of the record's forward")
        blk, g = res.blk, res.geom
        b, s, d, act = blk.batch, blk.seq_len, g.d_head, blk.act_dtype
        t = b * s
        proj = res.saved.proj_slab.view(t, g.n_qkvg)
        o_q, _o_g, o_k, _o_v = g.qkvg_offsets
        rq = torch.empty(t, g.h_q, d, dtype=act, device=proj.device)
        rk = torch.empty(t, g.h_kv, d, dtype=act, device=proj.device)
        st = _QkNormRope(g, batch=b, seq_len=s, dtype=act, want_rstd=False)
        st.check_support()
        st.compile()
        st.execute(_cols(proj, o_q, g.h_q, d), _cols(proj, o_k, g.h_kv, d), inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], q_out=rq, k_out=rk)
        torch.cuda.synchronize()
        res._rebuilt_qk = (rq, rk)
    return res._rebuilt_qk


def _slots(res) -> dict:
    """The block's materialised intermediates after execute, read out of the workspace: the e4m3 payloads as e4m3 views, their
    scale-factor blobs as flat uint8 views, the bf16 buffers they were quantized from (the gate backward's dO, the slab's V band, the
    dqkvg slab; the bf16 Q / K rebuild as the workspace region when the carve holds one, else materialised by ``_rebuild_qk`` -- the
    fused prologue quantizes it out of registers), the SDPA stage's bf16 outputs."""
    from cudnn.gated_attention_block.api import _sf_slot_bytes

    blk, g = res.blk, res.geom
    b, s = blk.batch, blk.seq_len
    t, d, act = b * s, g.d_head, blk.act_dtype
    lay = blk._layout()
    o_q, o_g, o_k, o_v = g.qkvg_offsets
    if lay.recompute >= 0 and lay.recompute_k >= 0:
        rq, rk = _view(res.ws, lay.recompute, (t, g.h_q, d), act), _view(res.ws, lay.recompute_k, (t, g.h_kv, d), act)
    else:
        rq, rk = _rebuild_qk(res)
    dqkvg = _view(res.ws, lay.dqkvg, (t, g.n_qkvg), act)
    proj = res.saved.proj_slab.view(t, g.n_qkvg)
    sf = {}
    for name, heads in (("sf_q", g.h_q), ("sf_q_T", g.h_q), ("sf_k", g.h_kv), ("sf_k_T", g.h_kv), ("sf_v", g.h_kv), ("sf_do", g.h_q), ("sf_do_T", g.h_q)):
        sf[name] = _view(res.ws, getattr(lay, name), (_sf_slot_bytes(b, heads, s, d),), torch.uint8)
    return dict(
        dy8=_view(res.ws, lay.dy8, (t, g.d_model), _E4M3),
        do8=_view(res.ws, lay.do8, (t, g.h_q, d), _E4M3),
        do_T8=_view(res.ws, lay.do_T8, (t, g.h_q, d), _E4M3),
        og8=_view(res.ws, lay.og8, (t, g.h_q, d), _E4M3) if lay.og8 >= 0 else None,
        q8=_view(res.ws, lay.q8, (t, g.h_q, d), _E4M3),
        q_T8=_view(res.ws, lay.q_T8, (t, g.h_q, d), _E4M3),
        k8=_view(res.ws, lay.k8, (t, g.h_kv, d), _E4M3),
        k_T8=_view(res.ws, lay.k_T8, (t, g.h_kv, d), _E4M3),
        v8=_view(res.ws, lay.v8, (t, g.h_kv, d), _E4M3),
        # the GEMM-canonical dQKVG payloads and blobs are carved by NEED -- `dqkvg8 / sf_dqkvg` with `need_dh`, the transposed
        # `dqkvg_t8 / sf_dqkvg_t` with `need_dw_qkvg` (a dgrad-only block carves neither of the latter: `lay.dqkvg_t8 == -1`, and a
        # view at -1 is an empty slice) -- so each is a view only where carved, the API's own `>= 0` gate; every consumer here is
        # gated on the same `blk.need_*` flag
        dqkvg8=_view(res.ws, lay.dqkvg8, (t, g.n_qkvg), _E4M3) if lay.dqkvg8 >= 0 else None,
        dqkvg_t8=_view(res.ws, lay.dqkvg_t8, (g.n_qkvg, t), _E4M3) if lay.dqkvg_t8 >= 0 else None,
        sf=sf,
        sf_dqkvg=_view(res.ws, lay.sf_dqkvg, (sf_blob_bytes(t, g.n_qkvg),), torch.uint8) if lay.sf_dqkvg >= 0 else None,
        sf_dqkvg_t=_view(res.ws, lay.sf_dqkvg_t, (sf_blob_bytes(g.n_qkvg, t),), torch.uint8) if lay.sf_dqkvg_t >= 0 else None,
        do=_view(res.ws, lay.do_gated, (t, g.h_q, d), act),  # B3 wrote dO = dO_gated * sigmoid(gate) IN PLACE over B2's output
        rq=rq,  # the bf16 Q / K rebuild the block quantizations are the bitwise quantize of (the fused prologue's own, out of registers)
        rk=rk,
        v_band=_cols(proj, o_v, g.h_kv, d),  # the slab's V band (strided), the v quantize's source
        gate=_cols(proj, o_g, g.h_q, d),
        dqkvg=dqkvg,
        dg=dqkvg[:, o_g : o_g + g.h_q * d],
        dq=_view(res.ws, lay.dq, (t, g.h_q, d), act),
        dk=_view(res.ws, lay.dk, (t, g.h_kv, d), act),
        dv=_view(res.ws, lay.dv, (t, g.h_kv, d), act),
        lay=lay,
    )


def _delta(res) -> torch.Tensor:
    """The block's ``delta`` region B3 wrote (ALWAYS under quant): fp32 ``[B, H_q, S_pad]``, the row's external delta."""
    lay = res.blk._layout()
    assert lay.delta >= 0, "the quantized backward always carves delta (the external delta is mandatory)"
    return _view(res.ws, lay.delta, tuple(res.blk._sdpa.delta_shape), torch.float32)


def _scalar_block(res) -> torch.Tensor:
    """The whole fp32 scalar block as one view (every slot, in ``QUANT_SCALAR_SLOTS`` order)."""
    lay = res.blk._layout()
    return _view(res.ws, lay.quant_scalars, (len(_api_const("QUANT_SCALAR_SLOTS")),), torch.float32)


def _adapter_region(res, name: str) -> tuple:
    """``(offset into the block's sdpa_bwd_ws slice, shape, dtype)`` of one scratch region of the MXFP8 adapter's OWN carve
    (``_scratch_shapes()`` walked in order with ``ws_align`` -- the row suite's idiom), never the arithmetic of one layout."""
    from cudnn.sdpa.fwd.api_dsl import ws_align

    off = 0
    for n, shape, dt in res.blk._sdpa._impl._scratch_shapes():
        if n == name:
            return off, tuple(int(x) for x in shape), dt
        off += ws_align(math.prod(shape) * dt.itemsize)
    raise KeyError(name)


def _adapter_tensor(res, name: str) -> torch.Tensor:
    """A region of the adapter's scratch as a typed view of the block's workspace (``sdpa_bwd_ws`` + the region's offset)."""
    lay = res.blk._layout()
    off, shape, dt = _adapter_region(res, name)
    return _view(res.ws, lay.sdpa_bwd_ws + off, shape, dt)


def _live_rows(part: torch.Tensor, s: int) -> torch.Tensor:
    """The live ``[:, :s]`` rows of a per-Q-head partial region.  The adapter carves ``dk_part`` / ``dv_part`` as ``(B, kv_rows, H_q, D)``
    with ``kv_rows`` the kv side's 256-row-padded S whenever that side is padded (``_scratch_shapes``: ``kv_rows = skvp if self._kv_padded
    else skv``), so at a padded cell the region holds MORE rows than the ``[B, S, H_q, D]`` per-head reference -- the pad rows are the
    fold's own scratch over zero-filled operands, never compared.  The identity at an un-padded S.  Pinned on the host against the
    adapter's own carve by ``test_the_per_head_partial_regions_are_compared_on_their_live_rows``."""
    assert part.dim() == 4 and part.shape[1] >= s, (tuple(part.shape), s)
    return part[:, :s]


# ---------------------------------------------------------------------------
# The bitwise layer: every payload and blob against the torch quantization of its own source, the scalars, the delta
# ---------------------------------------------------------------------------


def _mx_oracle_thd(src_thd: torch.Tensor, b: int, s: int, h: int):
    """``quantize_to_mxfp8`` on a ``[T, H, D]`` buffer permuted to BHSD (the quantizer suite's oracle plumbing): ``(row_data [T, H, D]
    e4m3, row_sf flat uint8, col_data [T, H, D] e4m3, col_sf flat uint8)`` in the SDPA's own scale-factor layouts."""
    _test_python_root()
    from sdpa.mxfp8_quant import quantize_to_mxfp8

    d = src_thd.shape[-1]
    bhsd = src_thd.reshape(b, s, h, d).permute(0, 2, 1, 3)
    row_d, _, row_sf, col_d, _, col_sf = quantize_to_mxfp8(bhsd, b, h, s, d, with_ref=False)
    to_thd = lambda x: x.permute(0, 2, 1, 3).reshape(b * s, h, d)  # noqa: E731
    return to_thd(row_d), row_sf.reshape(-1), to_thd(col_d), col_sf.reshape(-1)


def _sf_refs(src_thd: torch.Tensor, b: int, s: int, h: int):
    """The per-element fp32 dequant scales of the rowwise and the columnwise quantization of a ``[T, H, D]`` bf16 source, as BHSD
    ``[B, H, S, D]`` -- the row reference's ``sf_*_ref`` operands, re-quantized from the kernel's OWN bf16 source (bit-exact vs the
    kernel, pinned by the bitwise layer), never from a second quantization of fp64 values."""
    _test_python_root()
    from sdpa.mxfp8_quant import quantize_to_mxfp8

    d = src_thd.shape[-1]
    bhsd = src_thd.reshape(b, s, h, d).permute(0, 2, 1, 3)
    _, sf_d_ref, _, _, sf_s_ref, _ = quantize_to_mxfp8(bhsd, b, h, s, d, with_ref=True)
    return sf_d_ref, sf_s_ref


def _assert_mx_bitwise(src_thd: torch.Tensor, payload: torch.Tensor, sf_flat: torch.Tensor, axis: str, *, b: int, s: int, h: int, what: str) -> None:
    """``payload`` / ``sf_flat`` are BITWISE the torch quantization of ``src_thd`` in the SDPA's layout for ``axis`` (codes as uint8,
    the scale-factor bytes as laid out by ``swizzle_sf_rowwise`` / ``_columnwise``) -- the quantizer suite's bar at block level."""
    row_d, row_sf, col_d, col_sf = _mx_oracle_thd(src_thd.contiguous(), b, s, h)
    ref_d, ref_sf = (row_d, row_sf) if axis == "row" else (col_d, col_sf)
    n_bad_d = int((payload.view(torch.uint8) != ref_d.view(torch.uint8)).sum())
    assert ref_sf.numel() == sf_flat.numel(), (what, ref_sf.numel(), sf_flat.numel())
    n_bad_sf = int((sf_flat != ref_sf).sum())
    print(f"{what} ({axis}): {n_bad_d} of {payload.numel()} e4m3 codes and {n_bad_sf} of {sf_flat.numel()} scale bytes differ from the torch quantization")
    assert n_bad_d == 0, f"{what}: {n_bad_d} e4m3 codes differ from the torch quantization of the kernel's own source"
    assert n_bad_sf == 0, f"{what}: {n_bad_sf} scale-factor bytes differ from the torch quantization (byte ORDER or rounding)"


def _canonical_ref(x2d: torch.Tensor):
    """``[rows, K]`` bf16 -> (e4m3 codes ``[rows, K]``, the PADDED canonical F8_128x4 blob, the logical E8M0 bytes ``[rows, K/32]``):
    the oracle's rowwise quantizer and blob builder -- what the GEMM-canonical arms of the kernel must reproduce byte for byte."""
    codes, e = mx_quantize_rowwise_2d(x2d.contiguous())
    return codes, mx_swizzle_sf_rowwise_padded(e), e


def _assert_canonical_bitwise(x2d: torch.Tensor, payload: torch.Tensor, blob: torch.Tensor, what: str) -> None:
    codes, ref_blob, _e = _canonical_ref(x2d)
    n_bad_d = int((payload.view(torch.uint8) != codes.view(torch.uint8)).sum())
    assert ref_blob.numel() == blob.numel(), (what, ref_blob.numel(), blob.numel())
    n_bad_sf = int((blob != ref_blob).sum())
    print(f"{what} (canonical): {n_bad_d} of {payload.numel()} e4m3 codes and {n_bad_sf} of {blob.numel()} scale bytes differ from the torch quantization")
    assert n_bad_d == 0, f"{what}: {n_bad_d} e4m3 codes differ from the torch quantization of the kernel's own source"
    assert n_bad_sf == 0, f"{what}: {n_bad_sf} canonical scale-factor bytes differ (byte ORDER, rounding, or an unwritten pad)"


def _dead_slots() -> tuple:
    return tuple(n for n in _api_const("QUANT_SCALAR_SLOTS") if n not in _MXFP8_LIVE_SLOTS)


def _assert_quantizers_scalars_delta_bitwise(res) -> dict:
    """The BITWISE layer of a cell (module docstring): ``dy8`` against torch's saturating cast at the READ-BACK scale; ``og8`` against
    the forward's ``o8`` and the kernels' bf16 ``O_gated``; the seven SDPA-layout payloads and blobs against ``quantize_to_mxfp8`` of
    their own bf16 sources (``q8 / sf_q`` and ``k8 / sf_k`` ALSO ``torch.equal`` the FORWARD workspace's bytes); the two canonical
    dQKVG payloads and blobs against the oracle's rowwise quantizer and padded blob builder; every scalar of the block (the five
    live dY slots by the kernel's formula, the three live constants at the fp32 of the MxQuantSpec's values, the 21 dead slots exactly
    0.0); and ``delta`` (the chain's own ``dot_do_o`` with an exactly-zero pad tail).  Returns the intermediates (``_slots``)."""
    blk, g, sc, sp, saved = res.blk, res.geom, res.scalars, res.spec, res.saved
    b, s, d = res.batch, res.seq_len, g.d_head
    t = b * s
    v = _slots(res)
    # --- the per-tensor points: dy8, og8 -----------------------------------------------------------------------------
    _assert_e4m3_bitwise(v["dy8"], res.dy, sc["scale_dy"], "dy8")
    flay = res.fwd.blk._layout()
    if blk.need_dw_o:
        assert v["og8"] is not None
        _assert_og8_bitwise_the_kernels_o_gated(v["og8"], saved.o.view(t, g.h_q, d), v["gate"], sp.scale_o, _view(res.fwd.ws, flay.o8, (t, g.h_q, d), _E4M3))
    else:
        assert v["og8"] is None and v["lay"].og8 == -1
    # --- the SDPA-layout block quantizations, bitwise the torch quantization of their own sources -------------------
    _assert_mx_bitwise(v["do"], v["do8"], v["sf"]["sf_do"], "row", b=b, s=s, h=g.h_q, what="do8 / sf_do (the gate backward's bf16 dO)")
    _assert_mx_bitwise(v["do"], v["do_T8"], v["sf"]["sf_do_T"], "col", b=b, s=s, h=g.h_q, what="do_T8 / sf_do_T")
    _assert_mx_bitwise(v["rq"], v["q8"], v["sf"]["sf_q"], "row", b=b, s=s, h=g.h_q, what="q8 / sf_q (the bf16 rebuild)")
    _assert_mx_bitwise(v["rq"], v["q_T8"], v["sf"]["sf_q_T"], "col", b=b, s=s, h=g.h_q, what="q_T8 / sf_q_T")
    _assert_mx_bitwise(v["rk"], v["k8"], v["sf"]["sf_k"], "row", b=b, s=s, h=g.h_kv, what="k8 / sf_k")
    _assert_mx_bitwise(v["rk"], v["k_T8"], v["sf"]["sf_k_T"], "col", b=b, s=s, h=g.h_kv, what="k_T8 / sf_k_T")
    _assert_mx_bitwise(v["v_band"], v["v8"], v["sf"]["sf_v"], "row", b=b, s=s, h=g.h_kv, what="v8 / sf_v (the slab's V band, ROWWISE)")
    # the forward's own bytes: q8 / sf_q and k8 / sf_k are the forward's SDPA operands recomputed bit-exactly (v8 is NOT: the forward's is columnwise)
    from cudnn.gated_attention_block.api import _sf_slot_bytes

    for name, sf_name, h in (("q8", "sf_q", g.h_q), ("k8", "sf_k", g.h_kv)):
        fwd_bytes = _view(res.fwd.ws, getattr(flay, name), (t, h, d), _E4M3).view(torch.uint8)
        assert torch.equal(v[name].view(torch.uint8), fwd_bytes), f"{name}: the recomputed payload is not bitwise the forward's"
        fwd_sf = _view(res.fwd.ws, getattr(flay, sf_name), (_sf_slot_bytes(b, h, s, d),), torch.uint8)
        assert torch.equal(v["sf"][sf_name], fwd_sf), f"{sf_name}: the recomputed scale factors are not bitwise the forward's"
    # --- the GEMM-canonical block quantizations of dQKVG -------------------------------------------------------------------
    if blk.need_dh:
        _assert_canonical_bitwise(v["dqkvg"], v["dqkvg8"], v["sf_dqkvg"], "dqkvg8 / sf_dqkvg (rowwise over (T, N))")
    if blk.need_dw_qkvg:
        _assert_canonical_bitwise(v["dqkvg"].t(), v["dqkvg_t8"], v["sf_dqkvg_t"], "dqkvg_t8 / sf_dqkvg_t (transposed over (N, T))")
    # --- scalars, bitwise ----------------------------------------------------------------------------------------------
    assert sc["amax_dy"] == res.dy.float().abs().max().item(), (sc["amax_dy"], res.dy.float().abs().max().item())
    if res.grad_scaling == "current":
        assert sc["scale_dy"] == _grad_scale(sc["amax_dy"]), (sc["scale_dy"], _grad_scale(sc["amax_dy"]))
    else:
        assert sc["scale_dy"] == _f32(res.scale_dy_t.item())
    assert sc["descale_dy"] == _f32(1.0 / sc["scale_dy"])
    assert sc["scale_o"] == _f32(sp.scale_o) and sc["descale_o"] == _f32(1.0 / sp.scale_o) and sc["descale_w_o"] == _f32(sp.descale_w_o)
    assert sc["alpha_b1"] == _f32_mul(sc["descale_dy"], sc["descale_o"]) and sc["alpha_b2"] == _f32_mul(sc["descale_dy"], sc["descale_w_o"])
    for name in _dead_slots():
        assert sc[name] == 0.0, f"{name}: a dead slot of the MXFP8 arm reads {sc[name]!r}, not 0.0"
    assert all(np.isfinite(sc[n]) for n in _MXFP8_LIVE_SLOTS)
    # --- delta, bitwise ------------------------------------------------------------------------------------------------
    from test_sigmoid_gate_bwd import _chain_dot_do_o

    delta = _delta(res)
    want = _chain_dot_do_o(saved.o, v["do"].view(b, s, g.h_q, d))
    assert delta.shape == want.shape and torch.equal(delta[..., :s], want[..., :s]), "delta is not bitwise the chain's dot_do_o"
    assert torch.equal(delta[..., s:], torch.zeros_like(delta[..., s:])), "the delta pad tail must be exact zeros"
    return v


# ---------------------------------------------------------------------------
# The row's reference, per Q head: once-rounded dK / dV, the fold-modelled dV, the per-head partials
# ---------------------------------------------------------------------------


def _head_sf(sf: torch.Tensor, b: int, h_total: int, s: int, d: int, i: int) -> torch.Tensor:
    """Head ``i`` of a per-element scale tensor in the row reference's OWN ``[b*h, s, d]`` layout (``quantize_to_mxfp8``'s
    ``sf_*_ref``, which ``mxfp8_ref._dequant`` views to the payload's ``[b, 1, s, d]``): the batch-head axis is FLAT, so a head is a
    slice of the tensor re-viewed as ``[b, h, s, d]`` -- ``sf[:, i : i + 1]`` on the flat tensor slices the S axis instead (a
    ``[b*h, 1, d]`` tensor the dequant cannot view to ``[b, 1, s, d]``: a shape error on every cell).  Pinned on the host by
    ``test_the_per_head_scale_slices_compose_the_rows_dequant``."""
    assert sf.shape == (b * h_total, s, d), (tuple(sf.shape), (b * h_total, s, d))
    return sf.reshape(b, h_total, s, d)[:, i : i + 1]


def _row_reference(res, v: dict) -> dict:
    """The MXFP8 row's reference (``mxfp8_ref.compute_ref_backward``) over the block's OWN payloads, per Q HEAD (a group of one),
    with the per-element scales re-quantized from the kernel's bf16 sources, the record's exact LSE, the block's ``delta`` (the
    SAME tensor the kernel consumed), ``quantize_ds=True`` (the block-scaled dS chain), fp32 outputs per head.  Returns BSHD fp32
    tensors rounded like the row's bf16 output: ``dq`` (per head), ``dk`` / ``dv`` ONCE-rounded (the per-head fp32 partials summed
    in the fold kernel's ascending order, then rounded once -- the row's own reference), ``dv_fold`` the FOLD-MODELLED dV (each
    per-head partial rounded to bf16 first, the kernel's structure), and the per-head fp32 partials ``dk_parts`` / ``dv_parts``
    ``[B, S, H_q, D]`` (the kernel's ``dk_part`` / ``dv_part`` regions under GQA)."""
    _test_python_root()
    from sdpa.mxfp8_ref import compute_ref_backward

    blk, g, saved = res.blk, res.geom, res.saved
    b, s, d = blk.batch, blk.seq_len, g.d_head
    hq, hk = g.h_q, g.h_kv
    grp = hq // hk
    bhsd = lambda x, h: x.reshape(b, s, h, d).permute(0, 2, 1, 3).contiguous()  # noqa: E731
    sfq_d, sfq_s = _sf_refs(v["rq"], b, s, hq)
    sfk_d, sfk_s = _sf_refs(v["rk"], b, s, hk)
    sfv_d, _ = _sf_refs(v["v_band"].contiguous(), b, s, hk)
    sfdo_d, sfdo_s = _sf_refs(v["do"], b, s, hq)
    q8, qT8, k8, kT8, v8 = bhsd(v["q8"], hq), bhsd(v["q_T8"], hq), bhsd(v["k8"], hk), bhsd(v["k_T8"], hk), bhsd(v["v8"], hk)
    do8, doT8 = bhsd(v["do8"], hq), bhsd(v["do_T8"], hq)
    o16, do16 = saved.o.permute(0, 2, 1, 3), v["do"].reshape(b, s, hq, d).permute(0, 2, 1, 3)
    delta, lse = _delta(res)[..., :s], saved.lse
    right = 0 if g.is_causal else None
    sl = lambda x, i: x[:, i : i + 1]  # noqa: E731  -- head i of a BHSD payload / port, or of a [B, H, S] stats / delta
    dq_parts, dk_parts, dv_parts = [], [], []
    for h in range(hq):
        kv = h // grp
        dq_h, dk_h, dv_h, _dsink = compute_ref_backward(
            sl(q8, h), sl(qT8, h), sl(k8, kv), sl(kT8, kv), sl(v8, kv), sl(o16, h), sl(do16, h), sl(do8, h), sl(doT8, h), g.scale,
            _head_sf(sfq_d, b, hq, s, d, h), _head_sf(sfq_s, b, hq, s, d, h), _head_sf(sfk_d, b, hk, s, d, kv), _head_sf(sfk_s, b, hk, s, d, kv),
            _head_sf(sfv_d, b, hk, s, d, kv), _head_sf(sfdo_d, b, hq, s, d, h), _head_sf(sfdo_s, b, hq, s, d, h),
            torch_itype=_E4M3, torch_otype=torch.float32, left_bound=None, right_bound=right, diag_align=None,
            stats=sl(lse, h), quantize_ds=True, delta=sl(delta, h),
        )  # fmt: skip
        dq_parts.append(dq_h.float())
        dk_parts.append(dk_h.float())
        dv_parts.append(dv_h.float())
    dq = torch.cat(dq_parts, dim=1)  # [B, H_q, S, D]
    dk_once, dv_once, dv_fold = [], [], []
    for kv in range(hk):
        members = [kv * grp + m for m in range(grp)]  # the fold kernel's order: group members ascending, fp32 accumulation
        acc_k = dk_parts[members[0]].clone()
        acc_v = dv_parts[members[0]].clone()
        acc_v16 = dv_parts[members[0]].to(torch.bfloat16).float()
        for h in members[1:]:
            acc_k = acc_k + dk_parts[h]
            acc_v = acc_v + dv_parts[h]
            acc_v16 = acc_v16 + dv_parts[h].to(torch.bfloat16).float()
        dk_once.append(acc_k)
        dv_once.append(acc_v)
        dv_fold.append(acc_v16)
    r16 = lambda x: x.to(torch.bfloat16).float()  # noqa: E731
    bshd = lambda x: x.permute(0, 2, 1, 3).contiguous()  # noqa: E731
    return dict(
        dq=bshd(r16(dq)),
        dk=bshd(r16(torch.cat(dk_once, dim=1))),
        dv=bshd(r16(torch.cat(dv_once, dim=1))),
        dv_fold=bshd(r16(torch.cat(dv_fold, dim=1))),
        dk_parts=bshd(torch.cat(dk_parts, dim=1)),
        dv_parts=bshd(torch.cat(dv_parts, dim=1)),
    )


def _row_tol():
    """The MXFP8 SDPA row suite's own recipe -- its tolerance constants and ``assert_close_fp8_grad`` -- imported, never re-literalled."""
    frost_dir = os.path.join(_test_python_root(), "sdpa", "frost")
    if frost_dir not in sys.path:
        sys.path.insert(0, frost_dir)
    from sdpa.fp8 import assert_close_fp8_grad
    from test_sdpa_bwd_mxfp8_sm107 import _GRAD_TOL

    return _GRAD_TOL, assert_close_fp8_grad


def _row_keys(res) -> dict:
    """The reduction length feeding each ROW of a bf16 output: ``dh`` (a token row) over N, the weight gradients (an output row)
    over the tokens -- the ``keys`` of the ``1e-5 x rows x keys`` row budget."""
    return dict(dh=res.geom.n_qkvg, dw_qkvg=res.batch * res.seq_len, dw_o=res.batch * res.seq_len)


# ---------------------------------------------------------------------------
# The oracles (the fp8 oracle's MXFP8 sibling): fold-modelled (M), once-rounded (M), unquantized (U), seeded
# ---------------------------------------------------------------------------


def _oracle(res, *, modelled: bool, seeded: Optional[dict] = None, fold: str = "kernel") -> dict:
    """The MXFP8 backward oracle fed the block's OWN conditions: its read-back ``scale_dy``, the SAME ``delta`` the kernel consumed,
    the record's exact LSE, bf16 pre-gate O and GATE band, the block's bf16 dO, the block's own payloads and scale-factor blobs
    (the kernel's inputs) and the caller's transposed artifacts (the (M) dgrad / wgrad dequantize them through their blobs);
    ``fold`` = ``"kernel"`` (dK once-rounded from fp32 partials, dV per-Q-head bf16 partials in the fold kernel's order) or
    ``"once"``; ``seeded`` substitutes the block's bf16 dQ / dK / dV.  (U) keeps its own fp64 chain on purpose."""
    sc = res.scalars
    g, t = res.geom, res.batch * res.seq_len
    record = {}
    if modelled:
        v = _slots(res)
        record = dict(
            lse=res.saved.lse,
            o=res.saved.o,
            gate=v["gate"],
            do=v["do"],
            q8=v["q8"],
            sf_q=v["sf"]["sf_q"],
            q_T8=v["q_T8"],
            sf_q_T=v["sf"]["sf_q_T"],
            k8=v["k8"],
            sf_k=v["sf"]["sf_k"],
            k_T8=v["k_T8"],
            sf_k_T=v["sf"]["sf_k_T"],
            v8=v["v8"],
            sf_v=v["sf"]["sf_v"],
            do8=v["do8"],
            sf_do=v["sf"]["sf_do"],
            do_T8=v["do_T8"],
            sf_do_T=v["sf"]["sf_do_T"],
        )
    return gated_attention_block_mxfp8_bwd_reference(
        res.inp,
        RefGeometry(**res.geom_kw),
        res.spec,
        res.dy,
        scale_dy=sc["scale_dy"],
        delta=_delta(res)[..., : res.seq_len],
        modelled=modelled,
        seeded=seeded,
        fold=fold,
        **{k: res.art.get(k) for k in _ARTIFACTS},
        **record,
    )


def _oracle_m(res, fold: str = "kernel") -> dict:
    cache = getattr(res, "oracle_m", None) or {}
    if fold not in cache:  # memoised on the run: the report cell and the row-budget cell read the same oracle
        cache[fold] = _oracle(res, modelled=True, fold=fold)
        res.oracle_m = cache
    return cache[fold]


def _oracle_u(res) -> dict:
    return _oracle(res, modelled=False)


def _oracle_seeded(res) -> dict:
    v = _slots(res)
    b, s, g = res.batch, res.seq_len, res.geom
    seed = dict(dq=v["dq"].view(b, s, g.h_q, g.d_head), dk=v["dk"].view(b, s, g.h_kv, g.d_head), dv=v["dv"].view(b, s, g.h_kv, g.d_head))
    return _oracle(res, modelled=True, seeded=seed)


def _report_seeded_intermediates(res, v: dict, ref: dict) -> dict:
    """Localisation between B4 and the outputs (no new bound): the block's bf16 ``dqkvg`` bands (B3's dG, B5+B6's dQ_pre / dK_pre)
    against the seeded oracle's fp64 bands as a fraction of the bf16 block's bound, PRINTED; the slab's V band ``torch.equal`` the
    block's own dV slot (a copy, asserted); and the e4m3 flips of the two canonical dQKVG quantizations against the oracle's own
    bf16-rounded slab quantized AT THE KERNEL'S OWN SCALE BYTES -- the flip class of block scales is one e4m3 step x the block's
    ``2^(e - 127)``: a flip of ``dqkvg_t8`` at ``[n, t]`` moves ``dW_qkvg`` row ``n`` by ``flip x deq(h_t)[:, t]``, a flip of ``dqkvg8``
    at ``[t, n]`` moves ``dh`` row ``t`` by ``flip x deq(w_qkvg_t)[:, n]``.  Returns that flip evidence for
    ``_assert_seeded_dw_qkvg_row_budgeted``: ``flips_t`` (bool ``[N, T]``), ``deq_t`` / ``deq_t_ref`` (fp32 ``[N, T]``: the kernel's
    and the oracle's dequantized transposed gradient at the kernel's scales), ``dw_qkvg_rows`` / ``dh_rows`` and ``band_col_worst``
    (per slab column: the PRE-cast band's worst cell as a fraction of its band's bound)."""
    _test_python_root()
    from sdpa.mxfp8_quant import e8m0_to_float

    g = res.geom
    t, d, n = res.batch * res.seq_len, g.d_head, g.n_qkvg
    o_q, o_g, o_k, o_v = g.qkvg_offsets
    dqkvg = v["dqkvg"]
    bands = ((o_q, g.h_q, "dq_pre", "dq_pre"), (o_g, g.h_q, "dg", "dg"), (o_k, g.h_kv, "dk_pre", "dk_pre"), (o_v, g.h_kv, "dv", "dv_band"))
    ref_slab = torch.empty(t, n, dtype=torch.float64, device=dqkvg.device)
    band_col_worst = torch.empty(n, dtype=torch.float64, device=dqkvg.device)
    for off, heads, name, key in bands:
        _report_close(_cols(dqkvg, off, heads, d), ref[key], f"band {name} vs the seeded oracle")
        band_ref = ref[key].reshape(t, heads * d)
        ref_slab[:, off : off + heads * d] = band_ref
        band_bound = _ATOL_FRAC[dqkvg.dtype] * band_ref.abs().max() + _RTOL[dqkvg.dtype] * band_ref.abs()
        band_col_worst[off : off + heads * d] = ((_cols(dqkvg, off, heads, d).reshape(t, heads * d).double() - band_ref).abs() / band_bound).amax(dim=0)
    assert torch.equal(_cols(dqkvg, o_v, g.h_kv, d).contiguous(), v["dv"]), "the slab's V band is not bitwise the block's own dV slot"
    ref16 = ref_slab.to(torch.bfloat16)
    out = dict(
        band_col_worst=band_col_worst,
        dw_qkvg_rows=torch.zeros(0, dtype=torch.long, device=dqkvg.device),
        dh_rows=torch.zeros(0, dtype=torch.long, device=dqkvg.device),
    )
    if res.blk.need_dw_qkvg:
        # the kernel's transposed scale bytes [N, T/32] -> the oracle's bf16 slab^T quantized at THOSE scales
        e_t = mx_unswizzle_sf_rowwise(v["sf_dqkvg_t"], n, t)
        scale_t = e8m0_to_float(e_t).repeat_interleave(32, dim=1)  # [N, T]
        ref_t = ref16.t().float()
        codes_ref_t = quant_e4m3(ref_t / scale_t, 1.0)
        flips_t = v["dqkvg_t8"].view(torch.uint8) != codes_ref_t.view(torch.uint8)
        rows_n, _cols_t = torch.nonzero(flips_t, as_tuple=True)
        deq_t = v["dqkvg_t8"].float() * scale_t
        deq_t_ref = codes_ref_t.float() * scale_t
        out.update(flips_t=flips_t, deq_t=deq_t, deq_t_ref=deq_t_ref, dw_qkvg_rows=torch.unique(rows_n))
        n_pre = int((dqkvg != ref16).sum())
        print(
            f"dqkvg_t8 vs e4m3(bf16(seeded oracle dqkvg)^T) at the kernel's own block scales: {int(flips_t.sum())} of {flips_t.numel()} codes differ "
            f"({n_pre} bf16 cells differ before the cast); {int(out['dw_qkvg_rows'].numel())} dW_qkvg rows touched"
        )
    if res.blk.need_dh:
        e = mx_unswizzle_sf_rowwise(v["sf_dqkvg"], t, n)
        scale = e8m0_to_float(e).repeat_interleave(32, dim=1)  # [T, N]
        codes_ref = quant_e4m3(ref16.float() / scale, 1.0)
        flips = v["dqkvg8"].view(torch.uint8) != codes_ref.view(torch.uint8)
        rows_t, _cols_n = torch.nonzero(flips, as_tuple=True)
        out.update(flips=flips, dh_rows=torch.unique(rows_t))
        print(
            f"dqkvg8 vs e4m3(bf16(seeded oracle dqkvg)) at the kernel's own block scales: {int(flips.sum())} of {flips.numel()} codes differ; {int(out['dh_rows'].numel())} dh rows touched"
        )
    if v["og8"] is not None and ref.get("og8") is not None:
        og_flips = v["og8"].view(torch.uint8) != ref["og8"].reshape(v["og8"].shape).view(torch.uint8)
        _rows, cols_j = torch.nonzero(og_flips.reshape(t, -1), as_tuple=True)
        print(
            f"og8 vs the seeded oracle's og8: {int(og_flips.sum())} of {og_flips.numel()} codes differ; {int(torch.unique(cols_j).numel())} dW_o columns touched"
        )
    for name, keys in _row_keys(res).items():
        if res.grads.get(name) is not None and ref.get(name) is not None:
            n_out, n_rows = _rows_outside(res.grads[name], ref[name])
            print(
                f"{name} vs the seeded oracle: {n_out} of {n_rows} rows outside the bf16 bound (row budget 1e-5 x rows x keys = {_row_budget(n_rows, keys):.3g})"
            )
    return out


def _assert_seeded_dw_qkvg_row_budgeted(res, v: dict, ref: dict, flip_ev: dict, what: str) -> float:
    """The SEEDED ``dW_qkvg`` in the row-budgeted form WITH its attribution (the fp8 module's form over block scales): (1) the rows
    with a cell outside the bf16 block's bound stay within ``_row_budget`` (``1e-5 x rows x keys``, keys = T); (2) EVERY such row is a
    row a ``dqkvg_t8`` code flip touched (a code of block ``(n, t // 32)`` differing from the oracle's bf16 band quantized at the
    kernel's own scale byte); (3) for EVERY such row the PRE-cast slab column is itself inside the bf16 bound of its band against the
    seeded oracle -- the flip is the cast's rounding, not a band's miss.  Printed per row outside: its flip count, its pre-cast
    column's worst, and its worst before / after the flips' rank-1 term ``sum_t (deq(dqkvg_t8) - deq_ref)[n, t] x deq(h_t)[:, t]`` is
    removed.  Returns the per-cell worst (``_report_close``)."""
    got, ref64 = res.grads["dw_qkvg"], ref["dw_qkvg"]
    assert torch.isfinite(got).all(), f"{what}: non-finite cells"
    worst = _report_close(got, ref64, what)
    outside = _rows_outside_mask(got, ref64)
    rows_out = torch.nonzero(outside).flatten()
    n_out, n_rows = int(rows_out.numel()), int(outside.numel())
    budget = _row_budget(n_rows, _row_keys(res)["dw_qkvg"])
    unexplained = rows_out[~torch.isin(rows_out, flip_ev["dw_qkvg_rows"].to(rows_out.device))]
    band_worst = flip_ev["band_col_worst"].to(rows_out.device)[rows_out]
    not_the_casts = rows_out[band_worst > 1.0]
    if n_out:
        g = res.geom
        got64, r64 = got.detach().double(), ref64.detach().double()
        bound = _ATOL_FRAC[got.dtype] * r64.abs().max() + _RTOL[got.dtype] * r64.abs()
        h_t64 = mx_dequant_rowwise_2d(res.art["h_t"], res.art["h_t_sf"]).double()  # [d_model, T]
        code_diff = (flip_ev["deq_t"][rows_out] - flip_ev["deq_t_ref"][rows_out]).double()  # [rows out, T]
        flip_term = code_diff @ h_t64.t()  # [rows out, d_model]: the flips' rank-1 contributions to each row
        before = ((got64[rows_out] - r64[rows_out]).abs() / bound[rows_out]).amax(dim=1)
        after = ((got64[rows_out] - r64[rows_out] - flip_term).abs() / bound[rows_out]).amax(dim=1)
        n_flips = flip_ev["flips_t"][rows_out].sum(dim=1)
        for i, nn in enumerate(rows_out.tolist()):
            print(
                f"{what}: row {nn} ({_qkvg_band(g, nn)}) outside the bf16 bound -- worst {before[i].item():.3f} of the bound, {int(n_flips[i])} dqkvg_t8 flips "
                f"in its row, {after[i].item():.3f} after removing their rank-1 term; its pre-cast slab column at {band_worst[i].item():.3f} of its band's bound"
            )
    print(
        f"{what}: {n_out} of {n_rows} rows outside the bf16 bound (row budget 1e-5 x rows x keys = {budget:.3g}), {int(unexplained.numel())} of them untouched "
        f"by a dqkvg_t8 flip, {int(not_the_casts.numel())} with the pre-cast slab column itself outside its band's bound"
    )
    assert n_out <= budget, f"{what}: {n_out} of {n_rows} rows outside the bf16 bound exceed the row budget {budget:.3g} -- rows {rows_out.tolist()}"
    assert (
        unexplained.numel() == 0
    ), f"{what}: rows {unexplained.tolist()} are outside the bf16 bound and no dqkvg_t8 code flip touched them (not the cast's flip class)"
    assert not_the_casts.numel() == 0, (
        f"{what}: rows {not_the_casts.tolist()} are outside the bf16 bound and their PRE-cast slab columns are themselves outside the band's bound "
        f"({[round(x, 3) for x in band_worst[band_worst > 1.0].tolist()]} of it): a miss of the band upstream of the cast"
    )
    return worst


# ---------------------------------------------------------------------------
# ACCEPT (Rubin)
# ---------------------------------------------------------------------------


@requires_rubin
@_MATRIX
def test_mxfp8_stage_localised_bounds(cell):
    """Every stage of the MXFP8 backward against the bound calibrated FOR IT, on the block's own operands (module docstring): the
    bitwise layer (every payload and blob, the scalars, the delta); the gradients ``torch.equal`` a standalone
    ``SdpaBwdDslSm107Mxfp8(external_delta=False)`` run over the block's own payloads (the external delta IS the row's pre-pass
    here); B1 / B2 on the block's e4m3 operands under the GEMM bound (B2 as the composite dO under the bf16 bound: B3 overwrote
    it in place); B7 / B8 under the GEMM bound on fp64 of the operands dequantized THROUGH THE BLOBS (the orientation guard);
    the SDPA stage under the row's recipe against its once-rounded reference per gradient, the kernel's per-Q-head partials
    (fp32 ``dk_part``, bf16 ``dv_part``) against the per-head reference under the same recipe, the bf16 bound FORM printed against
    the once-rounded fold and the fold-modelled one (asserted once the first run recorded it); ``dh / dW_o / dW_*_norm`` against
    the oracle SEEDED with the block's own dQ / dK / dV, ``dW_qkvg`` row-budgeted with its attribution on ``dqkvg_t8`` flips."""
    res = _cell_backward(cell)
    blk, g, sc, sp, saved = res.blk, res.geom, res.scalars, res.spec, res.saved
    b, s, d = res.batch, res.seq_len, g.d_head
    t, grp = b * s, g.h_q // g.h_kv
    v = _assert_quantizers_scalars_delta_bitwise(res)
    # --- the GEMMs on the block's own operands ---------------------------------------------------------------------------
    dy8_64 = v["dy8"].double() * (1.0 / sc["scale_dy"])
    wo64 = res.inp["w_o"].double() * sp.descale_w_o  # [d_model, H_q*D]
    do_gated64 = dy8_64 @ wo64
    do_ref64 = do_gated64.to(torch.bfloat16).double().view(t, g.h_q, d) * torch.sigmoid(v["gate"].double())
    _assert_grad_close(v["do"], do_ref64, "dO (B2 + B3 composite; B2's output is overwritten in place)")
    if blk.need_dw_o:
        og8_64 = v["og8"].double() * (1.0 / sp.scale_o)
        _gemm_bound(res.grads["dw_o"], dy8_64.t() @ og8_64.view(t, g.h_q * d), "B1 dW_o = dy8^T . og8 (alpha_b1)")
    if blk.need_dw_qkvg:
        a64 = mx_dequant_rowwise_2d(v["dqkvg_t8"], v["sf_dqkvg_t"]).double()  # [N, T] through the kernel's own blob
        b64 = mx_dequant_rowwise_2d(res.art["h_t"], res.art["h_t_sf"]).double()  # [d_model, T] through the caller's blob
        _gemm_bound(res.grads["dw_qkvg"], a64 @ b64.t(), "B7 dW_qkvg = dqkvg_t8 . h_t^T (block scales through the blobs)")
    if blk.need_dh:
        a64 = mx_dequant_rowwise_2d(v["dqkvg8"], v["sf_dqkvg"]).double()  # [T, N]
        b64 = mx_dequant_rowwise_2d(res.art["w_qkvg_t"], res.art["w_qkvg_t_sf"]).double()  # [d_model, N]
        _gemm_bound(res.grads["dh"].view(t, g.d_model), a64 @ b64.t(), "B8 dh = dqkvg8 . w_qkvg_t^T (block scales through the blobs)")
    # --- the SDPA stage: bitwise the row's own pre-pass, then the row's recipe against its once-rounded reference ---------
    _assert_sdpa_stage_bitwise_the_rows_own_pre_pass(res, v)
    grad_tol, assert_close_fp8_grad = _row_tol()
    refs = _row_reference(res, v)
    for name, h, tag in (("dq", g.h_q, "dQ"), ("dk", g.h_kv, "dK"), ("dv", g.h_kv, "dV")):
        got = v[name].view(b, s, h, d).float()
        assert_close_fp8_grad(got, refs[name], grad_tol["atol"], grad_tol["rtol"], tag, keys=s, budget=1e-5)
        _report_stage_difference(v[name].view(b, s, h, d), refs[name], f"{tag} (once-rounded reference)")
    _report_stage_difference(v["dv"].view(b, s, g.h_kv, d), refs["dv_fold"], "dV (fold-modelled reference: per-head bf16 partials in the fold's order)")
    if _STAGE_BF16_FORM_ASSERTED:
        _assert_grad_close(v["dq"].view(b, s, g.h_q, d), refs["dq"].double(), "dQ vs the row's reference (the bf16 bound form)")
        _assert_grad_close(v["dk"].view(b, s, g.h_kv, d), refs["dk"].double(), "dK vs the once-rounded reference (the bf16 bound form)")
        _assert_grad_close(v["dv"].view(b, s, g.h_kv, d), refs["dv_fold"].double(), "dV vs the fold-modelled reference (the bf16 bound form)")
    else:
        for name, ref_name, h, tag in (("dq", "dq", g.h_q, "dQ"), ("dk", "dk", g.h_kv, "dK"), ("dv", "dv_fold", g.h_kv, "dV (fold-modelled)")):
            _report_close(
                v[name].view(b, s, h, d), refs[ref_name].double(), f"{tag} vs its reference, the bf16 bound form (printed until the first run records it)"
            )
    if grp > 1:
        # the kernel's per-Q-head partials under the same recipe: fp32 dk_part (rounded once by the fold), bf16 dv_part
        dk_region, dv_region = _adapter_tensor(res, "dk_part"), _adapter_tensor(res, "dv_part")
        dk_part, dv_part = _live_rows(dk_region, s), _live_rows(dv_region, s)  # a kv-padded carve's pad rows are the fold's own scratch
        print(
            f"{cell.id}: the row's per-Q-head partials -- dk_part {dk_region.dtype} {tuple(dk_region.shape)}, dv_part {dv_region.dtype} "
            f"{tuple(dv_region.shape)} (carved over the adapter's kv rows; compared on the {s} live rows)"
        )
        assert_close_fp8_grad(dk_part.float(), refs["dk_parts"], grad_tol["atol"], grad_tol["rtol"], "dk_part (per Q head)", keys=s, budget=1e-5)
        assert_close_fp8_grad(
            dv_part.float(), refs["dv_parts"].to(torch.bfloat16).float(), grad_tol["atol"], grad_tol["rtol"], "dv_part (per Q head, bf16)", keys=s, budget=1e-5
        )
        rms = lambda a, c: (a.double() - c.double()).norm().item() / max(c.double().norm().item(), 1e-300)  # noqa: E731
        dk_bshd, dv_bshd = v["dk"].view(b, s, g.h_kv, d), v["dv"].view(b, s, g.h_kv, d)  # the [B, S, H_kv, D] view the assertions above compare in
        print(
            f"{cell.id}: dK vs the once-rounded reference rel RMS {rms(dk_bshd, refs['dk']):.3g} (expected ~0: fp32 partials rounded once); "
            f"dV vs once-rounded {rms(dv_bshd, refs['dv']):.3g}, vs fold-modelled {rms(dv_bshd, refs['dv_fold']):.3g}"
        )
    # --- downstream of B4: the SEEDED oracle under the bf16 block's bound ------------------------------------------------
    ref = _oracle_seeded(res)
    flip_ev = _report_seeded_intermediates(res, v, ref)
    worst = {}
    for name in ("dh", "dw_o"):
        if res.grads[name] is not None:
            worst[name] = _assert_grad_close(res.grads[name], ref[name], f"{name} vs the seeded oracle")
    if res.grads["dw_qkvg"] is not None:
        worst["dw_qkvg"] = _assert_seeded_dw_qkvg_row_budgeted(res, v, ref, flip_ev, "dw_qkvg vs the seeded oracle")
    for name in ("dw_q_norm", "dw_k_norm"):
        if res.grads[name] is not None:
            worst[name] = _assert_dw_norm_close(res.grads[name], ref[name], ref[name + "_mass"], f"{name} vs the seeded oracle")
    print(f"{cell.id}: worst cells (fraction of the bound) {worst}")


def _assert_sdpa_stage_bitwise_the_rows_own_pre_pass(res, v: dict) -> None:
    """The block's dQ / dK / dV ``torch.equal`` a standalone ``SdpaBwdDslSm107Mxfp8(external_delta=False)`` run over the block's own
    payloads, blobs, LSE and the dead bf16 ports: on the MXFP8 row the external delta (the gate backward's ``rowsum(bf16 dO * bf16
    O)`` in ``dot_do_o``'s order) IS the row's own pre-pass -- the row's property, asserted at the block."""
    from cudnn.sdpa.bwd.api_dsl_sm107 import SdpaBwdDslSm107Mxfp8

    st = res.blk._sdpa
    impl = st._impl
    g, b, s, d = res.geom, res.batch, res.seq_len, res.geom.d_head
    kw = dict(
        sample_q=impl.q_desc,
        sample_k=impl.k_desc,
        sample_v=impl.v_desc,
        sample_o=impl.o_desc,
        sample_do=impl.do_desc,
        sample_stats=impl.stats_desc,
        sample_dq=impl.dq_desc,
        sample_dk=impl.dk_desc,
        sample_dv=impl.dv_desc,
        sample_q_T=impl.q_T_desc,
        sample_k_T=impl.k_T_desc,
        sample_do_T=impl.do_T_desc,
        sample_do_f16=impl.do_f16_desc,
        **{f"sample_{n}": impl.sf_descs[n] for n in st.sf_roles()},
        is_causal=bool(g.is_causal),
        causal_bottom_right=bool(g.causal_bottom_right),
        window_size_left=None if g.window_left < 0 else int(g.window_left),
        window_size_right=None if g.window_right < 0 else int(g.window_right),
        deterministic=False,
        scale_softmax=float(g.scale),
        seq_kv_lens_present=False,
        external_delta=False,
    )
    own = SdpaBwdDslSm107Mxfp8(**kw)
    own.check_support()
    own.compile()
    ws = torch.empty(own.scratch_workspace_bytes(), dtype=torch.uint8, device="cuda").fill_(0xFF)
    bshd = lambda x, h: x.view(b, s, h, d).transpose(1, 2)  # noqa: E731
    dq, dk, dv = (torch.full_like(v[n], float("nan")) for n in ("dq", "dk", "dv"))
    own.execute(
        bshd(v["q8"], g.h_q),
        bshd(v["k8"], g.h_kv),
        bshd(v["v8"], g.h_kv),
        res.saved.o.transpose(1, 2),
        bshd(v["do8"], g.h_q),
        res.saved.lse,
        bshd(dq, g.h_q),
        bshd(dk, g.h_kv),
        bshd(dv, g.h_kv),
        workspace=ws,
        current_stream=__import__("cuda.bindings.driver", fromlist=["CUstream"]).CUstream(torch.cuda.current_stream().cuda_stream),
        q_T_tensor=bshd(v["q_T8"], g.h_q),
        k_T_tensor=bshd(v["k_T8"], g.h_kv),
        do_T_tensor=bshd(v["do_T8"], g.h_q),
        do_f16_tensor=bshd(v["do"], g.h_q),
        **{n: v["sf"][n].view(*st.sf_shapes()[n]) for n in st.sf_roles()},
    )
    torch.cuda.synchronize()
    for name, got in (("dq", dq), ("dk", dk), ("dv", dv)):
        assert torch.equal(got, v[name]), f"{name}: the block's gradient is not bitwise the row's own pre-pass run over the same payloads"


@requires_rubin
@_MATRIX
def test_mxfp8_end_to_end_vs_the_oracles(cell):
    """End to end against (M) the fully MODELLED oracle with the fold modelled per gradient (dK once-rounded, dV per-Q-head bf16
    partials), (M) with a once-rounded fold (informational: the dV fold's cost in rows, visible per cell and per S) and (U) the
    unquantized-gradient STE oracle: ``cos``, ``max|diff| / max|ref|`` and the rows outside the bf16 bound against the row budget
    PRINTED per gradient; finiteness pinned.  The (M) fold-modelled one is asserted by the row-budget test once measured."""
    res = _cell_backward(cell)
    for name, ten in res.grads.items():
        if ten is not None:
            assert torch.isfinite(ten).all(), f"{name}: non-finite cells"
    keys = _row_keys(res)
    m = _print_end_to_end(f"{cell.id} (M fold-modelled)", res.grads, _oracle_m(res, "kernel"), keys=keys)
    m1 = _print_end_to_end(f"{cell.id} (M once-rounded)", res.grads, _oracle_m(res, "once"), keys=keys)
    u = _print_end_to_end(f"{cell.id} (U)", res.grads, _oracle_u(res), keys=keys)
    assert m and m1 and u


@requires_rubin
@_MATRIX
def test_mxfp8_end_to_end_modelled_is_row_budgeted(cell):
    """The (M) end-to-end in the ONE form named for it -- the bf16 block's bound with the SDPA stage's flip class propagated and
    budgeted by ROWS like ``assert_close_fp8_grad`` (``1e-5 x rows x keys``, at least 1) -- against the FOLD-MODELLED oracle on every
    produced bf16 output; PRINTED on the first Rubin run and ASSERTED once that run recorded its margins (``_M_ROW_BUDGET_ASSERTED``),
    never widened."""
    res = _cell_backward(cell)
    ref = _oracle_m(res, "kernel")
    m = _print_end_to_end(f"{cell.id} (M fold-modelled)", res.grads, ref, keys=_row_keys(res))
    assert m, cell.id
    over = {n: (m[n]["rows_outside"], m[n]["rows"], m[n]["row_budget"]) for n in _M_OUTPUTS if n in m and m[n]["rows_outside"] > m[n]["row_budget"]}
    print(f"{cell.id}: (M) outputs over the row budget: {over or 'none'} (asserted: {_M_ROW_BUDGET_ASSERTED})")
    if _M_ROW_BUDGET_ASSERTED:
        assert not over, f"{cell.id}: (M) rows outside the bf16 bound exceed the 1e-5 x rows x keys row budget (rows outside, rows, budget): {over}"


@requires_rubin
@_KNOB_SETS
def test_mxfp8_two_runs_are_bitwise(knobs):
    """Two executes of the SAME block over the same record / dy / artifacts -- the second into a workspace poisoned 0xFF and
    NaN-filled gradients -- are ``torch.equal`` on every gradient AND on the scalar block, and so is a FRESH block over the same
    record, under every knob set: no atomic anywhere on the MXFP8 chain (the dY amax is a max over per-CTA partials)."""
    res = _cell_backward(_BITWISE_CELL, **knobs)
    sb1 = _scalar_block(res).clone()
    ws2 = torch.empty_like(res.ws).fill_(0xFF)
    grads2 = _alloc_grads(res.blk, fill=float("nan"))
    _execute_mx(res.blk, res.inp, res.saved, res.dy, grads2, ws2, res.art, scale_dy=res.scale_dy_t)
    torch.cuda.synchronize()
    for name, ten in grads2.items():
        if ten is not None:
            assert torch.equal(ten, res.grads[name]), f"{name}: a second execute of the same block differs (knobs={knobs})"
    assert torch.equal(_view(ws2, res.blk._layout().quant_scalars, sb1.shape, torch.float32), sb1), "the scalar block differs between two executes of one block"
    blk, ws, grads = _twin_mx(res, **knobs)
    for name, ten in grads.items():
        if ten is not None:
            assert torch.equal(ten, res.grads[name]), f"{name}: two runs differ (knobs={knobs})"
    assert torch.equal(_view(ws, blk._layout().quant_scalars, sb1.shape, torch.float32), sb1), "the scalar block differs between two runs"


@requires_rubin
def test_mxfp8_fuse_wgrad_overlap_is_bitwise_the_in_order_block():
    """``fuse_wgrad_overlap=True`` is a scheduling knob: every gradient and the scalar block ``torch.equal`` the in-order block's; the
    side-stream GEMMs read ``alpha_b1`` and ``og8`` (B1, forked after the dual-axis dO quantize) and ``dqkvg_t8`` with its blob (B7,
    forked after the fused epilogue wrote it), all written on the launch stream before their fork events -- the scalar block is
    poisoned between the runs so a mis-placed fork cannot hide behind an equal value."""
    res = _cell_backward(_BITWISE_CELL)
    blk, ws, grads = _twin_mx(res, fuse_wgrad_overlap=True)
    for name, ten in grads.items():
        if ten is not None:
            assert torch.equal(ten, res.grads[name]), f"{name}: fuse_wgrad_overlap differs from the in-order block"
    assert torch.equal(_view(ws, blk._layout().quant_scalars, (len(_api_const("QUANT_SCALAR_SLOTS")),), torch.float32), _scalar_block(res))


@requires_rubin
def test_mxfp8_fuse_gate_bwd_is_inert():
    """The external delta is MANDATORY under an MxQuantSpec, so ``fuse_gate_bwd`` has no second arm: both values give ``torch.equal``
    gradients and scalar blocks; the stage's adapter is built with ``external_delta=True`` either way."""
    res = _cell_backward(_BITWISE_CELL)
    assert res.blk._sdpa._impl.external_delta is True
    blk, ws, grads = _twin_mx(res, fuse_gate_bwd=True)
    assert blk._sdpa._impl.external_delta is True
    for name, ten in grads.items():
        if ten is not None:
            assert torch.equal(ten, res.grads[name]), f"{name}: fuse_gate_bwd changed the gradients under the MXFP8 arm"


@requires_rubin
def test_mxfp8_delayed_dy_replays_current_bitwise():
    """A ``grad_scaling="delayed"`` block fed the "current" run's READ-BACK ``scale_dy`` gives ``torch.equal`` gradients and an equal
    scalar block: the two recipes are ONE code path with two writers of the one per-tensor scale of this pipeline."""
    res = _cell_backward(_BITWISE_CELL)
    blk, ws, grads = _twin_mx(res, grad_scaling="delayed", scale_dy=res.scalars["scale_dy"])
    for name, ten in grads.items():
        if ten is not None:
            assert torch.equal(ten, res.grads[name]), f"{name}: the delayed replay differs from the current run"
    assert torch.equal(_view(ws, blk._layout().quant_scalars, (len(_api_const("QUANT_SCALAR_SLOTS")),), torch.float32), _scalar_block(res))
    delayed_cell = _cell_backward(_BY_ID["s512_causal_b2_delayed-norm"])
    for name, ten in delayed_cell.grads.items():
        if ten is not None:
            assert torch.equal(ten, res.grads[name]), f"{name}: the delayed matrix cell differs from the current one"


_LAUNCH_CELL_IDS = [
    "s512_causal_b2-norm",
    "s512_causal_b2-rope_only",
    "s512_causal_b1_mha-norm",
    "s1024_dense_b1_mha-norm",
    "s992_causal_b1-norm",
    "s1008_causal_b2-norm",
    "s1000_causal_b1_dgrad_only-norm",
    "s992_causal_b1_mha-norm",
    "s384_causal_b1-norm",
    "s512_causal_b1_397b-norm",
]


@requires_rubin
@pytest.mark.parametrize("cell_id", _LAUNCH_CELL_IDS)
def test_mxfp8_launch_count_is_honest(cell_id):
    """CUPTI kernel records of one execute == the COMPUTED expectation (``mxfp8_expected_launches``: the block's rows by the cell's
    needs + the row's terms) == the formula recomputed from the ADAPTER's own facts (``mxfp8_launch_formula_from_facts``); no hidden
    memcpy and NO memset on the execute path (the scalar init and the row's fills are kernels, counted)."""
    from torch.profiler import ProfilerActivity, profile

    cell = _BY_ID[cell_id]
    res = _cell_backward(cell)
    blk = res.blk
    expected = mxfp8_expected_launches(cell)
    formula = mxfp8_launch_formula_from_facts(blk)
    grads = _alloc_grads(blk)
    _execute_mx(blk, res.inp, res.saved, res.dy, grads, res.ws, res.art, scale_dy=res.scale_dy_t)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        _execute_mx(blk, res.inp, res.saved, res.dy, grads, res.ws, res.art, scale_dy=res.scale_dy_t)
        torch.cuda.synchronize()
    names = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    if not names:
        pytest.skip("torch.profiler recorded no CUDA events (CUPTI unavailable on this node); the launch count is unverified here")
    memsets = [n for n in names if "memset" in n.lower()]
    memcpys = [n for n in names if "memcpy" in n.lower()]
    kernels = [n for n in names if n not in memsets and n not in memcpys]
    print(
        f"\n{cell_id}: {len(kernels)} kernels (formula {formula}, expected {expected}), {len(memsets)} memsets, {len(memcpys)} memcpys:\n  "
        + "\n  ".join(names)
    )
    assert not memcpys, f"a hidden copy on the execute path: {memcpys}"
    assert not memsets, f"a hidden memset on the execute path: {memsets}"
    assert formula == expected, (formula, expected)
    assert len(kernels) == expected, (len(kernels), expected, kernels)


@requires_rubin
@pytest.mark.parametrize("cell", _LAUNCH_ONLY_CELLS, ids=[c.id for c in _LAUNCH_ONLY_CELLS])
def test_mxfp8_launch_only_cells_are_finite_and_quantize_bitwise(cell):
    """The launch-count-only cells run the whole backward for their count -- this is their numerics layer, without an oracle:
    every gradient finite and the BITWISE layer of the matrix cells."""
    res = _cell_backward(cell)
    for name, ten in res.grads.items():
        if ten is not None:
            assert torch.isfinite(ten).all(), f"{cell.id}: {name} has non-finite cells"
    _assert_quantizers_scalars_delta_bitwise(res)


def _spread_forward(cell: _Cell):
    """An MXFP8 training forward over the cell's geometry on a dataset with a per-token / per-row POWER-OF-TWO spread -- ``h[t, :] *=
    2^((t % 8) - 4)``, ``W_qkvg[n, :] *= 2^((n % 8) - 4)`` on the bf16 inputs before their quantization -- so the E8M0 bytes of a
    blob over the matrix and over its transpose differ at most positions BY CONSTRUCTION (a token's own power of two vs the max
    over its 32-token block); a magnitude-uniform dataset gives both orientations near-identical bytes and makes the orientation
    numerically moot.  Returns the forward run (record + inputs + workspace), ``_run_training_quant``'s shape."""
    from test_block_training_forward import _execute_quant

    geom_kw, b, s = cell.geom_kw, cell.b, cell.s
    rg, geom = RefGeometry(**geom_kw), GatedAttentionBlockGeometry(**geom_kw)
    inp16 = dict(make_inputs(rg, batch=b, seq_len=s, dtype=torch.bfloat16))
    t, dm, n = b * s, geom.d_model, geom.n_qkvg
    t_spread = (2.0 ** ((torch.arange(t, device="cuda") % 8) - 4)).to(torch.float32)
    n_spread = (2.0 ** ((torch.arange(n, device="cuda") % 8) - 4)).to(torch.float32)
    inp16["h"] = (inp16["h"].reshape(t, dm).float() * t_spread[:, None]).to(torch.bfloat16).reshape(b, s, dm).contiguous()
    inp16["w_qkvg"] = (inp16["w_qkvg"].float() * n_spread[:, None]).to(torch.bfloat16).contiguous()
    inp_q, desc = quantize_block_inputs_mxfp8(inp16)
    spec = MxQuantSpec(**desc, scale_o=mxfp8_calibrated_scale_o(inp_q, rg))
    out = torch.empty(b, s, dm, device="cuda", dtype=torch.bfloat16)
    blk = GatedAttentionBlockFwd(
        inp_q["h"], inp_q["w_qkvg"], inp_q["w_q_norm"], inp_q["w_k_norm"], inp_q["cos"], inp_q["sin"], inp_q["w_o"], out, geom,
        quant=spec, sample_h_sf=inp_q["h_sf"], sample_w_qkvg_sf=inp_q["w_qkvg_sf"], save_for_backward=True,
    )  # fmt: skip
    r = SimpleNamespace(blk=blk, inp16=inp16, inp=inp_q, spec=spec, out=out, family="mxfp8", geom=geom, geom_kw=geom_kw, batch=b, seq_len=s)
    r.saved = _alloc_saved(geom, inp_q, b, s, save_mode="proj_slab", act_dtype=torch.bfloat16)
    blk.check_support()
    blk.compile()
    r.ws = torch.empty(blk.get_workspace_size(), dtype=torch.uint8, device="cuda")
    _execute_quant(r, r.ws, saved=r.saved)
    torch.cuda.synchronize()
    return r


@requires_rubin
@_MATRIX
def test_mxfp8_wrong_orientation_blob_fails_the_gemm_bound(cell):
    """The orientation guard, on EVERY accept cell and on the pin's OWN spread dataset: the byte count of a scale-factor blob is
    symmetric in ``(rows, K)``, so the forward's ``h_sf`` handed as ``h_t_sf`` (and ``w_qkvg_sf`` as ``w_qkvg_t_sf``) passes every host
    check -- and the B7 / B8 GEMM bound, whose REFERENCE dequantizes the artifact through the CORRECT blob while only the block
    was handed the wrong one, must FAIL; the control (the right artifacts) passes the same bound first.  Asserted: the two blobs
    are not ``torch.equal`` (else nothing could fail), the control passes, the wrong blob fails (``pytest.raises(AssertionError)``)."""
    fwd = _spread_forward(cell)
    res = _backward_mxfp8(cell.geom_kw, cell.b, cell.s, memo=False, fwd=fwd, **cell.bwd_kw)
    blk, g, art = res.blk, res.geom, res.art
    t = res.batch * res.seq_len
    v = _slots(res)
    checked = False
    if blk.need_dw_qkvg:
        wrong = fwd.inp["h_sf"]
        assert wrong.numel() == art["h_t_sf"].numel() and not torch.equal(
            wrong, art["h_t_sf"]
        ), "the un-transposed blob must have the same byte count and different bytes"
        a64 = mx_dequant_rowwise_2d(v["dqkvg_t8"], v["sf_dqkvg_t"]).double()
        b64 = mx_dequant_rowwise_2d(art["h_t"], art["h_t_sf"]).double()  # the CORRECT blob, in the reference
        _gemm_bound(res.grads["dw_qkvg"], a64 @ b64.t(), f"{cell.id}: B7 with the right h_t_sf (control)")
        _blk, _ws, grads = _twin_mx(res, art={**art, "h_t_sf": wrong})
        with pytest.raises(AssertionError):
            _gemm_bound(grads["dw_qkvg"], a64 @ b64.t(), f"{cell.id}: B7 with the forward's h_sf as h_t_sf (must FAIL)")
        checked = True
    if blk.need_dh:
        wrong = fwd.inp["w_qkvg_sf"]
        assert wrong.numel() == art["w_qkvg_t_sf"].numel() and not torch.equal(wrong, art["w_qkvg_t_sf"])
        a64 = mx_dequant_rowwise_2d(v["dqkvg8"], v["sf_dqkvg"]).double()
        b64 = mx_dequant_rowwise_2d(art["w_qkvg_t"], art["w_qkvg_t_sf"]).double()
        _gemm_bound(res.grads["dh"].view(t, g.d_model), a64 @ b64.t(), f"{cell.id}: B8 with the right w_qkvg_t_sf (control)")
        _blk, _ws, grads = _twin_mx(res, art={**art, "w_qkvg_t_sf": wrong})
        with pytest.raises(AssertionError):
            _gemm_bound(grads["dh"].view(t, g.d_model), a64 @ b64.t(), f"{cell.id}: B8 with the forward's w_qkvg_sf as w_qkvg_t_sf (must FAIL)")
        checked = True
    assert checked


_DY_SHIFT = 13


@requires_rubin
@_MATRIX
def test_mxfp8_backward_is_bitwise_equivariant_under_a_power_of_two_dy_scaling(cell):
    """A power-of-two scaling of dY (``2^-13``) leaves the whole chain exactly equivariant: ``scale_dy`` absorbs it (``dy8`` bitwise),
    every dY-independent payload and blob (q / k / v) is bitwise, every dY-dependent block quantization keeps its e4m3 CODES bitwise
    while its E8M0 bytes shift by exactly 13 (a power of two moves the exponent; zero blocks and pad bytes stay 0), the fp32 sums and
    bf16 stores scale exactly -- so every gradient is bitwise ``2^-13 x`` the unit run's and the scalar block's live slots scale by the
    power of two.  Pins the on-device ``scale_dy`` and the dynamic range of every cast on the chain."""
    res = _cell_backward(cell)
    f = 2.0**-_DY_SHIFT
    dy2 = (res.dy.float() * f).to(torch.bfloat16)
    assert torch.equal(dy2.float(), res.dy.float() * f), "the scaled dy must be exact in bf16"
    blk = _declare_mx_bwd(dy2, res.saved, res.inp, res.geom, quant=res.spec, grad_scaling=res.grad_scaling, **cell.bwd_kw)
    blk.check_support()
    blk.compile()
    ws = torch.empty(blk.get_workspace_size(), dtype=torch.uint8, device="cuda").fill_(0xFF)
    grads = _alloc_grads(blk, fill=float("nan"))
    sdy = _dev_scalar(res.scalars["scale_dy"] * 2.0**_DY_SHIFT) if res.grad_scaling == "delayed" else None
    _execute_mx(blk, res.inp, res.saved, dy2, grads, ws, res.art, scale_dy=sdy)
    torch.cuda.synchronize()
    res2 = SimpleNamespace(
        blk=blk,
        ws=ws,
        geom=res.geom,
        batch=res.batch,
        seq_len=res.seq_len,
        saved=res.saved,
        inp=res.inp,  # the rebuild reference of `_slots` (the fused chain carves no bf16 Q / K) reads the norm weights and cos / sin from it
        scalars={k: float(x.item()) for k, x in blk.quant_scalars(ws).items()},
    )
    v1, v2 = _slots(res), _slots(res2)
    assert torch.equal(v2["dy8"].view(torch.uint8), v1["dy8"].view(torch.uint8)), "dy8: scale_dy did not absorb the power of two"
    for name in ("q8", "q_T8", "k8", "k_T8", "v8"):
        assert torch.equal(v2[name].view(torch.uint8), v1[name].view(torch.uint8)), f"{name}: a dY-independent payload changed"
    for name in ("sf_q", "sf_q_T", "sf_k", "sf_k_T", "sf_v"):
        assert torch.equal(v2["sf"][name], v1["sf"][name]), f"{name}: a dY-independent blob changed"

    def shifted(unit: torch.Tensor, scaled: torch.Tensor, what: str) -> None:
        live = unit != 0
        assert torch.equal(scaled[~live], torch.zeros_like(scaled[~live])), f"{what}: a zero / pad scale byte became non-zero"
        diff = unit[live].int() - scaled[live].int()
        assert bool((diff == _DY_SHIFT).all()), f"{what}: E8M0 bytes did not shift by exactly {_DY_SHIFT} ({int((diff != _DY_SHIFT).sum())} bytes off)"

    for pay, sf_name in (("do8", "sf_do"), ("do_T8", "sf_do_T")):
        assert torch.equal(v2[pay].view(torch.uint8), v1[pay].view(torch.uint8)), f"{pay}: the e4m3 codes changed under a power-of-two dY"
        shifted(v1["sf"][sf_name], v2["sf"][sf_name], sf_name)
    if blk.need_dh:
        assert torch.equal(v2["dqkvg8"].view(torch.uint8), v1["dqkvg8"].view(torch.uint8)), "dqkvg8: the codes changed"
        shifted(v1["sf_dqkvg"], v2["sf_dqkvg"], "sf_dqkvg")
    if blk.need_dw_qkvg:
        assert torch.equal(v2["dqkvg_t8"].view(torch.uint8), v1["dqkvg_t8"].view(torch.uint8)), "dqkvg_t8: the codes changed"
        shifted(v1["sf_dqkvg_t"], v2["sf_dqkvg_t"], "sf_dqkvg_t")
    if v1["og8"] is not None:
        assert torch.equal(v2["og8"].view(torch.uint8), v1["og8"].view(torch.uint8))
    for name, ten in grads.items():
        if ten is not None:
            want = (res.grads[name].float() * f).to(ten.dtype)
            assert torch.equal(ten, want), f"{name}: not bitwise 2^-{_DY_SHIFT} x the unit run's"
    sc1, sc2 = res.scalars, res2.scalars
    assert sc2["amax_dy"] == sc1["amax_dy"] * f and sc2["scale_dy"] == sc1["scale_dy"] * 2.0 ** _DY_SHIFT and sc2["descale_dy"] == sc1["descale_dy"] * f
    assert sc2["alpha_b1"] == sc1["alpha_b1"] * f and sc2["alpha_b2"] == sc1["alpha_b2"] * f
    for name in _dead_slots():
        assert sc2[name] == 0.0, name


@requires_rubin
def test_mxfp8_cuda_graph_capture_replays_bitwise():
    """One ``execute`` captured into a CUDA graph on a side torch stream replays bitwise the eager run and recomputes over a NEW
    ``dy`` and NEW artifact bytes written IN PLACE through the captured pointers (the scalar init, the amax partials, every block
    quantize and the block-scale GEMMs are device work on the launch stream: capturable, no host readback).  The capture itself
    launches nothing; the block allocates nothing."""
    res = _cell_backward(_BY_ID["s256_causal_b1-norm"])
    blk = res.blk
    dy2 = res.dy.clone()
    art2 = {k: v.clone() for k, v in res.art.items()}
    ws = torch.empty_like(res.ws)
    grads = _alloc_grads(blk, fill=float("nan"))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _execute_mx(blk, res.inp, res.saved, dy2, grads, ws, art2)  # warm-up on the capture stream
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    for ten in grads.values():
        if ten is not None:
            ten.fill_(float("nan"))
    ws.fill_(0xFF)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=stream):
            _execute_mx(blk, res.inp, res.saved, dy2, grads, ws, art2)
        torch.cuda.synchronize()
        for name, ten in grads.items():
            if ten is not None:
                assert torch.isnan(ten).all(), f"{name}: the capture launched work"
        graph.replay()
        torch.cuda.synchronize()
        for name, ten in grads.items():
            if ten is not None:
                assert torch.equal(ten, res.grads[name]), f"{name}: the replay differs from the eager run"
        # new inputs through the captured pointers: a new dy, and the artifacts of a re-spread h (new codes AND new blob bytes)
        dy3 = _make_dy(res.out, seed=7)
        inp16b = dict(res.inp16)
        inp16b["h"] = (inp16b["h"].float() * 0.5).to(torch.bfloat16)
        art3 = _artifacts_of(inp16b, res.geom)
        dy2.copy_(dy3)
        for k in art2:
            art2[k].copy_(art3[k])
        ws.fill_(0xFF)
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        ref = _alloc_grads(blk, fill=float("nan"))
        ws_ref = torch.empty_like(ws).fill_(0xFF)
        _execute_mx(blk, res.inp, res.saved, dy3, ref, ws_ref, art3)
        torch.cuda.synchronize()
        for name, ten in grads.items():
            if ten is not None:
                assert torch.isfinite(ten).all() and torch.equal(ten, ref[name]), f"{name}: the replay over new inputs differs from eager"
    finally:  # a graph left to the cyclic GC resets itself inside a later test's capture
        graph.reset()


@requires_rubin
@_KNOB_SETS
def test_mxfp8_workspace_size_is_honest(knobs):
    """``get_workspace_size()`` is exact and never exceeded: the carve (every MXFP8 region present, 256-B aligned: the payloads, the
    blobs, the scalar block, the dY partials; NO bf16 rebuild slots -- the fused prologue's MX epilogue writes the four Q / K payloads
    out of registers, ``mx_prologue_arm == "mx_epilogue"`` --; no dO / dG / band partials) + the MXFP8 adapter's scratch (no
    ``delta`` region: the block's own one takes its place) + the GEMM scratch (the max over the four K64 plans: two per-tensor with
    an alpha, two block-scale without); a buffer 4096 B larger keeps its tail untouched; two executes allocate nothing; every e4m3
    payload and every scale-factor blob is WRITTEN in full (no 0xFF byte survives: 0xFF is the e4m3 NaN and the E8M0 NaN, which a
    finite source never produces -- a pad byte is 0x00), and so are the fp32 ``delta`` region and the scalar block's slots."""
    from cudnn.gated_attention_block.api import _WS_ALIGN, _sf_slot_bytes

    res = _cell_backward(_BITWISE_CELL, **knobs)
    blk = res.blk
    size = blk.get_workspace_size()
    lay = blk._layout()
    assert size == lay.total_bytes and size % _WS_ALIGN == 0
    plans = blk.gemm_plans
    assert len(plans) == 4 and all(p.mma_tile_k_bytes == 64 for p in plans.values())
    assert plans["out_proj_dgrad"].has_alpha and plans["out_proj_wgrad"].has_alpha and not plans["out_proj_dgrad"].block_scale
    assert plans["qkv_gate_wgrad"].block_scale and plans["qkv_gate_dgrad"].block_scale and not plans["qkv_gate_wgrad"].has_alpha
    assert lay.gemm_scratch_bytes == max(p.workspace_bytes for p in plans.values()) >= 1
    assert lay.sdpa_bwd_bytes == blk._sdpa.scratch_workspace_bytes()
    assert lay.quant_scalars >= 0 and lay.quant_scalars % 256 == 0 and lay.delta >= 0 and lay.o_gated == -1 and lay.recompute_v == -1
    assert blk.mx_prologue_arm == blk._prologue.arm == "mx_epilogue"
    assert (
        lay.recompute == -1 and lay.recompute_k == -1
    ), "no bf16 rebuild region under the MX-epilogue prologue arm (the payloads are written out of registers)"
    assert (
        lay.amax_partials >= 0 and lay.amax_partials % _WS_ALIGN == 0 and lay.amax_partials_n == blk._prologue.n_partials_cap >= blk._prologue.n_partials() >= 1
    )
    assert (lay.amax_partials_do, lay.amax_partials_dg, lay.gate_partials_n, lay.amax_partials_bands, lay.band_partials_n) == (-1, -1, 0, -1, 0)
    g, b, s = blk.geom, blk.batch, blk.seq_len
    t, d, n = b * s, g.d_head, g.n_qkvg
    regions = dict(
        dy8=(lay.dy8, t * g.d_model),
        do8=(lay.do8, t * g.h_q * d),
        do_T8=(lay.do_T8, t * g.h_q * d),
        q8=(lay.q8, t * g.h_q * d),
        q_T8=(lay.q_T8, t * g.h_q * d),
        k8=(lay.k8, t * g.h_kv * d),
        k_T8=(lay.k_T8, t * g.h_kv * d),
        v8=(lay.v8, t * g.h_kv * d),
        dqkvg8=(lay.dqkvg8, t * n),
        dqkvg_t8=(lay.dqkvg_t8, n * t),
        sf_dqkvg=(lay.sf_dqkvg, sf_blob_bytes(t, n)),
        sf_dqkvg_t=(lay.sf_dqkvg_t, sf_blob_bytes(n, t)),
    )
    for name, heads in (("sf_q", g.h_q), ("sf_q_T", g.h_q), ("sf_k", g.h_kv), ("sf_k_T", g.h_kv), ("sf_v", g.h_kv), ("sf_do", g.h_q), ("sf_do_T", g.h_q)):
        regions[name] = (getattr(lay, name), _sf_slot_bytes(b, heads, s, d))
    if lay.og8 >= 0:
        regions["og8"] = (lay.og8, t * g.h_q * d)
    for name, (off, nbytes) in regions.items():
        assert off >= 0 and off % _WS_ALIGN == 0, name
    ws = torch.full((size + 4096,), 0xFF, dtype=torch.uint8, device="cuda")
    grads = _alloc_grads(blk)
    _execute_mx(blk, res.inp, res.saved, res.dy, grads, ws[:size], res.art, scale_dy=res.scale_dy_t)
    torch.cuda.synchronize()
    gc.collect()
    live = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    n0 = torch.cuda.memory_stats()["allocation.all.allocated"]
    _execute_mx(blk, res.inp, res.saved, res.dy, grads, ws[:size], res.art, scale_dy=res.scale_dy_t)
    _execute_mx(blk, res.inp, res.saved, res.dy, grads, ws[:size], res.art, scale_dy=res.scale_dy_t)
    torch.cuda.synchronize()
    n1 = torch.cuda.memory_stats()["allocation.all.allocated"]
    peak = torch.cuda.max_memory_allocated()
    assert n1 == n0, f"the MXFP8 backward made {n1 - n0} CUDA allocation(s) on the execute path (allocation.all.allocated {n0} -> {n1})"
    assert peak <= live, f"a temporary on the MXFP8 backward's execute path: the allocator peak rose from {live} to {peak} bytes"
    assert torch.equal(ws[size:], torch.full((4096,), 0xFF, dtype=torch.uint8, device="cuda")), "bytes past get_workspace_size() were written"
    for name, ten in grads.items():
        if ten is not None:
            assert torch.equal(ten, res.grads[name]), name
    sb = _view(ws[:size], lay.quant_scalars, (len(_api_const("QUANT_SCALAR_SLOTS")),), torch.float32)
    assert torch.isfinite(sb).all(), "a scalar slot was never written (0xFF = NaN)"
    for name, (off, nbytes) in regions.items():
        survivors = int((ws[off : off + nbytes] == 0xFF).sum())
        assert survivors == 0, f"{name}: {survivors} of {nbytes} bytes still hold the 0xFF poison -- never written"
    # (no bf16 rebuild region to check: the four Q / K payloads above ARE the rebuild's product, every byte of them written)
    delta = _view(ws[:size], lay.delta, tuple(blk._sdpa.delta_shape), torch.float32)
    assert torch.isfinite(delta).all(), "a delta element (pad rows included) was never written (0xFFFFFFFF = NaN)"


@requires_rubin
@_MATRIX
def test_mxfp8_amax_times_scale_never_exceeds_448(cell):
    """The one published (amax, scale) pair of this pipeline -- ``amax_dy * scale_dy <= 448`` by the in-kernel formula for the amax the
    pass read, and the scale the largest such power of two (margin 0); every block quantization's own scale is per block (no pair
    to check: its E8M0 is ``ceil(log2(amax / 448))`` by construction, pinned bitwise)."""
    res = _cell_backward(cell)
    sc = res.scalars
    prod = sc["amax_dy"] * sc["scale_dy"]
    print(f"{cell.id}: amax_dy * scale_dy = {prod:.4g}")
    assert prod <= FP8_E4M3_MAX
    if res.grad_scaling == "current":
        assert sc["amax_dy"] * (2.0 * sc["scale_dy"]) > FP8_E4M3_MAX or sc["amax_dy"] == 0.0, "the scale is not the largest power of two (margin 0)"


@requires_rubin
@pytest.mark.parametrize("how", ["ambient", "explicit"])
def test_mxfp8_a_caller_stream_orders_every_stage(how):
    """Every stage -- the fused prologue, the dY quantize, the dual-axis dO quantize, the four GEMMs, the fused epilogue, the MXFP8
    SDPA adapter -- launches on ONE stream, the caller's: ambient (``with torch.cuda.stream(s):``) or explicit (``current_stream=``).  The default
    stream is parked behind a long spin and the workspace is zeroed on the side stream right after the block, so a stage enqueued
    on the default stream runs late and the gradients differ from the default-stream run -- which they must equal BITWISE."""
    import cuda.bindings.driver as cuda_drv

    res = _cell_backward(_BY_ID["s256_causal_b1-norm"])
    side = torch.cuda.Stream()
    ws2 = torch.zeros_like(res.ws)
    grads2 = _alloc_grads(res.blk, fill=0)
    torch.cuda.synchronize()
    park_the_default_stream()
    with torch.cuda.stream(side):
        dy2 = res.dy.clone()
        art2 = {k: v.clone() for k, v in res.art.items()}
        cs = None if how == "ambient" else cuda_drv.CUstream(side.cuda_stream)
        _execute_mx(res.blk, res.inp, res.saved, dy2, grads2, ws2, art2, current_stream=cs)
        ws2.zero_()
    torch.cuda.synchronize()
    for name, ten in grads2.items():
        if ten is not None:
            assert torch.equal(ten, res.grads[name]), f"{name}: a stage escaped the caller's stream ({how})"


@requires_rubin
def test_mxfp8_convenience_wrapper_matches_the_class():
    """``gated_attention_block_backward(..., quant=MxQuantSpec, h_t=, h_t_sf=, w_qkvg_t=, w_qkvg_t_sf=)`` allocates, caches the compiled
    block (``quant``, ``grad_scaling`` and the artifacts' PRESENCE in the key) and delegates: its outputs are ``torch.equal`` the class
    path's, and the per-call workspace's views are released at return."""
    res = _cell_backward(_BY_ID["s256_causal_b1-norm"])
    inp, saved = res.inp, res.saved
    for t_ in (saved.h, inp["w_qkvg"], inp["w_o"], inp["w_q_norm"], inp["w_k_norm"]):
        t_.requires_grad_(True)
    try:
        out = gated_attention_block_backward(
            res.dy, saved, inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], res.geom, quant=res.spec, **res.art
        )  # fmt: skip
        torch.cuda.synchronize()
        for name in ("dh", "dw_qkvg", "dw_o", "dw_q_norm", "dw_k_norm"):
            assert torch.equal(out[name], res.grads[name]), name
        cached = [bk for bk in _api_bwd._BWD_CACHE.values() if isinstance(bk.quant, MxQuantSpec) and bk.batch == res.batch and bk.seq_len == res.seq_len]
        assert cached and all(bk._ws_views is None for bk in cached), "the wrapper left the per-call workspace's views cached"
    finally:
        for t_ in (saved.h, inp["w_qkvg"], inp["w_o"], inp["w_q_norm"], inp["w_k_norm"]):
            t_.requires_grad_(False)


@requires_rubin
def test_mxfp8_execute_contracts_are_typed():
    """On a COMPILED MXFP8 block (execute refuses an uncompiled block first), the appended inputs both directions (Rule 1), each a
    ``ValueError`` naming the attribute before any launch: ``scale_dp`` / ``scale_do`` / ``scale_dqkvg`` given; ``scale_dy`` given under
    "current" and missing under "delayed"; an artifact missing; a ``.t()`` VIEW of the un-transposed codes as ``h_t`` / ``w_qkvg_t``;
    a wrong blob byte count; a bf16 ``h_t``."""
    res = _cell_backward(_BY_ID["s256_causal_b1-norm"])
    blk, inp, saved, dy, ws, art = res.blk, res.inp, res.saved, res.dy, res.ws, res.art
    g, t = blk.geom, blk.batch * blk.seq_len
    grads = _alloc_grads(blk)
    run = lambda **kw: _execute_mx(blk, inp, saved, dy, grads, ws, art, **kw)  # noqa: E731
    for name in ("scale_dp", "scale_do", "scale_dqkvg"):
        with pytest.raises(ValueError, match=name):
            blk.execute(
                dy,
                saved,
                inp["w_qkvg"],
                inp["w_q_norm"],
                inp["w_k_norm"],
                inp["cos"],
                inp["sin"],
                inp["w_o"],
                workspace=ws,
                **grads,
                **art,
                **{name: _dev_scalar(1.0)},
            )
    with pytest.raises(ValueError, match="scale_dy"):
        run(scale_dy=_dev_scalar(1.0))
    for name in _ARTIFACTS:
        with pytest.raises(ValueError, match=name):
            run(**{name: None})
    with pytest.raises(ValueError, match="h_t"):
        run(h_t=inp["h"].reshape(t, g.d_model).t())
    with pytest.raises(ValueError, match="w_qkvg_t"):
        run(w_qkvg_t=inp["w_qkvg"].t())
    with pytest.raises(ValueError, match="h_t_sf"):
        run(h_t_sf=art["h_t_sf"][:-512])
    with pytest.raises(ValueError, match="h_t"):
        run(h_t=art["h_t"].to(torch.bfloat16))
    delayed = _cell_backward(_BY_ID["s512_causal_b2_delayed-norm"])
    with pytest.raises(ValueError, match="scale_dy"):
        _execute_mx(delayed.blk, delayed.inp, delayed.saved, delayed.dy, _alloc_grads(delayed.blk), delayed.ws, delayed.art)


@requires_rubin
def test_mxfp8_sf_order_s_sweep_does_not_degrade_with_s():
    """The scale-factor layout detector on the BACKWARD's cosines at B = 1, dense, S in {128, 256, 384, 512}: a cosine DEGRADING with S
    is a D-plane-major blob read per tile (the plane stride grows with S), a CONSTANT offset an atom bug.  Every S must pass the gate
    (``cos >= 0.99``) against the fold-modelled (M) oracle on every produced gradient AND the sweep must be flat within 5e-3."""
    cs = {}
    for s in (128, 256, 384, 512):
        res = _backward_mxfp8({**_COMMON, "qk_norm": True, "is_causal": False}, 1, s)
        ref = _oracle_m(res, "kernel")
        cs[s] = {name: _cos(res.grads[name], ref[name]) for name in ("dh", "dw_qkvg", "dw_o")}
    print("\nmxfp8 backward SF-order S-sweep (B=1 dense): " + "  ".join(f"S={s}: " + " ".join(f"{k}={c:.6f}" for k, c in d.items()) for s, d in cs.items()))
    for name in ("dh", "dw_qkvg", "dw_o"):
        vals = [cs[s][name] for s in cs]
        assert all(c > 0.99 for c in vals), (name, vals)
        assert max(vals) - min(vals) < 5e-3, f"{name}: the cosine drifts with S -- a scale-factor layout bug: {vals}"


@requires_rubin
def test_mxfp8_quant_scalars_hold_the_plan_time_constants():
    """After an execute the scalar block holds the MXFP8 arm's eight live slots and 21 exact zeros: the three live constants
    (``scale_o``, ``descale_o``, ``descale_w_o``) at the fp32 RN of the MxQuantSpec's values, the eleven other constants 0.0; the five live
    DYNAMIC slots (``amax_dy / scale_dy / descale_dy / alpha_b1 / alpha_b2``) finite and bitwise the kernel's formulas from the read-back
    amax, the ten dead dynamic slots 0.0."""
    res = _cell_backward(_BITWISE_CELL)
    sc, vals, sp = res.scalars, res.blk._quant_const_values(), res.spec
    consts = _api_const("QUANT_CONST_SLOTS")
    for name in consts:
        assert sc[name] == _f32(vals[name]), (name, sc[name], vals[name])
    assert sc["scale_o"] == _f32(sp.scale_o) and sc["descale_o"] == _f32(1.0 / sp.scale_o) and sc["descale_w_o"] == _f32(sp.descale_w_o)
    assert all(sc[n] == 0.0 for n in consts if n not in _MXFP8_LIVE_CONSTS)
    dynamic = [n for n in _api_const("QUANT_SCALAR_SLOTS") if n not in consts]
    live_dyn = [n for n in _MXFP8_LIVE_SLOTS if n in dynamic]
    assert len(live_dyn) == 5 and len(dynamic) == 15
    assert all(np.isfinite(sc[n]) and sc[n] != 0.0 for n in live_dyn), {n: sc[n] for n in live_dyn}
    assert sc["scale_dy"] == _grad_scale(sc["amax_dy"]) and sc["descale_dy"] == _f32(1.0 / sc["scale_dy"])
    assert sc["alpha_b1"] == _f32_mul(sc["descale_dy"], sc["descale_o"]) and sc["alpha_b2"] == _f32_mul(sc["descale_dy"], sc["descale_w_o"])
    assert all(sc[n] == 0.0 for n in dynamic if n not in live_dyn), {n: sc[n] for n in dynamic if n not in live_dyn}
    assert sum(1 for n in _api_const("QUANT_SCALAR_SLOTS") if sc[n] != 0.0) == 8


def _device_records_of(prof, path) -> tuple:
    """``(records, streams)`` of a profile's device-side activity -- kernels, memcpys, memsets -- read from the exported chrome trace."""
    import json

    prof.export_chrome_trace(str(path))
    with open(path) as f:
        events = json.load(f)["traceEvents"]
    device = [e for e in events if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    return [(e["cat"], e["name"]) for e in device], {e.get("args", {}).get("stream") for e in device}


@requires_rubin
@pytest.mark.parametrize("how", ["class", "wrapper"])
def test_mxfp8_first_use_on_an_explicit_stream_reads_nothing_the_ambient_stream_wrote(how, tmp_path):
    """A FRESH block (compiled inside the test) executed for the FIRST time on an explicit non-blocking stream that is NOT the
    ambient one: the plan-time constants are slots the standalone scalar-init launch stores from its kernel arguments ON THE
    EXECUTION STREAM -- ``compile()`` records NO device-side activity at all, and every device-side record of the first ``execute``
    ran on ONE stream, with every gradient and the whole scalar block bitwise the synchronised run's; the WRAPPER: a cache-miss call
    compiles inside, and every device-side record of the whole call ran on that one stream."""
    import cuda.bindings.driver as cuda_drv
    from torch.profiler import ProfilerActivity, profile

    res = _cell_backward(_BY_ID["s256_causal_b1-norm"])
    inp, saved, art = res.inp, res.saved, res.art
    side = torch.cuda.Stream()
    ambient = torch.cuda.default_stream()
    assert torch.cuda.current_stream() == ambient and side != ambient
    cs = cuda_drv.CUstream(side.cuda_stream)
    torch.cuda.synchronize()
    if how == "class":
        blk = _declare_mx_bwd(res.dy, saved, inp, res.geom, quant=res.spec)
        blk.check_support()
        ws = torch.empty_like(res.ws).fill_(0xFF)
        grads = _alloc_grads(res.blk, fill=float("nan"))
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            blk.compile()
            torch.cuda.synchronize()
        compile_records, _ = _device_records_of(prof, tmp_path / "compile.json")
        assert torch.cuda.memory_allocated() == before, "compile() allocated a CUDA tensor"
        assert blk.get_workspace_size() == ws.numel()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            _execute_mx(blk, inp, saved, res.dy, grads, ws, art, current_stream=cs)
            side.synchronize()
        records, streams = _device_records_of(prof, tmp_path / "first_use_class.json")
        if not records:
            pytest.skip("torch.profiler recorded no CUDA events (CUPTI unavailable on this node); the compile-enqueues-nothing pin is unverified here")
        assert compile_records == [], f"compile() enqueued device work -- a fill an execute on another stream could race: {compile_records}"
        assert len(streams) == 1, f"the first execute ran on {len(streams)} streams ({streams}): device work escaped the execution stream -- {records}"
        got_block = _view(ws, blk._layout().quant_scalars, (len(_api_const("QUANT_SCALAR_SLOTS")),), torch.float32)
    else:
        for t_ in (saved.h, inp["w_qkvg"], inp["w_o"], inp["w_q_norm"], inp["w_k_norm"]):
            t_.requires_grad_(True)
        kept = dict(_api_bwd._BWD_CACHE)
        _api_bwd._BWD_CACHE.clear()
        try:
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                out = gated_attention_block_backward(
                    res.dy, saved, inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], res.geom, current_stream=cs, quant=res.spec, **art
                )  # fmt: skip
                side.synchronize()
            grads = {k: out[k] for k in ("dh", "dw_qkvg", "dw_o", "dw_q_norm", "dw_k_norm")}
            got_block = None
        finally:
            _api_bwd._BWD_CACHE.clear()
            _api_bwd._BWD_CACHE.update(kept)
            for t_ in (saved.h, inp["w_qkvg"], inp["w_o"], inp["w_q_norm"], inp["w_k_norm"]):
                t_.requires_grad_(False)
        records, streams = _device_records_of(prof, tmp_path / "first_use_wrapper.json")
        if not records:
            pytest.skip("torch.profiler recorded no CUDA events (CUPTI unavailable on this node); the one-stream pin is unverified here")
        assert len(streams) == 1, f"the first use ran on {len(streams)} streams ({streams}): device work escaped the execution stream -- {records}"
    torch.cuda.synchronize()
    for name, ten in grads.items():
        if ten is not None:
            assert torch.equal(ten, res.grads[name]), f"{name}: the first use on the explicit stream differs from the synchronised run ({how})"
    if got_block is not None:
        assert torch.equal(got_block, _scalar_block(res)), "the scalar block (the plan-time constants included) differs from the synchronised run"


@requires_rubin
def test_mxfp8_dense_tail_has_no_record():
    """A dense ``S % 128 != 0`` has no MXFP8 record to run a backward over: the quantized forward declines it typed."""
    _dense_tail_declined({**_COMMON, "qk_norm": True, "is_causal": False}, 1, 992, "mxfp8")


# ---------------------------------------------------------------------------
# REJECT (any CUDA device; a DECLARED block, no compile, no launch) -- attribute names only
# ---------------------------------------------------------------------------


def _declare_then_check(make):
    make().check_support()


@requires_cuda
def test_mxfp8_rejects_match_the_attribute_names():
    """Every typed decline of the MXFP8 backward matched by ATTRIBUTE NAME (the prose is pinned in ``test_block_backward.py``):
    ``thd`` with an MxQuantSpec, an e4m3 weight under the fp4 weight modes (``w_qkvg`` under an e2m1 ``w_qkvg_dtype``, ``w_o`` under ``o_fp4``:
    the modes themselves construct -- their backward is ``test_block_backward_fp4.py``'s), e5m2 codes, an fp16 ``dy``, bf16 weights with an
    MxQuantSpec, the ``B*S % 32`` rule under ``need_dw_qkvg`` (served without), padding, ``fuse_wgrad_overlap`` without a wgrad, ``need_*``
    both ways."""
    from cudnn.gated_attention_block.api import Fp4Format

    b, s = 1, 256
    with pytest.raises(ValueError, match="thd"):
        _declare_bwd_mxfp8(
            dict(_COMMON),
            1,
            256,
            thd=True,
            num_sequences=2,
            max_seq_len=128,
            saved_replace=dict(seq_lens=torch.tensor([128, 128], dtype=torch.int32, device="cuda"), seq_lens_form="lengths"),
        )
    with pytest.raises(ValueError, match="w_qkvg"):  # the MXFP4 weight mode constructs; its e4m3 w_qkvg is the weight-dtype decline
        _declare_then_check(lambda: _declare_bwd_mxfp8(dict(_COMMON), b, s, spec=MxQuantSpec(descale_w_o=0.125, w_qkvg_dtype=torch.float4_e2m1fn_x2)).blk)
    with pytest.raises(ValueError, match="w_o"):  # the fp4 O mode constructs; its e4m3 w_o is the weight-dtype decline
        _declare_then_check(lambda: _declare_bwd_mxfp8(dict(_COMMON), b, s, spec=MxQuantSpec(descale_w_o=1.0, scale_o=1.0, o_fp4=Fp4Format.NVFP4)).blk)
    with pytest.raises(NotImplementedError, match="MxQuantSpec|e5m2"):
        _declare_bwd_mxfp8(dict(_COMMON), b, s, spec=MxQuantSpec(descale_w_o=0.125, dtype=torch.float8_e5m2))
    with pytest.raises(ValueError, match="dy|bfloat16"):
        _declare_then_check(lambda: _declare_bwd_mxfp8(dict(_COMMON), b, s, dy_dtype=torch.float16).blk)
    with pytest.raises(ValueError, match="w_qkvg|w_o"):
        _declare_then_check(lambda: _declare_bwd_mxfp8(dict(_COMMON), b, s, w_dtype=torch.bfloat16).blk)
    with pytest.raises(ValueError, match="need_dw_qkvg"):
        _declare_then_check(lambda: _declare_bwd_mxfp8(dict(_COMMON), 1, 1000).blk)
    try:
        _declare_bwd_mxfp8(dict(_COMMON), 1, 1000, need_dw_qkvg=False).blk.check_support()
    except NotImplementedError as e:  # off Rubin the device gate is the only decline left, and it names Rubin
        assert _cc() != _SM107 and "Rubin" in str(e), str(e)
    with pytest.raises(NotImplementedError, match="seq_lens"):
        _declare_then_check(lambda: _declare_bwd_mxfp8(dict(_COMMON), b, s, seq_lens_present=True).blk)
    with pytest.raises(ValueError, match="fuse_wgrad_overlap"):
        _declare_then_check(lambda: _declare_bwd_mxfp8(dict(_COMMON), b, s, need_dw_o=False, need_dw_qkvg=False, fuse_wgrad_overlap=True).blk)
    with pytest.raises(ValueError, match="need_dh"):
        _declare_then_check(lambda: _declare_bwd_mxfp8(dict(_COMMON), b, s, need_dh=False, need_dw_qkvg=False, need_dw_o=False, need_dw_norms=False).blk)


@requires_cuda
def test_the_per_head_scale_slices_compose_the_rows_dequant():
    """Host, any CUDA device: ``_head_sf`` (the per-head reference's scale slices) composes the row reference's dequant -- for a
    ``[B, H, S, D]`` bf16 source quantized by ``quantize_to_mxfp8``, dequantizing head ``i`` of the payload with head ``i``'s slice of
    the ``[b*h, s, d]`` per-element scales is ``torch.equal`` head ``i`` of the whole-tensor dequant, rowwise and columnwise, at B = 2
    (a FLAT batch-head axis) and at an S that is no multiple of 128 (the quantizer's pad never reaches the reference); the naive
    ``sf[:, i : i + 1]`` on the flat tensor is the S-axis slice and cannot be viewed to the head's payload shape."""
    _test_python_root()
    from sdpa.mxfp8_quant import quantize_to_mxfp8
    from sdpa.mxfp8_ref import _dequant

    b, h, s, d = 2, 4, 96, _COMMON["d_head"]
    src = torch.randn(b, h, s, d, generator=torch.Generator(device="cuda").manual_seed(7), device="cuda").to(torch.bfloat16)
    pay_d, sf_d, _, pay_s, sf_s, _ = quantize_to_mxfp8(src, b, h, s, d, with_ref=True)
    for axis, pay, sf in (("row", pay_d, sf_d), ("col", pay_s, sf_s)):
        assert sf.shape == (b * h, s, d), (axis, tuple(sf.shape))
        whole = _dequant(pay, sf)
        for i in range(h):
            head = _dequant(pay[:, i : i + 1], _head_sf(sf, b, h, s, d, i))
            assert torch.equal(head, whole[:, i : i + 1]), (axis, i)
        with pytest.raises(RuntimeError):
            _dequant(pay[:, 0:1], sf[:, 0:1])  # dim 1 of the FLAT tensor is S: a [b*h, 1, d] slice, not a head


@requires_cuda
def test_the_per_head_partial_regions_are_compared_on_their_live_rows():
    """Host, any CUDA device, no compile and no launch: the MXFP8 adapter carves its per-Q-head ``dk_part`` / ``dv_part`` regions over
    the kv side's 256-row-padded S whenever that side is padded (``_scratch_shapes``: ``kv_rows = skvp if self._kv_padded else skv``),
    so at the padded GQA cells (S = 992, 1008) the regions hold MORE rows than the ``[B, S, H_q, D]`` per-head reference and the
    un-sliced comparison is a shape error; ``_live_rows`` is the ``[:, :s]`` the stage-localised layer compares on -- run here against
    the adapter's OWN carve, built the block's way (``_SdpaBwdMxfp8._build_impl``; the plan is constructor arithmetic, no
    ``check_support``), at the two padded cells and at an un-padded one (the identity), with the pad rows poisoned (NaN) to show
    they never reach the row's ``assert_close_fp8_grad``."""
    grad_tol, assert_close_fp8_grad = _row_tol()
    for cell in (_BY_ID["s992_causal_b1-norm"], _BY_ID["s1008_causal_b2-norm"], _BITWISE_CELL):
        g = GatedAttentionBlockGeometry(**cell.geom_kw)
        b, s, d = cell.b, cell.s, g.d_head
        st = _api_bwd._SdpaBwdMxfp8(g, batch=b, seq_len=s, grad_dtype=torch.bfloat16, device=torch.device("cuda"))
        # the carve is construction-time arithmetic (the adapter's pads, group and dS policy are fixed in its constructor); its
        # check_support() gates the block-scaled dS policy on the Rubin line and is not needed for the plan
        impl = st._ensure_impl()
        plan = {n: (tuple(int(x) for x in shape), dt) for n, shape, dt in impl._scratch_shapes()}
        kv_rows = int(impl._skv_pad) if impl._kv_padded else s
        assert _padded(s)[1] == bool(impl._kv_padded) and kv_rows >= s, (cell.id, kv_rows, s)
        for name in ("dk_part", "dv_part"):
            shape, dt = plan[name]
            assert shape == (b, kv_rows, g.h_q, d), (cell.id, name, shape)
            ref = torch.randn(b, s, g.h_q, d, generator=torch.Generator(device="cuda").manual_seed(3), device="cuda") * 0.01
            region = torch.full(shape, float("nan"), dtype=dt, device="cuda")
            region[:, :s] = ref.to(dt)
            live = _live_rows(region, s)
            assert live.shape == ref.shape and live.data_ptr() == region.data_ptr() and torch.isfinite(live.float()).all()
            assert_close_fp8_grad(live.float(), ref.to(dt).float(), grad_tol["atol"], grad_tol["rtol"], f"{cell.id} {name}", keys=s, budget=1e-5)
            if kv_rows != s:
                with pytest.raises(RuntimeError):
                    (region.float() - ref).abs()  # the un-sliced comparison: the shape error this pin guards against
        print(f"{cell.id}: dk_part / dv_part carved over {kv_rows} kv rows, compared on {s} live rows (kv padded: {bool(impl._kv_padded)})")


@requires_cuda
def test_the_matrix_declares_what_the_module_says():
    """Host, no launch: every matrix row is a shape the MXFP8 forward CAN record (a causal tail at S % 128 != 0 and a dense multiple
    of 128 only), every cell with the projection weight gradient has ``B*S % 32 == 0`` (the dgrad-only row is the one ragged T), the
    delayed cell shares the bitwise cell's geometry, the two launch-count-only cells reach arms no matrix cell does, the stage list
    the matrix declares against is the module's, and the (M) layer is a plain assertion on every cell once switched on (no xfail)."""
    for c in _CELLS + _LAUNCH_ONLY_CELLS:
        assert c.causal or c.s % 128 == 0, f"{c.id}: a dense S % 128 != 0 has no record"
        assert c.need_dw_qkvg is False or c.t % 32 == 0, c.id
    ragged = [c for c in _CELLS if c.t % 32]
    assert [c.id for c in ragged] == ["s1000_causal_b1_dgrad_only-norm"] and not ragged[0].need_dw_qkvg and ragged[0].need_dw_o
    for c in _LAUNCH_ONLY_CELLS:
        assert c.causal and c.id not in {m.id for m in _CELLS}, c.id
    pad_mha, kv_only = _BY_ID["s992_causal_b1_mha-norm"], _BY_ID["s384_causal_b1-norm"]
    assert _padded(pad_mha.s) == (True, True) and pad_mha.h_kv == _COMMON["h_q"] and _padded(kv_only.s) == (False, True) and kv_only.h_kv < _COMMON["h_q"]
    delayed, base = _BY_ID["s512_causal_b2_delayed-norm"], _BITWISE_CELL
    assert delayed.grad_scaling == "delayed" and (delayed.s, delayed.causal, delayed.b, delayed.h_kv, delayed.qk_norm) == (
        base.s,
        base.causal,
        base.b,
        base.h_kv,
        base.qk_norm,
    )
    assert [c.id for c in _CELLS if c.grad_scaling != "current"] == [delayed.id]
    assert len(_GQA_CELLS) == 12 and all(c.group > 1 for c in _GQA_CELLS)
    assert len(_MXFP8_STAGES) == 11 and _MXFP8_STAGES.count("_QuantizeMxfp8") == 1  # the fused chain: one standalone (dual-axis dO) quantize
    assert _MXFP8_STAGES[0] == "_MxQuantPrologue" and _MXFP8_STAGES[-3] == "_MxQuantEpilogue"
    assert set(_LAUNCH_CELL_IDS) <= set(_BY_ID)
    assert isinstance(_M_ROW_BUDGET_ASSERTED, bool) and isinstance(_STAGE_BF16_FORM_ASSERTED, bool)

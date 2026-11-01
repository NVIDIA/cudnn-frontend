# Placement table generator

`gen.py` turns a frozen FROST-vs-backend sweep into entries of `python/cudnn/sdpa/placement_table.py`, the measured table that `fwd/placement.place` and `bwd/heuristics._leads` consult before their hand-written rules (Rule S12 in `python/cudnn/sdpa/AGENTS.md`).

## Workflow

1. **Freeze a study.** Write the training grid and a seeded random holdout, record their sha256, and only then time anything. A study JSON names:
   - `id`, `row` (the FROST engine spec name), `device_cc`, `dtype` (`bf16` / `f16`);
   - `adapter`: `inference` for the generation phase of `benchmark/attention_inference`, or `training` for `benchmark/attention_training`;
   - head pair defaults (`d_qk`, `d_v`), an optional `layout` default;
   - the `train` / `holdout` JSONL paths.
2. **Sweep on the target part.** Run the `cudnn` (backend only), `cudnn_oss` (FROST first) and `cudnn_auto` (default dispatch) legs. Write one JSON line per (case, leg) with `ms`, `plan` and the `case` dict. Use a full-power board, and record co-tenant GPU processes at start and end.
3. **Fit:** `python benchmark/placement_gen/gen.py study.json`. Every case becomes graph facts, classified with `placement_table.class_key`, the same code the runtime uses. Per class, every scalar in `placement_table.FEATURES` is fit (today: `cost`, the roofline proxy; `tiles`, the parallel tiles before a KV split; `rows`, the query rows b * h_q * s_q; `kv`, the KV tokens per request; `q`, the query tokens per request). Each fit is:
   - `always`;
   - `never`;
   - or one threshold halfway between the last training loss and the first win past it, in the direction that keeps more wins.

   A threshold gives up the wins on its far side, so `always` / `never` replace it when every loss they accept is
   marginal (within twice `--margin`) and their worst training cell, then their geomean regret, is lower.

   The feature with the lowest training regret is kept and recorded in the entry. Cells within `--margin` of parity (default 3%) never move a threshold. A class that no feature can split monotonically is **NON-MONOTONE** and gets no entry; refine the class key instead.
4. **Qualify on the frozen holdout, once, at the frozen head.** The holdout must exercise every class the entries cover (on- and off-tile S_kv, every layout), was never used to fit or to choose a feature or constant, and is consumed by the run.
5. **Score.** The tool prints generated vs measured-default vs backend-only regret on train and holdout. The holdout must show no case where the generated choice is slower than the measured default beyond noise.
6. **Emit:** `gen.py study.json --emit` prints the entries. Paste them as one block under a comment naming the study, part, cuDNN version and case counts.

Entries carry the study id. Re-run the study when a kernel change touches the row's capabilities.

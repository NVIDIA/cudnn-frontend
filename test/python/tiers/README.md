# Test tiers: SMOKE, FULL, NIGHTLY

Three nested selections of the `L0` matrix for LOCAL runs of the FROST test suites (`sdpa/frost`,
`gated_attention_block/cutedsl`, `sdpa/graph/test_mhas_v2.py`, ...). CI keeps `-m L0` (`pytest.ini` `addopts`) and is not
changed by anything in this directory.

| tier | selection | definition | target |
|---|---|---|---|
| SMOKE | `-m smoke` | one cell per code path -- engine row x dtype x mask arm x layout (dense / THD / paged / split) x prepared / graph API -- at the smallest shape: `smoke_<arch>.txt` | minutes on one GPU |
| FULL | `-m "L0 and not nightly_only"` | the functional matrix: every L0 cell minus `nightly_only.txt` | the gate for a kernel change; under an hour on 4 GPUs with 4 processes each (the `sdpa/frost` + gated-block + `test_mhas_v2` matrix measured 26.5 min with a cold and 12.5 min with a warm compiled-plan cache on Rubin, cc 10.7, 204 SMs) |
| NIGHTLY | `-m L0`, or `-m "L0 or L1 or L2 or L3 or L4"` | everything, the `nightly_only` cells included | overnight |

SMOKE is a subset of FULL, FULL of CI's L0, CI's L0 of NIGHTLY.

## How the markers get on the cells

`test/python/conftest.py` applies the two markers at collection from the lists here (`_apply_tier_markers`), so no test file
carries a tier and moving a cell is a one-line edit of a list:

- `smoke_<arch>.txt` -- one list per compute capability, `arch` = `cc<major><minor>` of the GPU the process runs on (`cc107`
  on Rubin, `cc100` on B200 / GB200). A capability without a list has an EMPTY SMOKE tier -- `-m smoke` then selects nothing
  (exit 5) and `conftest.py` says so in one `[tiers] no SMOKE list for this GPU ...` line on stderr. `CUDNN_TEST_TIER_ARCH=cc107`
  applies another arch's list on any host (to inspect or collect it).
- `nightly_only.txt` -- shared by every arch; `nightly_only_reasons.tsv` gives, per row, the reason and the FULL cell that
  keeps its arm (`arm:<tokens>` = shared dtype / mask / family / shape tokens; `module` = same module only -- review those).
- `smoke_<arch>_code_paths.tsv` -- the code path each SMOKE cell stands for (one line per listed id; the reviewer's aid).

Lines are node ids relative to `test/python` (what `pytest --collect-only -q` prints from there); `#` starts a comment.
The cc 10.7 sweep cells of `test_mhas_v2.py` (the `_cc107_sweep` functions) are defined at the default `MHAS_CC107_MULT`
(4): the case count is part of every seed, so `[test1]` draws another geometry under a different multiplier -- a SMOKE cell
there names the default-multiplier draw.
`test_tiers.py` (L0) asserts that every listed id still collects and that `-m smoke` / `-m nightly_only` select exactly the
listed cells, so a renamed or re-parametrized test cannot silently drop out of a tier -- rename the id in the list in the
same commit.

## Rules for a cell to be listed

SMOKE -- every FROST engine row and every compile-time (`const_expr`) mask / layout arm the row serves is compiled and
executed once:

1. One cell per (engine row, dtype, mask arm, layout, API), at the smallest shape that reaches the arm. A shape that trips
   a different arm (a tail tile, a ring wrap, a split) is its own code path and gets its own cell.
2. Prefer the cell with the strongest oracle for that path (an fp32 / fp64 reference over a bitwise twin check).
3. Host-only contract cells (declines, validators, source and SASS / PTX pins) are not SMOKE -- they cost nothing in FULL
   and prove no kernel.
4. A code path with no runnable L0 cell is a gap to fill in the suite, not a reason to list a non-L0 cell.
5. A compile-time softmax LEVER is a `const_expr` arm too: `softmax_precision=HALF` (`SOFTMAX_F16`), the pre-folded scale
   (`SCALE_PREFOLDED`) and their fused stats-less arm (`_FUSED_SHIFT_CVT`) each get one cell per kernel on the dense arm, with
   Stats where the lever changes the LSE denominator (HALF alone, FLOAT + fold) and without where the fused arm needs it; the
   e5m2 member of the HALF cast is covered once (the helpers are shared). The lever x mask-arm cross product, the other dtype
   of every f16 kernel and the B / H variations stay in FULL.

NIGHTLY-only -- a cell leaves FULL only with a named twin that keeps its arm (`nightly_only_reasons.tsv`, third column):

1. A codegen pin -- the function BODY compiles a kernel and inspects its SASS / PTX / cubin or a PTX digest -- whose module
   holds a numerics cell of the same dtype / mask / family / shape tokens. Classified by what the body does, never by the
   test's name (`pin` also matches wraPPINg, sPINning and maPPINg; a test named `rendering` is as often a GPU numerics
   test). A codegen pin with no numerics cell in its module stays in FULL.
2. A B / H / seed twin -- a cell that differs from a kept cell of the SAME function only in batch, MHA head count or seed
   (same engine row, mask arm, `const_expr` arm and reference; only the grid size differs). Never twinned: a function with
   a kv-head key, or whose name mentions head / const / group / pack / gqa / mqa / thd / count / slot / budget / map /
   prefix / metadata / parallel / capacity / degenerate / ragged / tail / sequence; an arm named in the id (gqa / mqa /
   mha) is part of the key.
3. A big-shape (S >= 16K) oracle cell, only when a smaller sibling of the same function keeps the arm. (None on the tree
   today; a cell that is its own contract -- an Int32-overflow probe -- stays in FULL.)
4. Never: `gpu_exclusive` cells, reject / decline / typed-error cells, or any cell whose demotion would change a tolerance.

Counts at the time of writing: SMOKE cc 10.7 = 350 cells (258 hand-picked + 92 rule-added), cc 10.0 = 267 (135 + 132);
NIGHTLY-only = 120 (96 codegen pins + 24 B / H / seed twins). Lists for other capabilities are added as they are reviewed.

## Running

From `test/python`, with the FROST engines opted in (`CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1`); keep the `sdpa/frost/` paths
contiguous on the command line (`test/AGENTS.md`):

```bash
pytest -m smoke -n 4 --dist loadgroup sdpa/frost gated_attention_block/cutedsl sdpa/graph/test_mhas_v2.py   # SMOKE, one GPU
pytest -m "L0 and not nightly_only" sdpa/frost gated_attention_block/cutedsl sdpa/graph/test_mhas_v2.py      # FULL
```

Several pytest processes on one GPU: set `CUDNN_TEST_SHARED_GPU=1` in each so the GPU memory gate of `conftest.py` is armed
for them (see `test/AGENTS.md`).

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""SM107 dense d128 BF16/FP16 accumulation.

1. Identity: faithful one-CTA five-product port from SHA850a7e99.
   Intended target: python/cudnn/sdpa/bwd/kernels/sm107/bprop_d128_accum_f16.py.
   Source K: kernels/sm100/bprop_d128_accum_f16.py, all ranges below are K lines.

2. Computation / numerics:
   S^T=K Q^T; P^T=exp(scale*S^T - natural-log LSE).
   dP^T=V dO^T; dS^T=scale*P^T*(dP^T-delta).
   dV+=P^T dO; dK+=dS^T Q; dQ=dS K, FP32 reduce-add workspace then native cast.
   Keep source fp32/packed operation order, tails, mask and SELECT zero handling.

3. Item / shapes:
   One (batch,head,128-KV-row block); iterate128-Q-row tiles.
   Every MMA M=N=K=128, group1, K_HW16, kind F16, k_dim0.
   Prologue S; each tile dP(4+4 steps),dV,lookahead S,dK,dQ.
   Q stages2; dO stage1 with2 spatial half-full events; scheduler stages2.

4. Warp map / registers:
   softmax0..7:144 INCREASE; epilogue8..11:144 INCREASE;
   MMA12/load13/store14/scheduler15:80 DECREASE, equal producer counts.
   8*144+4*144+4*80=2048=entry128*16. min_blocks_per_mp1.

5. Barriers -- BINDING source ledger, all Scope.LOCAL, group1,22 names/33 slots.
   Source sites below are K lines at SHA850a7e99, not this module's line numbers.
   SM=softmax8 warps, EP=epilogue4, MM/LD/ST/SC=one warp each.
   e=one elected lane per calling warp; b=bare32 lanes per warp; one CTA.
   ONE=CFG.ONE_LANE(1), SOFT=CFG.SOFTMAX_LANES(256),
   EPI=CFG.EPILOGUE_LANES(128), STORE=CFG.STORE_LANES(32),
   READ=CFG.SCHED_CONSUMER_WARPS(15). Every equation is per slot/per phase.
   MMA=MMA_COMMIT, TMA=TMA_LOAD, THR=THREAD. A=arrive, W=wait.
   p0=PipelineState.start(), p1=PipelineState.start(phase=1).

   name[slots]@byte | producer / guarded issuer / SUM == init | bytes | observer W / carried state
   k_empty[1]@230400 | MMA MM:e A527,1*1*1=ONE OK | - | LD W565,kv p1
   k_full[1]@230416 | TMA LD:pred=e A566,1*1*1=ONE OK |32768| MM W491,kv p0
   v_empty[1]@230432 | MMA MM:e A528,1*1*1=ONE OK | - | LD W568,kv p1
   v_full[1]@230448 | TMA LD:pred=e A569,1*1*1=ONE OK |32768| MM W496,kv p0
   do_empty[1]@230464 | MMA MM:e A508,1*1*1=ONE OK | - | LD W580,do p1
   do_full[2]@230480 | TMA LD:pred=e A582/half,1*1*1=ONE OK |16384/half| MM W501/503,do p0
   q_empty[2]@230496 | MMA MM:e A518,1*1*1=ONE OK | - | LD W573,q p1/depth2
   q_full[2]@230512 | TMA LD:pred=e A574,1*1*1=ONE OK |32768| MM W492/511,q/q_next p0/depth2
   s_done[1]@230528 | MMA MM:e A495 or514,1*1*1=ONE OK | - | SM W314,state p0
   dp_done[1]@230544 | MMA MM:e A505,1*1*1=ONE OK | - | SM W319,state p0
   dq_done[1]@230560 | MMA MM:e A521 live or531 zero,1*1*1=ONE OK | - | EP W409/425,dq p0
   dp_empty[1]@230592 | THR EP:b A414,32*4*1=EPI OK | - | MM W500,dp_empty p1
   p_ready[1]@230608 | THR SM:b A318,32*8*1=SOFT OK | - | MM W506,ready p0
   ds_ready[1]@230624 | THR SM:b A323,32*8*1=SOFT OK | - | MM W515,ready p0; EP W398,ds p0
   ds_smem[1]@230640 | THR EP:b A408,32*4*1=EPI OK | - | MM W519,ready p0
   dq_empty[4]@230656 | THR ST:b A623 bootstrap or634 release,32*1*1=STORE OK | - | EP W402/404/417/433/360,empty p0
   dq_full[4]@230688 | THR EP:b A420/slice,32*4*1=EPI OK | - | ST W627/slice,dq p0
   dv_full[1]@230720 | THR EP:b A367,32*4*1=EPI OK | - | ST W636,out p0
   dk_full[1]@230736 | THR EP:b A365,32*4*1=EPI OK | - | ST W639,out p0
   dvk_empty[1]@230752 | THR ST:b A644,32*1*1=STORE OK | - | MM W497,out_empty p1
   sched_full[2]@230768 | TMA SC:e A656,1*1*1=ONE OK |16| all15 consumers W174 plus SC W660,sched p0/depth2
   sched_empty[2]@230784 | THR SM:e A309 or328 + EP:e395 + MM:e488 + LD:e562 + ST:e621,
                        |     (8+4+1+1+1)*1*1=READ OK | - | SC W654,state p0/depth2

   Named resources are separate from the33 mbarrier slots:
   Stats idCFG.SOFTMAX_BARRIER(1): SM:b,32*8=SOFT(256), both K:202 and219,
     before Stats/delta overwrite then after cooperative stores;2*N joins/item.
     _stats_subset_fence requests unaligned subset synchronization through
     the generic tile_dsl helper with aligned=False. Both scheduling/memory
     fences are retained; the shared helper's default remains unchanged.
   CTA id0: all16*32=CFG.THREADS(512), init publication K:707 and final join K:721.
   No role may reuse id0 for a subset. All active slots initialize once;
   dq_done[1:3],dp_empty[1],ds_smem[1] are reserved padding, not active stages.
   BAR_LAYOUT values/kinds/offsets match source exactly; counts use named CFG fields.

6. Init / phase / drains -- BINDING; no topology or cadence deviation permitted.
   Caller scaffolding order, preserving K:685,692..708:
     warp = make_warp_uniform(warp_idx()); construct slab/bars/tmem_slot/sched.
     if warp ==0: collectively allocate CFG.TMEM_COLS(512), group1;
       collectively relinquish permit; full-mask bar_warp_sync (K:699..701).
     OUTSIDE that branch call _init_and_publish(bars, warp) with all512 threads.
     Helper: if warp==0 AND elect, init each Bars field's every active stage;
       then fence_mbarrier_init and full-CTA sync outside both guards.
     Only after helper return load tmem_slot and dispatch all six source roles.
   The helper must not allocate/relinquish/read TMEM or bootstrap an arrival.
   At kernel end all roles consume final invalid CLC response, full-CTA sync,
   then warp0 AFTER_THREAD_SYNC and collective matching512-column group1 free.
   No exclusive allocation and no cluster sync/drain in this group1 kernel.

   PAIRING (per slot's generation k): p0 wait k consumes completion k;
   p1 wait k consumes release k-1, with virtual initial release already available.
   Loader q/do/kv, MM dp_empty/out_empty start p1; every other listed state p0.
   No MMA_COMMIT bootstrap. ST A623 explicitly supplies32 arrivals per dq_empty
   slice per owner, and EP empty_state starts p0. Its bootstrap and later source
   releases are DISTINCT phases, never additive credits in one phase.
   dO[0/1] are spatial halves, dQ[0..3] spatial slices, not temporal phase indices.
   Q/scheduler slot and parity come from carried PipelineState, depth2.

   N=Q iterations in one owner; L=prior Q iterations,V=prior nonzero owners,
   Z=prior zero owners,j=prior owners. State generations/cadences:
     Q: idx=L mod2,full phase=floor(L/2) mod2,empty phase=1 XOR full;advance N.
     K/V: full generation V,pre-armed empty V-1;advance only if N>0.
     do/softmax/ready/EP ds/ST dq: generation L;advance N.
     MM dp_empty: wait generation L-1;advance N, never on zero owners.
     EP dq: generation L+Z;advance N if live,else1 at K:425..426.
     EP empty: generation L+j;advance N then1 per owner at K:436.
     ST out/MM out_empty: generation j / pre-armed j-1;advance every owner.
     Every scheduler state: idx=j mod2,phase=floor(j/2) mod2;advance every response.
   These are audit equations, not permission to replace PipelineState with bit math.
   Fires for N=0/1/2/3:
     live Q rows:0/1/2/3; K/V:0/1/1/1; dq_done:1/1/2/3;
     each dq_empty:1/2/3/4; dv/dk/dvk/scheduler:1/1/1/1; Stats joins:0/2/4/6.
   Keep q_next captured K:509 before state assignment K:522. Carry every state
   across live-dead-dead-live owners and ring wraps; do not reset per work item.

   Role contracts (source sequence remains binding):
   _softmax_loop: W s_done314,dp_done319; bare256 P318/dS323 publications after
     tcgen05 STORE wait and BEFORE_THREAD_SYNC. Stats joins at both202/219.
     Elected8 scheduler credits at309 dense OR328 causal-late, not both.
   _epilogue_loop: W ds_ready398; dS LOAD wait401; W dq_empty0/1 at402/404;
     STS/proxy406/before407 ->bare128 ds_smem408. W dq_done409, LOAD wait412,
     before413 ->bare128 dp_empty414. Per slice W417,STS/proxy419 ->bare128 full420.
     Zero owner instead W dq_done425 and advances dq_state once.
     BEFORE native output publications, W ALL FOUR terminal dq_empty slices433,
     including zero owners. Native specializations still W slice1/3 at360 and
     publish dv367/dk365 bare128 after proxy362/before363. Elected4 sched credits395.
   _mma_loop: all commits elected1,group1. Source order: S prologue494/495,
     W out_empty497 EVERY owner,then dP502/504/505,dV507/do_empty508,
     lookahead S513/514 if live,dK517/q_empty518,dQ520/dq_done521.
     W dp_empty500; both do_full501/503; p_ready506; ds_ready515;ds_smem519.
     K/V empties527/528 only live; otherwise dq_done531 AFTER out_empty497.
     Elected1 sched credit488. Never treat zero dq_done as an independent bootstrap.
   _load_loop: W K/V empty565/568 ->expect_tx full566/569 at32768 each;
     W Q empty573 ->expect_tx574 at32768; W dO empty580 ->two half arms582
     at16384 each,then matching loads584. TMA issues elected by their helpers.
     Elected1 sched credit562. Empty-owner guards skip both side's live events.
   _store_loop: bare32 per-slice bootstrap623 per owner. Per live tile W full627,
     elected reduction630,warp-uniform commit631; waits633 in3,2,1,0 order then
     bare32 per-slice release634. W dv_full636/dk_full639; native stores638/640
     remain warp-uniform; commit642/wait0 at643 ->bare32 dvk_empty644.
     Elected1 sched credit621.
   _scheduler_loop: W empty654; elected expect_tx656 plus CLC657;tx16,slot stride32.
     Full-warp sync659,W full660,decode661,advance662. No extra self-credit.
     All15 consumer warps W full174,read/broadcast/proxy-fence payload before
     future slot credits. Final invalid response is observed by all16 warps.

   Phase reuse / alias requirements:
   - Every pre-armed empty's first real flip depends on that wait's preceding point.
     out_empty497 ->live521 orzero531 ->EP W409/425 ->outputs ->ST644.
   - Both ds_ready observers have back-edges: MM W515 ->next dP505 ->next323;
     each EP W398 ->ds_smem408 aggregate ->MM W519 ->next dP505 ->next323.
     dp_empty414 ->MM W500 separately protects completion of the dQ TMEM readout.
   - S/P write sequence: S ->SM LOAD wait ->P STORE wait/fence ->p_ready ->dV
     read ->lookahead S on the same MMA stream; no aliased write may move earlier.
   - dP/dS/dQ sequence: dP ->SM LOAD wait ->packed dS ->dK and EP readout;
     EP readout/STS ->ds_smem ->dQ overwrite ->EP LOAD wait ->dp_empty ->next dP.
   - ALIAS sequence: dS32KiB ->dQ64KiB ->native dV/dK32KiB each. Per-slice
     source wait/release gates every overwrite. N=0 skips dS/dQ writes but retains
     bootstrap,terminal observations and native zero stores. No new bank/tenant.
   - dq_empty W433 observes every completed terminal phase; native W360 at1/3
     only covers storage order of siblings and cannot replace observations at0/2.
     All-four terminal waits ->native publications ->ST drain643 ->next bootstrap623.
   - dq_full W627 ->source release634 ->EP W417 ->next full420,per slice.
   - Every scheduler payload reader returns its own counted credit; all15 gate
     reuse; scheduler self-read is program-ordered. Retain causal-late softmax credit.
   There are no cross-CTA async arrivals, so cluster drains are not applicable.
   Per-owner dQ terminal waits are mandatory local phase reuse, not cluster drains.
   Preserve _wait full-mask reconvergence K:115, AFTER_THREAD_SYNC K:121,
   all TMEM wait/fence-before-release and generic->async proxy fences.

7. Bounds / scheduler:
   Port _decode_flat(157..169), _next_tile(173..182), _bounds(191..195).
   Every role uses identical inputs. Two-stage CLC payload stride32B/tx16B;
   exactly15 elected consumer-warp credits, no scheduler self-credit.

8. SMEM declaration order (bytes; all operand bases1024-aligned):
   Q@0:65536,TMA->S/dK; dO@65536:32768,TMA->dP/dV;
   K@98304:32768,TMA->S/dQ; V@131072:32768,TMA->dP;
   ALIAS@163840:65536,vectorSTS->dS MMA/dQ reduction/native dV+dK stores;
   Stats@229376:512 +delta@229888:512; scaffold@230400:1024.
   Total231424B of dynamic shared memory; scaffold and operand storage included.
   SW128 matches descriptors; STS row stride128 uses XOR(row%8,vec).
   Retain DESC_VERSION1; _operand's source layout enum2 is unchanged.
   ALIAS dS first32768 ->dQ65536 ->native dV/dK two32768 tenants.
   Stats remain in SMEM. No new ring or extra staging.

9. TMEM:
   S[0,128), packedP[0,32)U[64,96);
   dP/dQ[128,256), packeddS[128,160)U[192,224);
   dV[256,384),dK[384,512). All512 allocated/deallocated with one constant.
   No exclusive allocation. Keep LOAD/STORE waits before every slot release.

10. Tensor maps: IO box128x64x2=16384B, full operand32768B;
    dQ FP32 box128x32x4=16384B, four spatial slices.
    Keep true extents/OOB zero fill and clipped output coordinates.

11. Optional arms: dense/top-left causal only initially; no optimization knobs.
    THD/grouped acceptance follows separately; do not claim from dense tests.

12. Edge matrix: N0/1/2/3, live-dead-live, Q/scheduler wraps, tails,
    masked rows, one CTA, multiple batch/heads.

13. Performance: no speedup or speed-of-light claim for this port.

14. Launch ABI: kernel pointer/tensor-map sequence unchanged; block512;
    prepared chain delta+zero -> accumulation -> cast; append-only ABI.
    Native SM107 route/cache/symbol names separate from SM100; runtime
    public adapter DSL floor4.8.1 and SM107 admission are typed host gates.
"""

from typing import NamedTuple

from cudnn.frost.tile_dsl.requirements import requirement_error

_requirement = requirement_error("SM107 d128 accumulation", (10, 7))
if _requirement is not None:
    raise NotImplementedError(_requirement)

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import arith
from cutlass.base_dsl.typing import Pointer
from cutlass.experimental import primitives as nvvm
from cutlass.experimental.cuda import tensor_map as tmap

from cudnn.frost.tile_dsl.barrier import MBarrier, PipelineState, Producer, Scope, advance, named_barrier_fence
from cudnn.frost.tile_dsl.handles import GmemTileTma, MmaDesc, SmemTile
from cudnn.frost.tile_dsl.mma import mma_ss, mma_ts_step
from cudnn.frost.tile_dsl.pointwise import ffma2, fmul2, fp32_to_fp16
from cudnn.frost.tile_dsl.scheduler import Sched, read_clc_payload
from cudnn.frost.tile_dsl.tma import tma_load_subtiles, tma_load_tile, tma_store_commit, tma_store_tile, tma_store_wait

from ...config_sm107_d128_accum import BAR_LAYOUT, CFG, OPERAND_DESCRIPTOR

SCAFFOLD_ONLY = False
LOG2E = 1.4426950408889634


class Bars(NamedTuple):
    k_empty: MBarrier
    k_full: MBarrier
    v_empty: MBarrier
    v_full: MBarrier
    do_empty: MBarrier
    do_full: MBarrier
    q_empty: MBarrier
    q_full: MBarrier
    s_done: MBarrier
    dp_done: MBarrier
    dq_done: MBarrier
    dp_empty: MBarrier
    p_ready: MBarrier
    ds_ready: MBarrier
    ds_smem: MBarrier
    dq_empty: MBarrier
    dq_full: MBarrier
    dv_full: MBarrier
    dk_full: MBarrier
    dvk_empty: MBarrier
    sched_full: MBarrier
    sched_empty: MBarrier


def _bars(slab):
    kinds = {"mma": Producer.MMA_COMMIT, "tma": Producer.TMA_LOAD, "thread": Producer.THREAD}
    return Bars(
        *(
            MBarrier(
                cutlass.Array(
                    Pointer(slab.data_ptr() + off, dtype=cutlass.Int64).ir_value(),
                    shape=slots,
                    dtype=cutlass.Int64,
                    alignment=CFG.BARRIER_ALIGNMENT,
                ),
                stages=slots,
                init_count=count,
                producer=kinds[kind],
                scope=Scope.LOCAL,
                try_wait=True,
            )
            for _, off, slots, count, kind in BAR_LAYOUT
        )
    )


@cute.jit
def _init_and_publish(bars, warp):
    # Called by ALL512 threads after warp0's collective alloc/relinquish/warp sync.
    # Keep allocation and the post-publication tmem_slot load in the kernel caller.
    if warp == 0:
        if nvvm.elect_sync():
            for i in cutlass.range_constexpr(len(bars)):
                for stage in cutlass.range_constexpr(bars[i].stages):
                    bars[i][stage].init()
    nvvm.fence_mbarrier_init()
    nvvm.barrier_cta_sync()


@cute.jit
def _stats_subset_fence():
    # Preserve both inline scheduling fences for the participating subset.
    named_barrier_fence(CFG.SOFTMAX_BARRIER, CFG.SOFTMAX_LANES, aligned=False)


@cute.jit
def _stats(slab, lse, delta, row, group, qbase, sq, batch, head, scale):
    # Every one of256 lanes participates in both named barriers. Each group
    # writes a different128-float vector, one row per lane.
    _stats_subset_fence()
    idx = qbase + row
    valid = idx < sq
    if group == 0:
        # Tail rows past S_q read the last Stats row (in bounds) and publish
        # -inf: their zero-filled Q / dO columns then yield exp2(-inf) = 0.
        clamped = cutlass.Int32(arith.select(valid.ir_value(), idx.ir_value(), (sq - 1).ir_value()))
        val = cutlass.Float32(lse[batch, head, clamped])
        stat = -cutlass.Float32(LOG2E) * val
        if val == cutlass.Float32(float("-inf")):
            stat = cutlass.Float32(0.0)
        stat = cutlass.Float32(arith.select(valid.ir_value(), stat.ir_value(), cutlass.Float32(float("-inf")).ir_value()))
        _view(slab, CFG.STATS, cutlass.Float32, CFG.N).subview(row).store(stat)
    else:
        # delta spans the rounded S_q; dot_do_o_kernel zeroes its tail rows.
        stat = -scale * cutlass.Float32(delta[batch, head, idx])
        _view(slab, CFG.DELTA, cutlass.Float32, CFG.N).subview(row).store(stat)
    _stats_subset_fence()


def _view(slab, offset, dtype, count, alignment=CFG.BARRIER_ALIGNMENT):
    ptr = Pointer(slab.data_ptr() + offset, dtype=dtype)
    return cutlass.Array(ptr.ir_value(), shape=count, dtype=dtype, alignment=alignment)


def _operand(slab, offset, dtype, *, transpose=False, fp32=False):
    box = CFG.DQ_BOX if fp32 else CFG.HEAD_BOX
    bpe = CFG.BPE_ACC if fp32 else CFG.BPE
    return SmemTile(
        base=_view(slab, offset, dtype, CFG.M * CFG.D, alignment=CFG.SMEM_ALIGNMENT),
        elems_per_stage=CFG.M * CFG.D,
        leading_byte_offset=CFG.M * CFG.SWIZZLE_BYTES if transpose else OPERAND_DESCRIPTOR["leading_byte_offset"],
        stride_byte_offset=OPERAND_DESCRIPTOR["stride_byte_offset"],
        layout=CFG.SWIZZLE_LAYOUT,
        tma_loads_per_tile=CFG.D // box,
        tma_granu_elems=box,
        tma_subtile_stride_elems=CFG.M * box,
        desc_version=CFG.DESC_VERSION,
    )


def _mma_desc(dtype, *, atranspose=False, btranspose=False):
    return MmaDesc(
        M=CFG.M,
        N=CFG.N,
        K=CFG.D,
        bpe_a=CFG.BPE,
        bpe_b=CFG.BPE,
        tile_k_hw=CFG.K_HW,
        atranspose=atranspose,
        btranspose=btranspose,
        cta_group=CFG.CTA_MMA,
        idesc=nvvm.Tcgen05InstrDesc.build(
            c_dtype=cutlass.Float32, a_dtype=dtype, b_dtype=dtype, m_dim=CFG.M, n_dim=CFG.N, a_major=int(atranspose), b_major=int(btranspose)
        ),
        kind=nvvm.Tcgen05MMAKind.F16,
    )


@cute.jit
def _decode_flat(tile, nheads, nk, swizzle: cutlass.Constexpr[int], batch_heads: cutlass.Constexpr[int]):
    group = tile // (swizzle * nk)
    inner = tile % (swizzle * nk)
    block = cutlass.Int32(0)
    residual = cutlass.Int32(0)
    if group < batch_heads // swizzle:
        block = inner // swizzle
        residual = inner % swizzle
    else:
        block = inner // max(batch_heads % swizzle, 1)
        residual = inner % max(batch_heads % swizzle, 1)
    hb = group * swizzle + residual
    return block, hb % nheads, hb // nheads


def _tensor_map(tensor, box):
    # Logical TMA coordinates (d, sequence, head, batch), source-proved order.
    return tmap.create_tensor_map_tiled_from_view(tensor, box_dims=(1, CFG.M, 1, box), stride_order=(3, 1, 2, 0), swizzle=tmap.TensorMapSwizzle.s128b)


@cute.jit
def _wait(bar, phase):
    bar.wait(phase)
    # All full-warp consumers reconverge before any elected credit/issue.
    nvvm.bar_warp_sync(cute.arch.FULL_MASK)


@cute.jit
def _wait_tcgen(bar, phase):
    _wait(bar, phase)
    nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)


@cute.jit
def _commit(bar):
    if nvvm.elect_sync():
        bar.arrive(cta_group=CFG.CTA_MMA)


@cute.jit
def _credit(bar):
    if nvvm.elect_sync():
        bar.arrive()


@cute.jit
def _next_tile(sched, bars, state, nheads, nk, causal: cutlass.Constexpr[bool], swizzle: cutlass.Constexpr[int], batch_heads: cutlass.Constexpr[int]):
    _wait(bars.sched_full[state.idx], state.phase)
    x, yz, valid = read_clc_payload(sched, state.idx * (CFG.SCHED_PAYLOAD_STAGE_BYTES // CFG.BPE_ACC), warp_broadcast=True)
    if cutlass.const_expr(causal):
        block, head, batch = _decode_flat(x, nheads, nk, swizzle, batch_heads)
    else:
        block = x
        head = yz & 0xFFFF
        batch = (yz >> 16) & 0xFFFF
    return block, head, batch, valid, advance(state, CFG.SCHED_STAGES)


@cute.jit
def _bounds(block, sq, causal: cutlass.Constexpr[bool]):
    lo = cutlass.Int32(0)
    if cutlass.const_expr(causal):
        lo = block
    return lo, (sq + CFG.N - 1) // CFG.N


@cute.jit
def _tmem_load(address, n: cutlass.Constexpr[int], dtype: cutlass.Constexpr = cutlass.Float32):
    return nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(address, dtype), num=n)


@cute.jit
def _tmem_store(address, values):
    nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(address, cutlass.Int32), values)


@cute.jit
def _p_values(slab, tmem_base, row, group, block, qtile, lo, skv, scale, causal: cutlass.Constexpr[bool], dtype: cutlass.Constexpr):
    address = tmem_base + group * CFG.D_HALF
    a = _tmem_load(address, CFG.SOFTMAX_TMEM_LOAD_COLS)
    b = _tmem_load(address + CFG.SOFTMAX_TMEM_LOAD_COLS, CFG.SOFTMAX_TMEM_LOAD_COLS)
    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
    scores = cutlass.Vector.from_elements(
        tuple(a[i] for i in range(CFG.SOFTMAX_TMEM_LOAD_COLS)) + tuple(b[i] for i in range(CFG.SOFTMAX_TMEM_LOAD_COLS)), cutlass.Float32
    )
    p = []
    packed = []
    scaled = scale * cutlass.Float32(LOG2E)
    neg_inf = cutlass.Float32(float("-inf"))
    kr = block * CFG.M + row
    # The tail KV tile's zero-filled key rows (kr >= skv) must not exponentiate:
    # exp2(-stat) overflows for a very negative LSE, and inf times their zero K
    # would put NaN into dQ.  Only the last KV tile pays the select.
    key_tail = (block + 1) * CFG.M > skv
    key_valid = kr < skv
    for chunk in cutlass.range_constexpr(CFG.SOFTMAX_CHUNKS):
        stat = (
            _view(slab, CFG.STATS, cutlass.Float32, CFG.N)
            .subview(group * CFG.D_HALF + chunk * CFG.STATS_VECTOR_ELEMS)
            .load(vector_size=CFG.STATS_VECTOR_ELEMS, alignment=CFG.STATS_VECTOR_ALIGNMENT)
        )
        for pair in cutlass.range_constexpr(CFG.SOFTMAX_PAIRS_PER_CHUNK):
            i = chunk * CFG.STATS_VECTOR_ELEMS + pair * CFG.FP32_PAIR_ELEMS
            s0, s1 = scores[i], scores[i + 1]
            if cutlass.const_expr(causal):
                if qtile == lo:
                    qc = qtile * CFG.N + group * CFG.D_HALF + i
                    s0 = cutlass.Float32(arith.select((kr <= qc).ir_value(), s0.ir_value(), neg_inf.ir_value()))
                    s1 = cutlass.Float32(arith.select((kr <= qc + 1).ir_value(), s1.ir_value(), neg_inf.ir_value()))
            if key_tail:
                s0 = cutlass.Float32(arith.select(key_valid.ir_value(), s0.ir_value(), neg_inf.ir_value()))
                s1 = cutlass.Float32(arith.select(key_valid.ir_value(), s1.ir_value(), neg_inf.ir_value()))
            e0, e1 = ffma2(scaled, scaled, s0, s1, stat[pair * CFG.FP32_PAIR_ELEMS], stat[pair * CFG.FP32_PAIR_ELEMS + 1])
            p0 = cute.math.exp2(e0, fastmath=True)
            p1 = cute.math.exp2(e1, fastmath=True)
            p.extend((p0, p1))
            packed.append(fp32_to_fp16(p0, p1, dtype=dtype))
    _tmem_store(address, cutlass.Vector.from_elements(tuple(packed), cutlass.Int32))
    return cutlass.Vector.from_elements(tuple(p), cutlass.Float32)


@cute.jit
def _ds_values(slab, tmem_base, group, p, scale, dtype: cutlass.Constexpr):
    address = tmem_base + CFG.DP_DS_DQ_COL + group * CFG.D_HALF
    a = _tmem_load(address, CFG.SOFTMAX_TMEM_LOAD_COLS)
    b = _tmem_load(address + CFG.SOFTMAX_TMEM_LOAD_COLS, CFG.SOFTMAX_TMEM_LOAD_COLS)
    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
    dp = cutlass.Vector.from_elements(
        tuple(a[i] for i in range(CFG.SOFTMAX_TMEM_LOAD_COLS)) + tuple(b[i] for i in range(CFG.SOFTMAX_TMEM_LOAD_COLS)), cutlass.Float32
    )
    packed = []
    for chunk in cutlass.range_constexpr(CFG.SOFTMAX_CHUNKS):
        stat = (
            _view(slab, CFG.DELTA, cutlass.Float32, CFG.N)
            .subview(group * CFG.D_HALF + chunk * CFG.STATS_VECTOR_ELEMS)
            .load(vector_size=CFG.STATS_VECTOR_ELEMS, alignment=CFG.STATS_VECTOR_ALIGNMENT)
        )
        for pair in cutlass.range_constexpr(CFG.SOFTMAX_PAIRS_PER_CHUNK):
            i = chunk * CFG.STATS_VECTOR_ELEMS + pair * CFG.FP32_PAIR_ELEMS
            d0, d1 = ffma2(scale, scale, dp[i], dp[i + 1], stat[pair * CFG.FP32_PAIR_ELEMS], stat[pair * CFG.FP32_PAIR_ELEMS + 1])
            d0, d1 = fmul2(d0, d1, p[i], p[i + 1])
            packed.append(fp32_to_fp16(d0, d1, dtype=dtype))
    _tmem_store(address, cutlass.Vector.from_elements(tuple(packed), cutlass.Int32))


@cute.jit
def _softmax_loop(
    slab,
    bars,
    sched,
    tmem_base,
    lse,
    delta,
    block,
    head,
    batch,
    sq,
    skv,
    nk,
    nheads,
    batch_heads: cutlass.Constexpr[int],
    swizzle: cutlass.Constexpr[int],
    scale,
    causal: cutlass.Constexpr[bool],
    dtype: cutlass.Constexpr,
):
    nvvm.setmaxregister(CFG.SOFTMAX_REGS, nvvm.SetMaxRegisterAction.INCREASE)
    tid = cute.arch.thread_idx()[0]
    row = tid % CFG.M
    group = tid // CFG.M
    state = PipelineState.start()
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        if cutlass.const_expr(not causal):
            _credit(bars.sched_empty[sched_state.idx])
        lo, hi = _bounds(block, sq, causal)
        if lo < hi:
            _stats(slab, lse, delta, row, group, lo * CFG.N, sq, batch, head, scale)
        for qtile in cutlass.range(lo, hi, 1, unroll=1):
            _wait_tcgen(bars.s_done, state.phase)
            p = _p_values(slab, tmem_base, row, group, block, qtile, lo, skv, scale, causal, dtype)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
            bars.p_ready.arrive()
            _wait_tcgen(bars.dp_done, state.phase)
            _ds_values(slab, tmem_base, group, p, scale, dtype)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
            bars.ds_ready.arrive()
            if qtile + 1 < hi:
                _stats(slab, lse, delta, row, group, (qtile + 1) * CFG.N, sq, batch, head, scale)
            state = advance(state, CFG.SINGLE_STAGE)
        if cutlass.const_expr(causal):
            _credit(bars.sched_empty[sched_state.idx])
        block, head, batch, valid, sched_state = _next_tile(sched, bars, sched_state, nheads, nk, causal, swizzle, batch_heads)


@cute.jit
def _store_words(slab, byte_offset, row, words):
    for vec in cutlass.range_constexpr(CFG.STORE_VECS_PER_ROW):
        off = byte_offset + row * CFG.SWIZZLE_BYTES + ((row % CFG.SWIZZLE_ROWS) ^ vec) * CFG.STORE_VECTOR_BYTES
        values = cutlass.Vector.from_elements(tuple(words[vec * CFG.STORE_VECTOR_ELEMS + k] for k in range(CFG.STORE_VECTOR_ELEMS)), cutlass.Int32)
        Pointer(slab.data_ptr() + off, dtype=cutlass.Int32).store(values, alignment=CFG.STORE_VECTOR_BYTES)


@cute.jit
def _store_dq_slice(slab, row, values, index: cutlass.Constexpr[int]):
    for vec in cutlass.range_constexpr(CFG.STORE_VECS_PER_ROW):
        off = CFG.ALIAS + index * CFG.SLICE_BYTES + row * CFG.SWIZZLE_BYTES + ((row % CFG.SWIZZLE_ROWS) ^ vec) * CFG.STORE_VECTOR_BYTES
        v = cutlass.Vector.from_elements(
            tuple(values[index * CFG.DQ_BOX + vec * CFG.STORE_VECTOR_ELEMS + k] for k in range(CFG.STORE_VECTOR_ELEMS)), cutlass.Float32
        )
        Pointer(slab.data_ptr() + off, dtype=cutlass.Float32).store(v, alignment=CFG.STORE_VECTOR_BYTES)


@cute.jit
def _native_epilogue(slab, tmem_base, bars, row, live, empty_phase, dtype: cutlass.Constexpr, dk: cutlass.Constexpr[bool]):
    plane = CFG.DK_COL if cutlass.const_expr(dk) else CFG.DV_COL
    base = CFG.ALIAS + (CFG.OPERAND_BYTES if cutlass.const_expr(dk) else 0)
    last_slice = CFG.DQ_SLICES - 1 if cutlass.const_expr(dk) else CFG.DQ_SLICES // (CFG.D // CFG.D_HALF) - 1
    for half in cutlass.range_constexpr(CFG.D // CFG.D_HALF):
        vals = cutlass.Vector.from_elements(tuple(cutlass.Float32(0.0) for _ in range(CFG.D_HALF)), cutlass.Float32)
        if live:
            vals = _tmem_load(tmem_base + plane + half * CFG.D_HALF, CFG.D_HALF)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
        words = cutlass.Vector.from_elements(
            tuple(
                fp32_to_fp16(vals[CFG.FP32_PAIR_ELEMS * i], vals[CFG.FP32_PAIR_ELEMS * i + 1], dtype=dtype) for i in range(CFG.D_HALF // CFG.FP32_PAIR_ELEMS)
            ),
            cutlass.Int32,
        )
        if cutlass.const_expr(half == 0):
            _wait(bars.dq_empty[last_slice], empty_phase)
        _store_words(slab, base + half * CFG.SLICE_BYTES, row, words)
    nvvm.fence_proxy("async.shared", space="cta")
    nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
    if cutlass.const_expr(dk):
        bars.dk_full.arrive()
    else:
        bars.dv_full.arrive()


@cute.jit
def _epilogue_loop(
    slab,
    bars,
    sched,
    tmem_base,
    block,
    head,
    batch,
    sq,
    nk,
    nheads,
    batch_heads: cutlass.Constexpr[int],
    swizzle: cutlass.Constexpr[int],
    causal: cutlass.Constexpr[bool],
    dtype: cutlass.Constexpr,
):
    nvvm.setmaxregister(CFG.EPILOGUE_REGS, nvvm.SetMaxRegisterAction.INCREASE)
    row = cute.arch.thread_idx()[0] % CFG.M
    ds_state = PipelineState.start()
    dq_state = PipelineState.start()
    empty_state = PipelineState.start()
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit(bars.sched_empty[sched_state.idx])
        lo, hi = _bounds(block, sq, causal)
        for qtile in cutlass.range(lo, hi, 1, unroll=1):
            _wait_tcgen(bars.ds_ready, ds_state.phase)
            ds0 = _tmem_load(tmem_base + CFG.DP_DS_DQ_COL, CFG.PACK_GROUP_COLS, cutlass.Int32)
            ds1 = _tmem_load(tmem_base + (CFG.DP_DS_DQ_COL + CFG.D_HALF), CFG.PACK_GROUP_COLS, cutlass.Int32)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
            _wait(bars.dq_empty[0], empty_state.phase)
            _store_words(slab, CFG.ALIAS, row, ds0)
            _wait(bars.dq_empty[1], empty_state.phase)
            _store_words(slab, CFG.ALIAS + CFG.SLICE_BYTES, row, ds1)
            nvvm.fence_proxy("async.shared", space="cta")
            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
            bars.ds_smem.arrive()
            _wait_tcgen(bars.dq_done, dq_state.phase)
            dq0 = _tmem_load(tmem_base + CFG.DP_DS_DQ_COL, CFG.D_HALF)
            dq1 = _tmem_load(tmem_base + (CFG.DP_DS_DQ_COL + CFG.D_HALF), CFG.D_HALF)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
            bars.dp_empty.arrive()
            vals = cutlass.Vector.from_elements(tuple(dq0[i] for i in range(CFG.D_HALF)) + tuple(dq1[i] for i in range(CFG.D_HALF)), cutlass.Float32)
            for i in cutlass.range_constexpr(CFG.DQ_SLICES):
                _wait(bars.dq_empty[i], empty_state.phase)
                _store_dq_slice(slab, row, vals, i)
                nvvm.fence_proxy("async.shared", space="cta")
                bars.dq_full[i].arrive()
            ds_state = advance(ds_state, CFG.SINGLE_STAGE)
            dq_state = advance(dq_state, CFG.SINGLE_STAGE)
            empty_state = advance(empty_state, CFG.SINGLE_STAGE)
        if lo >= hi:
            _wait_tcgen(bars.dq_done, dq_state.phase)
            dq_state = advance(dq_state, CFG.SINGLE_STAGE)
        # Observe every terminal source-credit phase before the next CLC
        # owner bootstraps these barriers again. Native dV/dK alone wait on
        # slices 1/3, leaving slices 0/2 unobserved after the final dQ or a
        # zero-trip owner. Each completed mbarrier phase needs a successful
        # wait before an arrival in its next phase.
        for i in cutlass.range_constexpr(CFG.DQ_SLICES):
            _wait(bars.dq_empty[i], empty_state.phase)
        _native_epilogue(slab, tmem_base, bars, row, lo < hi, empty_state.phase, dtype, False)
        _native_epilogue(slab, tmem_base, bars, row, lo < hi, empty_state.phase, dtype, True)
        empty_state = advance(empty_state, CFG.SINGLE_STAGE)
        block, head, batch, valid, sched_state = _next_tile(sched, bars, sched_state, nheads, nk, causal, swizzle, batch_heads)


# Ported from kernels/sm100/bprop_d128_accum_f16.py:441-445 and449-532.
# Named CFG substitutions only; source operation/wait/phase order is unchanged.
@cute.jit
def _ts(desc, plane, bdesc, dst, accum):
    for k in cutlass.range_constexpr(CFG.D // CFG.K_HW):
        mma_ts_step(
            desc,
            nvvm.make_tmem_ptr(plane + (k // (CFG.D_HALF // CFG.K_HW)) * CFG.PACK_GROUP_COLS, cutlass.Int8),
            bdesc,
            dst,
            k_idx=k,
            accumulate=accum if cutlass.const_expr(k == 0) else True,
        )


@cute.jit
def _mma_loop(
    slab,
    bars,
    sched,
    tmem_base,
    block,
    head,
    batch,
    sq,
    nk,
    nheads,
    batch_heads: cutlass.Constexpr[int],
    swizzle: cutlass.Constexpr[int],
    causal: cutlass.Constexpr[bool],
    dtype: cutlass.Constexpr,
):
    nvvm.setmaxregister(CFG.SERVICE_REGS, nvvm.SetMaxRegisterAction.DECREASE)
    score = _mma_desc(dtype)
    grad = _mma_desc(dtype, btranspose=True)
    dqdesc = _mma_desc(dtype, atranspose=True, btranspose=True)
    kd = _operand(slab, CFG.K, dtype).desc()
    vd = _operand(slab, CFG.V, dtype).desc()
    dod = _operand(slab, CFG.DO, dtype).desc()
    dot = _operand(slab, CFG.DO, dtype, transpose=True).desc()
    dst = _operand(slab, CFG.ALIAS, dtype, transpose=True).desc()
    kt = _operand(slab, CFG.K, dtype, transpose=True).desc()
    s = nvvm.make_tmem_ptr(tmem_base, cutlass.Float32)
    dp = nvvm.make_tmem_ptr(tmem_base + CFG.DP_DS_DQ_COL, cutlass.Float32)
    dv = nvvm.make_tmem_ptr(tmem_base + CFG.DV_COL, cutlass.Float32)
    dk = nvvm.make_tmem_ptr(tmem_base + CFG.DK_COL, cutlass.Float32)
    q_state = PipelineState.start()
    do_state = PipelineState.start()
    kv_state = PipelineState.start()
    ready_state = PipelineState.start()
    dp_empty = PipelineState.start(phase=1)
    out_empty = PipelineState.start(phase=1)
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit(bars.sched_empty[sched_state.idx])
        lo, hi = _bounds(block, sq, causal)
        if lo < hi:
            _wait_tcgen(bars.k_full, kv_state.phase)
            _wait_tcgen(bars.q_full[q_state.idx], q_state.phase)
            qd = _operand(slab, CFG.Q + q_state.idx * CFG.OPERAND_BYTES, dtype).desc()
            mma_ss(score, kd, qd, s)
            _commit(bars.s_done)
            _wait_tcgen(bars.v_full, kv_state.phase)
        _wait_tcgen(bars.dvk_empty, out_empty.phase)
        out_empty = advance(out_empty, CFG.SINGLE_STAGE)
        for qtile in cutlass.range(lo, hi, 1, unroll=1):
            _wait_tcgen(bars.dp_empty, dp_empty.phase)
            _wait_tcgen(bars.do_full[0], do_state.phase)
            mma_ss(score, vd, dod, dp, k_start=0, k_count=CFG.D_HALF // CFG.K_HW)
            _wait_tcgen(bars.do_full[1], do_state.phase)
            mma_ss(score, vd, dod, dp, k_start=CFG.D_HALF // CFG.K_HW, k_count=CFG.D_HALF // CFG.K_HW, accumulate=True)
            _commit(bars.dp_done)
            _wait_tcgen(bars.p_ready, ready_state.phase)
            _ts(grad, tmem_base, dot, dv, qtile > lo)
            _commit(bars.do_empty)
            q_next = advance(q_state, CFG.Q_STAGES)
            if qtile + 1 < hi:
                _wait_tcgen(bars.q_full[q_next.idx], q_next.phase)
                qd_next = _operand(slab, CFG.Q + q_next.idx * CFG.OPERAND_BYTES, dtype).desc()
                mma_ss(score, kd, qd_next, s)
                _commit(bars.s_done)
            _wait_tcgen(bars.ds_ready, ready_state.phase)
            qt = _operand(slab, CFG.Q + q_state.idx * CFG.OPERAND_BYTES, dtype, transpose=True).desc()
            _ts(grad, tmem_base + CFG.DP_DS_DQ_COL, qt, dk, qtile > lo)
            _commit(bars.q_empty[q_state.idx])
            _wait_tcgen(bars.ds_smem, ready_state.phase)
            mma_ss(dqdesc, dst, kt, dp)
            _commit(bars.dq_done)
            q_state = q_next
            do_state = advance(do_state, CFG.SINGLE_STAGE)
            ready_state = advance(ready_state, CFG.SINGLE_STAGE)
            dp_empty = advance(dp_empty, CFG.SINGLE_STAGE)
        if lo < hi:
            _commit(bars.k_empty)
            _commit(bars.v_empty)
            kv_state = advance(kv_state, CFG.SINGLE_STAGE)
        else:
            _commit(bars.dq_done)
        block, head, batch, valid, sched_state = _next_tile(sched, bars, sched_state, nheads, nk, causal, swizzle, batch_heads)


@cute.jit
def _load_loop(
    slab,
    bars,
    sched,
    q_desc,
    k_desc,
    v_desc,
    do_desc,
    block,
    head,
    batch,
    sq,
    nk,
    nheads,
    batch_heads: cutlass.Constexpr[int],
    swizzle: cutlass.Constexpr[int],
    causal: cutlass.Constexpr[bool],
    dtype: cutlass.Constexpr,
):
    nvvm.setmaxregister(CFG.SERVICE_REGS, nvvm.SetMaxRegisterAction.DECREASE)
    q_state = PipelineState.start(phase=1)
    do_state = PipelineState.start(phase=1)
    kv_state = PipelineState.start(phase=1)
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit(bars.sched_empty[sched_state.idx])
        lo, hi = _bounds(block, sq, causal)
        if lo < hi:
            _wait(bars.k_empty, kv_state.phase)
            bars.k_full.arrive(n_bytes=CFG.OPERAND_BYTES, pred=nvvm.elect_sync())
            tma_load_tile(_operand(slab, CFG.K, dtype), GmemTileTma(k_desc)(0, block * CFG.M, head, batch), bars.k_full.smem_ptr)
            _wait(bars.v_empty, kv_state.phase)
            bars.v_full.arrive(n_bytes=CFG.OPERAND_BYTES, pred=nvvm.elect_sync())
            tma_load_tile(_operand(slab, CFG.V, dtype), GmemTileTma(v_desc)(0, block * CFG.M, head, batch), bars.v_full.smem_ptr)
            kv_state = advance(kv_state, CFG.SINGLE_STAGE)
        for qtile in cutlass.range(lo, hi, 1, unroll=1):
            _wait(bars.q_empty[q_state.idx], q_state.phase)
            bars.q_full[q_state.idx].arrive(n_bytes=CFG.OPERAND_BYTES, pred=nvvm.elect_sync())
            tma_load_tile(
                _operand(slab, CFG.Q + q_state.idx * CFG.OPERAND_BYTES, dtype),
                GmemTileTma(q_desc)(0, qtile * CFG.N, head, batch),
                bars.q_full[q_state.idx].smem_ptr,
            )
            _wait(bars.do_empty, do_state.phase)
            for half in cutlass.range_constexpr(CFG.DO_HALF_SLOTS):
                bars.do_full[half].arrive(n_bytes=CFG.SLICE_BYTES, pred=nvvm.elect_sync())
            for half in cutlass.range_constexpr(CFG.DO_HALF_SLOTS):
                tma_load_subtiles(
                    _operand(slab, CFG.DO, dtype), GmemTileTma(do_desc)(0, qtile * CFG.N, head, batch), bars.do_full[half].smem_ptr, half, CFG.SINGLE_STAGE
                )
            q_state = advance(q_state, CFG.Q_STAGES)
            do_state = advance(do_state, CFG.SINGLE_STAGE)
        block, head, batch, valid, sched_state = _next_tile(sched, bars, sched_state, nheads, nk, causal, swizzle, batch_heads)


@cute.jit
def tma_reduce_subtile(smem_tile, gmem_slice, subtile: cutlass.Constexpr[int]):
    """One elected source-sized FP32 add; caller owns commit and wait/credit."""
    desc = gmem_slice.tma_desc.get_ptr()
    coords = [gmem_slice.coord_d + cutlass.Int32(subtile * smem_tile.tma_granu_elems)] + list(gmem_slice.coords[1:])
    src = smem_tile.base.subview(subtile * smem_tile.tma_subtile_stride_elems)
    if nvvm.elect_sync():
        nvvm.cp_async_bulk_tensor_reduce(desc, src, "add", coords)


# DIAGNOSTIC lever ``diag_delete_dq_reduce_add`` (``Constexpr[bool]``, default False = the shipped kernel, rendered PTX-identical):
# the store warp skips the four fp32 dQ reduce-add issues per q tile and nothing else (every dq_full wait, commit, wait_group and
# dq_empty arrive stays: the barrier table is unchanged).  Armed, the OUTPUT IS WRONG BY DESIGN (dQ stays zero) -- a timing-only
# ceiling for the L2 reduce-add term; the dense MHA entry never arms it, the grouped dense kernel threads it from its adapter record.
# The THD twin's header (bprop_d128_accum_thd_f16.py) carries the full statement.
@cute.jit
def _store_loop(
    slab,
    bars,
    sched,
    dq_desc,
    dk_desc,
    dv_desc,
    block,
    head,
    batch,
    sq,
    nk,
    nheads,
    batch_heads: cutlass.Constexpr[int],
    swizzle: cutlass.Constexpr[int],
    causal: cutlass.Constexpr[bool],
    dtype: cutlass.Constexpr,
    diag_delete_dq_reduce_add: cutlass.Constexpr[bool] = False,
):
    nvvm.setmaxregister(CFG.SERVICE_REGS, nvvm.SetMaxRegisterAction.DECREASE)
    dq_state = PipelineState.start()
    out_state = PipelineState.start()
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit(bars.sched_empty[sched_state.idx])
        for i in cutlass.range_constexpr(CFG.DQ_SLICES):
            bars.dq_empty[i].arrive()  # all32 lanes, one bootstrap per KV tile
        lo, hi = _bounds(block, sq, causal)
        for qtile in cutlass.range(lo, hi, 1, unroll=1):
            for i in cutlass.range_constexpr(CFG.DQ_SLICES):
                _wait(bars.dq_full[i], dq_state.phase)
                # DIAGNOSTIC D2 (default off = the shipped code): only the reduce-add issue is deleted; commit / wait / arrives stay.
                if cutlass.const_expr(not diag_delete_dq_reduce_add):
                    tma_reduce_subtile(_operand(slab, CFG.ALIAS, cutlass.Float32, fp32=True), GmemTileTma(dq_desc)(0, qtile * CFG.N, head, batch), i)
                tma_store_commit()
            for i in cutlass.range_constexpr(CFG.DQ_SLICES):
                tma_store_wait((CFG.DQ_SLICES - 1) - i)
                bars.dq_empty[i].arrive()  # all32 after corresponding source read
            dq_state = advance(dq_state, CFG.SINGLE_STAGE)
        _wait(bars.dv_full, out_state.phase)
        # Exact backend distinction: native TMA store is warp-uniform, no elect.
        tma_store_tile(_operand(slab, CFG.ALIAS, dtype), GmemTileTma(dv_desc)(0, block * CFG.M, head, batch))
        _wait(bars.dk_full, out_state.phase)
        tma_store_tile(_operand(slab, CFG.ALIAS + CFG.OPERAND_BYTES, dtype), GmemTileTma(dk_desc)(0, block * CFG.M, head, batch))
        out_state = advance(out_state, CFG.SINGLE_STAGE)
        tma_store_commit()
        tma_store_wait(0)
        bars.dvk_empty.arrive()
        block, head, batch, valid, sched_state = _next_tile(sched, bars, sched_state, nheads, nk, causal, swizzle, batch_heads)


@cute.jit
def _scheduler_loop(bars, sched):
    nvvm.setmaxregister(CFG.SERVICE_REGS, nvvm.SetMaxRegisterAction.DECREASE)
    state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        _wait(bars.sched_empty[state.idx], state.phase)
        if nvvm.elect_sync():
            bars.sched_full[state.idx].arrive(n_bytes=CFG.SCHED_TX_BYTES)
            nvvm.clusterlaunchcontrol_try_cancel(
                sched.tile_id_smem.subview(state.idx * (CFG.SCHED_PAYLOAD_STAGE_BYTES // CFG.BPE_ACC)), bars.sched_full[state.idx].smem_ptr, multicast=1
            )
        nvvm.fence_proxy("async.shared", space="cta")
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
        _wait(bars.sched_full[state.idx], state.phase)
        _, _, valid = read_clc_payload(sched, state.idx * (CFG.SCHED_PAYLOAD_STAGE_BYTES // CFG.BPE_ACC), warp_broadcast=True)
        state = advance(state, CFG.SCHED_STAGES)


@cute.kernel
def bprop_d128_accum_sm107_kernel(
    q_desc: cutlass.GridConstant[tmap.TensorMap],
    k_desc: cutlass.GridConstant[tmap.TensorMap],
    v_desc: cutlass.GridConstant[tmap.TensorMap],
    do_desc: cutlass.GridConstant[tmap.TensorMap],
    dq_desc: cutlass.GridConstant[tmap.TensorMap],
    dk_desc: cutlass.GridConstant[tmap.TensorMap],
    dv_desc: cutlass.GridConstant[tmap.TensorMap],
    lse: cute.Tensor,
    delta: cute.Tensor,
    sq: cutlass.Int32,
    skv: cutlass.Int32,
    nheads: cutlass.Int32,
    batch_heads: cutlass.Constexpr[int],
    scale: cutlass.Float32,
    causal: cutlass.Constexpr[bool],
    swizzle: cutlass.Constexpr[int],
    dtype: cutlass.Constexpr,
):
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    x, y, z = cute.arch.block_idx()
    nk = (skv + CFG.M - 1) // CFG.M
    if cutlass.const_expr(causal):
        block, head, batch = _decode_flat(x, nheads, nk, swizzle, batch_heads)
    else:
        block, head, batch = x, y, z
    slab = cutlass.Array(cutlass.Uint8, CFG.SMEM_BYTES, alignment=CFG.SMEM_ALIGNMENT, space=cutlass.AddressSpace.smem)
    bars = _bars(slab)
    tmem_slot = _view(slab, CFG.TMEM_HANDLE, cutlass.Int32, CFG.SINGLE_STAGE, alignment=CFG.BPE_ACC)
    sched_words = _view(slab, CFG.SCHED_PAYLOAD, cutlass.Int32, CFG.SCHED_PAYLOAD_WORDS, alignment=CFG.BARRIER_ALIGNMENT)
    sched = Sched(bars.sched_full.base_ptr, bars.sched_empty.base_ptr, sched_words, x, y, z)
    if warp == 0:
        # All 512 TMEM columns belong to this CTA; use the portable allocation form.
        nvvm.tcgen05_alloc(tmem_slot, cutlass.Int32(CFG.TMEM_COLS), group=nvvm.CTAGroup.CTA_1)
        nvvm.tcgen05_relinquish_alloc_permit(group=nvvm.CTAGroup.CTA_1)
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
    _init_and_publish(bars, warp)
    tmem_base = tmem_slot.load()
    if warp < CFG.SOFTMAX_WARPS:
        _softmax_loop(slab, bars, sched, tmem_base, lse, delta, block, head, batch, sq, skv, nk, nheads, batch_heads, swizzle, scale, causal, dtype)
    elif warp < CFG.MMA_WARP:
        _epilogue_loop(slab, bars, sched, tmem_base, block, head, batch, sq, nk, nheads, batch_heads, swizzle, causal, dtype)
    elif warp == CFG.MMA_WARP:
        _mma_loop(slab, bars, sched, tmem_base, block, head, batch, sq, nk, nheads, batch_heads, swizzle, causal, dtype)
    elif warp == CFG.LOAD_WARP:
        _load_loop(slab, bars, sched, q_desc, k_desc, v_desc, do_desc, block, head, batch, sq, nk, nheads, batch_heads, swizzle, causal, dtype)
    elif warp == CFG.STORE_WARP:
        _store_loop(slab, bars, sched, dq_desc, dk_desc, dv_desc, block, head, batch, sq, nk, nheads, batch_heads, swizzle, causal, dtype)
    else:
        _scheduler_loop(bars, sched)
    nvvm.barrier_cta_sync()
    if warp == 0:
        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
        nvvm.tcgen05_dealloc(nvvm.make_tmem_ptr(tmem_slot.load(), cutlass.Int8), cutlass.Int32(CFG.TMEM_COLS), group=nvvm.CTAGroup.CTA_1)


bprop_d128_accum_sm107_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def host(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    do: cute.Tensor,
    lse: cute.Tensor,
    delta: cute.Tensor,
    dq: cute.Tensor,
    dk: cute.Tensor,
    dv: cute.Tensor,
    scale: cutlass.Float32,
    causal: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    if cutlass.const_expr(q.element_type not in (cutlass.BFloat16, cutlass.Float16)):
        raise ValueError("dense d128 accumulation requires BF16/FP16")
    if cutlass.const_expr(q.shape[3] != CFG.D or k.shape[3] != CFG.D or q.shape[2] != k.shape[2]):
        raise ValueError("dense d128 accumulation requires dense MHA d128")
    if cutlass.const_expr(q.shape[1] <= 0 or k.shape[1] <= 0):
        raise ValueError("dense d128 accumulation requires positive Sq/Skv")
    nk = (k.shape[1] + CFG.M - 1) // CFG.M
    heads_per_l2 = CFG.SOURCE_SCHED_L2_BYTES // (q.shape[1] * (CFG.D + CFG.D) * CFG.BPE)
    swizzle = 1 << (heads_per_l2.bit_length() - 1) if heads_per_l2 > 0 else 1
    grid = (nk * q.shape[2] * q.shape[0], 1, 1) if causal else (nk, q.shape[2], q.shape[0])
    bprop_d128_accum_sm107_kernel(
        _tensor_map(q, CFG.HEAD_BOX),
        _tensor_map(k, CFG.HEAD_BOX),
        _tensor_map(v, CFG.HEAD_BOX),
        _tensor_map(do, CFG.HEAD_BOX),
        _tensor_map(dq, CFG.DQ_BOX),
        _tensor_map(dk, CFG.HEAD_BOX),
        _tensor_map(dv, CFG.HEAD_BOX),
        lse,
        delta,
        q.shape[1],
        k.shape[1],
        q.shape[2],
        q.shape[0] * q.shape[2],
        scale,
        causal,
        swizzle,
        q.element_type,
    ).launch(grid=grid, block=(CFG.THREADS, 1, 1), stream=stream, min_blocks_per_mp=CFG.MIN_BLOCKS_PER_MP)

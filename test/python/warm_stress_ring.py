# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The self-test ring of ``warm_stress`` and of ``cudnn.frost.tile_dsl.consume``: one kernel, a compile-time release form.

A zero-math ring.  Per thread block: warp 0 lane 0 refills one of two 4,096-byte stages with a bulk copy from global memory;
the 32 lanes of warp 1 wait for the fill, load five 16-byte vectors each, pass a named barrier of the 32 lanes, RELEASE the
stage, and only then use what they loaded (one Int32 sum per lane over all fills is the output).  Nothing a lane loads feeds
an address, a loop bound or a branch, so wrong data cannot hang or fault the kernel.

``RELEASE`` selects how the stage is released:

* the forms of ``tile_dsl.consume`` (``HELPER_FORMS``): the release depends on every loaded word and the ring must read 0
  wrong bytes in every launch;
* ``plain``: an elected plain arrive.  THE UNSAFE TWIN -- the same kernel with the loads unread at the release.  It is the
  positive control of ``warm_stress``: where it fails, the stress can see the defect class on that GPU.  A data-race
  checker reports this form by construction; it is not to be "fixed";
* ``folded_zero`` and ``guarded``: the helper's chain written inline with a mask the toolchain can see through (an inline-asm
  zero; the kernel argument under an entry guard).  Both lose the dependence without any error and are kept as controls of
  the two sentences in the contract that forbid them.

Barrier table (per thread block; the only initialising lane is thread 0, then ``fence.mbarrier_init`` and a CTA barrier):

    full[2]    init 1                          the producer's expect_tx arrive (warp 0, lane 0) + the copy's bytes
    empty[2]   init SUM(issuing lanes x count) of the consumer warp(s) per phase: 1 for an elected arrive, 32 for every
               lane, x COUNT for the counted forms, 2 for the scheduler credit in clusters of two (each warp credits both
               thread blocks); per form in HELPER_FORMS below, 1 for the three control forms

Geometry: every lane of one vector load reads the same shared-memory banks (lanes 128 bytes apart) and the default grid is
three thread blocks per SM.  The unsafe twin was measured to fail in every launch in this geometry and in none with lanes
16 bytes apart or with one thread block per SM; keep both properties when changing the ring.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.experimental.primitives as nvvm

from cudnn.frost.tile_dsl.consume import consume_then_arrive, consume_then_arrive_on_peer, consume_then_read_tile_id_arrive, release_zero

STAGES = 2
LOADS = 5  # 16-byte vector loads per lane and fill
VEC_WORDS = 4
WORDS_PER_RELEASE = LOADS * VEC_WORDS
FILL_WORDS = 1024  # Int32 per fill: 4,096 bytes
LANE_WORDS = 32  # lane pitch: 128 bytes, one bank cycle
THREADS = 64
FILLS = 16  # fills per thread block and launch; each stage is refilled FILLS // STAGES - 1 times
BLOCKS_PER_SM = 3
COUNT = 2  # the counted forms arrive twice per issuing lane

CONTROL_FORMS = ("plain", "folded_zero", "guarded")
# form -> (cluster size the form needs, arrivals on each thread block's empty barrier per phase)
HELPER_FORMS = {
    "elected": (1, 1),  # consume_then_arrive(pred=elect): one lane
    "all": (1, 32),  # consume_then_arrive(): every lane of the warp
    "counted": (1, COUNT),  # consume_then_arrive(pred=elect, count=COUNT)
    "counted_all": (1, 32 * COUNT),  # consume_then_arrive(count=COUNT)
    "peer": (2, 1),  # consume_then_arrive_on_peer(own rank, pred=elect): the release-cta cluster arrive
    "peer_all": (2, 32),  # consume_then_arrive_on_peer(own rank): every lane
    "peer_relaxed": (2, 1),  # consume_then_arrive_on_peer(own rank, pred=elect, release_cta=False)
    "sched1": (1, 1),  # consume_then_read_tile_id_arrive(cga 1)
    "sched2": (2, 2),  # consume_then_read_tile_id_arrive(cga 2): each warp credits BOTH thread blocks of the cluster
}
if 31 * LANE_WORDS + WORDS_PER_RELEASE > FILL_WORDS:
    raise ValueError("the lanes' loads do not fit one fill")


def cluster_of(release: str, cluster: int = 0) -> int:
    """The cluster size a form launches with; a control form takes the caller's (the twin of a cluster form runs in clusters)."""
    if release in HELPER_FORMS:
        return HELPER_FORMS[release][0]
    if release not in CONTROL_FORMS:
        raise ValueError(f"unknown release form {release!r}")
    return cluster if cluster else 1


def _empty_arrivals(release: str) -> int:
    return HELPER_FORMS[release][1] if release in HELPER_FORMS else 1


@cute.kernel
def warm_stress_ring_kernel(
    mX: cute.Tensor,
    mOut: cute.Tensor,
    fills: cutlass.Int32,
    rz_arg: cutlass.Int32,
    RELEASE: cutlass.Constexpr[str],
    AMPLIFY: cutlass.Constexpr[bool],
    CLUSTER: cutlass.Constexpr[int],
):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    warp = tid // 32
    lane = tid % 32
    rank = cutlass.Int32(cute.arch.block_idx_in_cluster()) if cutlass.const_expr(CLUSTER > 1) else cutlass.Int32(0)
    rz = release_zero(rz_arg) if cutlass.const_expr(RELEASE in HELPER_FORMS) else None

    stage = cutlass.Array(cutlass.Int32, STAGES * FILL_WORDS, alignment=1024, space=cutlass.AddressSpace.smem)
    mbar = cutlass.Array(cutlass.Int64, 2 * STAGES, alignment=8, space=cutlass.AddressSpace.smem)  # full[0:2], empty[2:4]
    late = cutlass.Array(cutlass.Int32, 4, alignment=16, space=cutlass.AddressSpace.smem)  # AMPLIFY: a zero word loaded behind the release

    if tid == 0:
        for j in cutlass.range_constexpr(STAGES):
            nvvm.mbarrier_init(mbar.subview(j), 1)
            nvvm.mbarrier_init(mbar.subview(STAGES + j), _empty_arrivals(RELEASE))
        if cutlass.const_expr(AMPLIFY):
            late.subview(0).store(cutlass.Int32(0))
    nvvm.fence_mbarrier_init()
    nvvm.barrier_cta_sync()
    if cutlass.const_expr(CLUSTER > 1):
        # every thread block's barriers exist before a cross-block arrive can land on them
        nvvm.barrier_cluster_arrive_relaxed_aligned()
        nvvm.barrier_cluster_wait_aligned()

    # the `guarded` control puts the consumer under an entry guard on the kernel argument; every other form compares constants
    guard = cutlass.Int32(0)
    if cutlass.const_expr(RELEASE == "guarded"):
        guard = rz_arg

    if warp == 0:
        if lane == 0:
            i = cutlass.Int32(0)
            while i < fills:
                s = i % STAGES
                if i >= STAGES:
                    # the stage was released by the consumer's arrive(s) of fill i - STAGES
                    while not nvvm.mbarrier_try_wait_parity(mbar.subview(STAGES + s), ((i // STAGES) - 1) & 1):
                        pass
                nvvm.mbarrier_arrive_expect_tx(mbar.subview(s), FILL_WORDS * 4)
                src = mX.iterator + cutlass.Int64(bid * fills + i) * FILL_WORDS
                nvvm.cp_async_bulk_shared_cluster_global(stage.subview(s * FILL_WORDS), src, mbar.subview(s), FILL_WORDS * 4)
                i = i + 1
    elif warp == 1:
        if guard == cutlass.Int32(0):
            acc = cutlass.Int32(0)
            i = cutlass.Int32(0)
            while i < fills:
                s = i % STAGES
                while not nvvm.mbarrier_try_wait_parity(mbar.subview(s), (i // STAGES) & 1):
                    pass
                vecs = []
                words = []
                for j in cutlass.range_constexpr(LOADS):
                    vec = stage.load(s * FILL_WORDS + lane * LANE_WORDS + j * VEC_WORDS, vector_size=VEC_WORDS, alignment=16)
                    vecs.append(vec)
                    for e in cutlass.range_constexpr(VEC_WORDS):
                        words.append(vec[e])
                nvvm.barrier_cta_sync(1, thread_count=32)
                mb_empty = mbar.subview(STAGES + s)

                # ---- the release ----
                if cutlass.const_expr(RELEASE == "plain"):
                    nvvm.inline_ptx("mbarrier.arrive.shared::cta.b64 _, [{$r0}];", read_only_args=[mb_empty], predicate=nvvm.elect_sync())
                elif cutlass.const_expr(RELEASE in ("folded_zero", "guarded")):
                    fold = words[0]
                    for w in words[1:]:
                        fold = fold + w
                    mask = rz_arg if cutlass.const_expr(RELEASE == "guarded") else nvvm.inline_ptx("mov.b32 $0, 0;", write_only_types=[cutlass.Int32])
                    nvvm.inline_ptx(
                        "mbarrier.arrive.shared::cta.b64 _, [{$r0}];",
                        read_only_args=[mbar.subview(STAGES + s + (fold & mask))],
                        predicate=nvvm.elect_sync(),
                    )
                elif cutlass.const_expr(RELEASE == "elected"):
                    consume_then_arrive(mb_empty, vecs, rz, pred=nvvm.elect_sync())
                elif cutlass.const_expr(RELEASE == "all"):
                    consume_then_arrive(mb_empty, words, rz)
                elif cutlass.const_expr(RELEASE == "counted"):
                    consume_then_arrive(mb_empty, (vecs[0], (vecs[1], vecs[2]), [vecs[3], vecs[4]]), rz, pred=nvvm.elect_sync(), count=COUNT)
                elif cutlass.const_expr(RELEASE == "counted_all"):
                    consume_then_arrive(mb_empty, vecs, rz, count=COUNT)
                elif cutlass.const_expr(RELEASE == "peer"):
                    consume_then_arrive_on_peer(mb_empty, rank, vecs, rz, pred=nvvm.elect_sync())
                elif cutlass.const_expr(RELEASE == "peer_all"):
                    consume_then_arrive_on_peer(mb_empty, rank, vecs, rz)
                elif cutlass.const_expr(RELEASE == "peer_relaxed"):
                    consume_then_arrive_on_peer(mb_empty, rank, vecs, rz, pred=nvvm.elect_sync(), release_cta=False)
                elif cutlass.const_expr(RELEASE == "sched1"):
                    consume_then_read_tile_id_arrive(mb_empty, 1, vecs, rz)
                else:
                    consume_then_read_tile_id_arrive(mb_empty, CLUSTER, vecs, rz)

                # ---- the use ----
                if cutlass.const_expr(AMPLIFY):
                    zero_late = late.load(0)
                    for w in words:
                        acc = acc + (w ^ zero_late)
                else:
                    for w in words:
                        acc = acc + w
                i = i + 1
            mOut[bid * 32 + lane] = acc
    if cutlass.const_expr(CLUSTER > 1):
        # a cross-block arrive must find its target resident: no thread block leaves before the cluster is done
        nvvm.barrier_cluster_arrive_relaxed_aligned()
        nvvm.barrier_cluster_wait_aligned()


warm_stress_ring_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def warm_stress_ring_launch(
    x: cute.Tensor,
    out: cute.Tensor,
    fills: cutlass.Int32,
    blocks: cutlass.Int32,
    RELEASE: cutlass.Constexpr[str],
    AMPLIFY: cutlass.Constexpr[bool],
    CLUSTER: cutlass.Constexpr[int],
    stream: cuda.CUstream,
):
    # cutlass.Int32(0) is the release-zero: the only value this host ever passes
    if cutlass.const_expr(CLUSTER > 1):
        warm_stress_ring_kernel(x, out, fills, cutlass.Int32(0), RELEASE, AMPLIFY, CLUSTER).launch(
            grid=(blocks, 1, 1), block=(THREADS, 1, 1), cluster=(CLUSTER, 1, 1), stream=stream
        )
    else:
        warm_stress_ring_kernel(x, out, fills, cutlass.Int32(0), RELEASE, AMPLIFY, CLUSTER).launch(grid=(blocks, 1, 1), block=(THREADS, 1, 1), stream=stream)


@cute.kernel
def warm_stress_poison_kernel(pattern: cutlass.Int32, WORDS: cutlass.Constexpr[int]):
    """Overwrite this thread block's whole shared-memory allocation with ``pattern``; touches nothing else."""
    tid, _, _ = cute.arch.thread_idx()
    pool = cutlass.Array(cutlass.Int32, WORDS, alignment=16, space=cutlass.AddressSpace.smem)
    i = cutlass.Int32(tid)
    while i < WORDS:
        pool.subview(i).store(pattern)
        i = i + THREADS


warm_stress_poison_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def warm_stress_poison_launch(pattern: cutlass.Int32, blocks: cutlass.Int32, WORDS: cutlass.Constexpr[int], stream: cuda.CUstream):
    warm_stress_poison_kernel(pattern, WORDS).launch(grid=(blocks, 1, 1), block=(THREADS, 1, 1), stream=stream)


def ring_blocks(sm_count: int, cluster: int = 1, blocks_per_sm: int = BLOCKS_PER_SM) -> int:
    """Thread blocks of one ring launch: ``blocks_per_sm`` per SM, rounded up to whole clusters."""
    blocks = sm_count * blocks_per_sm
    return ((blocks + cluster - 1) // cluster) * cluster


def ring_input(blocks: int, fills: int, device, seed: int = 20261111):
    """``(x, reference)``: the fills of every thread block and the independent expected output.

    ``x`` is Int32 ``[blocks * fills * FILL_WORDS]``.  Lane ``l`` of block ``b`` loads the words
    ``[l * LANE_WORDS, l * LANE_WORDS + WORDS_PER_RELEASE)`` of each of its fills; ``reference[b * 32 + l]`` is their Int32
    (wrapping) sum over all fills, computed with torch from ``x`` alone.
    """
    import torch

    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randint(-(2**31), 2**31 - 1, (blocks * fills * FILL_WORDS,), dtype=torch.int32, generator=gen).to(device)
    index = (torch.arange(32).view(-1, 1) * LANE_WORDS + torch.arange(WORDS_PER_RELEASE).view(1, -1)).reshape(-1).to(device)
    picked = x.view(blocks, fills, FILL_WORDS)[:, :, index].view(blocks, fills, 32, WORDS_PER_RELEASE)
    reference = picked.to(torch.int64).sum(dim=(1, 3)).to(torch.int32).reshape(-1).contiguous()
    return x, reference


def locate(_name: str, byte_offset: int) -> str:
    """Where an output byte of the ring comes from."""
    record = byte_offset // 4
    return f"thread block {record // 32}, lane {record % 32}"


class Ring:
    """One compiled ring form bound to its inputs: ``launch()`` is ONE launch on the stream it was built for."""

    def __init__(
        self, release: str, *, device=None, amplify: bool = True, cluster: int = 0, fills: int = FILLS, blocks_per_sm: int = BLOCKS_PER_SM, blocks: int = 0
    ):
        import torch
        from cutlass.cute.runtime import from_dlpack

        self.release = release
        self.amplify = bool(amplify)
        self.cluster = cluster_of(release, cluster)
        self.fills = int(fills)
        self.device = torch.device("cuda") if device is None else torch.device(device)
        self.sm_count = torch.cuda.get_device_properties(self.device).multi_processor_count
        self.blocks = int(blocks) if blocks else ring_blocks(self.sm_count, self.cluster, blocks_per_sm)
        if self.blocks % self.cluster:
            raise ValueError(f"{self.blocks} thread blocks do not fill clusters of {self.cluster}")
        if self.fills < 2 * STAGES:
            raise ValueError("the ring needs at least two fills per stage: a stage that is never refilled cannot show the defect")
        self.x, self.reference = ring_input(self.blocks, self.fills, self.device)
        self.out = torch.zeros(self.blocks * 32, dtype=torch.int32, device=self.device)
        self._x = from_dlpack(self.x, assumed_align=16)
        self._out = from_dlpack(self.out, assumed_align=16)
        with torch.cuda.device(self.device):
            self._stream = cuda.CUstream(torch.cuda.current_stream(self.device).cuda_stream)
            self._fn = cute.compile(warm_stress_ring_launch, self._x, self._out, self.fills, self.blocks, release, self.amplify, self.cluster, self._stream)

    @property
    def refills_per_launch(self) -> int:
        """Refills of one stage inside one launch, per thread block."""
        return self.fills // STAGES - 1

    def launch(self) -> None:
        self._fn(self._x, self._out, self.fills, self.blocks, self._stream)


class Poison:
    """A launch that overwrites every SM's shared memory: ``blocks_per_sm`` thread blocks per SM, each filling ``smem_bytes``."""

    def __init__(self, *, device=None, smem_bytes: int = 0, blocks_per_sm: int = 4):
        import torch

        self.device = torch.device("cuda") if device is None else torch.device(device)
        index = self.device.index if self.device.index is not None else torch.cuda.current_device()
        if not smem_bytes:
            (err,) = cuda.cuInit(0)
            err, handle = cuda.cuDeviceGet(index)
            err, smem_bytes = cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, handle)
            if int(err) != 0 or smem_bytes <= 0:
                raise RuntimeError(f"cannot read the device's shared memory per thread block (error {err})")
        self.words = (int(smem_bytes) // 4 // THREADS) * THREADS
        self.blocks = torch.cuda.get_device_properties(self.device).multi_processor_count * blocks_per_sm
        with torch.cuda.device(self.device):
            self._stream = cuda.CUstream(torch.cuda.current_stream(self.device).cuda_stream)
            self._fn = cute.compile(warm_stress_poison_launch, cutlass.Int32(0), self.blocks, self.words, self._stream)

    def launch(self, pattern: int = 0) -> None:
        self._fn(pattern, self.blocks, self._stream)

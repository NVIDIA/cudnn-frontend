# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""SM100/SM107 f16 forward launches, prepared once and bound per call.

Three owners, one implementation each:

* **Observation** — :class:`BufferFacts`: what the caller's buffer is (address, dtype, device,
  the element span the producer guarantees, shape / strides). Read from the graph's normalized
  ``VariantPack`` (:func:`facts_of_roles`) or from a torch tensor handed to the standalone
  adapter (:func:`facts_of_tensor`). Nothing downstream looks at a buffer object again.
* **Semantics** — :class:`ThdLaunchSpec`, built by the adapter after ``compile()``: the
  positional argument template of the explicit host entry with every plan constant filled,
  the argument slot of every runtime field, the per-operand rules (dtype, alignment, extent),
  the capacity formulas, the workspace regions, the declared per-call operation (padded-Stats
  ``-inf`` seed) and the read-only dummies it owns.
* **Binding** — :func:`bind_thd`: applies the spec's rules to this call's facts and returns an
  independent argument frame (or None when no Q token is addressable). Lookups, integer
  arithmetic and writes; no ``cute`` objects, no torch views, no device allocation, no compile.

The graph plan (:class:`PreparedThdLaunch`) and the adapter's ``execute()`` both go through
``bind_thd`` and the artifact's positional tvm-ffi entry.

Dense launches use :class:`DenseLaunchSpec` and :func:`bind_dense`. A split plan adds an
immutable :class:`SplitCombineSpec`; :func:`bind_dense_split` binds the caller's workspace
and final outputs before either launch. Partial LSE remains natural-log even when final
Stats are absent or use log2. Every execution owns both argument frames.
"""

from __future__ import annotations

import inspect
import math
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Tuple

from cudnn.frost import buffers as _buffers
from cudnn.frost.compiled_cache import positional_entry

_ALIGN_TMA = 16
_ALIGN_F32 = 4
_DLPACK_CUDA = 2
_DLPACK_CPU = 1
_DTYPE_BY_CODE = {(code, bits): name for name, (code, bits) in _buffers.DTYPES.items()}


class BufferFacts(NamedTuple):
    """What a caller buffer is, observed once (see the module docstring)."""

    ptr: int
    dtype: str  # bare name ("bfloat16"); "" when unknown
    device: Tuple[int, int]  # DLPack (device_type, device_id); (-1, -1) unknown
    span: int  # element span the producer guarantees, in the DECLARED element width; -1 unknown (a bare address)
    shape: Tuple[int, ...]
    strides: Tuple[int, ...]

    @property
    def numel(self) -> int:
        n = 1
        for e in self.shape:
            n *= int(e)
        return n

    @property
    def contiguous(self) -> bool:
        return _buffers.is_contiguous(self.shape, self.strides)


def facts_of_tensor(t) -> Optional[BufferFacts]:
    """Facts of a torch tensor (the standalone adapter's operands); None for None."""
    if t is None:
        return None
    shape, strides = tuple(t.shape), tuple(t.stride())
    n = 1
    for e in shape:
        n *= int(e)
    span = n if (n == 0 or t.is_contiguous()) else 1 + sum((int(s) - 1) * int(st) for s, st in zip(shape, strides))
    dev = t.device
    device = (_DLPACK_CUDA, int(dev.index if dev.index is not None else 0)) if dev.type == "cuda" else (_DLPACK_CPU, 0)  # a known CPU tensor is not "unknown"
    return BufferFacts(t.data_ptr(), str(t.dtype).split(".")[-1], device, span, shape, strides)


def facts_of_roles(pack, indices: List[int]) -> List[BufferFacts]:
    """Project normalized operands into immutable facts in one native crossing.

    Effective dtype/geometry follow graph declarations and overrides; span and
    device retain the producer's observations. No operand objects are built.
    """
    return pack.native._facts_as(indices, BufferFacts, _DTYPE_BY_CODE)


class ThdLaunchSpec:
    """Plan-time facts of one THD f16 launch (built by :func:`build_thd_spec`); read-only after
    build except for its bounded cache of immutable validated geometry."""

    __slots__ = (
        "fn",
        "owner",
        "order",
        "index",
        "template",
        "b",
        "qh",
        "kh",
        "d_qk",
        "d_v",
        "paged",
        "page_size",
        "paged_hnd",
        "decl",
        "expect",
        "has_lse",
        "has_sink",
        "lse_padded",
        "lse_head_major",
        "lse_head_stride",
        "lse_stride",
        "s_q_max",
        "total_q",
        "total_kv",
        "n_q_lens",
        "n_kv_lens",
        "lens_form",
        "off_o_desc",
        "scratch_bytes",
        "neg_inf",
        "device_index",
        "_dummies",
        "_geometry_cache",
    )

    def frame(self) -> List[Any]:
        return list(self.template)

    def dummy(self, key: str) -> int:
        """Address of a zero-filled read-only device buffer owned by this spec; every one the frame can
        need is allocated and initialized at build (:func:`build_thd_spec`), never during execute."""
        return self._dummies[key].data_ptr()


def _zeroed_device_buffer(nbytes: int, device_index: int) -> "_buffers.DeviceBuffer":
    """A ``cuMemAlloc`` buffer zeroed synchronously (cuMemsetD32 + stream sync at build): ready for
    whatever stream later reads it."""
    from cuda.bindings import driver as _drv

    buf = _buffers.DeviceBuffer(int(nbytes), device_index)
    (err,) = _drv.cuMemsetD32(buf.data_ptr(), 0, (int(nbytes) + 3) // 4)
    if int(err) != 0:
        raise RuntimeError(f"cudnn.sdpa: cuMemsetD32 failed: {err}")
    (err,) = _drv.cuStreamSynchronize(_drv.CUstream(0))
    if int(err) != 0:
        raise RuntimeError(f"cudnn.sdpa: cuStreamSynchronize failed: {err}")
    return buf


# The host slot vocabulary the prepared launch binds: constants written once at build, and slots bind_thd writes per call.
_FILLED_AT_BUILD = frozenset(
    "q_strides o_strides k_strides v_strides lse_strides lse_ext scale_softmax_log2 n_thd_units seq_q_lens_addr thd_lens_form o_partial_ptr "
    "block_table_ptr block_table_v_ptr table_strides n_pages gate_ptr gate_strides".split()
)
_FILLED_PER_CALL = frozenset(
    "q_ptr k_ptr v_ptr o_ptr lse_ptr sinks_ptr meta_ptr o_desc_ptr problem_size k_strides v_strides lse_ext n_pages thd_q_lens_ptr thd_kv_lens_ptr "
    "block_table_ptr block_table_v_ptr table_strides stream".split()
)


def _compiled_paged_hnd(api) -> bool:
    """The in-page layout kind the artifact was compiled for (``paged_hnd``: the row stride below the head
    stride in the kernel's (page, row, head, d) order), read the way ``_explicit_compile_kwargs`` derived it."""
    ps = api._paged_pool_stride(api.k_desc)
    return bool(int(ps[1]) < int(ps[2]))


def _pool_is_hnd(f: BufferFacts) -> bool:
    """A (n_pages, KH, page_size, D) pool container is HND when its row stride is below its head stride."""
    return int(f.strides[2]) < int(f.strides[1])


def _covering(shape: Tuple[int, ...], strides: Tuple[int, ...]) -> bool:
    """No two distinct indices of ``shape`` address the same element: sorted by stride, each stride covers
    the extents below it (extent-1 axes are free)."""
    axes = sorted((int(st), int(sz)) for sz, st in zip(shape, strides) if int(sz) > 1)
    need = 1
    for st, sz in axes:
        if st < need:
            return False
        need = st * sz
    return True


def _positional_order(api) -> Tuple[Any, Any, List[str]]:
    """``(raw positional entry, compiled artifact, host slot order)`` of the adapter's compiled kernel;
    NotImplementedError when the artifact has no positional entry or the host is not the expected shape."""
    km = api._k_mod
    compiled = api._compiled_kernel
    raw = positional_entry(compiled)
    if raw is None:
        raise NotImplementedError("the compiled artifact exposes no positional tvm-ffi entry")
    order = [n for n, p in inspect.signature(km._host).parameters.items() if "Constexpr" not in str(p.annotation)]
    if order[-1] != "stream":
        raise NotImplementedError(f"{km.__name__}: the host entry does not end with the stream parameter: {order[-3:]}")
    wrapper_sig = getattr(compiled, "_kwargs_wrapper", None) or compiled
    try:
        seen = list(inspect.signature(wrapper_sig).parameters)
    except (TypeError, ValueError):
        seen = None
    if seen is not None and seen != order:
        raise NotImplementedError(f"{km.__name__}: positional ABI {order} disagrees with the compiled wrapper {seen}")
    return raw, compiled, order


def build_thd_spec(api, *, scale_softmax: Optional[float]) -> ThdLaunchSpec:
    """The adapter's plan-time facts as a :class:`ThdLaunchSpec`; NotImplementedError when the
    artifact has no positional entry or the host signature is not the expected shape."""
    km = api._k_mod
    raw, compiled, order = _positional_order(api)
    plan = api._thd_plan()
    s = ThdLaunchSpec()
    s.fn, s.owner, s.order = raw, compiled, order
    s.index = {n: i for i, n in enumerate(order)}
    s.b, s.qh, s.kh, s.d_qk, s.d_v = api.batch_size, api.h_q, api.h_kv, api.head_dim_qk, api.head_dim_v
    s.paged, s.page_size = bool(api.paged), int(api.paged_page_size or 0)
    s.paged_hnd = _compiled_paged_hnd(api) if s.paged else False
    s.decl = dict(q=plan.q, k=plan.k, v=plan.v, o=plan.o)  # (h, d, token_stride, head_stride, elem_stride, row_span)
    s.expect = {n: str(getattr(api, f"{n}_desc").dtype).split(".")[-1] for n in ("q", "k", "v", "o")}
    s.has_lse, s.has_sink = api.lse_desc is not None, bool(api.has_sink)
    s.lse_padded, s.lse_head_major = bool(api.thd_stats_padded), bool(api.thd_stats_head_major)
    s.lse_head_stride = int(api.thd_stats_head_stride or 0)
    s.lse_stride = tuple(int(x) for x in api._lse_stride) if s.lse_padded else None
    s.s_q_max = int(api.s_q_max)
    s.total_q = None if plan.total_q is None else int(plan.total_q)
    s.total_kv = None if plan.total_kv is None else int(plan.total_kv)
    s.n_q_lens, s.n_kv_lens, s.lens_form = int(plan.n_q_lens), int(plan.n_kv_lens), int(plan.lens_form)
    s.off_o_desc, s.scratch_bytes = int(plan.off_o_desc), int(plan.scratch_bytes)
    s.neg_inf = _buffers.init_word("fp32", float("-inf"))
    s.device_index = int(api.q_desc.device.index or 0)
    s._dummies = {}
    s._geometry_cache = None
    if not s.has_sink:
        s._dummies["sinks"] = _zeroed_device_buffer(s.qh * 4, s.device_index)
    if not s.paged:  # the all-KV-zero clamp's V stub: one (kh, d_v) row of zeros
        s._dummies["v_stub"] = _zeroed_device_buffer(s.kh * s.d_v * _buffers.DTYPE_ITEMSIZE[s.expect["v"]], s.device_index)
    scale = float(api.scale_softmax if scale_softmax is None or scale_softmax == 0.0 else scale_softmax)

    t: List[Any] = [None] * len(order)

    def put(name, value):
        if name in s.index:
            t[s.index[name]] = value

    q, k, v, o = plan.q, plan.k, plan.v, plan.o
    put("q_strides", (q[2], q[2], q[3]))
    put("o_strides", (o[2], o[2], o[3]))
    if not s.paged:
        put("k_strides", (k[2], k[2], k[3]))
        put("v_strides", (v[2], v[2], v[3]))
    if s.lse_padded:
        put("lse_strides", s.lse_stride)
        put("lse_ext", s.s_q_max)
    else:
        put("lse_strides", (0, 0, 0))
        put("lse_ext", s.lse_head_stride)  # compact head-major: the token capacity, written per call
    put("scale_softmax_log2", scale * math.log2(math.e))
    put("n_thd_units", int(plan.units))
    put("seq_q_lens_addr", 0)
    put("thd_lens_form", s.lens_form)
    put("o_partial_ptr", None)
    put("block_table_ptr", None)
    put("block_table_v_ptr", None)
    put("table_strides", (0, 0))
    put("n_pages", 0)
    put("gate_ptr", None)  # gate-capable hosts declare the slots; the prepared domain excludes the gate
    put("gate_strides", (0, 0, 0))
    unfilled = sorted(set(order) - _FILLED_AT_BUILD - _FILLED_PER_CALL)
    if unfilled:
        raise NotImplementedError(f"{km.__name__}: host slots {unfilled} are not bound by the prepared THD launch")
    s.template = t
    return s


def _capacity(f: BufferFacts, geo: Tuple[int, int, int, int], name: str) -> int:
    """Token capacity of a packed operand under the resolved (ts, hs, es, row_span) geometry."""
    if f.span < 0:
        raise ValueError(f"cudnn.sdpa: " + (f"{name} was passed as a bare address; a ragged operand needs a sized buffer"))
    ts, row = geo[0], geo[3]
    return 0 if f.span < row else (f.span - row) // ts + 1


class ResolvedGeometry(NamedTuple):
    """The runtime geometry one THD call binds: the batch it runs, and per packed role the
    (token, head, elem) strides and the row span the capacities are computed from."""

    b: int
    roles: Mapping[str, Tuple[int, int, int, int]]  # name -> (ts, hs, es, row_span)


def resolve_thd_geometry(spec: ThdLaunchSpec, facts: Dict[str, Optional[BufferFacts]]) -> ResolvedGeometry:
    """Validate this call's effective geometry against the plan's fixed specialization and return
    what the binder writes; raise ValueError outside the supported domain. One implementation for
    admission and binding (no separate boolean query with its own rules).

    Fixed by the artifact: head counts (GQA specialization), head dims, layout kind, page size.
    Runtime per call: the batch (at most the declared one: workspace and unit envelope are sized
    for it), and each packed operand's token / head strides (TMA-expressible: elem stride 1,
    16-byte multiples, covering)."""
    cu_q, cu_kv = bool(spec.lens_form & 1), bool(spec.lens_form & 2)
    q_lens, kv_lens = facts.get("q_lens"), facts.get("kv_lens")
    if q_lens is None or kv_lens is None:
        raise ValueError(f"cudnn.sdpa: " + ("THD execute requires seq_q_lens and seq_kv_lens"))
    # Only the pure layout calculation is reusable. Addresses, observed storage spans,
    # producer devices, workspace and stream are still validated/bound on EVERY call
    # by bind_thd, including its per-call padded-Stats seed. Geometry changes (also
    # execute-time overrides) take the same admission rules below.
    names = ("q", "o") + (() if spec.paged else ("k", "v"))
    key = (q_lens.shape, kv_lens.shape, tuple((facts[name].shape, facts[name].strides) for name in names))
    cached = spec._geometry_cache
    if cached is not None and cached[0] == key:
        return cached[1]
    b = q_lens.numel - (1 if cu_q else 0)
    if b <= 0 or b > spec.b:
        raise ValueError(f"cudnn.sdpa: " + (f"seq_q_lens describes {b} sequences; this plan is prepared for 1..{spec.b}"))
    if kv_lens.numel != b + (1 if cu_kv else 0):
        raise ValueError(f"cudnn.sdpa: " + (f"seq_kv_lens must describe the same {b} sequences as seq_q_lens; got {kv_lens.numel} elements"))
    if spec.has_lse and spec.lse_padded and b != spec.b:
        raise ValueError(f"cudnn.sdpa: " + (f"a per-batch padded Stats buffer is declared for {spec.b} sequences; running {b} is not supported"))
    roles: Dict[str, Tuple[int, int, int, int]] = {}
    for name in names:
        f = facts[name]
        decl = spec.decl[name]
        h, d = decl[0], decl[1]
        st, sh = f.strides, f.shape
        if len(st) == 4:  # the graph's (B, H, S, D) declaration, possibly overridden
            if int(sh[0]) != b or int(sh[1]) != h or int(sh[3]) != d:
                raise ValueError(f"cudnn.sdpa: " + (f"{name}: effective shape {tuple(sh)} must be ({b}, {h}, S, {d}) for this plan"))
            ts, hs, es = int(st[2]), int(st[1]), int(st[3])
        elif len(st) == 3:  # the caller's packed (T, H, D)
            if int(sh[1]) != h or int(sh[2]) != d:
                raise ValueError(f"cudnn.sdpa: " + (f"{name}: a packed THD buffer is (T, {h}, {d}); got {tuple(sh)}"))
            ts, hs, es = int(st[0]), int(st[1]), int(st[2])
        else:
            if True:
                raise ValueError(f"cudnn.sdpa: " + (f"{name}: a THD operand is (T, H, D) or the graph's (B, H, S, D); got rank {len(st)}"))
        if f.numel == 0:
            ts, hs, es = decl[2], decl[3], decl[4]
        width = _buffers.DTYPE_ITEMSIZE[spec.expect[name]]
        if es != 1:
            raise ValueError(f"cudnn.sdpa: " + (f"{name}: the head dim must be contiguous (elem stride 1); got {es}"))
        if hs < d or (hs * width) % _ALIGN_TMA != 0:
            raise ValueError(f"cudnn.sdpa: " + (f"{name}: head stride {hs} must cover {d} elements and be a 16-byte multiple"))
        if ts < (h - 1) * hs + d or (ts * width) % _ALIGN_TMA != 0:
            raise ValueError(f"cudnn.sdpa: " + (f"{name}: token stride {ts} must cover the {h} heads and be a 16-byte multiple"))
        roles[name] = (ts, hs, es, (h - 1) * hs + (d - 1) * es + 1)
    geometry = ResolvedGeometry(b, MappingProxyType(roles))
    # Publish one immutable record, never a separately updated key/value pair: concurrent
    # calls with different geometry may replace the cache but keep their own geometry.
    # One entry bounds retained metadata independently of the number of batch shapes.
    spec._geometry_cache = (key, geometry)
    return geometry


def _stats_layout_is_the_compiled_kind(spec: ThdLaunchSpec, lse: BufferFacts) -> None:
    """The host builds the Stats tensor from the compiled layout kind (token-major (T, H), head-major
    (1, H, ext), or the declared padded strides), never from the effective strides. Compact storage of
    rank <= 2 carries no layout that could contradict the kind (the kind is how that storage is written;
    this is what the tensor path bound too); a described rank-3 / rank-4 geometry must be the kind."""
    if lse.numel == 0:
        return
    st, sh = lse.strides, lse.shape
    qh = spec.qh
    if spec.lse_padded:
        # The declared per-batch strides apply over the caller's STORAGE (the padded contract): any
        # contiguous allocation of the right size is that storage, whatever its own dim order; a
        # strided view is accepted only when it IS the declared (B, H, S) layout.
        if len(sh) in (3, 4) and (int(st[0]), int(st[1]), int(st[2])) == tuple(spec.lse_stride):
            return
        if not lse.contiguous:
            raise ValueError(f"cudnn.sdpa: padded lse_tensor strides {tuple(st)} must be the declared {tuple(spec.lse_stride)} or contiguous storage")
        return
    if len(sh) <= 2 and lse.contiguous:
        return  # flat (T*H) or (T, H) / (H, ext) storage: written in the compiled kind
    if len(sh) == 4:  # the graph's (B, H, S, 1)
        h_st, t_st = int(st[1]), int(st[2])
    elif len(sh) == 3 and spec.lse_head_major:  # (1, H, ext)
        h_st, t_st = int(st[1]), int(st[2])
    elif len(sh) == 3:  # (T, H, 1) token-major
        h_st, t_st = int(st[1]), int(st[0])
    elif len(sh) == 2:  # strided (T, H) / (H, ext)
        h_st, t_st = (int(st[0]), int(st[1])) if spec.lse_head_major else (int(st[1]), int(st[0]))
    else:
        raise ValueError(f"cudnn.sdpa: lse_tensor: unsupported Stats geometry {tuple(sh)} / {tuple(st)}")
    if spec.lse_head_major:
        if t_st != 1:
            raise ValueError(f"cudnn.sdpa: head-major lse_tensor must have the token axis contiguous; got stride {t_st}")
        if spec.lse_head_stride and h_st != spec.lse_head_stride:
            raise ValueError(f"cudnn.sdpa: head-major lse_tensor head stride {h_st} must be the declared {spec.lse_head_stride}")
    elif h_st != 1 or t_st != qh:
        raise ValueError(f"cudnn.sdpa: token-major lse_tensor must be packed (T, H): head stride 1, token stride {qh}; got head {h_st}, token {t_st}")


def _on_plan_device(spec, name: str, f: BufferFacts) -> None:
    # one device rule for every bound role: a KNOWN producer device must be the plan's CUDA device
    if f.device[0] != -1 and f.device != (_DLPACK_CUDA, spec.device_index):
        raise ValueError(f"cudnn.sdpa: " + (f"{name} must be on CUDA device {spec.device_index} (this plan's); got DLPack device {f.device}"))


def _bind_paged_kv(spec, frame: List[Any], ix: Dict[str, int], facts: Dict[str, Optional[BufferFacts]], k: BufferFacts, v: BufferFacts, b: int) -> int:
    """Bind the page pools and tables of a paged launch; returns SKV = max_pages * page_size. Shared by
    the THD and dense binders (one implementation of the paged domain)."""
    # K/V are page pools (n_pages, page_size, KH, D) in the kernel's order: a permutation of the
    # container's (n_pages, KH, page_size, D) strides; SKV = max_pages * page_size
    bt, btv = facts.get("block_table"), facts.get("block_table_v")
    if bt is None or btv is None:
        raise ValueError(f"cudnn.sdpa: " + ("paged KV requires paged_attention_k_table / paged_attention_v_table buffers"))
    if bt.dtype != "int32" or btv.dtype != "int32":
        raise ValueError(f"cudnn.sdpa: " + ("the page tables must be int32"))
    _on_plan_device(spec, "paged_attention_k_table", bt)
    _on_plan_device(spec, "paged_attention_v_table", btv)

    def table(name, f):
        if len(f.shape) == 4:  # the graph's (B, 1, max_pages, 1) declaration
            shape, strides = (int(f.shape[0]), int(f.shape[2])), (int(f.strides[0]), int(f.strides[2]))
        else:
            shape, strides = tuple(int(x) for x in f.shape), tuple(int(x) for x in f.strides)
        if len(shape) != 2:
            raise ValueError(f"cudnn.sdpa: " + (f"{name} must be (B, max_pages); got {f.shape}"))
        return shape, strides

    (tb, max_pages), table_strides = table("paged_attention_k_table", bt)
    (tbv, max_pages_v), table_strides_v = table("paged_attention_v_table", btv)
    if tb < b or tbv < b:
        raise ValueError(f"cudnn.sdpa: " + (f"the page tables describe {tb} / {tbv} sequences; this call runs {b}"))
    if max_pages_v != max_pages or table_strides_v != table_strides:
        raise ValueError(
            f"cudnn.sdpa: "
            + ("paged_attention_k_table and paged_attention_v_table must share (max_pages) and strides: the host walks both with one stride pair")
        )
    # the host addresses rows 0..B-1 and pages 0..max_pages-1 of BOTH tables
    need = (b - 1) * table_strides[0] + (max_pages - 1) * table_strides[1] + 1
    if bt.span >= 0 and bt.span < need:
        raise ValueError(f"cudnn.sdpa: " + (f"paged_attention_k_table spans {bt.span} elements; ({b}, {max_pages}) with strides {table_strides} needs {need}"))
    if btv.span >= 0 and btv.span < need:
        raise ValueError(f"cudnn.sdpa: " + (f"paged_attention_v_table spans {btv.span} elements; ({b}, {max_pages}) with strides {table_strides} needs {need}"))
    # the pools: (n_pages, KH, page_size, D) containers, head dim contiguous, one page count for K and V,
    # and the in-page layout kind the artifact was compiled for (its TMA descriptors order (row, head) by it)
    for name, f, d in (("k", k, spec.d_qk), ("v", v, spec.d_v)):
        if len(f.shape) != 4 or int(f.shape[1]) != spec.kh or int(f.shape[2]) != spec.page_size or int(f.shape[3]) != d:
            raise ValueError(f"cudnn.sdpa: " + (f"{name}: a page pool is (n_pages, {spec.kh}, {spec.page_size}, {d}); got {tuple(f.shape)}"))
        if int(f.strides[3]) != 1:
            raise ValueError(f"cudnn.sdpa: " + (f"{name}: the page pool's head dim must be contiguous"))
        if _pool_is_hnd(f) != spec.paged_hnd:
            raise ValueError(
                f"cudnn.sdpa: " + (f"{name}: this artifact was compiled for {'HND' if spec.paged_hnd else 'NHD'} page pools; got strides {tuple(f.strides)}")
            )
    if int(k.shape[0]) != int(v.shape[0]):
        raise ValueError(f"cudnn.sdpa: " + (f"K and V pools must hold the same number of pages; got {k.shape[0]} and {v.shape[0]}"))
    t_kv = max_pages * spec.page_size
    frame[ix["k_strides"]] = (int(k.strides[0]), int(k.strides[2]), int(k.strides[1]))
    frame[ix["v_strides"]] = (int(v.strides[0]), int(v.strides[2]), int(v.strides[1]))
    frame[ix["block_table_ptr"]], frame[ix["block_table_v_ptr"]] = bt.ptr, btv.ptr
    frame[ix["table_strides"]] = (int(table_strides[0]), int(table_strides[1]))
    frame[ix["n_pages"]] = int(k.shape[0])
    return t_kv


def bind_thd(spec: ThdLaunchSpec, facts: Dict[str, Optional[BufferFacts]], workspace_ptr: int, stream, stream_int: int) -> Optional[List[Any]]:
    """This call's argument frame for ``spec`` from the operands' facts (roles ``q k v o lse sinks
    q_lens kv_lens`` and, paged, ``block_table block_table_v``); None when no Q token is
    addressable. Runs the declared per-call operation (padded-Stats seed) on ``stream_int``."""
    ix = spec.index
    frame = spec.frame()

    def on_plan_device(name: str, f: BufferFacts) -> None:
        _on_plan_device(spec, name, f)

    def operand(name: str) -> BufferFacts:
        f = facts.get(name)
        if f is None:
            raise ValueError(f"cudnn.sdpa: " + (f"{name} is required"))
        if f.dtype != spec.expect[name]:
            raise ValueError(f"cudnn.sdpa: " + (f"{name}: runtime buffer dtype {f.dtype} does not match its declaration ({spec.expect[name]})"))
        on_plan_device(name, f)
        if f.ptr % _ALIGN_TMA != 0:
            raise ValueError(
                f"cudnn.sdpa: "
                + (f"{name}: runtime buffer base address must be 16-byte aligned (TMA global-address rule); got data_ptr() % 16 == {f.ptr % _ALIGN_TMA}")
            )
        return f

    def lens(name: str, n: int) -> int:
        f = facts.get(name)
        if f is None:
            raise ValueError(f"cudnn.sdpa: " + (f"{name} is required"))
        on_plan_device(name, f)
        if f.dtype != "int32":
            raise ValueError(f"cudnn.sdpa: " + (f"{name} must be int32; got {f.dtype}"))
        if f.numel != n:
            raise ValueError(f"cudnn.sdpa: " + (f"{name} must have {n} elements; got {f.numel}"))
        if not f.contiguous:
            raise ValueError(f"cudnn.sdpa: " + (f"{name} must be contiguous (read as a flat ({n},) operand)"))
        return f.ptr

    q, k, v, o = operand("q"), operand("k"), operand("v"), operand("o")
    geo = resolve_thd_geometry(spec, facts)
    frame[ix["q_ptr"]], frame[ix["k_ptr"]], frame[ix["v_ptr"]], frame[ix["o_ptr"]] = q.ptr, k.ptr, v.ptr, o.ptr
    for name, slot in (("q", "q_strides"), ("o", "o_strides")) + (() if spec.paged else (("k", "k_strides"), ("v", "v_strides"))):
        ts, hs, _es, _row = geo.roles[name]
        frame[ix[slot]] = (ts, ts, hs)  # the extent-1 batch dim binds the token stride (never stepped)
    frame[ix["thd_q_lens_ptr"]] = lens("q_lens", geo.b + (1 if spec.lens_form & 1 else 0))
    frame[ix["thd_kv_lens_ptr"]] = lens("kv_lens", geo.b + (1 if spec.lens_form & 2 else 0))

    lse = facts.get("lse")
    lse_cap = None
    if spec.has_lse:
        if lse is None:
            raise ValueError(f"cudnn.sdpa: " + ("lse_tensor is required by this compiled specialization"))
        on_plan_device("lse_tensor", lse)
        if lse.dtype != "float32":
            raise ValueError(f"cudnn.sdpa: " + (f"lse_tensor must be float32; got {lse.dtype}"))
        if lse.ptr % _ALIGN_F32 != 0:
            raise ValueError(f"cudnn.sdpa: " + ("lse_tensor must be 4-byte aligned"))
        if spec.lse_padded:
            expected = spec.b * spec.qh * spec.s_q_max
            if lse.numel != expected:
                raise ValueError(f"cudnn.sdpa: " + (f"padded lse_tensor must have B*H_q*S_q_max = {expected} elements; got {lse.numel}"))
        elif spec.lse_head_major and spec.lse_head_stride:
            if lse.numel < spec.qh * spec.lse_head_stride:
                raise ValueError(
                    f"cudnn.sdpa: " + (f"head-major lse_tensor must hold H_q*head_stride = {spec.qh * spec.lse_head_stride} elements; got {lse.numel}")
                )
        else:
            if lse.span < 0:
                raise ValueError(f"cudnn.sdpa: " + ("lse_tensor: a ragged Stats operand needs a sized buffer, not a bare address"))
            lse_cap = lse.span // spec.qh
        _stats_layout_is_the_compiled_kind(spec, lse)  # after the size checks: a mis-sized buffer reports its size, not its layout
        frame[ix["lse_ptr"]] = lse.ptr
    else:
        if lse is not None:
            raise ValueError(f"cudnn.sdpa: " + ("this specialization was compiled without a Stats output; construct the API without sample_lse"))

    def seed_padded():
        shape = (spec.b, spec.qh, spec.s_q_max)
        if _buffers.is_contiguous(shape, spec.lse_stride):
            _buffers.fill_word_async(lse.ptr, math.prod(shape), spec.neg_inf, stream_int)
        else:
            _buffers.fill_word_strided_async(lse.ptr, shape, spec.lse_stride, 4, spec.neg_inf, stream_int)

    t_q = min(_capacity(q, geo.roles["q"], "q"), _capacity(o, geo.roles["o"], "o"))
    if spec.total_q is not None:
        t_q = min(t_q, spec.total_q)
    if lse_cap is not None:
        t_q = min(t_q, lse_cap)
    if t_q == 0:
        if spec.has_lse and spec.lse_padded:
            seed_padded()
        return None

    if spec.paged:
        t_kv = _bind_paged_kv(spec, frame, ix, facts, k, v, geo.b)
    else:
        t_kv = min(_capacity(k, geo.roles["k"], "k"), _capacity(v, geo.roles["v"], "v"))
        if spec.total_kv is not None:
            t_kv = min(t_kv, spec.total_kv)
        if t_kv == 0:
            # all-KV-zero clamp: one packed row of K aliases Q's storage, V a zero stub; the kernel reads no K/V row
            kh, d_qk, d_v = spec.kh, spec.d_qk, spec.d_v
            t_kv = 1
            frame[ix["k_ptr"]] = q.ptr
            frame[ix["k_strides"]] = (kh * d_qk, kh * d_qk, d_qk)
            frame[ix["v_ptr"]] = spec.dummy("v_stub")
            frame[ix["v_strides"]] = (kh * d_v, kh * d_v, d_v)
    if spec.has_lse and spec.lse_head_major and not spec.lse_head_stride:
        frame[ix["lse_ext"]] = t_q
    frame[ix["problem_size"]] = (geo.b, spec.qh, spec.kh, t_q, t_kv, 0)  # units / workspace are sized for spec.b >= geo.b

    sinks = facts.get("sinks")
    if spec.has_sink:
        if sinks is None:
            raise ValueError(f"cudnn.sdpa: " + ("sinks is required by this compiled specialization"))
        on_plan_device("sinks", sinks)
        if sinks.dtype != "float32" or sinks.numel != spec.qh or not sinks.contiguous:
            raise ValueError(f"cudnn.sdpa: " + (f"sinks must be a contiguous ({spec.qh},) float32 tensor"))
        frame[ix["sinks_ptr"]] = sinks.ptr
    else:
        if sinks is not None:
            raise ValueError(f"cudnn.sdpa: " + ("this specialization was compiled without a sink; construct the API with has_sink"))
        frame[ix["sinks_ptr"]] = spec.dummy("sinks")

    if workspace_ptr % _ALIGN_TMA != 0:
        raise ValueError(f"cudnn.sdpa: " + (f"the workspace must be 16-byte aligned; got 0x{workspace_ptr:x}"))
    frame[ix["meta_ptr"]] = workspace_ptr
    frame[ix["o_desc_ptr"]] = workspace_ptr + spec.off_o_desc
    frame[ix["stream"]] = stream
    if spec.has_lse and spec.lse_padded:
        seed_padded()  # declared per-call operation: rows past each length read -inf
    return frame


class PreparedThdLaunch:
    """The graph plan's THD f16 launch: the spec plus this graph's operand uids."""

    def __init__(self, spec: ThdLaunchSpec, binding):
        self.spec = spec
        uids = {
            "q": binding.q.get_uid(),
            "k": binding.k.get_uid(),
            "v": binding.v.get_uid(),
            "o": binding.o.get_uid(),
            "q_lens": (binding.cu_seq_len_q if spec.lens_form & 1 else binding.seq_len_q).get_uid(),
            "kv_lens": (binding.cu_seq_len_kv if spec.lens_form & 2 else binding.seq_len_kv).get_uid(),
        }
        if binding.stats is not None:
            uids["lse"] = binding.stats.get_uid()
        if spec.has_sink:
            uids["sinks"] = binding.sink_token.get_uid()
        if spec.paged:
            uids["block_table"] = binding.paged_k_table.get_uid()
            uids["block_table_v"] = binding.paged_v_table.get_uid()
        self._roles = list(uids)
        self._uids = [uids[r] for r in self._roles]
        self._indices: Optional[List[int]] = None

    def execute(self, pack, workspace_ptr: int, stream, stream_int: int) -> None:
        indices = self._indices
        if indices is None:
            try:
                indices = self._indices = [pack.index_of(u) for u in self._uids]
            except KeyError as exc:
                raise ValueError(f"cudnn.sdpa: tensor uid {exc} is bound by the plan but is not an operand of this graph") from exc
        facts = dict(zip(self._roles, facts_of_roles(pack, indices)))
        frame = bind_thd(self.spec, facts, workspace_ptr, stream, stream_int)
        if frame is not None:
            self.spec.fn(*frame)


# ---------------------------------------------------------------------------------------------------
# Dense (padded) launch: the same explicit host, every extent and stride read from the operands.
# ---------------------------------------------------------------------------------------------------

# Constants written at build, and slots bind_dense writes per call.
_FILLED_AT_BUILD_DENSE = frozenset(
    "lse_strides lse_ext scale_softmax_log2 n_thd_units seq_q_lens_addr thd_q_lens_ptr thd_kv_lens_ptr thd_lens_form o_partial_ptr "
    "block_table_ptr block_table_v_ptr table_strides n_pages gate_ptr gate_strides meta_ptr o_desc_ptr".split()
)
_FILLED_PER_CALL_DENSE = frozenset(
    "q_ptr k_ptr v_ptr o_ptr q_strides k_strides v_strides o_strides lse_ptr lse_strides sinks_ptr meta_ptr problem_size seq_q_lens_addr "
    "o_partial_ptr block_table_ptr block_table_v_ptr table_strides n_pages gate_ptr gate_strides stream".split()
)


class SplitCombineSpec(NamedTuple):
    """Immutable combine artifact and workspace geometry; addresses are bound per call."""

    fn: Any
    owner: Any
    o: BufferFacts
    lse: BufferFacts
    lse_offset: int
    output_dtype: str
    has_stats: bool


class DenseLaunchSpec:
    """Plan-time facts of one dense f16 launch (built by :func:`build_dense_spec`); read-only after build."""

    __slots__ = (
        "fn",
        "owner",
        "order",
        "index",
        "template",
        "b",
        "qh",
        "kh",
        "s_q_max",
        "s_k_max",
        "d_qk",
        "d_v",
        "paged",
        "page_size",
        "paged_hnd",
        "expect",
        "elem_bytes",
        "has_lse",
        "has_sink",
        "seq_kv_present",
        "seq_q_present",
        "gate_expect",
        "split",
        "fp32_partial",
        "combine",
        "tile_n",
        "causal",
        "causal_bottom_right",
        "window_right",
        "shape_fixed",
        "lpt_grid_fixed",
        "device_index",
        "_dummies",
    )

    def frame(self) -> List[Any]:
        return list(self.template)

    def dummy(self, key: str) -> int:
        return self._dummies[key].data_ptr()

    def kv_tail_admitted(self, s_q: int, s_kv: int) -> bool:
        """The compiled artifact's KV-tail contract (the adapter's check_support rule, applied to the
        RUNTIME extents): a partial last KV tile is zero-filled by TMA but only MASKED on the padded /
        causal paths, so S_kv must be a tile multiple unless per-batch KV lengths carry the real lengths
        or the causal diagonal provably covers the tail."""
        if self.paged or s_kv % self.tile_n == 0 or self.seq_kv_present:
            return True
        if not self.causal:
            return False
        return (self.causal_bottom_right and self.window_right == 0) or (not self.causal_bottom_right and s_q + self.window_right <= s_kv)


def build_dense_spec(api, *, scale_softmax: Optional[float]) -> DenseLaunchSpec:
    """The adapter's plan-time facts for a dense (padded) launch: the positional entry, the argument
    template with every constant filled, the fixed specialization the binder checks each call against,
    and the read-only dummies (zeroed at build)."""
    km = api._k_mod
    raw, compiled, order = _positional_order(api)
    s = DenseLaunchSpec()
    s.fn, s.owner, s.order = raw, compiled, order
    s.index = {n: i for i, n in enumerate(order)}
    s.b, s.qh, s.kh, s.d_qk, s.d_v = int(api.batch_size), int(api.h_q), int(api.h_kv), int(api.head_dim_qk), int(api.head_dim_v)
    s.s_q_max, s.s_k_max = int(api.s_q_max), int(api.s_k_max)
    if getattr(km.CFG, "PACK_GQA", False) and s.qh != s.kh * km.CFG.QH_PER_KH:
        raise ValueError("cudnn.sdpa: runtime head counts must match the compiled PackGQA ratio")
    if hasattr(km, "N_Q") and s.s_q_max * km.HEADS_PER_TILE > km.N_Q:
        raise ValueError(f"cudnn.sdpa: decode query rows exceed the compiled {km.N_Q}-row tile")
    s.paged, s.page_size = bool(api.paged), int(api.paged_page_size or 0)
    s.paged_hnd = _compiled_paged_hnd(api) if s.paged else False
    s.split = int(api.split_kv)
    s.fp32_partial = bool(api._fp32_partial_split()) if s.split > 1 else False
    s.expect = {n: str(getattr(api, f"{n}_desc").dtype).split(".")[-1] for n in ("q", "k", "v", "o")}
    if s.split > 1:  # the main kernel writes the split-major partial slabs, in the partial dtype
        s.expect["o"] = str(api._partial_torch_dtype()).split(".")[-1]
    s.elem_bytes = {n: _buffers.DTYPE_ITEMSIZE[t] for n, t in s.expect.items()}
    s.has_lse = (api.lse_desc is not None) or s.split > 1
    s.has_sink = bool(api.has_sink)
    s.seq_kv_present, s.seq_q_present = bool(api.seq_kv_lens_present), bool(api.seq_q_lens_present)
    s.gate_expect = str(api.gate_desc.dtype).split(".")[-1] if getattr(api, "gate_desc", None) is not None else None
    s.tile_n = int(getattr(km.CFG, "TILE_N", 128))
    # the COMPILED mask kind: the d192 lowering may have rewritten a square bottom-right mask as top-left
    s.causal = bool(api.is_causal)
    s.causal_bottom_right = bool(getattr(km.CFG, "BOTTOM_RIGHT", getattr(api, "causal_bottom_right", False)))
    s.window_right = int(api.window_size_right or 0)
    # Lowering canonicalizations that read the DECLARED (S_q, S_kv) pin the runtime extents to them: the square
    # bottom-right -> top-left rewrite (equal only for S_q == S_kv) and the 8K LPT-L2 head grouping of the d192 flavor.
    requested_br = bool(getattr(api, "causal_bottom_right", False))
    s.shape_fixed = (s.causal and requested_br and not s.causal_bottom_right) or (  # compiled top-left for a requested bottom-right: square only
        tuple(getattr(api, "flavor", ())) == (192, 128) and s.s_q_max == s.s_k_max == 8192  # the 8K LPT-L2 head grouping
    )
    # A grouped-LPT schedule decodes tile coordinates from a COMPILED Q-tile count and head group (derived from the
    # declared S_q and batch x heads): such an artifact serves exactly the declared (batch, S_q). The f16 kernels
    # compile neither (both fields are the FP8 flavors'), so this pins nothing today and guards their joining.
    params = getattr(km, "PARAMS", None)
    s.lpt_grid_fixed = int(getattr(params, "lpt_head_group", 1)) > 1 or int(getattr(params, "lpt_q_tiles", 0)) > 0
    s.device_index = int(api.q_desc.device.index or 0)
    s.combine = None
    if s.split > 1:
        from cudnn.sdpa.fwd.api_dsl import ws_align
        from cudnn.sdpa.fwd.kernels.sm100 import split_combine

        owner = split_combine.compile_ptr(
            dtype_o=api._combine_dtype_tag(), dtype_partial=api._partial_dtype_tag(), has_lse=api.lse_desc is not None, stats_log2=api.stats_log2
        )
        fn = positional_entry(owner)
        if fn is None:
            raise NotImplementedError("the split combine artifact exposes no positional tvm-ffi entry")
        rows, sq, h, d = s.split * s.b, s.s_q_max, s.qh, s.d_v
        o_size, lse_size = rows * sq * h * d, rows * h * sq
        device = (_DLPACK_CUDA, s.device_index)
        s.combine = SplitCombineSpec(
            fn,
            owner,
            BufferFacts(0, s.expect["o"], device, o_size, (rows, h, sq, d), (sq * h * d, d, h * d, 1)),
            BufferFacts(0, "float32", device, lse_size, (rows, h, sq), (h * sq, sq, 1)),
            ws_align(o_size * s.elem_bytes["o"]),
            str(api.o_desc.dtype).split(".")[-1],
            api.lse_desc is not None,
        )
    s._dummies = {}
    if not s.has_sink:
        s._dummies["sinks"] = _zeroed_device_buffer(s.qh * 4, s.device_index)
    if not s.seq_kv_present:
        s._dummies["meta"] = _zeroed_device_buffer(max(s.b, 1) * 4, s.device_index)
    s._dummies["o_desc"] = _zeroed_device_buffer(128, s.device_index)
    scale = float(api.scale_softmax if scale_softmax is None or scale_softmax == 0.0 else scale_softmax)

    t: List[Any] = [None] * len(order)

    def put(name, value):
        if name in s.index:
            t[s.index[name]] = value

    put("lse_strides", (0, 0, 0))
    put("lse_ext", 0)
    put("scale_softmax_log2", scale * math.log2(math.e))
    put("n_thd_units", 0)
    put("seq_q_lens_addr", 0)
    put("thd_q_lens_ptr", None)
    put("thd_kv_lens_ptr", None)
    put("thd_lens_form", None)
    put("o_partial_ptr", None)
    put("block_table_ptr", None)
    put("block_table_v_ptr", None)
    put("table_strides", (0, 0))
    put("n_pages", 0)
    put("gate_ptr", None)
    put("gate_strides", (0, 0, 0))
    put("meta_ptr", None if s.seq_kv_present else s.dummy("meta"))
    put("o_desc_ptr", s.dummy("o_desc"))
    unfilled = sorted(set(order) - _FILLED_AT_BUILD_DENSE - _FILLED_PER_CALL_DENSE)
    if unfilled:
        raise NotImplementedError(f"{km.__name__}: host slots {unfilled} are not bound by the prepared dense launch")
    s.template = t
    return s


class _DenseRole(NamedTuple):
    ptr: int
    strides: Tuple[int, int, int]  # (batch, seq, head) element strides
    b: int
    s: int


def _dense_role(
    spec: DenseLaunchSpec,
    facts: Dict[str, Optional[BufferFacts]],
    name: str,
    heads: int,
    d: int,
    s_max: int,
    expect: str,
    b_mult: int = 1,
    *,
    tma: bool = True,
) -> _DenseRole:
    """One BHSD operand of a dense launch: dtype / device / alignment, the fixed head count and head
    dim, batch and sequence within the plan's envelope, and a layout TMA binds zero-copy — head dim
    contiguous, seq and head strides 16-byte multiples, token-major and covering (the shared rule of
    ``config_sm100.bshd_zero_copy_stride``); a smaller buffer than the geometry claims is rejected."""
    f = facts.get(name)
    if f is None:
        raise ValueError(f"cudnn.sdpa: {name} is required")
    if f.dtype != expect:
        raise ValueError(f"cudnn.sdpa: {name}: runtime buffer dtype {f.dtype} does not match its declaration ({expect})")
    _on_plan_device(spec, name, f)
    align = _ALIGN_TMA if tma else _buffers.DTYPE_ITEMSIZE[expect]
    if f.ptr % align != 0:
        raise ValueError(f"cudnn.sdpa: {name}: runtime buffer base address must be {align}-byte aligned; got data_ptr() % {align} == {f.ptr % align}")
    if len(f.shape) != 4:
        raise ValueError(f"cudnn.sdpa: {name}: a dense operand is (B, H, S, D); got {tuple(f.shape)}")
    b, h, seq, dd = (int(x) for x in f.shape)
    if h != heads or dd != d:
        raise ValueError(f"cudnn.sdpa: {name}: this plan was built for {heads} heads of dim {d}; got {tuple(f.shape)}")
    if b < 1 or b > spec.b * b_mult:
        raise ValueError(f"cudnn.sdpa: {name}: batch {b} is outside this plan's envelope (1..{spec.b * b_mult})")
    if seq < 1 or seq > s_max:
        raise ValueError(f"cudnn.sdpa: {name}: sequence length {seq} is outside this plan's envelope (1..{s_max})")
    # ONE layout predicate for admission and execution (the lowering's attach decision uses it too):
    # singleton axes canonicalized, then BSHD-compact or the zero-copy rule
    from cudnn.sdpa.fwd.config_sm100 import dense_bind_strides

    if tma:
        bound = dense_bind_strides((b, h, seq, dd), tuple(int(x) for x in f.strides), _buffers.DTYPE_ITEMSIZE[expect])
    else:
        from cudnn.sdpa.graph_analyzer import dense_layout_ok

        if not dense_layout_ok(f.shape, f.strides):
            raise ValueError(f"cudnn.sdpa: {name}: split output needs contiguous D and non-overlapping strides; got {f.shape} / {f.strides}")
        st = tuple(int(v) if int(n) > 1 else 0 for n, v in zip(f.shape, f.strides))
        bound = st[0], st[2], st[1]
    if bound is None:
        raise ValueError(
            f"cudnn.sdpa: {name}: shape {tuple(f.shape)} strides {tuple(f.strides)} is not a layout the kernel binds zero-copy: the head dim must be "
            f"contiguous, seq and head strides 16-byte multiples, and the layout token-major and covering (config_sm100.dense_bind_strides)"
        )
    bs, ss, hs = bound
    need = (b - 1) * bs + (h - 1) * hs + (seq - 1) * ss + d
    if f.span >= 0 and f.span < need:
        raise ValueError(f"cudnn.sdpa: {name} spans {f.span} elements; its geometry {tuple(f.shape)} / {tuple(f.strides)} needs {need}")
    return _DenseRole(f.ptr, (bs, ss, hs), b, seq)


def _dense_lse(spec: DenseLaunchSpec, lse: Optional[BufferFacts], b: int, s_q: int, *, required: bool):
    """Validate main or final Stats with the same device, span and non-aliasing contract."""
    if required:
        if lse is None:
            raise ValueError("cudnn.sdpa: lse_tensor is required by this compiled specialization")
        _on_plan_device(spec, "lse_tensor", lse)
        if lse.dtype != "float32":
            raise ValueError(f"cudnn.sdpa: lse_tensor must be float32; got {lse.dtype}")
        if lse.ptr % _ALIGN_F32 != 0:
            raise ValueError("cudnn.sdpa: lse_tensor must be 4-byte aligned")
        sh, st = tuple(int(x) for x in lse.shape), tuple(int(x) for x in lse.strides)
        if len(sh) == 4 and sh[3] == 1:
            sh, st = sh[:3], st[:3]
        if len(sh) != 3 or sh[0] < b or sh[1] != spec.qh or sh[2] < s_q:
            raise ValueError(f"cudnn.sdpa: lse_tensor must be ({b}, {spec.qh}, {s_q}[, 1]) or larger; got {tuple(lse.shape)}")
        need = (b - 1) * st[0] + (spec.qh - 1) * st[1] + (s_q - 1) * st[2] + 1
        if lse.span >= 0 and lse.span < need:
            raise ValueError(f"cudnn.sdpa: lse_tensor spans {lse.span} elements; its geometry needs {need}")
        if not _covering((b, spec.qh, s_q), st):
            raise ValueError(f"cudnn.sdpa: lse_tensor strides {st} alias distinct (batch, head, row) entries onto one address (a write race)")
        return lse.ptr, (st[0], st[1], st[2])
    elif lse is not None:
        raise ValueError("cudnn.sdpa: this specialization was compiled without a Stats output; construct the API without sample_lse")

    return None, (0, 0, 0)


def bind_dense(spec: DenseLaunchSpec, facts: Dict[str, Optional[BufferFacts]], stream, stream_int: int) -> List[Any]:
    """This call's argument frame for a dense launch from the operands' facts (roles ``q k v o`` and,
    as compiled, ``lse sinks seq_kv seq_q gate`` and the paged ``block_table block_table_v``). Under a
    split the ``o`` / ``lse`` roles are the split-major partial slabs (batch extent split * B)."""
    ix = spec.index
    frame = spec.frame()
    q = _dense_role(spec, facts, "q", spec.qh, spec.d_qk, spec.s_q_max, spec.expect["q"])
    o = _dense_role(spec, facts, "o", spec.qh, spec.d_v, spec.s_q_max, spec.expect["o"], b_mult=spec.split)
    b, s_q = q.b, q.s
    if o.b != b * spec.split or o.s != s_q:
        raise ValueError(
            f"cudnn.sdpa: o is ({o.b}, {spec.qh}, {o.s}, {spec.d_v}) but q runs batch {b} x seq {s_q}" + (f" ({spec.split} splits)" if spec.split > 1 else "")
        )
    frame[ix["q_ptr"]], frame[ix["q_strides"]] = q.ptr, q.strides
    frame[ix["o_ptr"]], frame[ix["o_strides"]] = o.ptr, o.strides
    if spec.paged:
        k, v = facts.get("k"), facts.get("v")
        if k is None or v is None:
            raise ValueError("cudnn.sdpa: k and v are required")
        for name, f, expect in (("k", k, spec.expect["k"]), ("v", v, spec.expect["v"])):
            if f.dtype != expect:
                raise ValueError(f"cudnn.sdpa: {name}: runtime buffer dtype {f.dtype} does not match its declaration ({expect})")
            _on_plan_device(spec, name, f)
            if f.ptr % _ALIGN_TMA != 0:
                raise ValueError(f"cudnn.sdpa: {name}: runtime buffer base address must be 16-byte aligned")
        frame[ix["k_ptr"]], frame[ix["v_ptr"]] = k.ptr, v.ptr
        s_kv = _bind_paged_kv(spec, frame, ix, facts, k, v, b)
    else:
        k = _dense_role(spec, facts, "k", spec.kh, spec.d_qk, spec.s_k_max, spec.expect["k"])
        v = _dense_role(spec, facts, "v", spec.kh, spec.d_v, spec.s_k_max, spec.expect["v"])
        if k.b != b or v.b != b or k.s != v.s:
            raise ValueError(f"cudnn.sdpa: k / v run batch {k.b} / {v.b} x seq {k.s} / {v.s}; q runs batch {b}")
        s_kv = k.s
        frame[ix["k_ptr"]], frame[ix["k_strides"]] = k.ptr, k.strides
        frame[ix["v_ptr"]], frame[ix["v_strides"]] = v.ptr, v.strides
    if spec.shape_fixed and (s_q, s_kv) != (spec.s_q_max, spec.s_k_max):
        raise ValueError(
            f"cudnn.sdpa: this artifact was lowered for exactly S_q={spec.s_q_max}, S_kv={spec.s_k_max} (a square-mask / schedule canonicalization "
            f"read the declared extents); it does not serve ({s_q}, {s_kv})"
        )
    if spec.lpt_grid_fixed and (b, s_q) != (spec.b, spec.s_q_max):
        raise ValueError(
            f"cudnn.sdpa: this artifact's grouped LPT schedule was compiled for exactly batch={spec.b}, S_q={spec.s_q_max}; "
            f"it does not serve batch={b}, S_q={s_q}"
        )
    if not spec.kv_tail_admitted(s_q, s_kv):
        raise ValueError(
            f"cudnn.sdpa: S_kv ({s_kv}) must be a multiple of {spec.tile_n} for this artifact unless per-batch KV lengths are present or the "
            f"causal mask covers the KV tail (the compiled specialization does not mask a partial last tile)"
        )
    frame[ix["problem_size"]] = (b, spec.qh, spec.kh, s_q, s_kv, 0)
    if spec.fp32_partial:
        frame[ix["o_partial_ptr"]] = o.ptr

    frame[ix["lse_ptr"]], frame[ix["lse_strides"]] = _dense_lse(spec, facts.get("lse"), b * spec.split, s_q, required=spec.has_lse)

    sinks = facts.get("sinks")
    if spec.has_sink:
        if sinks is None:
            raise ValueError("cudnn.sdpa: sinks is required by this compiled specialization")
        _on_plan_device(spec, "sinks", sinks)
        if sinks.dtype != "float32" or sinks.numel != spec.qh or not sinks.contiguous:
            raise ValueError(f"cudnn.sdpa: sinks must be a contiguous ({spec.qh},) float32 tensor")
        if sinks.ptr % _ALIGN_F32:
            raise ValueError("cudnn.sdpa: sinks must be 4-byte aligned")
        if sinks.span >= 0 and sinks.span < spec.qh:
            raise ValueError(f"cudnn.sdpa: sinks spans {sinks.span} elements; this launch reads {spec.qh}")
        frame[ix["sinks_ptr"]] = sinks.ptr
    else:
        if sinks is not None:
            raise ValueError("cudnn.sdpa: this specialization was compiled without a sink; construct the API with has_sink")
        frame[ix["sinks_ptr"]] = spec.dummy("sinks")

    def lens(name: str) -> int:
        f = facts.get(name)
        if f is None:
            raise ValueError(f"cudnn.sdpa: {name} is required by this compiled specialization")
        _on_plan_device(spec, name, f)
        if f.dtype != "int32" or not f.contiguous or f.numel < b:
            raise ValueError(f"cudnn.sdpa: {name} must be a contiguous int32 tensor of at least {b} elements; got {f.dtype} x {f.numel}")
        if f.ptr % _ALIGN_F32:
            raise ValueError(f"cudnn.sdpa: {name} must be 4-byte aligned")
        if f.span >= 0 and f.span < b:
            raise ValueError(f"cudnn.sdpa: {name} spans {f.span} elements; this launch reads {b}")
        return f.ptr

    if spec.seq_kv_present:
        frame[ix["meta_ptr"]] = lens("seq_kv_lens")
    if spec.seq_q_present:
        frame[ix["seq_q_lens_addr"]] = lens("seq_q_lens")

    if spec.gate_expect is not None:
        g = _dense_role(spec, facts, "gate", spec.qh, spec.d_v, spec.s_q_max, spec.gate_expect)
        if g.b != b or g.s != s_q:
            raise ValueError(f"cudnn.sdpa: gate is ({g.b}, {spec.qh}, {g.s}, {spec.d_v}) but q runs batch {b} x seq {s_q}")
        frame[ix["gate_ptr"]], frame[ix["gate_strides"]] = g.ptr, g.strides
    elif facts.get("gate") is not None:
        raise ValueError("cudnn.sdpa: this specialization was compiled without an epilogue gate")
    frame[ix["stream"]] = stream
    return frame


def bind_dense_split(spec: DenseLaunchSpec, facts: Dict[str, Optional[BufferFacts]], workspace_ptr: int, stream, stream_int: int):
    """Bind partial workspace and final strided outputs before either split kernel launches."""
    combine = spec.combine
    if not workspace_ptr or workspace_ptr % _ALIGN_TMA:
        raise ValueError("cudnn.sdpa: split workspace must be non-null and 16-byte aligned")
    q = facts.get("q")
    if q is None or len(q.shape) != 4 or (q.shape[0], q.shape[2]) != (spec.b, spec.s_q_max):
        raise ValueError(f"cudnn.sdpa: a split launch runs the declared (B, S_q) = ({spec.b}, {spec.s_q_max})")
    o = _dense_role(spec, facts, "o", spec.qh, spec.d_v, spec.s_q_max, combine.output_dtype, tma=False)
    if (o.b, o.s) != (spec.b, spec.s_q_max):
        raise ValueError("cudnn.sdpa: split output must match the declared (B, S_q)")
    lse_ptr, lse_strides = _dense_lse(spec, facts.get("lse"), spec.b, spec.s_q_max, required=combine.has_stats)
    lse_partial_ptr = workspace_ptr + combine.lse_offset
    partials = dict(facts, o=combine.o._replace(ptr=workspace_ptr), lse=combine.lse._replace(ptr=lse_partial_ptr))
    frame = bind_dense(spec, partials, stream, stream_int)
    combine_args = (
        workspace_ptr,
        lse_partial_ptr,
        o.ptr,
        lse_ptr,
        (spec.b, spec.qh, spec.s_q_max, spec.d_v),
        spec.split,
        (*o.strides, 1),
        lse_strides,
        stream,
    )
    return frame, combine_args


class PreparedDenseLaunch:
    """The graph plan's dense f16 launch: the spec plus this graph's operand uids."""

    def __init__(self, spec: DenseLaunchSpec, binding, *, seq_kv_src=None, seq_q_src=None, gate_src=None):
        self.spec = spec
        uids = {"q": binding.q.get_uid(), "k": binding.k.get_uid(), "v": binding.v.get_uid(), "o": binding.o.get_uid()}
        if binding.stats is not None:
            uids["lse"] = binding.stats.get_uid()
        if spec.has_sink:
            uids["sinks"] = binding.sink_token.get_uid()
        if spec.seq_kv_present:
            uids["seq_kv_lens"] = seq_kv_src.get_uid()
        if spec.seq_q_present:
            uids["seq_q_lens"] = seq_q_src.get_uid()
        if spec.paged:
            uids["block_table"] = binding.paged_k_table.get_uid()
            uids["block_table_v"] = binding.paged_v_table.get_uid()
        if spec.gate_expect is not None:
            uids["gate"] = gate_src.get_uid()
        self._roles = list(uids)
        self._uids = [uids[r] for r in self._roles]
        self._indices: Optional[List[int]] = None

    def execute(self, pack, workspace_ptr: int, stream, stream_int: int) -> None:
        indices = self._indices
        if indices is None:
            try:
                indices = self._indices = [pack.index_of(u) for u in self._uids]
            except KeyError as exc:
                raise ValueError(f"cudnn.sdpa: tensor uid {exc} is bound by the plan but is not an operand of this graph") from exc
        facts = dict(zip(self._roles, facts_of_roles(pack, indices)))
        if self.spec.combine is not None:
            frame, combine_args = bind_dense_split(self.spec, facts, workspace_ptr, stream, stream_int)
            self.spec.fn(*frame)
            self.spec.combine.fn(*combine_args)
        else:
            self.spec.fn(*bind_dense(self.spec, facts, stream, stream_int))

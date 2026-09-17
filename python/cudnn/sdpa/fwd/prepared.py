# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""The SM100 THD (packed / ragged) f16 forward launch, prepared once and bound per call.

Three owners, one implementation each:

* **Observation** — :class:`BufferFacts`: what the caller's buffer is (address, dtype, device,
  the element span the producer guarantees, shape / strides). Read from the graph's normalized
  ``VariantPack`` (:func:`facts_of_pack`) or from a torch tensor handed to the standalone
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
"""

from __future__ import annotations

import inspect
import math
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

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
    """Facts of several variant-pack operands in one native crossing (see :func:`facts_of_pack`)."""
    out = []
    for ptr, code, bits, dev_type, dev_id, nbytes, shape, stride in pack.native.facts(list(indices)):
        width = max(1, (int(bits) + 7) // 8)
        out.append(
            BufferFacts(int(ptr), _DTYPE_BY_CODE.get((code, bits), ""), (dev_type, dev_id), -1 if nbytes < 0 else nbytes // width, tuple(shape), tuple(stride))
        )
    return out


def facts_of_pack(pack, index: int) -> BufferFacts:
    """Facts of variant-pack operand ``index``: the producer's observed span / device, the
    effective (graph-described, overridden) geometry; no operand object is built."""
    native = pack.native
    code_bits = tuple(native.dtype(index))
    nbytes = native.observed_bytes(index)
    width = max(1, (int(code_bits[1]) + 7) // 8) if len(code_bits) == 2 else 1
    return BufferFacts(
        native.pointer(index),
        _DTYPE_BY_CODE.get(code_bits, ""),
        tuple(native.observed_device(index)),
        -1 if nbytes < 0 else nbytes // width,  # producer bytes -> elements of the EFFECTIVE (declared) dtype
        tuple(native.shape(index)),
        tuple(native.stride(index)),
    )


class ThdLaunchSpec:
    """Plan-time facts of one THD f16 launch (built by :func:`build_thd_spec`); read-only after
    build except for the lazily allocated read-only dummies it owns."""

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


def build_thd_spec(api, *, scale_softmax: Optional[float]) -> ThdLaunchSpec:
    """The adapter's plan-time facts as a :class:`ThdLaunchSpec`; NotImplementedError when the
    artifact has no positional entry or the host signature is not the expected shape."""
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
    plan = api._thd_plan()
    s = ThdLaunchSpec()
    s.fn, s.owner, s.order = raw, compiled, order
    s.index = {n: i for i, n in enumerate(order)}
    s.b, s.qh, s.kh, s.d_qk, s.d_v = api.batch_size, api.h_q, api.h_kv, api.head_dim_qk, api.head_dim_v
    s.paged, s.page_size = bool(api.paged), int(api.paged_page_size or 0)
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
    roles: Dict[str, Tuple[int, int, int, int]]  # name -> (ts, hs, es, row_span)


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
    b = q_lens.numel - (1 if cu_q else 0)
    if b <= 0 or b > spec.b:
        raise ValueError(f"cudnn.sdpa: " + (f"seq_q_lens describes {b} sequences; this plan is prepared for 1..{spec.b}"))
    if kv_lens.numel != b + (1 if cu_kv else 0):
        raise ValueError(f"cudnn.sdpa: " + (f"seq_kv_lens must describe the same {b} sequences as seq_q_lens; got {kv_lens.numel} elements"))
    if spec.has_lse and spec.lse_padded and b != spec.b:
        raise ValueError(f"cudnn.sdpa: " + (f"a per-batch padded Stats buffer is declared for {spec.b} sequences; running {b} is not supported"))
    roles: Dict[str, Tuple[int, int, int, int]] = {}
    for name in ("q", "o") + (() if spec.paged else ("k", "v")):
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
    return ResolvedGeometry(b, roles)


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


def bind_thd(spec: ThdLaunchSpec, facts: Dict[str, Optional[BufferFacts]], workspace_ptr: int, stream, stream_int: int) -> Optional[List[Any]]:
    """This call's argument frame for ``spec`` from the operands' facts (roles ``q k v o lse sinks
    q_lens kv_lens`` and, paged, ``block_table block_table_v``); None when no Q token is
    addressable. Runs the declared per-call operation (padded-Stats seed) on ``stream_int``."""
    ix = spec.index
    frame = spec.frame()

    def on_plan_device(name: str, f: BufferFacts) -> None:
        # one device rule for every bound role: a KNOWN producer device must be the plan's CUDA device
        if f.device[0] != -1 and f.device != (_DLPACK_CUDA, spec.device_index):
            raise ValueError(f"cudnn.sdpa: " + (f"{name} must be on CUDA device {spec.device_index} (this plan's); got DLPack device {f.device}"))

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
        # K/V are page pools (n_pages, page_size, KH, D) in the kernel's order: a permutation of the
        # container's (n_pages, KH, page_size, D) strides; SKV = max_pages * page_size
        bt, btv = facts.get("block_table"), facts.get("block_table_v")
        if bt is None or btv is None:
            raise ValueError(f"cudnn.sdpa: " + ("paged KV requires paged_attention_k_table / paged_attention_v_table buffers"))
        if bt.dtype != "int32" or btv.dtype != "int32":
            raise ValueError(f"cudnn.sdpa: " + ("the page tables must be int32"))
        on_plan_device("paged_attention_k_table", bt)
        on_plan_device("paged_attention_v_table", btv)

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
        if tb < geo.b or tbv < geo.b:
            raise ValueError(f"cudnn.sdpa: " + (f"the page tables describe {tb} / {tbv} sequences; this call runs {geo.b}"))
        if max_pages_v != max_pages or table_strides_v != table_strides:
            raise ValueError(
                f"cudnn.sdpa: "
                + ("paged_attention_k_table and paged_attention_v_table must share (max_pages) and strides: the host walks both with one stride pair")
            )
        # the host addresses rows 0..B-1 and pages 0..max_pages-1 of BOTH tables
        need = (geo.b - 1) * table_strides[0] + (max_pages - 1) * table_strides[1] + 1
        if bt.span >= 0 and bt.span < need:
            raise ValueError(
                f"cudnn.sdpa: " + (f"paged_attention_k_table spans {bt.span} elements; ({geo.b}, {max_pages}) with strides {table_strides} needs {need}")
            )
        if btv.span >= 0 and btv.span < need:
            raise ValueError(
                f"cudnn.sdpa: " + (f"paged_attention_v_table spans {btv.span} elements; ({geo.b}, {max_pages}) with strides {table_strides} needs {need}")
            )
        # the pools: (n_pages, KH, page_size, D) containers, head dim contiguous, one page count for K and V
        for name, f, d in (("k", k, spec.d_qk), ("v", v, spec.d_v)):
            if len(f.shape) != 4 or int(f.shape[1]) != spec.kh or int(f.shape[2]) != spec.page_size or int(f.shape[3]) != d:
                raise ValueError(f"cudnn.sdpa: " + (f"{name}: a page pool is (n_pages, {spec.kh}, {spec.page_size}, {d}); got {tuple(f.shape)}"))
            if int(f.strides[3]) != 1:
                raise ValueError(f"cudnn.sdpa: " + (f"{name}: the page pool's head dim must be contiguous"))
        if int(k.shape[0]) != int(v.shape[0]):
            raise ValueError(f"cudnn.sdpa: " + (f"K and V pools must hold the same number of pages; got {k.shape[0]} and {v.shape[0]}"))
        t_kv = max_pages * spec.page_size
        frame[ix["k_strides"]] = (int(k.strides[0]), int(k.strides[2]), int(k.strides[1]))
        frame[ix["v_strides"]] = (int(v.strides[0]), int(v.strides[2]), int(v.strides[1]))
        frame[ix["block_table_ptr"]], frame[ix["block_table_v_ptr"]] = bt.ptr, btv.ptr
        frame[ix["table_strides"]] = (int(table_strides[0]), int(table_strides[1]))
        frame[ix["n_pages"]] = int(k.shape[0])
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
        if spec.has_lse:
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

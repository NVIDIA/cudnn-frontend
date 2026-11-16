# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests of the kernel census of the model samples (``samples/models/model_samples/census.py``) and of the kernel check a
sample makes with ``--check-kernels``: the canonical ids of the kernel families a sample launches, the reader of an exported
profiler trace, the wiring order of a decode step and its call log, the form check on the objects, and the comparison of launch
records with the plan on constructed records -- every constructed defect must FAIL with its own sentence (an extra launch, a
missing one, a swapped pair, a kernel of a fallback family where none is declared, the expected name with another grid,
another block or another shared memory, another symbol, a kernel nobody declares, an undeclared copy).  Then the same check
through a real ``Session`` on the CPU and the sample's own checks, with stand-in providers whose launches a stand-in profiler
records: the clean stand-ins pass, and an adapter that DECLARES one kernel and launches another fails.  With the profiler out,
the run is ``RAN ... (kernels not observed: <reason>)`` and never ``PASS``: held on the last line of the sample's own
``main()``, run to its end on the CPU.

And the target kernel set (``samples/models/model_samples/qwen_dense/kernel_census.json``, the stacks ``native`` and
``fe-entries``): the census file against its own three numbers and against the wiring of the model, the identity of every
block call, every reason a GPU class is not claimed, a target launch in another shape than the recorded one, "claimed
implies no fallback row and a recorded basis", the adapter of the attention gate + quantise entry over a stand-in entry,
its twin and its band on stated tensors, and ``main()`` with ``--stack native`` to its last line.  No GPU, no weights, no
profiler."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cudnn")

pytestmark = [pytest.mark.L0]

MODELS = Path(__file__).resolve().parents[3] / "samples" / "models"
GIB = 1024**3

# kernel names as the profiler reports them (heads of the names of one traced default run; a name is matched by its head)
SYMBOL = (
    "cudnn_kernel_frost_gdn_decode_fused_CfgTHREADS256TPR4K_RESIDENTTrueWARPGROUP_SPLITTrueCTAS_PER_SM0STATE_LOAD_PATHbulkSTATE_STREAMFalse"
    "MIN_CTAS_PER_SM1PDLFalsePDL_RELEASEFalsePIN_ENTRY_L_0"
)
SHAPE = ((96, 1, 1), (256, 1, 1), 42112)
TARGET_SYMBOL = SYMBOL.replace("PDLFalsePDL_RELEASEFalsePIN_ENTRY_L_0", "PDLTruePDL_RELEASETruePIN_ENTRY_LOA_0")  # the wait + release form
GATE_SYMBOL = (
    "cudnn_kernel_frost_attention_gate_quant_CfgTHREADS256PDL_RELEASETrue_tensorptrbf16gmemalign16o1_tensorptrbf16gmemalign16o1_tensorptrf32gmemo11_"
    "tensorptri8gmemalign16o1__0"
)
GATE_SHAPE = ((3, 1, 1), (256, 1, 1), 0)
GATHER_SHAPE = ((40, 1, 1), (128, 1, 1), 0)
NORM_SHAPE = ((1, 1, 1), (640, 1, 1), 80)  # of the add + norm + quantise entry at 1 row: one thread block per row
#: the symbol of that entry's kernel in two of its forms, as a traced run reported them: the form's text follows the kernel's name
NORM_SYMBOL = (
    "cudnn_kernel_frost_add_rms_norm_quant_h5120_add_nvfp4_wait_release_eps358637bd_tensorptrbf16gmemalign16o1_tensorptrbf16_gmem_align16_o_1_"
    "tensorptrbf16gmemalign16o1_tensorptrf32_gmem_o_1_0"
)
NORM_SYMBOL_FIRST = (
    "cudnn_kernel_frost_add_rms_norm_quant_h5120_store_bf16_e4m3_static_wait_release_eps358637bd_tensorptrbf16gmemalign16o1_tensorptrbf16_gmem_align16_o_1_"
    "tensorptrbf16gmemalign16o1_tensorpt_0"
)
RECORDED = {"torch": "2.14.0+cu130", "nvidia-cutlass-dsl": "4.8.0"}  # installed versions equal to the census's recorded ones
CENSUS = "target census plain / cc 10.0 / 148 SMs (recorded 2026-11-12)"
#: what the census line says was compared when the gather, the fused step, the gate and the five norms of the two-layer model
#: held: three numbers
COMPARED_8 = (
    "8 launches of a step compared by launch shape (2 the target's own compiled kernels + 6 cuDNN Frontend entries with byte-equal results "
    "+ 0 declared stand-ins)"
)
COMPARED_2 = (
    "2 launches of a step compared by launch shape (2 the target's own compiled kernels + 0 cuDNN Frontend entries with byte-equal results "
    "+ 0 declared stand-ins)"
)
WHOLE = "the whole target kernel set of a step of this model (what a complete stack launches; this stack's share is the next line): "
GATHER = "void at::native::(anonymous namespace)::indexSelectSmallIndex<BFloat16, int, unsigned int, 2, 2, -2>(at::cuda::detail::TensorInfo<...>)"
ELEMENTWISE = "void at::native::vectorized_elementwise_kernel<2, at::native::CUDAFunctorOnSelf_add<double>, std::array<char*, 2ul> >(int, ...)"
ARGMAX = "void at::native::reduce_kernel<512, 1, at::native::ReduceOp<BFloat16, at::native::ArgMaxOps<BFloat16>, unsigned int, long, 4, 4> >(...)"
SOFTMAX = "void (anonymous namespace)::softmax_warp_forward<double, double, double, 3, false, false, 32>(double*, double const*, int, int, int, ...)"
ARANGE = "void (anonymous namespace)::elementwise_kernel_with_index<int, at::native::arange_cuda_out(...)::{lambda(long)#1}>(int, ...)"
NVJET = "nvjet_sm100_tst_64x8_64x16_2x1_v_bz_splitK_TNT"
SPLIT_K = "void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, float, __nv_bfloat16, __nv_bfloat16, true>(...)"
GEMV = "std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, double, double, double, double, false, true, cublasGemvParamsEx<int, ...> >(...)"
GEMM_FP64 = "void cutlass::Kernel2<cutlass_80_tensorop_d884gemm_32x64_16x4_nn_align1>(cutlass_80_tensorop_d884gemm_32x64_16x4_nn_align1::Params)"
HOST_READ = "Memcpy DtoH (Device -> Pageable)"
COPY_IN = "Memcpy HtoD (Pageable -> Device)"


@pytest.fixture(scope="module", autouse=True)
def helper_layer():
    """``samples/models`` on ``sys.path`` for this module only: the path and every ``model_samples`` module leave with it."""
    sys.path.insert(0, str(MODELS))
    yield
    sys.path.remove(str(MODELS))
    for name in [n for n in sys.modules if n == "model_samples" or n.startswith("model_samples.")]:
        del sys.modules[name]


class _Clock:
    """Launch records with start times in the order they are made."""

    def __init__(self):
        self.now = 0.0

    def __call__(self, name, shape=((1, 1, 1), (128, 1, 1), 0), kind="kernel"):
        from model_samples.census import Launch

        self.now += 1.0
        if kind == "copy":
            return Launch(name, "copy", self.now, self.now + 0.5, bytes=4)
        return Launch(name, "kernel", self.now, self.now + 0.5, *shape, 32)


def _sites(adjacent=False):
    """The sites of one step of a small wiring: a gather (exact, by name), a reference chain, the fused step (exact, with
    the entry's stated symbol and launch shape), a multiply over a copy, the host path.  ``adjacent`` leaves the chain out:
    the two exact rows then follow each other."""
    from model_samples.census import Facts, Kernel, Site

    facts = Facts((SYMBOL,), {"PDL": False, "PDL_RELEASE": False, "THREADS": 256}, SHAPE)
    form = (("PDL", False), ("PDL_RELEASE", False))
    sites = [
        Site("embedding", -1, (Kernel("embedding_gather", "PyTorch", "exact", launches=1),), "torch.index_select"),
        Site("add_norm_quantize", 0, (Kernel("pytorch", "PyTorch"),), "torch_ref.add_norm_quantize"),
        Site("linear_attention.step", 0, (Kernel("gdn_decode_fused", "cuDNN FE", "exact", launches=1, shape="entry"),), "cudnn.GdnDecodeFused", form, facts),
        Site("projection.fp8.out", 0, (Kernel("cublas", "cuBLAS"), Kernel("pytorch", "PyTorch")), "torch.matmul over a bf16 copy"),
        Site("token_choice", -1, (Kernel("pytorch", "PyTorch", "host"),), "torch.argmax"),
    ]
    return [site for site in sites if not (adjacent and site.block == "add_norm_quantize")]


def _step(adjacent=False, fused=(SYMBOL, SHAPE)):
    """The launch records of one clean step of :func:`_sites`, ending in its host read."""
    at = _Clock()
    records = [at(GATHER, ((40, 1, 1), (128, 1, 1), 0))]
    if not adjacent:
        records += [at(ELEMENTWISE), at(COPY_IN, kind="copy"), at(ELEMENTWISE)]
    records += [at(*fused), at(ELEMENTWISE), at(NVJET, ((8, 1, 1), (256, 1, 1), 156168)), at(SPLIT_K), at(ELEMENTWISE), at(ARGMAX), at(HOST_READ, kind="copy")]
    return records


def _compare(steps, adjacent=False):
    from model_samples import census

    return census.compare(steps, _sites(adjacent))


def test_canonical_ids_of_the_kernel_families_a_sample_launches():
    """A name maps to the id of the first pattern that names it: an entry's symbol before any other cuDNN Frontend kernel, the
    gather before PyTorch's family, cuBLAS before PyTorch.  A copy is a copy in both spellings the profiler has for one."""
    from model_samples import census
    from model_samples.census import Launch, canonical

    def kernel(name):
        return canonical(Launch(name, "kernel", 0.0, 1.0))

    assert kernel(SYMBOL) == "gdn_decode_fused" and kernel(SYMBOL.replace("PDLFalsePDL_RELEASEFalse", "PDLTruePDL_RELEASETrue")) == "gdn_decode_fused"
    assert kernel(TARGET_SYMBOL) == "gdn_decode_fused" and kernel(GATE_SYMBOL) == "gate_quant", "each entry's kernel has an id of its own"
    assert kernel(NORM_SYMBOL) == kernel(NORM_SYMBOL_FIRST) == "add_norm_quant", "one id for every form of the norm entry's kernel"
    assert kernel("cudnn_kernel_frost_gdn_verify_fused_CfgTPR4K_RESIDENTTrue_0") == "cudnn_fe", "a cuDNN Frontend kernel no pattern names is no family's member"
    assert kernel(GATHER) == "embedding_gather" and kernel(GATHER.replace("SmallIndex", "LargeIndex")) == "embedding_gather"
    assert census.FAMILY_OF["embedding_gather"] == "pytorch", "the gather is a PyTorch kernel with an id of its own"
    for name in (ELEMENTWISE, ARGMAX, SOFTMAX, ARANGE, "void at::native::(anonymous namespace)::CatArrayBatchedCopy_alignedK_contig<...>(...)"):
        assert kernel(name) == "pytorch", name
    for name in (
        NVJET,
        SPLIT_K,
        GEMV,
        GEMM_FP64,
        "nvjet_sm120_tst_128x8_64x12_4x1_v_bz_TNT",
        "void gemmk1_kernel<float, 256, 5, false, cublasGemmk1Params<float> >(...)",
        "ampere_sgemm_64x64_nn",
    ):
        assert kernel(name) == "cublas", name
    assert kernel("some_other_library_kernel<float>") == "", "a kernel no pattern names has no id: the check names it"
    assert canonical(Launch(COPY_IN, "copy", 0.0, 1.0)) == "copy" and canonical(Launch("Memset (Device)", "copy", 0.0, 1.0)) == "copy"
    assert canonical(Launch("memcpy32_post", "kernel", 0.0, 1.0)) == "copy", "the copy node of a replayed launch graph can come as a kernel record"
    assert census.copy_name("memcpy32_post") and census.copy_name(HOST_READ) and not census.copy_name(NVJET)
    from model_samples.providers import factories

    named = [kernel_id for kernel_id, _ in census.FAMILIES]
    judged = {kernel.id for adapter in factories().values() for kernel in adapter.kernels if kernel.relation == "exact"}
    assert judged == {"gdn_decode_fused", "gate_quant", "add_norm_quant", "embedding_gather"} and judged <= set(named), "a pattern of its own each"
    for kernel_id in ("gdn_decode_fused", "gate_quant", "add_norm_quant"):
        assert named.index(kernel_id) < named.index("cudnn_fe"), f"{kernel_id}: in front of the catch-all"
    with pytest.raises(ValueError, match="an exact kernel states its launches per call"):
        census.Kernel("k", "PyTorch", "exact")
    with pytest.raises(ValueError, match="only an exact kernel states"):
        census.Kernel("pytorch", "PyTorch", "fallback", launches=3)
    with pytest.raises(ValueError, match="is not declared"):
        census.Kernel("k", "PyTorch", "some other relation")


def _trace_text(events):
    """An exported profiler trace as ``export_chrome_trace`` lays it out: a header, then the ``traceEvents`` array."""
    head = {"schemaVersion": 1, "deviceProperties": [{"id": 0, "name": "stated device", "numSms": 148}], "displayTimeUnit": "ms"}
    return json.dumps(head, indent=2)[:-2] + ',\n  "traceEvents": [\n' + ",\n".join(json.dumps(event, indent=2) for event in events) + "\n  ]\n}"


def _kernel_event(name, ts, grid=(96, 1, 1), block=(256, 1, 1), shared=42112, registers=103):
    arguments = {"device": 0, "stream": 7, "correlation": int(ts), "registers per thread": registers, "shared memory": shared, "grid": list(grid)}
    arguments.update({"block": list(block), "graph id": 0, "graph node id": 0})
    return {"ph": "X", "cat": "kernel", "name": name, "pid": 0, "tid": 7, "ts": ts, "dur": 2.5, "args": arguments}


def _copy_event(name, ts, size=4, cat="gpu_memcpy"):
    return {
        "ph": "X",
        "cat": cat,
        "name": name,
        "pid": 0,
        "tid": 7,
        "ts": ts,
        "dur": 1.0,
        "args": {"stream": 7, "bytes": size, "graph id": 2, "graph node id": 8589934593},
    }


def test_the_exported_trace_is_read_record_by_record(tmp_path):
    """Kernel and copy records are kept with their launch arguments, in start order; the runtime's calls, the flow arrows and
    the metadata are passed over.  The file is read in chunks: a chunk boundary inside an event, or inside the key that opens
    the array, changes nothing.  A file that is not such a trace, or ends inside its array, is a ``ValueError``."""
    from model_samples import census

    events = [
        {"ph": "M", "name": "process_name", "ts": 1.0, "pid": 0, "tid": 0, "args": {"name": "python"}},
        _kernel_event(SYMBOL, 3000.0),
        {"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "pid": 1, "tid": 1, "ts": 900.0, "dur": 4.0, "args": {"correlation": 20}},
        _kernel_event(GATHER, 1000.0, grid=(40, 1, 1), block=(128, 1, 1), shared=0, registers=28),
        {"ph": "f", "id": 20, "cat": "ac2g", "name": "ac2g", "pid": 0, "tid": 7, "ts": 1000.0, "bp": "e"},
        _copy_event(COPY_IN, 2000.0),
        {"ph": "X", "cat": "kernel", "name": "kernel_without_arguments", "pid": 0, "tid": 7, "ts": 3500.0, "dur": 1.0},
        _copy_event("Memset (Device)", 3800.0, size=64, cat="gpu_memset"),
        _copy_event(HOST_READ, 4000.0),
    ]
    path = tmp_path / "trace.json"
    path.write_text(_trace_text(events))
    records = census.read_trace(str(path))
    assert [r.name for r in records] == [GATHER, COPY_IN, SYMBOL, "kernel_without_arguments", "Memset (Device)", HOST_READ], "start order; six of nine events"
    assert [r.kind for r in records] == ["kernel", "copy", "kernel", "kernel", "copy", "copy"]
    fused = records[2]
    assert (
        fused.shape == SHAPE and fused.registers == 103 and fused.graph == (0, 0) and (fused.start, fused.end) == pytest.approx((3.0, 3.0025))
    ), "milliseconds"
    assert records[0].shape == ((40, 1, 1), (128, 1, 1), 0) and records[1].bytes == 4 and records[1].graph == (2, 8589934593)
    assert records[3].shape == (None, None, None) and records[3].registers is None, "a record without launch arguments carries none"
    for chunk in (1, 7, 64, 333):
        assert census.read_trace(str(path), chunk=chunk) == records, f"read in chunks of {chunk} characters"
    steps, rest = census.split_steps(records + [records[0]])
    assert [len(step) for step in steps] == [6] and steps[0][-1].name == HOST_READ and rest == [records[0]], "a step ends with its host read"
    whole = path.read_text()
    path.write_text(whole[: whole.index("kernel_without_arguments")])
    with pytest.raises(ValueError, match="ends inside its traceEvents array"):
        census.read_trace(str(path), chunk=64)
    path.write_text('{"schemaVersion": 1, "events": []}')
    with pytest.raises(ValueError, match="has no traceEvents array"):
        census.read_trace(str(path))


def test_the_step_facts_count_a_copy_in_either_spelling():
    from model_samples.timing import step_facts

    records = [("kernel_a", 0.0, 1.0), ("memcpy32_post", 1.0, 2.0), ("Memcpy DtoD (Device -> Device)", 2.0, 3.0), (HOST_READ, 3.0, 4.0)]
    assert step_facts(records) == [(4.0, 0.0, 1, 3)], "one kernel and three copies: the copy node of a replayed graph is a copy"


def test_the_wiring_order_of_a_decode_step():
    """``wiring()`` is the order ``step`` calls its blocks in, stated without running it: its counts are the model's call
    counts, and a step of the tiny geometry on the CPU makes exactly these calls."""
    from collections import Counter

    from model_samples.qwen_dense import QwenDense

    full = QwenDense()
    wiring = full.wiring()
    assert len(wiring) == full.calls_per_step() == 484 and Counter(block for block, _ in wiring) == {b: sum(c) for b, c in full.calls().items()}
    assert wiring[0] == ("embedding", -1) and wiring[-3:] == (("add_norm_quantize", 64), ("output_head", -1), ("token_choice", -1))
    assert [block for block, layer in wiring if layer == 3] == [
        "add_norm_quantize",
        "projection.fp8.in",
        "attention.prepare",
        "attention.core",
        "attention.gate_quantize",
        "projection.fp8.out",
        "add_norm_quantize",
        "mlp.gate_up",
        "mlp.down",
    ], "the fourth layer is an attention layer: nine calls"
    assert [block for block, layer in wiring if layer == 0][2] == "linear_attention.step" and len([1 for _, layer in wiring if layer == 0]) == 7
    reduced = QwenDense.reduced(2)
    assert len(reduced.wiring()) == reduced.calls_per_step() == 20
    session, text = _tiny_session(sequences=2, stack="reference")
    session.prefill([text.encode("ab" + str(s)) for s in range(2)])
    session._call_log = []
    session._advance(torch.zeros(2, dtype=torch.int32))
    call_log, session._call_log = session._call_log, None
    assert [(call.block, call.layer) for call in call_log] == list(session.model.wiring()), "what a step calls is the wiring"
    assert {call.rows for call in call_log} == {2} and {(call.instance, call.form) for call in call_log} == {("", ())}, "no entry instance, nothing passed"

    class Late:  # a block that knows what served a call only once the call was made
        said = "before the call"

        def __call__(self, operand):
            self.said = f"after the call on {operand}"
            return operand

        def serving(self):
            return self.said, ()

    from model_samples.census import Call

    session.plan["late"], session._call_log = Late(), []
    result = session._call("late", -1, 7)
    assert result == 7 and session._call_log == [Call("late", -1, 2, "after the call on 7", ())], "the line is written after the call"
    session._call_log = None


def test_a_clean_step_passes_and_says_what_it_attributed():
    from model_samples import census

    sites = _sites()
    traced = census.Traced.of(sites, [], _step() + _step(), 2, "", rows=1)
    status, lines = traced.judge()
    assert status == census.PASS and lines == [], lines
    assert traced.summary() == (
        "2 traced steps: 9 launches per step; 2 on exact rows, in the wiring order; 7 attributed by family to fallback rows and the host path "
        "(cublas 2, pytorch 5); 2 copies per step"
    )
    details = traced.details()
    assert details[0] == "embedding: torch.index_select, 1 launch per step (1 call x 1 per call); embedding_gather by name (no launch shape is declared for it)"
    step = "linear_attention.step: cudnn.GdnDecodeFused [PDL False, PDL_RELEASE False], 1 launch per step (1 call x 1 per call); "
    assert details[1] == step + f"the entry states grid (96, 1, 1), block (256, 1, 1), shared memory 42,112 B and the kernel {SYMBOL}"
    assert details[2].startswith("chosen by a library inside the fallback rows (printed, not gated): cublas: ") and NVJET + " x 1" in details[2]
    assert not any("pytorch" in line for line in details[2:]), "PyTorch's own kernels are not a library's choice"


def test_every_constructed_defect_fails_with_its_own_sentence():
    """One defect per case, on the records of an otherwise clean step.  Each must fail, and with the sentence that names it."""
    at = _Clock()
    at.now = 100.0
    step = _step()
    fused = step.index(next(r for r in step if r.name == SYMBOL))
    plan = "the plan 1 = 1 call(s) x 1 per call"

    def one(result, expected):
        assert not result.ok and result.sentences() == [f"step 0: {expected}"], result.sentences()

    # an extra launch of an exact kernel
    extra = step[: fused + 1] + [at(SYMBOL, SHAPE)] + step[fused + 1 :]
    one(
        _compare([extra]),
        f"launched gdn_decode_fused ({SYMBOL[:93] + '...'!r}) after linear_attention.step (layer 0), where the plan has no such launch: "
        f"the step has 2 launch(es) of gdn_decode_fused, {plan}",
    )
    # a missing launch
    one(
        _compare([step[:fused] + step[fused + 1 :]]),
        f"linear_attention.step (layer 0) did not launch gdn_decode_fused: the step has 0 launch(es) of gdn_decode_fused, {plan}",
    )
    # a swapped pair
    adjacent = _step(adjacent=True)
    swapped = [adjacent[1], adjacent[0]] + adjacent[2:]
    one(
        _compare([swapped], adjacent=True),
        f"the exact launches are not in the wiring order: launch 1 of 2 is gdn_decode_fused ({SYMBOL[:93] + '...'!r}), the wiring has embedding there; "
        "it is swapped with linear_attention.step (layer 0)",
    )
    # a kernel of a fallback family between two exact rows
    one(
        _compare([adjacent[:1] + [at(NVJET)] + adjacent[1:]], adjacent=True),
        f"launched {NVJET!r} between embedding and linear_attention.step (layer 0): a kernel of the fallback family cublas, which no row there declares; "
        "nothing may sit between two exact rows",
    )
    # ... and one of a family that the rows between two exact rows do not declare (a reference chain declares PyTorch's kernels only)
    one(
        _compare([step[:1] + [at(NVJET)] + step[1:]]),
        f"launched {NVJET!r} between embedding and linear_attention.step (layer 0): a kernel of the fallback family cublas, which no row there declares",
    )
    # the expected name with another grid, another block, another shared memory
    stated = "cudnn.GdnDecodeFused states grid (96, 1, 1), block (256, 1, 1), shared memory 42,112 B"
    for shape, text in (
        (((48, 1, 1), (256, 1, 1), 42112), "grid (48, 1, 1), block (256, 1, 1), shared memory 42,112 B"),
        (((96, 1, 1), (128, 1, 1), 42112), "grid (96, 1, 1), block (128, 1, 1), shared memory 42,112 B"),
        (((96, 1, 1), (256, 1, 1), 42128), "grid (96, 1, 1), block (256, 1, 1), shared memory 42,128 B"),
    ):
        one(_compare([_step(fused=(SYMBOL, shape))]), f"linear_attention.step (layer 0) launched gdn_decode_fused with {text}; {stated}")
    # another symbol of the same kernel: the entry did not compile it for this run
    other = SYMBOL.replace("PDLFalsePDL_RELEASEFalse", "PDLTruePDL_RELEASETrue")
    one(
        _compare([_step(fused=(other, SHAPE))]),
        f"linear_attention.step (layer 0) launched {other!r}; cudnn.GdnDecodeFused names {SYMBOL!r} for this run",
    )
    # a kernel no pattern names, and a cuDNN Frontend kernel no block declares
    one(
        _compare([step[:2] + [at("some_other_library_kernel<float>")] + step[2:]]),
        "launched 'some_other_library_kernel<float>' between embedding and linear_attention.step (layer 0); no block of the plan launches it",
    )
    one(
        _compare([step[:-1] + [at("cudnn_kernel_frost_gdn_verify_fused_CfgTPR4_0"), step[-1]]]),
        "launched 'cudnn_kernel_frost_gdn_verify_fused_CfgTPR4_0' after linear_attention.step (layer 0): the kernel cudnn_fe, "
        "which no block of the plan declares",
    )
    # a copy between two exact rows
    one(
        _compare([adjacent[:1] + [at("Memcpy DtoD (Device -> Device)", kind="copy")] + adjacent[1:]], adjacent=True),
        "an undeclared copy 'Memcpy DtoD (Device -> Device)' between embedding and linear_attention.step (layer 0): declared are the one host read of a step "
        "and the copies of a fallback row",
    )
    # the same defect in every traced step is one line
    three = _compare([_step(fused=(SYMBOL, ((48, 1, 1), (256, 1, 1), 42112))) for _ in range(3)])
    assert three.sentences() == [
        "step 0 (and 2 more of the 3 traced steps): linear_attention.step (layer 0) launched gdn_decode_fused with grid (48, 1, 1), block (256, 1, 1), "
        f"shared memory 42,112 B; {stated}"
    ]
    assert _compare([step, _step()]).ok, "the control: the clean step passes"


def test_nothing_sits_between_two_launches_of_one_exact_row():
    """A block whose call launches its kernel twice: the two launches follow each other.  A launch between them fails, also
    where the same row declares a family for a chain of its own -- that chain runs before or after the pair, not inside it."""
    from model_samples.census import Kernel, Site, compare

    at = _Clock()
    pair = Kernel("gdn_decode_fused", "cuDNN FE", "exact", launches=2)
    host = Site("token_choice", -1, (Kernel("pytorch", "PyTorch", "host"),), "torch.argmax")
    alone = [Site("pair", 0, (pair,), "an entry that launches twice"), host]
    with_chain = [Site("pair", 0, (pair, Kernel("pytorch", "PyTorch")), "an entry that launches twice, and a chain"), host]
    first, second, choice, read = at(SYMBOL, SHAPE), at(SYMBOL, SHAPE), at(ARGMAX), at(HOST_READ, kind="copy")
    inside = (
        f"step 0: launched {ELEMENTWISE[:93] + '...'!r} inside the launches of pair (layer 0): a kernel of the fallback family pytorch, "
        "which no row there declares; nothing may sit between two exact rows"
    )
    for sites in (alone, with_chain):
        assert compare([[first, second, choice, read]], sites).ok and compare([[first, second, choice, read]], sites).exact == 2
        assert compare([[first, at(ELEMENTWISE), second, choice, read]], sites).sentences() == [inside]
    assert compare([[at(ELEMENTWISE), first, second, at(ELEMENTWISE), choice, read]], with_chain).ok, "the row's own chain, around the pair"
    assert compare([[first, choice, read]], alone).sentences() == [
        "step 0: pair (layer 0) did not launch gdn_decode_fused: the step has 1 launch(es) of gdn_decode_fused, the plan 2 = 1 call(s) x 2 per call"
    ]


def test_a_second_host_read_and_a_call_log_that_is_not_the_wiring_fail():
    from model_samples import census
    from model_samples.census import Call

    sites = _sites()
    step = _step()
    # a block that reads the device inside a step: the records cannot be cut into the traced steps
    traced = census.Traced.of(sites, [], step[:3] + [step[-1]] + step[3:], 1, "", rows=1)
    status, lines = traced.judge()
    assert status == census.FAIL and lines == [
        "copies: the records show 2 host reads for 1 traced steps: a block reads the device inside a step, and only the one host read of a step is declared"
    ]
    assert traced.summary() == "the launch records cannot be cut into the traced steps"
    # a step that did not make the calls of the wiring
    call_log = [[Call(site.block, site.layer, 1, "", site.form) for site in sites]]
    assert census.compare([step], sites, call_log).ok
    del call_log[0][1]
    assert census.compare([step], sites, call_log).sentences() == [
        "step 0: the step made 4 block calls, the wiring has 5; call 2 is linear_attention.step (layer 0), the wiring has add_norm_quantize (layer 0) there"
    ]
    short = [[Call(site.block, site.layer, 1, "", site.form) for site in sites[:-1]]]  # every call of the wiring but its last
    assert census.compare([step], sites, short).sentences() == [
        "step 0: the step made 4 block calls, the wiring has 5; call 5 is nothing, the wiring has token_choice there"
    ]


def test_the_form_is_checked_on_the_objects():
    """Item 0: every field a row names equals the entry's own form, and every logged call was served by an entry instance that
    states the row's form.  A launch record cannot show either; a row whose entry declines to state its launch is a wrong
    table, and so is a row that names no form for an entry that states its launch: nothing would be compared, and the entry
    would be called on its own pick."""
    from dataclasses import replace

    from model_samples import census
    from model_samples.census import Call, Facts

    sites = _sites()
    assert census.check_forms(sites) == []
    step = sites[2]
    other = replace(step, facts=Facts((SYMBOL,), {"PDL": True, "PDL_RELEASE": False}, SHAPE))
    assert census.check_forms([other]) == ["linear_attention.step: the row's form says PDL False, cudnn.GdnDecodeFused states PDL True for this run"]
    unknown = replace(step, form=(("PDL", False), ("TOKEN_TILE", 8)))
    assert census.check_forms([unknown]) == ["linear_attention.step: the row's form names TOKEN_TILE, which the form of cudnn.GdnDecodeFused does not have"]
    declined = replace(step, facts=None, wrong="cudnn.GdnDecodeFused serves at most 65535 rows; got rows=70000")
    wrong = "linear_attention.step: cudnn.GdnDecodeFused states no launch for this run (cudnn.GdnDecodeFused serves at most 65535 rows; got rows=70000): "
    assert census.check_forms([declined]) == [wrong + "the provider table is wrong"]
    # a call that another entry instance served: the instance states another form than the row, in every step alike -> one sentence
    other_instance = Call("linear_attention.step", 0, 1, "cudnn.GdnDecodeFused #2", (("PDL", True), ("PDL_RELEASE", True)))
    assert census.check_forms(sites, [[other_instance], [other_instance]]) == [
        "linear_attention.step (layer 0): the adapter served the call with the form [PDL True, PDL_RELEASE True] (cudnn.GdnDecodeFused #2); "
        "the row says [PDL False, PDL_RELEASE False]"
    ]
    # ... and a call that nothing with a form served (an adapter that passes nothing), under a row that names one
    assert census.check_forms(sites, [[Call("linear_attention.step", 0, 1, "", ())]]) == [
        "linear_attention.step (layer 0): the adapter served the call with the form [] (no entry instance); the row says [PDL False, PDL_RELEASE False]"
    ]
    assert census.check_forms(sites, [[Call(site.block, site.layer, 1, "", site.form) for site in sites]]) == [], "the control: every call as its row says"
    # a row that names no form for an entry that states its launch: every field would compare equal, because there is none
    bare = replace(step, form=(), facts=Facts((SYMBOL,), {"PDL": True, "PDL_RELEASE": True}, SHAPE))
    no_form = "linear_attention.step: the row names no form; cudnn.GdnDecodeFused would be called on its own pick"
    assert census.check_forms([bare]) == [no_form]
    assert census.check_forms([replace(bare, facts=None, wrong="declined")])[1:] == [no_form], "said also where the entry stated nothing"
    assert census.check_forms([replace(sites[0], form=())]) == [], "a kernel known by its name alone has no entry to pick anything"
    status, lines = census.Traced.of([sites[0], sites[1], bare, *sites[3:]], [], _step(), 1, "", rows=1).judge()
    assert status == census.FAIL and lines == [f"form: {no_form}"], "it fails a pass whose launch records are clean"
    # a form that does not hold fails a pass whose launch records are clean, and one that observed nothing
    for records, missing in ((_step(), ""), ([], "the profiler failed (no tracing library)")):
        status, lines = census.Traced.of([sites[0], sites[1], other, *sites[3:]], [], records, 1, missing, rows=1).judge()
        assert status == census.FAIL and lines == [
            "form: linear_attention.step: the row's form says PDL False, cudnn.GdnDecodeFused states PDL True for this run"
        ]


def test_kernels_that_were_not_observed_are_neither_a_pass_nor_a_failure():
    from model_samples import census

    sites = _sites()
    nothing = census.Traced.of(sites, [], [], 2, "the profiler failed (no tracing library)", rows=1)
    assert nothing.judge() == (census.NOT_OBSERVED, []) and nothing.unobserved == "the profiler failed (no tracing library)"
    assert nothing.summary() == "not observed: the profiler failed (no tracing library)"
    assert census.Traced.of(sites, [], [], 2, "", rows=1).judge()[0] == census.NOT_OBSERVED, "no record and no reason is still nothing observed"
    # records that carry names and no launch arguments: the name is compared, the launch shape cannot be
    bare = [record._replace(grid=None, block=None, shared_memory=None) for record in _step()]
    shapeless = census.Traced.of(sites, [], bare, 1, "", rows=1)
    assert shapeless.judge() == (census.NOT_OBSERVED, [])
    assert shapeless.unobserved == "the profiler's kernel records carry no launch arguments (grid, block, shared memory)"
    assert shapeless.summary().endswith("; launch shapes not observed: the profiler's kernel records carry no launch arguments (grid, block, shared memory)")
    # ... and a kernel that is missing is a failure all the same
    missing = census.Traced.of(sites, [], [record for record in bare if record.name != SYMBOL], 1, "", rows=1)
    assert missing.judge()[0] == census.FAIL


def test_provider_rows_state_a_form_and_a_rows_range():
    """``form`` is what the adapter passes to its entry; ``rows`` is a third selection range, sequences x tokens."""
    from model_samples.blocks import DeviceFacts, ProviderRow, select_row
    from model_samples.census import form_text
    from model_samples.qwen_dense import configs

    facts = DeviceFacts((10, 0), "stated device", 148, 180 * GIB, 80 * GIB)
    rows = [
        ProviderRow(("any",), (1, None), (1, None), "fe.gdn_decode_fused", "fe-cublas", serves="rows 1-8", rows=(1, 8), form=(("PDL", True),)),
        ProviderRow(("any",), (1, None), (1, None), "torch_ref.gdn_decode_step", "fe-cublas", serves="any"),
    ]
    assert select_row("b", rows, facts, 8, 1, "fe-cublas").key == "fe.gdn_decode_fused" and select_row("b", rows, facts, 2, 4, "fe-cublas").form == (
        ("PDL", True),
    )
    assert select_row("b", rows, facts, 3, 3, "fe-cublas").key == "torch_ref.gdn_decode_step", "3 sequences x 3 tokens are 9 rows: past the first row's range"
    with pytest.raises(NotImplementedError, match="cudnn.GdnDecodeFused serves rows 1-8; this run has 9 sequences x 1 tokens per sequence"):
        select_row("b", rows[:1], facts, 9, 1, "fe-cublas")
    assert rows[1].form == () and rows[1].rows == (1, None) and form_text(rows[1].form) == "" and form_text(rows[0].form) == " [PDL True]"
    step = configs.PROVIDERS["linear_attention.step"][0]
    assert step.form == (("PDL", False), ("PDL_RELEASE", False)), "the fe-cublas row says what the entry does when nothing is passed"


def _plan(sequences=1, layers=2, table=None, **kwargs):
    from model_samples.blocks import DeviceFacts, plan
    from model_samples.qwen_dense import QwenDense

    model = QwenDense.reduced(layers)
    facts = DeviceFacts((10, 0), "stated device", 148, 180 * GIB, 80 * GIB)
    return plan(table or model.provider_table, model.regimes, facts, sequences, model=model, **kwargs), model


def test_every_row_of_an_entry_that_states_its_launch_names_a_form():
    """The family's provider table: a row whose adapter declares a kernel that is judged against the entry's own launch facts
    names at least one field of the entry's form.  With none the adapter passes nothing, the entry is called on its own pick
    and the form check has nothing to compare -- so the check fails such a row, through the real adapter and entry."""
    from dataclasses import replace

    from model_samples import census
    from model_samples.providers import factories
    from model_samples.qwen_dense import QwenDense, configs

    build = factories()
    stated = [(block, row) for block, rows in configs.PROVIDERS.items() for row in rows if any(kernel.shape == "entry" for kernel in build[row.key].kernels)]
    assert ("linear_attention.step", "fe.gdn_decode_fused") in [(block, row.key) for block, row in stated], "the table has such rows"
    for block, row in stated:
        assert row.form, f"{block}: the row of {row.key} names no form: its entry would be called on its own pick"
    rows = QwenDense.reduced(2).provider_table["linear_attention.step"]
    table = dict(QwenDense.reduced(2).provider_table)
    table["linear_attention.step"] = [replace(rows[0], form=()), *rows[1:]]
    plan, model = _plan(table=table, options={"device": "cpu"})  # a host entry, built without any knob: its own pick
    sites = census.expected(plan, model)
    assert sites[3].form == () and sites[3].facts is not None and "cudnn.GdnDecodeFused: grid (" in census.stated(plan)
    assert census.check_forms(sites) == ["linear_attention.step: the row names no form; cudnn.GdnDecodeFused would be called on its own pick"]


def test_an_entry_that_states_no_launch_for_the_run_is_a_wrong_table(monkeypatch):
    """A block declares a kernel that is judged against its entry's launch facts, and the adapter has none for the run -- it
    answers nothing, or the entry declines the row count the row claims: the site carries the reason and the check fails."""
    from model_samples import census
    from model_samples.providers import fe

    plan, model = _plan(options={"device": "cpu"})
    wrong = "linear_attention.step: cudnn.GdnDecodeFused states no launch for this run ({}): the provider table is wrong"
    monkeypatch.setattr(fe.GdnDecodeFusedBlock, "launch_facts", lambda self, rows: None)
    sites = census.expected(plan, model)
    assert sites[3].facts is None and census.check_forms(sites) == [wrong.format("the adapter states no launch facts")]

    def declines(self, rows):
        raise NotImplementedError(f"cudnn.GdnDecodeFused serves at most 8 rows; got rows={rows}")

    monkeypatch.setattr(fe.GdnDecodeFusedBlock, "launch_facts", declines)
    assert census.check_forms(census.expected(plan, model, rows=9)) == [wrong.format("cudnn.GdnDecodeFused serves at most 8 rows; got rows=9")]
    assert "cudnn.GdnDecodeFused: states no launch for this run (cudnn.GdnDecodeFused serves at most 8 rows; got rows=9)" in census.stated(plan, rows=9)


def test_the_kernel_table_of_the_default_plan():
    """The first part of the kernel table, from the plan alone: per block its census entries and how they will be judged."""
    from model_samples import census

    plan, model = _plan(options={"device": "cpu"})  # a host entry: its launch facts are read without a GPU
    text = census.table(plan).splitlines()
    assert text[0] == "kernels, planned (decided before the load from the family's configs.py; stack fe-cublas, 1 sequence, cc 10.0, 148 SMs)"
    assert text[1].split() == ["block", "calls", "provider", "entry", "[form]", "kernel", "launches", "checked"]
    row = {line.split()[0]: " ".join(line.split()) for line in text[2:-1]}
    assert row["embedding"] == "embedding 1 PyTorch torch.index_select embedding_gather 1 name"
    step = "linear_attention.step 1 cuDNN FE cudnn.GdnDecodeFused [PDL False, PDL_RELEASE False] gdn_decode_fused 1 "
    step += "name and launch shape, as the entry states them"
    assert row["linear_attention.step"] == step
    assert row["mlp.down"] == "mlp.down 2 cuBLAS torch.matmul over a bf16 copy cublas, pytorch -- by family; fallback: bf16 copy of 4-bit weights"
    assert row["attention.core"].endswith("pytorch, cublas -- by family; fallback: reference chain over a plain per-layer cache")
    assert row["token_choice"] == "token_choice 1 PyTorch torch.argmax pytorch -- host path, not compared"
    assert text[-1] == (
        "  per step: 2 launches on 2 exact rows, checked one by one in the wiring order; the launches of 17 fallback calls and of the host path "
        "are attributed by family"
    )
    assert len(plan) == len(text) - 3, "one line per block of the plan"
    kinds = {name: tuple(kernel.relation for kernel in block.kernels) for name, block in plan.items()}
    assert all(kinds.values()), "every block of the plan declares its kernels"
    assert [name for name, relations in kinds.items() if "exact" in relations] == ["embedding", "linear_attention.step"]
    sites = census.expected(plan, model)
    assert [(site.block, site.layer) for site in sites] == list(model.wiring()) and sites[3].form == (("PDL", False), ("PDL_RELEASE", False))
    stated = sites[3].facts  # of the real entry, built for the host: the row's form is the entry's own answer, field by field
    assert stated is not None and (stated.form["PDL"], stated.form["PDL_RELEASE"], stated.form["ROWS"]) == (False, False, 1) and stated.symbols == ()
    assert census.check_forms(sites) == [] and len(stated.shape) == 3 and stated.shape[0][1:] == (1, 1), "one thread block column per row"
    assert "no kernel compiled yet" in census.stated(plan) and "cudnn.GdnDecodeFused [PDL False, PDL_RELEASE False]: grid (" in census.stated(plan)
    four, _ = _plan(sequences=4, layers=4)
    assert census.table(four).splitlines()[-1].startswith("  per step: 4 launches on 2 exact rows"), "three linear-attention layers and the gather"


def test_a_row_whose_form_the_adapter_cannot_pass_is_a_wrong_table():
    from dataclasses import replace

    from model_samples.blocks import DeviceFacts, plan
    from model_samples.qwen_dense import QwenDense

    model = QwenDense.reduced(2)
    rows = model.provider_table["linear_attention.step"]
    table = dict(model.provider_table)
    table["linear_attention.step"] = [replace(rows[0], form=(("PDL", True), ("TOKEN_TILE", 8))), *rows[1:]]
    with pytest.raises(
        ValueError, match=r"provider row fe.gdn_decode_fused: its form names TOKEN_TILE; this adapter passes PDL, PDL_RELEASE to cudnn.GdnDecodeFused"
    ):
        plan(table, model.regimes, DeviceFacts((10, 0), "stated device", 148, 0, 0), 1, model=model)


def test_the_adapter_builds_its_entry_from_the_rows_form_and_states_the_entrys_facts(monkeypatch):
    """The cuDNN Frontend adapter over a stand-in entry: the row's form reaches the constructor as the entry's own arguments,
    one entry instance per form, and ``launch_facts`` hands over the entry's three answers unchanged.  ``serving`` is an
    observation: after a call it names the instance that was called and the row's fields as THAT instance states them, so
    an entry built without the row's knobs and a call served from another instance both show in the call log."""
    from dataclasses import replace
    from types import SimpleNamespace

    import cudnn

    from model_samples import census
    from model_samples.blocks import DeviceFacts
    from model_samples.providers import fe, torch_ref
    from model_samples.qwen_dense import QwenDense, configs

    built = []

    class Entry:
        def __init__(self, device, eps, geometry=None, **knobs):
            self.knobs, self.calls = knobs, []
            built.append(self)

        def __call__(self, rows, *operands):
            self.calls.append(rows)
            return torch.zeros((rows, 4), dtype=torch.uint8)

        def kernel_symbols_for(self, rows):
            return tuple([SYMBOL] if rows == 1 else [])

        def form_for(self, rows):
            return {"PDL": bool(self.knobs.get("pdl")), "PDL_RELEASE": bool(self.knobs.get("pdl_release")), "ROWS": rows}

        def launch_shape_for(self, rows):
            if rows > 8:
                raise NotImplementedError(f"this stand-in serves 8 rows; got rows={rows}")
            return ((96, rows, 1), (256, 1, 1), 42112)

    monkeypatch.setattr(cudnn, "GdnDecodeFused", Entry, raising=False)
    monkeypatch.setattr(fe, "_ENTRIES", {})
    model = QwenDense.reduced(2)
    row = configs.PROVIDERS["linear_attention.step"][0]
    facts = DeviceFacts((10, 0), "stated device", 148, 0, 0)
    options = {"device": "cpu", "check_kernels": True}

    def call(block, rows):  # one call of the adapter on operands of ``rows`` rows; the stand-in entry reads none of them
        weights = SimpleNamespace(conv_weight=None, a_log=None, dt_bias=None, norm_weight=None, out_scale=None, ba_weight=None, out_scale_value=1.0)
        return block(torch.zeros((rows, 4)), None, None, None, weights, SimpleNamespace(conv=None, state=None))

    plain = fe.GdnDecodeFusedBlock(row, model, facts, options)
    plain.check_support(1, 1)
    assert plain.serving() == ("cudnn.GdnDecodeFused", ()), "before a call nothing has served and nothing was passed"
    stated = plain.launch_facts(1)
    assert built[0].knobs == {"pdl": False, "pdl_release": False} and plain.serving() == ("cudnn.GdnDecodeFused", ()), "stating the facts is not a call"
    assert stated == ((SYMBOL,), {"PDL": False, "PDL_RELEASE": False, "ROWS": 1}, ((96, 1, 1), (256, 1, 1), 42112))
    with pytest.raises(NotImplementedError, match="this stand-in serves 8 rows"):
        plain.launch_facts(9)
    call(plain, 1)
    assert built[0].calls == [1] and plain.serving() == ("cudnn.GdnDecodeFused #1", (("PDL", False), ("PDL_RELEASE", False)))
    waiting = fe.GdnDecodeFusedBlock(replace(row, form=(("PDL", True),)), model, facts, options)
    assert waiting.launch_facts(4).form["PDL"] is True and built[1].knobs == {"pdl": True} and waiting.serving() == ("cudnn.GdnDecodeFused", ())
    call(waiting, 4)
    assert built[1].calls == [4] and waiting.serving() == ("cudnn.GdnDecodeFused #2", (("PDL", True),)), "the fields the row names, as the instance states them"
    assert fe.GdnDecodeFusedBlock(row, model, facts, options).launch_facts(1) == stated and len(built) == 2, "one entry per form, shared by every block"
    assert "frost/launch_facts.py" in fe.SOURCE_FILES, "the source hash of a run covers the module the facts come from"
    chain = torch_ref.GdnDecodeStep(replace(row, key="torch_ref.gdn_decode_step"), model, facts, options)
    assert chain.serving() == ("", ()) and chain.launch_facts(1) is None, "a block without an entry passes nothing, whatever form its row names"

    # The call log is an observation of the object that was called, not a copy of the row.  (1) A call served from ANOTHER
    # instance than the one whose facts the check read -- an adapter that owns several: the row's fields equal the facts of
    # the right instance, so the first half of item 0 holds and only the call log shows it.
    site = census.Site("linear_attention.step", 0, plain.kernels, plain.entry, tuple(row.form), stated)
    assert census.check_forms([site], [[census.Call("linear_attention.step", 0, 1, *plain.serving())]]) == []
    with monkeypatch.context() as patch:
        patch.setattr(plain, "_build", lambda device: built[1])
        call(plain, 1)
    assert built[1].calls == [4, 1] and plain.serving() == ("cudnn.GdnDecodeFused #2", (("PDL", True), ("PDL_RELEASE", False)))
    assert census.check_forms([site], [[census.Call("linear_attention.step", 0, 1, *plain.serving())]]) == [
        "linear_attention.step (layer 0): the adapter served the call with the form [PDL True, PDL_RELEASE False] (cudnn.GdnDecodeFused #2); "
        "the row says [PDL False, PDL_RELEASE False]"
    ]
    # (2) A row that says PDL True over an adapter that passes nothing: the entry is built on its own pick, and the log says so.
    with monkeypatch.context() as patch:
        patch.setattr(fe, "_ENTRIES", {})
        patch.setattr(fe.GdnDecodeFusedBlock, "_arguments", lambda self: {})
        silent = fe.GdnDecodeFusedBlock(replace(row, form=(("PDL", True), ("PDL_RELEASE", True))), model, facts, options)
        call(silent, 2)
        assert built[2].knobs == {} and silent.serving() == ("cudnn.GdnDecodeFused #1", (("PDL", False), ("PDL_RELEASE", False)))
    # (3) An instance that states no form for the rows it has just served shows none: never the row's.

    def no_form(self, rows):
        raise NotImplementedError(f"no form at rows={rows}")

    with monkeypatch.context() as patch:
        patch.setattr(Entry, "form_for", no_form)
        mute = fe.GdnDecodeFusedBlock(row, model, facts, options)
        call(mute, 1)
        assert built[0].calls == [1, 1] and mute.serving() == ("cudnn.GdnDecodeFused #1", ())
    # the answer is asked of an instance once per row count, not once per call of a traced step
    asked = []
    with monkeypatch.context() as patch:
        patch.setattr(Entry, "form_for", lambda self, rows: asked.append(rows) or {"PDL": False, "PDL_RELEASE": False})
        again = fe.GdnDecodeFusedBlock(row, model, facts, options)
        for rows in (1, 1, 1, 2, 2):
            call(again, rows)
            assert again.serving() == ("cudnn.GdnDecodeFused #1", (("PDL", False), ("PDL_RELEASE", False)))
    assert asked == [1, 2]

    class Older:  # an installed package whose entry states no launch facts
        pass

    monkeypatch.setattr(cudnn, "GdnDecodeFused", Older, raising=False)
    with pytest.raises(NotImplementedError, match=r"the kernel check needs the launch facts of cudnn.GdnDecodeFused \(kernel_symbols_for, form_for"):
        fe.GdnDecodeFusedBlock(row, model, facts, options).check_support(1, 1)
    fe.GdnDecodeFusedBlock(row, model, facts, {"device": "cpu"}).check_support(1, 1)  # a run that does not check its kernels needs none


# ----------------------------------------------------------------------------------------------- through a session, on the CPU
def _tiny_model():
    from model_samples.qwen_dense import Geometry, QwenDense

    geometry = Geometry(
        layer_kinds=("linear_attention", "full_attention"),
        hidden=64,
        vocab=272,
        mlp=96,
        la_key_heads=1,
        la_value_heads=2,
        la_head_dim=8,
        attn_heads=2,
        attn_kv_heads=1,
        attn_head_dim=8,
        rotary_dim=4,
    )
    return QwenDense(geometry, reductions=("tiny geometry",))


def _census_of(model):
    """The family's census with its one section re-stated for the layer widths of ``model``: the stand-in providers of this
    module launch the recorded shapes whatever the widths, so a run of the tiny geometry is a run the section holds for."""
    import copy

    from model_samples import census
    from model_samples.qwen_dense import QwenDense

    data = copy.deepcopy(QwenDense.target().data)
    for section in data["configurations"]["plain"].values():
        section["geometry"] = {name: getattr(model.geometry, name) for name in section["geometry"]}
    return census.Target(data)


def _tiny_session(sequences, stack="reference"):
    from model_samples import Session, TextIO

    model = _tiny_model()
    text = TextIO.synthetic(model.geometry.vocab)
    session = Session(model, text, sequences=sequences, stack=stack, device="cpu", capacity=32)
    session.plan_table()
    session.load()
    return session, text


def _norm_symbol(form):
    """The symbol a stand-in instance of the norm entry states for a form: the form's text behind the kernel's name, as the
    entry writes it."""
    mode = "add" if form["ADD"] else ("store" if form.get("STORE_RESIDUAL") else "none")
    outputs = "_".join(part for part in ("bf16" if form["BF16_OUT"] else None, form["QUANT"]) if part)
    return f"cudnn_kernel_frost_add_rms_norm_quant_h64_{mode}_{outputs}_wait_release_eps358637bd_0"


def _stand_ins(
    monkeypatch,
    step_launches=None,
    declares=None,
    profiler="records",
    serves=None,
    gate_shape=GATE_SHAPE,
    gate_defect=False,
    norm_defect="",
    norm_serves=None,
    norm_shape=None,
):
    """The provider factories of a device that exists only as a log: every provider of the provider rows is a stand-in that
    runs the reference's arithmetic and writes the launches a real provider of its row would make into the log; a stand-in
    profiler (it replaces ``loop.trace_steps``) hands the log over as the launch records of a traced pass.

    ``step_launches`` replaces what the stand-in of the fused step launches (default: the entry's own symbol in the shape the
    entry states); ``declares`` replaces the census entries it declares; ``serves`` what it says served a call (default: its
    one entry instance, which states the form of the row).  ``profiler``: ``"records"``, or the reason it cannot run -- or a
    list of these, one per traced pass, the last one for every further pass.  The stand-in of the gate entry runs the
    kernel's fp32 chain, states and launches ``gate_shape`` and is compared with the gate's real twin; ``gate_defect``
    makes it write one wrong byte per call.

    The stand-in of the norm entry runs that kernel's fp32 chain in the form of the call's site -- one instance per form,
    the residual stream updated in place, the 4-bit block scales as the kernel forms them and in its 128 x 4 layout, the
    bf16 rows handed over unquantised where the form writes no codes -- states and launches one thread block per row (or
    ``norm_shape``) and is compared with the norm's real twin.  ``norm_serves(site)`` replaces what it says served a call; ``norm_defect`` makes
    it ``"extra"`` (launch one more kernel in front of its own), ``"missing"`` (not launch at the output head), ``"chain"`` (launch a
    chain of PyTorch kernels in place of its own), or write per call a wrong ``"code"``, block ``"scale"``, ``"bf16"``
    element or ``"residual"`` element."""
    from model_samples import loop
    from model_samples.census import Facts, Kernel, Launch
    from model_samples.providers import factories, torch_ref
    from model_samples.timing import Trace
    from model_samples.weights import BLOCK, ScaledTensor, TiledBlockScaledTensor, Unquantized, _e2m1, _e4m3_values, quantize_fp8, tile_scale_index

    log = []
    passes = [profiler] if isinstance(profiler, str) else list(profiler)

    def launch(name, shape=((1, 1, 1), (128, 1, 1), 0), kind="kernel"):
        at = float(len(log))
        log.append(Launch(name, kind, at, at + 0.5, bytes=4) if kind == "copy" else Launch(name, "kernel", at, at + 0.5, *shape, 32))

    def logging(base, *names, shape=((1, 1, 1), (128, 1, 1), 0)):
        class StandIn(base):
            def __call__(self, *operands):
                for name in names:
                    launch(name, shape, kind="copy" if name.startswith("Memcpy") else "kernel")
                return super().__call__(*operands)

        return StandIn

    class Step(torch_ref.GdnDecodeStep):
        """The stand-in of the fused step: one entry instance per form of a row, in the form the row names."""

        provider = "cuDNN FE"
        kernels = declares or (Kernel("gdn_decode_fused", "cuDNN FE", "exact", launches=1, shape="entry"),)

        def _symbol(self):
            return TARGET_SYMBOL if dict(self.row.form).get("PDL") else SYMBOL

        def launch_facts(self, rows):
            return Facts((self._symbol(),), dict(self.row.form), ((96, rows, 1), (256, 1, 1), 42112))

        def serving(self):  # what the stand-in's own entry instance states
            return serves or ("cudnn.GdnDecodeFused #1", tuple(self.row.form))

        def __call__(self, x, z, hidden, slot_ids, params, slots):
            for name, shape in step_launches or [(self._symbol(), ((96, x.shape[0], 1), (256, 1, 1), 42112))]:
                launch(name, shape)
            return super().__call__(x, z, hidden, slot_ids, params, slots)

    class Gate(torch_ref.GateQuantize):
        """The stand-in of the gate entry: the kernel's own chain in fp32 -- the gated value as a quotient, rounded once to
        bf16, then the static 8-bit quantisation -- one launch per call, and the real twin to be compared with."""

        provider = "cuDNN FE"
        kernels = (Kernel("gate_quant", "cuDNN FE", "exact", launches=1, shape="entry"),)
        _twin = None

        def launch_facts(self, rows):
            return Facts((GATE_SYMBOL,), dict(self.row.form), gate_shape)

        def serving(self):
            return "cudnn.AttentionGateQuantDecode #1", tuple(self.row.form)

        def twin(self, reference):
            if self._twin is None:
                self._twin = torch_ref.GateQuantizeQuotient(reference.row, self.model, self.facts, self.options)
            return self._twin

        def __call__(self, o, gate, quant):
            launch(GATE_SYMBOL, gate_shape)
            codes = quantize_fp8((o.float() / (1.0 + torch.exp(-gate.float()))).to(torch.bfloat16), quant.scale).codes
            if gate_defect:
                codes[0, 0] = 0x7E  # the largest code: of no bf16 value near a gated value of this model
            return ScaledTensor(codes, float(quant.scale))

    class Norm(torch_ref.AddNormQuantize):
        """The stand-in of the norm entry: the kernel's own chain in fp32 -- the sum, the normalised value rounded once to
        bf16, the codes of that bf16 value -- in the form of the call's site, one launch per call, and the real twin."""

        provider = "cuDNN FE"
        kernels = (Kernel("add_norm_quant", "cuDNN FE", "exact", launches=1, shape="entry"),)
        state_args = (1,)
        _twin = None
        _site = ""

        def _form(self, site):
            return dict(self.row.site_form(site))

        def launch_facts(self, rows, site=""):
            return Facts((_norm_symbol(self._form(site)),), self._form(site), norm_shape or ((rows, 1, 1), (640, 1, 1), 80))

        def serving(self):  # what the stand-in's own entry instance of the last call's form states
            if norm_serves is not None:
                return norm_serves(self._site)
            place = [site for site, _ in self.row.sites].index(self._site) + 1
            return f"cudnn.AddNormQuantDecode #{place}", tuple(self.row.site_form(self._site))

        def twin(self, reference):
            if self._twin is None:
                self._twin = torch_ref.AddNormQuantizeBracket(reference.row, self.model, self.facts, self.options)
                self._twin.forms = {site: self._form(site) for site, _ in self.row.sites}
            return self._twin

        def compared(self, out):
            q, normed, residual = out
            return (None if isinstance(q, Unquantized) else q, normed, residual)

        def __call__(self, branch, residual, weight, quant, site=""):
            form, rows = self._form(site), branch.shape[0]
            self._site = site
            if norm_defect == "chain":
                launch(ELEMENTWISE)
                launch(ELEMENTWISE)
            else:
                if norm_defect == "extra":
                    launch(ELEMENTWISE)  # in front of its own kernel: after the launch of the block before it
                if not (norm_defect == "missing" and site == "output_head"):
                    launch(_norm_symbol(form), norm_shape or ((rows, 1, 1), (640, 1, 1), 80))
            if residual is None:
                v, residual = branch.float(), branch.clone()
            else:
                v = residual.float() + branch.float()
                residual.copy_(v.to(torch.bfloat16))  # in place, as the kernel updates the stream
            y = ((v * torch.rsqrt(v.square().mean(-1, keepdim=True) + self.eps)) * (1.0 + weight.float())).to(torch.bfloat16)
            if norm_defect == "bf16":
                y[0, 0] = y[0, 0].float() * 1.5 + 1.0
            if norm_defect == "residual":
                residual[0, 0] = residual[0, 0].float() + 1.0
            if form["QUANT"] == "e4m3_static":
                q = quantize_fp8(y, quant.scale)
                if norm_defect == "code":
                    q.codes[0, 0] = 0x7E  # the largest code: of no bf16 value near a normed value of this model
            elif form["QUANT"] == "nvfp4":
                scales = torch_ref.nvfp4_block_scales(y.float().abs().unflatten(-1, (-1, BLOCK)).amax(-1), quant.scale)
                if norm_defect == "scale":
                    scales[0, 0] += 8  # one binade up: of no block maximum the bracket admits
                # the encode scale as the reference quantiser forms it (the reciprocal of block scale x scale, in fp32), the sign
                # kept on a zero: inside the bracket of the entry's two approximate reciprocals, and on an exact tie the
                # reference's choice -- so the token line of the tiny geometry (rows 64 wide, where one 4-bit code moves a
                # logit by several units) is held to the family's band as it is with the reference chain
                step = _e4m3_values(scales).repeat_interleave(BLOCK, dim=-1) * torch.tensor(float(quant.scale), dtype=torch.float32)
                encode = torch.where(step != 0, 1.0 / torch.where(step != 0, step, torch.ones_like(step)), torch.zeros_like(step))
                nibbles = _e2m1(y.float() * encode, signed_zero=True)
                if norm_defect == "code":
                    nibbles[0, 0] = nibbles[0, 0] ^ 0x8  # the other sign: outside every bracket of a non-zero value
                tiled = torch.zeros((128, y.shape[-1] // BLOCK), dtype=torch.uint8)
                tiled.view(-1)[tile_scale_index(rows, y.shape[-1] // BLOCK, "cpu")] = scales
                q = TiledBlockScaledTensor(nibbles[..., 0::2] | (nibbles[..., 1::2] << 4), tiled, float(quant.scale))
            else:  # the form ends at the bf16 rows: the consumer quantises
                q = None if quant is None else Unquantized(y, quant.fmt, float(quant.scale))
            return q, (y if form["BF16_OUT"] else None), residual

    stand_ins = {
        "fe.gdn_decode_fused": Step,
        "fe.attention_gate_quant_decode": Gate,
        "fe.add_norm_quant_decode": Norm,
        "cublas.bf16_dequantized": logging(torch_ref.Linear, ELEMENTWISE, NVJET, SPLIT_K, ELEMENTWISE),
        "cublas.bf16_dequantized+torch_ref.silu_mul_quantize": logging(torch_ref.GateUp, ELEMENTWISE, NVJET, ELEMENTWISE),
        "torch.index_select": logging(torch_ref.Embedding, GATHER, shape=GATHER_SHAPE),
        "torch.argmax": logging(torch_ref.TokenChoice, ARGMAX),
        "torch_ref.add_norm_quantize": logging(torch_ref.AddNormQuantize, ELEMENTWISE, COPY_IN, ELEMENTWISE),
        "torch_ref.attention_prepare": logging(torch_ref.AttentionPrepare, ARANGE, "Memcpy DtoD (Device -> Device)"),
        "torch_ref.attention": logging(torch_ref.Attention, GEMV, SOFTMAX, GEMM_FP64),
        "torch_ref.gate_quantize": logging(torch_ref.GateQuantize, ELEMENTWISE),
    }
    for key, stand_in in stand_ins.items():  # the stand-ins keep the census entries of the providers they stand for
        if not key.startswith("fe."):
            real = factories()[key]
            stand_in.kernels, stand_in.provider = real.kernels, real.provider

    def profiled(step, count, device=None, records=None):
        outcome = passes.pop(0) if len(passes) > 1 else passes[0]
        if outcome != "records":
            return None, outcome
        del log[:]
        for i in range(count):
            step(i)
            launch(HOST_READ, kind="copy")
        if records is not None:
            records.extend(log)
        return Trace(count, 1.0, 2.0, 3, 4, 5.0), ""

    monkeypatch.setattr(loop, "trace_steps", profiled)
    return dict(factories(), **stand_ins)


def _recorded_run(monkeypatch, sequences=2, stack="fe-cublas", check_kernels=True, **stand_in):
    """A decode of the tiny geometry on the CPU through the rows of ``stack``, on the stand-in providers and the stand-in
    profiler of :func:`_stand_ins` (``stand_in``: its options).  Returns ``(session, text, prompts)`` with the run recorded
    and the traced pass not yet made."""
    from model_samples import Session, TextIO, blocks, census

    from model_samples.qwen_dense import QwenDense

    build = _stand_ins(monkeypatch, **stand_in)
    monkeypatch.setattr(census, "installed_versions", lambda packages: dict(RECORDED))
    model = _tiny_model()
    monkeypatch.setattr(QwenDense, "_target", _census_of(model))
    text = TextIO.synthetic(model.geometry.vocab)
    session = Session(model, text, sequences=sequences, stack=stack, device="cpu", capacity=16, check_kernels=check_kernels)
    facts = blocks.DeviceFacts((10, 0), "stated device", 148, 0, 0)
    options = {"reference_precision": "fp64", "device": "cpu", "check_kernels": session.check_kernels}
    session.plan = blocks.plan(model.provider_table, model.regimes, facts, sequences, 1, stack, model=model, options=options, factories=build)
    session.load()
    prompts = [text.encode("ab" + str(s)) for s in range(sequences)]
    ids = session.prefill(prompts)
    session.read(ids)
    for _ in range(4):
        ids = session.step(ids)
        session.read(ids)
    return session, text, prompts


def _line(checks, start):
    return next(line for line in checks.lines if line.strip().startswith(start))


def test_the_kernel_check_of_a_run_holds_on_clean_stand_ins(monkeypatch):
    """The fifth check through a real session and the sample's checks: the traced pass keeps the call log and the launch
    records, and every launch is attributed.  A run that was not asked to check its kernels prints no such line."""
    from model_samples import census, tie_out

    session = _recorded_run(monkeypatch)[0]
    session.trace(4)  # check_kernels: the pass keeps its launch records
    traced = session.traced
    assert len(traced.steps) == 4 and len(traced.call_log) == 4 and all(len(calls) == 20 for calls in traced.call_log)
    assert traced.call_log[0][3] == census.Call("linear_attention.step", 0, 2, "cudnn.GdnDecodeFused #1", (("PDL", False), ("PDL_RELEASE", False)))
    checks = tie_out.run_checks(session, prompt_tokens=3, kernels=True)
    assert checks.ok and checks.kernels == census.PASS and checks.unobserved == "", checks.text()
    line = _line(checks, "kernels")
    assert (
        line.endswith("    PASS") and "4 traced steps: " in line and "2 on exact rows, in the wiring order" in line and "cublas " in line and "pytorch " in line
    ), line
    entry = _line(checks, "linear_attention.step: cudnn.GdnDecodeFused [PDL False, PDL_RELEASE False], 1 launch per step")
    assert "the entry states grid (96, 2, 1), block (256, 1, 1), shared memory 42,112 B and the kernel " + SYMBOL in entry
    assert _line(checks, "chosen by a library inside the fallback rows (printed, not gated): cublas: ")
    dumped = traced.dump()
    assert json.loads(json.dumps(dumped))["check"] == census.PASS and dumped["rows"] == 2 and len(dumped["steps"]) == 4
    assert dumped["fields"][:3] == ["name", "kernel", "kind"] and len(dumped["names"]) == len(set(dumped["names"])) < len(dumped["steps"][0])
    fused = next(record for record in dumped["steps"][0] if record[1] == "gdn_decode_fused")
    assert dumped["names"][fused[0]] == SYMBOL and fused[2:6] == ["kernel", (96, 2, 1), (256, 1, 1), 42112]
    site = dumped["sites"][3]
    assert site["block"] == "linear_attention.step" and site["instance"] == "cudnn.GdnDecodeFused #1" and site["form"] == {"PDL": False, "PDL_RELEASE": False}
    assert site["served"] == {"PDL": False, "PDL_RELEASE": False} and dumped["sites"][0]["served"] == {}, "from the call log: what the instance states"
    assert site["stated"] == {"symbols": [SYMBOL], "launch": [[96, 2, 1], [256, 1, 1], 42112]} and site["kernels"][0]["relation"] == "exact"
    session = _recorded_run(monkeypatch, check_kernels=False)[0]
    session.trace(4)  # the session's decision: no launch records
    quiet = tie_out.run_checks(session, prompt_tokens=3)
    assert session.traced is None and quiet.ok and quiet.kernels == "", "a pass that was not asked for launch records keeps none"
    assert not any(line.strip().startswith("kernels") for line in quiet.lines), "and the fifth check is printed only where it was asked for"
    session = _recorded_run(monkeypatch)[0]
    session.trace(4)
    own = tie_out.run_checks(session, prompt_tokens=3)  # the argument left out: the session's own decision
    assert own.ok and own.kernels == census.PASS and _line(own, "kernels").endswith("    PASS"), "a session that checks its kernels is checked unasked"
    session = _recorded_run(monkeypatch)[0]
    session.trace(4)
    off = tie_out.run_checks(session, prompt_tokens=3, kernels=False)
    assert off.ok and off.kernels == "", "a stack without a target may leave the check out"
    session = _recorded_run(monkeypatch)[0]
    session.trace(4, records=False)
    unasked = tie_out.run_checks(session, prompt_tokens=3, kernels=True)
    assert unasked.ok and unasked.kernels == census.NOT_OBSERVED and unasked.unobserved == "the traced pass kept no launch records"


def test_an_adapter_that_declares_one_kernel_and_launches_another_fails(monkeypatch):
    """The negative control.  A stand-in that declares the fused kernel and launches a chain of PyTorch kernels instead; one that
    launches the fused kernel in another shape than its entry states; and one that declares a reference chain and launches the
    fused kernel.  The other four checks hold in each run: only the kernel check sees it."""
    from model_samples import census, tie_out
    from model_samples.census import Kernel

    def run(**defect):
        session, _, _ = _recorded_run(monkeypatch, **defect)
        session.trace(4)
        checks = tie_out.run_checks(session, prompt_tokens=3, kernels=True)
        assert not checks.ok and checks.kernels == census.FAIL, checks.text()
        assert _line(checks, "kernels ").endswith("; 1 kind of difference between the launches and the plan    FAIL"), "a failing pass claims no attribution"
        for other in ("replay", "blocks", "tokens", "batch invariance"):
            assert _line(checks, other).endswith("PASS"), checks.text()
        return [line.strip() for line in checks.lines if line.strip().startswith("kernels: ")]

    chain = run(step_launches=[(ELEMENTWISE, ((1, 1, 1), (128, 1, 1), 0)), (ELEMENTWISE, ((1, 1, 1), (128, 1, 1), 0))])
    sentence = (
        "linear_attention.step (layer 0) did not launch gdn_decode_fused: the step has 0 launch(es) of gdn_decode_fused, the plan 1 = 1 call(s) x 1 per call"
    )
    assert chain == [f"kernels: step 0 (and 3 more of the 4 traced steps): {sentence}"], chain
    shape = run(step_launches=[(SYMBOL, ((96, 2, 1), (256, 1, 1), 74848))])
    assert shape == [
        "kernels: step 0 (and 3 more of the 4 traced steps): linear_attention.step (layer 0) launched gdn_decode_fused with grid (96, 2, 1), "
        "block (256, 1, 1), shared memory 74,848 B; cudnn.GdnDecodeFused states grid (96, 2, 1), block (256, 1, 1), shared memory 42,112 B"
    ], shape
    hidden = run(declares=(Kernel("pytorch", "PyTorch"), Kernel("cublas", "cuBLAS")))
    assert hidden == [
        f"kernels: step 0 (and 3 more of the 4 traced steps): launched {SYMBOL[:93] + '...'!r} after embedding: the kernel gdn_decode_fused, "
        "which no block of the plan declares"
    ], hidden


def test_a_call_served_by_another_entry_instance_fails_the_run(monkeypatch):
    """The call log through a real session.  The stand-in of the fused step launches the kernel its entry names, in the shape
    its entry states -- the launch records are clean -- but says that another entry instance served its calls, one that
    states another form than the row.  Only the call log shows it; the other four checks hold."""
    from model_samples import census, tie_out

    session = _recorded_run(monkeypatch, serves=("cudnn.GdnDecodeFused #2", (("PDL", True), ("PDL_RELEASE", True))))[0]
    session.trace(4)
    assert session.traced.call_log[0][3] == census.Call("linear_attention.step", 0, 2, "cudnn.GdnDecodeFused #2", (("PDL", True), ("PDL_RELEASE", True)))
    checks = tie_out.run_checks(session, prompt_tokens=3, kernels=True)
    assert not checks.ok and checks.kernels == census.FAIL, checks.text()
    assert [line.strip() for line in checks.lines if line.strip().startswith("kernels: ")] == [
        "kernels: form: linear_attention.step (layer 0): the adapter served the call with the form [PDL True, PDL_RELEASE True] "
        "(cudnn.GdnDecodeFused #2); the row says [PDL False, PDL_RELEASE False]"
    ], checks.text()
    for other in ("replay", "blocks", "tokens", "batch invariance"):
        assert _line(checks, other).endswith("PASS"), checks.text()
    assert session.traced.dump()["sites"][3]["served"] == {"PDL": True, "PDL_RELEASE": True}


def _sample_script(monkeypatch):
    """The sample script as a module, for its own functions; what loading it changes in the process leaves with the test."""
    import importlib.util
    import os

    monkeypatch.setattr(sys, "path", list(sys.path))  # the script puts samples/models on the path
    monkeypatch.setenv("KINETO_LOG_LEVEL", os.environ.get("KINETO_LOG_LEVEL", "6"))  # and sets a default for the profiler's log level
    spec = importlib.util.spec_from_file_location("model_sample_qwen_dense_00_decode_census", MODELS / "qwen_dense" / "00_decode.py")
    sample = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sample)
    return sample


def test_without_a_profiler_the_run_is_reported_and_never_a_pass(monkeypatch, capsys):
    """Where the profiler cannot run, the kernel check observed nothing: its line says so, the other checks stand, and the
    sample's last line is ``RAN <stack> stack (kernels not observed: <reason>)`` -- unless a check failed."""
    from model_samples import StopRule, census, tie_out

    reason = "the profiler failed (no tracing library)"
    session = _recorded_run(monkeypatch, profiler=reason)[0]
    session.trace(4)
    assert session.traced.steps is None and session.traced.call_log == [] and session.report().trace_missing == reason
    checks = tie_out.run_checks(session, prompt_tokens=3, kernels=True)
    assert checks.ok and checks.judged == 6 and checks.kernels == census.NOT_OBSERVED and checks.unobserved == reason, checks.text()
    assert _line(checks, "kernels ") == f"  kernels           not observed: {reason}    NOT OBSERVED"
    assert _line(checks, "linear_attention.step: cudnn.GdnDecodeFused").endswith(SYMBOL), "what the entry states is printed either way"

    sample = _sample_script(monkeypatch)
    assert sample._verdict("fe-cublas", True, 6, reason) == f"RAN fe-cublas stack (kernels not observed: {reason})"
    assert sample._verdict("fe-cublas", False, 6, reason) == "FAIL", "a failed check is a failure, observed kernels or not"
    assert sample._verdict("fe-cublas", True, 6) == sample._verdict("fe-cublas", True, 6, "") == "PASS"
    assert sample._verdict("reference", True, 0, reason) == "RAN reference stack (no provider checked)"

    # the sample's own loop function, on the CPU: no kernel activity can be recorded there at all
    session, text, prompts = _recorded_run(monkeypatch, profiler="needs a recorded step on a CUDA device")
    session.reset(fresh=True)
    session.new_report()
    capsys.readouterr()
    ok, judged, _ = sample._decode(session, text, StopRule(text.eos_ids, force_tokens=4), prompts, tie_out, check=True, checked_steps=None)
    out = capsys.readouterr().out
    assert ok and judged == 6 and session.traced.status == census.NOT_OBSERVED and session.traced.unobserved == "needs a recorded step on a CUDA device"
    assert "the tokens); the kernels of the traced steps against the plan" in out, "the line before the checks says that the kernels are checked too"
    assert "  kernels           not observed: needs a recorded step on a CUDA device    NOT OBSERVED" in out
    # whether the kernels are checked is the session's decision, read in one place: a session that does not check them prints no line
    session, text, prompts = _recorded_run(monkeypatch, check_kernels=False, profiler="needs a recorded step on a CUDA device")
    session.reset(fresh=True)
    session.new_report()
    capsys.readouterr()
    ok, judged, _ = sample._decode(session, text, StopRule(text.eos_ids, force_tokens=4), prompts, tie_out, check=True, checked_steps=None)
    assert ok and "  kernels  " not in capsys.readouterr().out, "without the flag the run prints no kernel line"


def test_the_traced_pass_hands_over_its_launch_records(monkeypatch, tmp_path):
    """``trace_steps`` with a list: the profiler's trace is exported and its kernel and copy records are appended, the step
    facts come from them; the records are handed over also where they cannot be cut into the steps that ran.  A trace that
    cannot be exported or read is a reason, not an error of the run -- and the reason says what failed, never where: the
    file is a temporary one, and no output of a sample names a local path."""
    import errno
    import os

    import torch.profiler

    from model_samples import timing

    def events(steps):
        out, ts = [], 1000.0
        for _ in range(steps):
            for name in (GATHER, SYMBOL):
                out.append(_kernel_event(name, ts))
                ts += 10.0
            out.append(_copy_event(HOST_READ, ts))
            ts += 10.0
        return out

    class Profile:
        steps, fail = 2, ""

        def __init__(self, activities=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def export_chrome_trace(self, path):
            Profile.path = path
            if self.fail == "no room":
                raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC), path)
            if self.fail == "names the file":
                raise RuntimeError(f"cannot save the trace to {path}\nsecond line")
            if self.fail == "cut short":
                Path(path).write_text(_trace_text(events(self.steps))[:900])
                return
            if self.fail == "writes nothing":
                return  # an exporter that logs what it cannot write and raises nothing
            if self.fail:
                raise RuntimeError(self.fail)
            Path(path).write_text(_trace_text(events(self.steps)))

        def events(self):
            raise AssertionError("with launch records asked for, the step facts come from the exported trace")

    monkeypatch.setattr(torch.profiler, "supported_activities", lambda: {torch.profiler.ProfilerActivity.CUDA})
    monkeypatch.setattr(torch.profiler, "profile", Profile)
    records, ran = [], []
    trace, reason = timing.trace_steps(ran.append, 2, records=records)
    assert reason == "" and ran == [0, 1] and (trace.steps, trace.kernels, trace.copies) == (2, 2, 1)
    assert [r.name for r in records] == [GATHER, SYMBOL, HOST_READ] * 2 and records[1].shape == SHAPE
    Profile.steps = 3
    records = []
    assert timing.trace_steps(ran.append, 2, records=records) == (None, "the records show 3 host reads for 2 steps") and len(records) == 9
    for fail, reason in (
        ("the trace could not be saved", "the profiler's trace could not be exported (the trace could not be saved)"),
        ("no room", "the profiler's trace could not be exported (No space left on device)"),
        ("names the file", "the profiler's trace could not be exported (cannot save the trace to trace.json)"),
        ("writes nothing", "the profiler exported no trace"),
        ("cut short", "the profiler's trace could not be read (the exported trace ends inside its traceEvents array)"),
    ):
        Profile.fail, records = fail, []
        outcome = timing.trace_steps(ran.append, 2, records=records)
        assert outcome == (None, reason) and records == [], fail
        folder = os.path.dirname(Profile.path)
        assert os.path.isabs(Profile.path) and not os.path.exists(folder), "the trace was a file of a temporary folder, which is gone"
        assert os.sep not in reason and os.path.basename(folder) not in reason, "the reason names no path"


def test_the_samples_flags_for_the_kernel_check(monkeypatch, capsys):
    sample = _sample_script(monkeypatch)
    assert not sample._arguments([]).check_kernels and sample._arguments([]).dump_census is None
    assert sample._arguments(["--check-kernels"]).check_kernels
    implied = sample._arguments(["--dump-census", "census.json"])
    assert implied.check_kernels and implied.dump_census == "census.json", "--dump-census implies the kernel check"
    for wrong in (["--stack", "reference", "--check-kernels"], ["--stack", "reference", "--dump-census", "c.json"]):
        with pytest.raises(SystemExit) as stop:
            sample._arguments(wrong)
        assert (
            stop.value.code == 2 and "--check-kernels and --dump-census are for a provider stack; --stack reference checks nothing" in capsys.readouterr().err
        )
    with pytest.raises(SystemExit):
        sample._arguments(["--help"])
    shown = " ".join(capsys.readouterr().out.split())
    for part in ("--check-kernels", "--dump-census FILE", "reports RAN (kernels not observed), never PASS", "implies --check-kernels"):
        assert part in shown, part
    with pytest.raises(ValueError, match=r"^--dump-census census.json: cannot be written there$"):
        sample._writable("--dump-census", "/nowhere/at/all/census.json")


def _sample_main(monkeypatch, capsys, arguments, total=0, free=80 * GIB):
    """``main()`` of the sample script on stated device facts (compute capability 10.0), up to its first load.  ``total``:
    the device's memory as a real device reports it; 0 states the facts only, and nothing is asked of an entry."""
    import importlib.util

    from model_samples import blocks

    model_samples = importlib.import_module("model_samples")  # the package as an object: its names are patched below

    def no_load(session):
        raise AssertionError("reached a load")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(blocks.DeviceFacts, "probe", classmethod(lambda cls, device: cls((10, 0), "stated device", 148, total, free)))
    monkeypatch.setattr(model_samples, "check_environment", lambda: model_samples.Environment("1.33.0", "1.33.0", True, "0123456789ab", "2.13.0", "4.8.0"))
    monkeypatch.setattr(model_samples.Session, "load", no_load)
    paths = list(sys.path)
    spec = importlib.util.spec_from_file_location("model_sample_qwen_dense_00_decode_census_main", MODELS / "qwen_dense" / "00_decode.py")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        return module.main(arguments), capsys.readouterr().out
    finally:
        sys.path[:] = paths


def test_the_kernel_table_is_printed_before_the_load_and_a_decline_is_a_skip(monkeypatch, capsys, tmp_path):
    """With ``--check-kernels`` the planned kernel table follows the served-by table, before anything is loaded; without the
    flag it is not printed.  What the check needs of the installed package is part of the plan: a package whose entry states
    no launch facts is a ``SKIP`` that names the fix, before the first load.  A census file that cannot be written is an
    ``ERROR`` before the plan."""
    import cudnn

    with pytest.raises(AssertionError, match="reached a load"):
        _sample_main(monkeypatch, capsys, ["--check-kernels"])
    out = capsys.readouterr().out
    served, kernels = out.index("served by (decided before the load"), out.index(
        "kernels, planned (decided before the load from model_samples/qwen_dense/configs.py"
    )
    assert served < kernels and "cudnn.GdnDecodeFused [PDL False, PDL_RELEASE False]" in out[kernels:] and "kernels, as the entries state them" not in out
    with pytest.raises(AssertionError, match="reached a load"):
        _sample_main(monkeypatch, capsys, [])
    assert "kernels, planned" not in capsys.readouterr().out
    code, out = _sample_main(monkeypatch, capsys, ["--dump-census", str(tmp_path / "missing" / "census.json")])
    assert code == 1 and out.splitlines()[-1] == "[00] ERROR --dump-census census.json: cannot be written there", out

    class Older:
        pass

    monkeypatch.setattr(cudnn, "GdnDecodeFused", Older, raising=False)
    code, out = _sample_main(monkeypatch, capsys, ["--check-kernels"])
    skip = "[00] SKIP the kernel check needs the launch facts of cudnn.GdnDecodeFused (kernel_symbols_for, form_for, launch_shape_for); "
    skip += "this nvidia-cudnn-frontend "
    assert code == 0 and out.splitlines()[-1].startswith(skip) and out.splitlines()[-1].endswith("or run without --check-kernels"), out


def _sample_run(monkeypatch, capsys, arguments, **stand_in):
    """``main()`` of the sample script to its LAST line, on the CPU: stated device facts (compute capability 10.0), the tiny
    geometry, and the stand-in providers and stand-in profiler of :func:`_stand_ins` (``stand_in``: its options).  Returns
    ``(exit code, the lines the run printed)``."""
    import importlib.util

    from model_samples import blocks, census
    from model_samples.qwen_dense import QwenDense

    model_samples = importlib.import_module("model_samples")  # the package as an object: its names are patched below
    build = _stand_ins(monkeypatch, **stand_in)
    monkeypatch.setattr(census, "installed_versions", lambda packages: dict(RECORDED))
    stated = blocks.DeviceFacts((10, 0), "stated device", 148, 180 * GIB, 80 * GIB)
    real_session = model_samples.Session

    def session(model, text, **options):  # the sample's session on the CPU, planned on the stated facts and the stand-ins
        made = real_session(model, text, **dict(options, device="cpu"))
        plan_options = {"reference_precision": made.reference_precision, "device": "cpu", "check_kernels": made.check_kernels}
        table, regimes = model.provider_table, model.regimes
        made.plan = blocks.plan(table, regimes, stated, made.sequences, 1, made.stack, made.overrides, model=model, options=plan_options, factories=build)
        return made

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (80 * GIB, 180 * GIB))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device=None: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device=None: 0)
    monkeypatch.setattr(blocks.DeviceFacts, "probe", classmethod(lambda cls, device: stated))
    monkeypatch.setattr(model_samples, "check_environment", lambda: model_samples.Environment("1.33.0", "1.33.0", True, "0123456789ab", "2.13.0", "4.8.0"))
    monkeypatch.setattr(model_samples, "Session", session)
    monkeypatch.setattr(QwenDense, "reduced", classmethod(lambda cls, layers=None: _tiny_model()))
    monkeypatch.setattr(QwenDense, "_target", _census_of(_tiny_model()))  # unless a caller has named another census
    paths = list(sys.path)
    spec = importlib.util.spec_from_file_location("model_sample_qwen_dense_00_decode_census_run", MODELS / "qwen_dense" / "00_decode.py")
    module = importlib.util.module_from_spec(spec)
    capsys.readouterr()
    try:
        spec.loader.exec_module(module)
        code = module.main(["--force-tokens", "4", *arguments])
    finally:
        sys.path[:] = paths
    return code, capsys.readouterr().out.splitlines()


def _kernel_lines(lines):
    return [line.strip() for line in lines if line.startswith("  kernels  ")]


def test_the_last_line_of_a_run_that_checks_its_kernels(monkeypatch, capsys, tmp_path):
    """The sample's ``main()`` with ``--check-kernels``, run to its last line.  Stand-in launch records that hold: ``[00] PASS``.
    A stand-in that launches another shape than its entry states: ``[00] FAIL``, exit code 1.  The profiler out: the kernels
    line reads ``NOT OBSERVED`` and the last line is ``[00] RAN fe-cublas stack (kernels not observed: <reason>)`` with exit
    code 0 -- never ``PASS``, also where only ONE of the run's sequence counts could not be observed."""
    code, lines = _sample_run(monkeypatch, capsys, ["--check-kernels", "--sequences", "2"])
    assert code == 0 and lines[-1] == "[00] PASS", lines[-3:]
    (line,) = _kernel_lines(lines)
    assert line.endswith("    PASS") and "2 on exact rows, in the wiring order" in line, line
    assert any(line.startswith("kernels, planned (decided before the load") for line in lines) and "[00] seeded weights" in lines[-2]

    code, lines = _sample_run(monkeypatch, capsys, ["--check-kernels", "--sequences", "2"], step_launches=[(SYMBOL, ((96, 2, 1), (256, 1, 1), 74848))])
    assert code == 1 and lines[-1] == "[00] FAIL", lines[-3:]
    assert _kernel_lines(lines)[0].endswith("; 1 kind of difference between the launches and the plan    FAIL")

    reason = "the profiler failed (no tracing library)"
    code, lines = _sample_run(monkeypatch, capsys, ["--check-kernels", "--sequences", "2"], profiler=reason)
    assert code == 0 and lines[-1] == f"[00] RAN fe-cublas stack (kernels not observed: {reason})", lines[-3:]
    assert _kernel_lines(lines) == [f"kernels           not observed: {reason}    NOT OBSERVED"] and "[00] PASS" not in lines
    others = [line for line in lines if line.startswith(("  replay  ", "  blocks  ", "  tokens  ", "  batch invariance  "))]
    assert len(others) == 4 and all(line.endswith("PASS") for line in others), "the other four checks held"

    # two sequence counts, the profiler out for the first one only: one pass that was not observed is a run that was not
    code, lines = _sample_run(monkeypatch, capsys, ["--check-kernels", "--sequences", "1,2"], profiler=[reason, "records"])
    first, second = _kernel_lines(lines)
    assert first.endswith("    NOT OBSERVED") and second.endswith("    PASS"), (first, second)
    assert code == 0 and lines[-1] == f"[00] RAN fe-cublas stack (kernels not observed: {reason})", lines[-3:]

    # without the flag the same run is a PASS and prints no kernel line: the profiler's records are not asked for
    code, lines = _sample_run(monkeypatch, capsys, ["--sequences", "2"], profiler=reason)
    assert code == 0 and lines[-1] == "[00] PASS" and _kernel_lines(lines) == [] and not any(line.startswith("kernels, planned") for line in lines)


def test_the_census_file_of_a_run(monkeypatch, capsys, tmp_path):
    """``--dump-census FILE`` through ``main()``: the launch records of the traced pass are written and the run says so by
    the file's own name; with nothing observed there is nothing to write, which is an ``ERROR`` that names the file and
    states the run's own verdict -- never an empty file under a ``RAN`` line."""
    target = tmp_path / "census.json"
    code, lines = _sample_run(monkeypatch, capsys, ["--dump-census", str(target), "--sequences", "2"])
    assert code == 0 and lines[-1] == "[00] PASS" and "[00] launch records of 1 traced pass(es) written to census.json" in lines, lines[-4:]
    written = json.loads(target.read_text())
    assert (written["sample"], written["stack"], written["schema"], written["traced_steps"]) == ("qwen_dense/00_decode", "fe-cublas", 1, 32)
    assert written["gpu"] == {"name": "stated device", "compute_capability": [10, 0], "sm_count": 148} and str(tmp_path) not in target.read_text()
    (run,) = written["runs"]
    assert (run["sequences"], run["prompt"], run["rows"], run["check"], run["not_observed"]) == (2, 0, 2, "PASS", "")
    assert _kernel_lines(lines)[0].startswith(f"kernels           {len(run['steps'])} traced steps: "), "every traced step of the kernels line is in the file"
    assert all(run["names"][step[-1][0]] == HOST_READ for step in run["steps"]) and len(run["steps"]) > 4, "each ends with its host read"
    step = run["sites"][3]
    assert step["instance"] == "cudnn.GdnDecodeFused #1" and step["form"] == step["served"] == {"PDL": False, "PDL_RELEASE": False}
    target.unlink()
    reason = "the profiler failed (no tracing library)"
    code, lines = _sample_run(monkeypatch, capsys, ["--dump-census", str(target), "--sequences", "2"], profiler=reason)
    ran = f"RAN fe-cublas stack (kernels not observed: {reason})"
    assert (
        code == 1 and lines[-1] == f"[00] ERROR --dump-census census.json: no launch records to write (kernels not observed: {reason}) (the run itself: {ran})"
    )
    assert not target.exists(), "nothing was observed: no file"


def test_a_dry_plan_with_its_kernel_table_imports_nothing_new():
    """In a fresh interpreter: after ``cudnn`` and PyTorch are loaded, importing the census, planning on stated device facts
    and printing the kernel table adds only ``model_samples`` and standard-library modules."""
    code = f"""
import sys
import cudnn, torch
cudnn.GdnDecodeFused
import importlib.metadata
before = set(sys.modules)
sys.path.insert(0, {str(MODELS)!r})
from model_samples import census
from model_samples.blocks import DeviceFacts, plan
from model_samples.qwen_dense import QwenDense
model = QwenDense.reduced(2)
planned = plan(model.provider_table, model.regimes, DeviceFacts((10, 0), "stated", 148, 1 << 37, 1 << 37), 1, model=model, options={{"check_kernels": True}})
table = census.table(planned)
sites = census.expected(planned, model, rows=1)
added = sorted({{name.split(".")[0] for name in set(sys.modules) - before}})
print("ADDED", [name for name in added if name not in sys.stdlib_module_names and name != "model_samples"])
print("TABLE", "gdn_decode_fused" in table, len(sites))
"""
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "ADDED []" in done.stdout and "TABLE True 20" in done.stdout, done.stdout + done.stderr


# ----------------------------------------------------------------------------------------------- the target kernel set
def _target_plan(layers=2, stack="native", cc=(10, 0), sms=148, sequences=1, overrides=None):
    """A plan of ``stack`` on stated device facts, with the family's census: ``(plan, model, target)``."""
    from model_samples.blocks import DeviceFacts, plan
    from model_samples.qwen_dense import QwenDense

    model = QwenDense() if layers == 64 else QwenDense.reduced(layers)
    facts = DeviceFacts(cc, "stated device", sms, 180 * GIB, 80 * GIB)
    return plan(model.provider_table, model.regimes, facts, sequences, 1, stack, overrides, model=model, options={"device": "cpu"}), model, QwenDense.target()


def test_the_census_file_of_the_target_kernel_set():
    """The checked-in census against itself and against the model: its three numbers are the sums over its own entry lists
    for 48 + 16 layers; its entries follow the wiring of the two-layer, the four-layer and the 64-layer model; every exact
    kernel of the ``native`` rows is a kernel of the file, in the file's form."""
    from model_samples import census
    from model_samples.qwen_dense import QwenDense, configs

    target = QwenDense.target()
    text = (MODELS / "model_samples" / "qwen_dense" / "kernel_census.json").read_text()
    assert target is QwenDense.target() and target.name == "kernel_census.json" and target.data["schema"] == 2, "read once"
    assert "registers" not in text, "registers per thread move with the compiler: recorded elsewhere, never checked in"
    assert set(target.data["layers"]) == set(QwenDense().geometry.layer_kinds)
    sections = target.data["configurations"]["plain"]
    assert list(sections) == ["cc10.0/sm148"], "one recorded class today"
    for key, section in sections.items():
        assert section["basis"] == "recorded" and section["claimed"] is False and section["recorded"]["layers"] == [48, 16], key
        for rows, parts in section["rows"].items():
            launches, code, port, stand_ins = target.numbers(parts, QwenDense().geometry.layer_kinds)
            assert {"launches": launches, "code": code, "port": port, "stand_in": stand_ins} == section["numbers"], f"{key}, {rows} rows: three numbers"
            assert (launches, code, port, stand_ins) == (484, 210, 273, 1) and code + port + stand_ins == launches
            assert target.numbers(parts, QwenDense.reduced(2).geometry.layer_kinds) == (20, 9, 10, 1)
            assert target.numbers(parts, QwenDense.reduced(4).geometry.layer_kinds) == (34, 15, 18, 1)
    for layers in (2, 4, 64):  # the census's entries of a step, beside the wiring: every call has its place, and one target entry has no block yet
        plan, model, _ = _target_plan(layers)
        claim = target.claim(plan, model, installed=RECORDED)
        assert len(claim.identities) == len(model.wiring()) and claim.absent == [("output_head.quantize", "fi_fp4_quantize", "")], layers
    plan, model, _ = _target_plan()
    exact = {name: [kernel.id for kernel in block.kernels if kernel.relation == "exact"] for name, block in plan.items()}
    served = {name: ids[0] for name, ids in exact.items() if ids}
    assert served == {
        "embedding": "embedding_gather",
        "add_norm_quantize": "add_norm_quant",
        "linear_attention.step": "gdn_decode_fused",
        "attention.gate_quantize": "gate_quant",
    }
    for block, kernel_id in served.items():
        meta = target.data["kernels"][kernel_id]
        same = "a change of a form the kernel census names updates kernel_census.json in the same commit"
        assert dict(plan.rows[block].form) == meta.get(
            "form", {}
        ), f"{block}: the form of its native row in configs.py and of {kernel_id} in the census differ; {same}"
        assert meta["entry"] == plan[block].entry, block
    # a kernel the target calls in several forms: the census names the form of every call, the row the form of every call site,
    # and for every call of the 64-layer model the two are the same form
    full_plan, full, _ = _target_plan(64)
    parts = sections["cc10.0/sm148"]["rows"]["1"]
    calls = target.sequence(parts, full.geometry.layer_kinds)
    by_call = [(site, facts) for (block, _, facts), site in zip(calls, full.sites()) if block == "add_norm_quantize"]
    assert len(by_call) == 129 and all(site for site, _ in by_call)
    for site, facts in by_call:
        wanted = {**target.data["kernels"]["add_norm_quant"]["form"], **facts["form"]}
        assert dict(full_plan.rows["add_norm_quantize"].site_form(site)) == wanted, f"add_norm_quantize at {site}: configs.py and the census differ; {same}"
    assert parts[census.FIRST_LAYER] == {"add_norm_quantize": {"form": {"ADD": False, "STORE_RESIDUAL": True}}}, "the first norm of a step starts the stream"
    assert by_call[0][1]["form"] == {"ADD": False, "STORE_RESIDUAL": True, "BF16_OUT": True, "QUANT": "e4m3_static"} and by_call[1][1]["form"]["ADD"] is True
    assert (
        by_call[0][1]["launch"] == by_call[1][1]["launch"] == [[1, 1, 1], [640, 1, 1], 80]
    ), "one launch shape for every form: the form is not in a launch record"
    assert configs.TARGET_STACKS == QwenDense.target_stacks == ("native", "fe-entries")
    # which kernels a stack of this repository calls an entry for is said per kernel, and held here: the three numbers count
    # the complete stack, and a reader of the file must be able to tell what exists today
    kernels = target.data["kernels"]
    assert {kernel_id for kernel_id, meta in kernels.items() if meta["entry"] is not None} == set(served.values())
    assert all("entry" in meta for meta in kernels.values()), "null where no stack calls an entry for the kernel yet"
    for kernel_id, meta in kernels.items():
        ported = meta.get("kind") == "port" and meta["entry"] is not None
        assert ("tie_out" in meta) == ("differences" in meta) == ported, f"{kernel_id}: a port a stack calls says where its byte equality is recorded"
        if ported:
            assert isinstance(meta["tie_out"], str) and "is not a test of this repository" in meta["tie_out"] and meta["differences"], kernel_id
            holds = {
                "gate_quant": ("docs/operations/attention/gate_quant_decode.md", "test/python/attention_gate/frost"),
                "add_norm_quant": ("docs/operations/normalizations/add_norm_quant_decode.md", "test/python/norm/frost"),
            }[kernel_id]
            for path in holds:
                assert path in meta["tie_out"] and (MODELS.parents[1] / path).exists(), f"{kernel_id}: what the repository holds for the entry"
    assert "on compute capability 10.0 -- no differing byte" in kernels["add_norm_quant"]["tie_out"], "the class the comparison was made on"
    assert len(kernels["add_norm_quant"]["differences"]) == 5, "what differs from the target kernel besides its results, as the entry lists it"
    assert [kernel_id for kernel_id, meta in kernels.items() if meta.get("kind") == "port" and meta["entry"] is None] == [
        "fp8_projection",
        "swiglu_fp4",
    ], "the ports no stack calls yet: 128 of the 273 launches"
    readme = " ".join((MODELS / "qwen_dense" / "README.md").read_text().split())
    for sentence in (
        "These are the numbers of the complete stack, not of today's. Of the 273, this stack launches 145 today: the norm entry's 129 and the gate entry's 16.",
        'And "byte-equal" is the entry\'s own record, not something this sample measures',
        "for the gate entry it was made on compute capability 10.3, and none is recorded for 10.0",
        "for the norm entry it was made on compute capability 10.0",
    ):
        assert sentence in readme, sentence
    about = " ".join(target.data["about"])
    assert "A port's byte equality is the entry's own record, not a measurement of the sample that reads this file" in about
    assert "They count what a complete stack launches" in about
    # every target kernel of the section but a tuner's pick has its launch shape: the section can be claimed without further data
    for key, section in sections.items():
        for rows, parts in section["rows"].items():
            for block, kernel_id, facts in target.sequence(parts, QwenDense().geometry.layer_kinds):
                free = kernels[kernel_id]["relation"] == "stand-in" or "selection" in kernels[kernel_id]
                assert ("launch" in facts) != free, f"{key}, {rows} rows: {block} ({kernel_id})"
                assert set(facts) <= set(census.ENTRY_FACTS), block


def test_the_identity_of_every_block_call_and_the_three_numbers():
    """How every block call of the ``native`` plan stands to the target, decided against the census from the row's kernel and
    form: 194 of the 484 launches of a step of the 64-layer model are target kernels, 49 the target's own compiled kernels
    and 145 cuDNN Frontend entries with byte-equal results (the norm entry's 129, in five forms, and the gate entry's 16);
    every other block is ``not yet``.  Always three numbers."""
    from model_samples import census
    from model_samples.qwen_dense import QwenDense, configs

    plan, model, target = _target_plan(64)
    claim = target.claim(plan, model, installed=RECORDED)
    numbers = claim.numbers()
    assert (numbers["launches"], numbers["code"], numbers["port"], numbers["stand_ins"]) == (484, 210, 273, 1)
    assert (numbers["target_code"], numbers["target_port"]) == (49, 145) and numbers["not_the_target"] == {}
    assert numbers["not_yet"] == {
        "projection.fp8.in": 64,
        "projection.fp8.out": 64,
        "mlp.gate_up": 64,
        "mlp.down": 64,
        "attention.prepare": 16,
        "attention.core": 16,
        "output_head": 1,
    }
    assert sum(numbers["not_yet"].values()) + 49 + 145 + 1 == 484 and numbers["stand_ins_not_yet"] == ["output_head.quantize"]
    whole, reached = claim.coverage()
    assert whole == (
        WHOLE + "484 launches per step = 210 the target's own compiled kernels + 273 cuDNN Frontend entries with byte-equal results + 1 declared stand-in"
    ), "the three numbers of the complete stack, said to be that: the 273 are not all launched by a stack today"
    assert reached == (
        "kernel identity, as planned: 194 of 484 launches of a step are target kernels (49 the target's own compiled kernels + 145 cuDNN Frontend entries "
        "with byte-equal results); not yet: projection.fp8.in 64, projection.fp8.out 64, mlp.gate_up 64, mlp.down 64, "
        "attention.prepare 16, attention.core 16, output_head 1; 1 declared stand-in (not yet: output_head.quantize)"
    )
    readme = (MODELS / "qwen_dense" / "README.md").read_text()
    assert whole in readme and reached in readme, "the two lines of the family README are the lines a run prints"
    assert configs.readme_table(QwenDense().calls_text(), "native") in readme, "regenerate the native table of the README from configs.readme_table()"
    assert configs.readme_table(QwenDense().calls_text(), "fe-entries") == configs.readme_table(QwenDense().calls_text(), "native"), "the same rows today"
    shapes = {"embedding": GATHER_SHAPE, "add_norm_quantize": NORM_SHAPE, "linear_attention.step": SHAPE, "attention.gate_quantize": GATE_SHAPE}
    assert claim.recorded == {
        index: shapes[block] for index, (block, _) in enumerate(model.wiring()) if block in shapes
    }, "the recorded launch shape of every target launch: 194 of them, as tuples"
    assert len(claim.recorded) == 194
    assert configs.readme_sites(QwenDense(), "native") in readme, "regenerate the call-site table of the README from configs.readme_sites()"
    from collections import Counter

    forms = Counter(site for site in model.sites() if site)
    assert forms == {"linear_attention.first": 1, "linear_attention": 47, "full_attention": 16, "mlp": 64, "output_head": 1}, "five forms: 1 / 47 / 16 / 64 / 1"

    plan, model, _ = _target_plan(2)
    claim = target.claim(plan, model, installed=RECORDED)
    assert claim.coverage()[1].startswith(
        "kernel identity, as planned: 8 of 20 launches of a step are target kernels (2 the target's own compiled kernels + 6 cuDNN Frontend entries "
        "with byte-equal results); not yet: projection.fp8.in 2, "
    )
    kinds = {block: (identity.relation, identity.kind) for block, identity in zip(claim.blocks, claim.identities)}
    for block in (
        "add_norm_quantize",
        "linear_attention.step",
        "attention.gate_quantize",
    ):  # the class their target forms ran on: 2 or more sequences gate them
        assert plan.rows[block].batch_invariant == ("cc10.0",), f"{block}: batch invariance of its native row is a gate on compute capability 10.0"
    assert kinds["embedding"] == kinds["linear_attention.step"] == ("target", "code") and kinds["attention.gate_quantize"] == ("target", "port")
    assert kinds["token_choice"] == ("host", "") and kinds["mlp.down"] == ("not yet", "code") and kinds["add_norm_quantize"] == ("target", "port")
    table = census.table(plan, claim).splitlines()
    assert table[1].split()[-1] == "identity" and table[0].startswith("kernels, planned (decided before the load from the family's configs.py; stack native,")
    row = {line.split()[0]: line for line in table[2:14]}
    assert row["embedding"].endswith("  name  target") or row["embedding"].rstrip().endswith("target")
    assert (
        row["linear_attention.step"].split("cudnn.GdnDecodeFused")[1].split()
        == "[PDL True, PDL_RELEASE True] gdn_decode_fused 1 name and launch shape, as the entry states them target".split()
    )
    gate = "[THREADS 256, PDL_RELEASE True] gate_quant 1 name and launch shape, as the entry states them target"
    assert row["attention.gate_quantize"].split("cudnn.AttentionGateQuantDecode")[1].split() == gate.split()
    assert row["mlp.down"].rstrip().endswith("by family                                        not yet (bf16 copy of 4-bit weights)")
    assert row["token_choice"].rstrip().endswith("host path, not compared                          --")
    norm = "[PDL True, PDL_RELEASE True] + one of 6 forms, by call site add_norm_quant 5 name and launch shape, as the entry states them target"
    assert row["add_norm_quantize"].split("cudnn.AddNormQuantDecode")[1].split() == norm.split()
    assert table[14:20] == [f"  add_norm_quantize at {site}:{census.form_text(fields)}" for site, fields in configs.NORM_FORMS], "the form of every call site"
    assert table[16] == "  add_norm_quantize at linear_attention: [ADD True, STORE_RESIDUAL False, BF16_OUT True, QUANT e4m3_static]"
    assert table[-3].startswith("  per step: 8 launches on 4 exact rows") and table[-2:] == [f"  {line}" for line in claim.coverage()]
    assert "identity" not in census.table(_plan(options={"device": "cpu"})[0]).splitlines()[1], "a stack without a target has no such column"

    # fe-entries: the same rows today, so the same identities
    other, model, _ = _target_plan(2, "fe-entries")
    assert target.claim(other, model, installed=RECORDED).identities == claim.identities and {row.key for row in other.rows.values()} == {
        row.key for row in plan.rows.values()
    }
    # a row selected with --provider: the fused step in its fe-cublas form is the same kernel in another form
    selected, model, _ = _target_plan(2, overrides={"linear_attention.step": "fe.gdn_decode_fused"})
    assert selected.overridden == ("linear_attention.step",) and selected.rows["linear_attention.step"].form == (("PDL", False), ("PDL_RELEASE", False))
    differs = target.claim(selected, model, installed=RECORDED)
    step = differs.identities[3]
    assert step.text() == "not the target (the row's form is [PDL False, PDL_RELEASE False], the target's [PDL True, PDL_RELEASE True])" and step.launch is None
    assert differs.numbers()["not_the_target"] == {"linear_attention.step": 1} and "; not the target: linear_attention.step 1; " in differs.coverage()[1]
    assert 3 not in differs.recorded and len(differs.recorded) == 7, "no recorded launch shape is held against a call that is not the target's"
    # compute capability 12.x: the gate and the norm entry do not serve there, the blocks stay on their reference chains, and the census has no section
    plan12, model, _ = _target_plan(2, cc=(12, 0), sms=170)
    assert plan12.rows["attention.gate_quantize"].key == "torch_ref.gate_quantize" and plan12.rows["linear_attention.step"].form == (
        ("PDL", True),
        ("PDL_RELEASE", True),
    )
    assert plan12.rows["add_norm_quantize"].key == "torch_ref.add_norm_quantize" and plan12.rows["add_norm_quantize"].site_form("mlp") == ()
    claim12 = target.claim(plan12, model, installed=RECORDED)
    assert not claim12.known and claim12.coverage() == [
        "kernel identity: not stated for this run (no recorded section for cc 12.0 / 170 SMs: recorded is cc 10.0 / 148 SMs)"
    ]
    assert "identity" not in census.table(plan12, claim12).splitlines()[1] and census.table(plan12, claim12).splitlines()[-1] == "  " + claim12.coverage()[0]
    # ... and with a section for that class the gate would read "not yet": a kernel of one class is never taken for another
    import copy

    data = copy.deepcopy(target.data)
    data["configurations"]["plain"]["cc12.0/sm170"] = copy.deepcopy(data["configurations"]["plain"]["cc10.0/sm148"])
    there = census.Target(data).claim(plan12, model, installed=RECORDED)
    gate = there.identities[there.blocks.index("attention.gate_quantize")]
    assert there.known and gate.text() == "not yet (reference chain)" and (there.numbers()["target_code"], there.numbers()["target_port"]) == (2, 0)
    assert there.numbers()["not_yet"]["add_norm_quantize"] == 5


def test_every_reason_a_class_is_not_claimed():
    """The census line says ``NOT CLAIMED`` with every reason that applies, in a fixed order: a row selected with
    ``--provider``, fallback rows, a kernel of the target the plan has no block for, no recorded section for the GPU class
    and SM count (both are named) or for the row count, versions other than the recorded ones."""
    from model_samples import census

    plan, model, target = _target_plan()
    claim = target.claim(plan, model, installed=RECORDED)
    fallback, absent = "fallback rows present", "the plan has no block for output_head.quantize"
    assert claim.reasons == [fallback, absent] and claim.name == CENSUS and (claim.basis, claim.claimed, claim.rows) == ("recorded", False, 1)
    assert claim.line(census.PASS) == f"{CENSUS}: NOT CLAIMED ({fallback}; {absent}); {COMPARED_8}: each in the recorded one"
    assert claim.compared() == (8, COMPARED_8, 0), "what item 7 compares: a count of target launches is three numbers here too"
    assert claim.line(census.NOT_OBSERVED) == f"{CENSUS}: NOT CLAIMED ({fallback}; {absent}); the launch shapes of its target launches were not observed"
    assert claim.line(census.FAIL) == f"{CENSUS}: NOT CLAIMED ({fallback}; {absent})", "a failed check says nothing about the recorded shapes"
    selected, model, _ = _target_plan(overrides={"linear_attention.step": "fe.gdn_decode_fused", "mlp.down": "torch_ref.linear"})
    assert target.claim(selected, model, installed=RECORDED).reasons == [
        "not the target set: linear_attention.step selected by --provider",
        "not the target set: mlp.down selected by --provider",
        fallback,
        absent,
    ]
    other_sms, model, _ = _target_plan(sms=152)
    claim = target.claim(other_sms, model, installed=RECORDED)
    assert claim.reasons == [fallback, "no recorded section for cc 10.0 / 152 SMs: recorded is cc 10.0 / 148 SMs"], "both SM counts are named"
    assert (
        claim.name == "target census plain"
        and not claim.known
        and claim.recorded == {}
        and claim.line(census.PASS)
        == (f"target census plain: NOT CLAIMED ({fallback}; no recorded section for cc 10.0 / 152 SMs: recorded is cc 10.0 / 148 SMs)")
    )
    four, model, _ = _target_plan(sequences=4)
    claim = target.claim(four, model, installed=RECORDED)
    assert claim.reasons == [fallback, "no recorded section for 4 rows on cc 10.0 / 148 SMs: recorded is 1 row"] and claim.name == CENSUS and not claim.known
    assert claim.coverage() == ["kernel identity: not stated for this run (no recorded section for 4 rows on cc 10.0 / 148 SMs: recorded is 1 row)"]
    older = {"torch": "2.13.0+cu130", "nvidia-cutlass-dsl": "4.8.1", "triton": "3.7.1"}
    versions = "versions other than the recorded ones: PyTorch 2.13.0 (recorded with 2.14.0), CuTe DSL 4.8.1 (recorded with 4.8.0)"
    assert target.claim(plan, model, installed=older).reasons == [fallback, absent, versions], "only the packages a provider of this area imports"
    assert target.claim(plan, model, installed={"torch": None, "nvidia-cutlass-dsl": "4.8.0"}).reasons == [fallback, absent], "not installed: not compared"
    found = census.installed_versions(("torch", "no-such-distribution-of-this-test"))
    assert found["no-such-distribution-of-this-test"] is None and found["torch"].split("+")[0] == torch.__version__.split("+")[0]
    assert set(target.data["validated"]["recorded"]) >= set(census.PACKAGES), "the census records a version of every package that is compared"
    dumped = target.claim(plan, model, installed=RECORDED).dump()
    assert dumped["claimed"] is False and dumped["not_claimed"] == [fallback, absent] and dumped["numbers"]["launches"] == 20
    assert (dumped["census"], dumped["configuration"], dumped["device"], dumped["basis"], dumped["recorded"]) == (
        "kernel_census.json",
        "plain",
        "cc 10.0 / 148 SMs",
        "recorded",
        "2026-11-12",
    )


def _small_census(claimed=True, basis="recorded", form=None):
    """A census of a step with three block calls -- a gather, one fused step, the host path -- as a constructed file."""
    step = {"origin": "cuDNN Frontend", "relation": "target", "kind": "code", "form": {"PDL": True, "PDL_RELEASE": True} if form is None else form}
    return {
        "schema": 2,
        "validated": {"recorded": {"torch": "2.14.0", "nvidia-cutlass-dsl": "4.8.0"}},
        "kernels": {"embedding_gather": {"origin": "PyTorch", "relation": "target", "kind": "code"}, "gdn_decode_fused": step},
        "layers": {"linear_attention": "layer"},
        "configurations": {
            "plain": {
                "cc10.0/sm148": {
                    "claimed": claimed,
                    "basis": basis,
                    "recorded": {"date": "2026-11-12"},
                    "geometry": {"hidden": 5120},
                    "rows": {
                        "1": {
                            "prologue": [["embedding", "embedding_gather", {"launch": [[40, 1, 1], [128, 1, 1], 0]}]],
                            "layer": [["linear_attention.step", "gdn_decode_fused", {"launch": [[96, 1, 1], [256, 1, 1], 42112]}]],
                            "epilogue": [],
                            "after_step": [],
                        }
                    },
                }
            }
        },
    }


def _small_plan(step_row=None, overridden=()):
    """The plan and the model of :func:`_small_census`: stand-in objects with what a claim reads of them."""
    from types import SimpleNamespace

    from model_samples.blocks import DeviceFacts, Plan, ProviderRow
    from model_samples.census import Kernel

    any_row = (("any",), (1, None), (1, None))
    rows = {
        "embedding": ProviderRow(*any_row, "torch.index_select", "native", serves="any"),
        "linear_attention.step": step_row or ProviderRow(*any_row, "fe.gdn_decode_fused", "native", serves="any", form=(("PDL", True), ("PDL_RELEASE", True))),
        "token_choice": ProviderRow(*any_row, "torch.argmax", "native", serves="any"),
    }
    step = (
        (Kernel("pytorch", "PyTorch"),)
        if rows["linear_attention.step"].fallback
        else (Kernel("gdn_decode_fused", "cuDNN FE", "exact", launches=1, shape="entry"),)
    )
    blocks = {
        "embedding": SimpleNamespace(kernels=(Kernel("embedding_gather", "PyTorch", "exact", launches=1),)),
        "linear_attention.step": SimpleNamespace(kernels=step),
        "token_choice": SimpleNamespace(kernels=(Kernel("pytorch", "PyTorch", "host"),)),
    }
    plan = Plan(blocks, rows, {}, DeviceFacts((10, 0), "stated device", 148, 0, 0), 1, 1, "native", None, "a test", overridden)
    model = SimpleNamespace(
        geometry=SimpleNamespace(layer_kinds=("linear_attention",), hidden=5120, vocab=272),
        wiring=lambda: (("embedding", -1), ("linear_attention.step", 0), ("token_choice", -1)),
    )
    return plan, model


def test_a_complete_census_is_equal_and_a_claimed_section_needs_a_recorded_basis_and_no_fallback_row():
    """``EQUAL`` is printed only for a section that says ``claimed``, on a plan whose every block call is on its target kernel,
    with the recorded versions -- and then only when the kernel check held.  The implication the other way is held too: a
    file that says ``claimed`` on a basis that is not ``recorded`` is refused when it is read, and one that says
    ``claimed`` for a class on which a row of the stack is still a fallback is caught here and never printed as ``EQUAL``."""
    import copy
    from dataclasses import replace

    from model_samples import census
    from model_samples.qwen_dense import QwenDense

    plan, model = _small_plan()
    complete = census.Target(_small_census()).claim(plan, model, installed=RECORDED)
    assert complete.reasons == [] and complete.claimed and complete.line(census.PASS) == f"{CENSUS}: EQUAL ({COMPARED_2}, 0 by membership alone)"
    assert (
        complete.line(census.FAIL) == f"{CENSUS}: NOT EQUAL (the kernel check did not hold)" and complete.line(census.NOT_OBSERVED) == f"{CENSUS}: NOT OBSERVED"
    )
    assert complete.coverage() == [
        "kernel identity, as planned: 2 launches per step = 2 the target's own compiled kernels + 0 cuDNN Frontend entries with byte-equal results "
        "+ 0 declared stand-ins"
    ], "three numbers, also where two of them are zero"
    assert complete.dump()["claimed"] is True and complete.recorded == {0: GATHER_SHAPE, 1: SHAPE}
    unclaimed = census.Target(_small_census(claimed=False)).claim(plan, model, installed=RECORDED)
    assert unclaimed.reasons == ["the census does not claim this class yet"] and unclaimed.line(census.PASS).startswith(
        f"{CENSUS}: NOT CLAIMED (the census does"
    )
    planned = census.Target(_small_census(claimed=False, basis="planned")).claim(plan, model, installed=RECORDED)
    assert planned.name == "target census plain / cc 10.0 / 148 SMs (planned)" and planned.reasons == ["the census does not claim this class yet"]
    assert census.Target(_small_census()).claim(plan, model, installed={"torch": "2.15.0"}).line(census.PASS) == (
        f"{CENSUS}: NOT CLAIMED (versions other than the recorded ones: PyTorch 2.15.0 (recorded with 2.14.0)); {COMPARED_2}: each in the recorded one"
    )
    # another form than the census's, without --provider: the table is not the target, and the line says which block
    row = plan.rows["linear_attention.step"]
    other_form, _ = _small_plan(replace(row, form=(("PDL", True),)))
    differs = census.Target(_small_census()).claim(other_form, model, installed=RECORDED)
    reason = "not the target set: linear_attention.step (the row's form is [PDL True], the target's [PDL True, PDL_RELEASE True])"
    assert differs.reasons == [reason] and differs.recorded == {0: GATHER_SHAPE}
    with_provider, _ = _small_plan(replace(row, form=(("PDL", True),)), overridden=("linear_attention.step",))
    assert census.Target(_small_census()).claim(with_provider, model, installed=RECORDED).reasons == [
        "not the target set: linear_attention.step selected by --provider"
    ]
    other_kernel = census.Target(_small_census())
    other_kernel.data["configurations"]["plain"]["cc10.0/sm148"]["rows"]["1"]["layer"][0][1] = "embedding_gather"
    assert other_kernel.claim(plan, model, installed=RECORDED).reasons == [
        "not the target set: linear_attention.step (the row launches gdn_decode_fused, the target embedding_gather)"
    ]

    # claimed implies a recorded basis: refused when the file is read
    with pytest.raises(ValueError, match=r"^kernel_census.json: plain / cc10.0/sm148: claimed on the basis planned; only a recorded section can be claimed$"):
        census.Target(_small_census(basis="planned"))
    # claimed implies no fallback row: a plan with a fallback row on a section that says claimed is caught, and is never EQUAL
    fallback_row, _ = _small_plan(replace(row, key="torch_ref.gdn_decode_step", form=(), fallback="reference chain"))
    caught = census.Target(_small_census()).claim(fallback_row, model, installed=RECORDED)
    assert (
        caught.claimed and caught.reasons == ["fallback rows present"] and caught.line(census.PASS).startswith(f"{CENSUS}: NOT CLAIMED (fallback rows present)")
    )
    assert caught.identities[1].text() == "not yet (reference chain)" and caught.dump()["claimed"] is False

    def claimed_wrongly(target, stack):
        """The sections of ``target`` that say claimed although the stack's plan on their class cannot be: the rule of the file."""
        wrong = []
        for configuration, sections in target.data["configurations"].items():
            for key, section in sections.items():
                cc, sms = key[2:].split("/sm")
                for rows in section["rows"]:
                    facts = (tuple(int(n) for n in cc.split(".")), int(sms))
                    plan, model, _ = _target_plan(64, stack, facts[0], facts[1], int(rows))
                    reasons = [r for r in target.claim(plan, model, configuration, installed=RECORDED).reasons if not r.startswith("the census does not claim")]
                    if section["claimed"] and (section["basis"] != "recorded" or reasons):
                        wrong.append((configuration, key, rows, reasons))
        return wrong

    real = QwenDense.target()
    for stack in QwenDense.target_stacks:
        assert (
            claimed_wrongly(real, stack) == []
        ), "a section of the checked-in census says claimed while a row of the stack is a fallback, or on a planned basis"
    tampered = census.Target(copy.deepcopy(real.data))
    tampered.data["configurations"]["plain"]["cc10.0/sm148"]["claimed"] = True
    assert claimed_wrongly(tampered, "native") == [
        ("plain", "cc10.0/sm148", "1", ["fallback rows present", "the plan has no block for output_head.quantize"])
    ], "the rule sees a census that claims too early"

    # a file that is not such a census is refused by its own name
    for change, sentence in (
        (lambda d: d.update(schema=1), "not a kernel census of schema 2"),
        (lambda d: d.pop("layers"), "a kernel census holds kernels, layers and configurations"),
        (lambda d: d["kernels"]["gdn_decode_fused"].update(kind="copy"), "kernel gdn_decode_fused: a kernel is a target of kind code or port, or a stand-in"),
        (lambda d: d["configurations"]["plain"].update({"cc10/148": {}}), "plain / cc10/148: a section is keyed by ccMAJOR.MINOR/smCOUNT"),
        (lambda d: d["configurations"]["plain"]["cc10.0/sm148"].pop("basis"), "plain / cc10.0/sm148: a section holds claimed (true or false), basis"),
        (lambda d: d["configurations"]["plain"]["cc10.0/sm148"]["rows"]["1"].pop("epilogue"), "plain / cc10.0/sm148, 1 rows: no entry list epilogue"),
        (
            lambda d: d["configurations"]["plain"]["cc10.0/sm148"]["rows"]["1"]["layer"].append(["mlp.down", "no_such_kernel", {}]),
            "is not [block, kernel of the",
        ),
    ):
        data = _small_census()
        change(data)
        with pytest.raises(ValueError) as refused:
            census.Target(data)
        assert str(refused.value).startswith("kernel_census.json: ") and sentence in str(refused.value), sentence
    # a census that does not follow the model's wiring
    swapped = _small_census()
    parts = swapped["configurations"]["plain"]["cc10.0/sm148"]["rows"]["1"]
    parts["prologue"], parts["layer"] = parts["layer"], parts["prologue"]
    with pytest.raises(
        ValueError, match="the census does not follow this model's wiring: call 1 of a step is embedding, the census has linear_attention.step there"
    ):
        census.Target(swapped).claim(plan, model, installed=RECORDED)
    longer = _small_census()
    longer["configurations"]["plain"]["cc10.0/sm148"]["rows"]["1"]["after_step"].append(
        ["embedding", "embedding_gather", {"launch": [[40, 1, 1], [128, 1, 1], 0]}]
    )
    with pytest.raises(ValueError, match="the census does not follow this model's wiring: it has embedding after the last call of a step"):
        census.Target(longer).claim(plan, model, installed=RECORDED)
    with pytest.raises(ValueError, match="the census has no entry list for the layer kind 'full_attention'"):
        census.Target(_small_census()).claim(*_target_plan()[:2], installed=RECORDED)


def test_reading_a_census_file(tmp_path):
    from model_samples import census

    path = tmp_path / "some_census.json"
    path.write_text(json.dumps(_small_census()))
    assert census.Target.load(str(path)).name == "some_census.json"
    path.write_text("{not json")
    with pytest.raises(ValueError, match=r"^some_census.json: not JSON \(") as refused:
        census.Target.load(str(path))
    assert str(tmp_path) not in str(refused.value)
    with pytest.raises(ValueError, match=r"^gone.json: No such file or directory$"):
        census.Target.load(str(tmp_path / "gone.json"))


def test_a_target_launch_in_another_shape_than_the_recorded_one_fails():
    """Item 7.  Every launch of a block call whose identity is ``target`` has the launch shape the census recorded for it --
    also where the entry itself states and launches another shape (item 5 then holds, and only the census sees it), and for
    a kernel whose shape no entry states (the gather)."""
    from dataclasses import replace

    from model_samples import census
    from model_samples.census import HOST, NOT_YET, TARGET, Claim, Facts, Identity

    sites = _sites()
    identities = [
        Identity(TARGET, "embedding_gather", "code", "", GATHER_SHAPE),
        Identity(NOT_YET, "add_norm_quant", "port", "reference chain"),
        Identity(TARGET, "gdn_decode_fused", "code", "", SHAPE),
        Identity(NOT_YET, "cublaslt_fp8_gemm", "code", "bf16 copy of 8-bit weights"),
        Identity(HOST),
    ]
    blocks = [site.block for site in sites]
    claim = Claim("kernel_census.json", "plain", "cc 10.0 / 148 SMs", 1, "2026-11-12", "recorded", False, blocks, identities, reasons=["fallback rows present"])
    assert claim.recorded == {0: GATHER_SHAPE, 2: SHAPE}
    clean = census.Traced.of(sites, [], _step(), 1, "", rows=1, claim=claim)
    assert clean.judge() == (census.PASS, [])
    assert clean.details()[0] == (
        "embedding: torch.index_select, 1 launch per step (1 call x 1 per call); embedding_gather by name, and in the launch shape the census records: "
        "grid (40, 1, 1), block (128, 1, 1), shared memory 0 B"
    )
    assert clean.details()[-3:] == [
        WHOLE + "4 launches per step = 3 the target's own compiled kernels + 1 cuDNN Frontend entry with byte-equal results + 0 declared stand-ins",
        "kernel identity, as planned: 2 of 4 launches of a step are target kernels (2 the target's own compiled kernels + 0 cuDNN Frontend entries "
        "with byte-equal results); not yet: add_norm_quantize 1, projection.fp8.out 1; 0 declared stand-ins",
        f"{CENSUS}: NOT CLAIMED (fallback rows present); {COMPARED_2}: each in the recorded one",
    ]
    assert clean.dump()["target"]["not_claimed"] == ["fallback rows present"] and [site["identity"] for site in clean.dump()["sites"]] == [
        "target",
        "not yet (reference chain)",
        "target",
        "not yet (bf16 copy of 8-bit weights)",
        "host path, not compared",
    ]
    # the fused step states and launches a grid of 48: item 5 holds, the census has 96
    other = ((48, 1, 1), (256, 1, 1), 42112)
    moved = [sites[0], sites[1], replace(sites[2], facts=Facts((SYMBOL,), {"PDL": False, "PDL_RELEASE": False}, other)), *sites[3:]]
    assert census.compare([_step(fused=(SYMBOL, other))], moved).ok, "without a census nothing is wrong with it"
    traced = census.Traced.of(moved, [], _step(fused=(SYMBOL, other)), 1, "", rows=1, claim=claim)
    status, lines = traced.judge()
    assert status == census.FAIL and lines == [
        "step 0: linear_attention.step (layer 0) launched gdn_decode_fused with grid (48, 1, 1), block (256, 1, 1), shared memory 42,112 B; "
        f"the {CENSUS} has grid (96, 1, 1), block (256, 1, 1), shared memory 42,112 B"
    ]
    assert traced.details()[-1] == f"{CENSUS}: NOT CLAIMED (fallback rows present)"
    # ... and so for each of the three parts of a launch shape: another block, another shared memory, the grid as recorded
    for other, text in (
        (((96, 1, 1), (128, 1, 1), 42112), "grid (96, 1, 1), block (128, 1, 1), shared memory 42,112 B"),
        (((96, 1, 1), (256, 1, 1), 42128), "grid (96, 1, 1), block (256, 1, 1), shared memory 42,128 B"),
    ):
        moved = [sites[0], sites[1], replace(sites[2], facts=Facts((SYMBOL,), {"PDL": False, "PDL_RELEASE": False}, other)), *sites[3:]]
        status, lines = census.Traced.of(moved, [], _step(fused=(SYMBOL, other)), 1, "", rows=1, claim=claim).judge()
        assert status == census.FAIL and lines == [
            f"step 0: linear_attention.step (layer 0) launched gdn_decode_fused with {text}; "
            f"the {CENSUS} has grid (96, 1, 1), block (256, 1, 1), shared memory 42,112 B"
        ], text
    other = ((48, 1, 1), (256, 1, 1), 42112)
    moved = [sites[0], sites[1], replace(sites[2], facts=Facts((SYMBOL,), {"PDL": False, "PDL_RELEASE": False}, other)), *sites[3:]]
    # the gather in another shape than the recorded one: no entry states a shape for it, the census does
    step = _step()
    step[0] = step[0]._replace(grid=(80, 1, 1))
    assert census.compare([step], sites).ok
    status, lines = census.Traced.of(sites, [], step, 1, "", rows=1, claim=claim).judge()
    assert status == census.FAIL and lines == [
        f"step 0: embedding launched embedding_gather with grid (80, 1, 1), block (128, 1, 1), shared memory 0 B; the {CENSUS} has grid (40, 1, 1), "
        "block (128, 1, 1), shared memory 0 B"
    ]
    # records without launch arguments: the recorded shapes were not observed
    bare = [record._replace(grid=None, block=None, shared_memory=None) for record in _step()]
    unseen = census.Traced.of([replace(site, facts=None) if site.facts else site for site in sites], [], bare, 1, "", rows=1, claim=claim)
    assert unseen.judge() == (census.NOT_OBSERVED, []) and unseen.details()[-1].endswith("; the launch shapes of its target launches were not observed")
    # a call that is not the target's (another form, a fallback row) is held to no recorded shape -- whatever its identity carries
    reason = "the row's form is [PDL False, PDL_RELEASE False], the target's [PDL True, PDL_RELEASE True]"
    claim.identities[2] = Identity(census.OTHER, "gdn_decode_fused", "code", reason, SHAPE)
    assert claim.recorded == {0: GATHER_SHAPE}, "only a call whose identity is target (or stand-in) is held to the census"
    assert census.Traced.of(moved, [], _step(fused=(SYMBOL, other)), 1, "", rows=1, claim=claim).judge() == (census.PASS, [])


def test_the_gate_entrys_adapter_over_a_stand_in_entry(monkeypatch):
    """``fe.attention_gate_quant_decode`` over a stand-in entry: the row's form reaches the constructor as the entry's own
    arguments (``THREADS`` in its config, ``PDL_RELEASE`` as its launch knob), the width is the attention output's, every call
    gets an output buffer of its own (the entry's one buffer would be overwritten by the next call), the scale is one device
    tensor per value, the call log reads the instance that was called, and the twin is the gate at the kernel's rounding
    points.  An entry that declines the geometry or the row count of the row is a wrong table: a ``ValueError``."""
    from dataclasses import replace

    import cudnn
    from cudnn.attention_gate.frost import Cfg

    from model_samples import census
    from model_samples.blocks import DeviceFacts, plan
    from model_samples.providers import Quant, fe, torch_ref
    from model_samples.qwen_dense import QwenDense, configs
    from model_samples.weights import ScaledTensor

    built = []

    class Entry:
        limit = 160

        def __init__(self, device, width, cfg=None, *, pdl_release=None):
            if width % 8:
                raise NotImplementedError(f"this stand-in serves widths that are a multiple of 8; got {width}")
            self.width, self.cfg, self.pdl_release, self.calls = width, cfg, pdl_release, []
            built.append(self)

        def compile(self, attn, gate, scale, out=None):
            self.calls.append(("compile", tuple(attn.shape), out))

        def __call__(self, attn, gate, scale, out=None):
            self.calls.append(("launch", tuple(attn.shape), out, scale))
            out.view(torch.uint8).fill_(len(self.calls))
            return out

        def kernel_symbols_for(self, rows):
            return (GATE_SYMBOL,)

        def form_for(self, rows):
            return {"THREADS": 256 if self.cfg is None else self.cfg.THREADS, "PDL_RELEASE": bool(self.pdl_release)}

        def launch_shape_for(self, rows):
            if rows > self.limit:
                raise NotImplementedError(f"this stand-in serves at most {self.limit} rows per call; got {rows}")
            return ((3 * rows, 1, 1), (256, 1, 1), 0)

    monkeypatch.setattr(cudnn, "AttentionGateQuantDecode", Entry, raising=False)
    monkeypatch.setattr(fe, "_ENTRIES", {})
    model = QwenDense.reduced(2)
    row = configs.PROVIDERS["attention.gate_quantize"][0]
    assert row.key == "fe.attention_gate_quant_decode" and row.form == (("THREADS", 256), ("PDL_RELEASE", True)) and row.classes == ("cc10.x",)
    assert row.rows == (1, 160) and row.band == "gate_fp8" and set(row.stacks) == {"native", "fe-entries"}
    stated = DeviceFacts((10, 0), "stated device", 148, 0, 0)
    options = {"device": "cpu", "check_kernels": True}
    block = fe.AttentionGateQuantDecodeBlock(row, model, stated, options)
    block.check_support(1, 1)
    assert built == [] and block.serving() == ("cudnn.AttentionGateQuantDecode", ()), "on stated facts nothing is built; before a call nothing has served"
    facts = block.launch_facts(1)
    assert (built[0].width, built[0].cfg, built[0].pdl_release) == (6144, Cfg(THREADS=256), True), "the row's form, as the entry's own arguments"
    assert facts == ((GATE_SYMBOL,), {"THREADS": 256, "PDL_RELEASE": True}, GATE_SHAPE) and block.kernels == (
        census.Kernel("gate_quant", "cuDNN FE", "exact", launches=1, shape="entry"),
    )
    quant = Quant("fp8", 0.0625)
    attn, gate = torch.zeros((2, 6144), dtype=torch.bfloat16), torch.zeros((2, 6144), dtype=torch.bfloat16)
    first = block(attn, gate, quant)
    second = block(attn, gate, quant)
    (_, shape, out1, scale1), (_, _, out2, scale2) = built[0].calls
    assert shape == (2, 6144) and out1.dtype == torch.float8_e4m3fn and tuple(out1.shape) == (2, 6144) and out1 is not out2, "a buffer of its own per call"
    assert isinstance(first, ScaledTensor) and first.scale == 0.0625 and first.codes.dtype == torch.uint8 and first.codes.data_ptr() == out1.data_ptr()
    assert int(first.codes[0, 0]) == 1 and int(second.codes[0, 0]) == 2, "the first call's bytes are still its own after the second call"
    assert (
        scale1 is scale2 and scale1.dtype == torch.float32 and scale1.tolist() == [0.0625]
    ), "one scale tensor per value: no copy to the device inside a later step"
    assert block.serving() == ("cudnn.AttentionGateQuantDecode #1", (("THREADS", 256), ("PDL_RELEASE", True)))
    block.compile(4, 1)
    assert [call[:2] for call in built[0].calls[2:]] == [("compile", (4, 6144)), ("launch", (4, 6144))] and len(
        built
    ) == 1, "compiled and launched once at load"
    assert fe.AttentionGateQuantDecodeBlock(row, model, stated, options).launch_facts(1) == facts and len(built) == 1, "one entry per form, shared"
    reference = torch_ref.GateQuantize(configs.PROVIDERS["attention.gate_quantize"][1], model, stated, options)
    twin = block.twin(reference)
    assert type(twin) is torch_ref.GateQuantizeQuotient and block.twin(reference) is twin and reference.twin(reference) is reference
    plain = fe.AttentionGateQuantDecodeBlock(replace(row, form=(("PDL_RELEASE", False),)), model, stated, options)
    assert plain.launch_facts(1).form == {"THREADS": 256, "PDL_RELEASE": False} and (built[1].cfg, built[1].pdl_release) == (None, False)
    plain(attn, gate, quant)
    assert plain.serving() == ("cudnn.AttentionGateQuantDecode #2", (("PDL_RELEASE", False),)), "its place among the entries of its kind"
    for name in (
        "attention_gate/frost/gate_quant_decode.py",
        "attention_gate/frost/kernel/gate_quant_decode_config.py",
        "attention_gate/frost/kernel/gate_quant_decode_f16.py",
    ):
        assert name in fe.SOURCE_FILES and (Path(cudnn.__file__).parent / name).is_file(), f"the source hash of a run covers {name}"

    with pytest.raises(ValueError, match="cudnn.AttentionGateQuantDecode writes static 8-bit codes; this block is asked for 'nvfp4'"):
        block(attn, gate, Quant("nvfp4", 1.0))
    with pytest.raises(ValueError, match=r"provider row fe.attention_gate_quant_decode: its form names PDL; this adapter passes THREADS, PDL_RELEASE"):
        fe.AttentionGateQuantDecodeBlock(replace(row, form=(("PDL", True),)), model, stated, options).check_support(1, 1)

    # on a real device the adapter asks the entry, at plan time, for the launch of the run's row count: a decline is a wrong table, not a SKIP
    real = DeviceFacts((10, 0), "stated device", 148, 180 * GIB, 80 * GIB)
    fe.AttentionGateQuantDecodeBlock(row, model, real, options).check_support(4, 1)
    monkeypatch.setattr(Entry, "limit", 3)
    wrong = (
        r"^the provider table is wrong: the row of fe.attention_gate_quant_decode claims this run \(4 rows on compute capability 10.0\), and "
        r"cudnn.AttentionGateQuantDecode declines it: this stand-in serves at most 3 rows per call; got 4$"
    )
    with pytest.raises(ValueError, match=wrong):
        fe.AttentionGateQuantDecodeBlock(row, model, real, options).check_support(4, 1)
    table = dict(model.provider_table)
    with pytest.raises(ValueError, match=wrong):  # through the plan: a ValueError, which a sample prints as an ERROR
        plan(table, model.regimes, real, 4, 1, "native", model=model, options=options)
    monkeypatch.setattr(fe, "_ENTRIES", {})
    odd = QwenDense.reduced(2)
    odd.geometry = replace(odd.geometry, attn_heads=3, attn_head_dim=5)  # 15 elements per row: not a width the stand-in serves
    with pytest.raises(ValueError, match=r"^the provider table is wrong: cudnn.AttentionGateQuantDecode declines the geometry of the row of fe.attention_gate"):
        fe.AttentionGateQuantDecodeBlock(row, odd, real, options).check_support(1, 1)

    class Older:  # an installed package without the entry, and one whose entry states no launch facts
        __version__ = "1.32.3"

        def __getattr__(self, name):
            raise AttributeError(name)

    monkeypatch.setattr(fe.AttentionGateQuantDecodeBlock, "_cudnn", lambda self: Older())
    with pytest.raises(NotImplementedError, match=r"needs nvidia-cudnn-frontend with the attention gate \+ quantise decode step \(>= 1.33.0\); this is 1.32.3"):
        fe.AttentionGateQuantDecodeBlock(row, model, stated, options).check_support(1, 1)
    monkeypatch.undo()
    monkeypatch.setattr(cudnn, "AttentionGateQuantDecode", type("NoFacts", (), {}), raising=False)
    with pytest.raises(NotImplementedError, match=r"the kernel check needs the launch facts of cudnn.AttentionGateQuantDecode \(kernel_symbols_for"):
        fe.AttentionGateQuantDecodeBlock(row, model, stated, options).check_support(1, 1)


def test_an_entry_that_declines_the_row_count_its_row_claims_is_an_error_of_the_table(monkeypatch):
    """The fused step's adapter on a real device: the row claims the run, so the entry must state a launch for its row
    count.  One that declines is ``the provider table is wrong`` -- a ``ValueError``, never a SKIP; on stated device facts
    nothing is built and nothing is asked."""
    import cudnn

    from model_samples.blocks import DeviceFacts, plan
    from model_samples.providers import fe
    from model_samples.qwen_dense import QwenDense

    class Entry:
        def __init__(self, device, eps, geometry=None, **knobs):
            pass

        def kernel_symbols_for(self, rows):
            return ()

        def form_for(self, rows):
            return {"PDL": False, "PDL_RELEASE": False}

        def launch_shape_for(self, rows):
            if rows > 2:
                raise NotImplementedError(f"this stand-in serves 2 rows; got rows={rows}")
            return ((96, rows, 1), (256, 1, 1), 42112)

    monkeypatch.setattr(cudnn, "GdnDecodeFused", Entry, raising=False)
    monkeypatch.setattr(fe, "_ENTRIES", {})
    model = QwenDense.reduced(2)
    real = DeviceFacts((10, 0), "stated device", 148, 180 * GIB, 80 * GIB)
    assert plan(model.provider_table, model.regimes, real, 2, model=model, options={"device": "cpu"}).rows["linear_attention.step"].key == "fe.gdn_decode_fused"
    with pytest.raises(ValueError) as wrong:
        plan(model.provider_table, model.regimes, real, 4, model=model, options={"device": "cpu"})
    assert str(wrong.value) == (
        "the provider table is wrong: the row of fe.gdn_decode_fused claims this run (4 rows on compute capability 10.0), and cudnn.GdnDecodeFused "
        "declines it: this stand-in serves 2 rows; got rows=4"
    )
    stated = DeviceFacts((10, 0), "stated device", 148, 0, 0)
    assert plan(model.provider_table, model.regimes, stated, 4, model=model, options={"device": "cpu"}).sequences == 4, "on stated facts nothing is asked"


def test_the_gates_twin_and_its_band_on_stated_tensors():
    """The gate at the kernel's rounding points, and the band its entry is judged with.  The kernel's chain in fp32 -- the gated
    value as the quotient ``attn / (1 + exp(-gate))``, rounded once to bf16, then the 8-bit code -- is inside the band
    against the twin that forms the quotient in fp64.  A byte that is the code of neither bf16 value around the gated
    value, a byte off the nearest value's code beyond the margin, and a NaN byte each fail the block."""
    from types import SimpleNamespace

    from model_samples import tie_out
    from model_samples.blocks import DeviceFacts
    from model_samples.providers import Quant, torch_ref
    from model_samples.qwen_dense import configs
    from model_samples.weights import quantize_fp8

    assert tie_out.GATE_MARGIN_ULP32 == 2.5, "the margin of the entry's page (docs/operations/attention/gate_quant_decode.md): it moves only with that page"
    page = (MODELS.parents[1] / "docs" / "operations" / "attention" / "gate_quant_decode.md").read_text()
    assert "more than 2.5 fp32 units away from" in " ".join(page.split()), "the band is the contract that page states"
    generator = torch.Generator().manual_seed(7)
    attn = torch.randn((8, 4096), generator=generator).to(torch.bfloat16)
    gate = (3 * torch.randn((8, 4096), generator=generator)).to(torch.bfloat16)
    special_attn = torch.tensor([448.0, -448.0, 0.0, -0.0, 1e-30, 3.3895e38, 1.0625, 17.0, 3.0, 1.0]).to(torch.bfloat16)
    special_gate = torch.tensor([40.0, -40.0, 0.0, 88.5, -89.0, 104.0, -700.0, 700.0, 0.0, 0.0]).to(torch.bfloat16)
    attn[0, :10], gate[0, :10] = special_attn, special_gate
    quant = Quant("fp8", 0.02)
    row = configs.PROVIDERS["attention.gate_quantize"][1]
    model = SimpleNamespace(norm_eps=1e-6)

    def kernel_chain(a, g):  # what the entry computes: fp32 throughout, one rounding to bf16, one to the code
        return quantize_fp8((a.float() / (1.0 + torch.exp(-g.float()))).to(torch.bfloat16), quant.scale).codes

    def judged(codes, precision="fp64"):
        twin = torch_ref.GateQuantizeQuotient(row, model, DeviceFacts((0, 0), "cpu"), {"reference_precision": precision})
        stats = tie_out.BlockStats("gate_fp8")
        tie_out.BlockTap._judge(stats, (codes,), tie_out._tensors(twin(attn, gate, quant)), [], [], 0, twin)
        return stats, twin

    codes = kernel_chain(attn, gate)
    stats, twin = judged(codes)
    assert stats.ok and stats.outside == 0 and stats.off_nearest == 0 and stats.total_bytes == codes.numel() and stats.finite
    assert stats.equal_bytes > 0.999 * stats.total_bytes, "an fp32 quotient leaves the nearest bf16 of the fp64 one on a few elements in a million at most"
    assert "% of 8-bit bytes are the code of the nearest bf16 value; " in stats.line() and stats.line().endswith("0 the code of neither")
    nearest, other, margin = torch_ref.gate_quotient_bracket(attn, gate, torch.float64)
    assert bool((margin[0, 8:10] == float("inf")).all()) and torch.equal(nearest[0, 8:10], other[0, 8:10]), "a gate of 0 halves: the quotient is a bf16 value"
    assert (
        nearest[0, 8:10].tolist() == [1.5, 0.5] and nearest[0, 4].item() == 0.0 and nearest[0, 6].item() == 0.0
    ), "an fp32 exponential that overflows gives +-0"
    assert bool(((nearest.double() - other.double()).abs() <= 2.0 ** (torch.frexp(nearest.double().abs()).exponent.clamp(min=-125) - 8)).all()), "neighbours"
    # the exponential of a gate below about -88.7 overflows in fp32, where the kernel forms it: the gated value is then +-0,
    # however large the attention output.  In fp64 alone that quotient is a finite value of ordinary size
    large = torch.tensor([3.3895e38, -3.3895e38, 448.0]).to(torch.bfloat16)
    below = torch.tensor([-89.0, -89.0, -90.0]).to(torch.bfloat16)
    assert bool((large.double() / (1.0 + torch.exp(-below.double()))).abs()[:2].gt(0.5).all()), "in fp64 the first two quotients are of order one"
    zero, zero_other, zero_margin = torch_ref.gate_quotient_bracket(large, below, torch.float64)
    assert zero.tolist() == [0.0, -0.0, 0.0] and bool((zero_margin == float("inf")).all()) and torch.equal(zero, zero_other), "the fp32 overflow, also in fp64"
    assert torch.signbit(zero).tolist() == [False, True, False], "the sign of the attention output is kept"
    # a quotient exactly on the midpoint of two bf16 values: half of an odd subnormal.  The nearest value is the even one of
    # the two, the other the odd one, and the margin is zero -- either code is the kernel's own arithmetic there
    odd = torch.tensor([1, 3, 5, 0x8001, 0x8003], dtype=torch.int32).to(torch.int16).view(torch.bfloat16)  # the bit patterns
    halves = torch.zeros(5).to(torch.bfloat16)  # a gate of 0: the quotient is half the attention output
    even, uneven, on_midpoint = torch_ref.gate_quotient_bracket(odd, halves, torch.float64)
    bits = lambda values: (values.view(torch.int16).to(torch.int32) & 0xFFFF).tolist()  # noqa: E731
    assert bits(even) == [0, 2, 2, 0x8000, 0x8002] and bits(uneven) == [1, 1, 3, 0x8001, 0x8001] and on_midpoint.tolist() == [0.0] * 5, "nearest-even"
    assert judged(kernel_chain(attn, gate), "ieee-fp32")[0].equal_bytes == codes.numel(), "an fp32-accumulated twin is the kernel's chain itself"
    other_codes, margin = twin.bracket
    twin_codes = quantize_fp8(nearest, quant.scale).codes

    # (1) a byte that is the code of neither bf16 value around the gated value
    wrong = codes.clone()
    wrong[3, 100] = 0x7E if int(codes[3, 100]) != 0x7E else 0x00
    stats, _ = judged(wrong)
    assert not stats.ok and stats.outside == 1 and stats.finite and stats.line().endswith("; 1 the code of neither")
    # (2) the code of the OTHER bf16 value where the gated value is far from the midpoint: inside the bracket, off the nearest
    far = ((other_codes != twin_codes) & (margin > 1000.0) & (codes == twin_codes)).nonzero()
    assert len(far), "the operands hold an element whose two bf16 neighbours have different codes"
    r, c = (int(i) for i in far[0])
    wrong = codes.clone()
    wrong[r, c] = other_codes[r, c]
    stats, _ = judged(wrong)
    assert not stats.ok and stats.outside == 0 and stats.off_nearest == 1 and "1 of them more than 2.5 fp32 units from the midpoint" in stats.line()
    # ... while the same choice INSIDE the margin is the kernel's own arithmetic and holds
    inside = wrong.clone()
    bracket = (other_codes.clone(), margin.clone())
    bracket[1][r, c] = 2.0

    def judged_inside():
        twin = torch_ref.GateQuantizeQuotient(row, model, DeviceFacts((0, 0), "cpu"), {})
        out = twin(attn, gate, quant)
        twin.bracket = bracket
        stats = tie_out.BlockStats("gate_fp8")
        tie_out.BlockTap._judge(stats, (inside,), tie_out._tensors(out), [], [], 0, twin)
        return stats

    assert judged_inside().ok and judged_inside().off_nearest == 0
    # (3) a NaN byte
    wrong = codes.clone()
    wrong[0, 0] = 0x7F
    stats, _ = judged(wrong)
    assert not stats.ok and not stats.finite and "NON-FINITE VALUES" in stats.line()
    # ... and a NaN operand, which the kernel and the twin both turn into the NaN byte: equal bytes, and not a pass
    nan_attn = attn.clone()
    nan_attn[1, 5] = float("nan")
    twin = torch_ref.GateQuantizeQuotient(row, model, DeviceFacts((0, 0), "cpu"), {})
    stats = tie_out.BlockStats("gate_fp8")
    tie_out.BlockTap._judge(stats, (kernel_chain(nan_attn, gate),), tie_out._tensors(twin(nan_attn, gate, quant)), [], [], 0, twin)
    assert not stats.ok and not stats.finite and stats.outside == 0 and stats.off_nearest == 0, "only the finite rule sees it"
    # the nearest bf16 value is decided against the midpoint, never by a cast that rounds twice: over two million seeded pairs
    # a few quotients round to the midpoint itself in fp32, where nearest-even then picks the farther of the two values
    many = torch.Generator().manual_seed(11)
    a = torch.randn((1 << 21,), generator=many).to(torch.bfloat16)
    g = (3 * torch.randn((1 << 21,), generator=many)).to(torch.bfloat16)
    nearest, other, margin = torch_ref.gate_quotient_bracket(a, g, torch.float64)
    quotient = a.double() / (1.0 + torch.exp(-g.double()))
    twice = quotient.float().to(torch.bfloat16)
    assert int((nearest != twice).sum()) > 0, "the pairs hold a quotient whose two-step rounding leaves the nearest bf16 value"
    assert bool(((quotient - nearest.double()).abs() <= (quotient - other.double()).abs()).all()), "the nearest value is never the farther one"
    assert float(margin[nearest != twice].max()) < tie_out.GATE_MARGIN_ULP32, "and such a quotient sits inside the band's margin"
    assert torch.equal(other[nearest != twice], twice[nearest != twice]), "there the bracket keeps the value the two-step rounding gives as its other end"
    # one such quotient on stated operands: 1.8984375 / (1 + exp(-0.02880859375)) = 0.96289063..., 0.14 fp32 units above the
    # midpoint of the bf16 values 0.9609375 and 0.96484375.  In fp32 it is the midpoint itself, which rounds to the even
    # value, 0.9609375: the farther one, and the one a kernel that forms the quotient in fp32 writes there (code 56 at the
    # scale 0.90625; the nearest value's code is 57).  The bracket holds both values, so a row of either code is inside the
    # band; a byte of a third code is not
    near_attn = torch.full((1, 6144), 1.8984375).to(torch.bfloat16)
    near_gate = torch.full((1, 6144), 0.02880859375).to(torch.bfloat16)
    near_quant = Quant("fp8", 0.90625)
    assert near_attn[0, 0].item() == 1.8984375 and near_gate[0, 0].item() == 0.02880859375, "bf16 values, as stated"
    quotient = float(near_attn[0, 0].double() / (1.0 + torch.exp(-near_gate[0, 0].double())))
    assert abs(quotient - 0.9628906331227847) < 1e-15 and 0.13 < (quotient - (0.9609375 + 0.96484375) / 2) / 2.0**-24 < 0.14
    assert float(torch.tensor(quotient, dtype=torch.float64).float()) == 0.962890625, "in fp32 the quotient is the midpoint of the two bf16 values"
    nearest, other, margin = torch_ref.gate_quotient_bracket(near_attn, near_gate, torch.float64)
    assert bool((nearest == 0.96484375).all()) and bool((other == 0.9609375).all()), "the nearest value, and the one the two-step rounding gives"
    assert 0.13 < float(margin.min()) and float(margin.max()) < 0.14
    assert quantize_fp8(nearest, near_quant.scale).codes.unique().tolist() == [57] and quantize_fp8(other, near_quant.scale).codes.unique().tolist() == [56]

    def judged_near(codes):
        twin = torch_ref.GateQuantizeQuotient(row, model, DeviceFacts((0, 0), "cpu"), {})
        stats = tie_out.BlockStats("gate_fp8")
        tie_out.BlockTap._judge(stats, (codes,), tie_out._tensors(twin(near_attn, near_gate, near_quant)), [], [], 0, twin)
        return stats

    farther = torch.full((1, 6144), 56, dtype=torch.uint8)
    stats = judged_near(farther)
    assert stats.ok and stats.outside == 0 and stats.off_nearest == 0 and stats.equal_bytes == 0 and stats.total_bytes == 6144, stats.line()
    assert "; 6,144 of 6,144 the code of the other one around the gated value, 0 of them more than 2.5 fp32 units" in stats.line()
    stats = judged_near(torch.full((1, 6144), 57, dtype=torch.uint8))
    assert stats.ok and stats.equal_bytes == 6144, "the nearest value's code"
    stats = judged_near(quantize_fp8((near_attn.float() / (1.0 + torch.exp(-near_gate.float()))).to(torch.bfloat16), near_quant.scale).codes)
    assert stats.ok and stats.outside == 0, "the fp32 chain of this host, whichever of the two values it rounds to"
    third = farther.clone()
    third[0, 0] = 55
    stats = judged_near(third)
    assert not stats.ok and stats.outside == 1 and stats.line().endswith("; 1 the code of neither")
    # (4) state a call updates is compared byte for byte by a band that has no rule for it
    stats = tie_out.BlockStats("bf16_rounding")
    out = torch.ones((1, 8), dtype=torch.bfloat16)
    tie_out.BlockTap._judge(stats, (out,), (out,), [(torch.zeros(4),)], [(torch.ones(4),)])
    assert not stats.ok and not stats.state_equal and stats.line().endswith("; STATE DIFFERS")
    with pytest.raises(ValueError, match="the gate's twin quantises to static 8-bit codes"):
        torch_ref.GateQuantizeQuotient(row, model, DeviceFacts((0, 0), "cpu"), {})(attn, gate, Quant("nvfp4", 1.0))
    assert "gate_fp8" in tie_out.BANDS and "2.5 fp32 units" in tie_out.BANDS["gate_fp8"]
    chain = torch_ref.GateQuantize(row, model, DeviceFacts((0, 0), "cpu"), {})  # the reference chain: it rounds elsewhere and keeps no bracket
    with pytest.raises(ValueError, match="the band gate_fp8 is judged against the gate at the kernel's rounding points"):
        tie_out.BlockTap._judge(tie_out.BlockStats("gate_fp8"), (codes,), tie_out._tensors(chain(attn, gate, quant)), [], [], 0, chain)


def test_the_gates_bracket_holds_both_bf16_values_around_the_quotient():
    """The bracket of the gate's twin is the two bf16 values around the quotient ``attn / (1 + exp(-gate))`` in every branch of
    its rounding: the nearest one first, the one on the other side of the quotient second -- also where the quotient is the
    midpoint of the two in fp32 and the two-step rounding (fp64 to fp32 to bf16) lands on the farther value, below the
    quotient or above it, for either sign.  A quotient that is a bf16 value has that value twice.  A bracket that names the
    nearest value twice elsewhere would refuse a kernel that rounds such a quotient to the other one, which its band allows."""
    from model_samples import tie_out
    from model_samples.providers import torch_ref

    def bracket(attn, gate):
        operands = (torch.tensor([attn]).to(torch.bfloat16), torch.tensor([gate]).to(torch.bfloat16))
        assert operands[0].item() == attn and operands[1].item() == gate, "the stated operands are bf16 values"
        nearest, other, margin = torch_ref.gate_quotient_bracket(*operands, torch.float64)
        quotient = float(operands[0].double() / (1.0 + torch.exp(-operands[1].double())))
        return nearest.item(), other.item(), margin.item(), quotient, float(torch.tensor(quotient, dtype=torch.float64).float().to(torch.bfloat16))

    # (attn, gate) -> (the nearest bf16 value, the other one); the quotient is the midpoint of the two in fp32, which rounds
    # to the even value: the FARTHER one, below the quotient in the first case and above it in the others
    on_a_midpoint = {
        (1.8984375, 0.02880859375): (0.96484375, 0.9609375),
        (1.8984375, -0.02880859375): (0.93359375, 0.9375),
        (1.3828125, -0.9921875): (0.373046875, 0.375),
        (1.21875, 0.3828125): (0.72265625, 0.7265625),
    }
    for (attn, gate), (near, far) in on_a_midpoint.items():
        for sign in (1.0, -1.0):
            nearest, other, margin, quotient, twice = bracket(sign * attn, gate)
            assert (nearest, other) == (sign * near, sign * far), (attn, gate, sign)
            assert twice == sign * far and abs(quotient - nearest) < abs(quotient - other), "the two-step rounding gives the farther value"
            assert min(nearest, other) < quotient < max(nearest, other) and 0.0 < margin < 0.5, "less than half an fp32 unit from the midpoint"
    # away from a midpoint the two-step rounding is the nearest value, and the other one is its neighbour across the quotient
    for (attn, gate), (near, far) in {(1.0, 1.0): (0.73046875, 0.734375), (1.0, 3.0): (0.953125, 0.94921875)}.items():
        for sign in (1.0, -1.0):
            nearest, other, margin, quotient, twice = bracket(sign * attn, gate)
            assert (nearest, other) == (sign * near, sign * far) and twice == nearest and margin > tie_out.GATE_MARGIN_ULP32, (attn, gate, sign)
            assert min(nearest, other) < quotient < max(nearest, other)
    # a quotient that is a bf16 value: that value twice, at no distance from a midpoint that matters
    for (attn, gate), value in {(3.0, 0.0): 1.5, (-1.0, 0.0): -0.5, (0.0, 2.0): 0.0, (448.0, 40.0): 448.0}.items():
        assert bracket(attn, gate)[:3] == (value, value, float("inf")), (attn, gate)
    # over two million seeded pairs: the two values are one bf16 step apart (or equal, where the quotient is one), the
    # quotient lies between them, the first is never the farther one -- and an fp32 chain rounds to one of the two everywhere
    many = torch.Generator().manual_seed(11)
    a = torch.randn((1 << 21,), generator=many).to(torch.bfloat16)
    g = (3 * torch.randn((1 << 21,), generator=many)).to(torch.bfloat16)
    nearest, other, margin = torch_ref.gate_quotient_bracket(a, g, torch.float64)
    quotient = a.double() / (1.0 + torch.exp(-g.double()))
    bits = lambda values: values.view(torch.int16).to(torch.int32) & 0x7FFF  # noqa: E731  the magnitude's bit pattern
    step = (bits(nearest) - bits(other)).abs()
    assert bool(((step == 1) | ((step == 0) & (margin == float("inf")))).all()), "neighbours, or one value where the quotient is a bf16 value"
    assert bool((torch.signbit(nearest) == torch.signbit(other)).all()), "on one side of zero"
    low, high = torch.minimum(nearest.double(), other.double()), torch.maximum(nearest.double(), other.double())
    assert bool(((low <= quotient) & (quotient <= high)).all()), "the quotient lies between the two"
    assert bool(((quotient - nearest.double()).abs() <= (quotient - other.double()).abs()).all()), "the first is the nearest"
    chain = (a.float() / (1.0 + torch.exp(-g.float()))).to(torch.bfloat16)
    assert bool(((chain == nearest) | (chain == other)).all()), "the kernel's chain in fp32 rounds to one of the two"
    twice = quotient.float().to(torch.bfloat16)
    assert int((nearest != twice).sum()) > 0 and torch.equal(other[nearest != twice], twice[nearest != twice]), "the farther value stays in the bracket"


def test_a_run_of_the_native_stack_through_its_last_line(monkeypatch, capsys, tmp_path):
    """The sample's ``main()`` with ``--stack native``, on stand-in providers, to its last line.  The kernel check runs without
    a flag; the planned table has the identity column and the three numbers; the gate entry is judged against its twin
    inside its band; the census line reads ``NOT CLAIMED (fallback rows present; ...)`` and, where the launches hold, says
    that the target launches have the recorded shape.  A target launch in another shape than the recorded one fails the run;
    a wrong byte of the gate fails its block; ``--provider`` is said; without a profiler the run is ``RAN native stack``."""
    target = tmp_path / "census.json"
    code, lines = _sample_run(monkeypatch, capsys, ["--stack", "native", "--sequences", "1", "--dump-census", str(target)])
    assert code == 0 and lines[-1] == "[00] PASS", lines[-4:]
    planned = lines.index(next(line for line in lines if line.startswith("kernels, planned (decided before the load")))
    assert lines[planned + 1].split()[-1] == "identity" and "stack native" in lines[planned]
    assert lines[planned + 14] == "  add_norm_quantize at linear_attention.first: [ADD False, STORE_RESIDUAL True, BF16_OUT True, QUANT e4m3_static]"
    assert lines[planned + 21].strip().startswith(WHOLE + "20 launches per step = 9 ")
    reached = "kernel identity, as planned: 8 of 20 launches of a step are target kernels (2 the target's own compiled kernels + 6 cuDNN Frontend entries with "
    assert lines[planned + 22].strip().startswith(reached + "byte-equal results); not yet: projection.fp8.in 2, ")
    (kernels,) = _kernel_lines(lines)
    assert kernels.endswith("    PASS") and "8 on exact rows, in the wiring order" in kernels
    not_claimed = f"    {CENSUS}: NOT CLAIMED (fallback rows present; the plan has no block for output_head.quantize)"
    assert not_claimed + f"; {COMPARED_8}: each in the recorded one" in lines
    gate = next(line for line in lines if line.strip().startswith("attention.gate_quantize ") and "calls" in line)
    assert gate.endswith("[band: gate_fp8]  PASS") and "0 the code of neither" in gate, gate
    norm = next(line for line in lines if line.strip().startswith("add_norm_quantize ") and "calls" in line)
    assert norm.endswith("[band: add_norm_quant]  PASS") and "residual stream equal" in norm and norm.split("; ")[-1].startswith("0 outside"), norm
    assert any(line.startswith("  blocks            8 of 8 provider blocks inside their bands") and line.endswith("PASS") for line in lines)
    served = next(line.strip() for line in lines if line.strip().startswith("add_norm_quantize, as the call log of a traced step has it (5 calls): "))
    assert [part.split(" x ")[0] for part in served.split(": ", 1)[1].split("; ")] == ["1", "2", "1", "1"], "per form, in the order of its first call"
    assert "2 x cudnn.AddNormQuantDecode #5 [PDL True, PDL_RELEASE True, ADD True, STORE_RESIDUAL False, BF16_OUT False, QUANT nvfp4]" in served
    written = json.loads(target.read_text())
    (run,) = written["runs"]
    assert written["stack"] == "native" and run["target"]["not_claimed"][0] == "fallback rows present" and run["target"]["numbers"]["target_port"] == 6
    norms = [site for site in run["sites"] if site["block"] == "add_norm_quantize"]
    assert [site["site"] for site in norms] == ["linear_attention.first", "mlp", "full_attention", "mlp", "output_head"]
    assert all(site["identity"] == "target" and site["served"] == site["form"] and site["instance"].startswith("cudnn.AddNormQuantDecode #") for site in norms)
    assert norms[1]["form"]["QUANT"] == "nvfp4" and norms[4]["form"] == {
        "PDL": True,
        "PDL_RELEASE": True,
        "ADD": True,
        "STORE_RESIDUAL": False,
        "BF16_OUT": True,
        "QUANT": None,
    }
    assert [site["identity"] for site in run["sites"] if site["block"] in ("embedding", "attention.gate_quantize", "mlp.down")] == [
        "target",
        "not yet (bf16 copy of 4-bit weights)",
        "target",
        "not yet (bf16 copy of 4-bit weights)",
    ]

    # the gate entry states and launches another grid than the census recorded: only the census sees it
    code, lines = _sample_run(monkeypatch, capsys, ["--stack", "native", "--sequences", "1"], gate_shape=((6, 1, 1), (256, 1, 1), 0))
    assert code == 1 and lines[-1] == "[00] FAIL", lines[-4:]
    differs = (
        "attention.gate_quantize (layer 1) launched gate_quant with grid (6, 1, 1), block (256, 1, 1), shared memory 0 B; "
        f"the {CENSUS} has grid (3, 1, 1), block (256, 1, 1), shared memory 0 B"
    )
    assert [line.split(": ", 2)[2] for line in lines if line.startswith("    kernels: step 0")] == [differs], "one kind of difference, in every traced step"
    assert not_claimed in lines, "a failed check says nothing about the recorded shapes"

    # one wrong byte of the gate per call: the block check fails, the kernel check holds
    code, lines = _sample_run(monkeypatch, capsys, ["--stack", "native", "--sequences", "1"], gate_defect=True)
    gate = next(line for line in lines if line.strip().startswith("attention.gate_quantize ") and "calls" in line)
    assert code == 1 and lines[-1] == "[00] FAIL" and gate.endswith("[band: gate_fp8]  FAIL") and _kernel_lines(lines)[0].endswith("    PASS"), gate

    # --provider: the fused step in its fe-cublas form.  The run holds; it is not the target set, and the line says so first
    selected = ["--stack", "native", "--sequences", "1", "--provider", "linear_attention.step=fe.gdn_decode_fused"]
    code, lines = _sample_run(monkeypatch, capsys, selected)
    assert code == 0 and lines[-1] == "[00] PASS", lines[-4:]
    assert (
        f"    {CENSUS}: NOT CLAIMED (not the target set: linear_attention.step selected by --provider; fallback rows present; the plan has no block for "
        "output_head.quantize); 7 launches of a step compared by launch shape (1 the target's own compiled kernel + 6 cuDNN Frontend entries with "
        "byte-equal results + 0 declared stand-ins): each in the recorded one"
    ) in lines
    assert any("not the target (the row's form is [PDL False, PDL_RELEASE False], the target's [PDL True, PDL_RELEASE True])" in line for line in lines)

    # no profiler: RAN, never PASS -- and the stack's name is in the line
    reason = "the profiler failed (no tracing library)"
    code, lines = _sample_run(monkeypatch, capsys, ["--stack", "native", "--sequences", "1"], profiler=reason)
    assert code == 0 and lines[-1] == f"[00] RAN native stack (kernels not observed: {reason})" and "[00] PASS" not in lines
    assert not_claimed + "; the launch shapes of its target launches were not observed" in lines

    # fe-entries at two sequences: the same rows; the census holds no 2-row section, so no identity is stated
    code, lines = _sample_run(monkeypatch, capsys, ["--stack", "fe-entries", "--sequences", "2"])
    assert code == 0 and lines[-1] == "[00] PASS" and _kernel_lines(lines)[0].endswith("    PASS"), lines[-4:]
    unstated = "no recorded section for 2 rows on cc 10.0 / 148 SMs: recorded is 1 row"
    assert f"  kernel identity: not stated for this run ({unstated})" in lines
    assert f"    {CENSUS}: NOT CLAIMED (fallback rows present; {unstated})" in lines
    invariance = next(line for line in lines if line.startswith("  batch invariance  "))
    assert "linear_attention.step equal (gated)" in invariance and "attention.gate_quantize equal (gated)" in invariance and invariance.endswith("PASS")
    assert "add_norm_quantize equal (gated)" in invariance, "a sequence alone writes the bytes it writes among two: the norm row gates it on this class"

    # the default stack is untouched by all of this: no kernel line without the flag
    code, lines = _sample_run(monkeypatch, capsys, ["--sequences", "1"])
    assert code == 0 and lines[-1] == "[00] PASS" and _kernel_lines(lines) == [] and not any("target" in line for line in lines)


def test_a_census_that_is_not_one_of_this_model_is_an_error_before_the_load(monkeypatch, capsys):
    """The census is read with the plan.  A file that does not follow the model's wiring, or is no census, ends the run in
    one ``ERROR`` line that names the file, before anything is loaded; a stack without a target never reads it."""
    from model_samples.qwen_dense import QwenDense

    def not_this_model(cls):
        raise ValueError(
            "kernel_census.json: the census does not follow this model's wiring: call 2 of a step is add_norm_quantize, the census has mlp.down there"
        )

    monkeypatch.setattr(QwenDense, "target", classmethod(not_this_model))
    code, out = _sample_main(monkeypatch, capsys, ["--stack", "native"])
    assert code == 1 and out.splitlines()[-1].startswith("[00] ERROR kernel_census.json: the census does not follow this model's wiring: call 2"), out
    with pytest.raises(AssertionError, match="reached a load"):
        _sample_main(monkeypatch, capsys, [])


def test_the_target_stacks_always_check_their_kernels(monkeypatch, capsys):
    from model_samples import Session, TextIO, loop

    sample = _sample_script(monkeypatch)
    assert sample.STACKS == loop.STACKS == ("fe-cublas", "native", "fe-entries", "reference")
    assert sample._arguments(["--stack", "native"]).stack == "native" and sample._arguments(["--stack", "fe-entries"]).stack == "fe-entries"
    with pytest.raises(SystemExit):
        sample._arguments(["--help"])
    shown = " ".join(capsys.readouterr().out.split())
    for part in (
        "native: the target kernel set of the decode step",
        "fe-entries: the cuDNN Frontend rows of native",
        "always on in the stacks native and fe-entries",
    ):
        assert part in shown, part
    model = _tiny_model()
    text = TextIO.synthetic(model.geometry.vocab)
    for stack, checks in (("native", True), ("fe-entries", True), ("fe-cublas", False), ("reference", False)):
        session = Session(model, text, stack=stack, device="cpu", capacity=8)
        assert session.check_kernels is checks, stack
    assert Session(model, text, stack="fe-cublas", device="cpu", capacity=8).claim() is None, "a stack without a target says nothing about one"
    with pytest.raises(AssertionError, match="reached a load"):  # through main(): the kernel table is printed without --check-kernels
        _sample_main(monkeypatch, capsys, ["--stack", "native", "--sequences", "1"])
    out = capsys.readouterr().out
    assert (
        "kernels, planned (decided before the load from model_samples/qwen_dense/configs.py; stack native" in out
        and "kernel identity, as planned: 8 of 20" in out
    )


def test_a_dry_plan_of_the_native_stack_imports_nothing_new():
    """In a fresh interpreter: planning the ``native`` stack on stated device facts, reading the census and stating the claim
    add only ``model_samples`` and standard-library modules -- the versions the census compares are read from the
    distributions' metadata, no package is imported for them."""
    code = f"""
import sys
import cudnn, torch
cudnn.GdnDecodeFused, cudnn.AttentionGateQuantDecode, cudnn.AddNormQuantDecode
import importlib.metadata
before = set(sys.modules)
sys.path.insert(0, {str(MODELS)!r})
from model_samples import census
from model_samples.blocks import DeviceFacts, plan
from model_samples.qwen_dense import QwenDense
model = QwenDense.reduced(2)
planned = plan(model.provider_table, model.regimes, DeviceFacts((10, 0), "stated", 148, 0, 0), 1, 1, "native", model=model, options={{"check_kernels": True}})
claim = model.target().claim(planned, model)
table = census.table(planned, claim)
added = sorted({{name.split(".")[0] for name in set(sys.modules) - before}})
print("ADDED", [name for name in added if name not in sys.stdlib_module_names and name not in ("model_samples", "cudnn")])
print("TABLE", "gate_quant" in table, "add_norm_quant" in table, "8 of 20" in table, claim.reasons[0])
"""
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "ADDED []" in done.stdout and "TABLE True True True fallback rows present" in done.stdout, done.stdout + done.stderr


# ----------------------------------------------------------------------------------------------- what the review of the stack found
def test_a_count_of_target_launches_in_the_readmes_carries_its_split():
    """A count of target launches is three numbers, never one, wherever it stands.  In the READMEs and the guide of this
    area every "N of the M launches of a step" -- and every count of launches "on target kernels" or of "target launches" --
    is followed by its split into the target's own compiled kernels and cuDNN Frontend entries."""
    import re

    counts = (
        r"\b\d+ of (?:the )?(?:484|20)\b(?! block calls)",  # of the launches of a step: the 64-layer and the two-layer model
        r"\b\d+ launches of every one of the \d+ traced steps on target kernels",
        r"\bthe \d+ (?:remaining )?target launch(?:es)?\b",
    )
    split = re.compile(r"\d+ (?:of )?the target's own compiled kernels? \+ \d+ (?:of a )?cuDNN Frontend entr(?:y|ies)|=\s*49 \+ 145\b")
    found = 0
    for name in ("README.md", "AGENTS.md", "qwen_dense/README.md"):
        text = " ".join((MODELS / name).read_text().split())
        for pattern in counts:
            for match in re.finditer(pattern, text):
                found += 1
                after = text[match.start() : match.end() + 260]
                assert split.search(after), f"{name}: {after[:120]!r} gives a count of target launches without its split"
    assert found >= 5, "the patterns still find the counts of the READMEs"
    area = " ".join((MODELS / "README.md").read_text().split())
    assert "193 of the 484 launches of a step of the 64-layer model, 48 the target's own compiled kernels + 145 cuDNN Frontend entries" in area
    assert "194 of the 484" not in area, "the three entries are 193 launches; the 194th is PyTorch's gather"
    for wrong in ("counts 3 of 20.", "(65 of the 484 launches of a step of the 64-layer model)"):  # the two sentences the review found
        assert not split.search(wrong + " " * 10), "the rule sees a count without its split"


def test_a_section_holds_for_the_layer_widths_it_was_recorded_with():
    """A census section is keyed by the GPU class, the SM count, the row count -- and the layer widths of the model: a
    checkpoint of the family with other widths has other grids (and, where an entry picks by width, other kernels).  Such
    a run is told that no identity is stated and is held to no recorded launch shape; the reduced default model keeps the
    published widths, so the section holds for it.  A launch shape that also depends on a field beyond the layer widths
    (the output head's: the vocabulary) names it, and is held only for a model that shares it."""
    import copy
    from dataclasses import replace
    from types import SimpleNamespace

    from model_samples import census
    from model_samples.blocks import DeviceFacts, plan as make_plan
    from model_samples.qwen_dense import Geometry, QwenDense

    target = QwenDense.target()
    section = target.data["configurations"]["plain"]["cc10.0/sm148"]
    full = Geometry()
    widths = ("hidden", "mlp", "la_key_heads", "la_value_heads", "la_head_dim", "attn_heads", "attn_kv_heads", "attn_head_dim")
    assert section["geometry"] == {name: getattr(full, name) for name in widths}, "the widths every recorded launch shape of a layer depends on"
    reduced = QwenDense.reduced(2).geometry
    assert all(getattr(reduced, name) == getattr(full, name) for name in widths) and reduced.vocab != full.vocab, "the reduced model keeps the widths"
    plan, model, _ = _target_plan(2)
    assert target.claim(plan, model, installed=RECORDED).known and len(target.claim(plan, model, installed=RECORDED).recorded) == 8

    stated = DeviceFacts((10, 0), "stated device", 148, 0, 0)  # stated facts: no entry is built for these widths
    other = QwenDense(replace(reduced, hidden=4096, la_key_heads=8, la_value_heads=32, attn_heads=16), ("other widths",))
    planned = make_plan(other.provider_table, other.regimes, stated, 1, 1, "native", model=other, options={"device": "cpu"})
    claim = target.claim(planned, other, installed=RECORDED)
    reason = (
        "no recorded section for this model's widths on cc 10.0 / 148 SMs: hidden 4096 (recorded 5120), la_key_heads 8 (recorded 16), "
        "la_value_heads 32 (recorded 48), attn_heads 16 (recorded 24)"
    )
    assert not claim.known and claim.identities == [] and claim.recorded == {}, "no identity, and no recorded launch shape is held against the run"
    assert claim.reasons == ["fallback rows present", reason]
    assert claim.coverage() == [f"kernel identity: not stated for this run ({reason})"]
    assert claim.line(census.PASS) == f"{CENSUS}: NOT CLAIMED (fallback rows present; {reason})"
    assert "identity" not in census.table(planned, claim).splitlines()[1]
    # the launches of such a run, in the shapes its own entries state, hold: nothing of the 27B's is asked of them
    sites = _sites()
    grids = {"gdn_decode_fused": ((64, 1, 1), (256, 1, 1), 42112)}
    sites[2].facts = census.Facts((SYMBOL,), {"PDL": False, "PDL_RELEASE": False}, grids["gdn_decode_fused"])
    step = _step(fused=(SYMBOL, grids["gdn_decode_fused"]))
    step[0] = step[0]._replace(grid=(32, 1, 1))  # the gather of a model 4,096 wide
    unheld = Claim = census.Claim(
        "kernel_census.json", "plain", "cc 10.0 / 148 SMs", 1, "2026-11-12", "recorded", False, [s.block for s in sites], [], reasons=[reason]
    )
    assert Claim.recorded == {} and census.Traced.of(sites, [], step, 1, "", rows=1, claim=unheld).judge() == (census.PASS, [])
    # with the wide model's row count missing as well, the widths are what is said: the model comes first
    four = make_plan(other.provider_table, other.regimes, stated, 4, 1, "native", model=other, options={"device": "cpu"})
    assert target.claim(four, other, installed=RECORDED).reasons == ["fallback rows present", reason]

    # a launch shape recorded with one more field of the geometry: held for a model that shares it, said for one that does not
    small_plan, small_model = _small_plan()
    data = _small_census()
    data["configurations"]["plain"]["cc10.0/sm148"]["rows"]["1"]["layer"][0][2]["at"] = {"vocab": 248320}
    apart = census.Target(copy.deepcopy(data)).claim(small_plan, small_model, installed=RECORDED)
    said = "no recorded launch shape of linear_attention.step for this model: vocab 272 (recorded 248320)"
    assert apart.known and apart.reasons == [said] and apart.recorded == {0: GATHER_SHAPE} and apart.identities[1].relation == census.TARGET
    assert apart.line(census.PASS).startswith(
        f"{CENSUS}: NOT CLAIMED ({said}); 1 launch of a step compared by launch shape (1 the target's own compiled kernel + "
    )
    assert apart.line(census.PASS).endswith(": each in the recorded one; 1 more by membership alone")
    shared = SimpleNamespace(geometry=SimpleNamespace(layer_kinds=("linear_attention",), hidden=5120, vocab=248320), wiring=small_model.wiring)
    together = census.Target(copy.deepcopy(data)).claim(small_plan, shared, installed=RECORDED)
    assert together.reasons == [] and together.recorded == {0: GATHER_SHAPE, 1: SHAPE}
    head = [entry for entry in section["rows"]["1"]["after_step"] if entry[0] == "output_head"]
    assert head[0][2]["at"] == {"vocab": full.vocab}, "the output head's launch shape is recorded with the vocabulary"
    # a section without its widths is no section; a claimed one names at least one
    for change, sentence in (
        (lambda s: s.pop("geometry"), "plain / cc10.0/sm148: a section holds geometry, the layer widths of the model it was taken on"),
        (lambda s: s.update(geometry={"hidden": "5120"}), "plain / cc10.0/sm148: a section holds geometry, the layer widths of the model it was taken on"),
        (lambda s: s.update(geometry={}), "plain / cc10.0/sm148: claimed without the layer widths it holds for"),
        (lambda s: s["rows"]["1"]["layer"][0][2].update(at={"vocab": "many"}), "at of linear_attention.step is not an object of geometry fields"),
    ):
        lacking = _small_census()
        change(lacking["configurations"]["plain"]["cc10.0/sm148"])
        with pytest.raises(ValueError) as refused:
            census.Target(lacking)
        assert str(refused.value).startswith("kernel_census.json: ") and sentence in str(refused.value), sentence


def test_the_checks_of_a_target_stack_look_at_its_kernels_unasked(monkeypatch):
    """Whether a run checks its kernels is the session's decision, read in one place.  A caller that follows the helper API
    -- ``Session(stack="native")``, ``trace``, ``run_checks`` -- without naming the kernel check gets it: the kernels line
    is printed and a launch under another symbol than the entry's fails the run.  The check of such a stack cannot be
    switched off."""
    from model_samples import census, tie_out

    other_form = [(SYMBOL, ((96, 1, 1), (256, 1, 1), 42112))]  # the symbol of the form WITHOUT wait and release; the entry states the target's

    def run(**stand_in):
        session = _recorded_run(monkeypatch, sequences=1, stack="native", check_kernels=False, **stand_in)[0]
        assert session.check_kernels is True, "a stack that declares a target kernel set checks its kernels, whatever it was told"
        session.trace(4)  # no argument: the session's decision
        return session

    checks = tie_out.run_checks(run(step_launches=other_form), prompt_tokens=3)  # as the next script of this stack would call it
    assert not checks.ok and checks.kernels == census.FAIL, checks.text()
    assert _line(checks, "kernels ").endswith("    FAIL")
    (sentence,) = [line.strip() for line in checks.lines if line.strip().startswith("kernels: step 0")]
    assert f"linear_attention.step (layer 0) launched {SYMBOL!r}; cudnn.GdnDecodeFused names {TARGET_SYMBOL!r} for this run" in sentence
    held = tie_out.run_checks(run(), prompt_tokens=3)
    assert held.ok and held.kernels == census.PASS and _line(held, "kernels ").endswith("    PASS"), held.text()
    assert any(f"{CENSUS}: NOT CLAIMED (" in line and COMPARED_8 in line for line in held.lines), "and the census line with what was compared"
    with pytest.raises(ValueError, match=r"^stack native declares a target kernel set: its kernel check cannot be switched off$"):
        tie_out.run_checks(run(), prompt_tokens=3, kernels=False)


def test_equal_says_what_was_compared_and_a_file_that_records_less_is_refused():
    """``EQUAL`` never stands for a comparison that was not made.  A section that says ``claimed`` records the launch shape
    of every target kernel -- only a kernel a library's tuner selects and a stand-in have none; an entry with a fact the
    reader does not know (a misspelt ``launch``) or with a ``launch`` that is no launch shape is refused in every section;
    and the census line says how many launches of a step were compared by launch shape, as three numbers, and how many
    by membership alone."""
    from model_samples import census

    plan, model = _small_plan()

    def step_facts(data):
        return data["configurations"]["plain"]["cc10.0/sm148"]["rows"]["1"]["layer"][0][2]

    def refused(data):
        with pytest.raises(ValueError) as error:
            census.Target(data)
        assert str(error.value).startswith("kernel_census.json: plain / cc10.0/sm148, 1 rows, layer: "), str(error.value)
        return str(error.value).split("layer: ", 1)[1]

    bare = _small_census()
    step_facts(bare).pop("launch")
    assert refused(bare) == (
        "claimed without the launch shape of linear_attention.step (gdn_decode_fused); only a kernel a library's tuner selects and a stand-in have none"
    )
    for claimed in (True, False):  # a misspelt key is a fact the reader does not know: never "nothing recorded"
        misspelt = _small_census(claimed=claimed)
        step_facts(misspelt)["lauch"] = step_facts(misspelt).pop("launch")
        assert refused(misspelt) == "linear_attention.step states lauch; an entry states launch and at and form"
    for bad in (
        [[96, 1], [256, 1, 1], 42112],
        [[96, 1, 1], [256, 1, 1]],
        [[96, 1, 1], [256, 1, 1], "42112"],
        [[96, 1, 1], [256, 1, 1], -1],
        [[0, 1, 1], [256, 1, 1], 0],
        [[96, 1, 1.0], [256, 1, 1], 0],
        [[True, 1, 1], [256, 1, 1], 0],
        {"grid": [96, 1, 1]},
    ):
        wrong = _small_census(claimed=False)
        step_facts(wrong)["launch"] = bad
        assert refused(wrong) == "the launch of linear_attention.step is not [grid of three, block of three, shared memory in bytes]", bad

    one = "1 launch of a step compared by launch shape (1 the target's own compiled kernel + 0 cuDNN Frontend entries with byte-equal results + 0 declared stand-ins)"
    # a section that claims nothing may record less: the launch is then held by membership alone, and the line says so
    loose = _small_census(claimed=False)
    step_facts(loose).pop("launch")
    claim = census.Target(loose).claim(plan, model, installed=RECORDED)
    assert claim.recorded == {0: GATHER_SHAPE} and claim.compared() == (1, one, 1)
    assert (
        claim.line(census.PASS)
        == f"{CENSUS}: NOT CLAIMED (the census does not claim this class yet); {one}: each in the recorded one; 1 more by membership alone"
    )
    # a kernel a library's tuner selects, in a section that says claimed: by membership, and EQUAL carries both counts
    tuned = _small_census()
    tuned["kernels"]["gdn_decode_fused"]["selection"] = "the library's tuner chooses the kernel"
    step_facts(tuned).pop("launch")
    claim = census.Target(tuned).claim(plan, model, installed=RECORDED)
    assert claim.reasons == [] and claim.line(census.PASS) == f"{CENSUS}: EQUAL ({one}, 1 by membership alone)"
    complete = census.Target(_small_census()).claim(plan, model, installed=RECORDED)
    assert complete.line(census.PASS) == f"{CENSUS}: EQUAL ({COMPARED_2}, 0 by membership alone)" and complete.compared() == (2, COMPARED_2, 0)
    # the wrong launch under such a claim: the fused step at half the recorded grid, which its entry states too -- NOT EQUAL
    sites = [site for site in _sites() if site.block in ("embedding", "linear_attention.step", "token_choice")]  # the calls of the small plan
    other = ((48, 1, 1), (256, 1, 1), 42112)
    sites[1].facts = census.Facts((SYMBOL,), {"PDL": False, "PDL_RELEASE": False}, other)
    at = _Clock()
    step = [at(GATHER, GATHER_SHAPE), at(SYMBOL, other), at(ARGMAX), at(HOST_READ, kind="copy")]
    assert census.compare([step], sites).ok, "without a census nothing is wrong with it"
    traced = census.Traced.of(sites, [], step, 1, "", rows=1, claim=complete)
    status, lines = traced.judge()
    assert status == census.FAIL and len(lines) == 1 and f"the {CENSUS} has grid (96, 1, 1)" in lines[0], lines
    assert traced.details()[-1] == f"{CENSUS}: NOT EQUAL (the kernel check did not hold)"


def test_the_first_use_of_an_entry_is_part_of_the_plan(monkeypatch):
    """An entry gates the installed toolchain where it builds its kernel.  On a device where a kernel can be built the
    adapter makes that first use while the run is planned, on scratch operands of the run's row count: a decline is then a
    ``NotImplementedError`` -- a SKIP before the load, in the entry's own words -- where the installation does not meet a
    requirement the adapter can name (the part's target by cuDNN Frontend's public probe; on compute capability 10.7 the
    CUDA 13 flavour of the CuTe DSL for the gate entry and for the norm entry), and a ``ValueError`` -- an ERROR -- for any
    other decline: those are the adapter's own operands, and a faulty adapter must not read as an unsupported GPU.  The
    norm entry is built once per form the run calls; its build takes no operand."""
    from types import SimpleNamespace

    import cudnn
    import cudnn.frost
    import cutlass.base_dsl.version_info as dsl_version

    from model_samples.blocks import DeviceFacts, plan
    from model_samples.providers import fe
    from model_samples.qwen_dense import QwenDense, configs

    flavour = "the attention gate + quantize decode step on compute capability 10.7 requires a CuTe DSL whose compiler is CUDA 13.4 or later"
    layout = "the fused step needs x rows contiguous with a 16-byte-multiple row stride"
    calls = []

    class Gate:
        decline = ""

        def __init__(self, device, width, cfg=None, *, pdl_release=None):
            self.width = width

        def compile(self, attn, gate, scale, out=None):
            calls.append(("gate compile", tuple(attn.shape), str(attn.device), tuple(out.shape)))
            if self.decline:
                raise NotImplementedError(self.decline)

        def __call__(self, attn, gate, scale, out=None):
            calls.append(("gate launch", tuple(attn.shape)))
            return out

        def kernel_symbols_for(self, rows):
            return (GATE_SYMBOL,)

        def form_for(self, rows):
            return {"THREADS": 256, "PDL_RELEASE": True}

        def launch_shape_for(self, rows):
            return ((3 * rows, 1, 1), (256, 1, 1), 0)

    class Step:
        decline = ""

        def __init__(self, device, eps, geometry=None, **knobs):
            pass

        def compile(self, rows, *operands):
            calls.append(("step compile", rows, tuple(operands[0].shape)))
            if self.decline:
                raise NotImplementedError(self.decline)

        def __call__(self, rows, *operands):
            calls.append(("step launch", rows))
            return operands[10]

        def kernel_symbols_for(self, rows):
            return (TARGET_SYMBOL,)

        def form_for(self, rows):
            return {"PDL": True, "PDL_RELEASE": True}

        def launch_shape_for(self, rows):
            return ((96, rows, 1), (256, 1, 1), 42112)

    class Norm:
        decline = ""

        def __init__(self, device, hidden, eps, *, add, bf16, quant, store_residual=False, pdl=None, pdl_release=None):
            self.form = (add, store_residual, bf16, quant)

        def compile(self, rows=None):
            calls.append(("norm compile", self.form))
            if self.decline:
                raise NotImplementedError(self.decline)

        def __call__(self, x, residual, weight, scale, out=None, codes=None, scales=None):
            calls.append(("norm launch", self.form, tuple(x.shape)))
            return out, codes, scales

        def kernel_symbols_for(self, rows):
            return (NORM_SYMBOL,)

        def form_for(self, rows):
            return dict(zip(("ADD", "STORE_RESIDUAL", "BF16_OUT", "QUANT"), self.form), PDL=True, PDL_RELEASE=True)

        def launch_shape_for(self, rows):
            return ((rows, 1, 1), (640, 1, 1), 80)

    monkeypatch.setattr(cudnn, "AttentionGateQuantDecode", Gate, raising=False)
    monkeypatch.setattr(cudnn, "GdnDecodeFused", Step, raising=False)
    monkeypatch.setattr(cudnn, "AddNormQuantDecode", Norm, raising=False)
    monkeypatch.setattr(fe, "_ENTRIES", {})
    monkeypatch.setattr(fe, "_FIRST_LAUNCH", {})
    monkeypatch.setattr(fe, "_outside_pool", lambda device, launch: (launch(), 0)[1])
    model = QwenDense.reduced(2)
    options = {"device": "cpu", "check_kernels": True}
    # the forms of the norm entry the two-layer model calls, in the order its row names their sites: (add, store, bf16, quant)
    norm_forms = [(False, True, True, "e4m3_static"), (True, False, False, "e4m3_static"), (True, False, False, "nvfp4"), (True, False, True, None)]
    norm_builds = [("norm compile", form) for form in norm_forms]
    assert fe.EntryBlock.first_use_devices == ("cuda",), "a kernel is built for a CUDA device only"

    def resolved(cc=(10, 0), sequences=4, total=180 * GIB):
        return plan(
            model.provider_table, model.regimes, DeviceFacts(cc, "stated device", 148, total, 80 * GIB), sequences, 1, "native", model=model, options=options
        )

    def planned(**run):
        """The table resolved and the first use of its entries made: what a session does once its memory sentence has held."""
        table = resolved(**run)
        table.first_use()
        return table

    planned()
    assert calls == [], "on a host device nothing is built while the run is planned"
    monkeypatch.setattr(fe.EntryBlock, "first_use_devices", ("cuda", "cpu"))  # this test's device: the stand-ins build nothing
    planned(total=0)
    assert calls == [], "nor on stated device facts"
    table = resolved()
    assert calls == [] and not table.first_used, "resolving the table builds nothing: the first use is a step of its own, after the memory sentence"
    table.first_use()
    assert calls == [
        *norm_builds,  # one build per form the run calls, none for a form of the row no call of this model is made in
        ("step compile", 4, (4, 10240)),
        ("gate compile", (4, 6144), "cpu", (4, 6144)),
    ], "the first use of each entry, at the run's row count"
    table.first_use()
    assert len(calls) == 6 and table.first_used, "made once per plan"
    del calls[:]
    table["linear_attention.step"].compile(4, 1)
    table["attention.gate_quantize"].compile(4, 1)
    assert [call[0] for call in calls] == [
        "step compile",
        "step launch",
        "gate compile",
        "gate launch",
    ], "after the load: the build again (the entry finds its kernel), one launch"
    del calls[:]
    table["add_norm_quantize"].compile(4, 1)
    assert calls == [
        event for form in norm_forms for event in (("norm compile", form), ("norm launch", form, (4, 5120)))
    ], "the same for every form of the norm entry"

    # (1) compute capability 10.7 with the CUDA 12 flavour of the CuTe DSL: the gate entry's own sentence, as a SKIP of the plan
    monkeypatch.setattr(Gate, "decline", flavour)
    monkeypatch.setattr(dsl_version, "CUDA_VERSION", SimpleNamespace(major=12, minor=9))
    monkeypatch.setattr(cudnn.frost, "cutedsl_runtime_available", lambda cc=None: True)
    with pytest.raises(NotImplementedError) as skipped:
        planned(cc=(10, 7))
    assert str(skipped.value) == flavour, "the entry's own words: they name the fix"
    # (2) the same decline where the adapter can name no unmet requirement: the adapter's operands, an ERROR -- never a SKIP
    monkeypatch.setattr(dsl_version, "CUDA_VERSION", SimpleNamespace(major=13, minor=4))
    for cc in ((10, 7), (10, 0)):
        with pytest.raises(ValueError) as wrong:
            planned(cc=cc)
        assert str(wrong.value) == f"cudnn.AttentionGateQuantDecode declined the operands this adapter built for its first use: {flavour}", cc
    monkeypatch.setattr(dsl_version, "CUDA_VERSION", SimpleNamespace(major=12, minor=9))
    with pytest.raises(ValueError, match=r"^cudnn.AttentionGateQuantDecode declined the operands this adapter built for its first use: "):
        planned(cc=(10, 3))  # either flavour serves 10.0 and 10.3: there the flavour explains nothing
    # (2b) the norm entry asks the same of the installation on compute capability 10.7, and is asked first: its own sentence is the
    # SKIP; with the CUDA 13.4 flavour, or on another class, the same decline is the adapter's -- an ERROR
    norm_flavour = "the add + RMSNorm + quantize decode step on compute capability 10.7 requires a CuTe DSL whose compiler is CUDA 13.4 or later"
    monkeypatch.setattr(Norm, "decline", norm_flavour)
    for compiler, cc, outcome in (
        ((12, 9), (10, 7), NotImplementedError),
        ((13, 0), (10, 7), NotImplementedError),
        ((13, 4), (10, 7), ValueError),
        ((12, 9), (10, 0), ValueError),
    ):
        monkeypatch.setattr(dsl_version, "CUDA_VERSION", SimpleNamespace(major=compiler[0], minor=compiler[1]))
        with pytest.raises(outcome) as declined:
            planned(cc=cc)
        own = (
            norm_flavour
            if outcome is NotImplementedError
            else f"cudnn.AddNormQuantDecode declined the operands this adapter built for its first use: {norm_flavour}"
        )
        assert str(declined.value) == own, (compiler, cc)
    monkeypatch.setattr(Norm, "decline", "")
    monkeypatch.setattr(dsl_version, "CUDA_VERSION", SimpleNamespace(major=12, minor=9))
    # (3) a CuTe DSL without the part's target (cuDNN Frontend's public probe): the fused step's decline is a SKIP too
    monkeypatch.setattr(Gate, "decline", "")
    monkeypatch.setattr(Step, "decline", "SM107 requires a CuTe DSL build supporting sm_107a; found 4.8.0 without that target")
    monkeypatch.setattr(cudnn.frost, "cutedsl_runtime_available", lambda cc=None: tuple(cc) != (10, 7))
    with pytest.raises(NotImplementedError, match=r"^SM107 requires a CuTe DSL build supporting sm_107a"):
        planned(cc=(10, 7))
    # (4) a layout decline of the fused step on an installation that serves the part: the adapter's operands
    monkeypatch.setattr(Step, "decline", layout)
    with pytest.raises(ValueError) as wrong:
        planned()
    assert str(wrong.value) == f"cudnn.GdnDecodeFused declined the operands this adapter built for its first use: {layout}"
    # (5) after the plan nothing skips: a decline at the first launch is an error of the run, and the sentence says whose it is
    monkeypatch.setattr(Step, "decline", "")
    table = planned()
    monkeypatch.setattr(Step, "decline", layout)
    with pytest.raises(RuntimeError, match=r"^cudnn.GdnDecodeFused declined the operands this adapter built for its first launch: the fused step needs"):
        table["linear_attention.step"].compile(4, 1)
    monkeypatch.setattr(cudnn.frost, "cutedsl_runtime_available", lambda cc=None: False)
    with pytest.raises(RuntimeError, match=r"^cudnn.GdnDecodeFused declines this installation at its first launch: the fused step needs"):
        table["linear_attention.step"].compile(4, 1)
    monkeypatch.setattr(cudnn.frost, "cutedsl_runtime_available", lambda cc=None: True)
    monkeypatch.setattr(Norm, "decline", "the add + RMSNorm + quantize decode step serves at most 1024 rows per call; got 4")
    with pytest.raises(RuntimeError, match=r"^cudnn.AddNormQuantDecode declined the operands this adapter built for its first launch: the add \+ RMSNorm"):
        table["add_norm_quantize"].compile(4, 1)
    for block in ("attention.gate_quantize", "add_norm_quantize"):
        row = configs.PROVIDERS[block][0]
        assert row.classes == (
            "cc10.x",
        ), f"{block}: the row claims every compute capability 10.x; what the entry needs there beyond the class is asked at plan time"


def test_a_run_that_does_not_fit_is_declined_before_an_entry_builds_anything(monkeypatch, capsys):
    """The first use of an entry builds its kernel on scratch operands as large as the run's own: at the 65,535 rows the row
    of the fused step admits, 192 GiB for the recurrent state alone.  So the memory sentence comes first.  On a real CUDA
    device a session resolves the tables of every sequence count, which allocates nothing, asks whether the whole run
    fits, and only then lets the entries make their first use -- still before anything is loaded.  A GPU that is too
    small gets the SKIP with the estimate on every provider stack, from the session and as the last line of the sample's
    ``main()``, and nothing was built for it.  A toolchain an entry declines stays a SKIP before the weights, also for a
    session that is loaded without its table."""
    import cudnn
    import cudnn.frost

    from model_samples import Session, TextIO, blocks
    from model_samples.providers import fe
    from model_samples.qwen_dense import QwenDense, configs

    events = []

    class Entry:
        """A stand-in entry: it builds nothing and says when it was asked to."""

        name, decline = "", ""

        def __init__(self, *arguments, **knobs):
            pass

        def compile(self, *operands):
            events.append(("build", self.name))
            if self.decline:
                raise NotImplementedError(self.decline)

        def kernel_symbols_for(self, rows):
            return (SYMBOL,)

        def form_for(self, rows):
            return {}

        def launch_shape_for(self, rows):
            return SHAPE

    class Step(Entry):
        name = "cudnn.GdnDecodeFused"

    class Gate(Entry):
        name = "cudnn.AttentionGateQuantDecode"

    class Norm(Entry):
        name = "cudnn.AddNormQuantDecode"

    def scratch(block, sequences, tokens):  # in place of the adapters' scratch operands: what would be allocated, and when
        events.append(("scratch", block.entry, sequences * tokens))
        return block._build(block.options["device"]), ()

    sentence = blocks.Plan.memory

    def memory(plan, need_bytes, what=""):
        events.append(("memory sentence", plan.stack))
        return sentence(plan, need_bytes, what)

    def no_weights(model, *arguments, **options):
        raise AssertionError("reached a load")

    gpu = {"total": 180 * GIB, "free": 1 * GIB}  # a real device, as it reports itself
    monkeypatch.setattr(cudnn, "GdnDecodeFused", Step, raising=False)
    monkeypatch.setattr(cudnn, "AttentionGateQuantDecode", Gate, raising=False)
    monkeypatch.setattr(cudnn, "AddNormQuantDecode", Norm, raising=False)
    monkeypatch.setattr(fe, "_ENTRIES", {})
    monkeypatch.setattr(fe, "_FIRST_LAUNCH", {})
    monkeypatch.setattr(fe.GdnDecodeFusedBlock, "_scratch", scratch)
    monkeypatch.setattr(fe.AttentionGateQuantDecodeBlock, "_scratch", scratch)
    monkeypatch.setattr(blocks.Plan, "memory", memory)
    monkeypatch.setattr(blocks.DeviceFacts, "probe", classmethod(lambda cls, device: cls((10, 0), "stated device", 148, gpu["total"], gpu["free"])))
    monkeypatch.setattr(QwenDense, "load", no_weights)
    model = QwenDense.reduced(2)
    text = TextIO.synthetic(model.geometry.vocab)
    state, conv = 65536 * 48 * 128 * 128 * 4, 65536 * 3 * 10240 * 2
    assert state == 192 * GIB and model.state_bytes(65535, 256) > state + conv, "the scratch state of a first use at 65,535 rows: one slot pool of the run"
    assert configs.PROVIDERS["linear_attention.step"][0].sequences == (1, 65535), "the row count the fused step's rows admit"

    def run(stack, *counts):
        del events[:]
        first = Session(model, text, sequences=counts[0], stack=stack, device="cuda:0")
        later = [Session(model, text, sequences=count, stack=stack, device="cuda:0", share=first) for count in counts[1:]]
        return first, first.plan_table(later)

    def norm_builds(stack):
        """The norm entry's first use in a stack that has its row: one build per form the two-layer model calls (four), in
        front of the other entries' -- and no scratch operand: its build takes none."""
        return [] if stack == "fe-cublas" else [("build", "cudnn.AddNormQuantDecode")] * 4

    def first_uses(stack, *counts):
        entries = ("cudnn.GdnDecodeFused",) if stack == "fe-cublas" else ("cudnn.GdnDecodeFused", "cudnn.AttentionGateQuantDecode")
        return [event for rows in counts for event in (*norm_builds(stack), *(e for entry in entries for e in (("scratch", entry, rows), ("build", entry))))]

    target = "SM100 requires a CuTe DSL build supporting sm_100a; found 4.8.0 without that target"
    assert configs.PROVIDER_STACKS == ("fe-cublas", "native", "fe-entries")
    for stack in configs.PROVIDER_STACKS:
        # (1) the run needs more than the GPU has in total: the estimate, and not one scratch operand
        gpu.update(total=180 * GIB, free=1 * GIB)
        with pytest.raises(NotImplementedError) as skipped:
            run(stack, 65535)
        assert str(skipped.value) == (
            f"stack {stack} needs about 263.5 GiB on this GPU; this GPU has 180.0 GiB in total: the run does not fit on it "
            "(CUDA_VISIBLE_DEVICES selects another GPU)"
        )
        assert events == [("memory sentence", stack)], f"{stack}: nothing is built for a run that does not fit"
        # (2) the default run where too little is free: the same, for both sequence counts of the run
        few = rf"^stack {stack} needs about 3\.7 GiB on this GPU for 1 and 4 sequences, one count after the other; 1\.0 GiB are free: free memory on this GPU"
        with pytest.raises(NotImplementedError, match=few):
            run(stack, 1, 4)
        assert events == [("memory sentence", stack)], stack
        del events[:]
        with monkeypatch.context() as scoped:  # the sample's own run on such a device; what it patches ends with it
            code, out = _sample_main(scoped, capsys, ["--stack", stack], total=180 * GIB, free=1 * GIB)
        assert code == 0 and out.splitlines()[-1] == (
            f"[00] SKIP stack {stack} needs about 3.7 GiB on this GPU for 1 and 4 sequences, one count after the other; 1.0 GiB are free: "
            "free memory on this GPU, or select another one with CUDA_VISIBLE_DEVICES"
        ), out
        assert events == [("memory sentence", stack)], f"{stack}: the sample's own run builds nothing either"
        # (3) with room: the memory sentence, then the first use of every entry for every sequence count, then nothing more
        gpu.update(free=80 * GIB)
        first, table = run(stack, 1, 4)
        assert "this stack needs about 3.7 GiB on this GPU for 1 and 4 sequences, one count after the other; 80.0 GiB are free" in table
        assert events == [("memory sentence", stack), *first_uses(stack, 1, 4)], f"{stack}: the budget first, then the builds"
        with pytest.raises(AssertionError, match="reached a load"):
            first.load()
        assert events == [("memory sentence", stack), *first_uses(stack, 1, 4)], "the load finds the first use made"
        # (4) a toolchain the entry declines, with room: a SKIP in the entry's own words before any weight is loaded -- from
        # the table, from the sample's last line, and from a session that is loaded without its table
        monkeypatch.setattr(Step, "decline", target)
        monkeypatch.setattr(cudnn.frost, "cutedsl_runtime_available", lambda cc=None: False)
        with pytest.raises(NotImplementedError) as skipped:
            run(stack, 1, 4)
        assert str(skipped.value) == target and events == [
            ("memory sentence", stack),
            *norm_builds(stack),
            ("scratch", "cudnn.GdnDecodeFused", 1),
            ("build", "cudnn.GdnDecodeFused"),
        ]
        with monkeypatch.context() as scoped:
            code, out = _sample_main(scoped, capsys, ["--stack", stack], total=180 * GIB, free=80 * GIB)
        assert code == 0 and out.splitlines()[-1] == f"[00] SKIP {target}", out
        del events[:]
        with pytest.raises(NotImplementedError) as skipped:
            Session(model, text, sequences=1, stack=stack, device="cuda:0").load()
        assert str(skipped.value) == target and events == [
            *norm_builds(stack),
            ("scratch", "cudnn.GdnDecodeFused", 1),
            ("build", "cudnn.GdnDecodeFused"),
        ], stack
        monkeypatch.setattr(Step, "decline", "")
        monkeypatch.setattr(cudnn.frost, "cutedsl_runtime_available", lambda cc=None: True)


# ----------------------------------------------------------------------------------------------- an entry that a step calls in several forms
def test_the_call_sites_of_a_decode_step_and_the_forms_a_row_names_for_them(monkeypatch):
    """The norm block is called at several sites of a step, and a kernel that serves it is called in another form at each.
    ``sites()`` names the site of every call of the wiring -- what the normed rows feed, and whether a residual stream
    exists yet -- ``step`` hands the same name to the block as its last operand, and a provider row names the form of
    each site: six forms, of which the 64-layer model calls five, 1 / 47 / 16 / 64 / 1 times a step."""
    from collections import Counter
    from types import SimpleNamespace

    from model_samples import census
    from model_samples.qwen_dense import QwenDense, configs

    full = QwenDense()
    wiring, sites = full.wiring(), full.sites()
    assert len(sites) == len(wiring) == 484 and census.call_sites(full) == sites
    named = [(block, layer, site) for (block, layer), site in zip(wiring, sites) if site]
    assert len(named) == 129 and all(block == "add_norm_quantize" for block, _, _ in named), "only the norm block is called in several forms"
    assert Counter(site for _, _, site in named) == {"linear_attention.first": 1, "linear_attention": 47, "full_attention": 16, "mlp": 64, "output_head": 1}
    assert named[:3] == [("add_norm_quantize", 0, "linear_attention.first"), ("add_norm_quantize", 0, "mlp"), ("add_norm_quantize", 1, "linear_attention")]
    assert named[6:8] == [("add_norm_quantize", 3, "full_attention"), ("add_norm_quantize", 3, "mlp")] and named[-1] == ("add_norm_quantize", 64, "output_head")
    assert [site for site in QwenDense.reduced(1).sites() if site] == [
        "full_attention.first",
        "mlp",
        "output_head",
    ], "a model that starts with an attention layer"
    from model_samples.qwen_dense import model as family

    assert {site.removesuffix(family.FIRST) for _, _, site in named} == set(family.NORM_SITES) and family.FIRST == ".first"
    assert [site for site in QwenDense.reduced(2).sites() if site] == ["linear_attention.first", "mlp", "full_attention", "mlp", "output_head"]
    bare = SimpleNamespace(wiring=lambda: (("embedding", -1), ("token_choice", -1)))
    assert census.call_sites(bare) == ("", ""), "a model that names no call sites: every block in one form"
    with pytest.raises(ValueError, match="^the model names 1 call sites for the 2 calls of its wiring$"):
        census.call_sites(SimpleNamespace(wiring=bare.wiring, sites=lambda: ("",)))

    # what the step hands the block is the site of that call: seen on a step of the reference stack
    session, _ = _tiny_session(1)
    seen = []

    def tap(name, layer, block, operands):
        seen.append((name, layer, operands[4] if name == "add_norm_quantize" else ""))
        return block(*operands)

    session.tap = tap
    session._advance(torch.zeros(1, dtype=torch.int32))
    assert seen == [(block, layer, site) for (block, layer), site in zip(session.model.wiring(), session.model.sites())]
    assert [site for _, _, site in seen if site] == ["linear_attention.first", "mlp", "full_attention", "mlp", "output_head"]

    # the row: the launch form every call has, and per call site the residual mode and the outputs
    row = configs.PROVIDERS["add_norm_quantize"][0]
    assert row.key == "fe.add_norm_quant_decode" and row.form == (("PDL", True), ("PDL_RELEASE", True)) and row.sites == configs.NORM_FORMS
    assert row.classes == ("cc10.x",) and row.rows == (1, 1024) and row.sequences == (1, 1024) and row.band == "add_norm_quant"
    assert set(row.stacks) == {"native", "fe-entries"} and row.batch_invariant == ("cc10.0",) and not row.fallback and row.ci
    assert configs.PROVIDERS["add_norm_quantize"][1].key == "torch_ref.add_norm_quantize", "the reference chain stays the row of every other class and stack"
    assert {site for site, _ in row.sites} >= {site for _, _, site in named} and len(row.sites) == len({fields for _, fields in row.sites}) == 6
    for site, fields in row.sites:
        assert [name for name, _ in fields] == ["ADD", "STORE_RESIDUAL", "BF16_OUT", "QUANT"], f"{site}: the entry has no default for these"
        assert dict(fields)["ADD"] != dict(fields)["STORE_RESIDUAL"] and dict(fields)["STORE_RESIDUAL"] == site.endswith(".first"), site
    assert row.site_form("mlp") == (("PDL", True), ("PDL_RELEASE", True), ("ADD", True), ("STORE_RESIDUAL", False), ("BF16_OUT", False), ("QUANT", "nvfp4"))
    assert dict(row.site_form("output_head"))["QUANT"] is None and dict(row.site_form("linear_attention"))["BF16_OUT"] is True
    with pytest.raises(
        ValueError, match=r"^provider row fe.add_norm_quant_decode: no form for the call site 'mlp.gate'; the row names linear_attention.first, "
    ):
        row.site_form("mlp.gate")
    with pytest.raises(ValueError, match=r"no form for the call site ''"):
        row.site_form()
    plain = configs.PROVIDERS["linear_attention.step"][1]
    assert plain.sites == () and plain.site_form() == plain.site_form("any site") == plain.form, "a row that names no sites is called in its one form"
    assert configs.readme_sites(full, "native").splitlines()[2:] == [
        "| residual add + RMSNorm + quantise | `linear_attention.first` | 1 | [ADD False, STORE_RESIDUAL True, BF16_OUT True, QUANT e4m3_static] |",
        "| residual add + RMSNorm + quantise | `full_attention.first` | 0 | [ADD False, STORE_RESIDUAL True, BF16_OUT False, QUANT e4m3_static] |",
        "| residual add + RMSNorm + quantise | `linear_attention` | 47 | [ADD True, STORE_RESIDUAL False, BF16_OUT True, QUANT e4m3_static] |",
        "| residual add + RMSNorm + quantise | `full_attention` | 16 | [ADD True, STORE_RESIDUAL False, BF16_OUT False, QUANT e4m3_static] |",
        "| residual add + RMSNorm + quantise | `mlp` | 64 | [ADD True, STORE_RESIDUAL False, BF16_OUT False, QUANT nvfp4] |",
        "| residual add + RMSNorm + quantise | `output_head` | 1 | [ADD True, STORE_RESIDUAL False, BF16_OUT True, QUANT None] |",
    ]


def test_a_call_served_in_the_form_of_another_call_site_fails_the_form_check():
    """Item 0 per call site, on the plan of the 64-layer model with the real adapter on stated device facts.  Every call of
    the norm block carries the form of ITS site and the facts of the entry instance of that form; the forms cannot be told
    apart by a launch record (one launch shape, one kernel id).  A call log in which one call was served by the instance
    of a neighbouring form fails with a sentence that names the call, its site and both forms -- also where it is the
    other norm call of the same layer."""
    from collections import Counter
    from dataclasses import replace

    from model_samples import census

    plan, model, _ = _target_plan(64)
    sites = census.expected(plan, model)
    norms = [(index, site) for index, site in enumerate(sites) if site.block == "add_norm_quantize"]
    assert Counter(site.site for _, site in norms) == {"linear_attention.first": 1, "linear_attention": 47, "full_attention": 16, "mlp": 64, "output_head": 1}
    row = plan.rows["add_norm_quantize"]
    for _, site in norms:
        assert site.form == row.site_form(site.site) and site.wrong == "" and tuple(site.facts.shape) == NORM_SHAPE, site.site
        assert all(site.facts.form[name] == value for name, value in site.form) and site.facts.form["HIDDEN"] == 5120, "the entry instance of that form"
        assert site.row == f"add_norm_quantize at {site.site}" and site.name == f"add_norm_quantize (layer {site.layer})"
    assert len({site.form for _, site in norms}) == 5 and len({tuple(site.facts.shape) for _, site in norms}) == 1, "five forms, one launch shape"
    assert [site.site for site in sites if site.block != "add_norm_quantize"] == [""] * 355 and sites[3].row == "linear_attention.step"
    assert census.check_forms(sites) == []
    clean = [census.Call(site.block, site.layer, 1, "an entry instance", site.form) for site in sites]
    assert census.check_forms(sites, [clean, clean]) == [], "every call served in the form of its site"

    at = {(site.layer, site.site): index for index, site in norms}
    mlp5, input5, input3 = at[(5, "mlp")], at[(5, "linear_attention")], at[(3, "full_attention")]
    eight_bit = "[PDL True, PDL_RELEASE True, ADD True, STORE_RESIDUAL False, BF16_OUT False, QUANT e4m3_static]"
    four_bit = "[PDL True, PDL_RELEASE True, ADD True, STORE_RESIDUAL False, BF16_OUT False, QUANT nvfp4]"
    wrong = list(clean)
    wrong[mlp5] = census.Call("add_norm_quantize", 5, 1, "cudnn.AddNormQuantDecode #3", sites[input3].form)  # the form of an attention layer's input norm
    sentence = (
        f"add_norm_quantize (layer 5), the call at mlp: the adapter served the call with the form {eight_bit} (cudnn.AddNormQuantDecode #3); "
        f"the row says {four_bit}"
    )
    assert census.check_forms(sites, [wrong, wrong]) == [sentence], "one sentence per kind of difference"
    swapped = list(clean)  # the two norm calls of one layer, each served in the other's form: both are named, neither hides the other
    swapped[mlp5], swapped[input5] = clean[input5], clean[mlp5]
    found = census.check_forms(sites, [swapped])
    assert len(found) == 2 and found[0].startswith("add_norm_quantize (layer 5), the call at linear_attention: the adapter served the call with the form")
    assert found[1].startswith("add_norm_quantize (layer 5), the call at mlp: ") and found[1].endswith(f"the row says {four_bit}")
    unserved = list(clean)
    unserved[mlp5] = census.Call("add_norm_quantize", 5, 1, "", ())
    assert census.check_forms(sites, [unserved]) == [
        f"add_norm_quantize (layer 5), the call at mlp: the adapter served the call with the form [] (no entry instance); the row says {four_bit}"
    ]
    # a row whose form is not the entry's at one site: named with the site, once
    other = [replace(site, form=(("PDL", False), *site.form[1:])) if site.site == "mlp" else site for site in sites]
    assert census.check_forms(other) == ["add_norm_quantize at mlp: the row's form says PDL False, cudnn.AddNormQuantDecode states PDL True for this run"]
    # a call site the row has no form for: a wrong table, said on the site
    lacking = replace(row, sites=tuple(entry for entry in row.sites if entry[0] != "output_head"))
    plan.rows["add_norm_quantize"] = lacking
    missing = [site for site in census.expected(plan, model) if site.site == "output_head"]
    assert (
        len(missing) == 1
        and missing[0].facts is None
        and missing[0].wrong.startswith("provider row fe.add_norm_quant_decode: no form for the call site 'output_head'")
    )
    assert census.check_forms(missing)[0].startswith("add_norm_quantize at output_head: cudnn.AddNormQuantDecode states no launch for this run (provider row")


def test_the_norm_entrys_adapter_over_a_stand_in_entry(monkeypatch):
    """``fe.add_norm_quant_decode`` over a stand-in entry.  One adapter, one entry instance per form: the form of a call's
    site reaches the constructor as the entry's own arguments, a call is served by the instance of its site and by no
    other, and the call log reads the instance that was called.  The residual stream is the kernel's to update in place;
    everything else it writes goes into buffers of this call's own.  A consumer on a fallback row gets the kernel's bytes
    as they are -- the block scales in the kernel's layout, the bf16 rows unquantised where the form writes no codes --
    and nothing but the entry is launched.  A call whose operands do not fit the form of its site, a form the adapter
    cannot pass, and an entry that declines the width or the row count are wrong tables: ``ValueError``."""
    from dataclasses import replace

    import cudnn

    from model_samples import census
    from model_samples.blocks import DeviceFacts, plan
    from model_samples.providers import Quant, fe, torch_ref
    from model_samples.qwen_dense import QwenDense, configs
    from model_samples.weights import ScaledTensor, TiledBlockScaledTensor, Unquantized

    built = []

    class Entry:
        limit = 1024

        def __init__(self, device, hidden, eps, *, add, bf16, quant, store_residual=False, pdl=None, pdl_release=None):
            if hidden != 5120:
                raise NotImplementedError(f"this stand-in serves hidden 5120; got hidden {hidden}")
            self.arguments = dict(hidden=hidden, eps=eps, add=add, bf16=bf16, quant=quant, store_residual=store_residual, pdl=pdl, pdl_release=pdl_release)
            self.calls = []
            built.append(self)

        def compile(self, rows=None):
            self.calls.append(("compile", rows))

        def __call__(self, x, residual, weight, scale, out=None, codes=None, scales=None):
            self.calls.append(("launch", tuple(x.shape), residual, scale, out, codes, scales))
            mark = len(self.calls)
            if self.arguments["add"]:
                residual.copy_((residual.float() + x.float()).to(torch.bfloat16))  # in place, as the kernel does
            elif self.arguments["store_residual"]:
                residual.copy_(x)
            for written in (out, codes, scales):
                if written is not None:
                    written.view(torch.uint8).fill_(mark)
            return out, codes, scales

        def kernel_symbols_for(self, rows):
            return (_norm_symbol(self.form_for(rows)).replace("h64", "h5120"),)

        def form_for(self, rows):
            a = self.arguments
            return dict(
                HIDDEN=a["hidden"], EPS=a["eps"], ADD=a["add"], STORE_RESIDUAL=a["store_residual"], BF16_OUT=a["bf16"], QUANT=a["quant"], PDL=bool(a["pdl"]),
                PDL_RELEASE=bool(a["pdl"]) if a["pdl_release"] is None else a["pdl_release"],
            )  # fmt: skip

        def launch_shape_for(self, rows):
            if rows > self.limit:
                raise NotImplementedError(f"this stand-in serves at most {self.limit} rows per call; got {rows}")
            return ((rows, 1, 1), (640, 1, 1), 80)

    monkeypatch.setattr(cudnn, "AddNormQuantDecode", Entry, raising=False)
    monkeypatch.setattr(fe, "_ENTRIES", {})
    model = QwenDense.reduced(2)
    row = configs.PROVIDERS["add_norm_quantize"][0]
    stated = DeviceFacts((10, 0), "stated device", 148, 0, 0)
    options = {"device": "cpu", "check_kernels": True}
    block = fe.AddNormQuantDecodeBlock(row, model, stated, options)
    block.check_support(1, 1)
    assert built == [] and block.serving() == ("cudnn.AddNormQuantDecode", ()), "on stated facts nothing is built; before a call nothing has served"
    assert block.kernels == (census.Kernel("add_norm_quant", "cuDNN FE", "exact", launches=1, shape="entry"),) and block.state_args == (1,)
    assert block._sites() == ("linear_attention.first", "full_attention", "mlp", "output_head"), "the forms the two-layer model calls, in the row's order"
    facts = block.launch_facts(1, "mlp")
    assert len(built) == 1 and built[0].arguments == dict(
        hidden=5120, eps=1e-6, add=True, bf16=False, quant="nvfp4", store_residual=False, pdl=True, pdl_release=True
    ), "the form of the call site, as the entry's own arguments"
    assert (
        facts.shape == NORM_SHAPE
        and facts.form["QUANT"] == "nvfp4"
        and facts.symbols == ("cudnn_kernel_frost_add_rms_norm_quant_h5120_add_nvfp4_wait_release_eps358637bd_0",)
    )
    assert block.launch_facts(4, "mlp").shape == ((4, 1, 1), (640, 1, 1), 80) and len(built) == 1, "one instance per form, whatever the row count"

    x = torch.ones((2, 5120), dtype=torch.bfloat16)
    weight = torch.zeros(5120, dtype=torch.bfloat16)
    eight, four = Quant("fp8", 0.0625), Quant("nvfp4", 0.03125)
    # the first call of a step: no residual stream yet -- the form stores the operand, into a tensor of this call's own
    q1, normed1, residual = block(x, None, weight, eight, "linear_attention.first")
    first = built[1]
    assert (
        first.arguments["add"] is False
        and first.arguments["store_residual"] is True
        and first.arguments["bf16"] is True
        and first.arguments["quant"] == "e4m3_static"
    )
    _, shape, stored, scale1, out1, codes1, scales1 = first.calls[-1]
    assert shape == (2, 5120) and stored is residual and residual is not x and torch.equal(residual, x) and scales1 is None
    assert isinstance(q1, ScaledTensor) and q1.scale == 0.0625 and q1.codes.dtype == torch.uint8 and q1.codes.data_ptr() == codes1.data_ptr()
    assert codes1.dtype == torch.float8_e4m3fn and tuple(codes1.shape) == (2, 5120) and normed1 is out1 and out1.dtype == torch.bfloat16
    assert scale1.dtype == torch.float32 and scale1.tolist() == [0.0625], "8-bit codes: the consumer's static scale, as it is"
    assert block.serving() == ("cudnn.AddNormQuantDecode #2", row.site_form("linear_attention.first")), "the instance that was called, and what it states"
    # the norm in front of an MLP: the residual stream is updated in place; 4-bit codes with their block scales in the kernel's layout
    q2, normed2, same = block(x, residual, weight, four, "mlp")
    _, _, updated, scale2, out2, codes2, scales2 = built[0].calls[-1]
    assert same is residual and updated is residual and bool((residual == 2.0).all()), "in place: the stream is the kernel's to update"
    assert normed2 is None and out2 is None and isinstance(q2, TiledBlockScaledTensor) and q2.scale == 0.03125
    assert q2.codes is codes2 and codes2.dtype == torch.uint8 and tuple(codes2.shape) == (2, 2560)
    assert tuple(scales2.shape) == (128, 320) and scales2.dtype == torch.float8_e4m3fn and q2.tiled_scales.data_ptr() == scales2.data_ptr()
    assert scale2.tolist() == [32.0] and scale2.dtype == torch.float32, "4-bit codes: the global scale a consumer divides by, the fp32 reciprocal"
    assert block.serving() == ("cudnn.AddNormQuantDecode #1", row.site_form("mlp"))
    q3, _, _ = block(x, residual, weight, four, "mlp")
    assert (
        built[0].calls[-1][3] is scale2 and q3.codes is not q2.codes and int(q2.codes[0, 0]) != int(q3.codes[0, 0])
    ), "one scale tensor per value; buffers of its own per call"
    assert tuple(block.compared((q2, None, residual))[0].tensors()[1].shape) == (2, 320), "compared row by row: the block scales gathered out of the layout"
    # the final norm: the form ends at the bf16 rows; the 4-bit operand of the consumer is handed over unquantised
    q4, normed4, _ = block(x, residual, weight, four, "output_head")
    head = built[2]
    assert head.arguments["bf16"] is True and head.arguments["quant"] is None and head.calls[-1][3] is None and head.calls[-1][5] is None
    assert isinstance(q4, Unquantized) and q4.data is normed4 and (q4.fmt, q4.scale) == ("nvfp4", 0.03125)
    assert block.compared((q4, normed4, residual)) == (None, normed4, residual), "what the consumer quantises is not the kernel's output"
    assert block(x, residual, weight, None, "output_head")[0] is None, "no quantised operand asked for: none handed over"
    assert block.serving() == ("cudnn.AddNormQuantDecode #3", row.site_form("output_head")) and len(built) == 3, "three forms called so far: three instances"
    assert (
        fe.AddNormQuantDecodeBlock(row, model, stated, options).launch_facts(1, "mlp") == facts and len(built) == 3
    ), "one entry per form, shared by every session"
    block.compile(4, 1)
    assert len(built) == 4, "at load: every form the model calls (the fourth is the attention layer's input norm)"
    assert [call[:2] for call in built[3].calls] == [("compile", None), ("launch", (4, 5120))] and built[3].arguments["quant"] == "e4m3_static"
    assert [call[0] for call in built[0].calls[-2:]] == ["compile", "launch"] and built[0].calls[-1][1] == (
        4,
        5120,
    ), "compiled and launched once per form at load"
    reference = torch_ref.AddNormQuantize(configs.PROVIDERS["add_norm_quantize"][1], model, stated, options)
    twin = block.twin(reference)
    assert type(twin) is torch_ref.AddNormQuantizeBracket and block.twin(reference) is twin and reference.twin(reference) is reference
    assert twin.forms == {site: dict(row.site_form(site)) for site, _ in row.sites}, "the twin writes what the form of each call site writes"
    for name in ("norm/frost/add_norm_quant_decode.py", "norm/frost/kernel/add_norm_quant_decode_config.py", "norm/frost/kernel/add_norm_quant_decode_f16.py"):
        assert name in fe.SOURCE_FILES and (Path(cudnn.__file__).parent / name).is_file(), f"the source hash of a run covers {name}"

    # a call whose operands are not what the form of its site takes: the table or the wiring is wrong, and the adapter decides nothing
    for arguments, sentence in (
        ((x, None, weight, four, "mlp"), "its form at mlp is ADD True, STORE_RESIDUAL False; the step has no residual stream yet there"),
        (
            (x, residual, weight, eight, "linear_attention.first"),
            "its form at linear_attention.first is ADD False, STORE_RESIDUAL True; the step has a residual stream there",
        ),
        ((x, residual, weight, eight, "mlp"), "its form at mlp writes nvfp4 codes; the step asks for a fp8 operand there"),
        ((x, residual, weight, None, "mlp"), "its form at mlp writes nvfp4 codes; the step asks for no quantised operand there"),
        ((x, residual, weight, four, "full_attention"), "its form at full_attention writes e4m3_static codes; the step asks for a nvfp4 operand there"),
    ):
        with pytest.raises(ValueError) as wrong:
            block(*arguments)
        assert str(wrong.value) == f"provider row fe.add_norm_quant_decode: {sentence}", sentence
    with pytest.raises(ValueError, match="cudnn.AddNormQuantDecode writes static 8-bit or 4-bit codes; this block is asked for 'int8'"):
        block(x, residual, weight, Quant("int8", 1.0), "mlp")
    with pytest.raises(ValueError, match=r"^provider row fe.add_norm_quant_decode: no form for the call site 'mlp.gate'"):
        block(x, residual, weight, four, "mlp.gate")
    named = replace(row, form=(("PDL", True), ("THREADS", 640)))
    with pytest.raises(
        ValueError, match=r"its form at linear_attention.first names THREADS; this adapter passes ADD, BF16_OUT, QUANT, STORE_RESIDUAL, PDL, PDL_RELEASE"
    ):
        fe.AddNormQuantDecodeBlock(named, model, stated, options).check_support(1, 1)
    short = replace(row, sites=(("mlp", (("ADD", True), ("BF16_OUT", False))),))
    with pytest.raises(ValueError, match=r"its form at mlp does not name QUANT; .* which has no default for ADD, BF16_OUT, QUANT$"):
        fe.AddNormQuantDecodeBlock(short, model, stated, options).check_support(1, 1)

    # on a real device the adapter asks the instance of every form the run calls for the launch of its row count
    real = DeviceFacts((10, 0), "stated device", 148, 180 * GIB, 80 * GIB)
    fe.AddNormQuantDecodeBlock(row, model, real, options).check_support(4, 1)
    monkeypatch.setattr(Entry, "limit", 3)
    declined = (
        r"^the provider table is wrong: the row of fe.add_norm_quant_decode claims this run \(4 rows on compute capability 10.0\), and "
        r"cudnn.AddNormQuantDecode declines it: this stand-in serves at most 3 rows per call; got 4$"
    )
    with pytest.raises(ValueError, match=declined):
        fe.AddNormQuantDecodeBlock(row, model, real, options).check_support(4, 1)
    with pytest.raises(ValueError, match=declined):  # through the plan: a ValueError, which a sample prints as an ERROR
        plan(dict(model.provider_table), model.regimes, real, 4, 1, "native", model=model, options=options)
    monkeypatch.setattr(fe, "_ENTRIES", {})
    narrow = QwenDense.reduced(2)
    narrow.geometry = replace(narrow.geometry, hidden=4096)
    with pytest.raises(ValueError) as wrong:
        fe.AddNormQuantDecodeBlock(row, narrow, real, options).check_support(1, 1)
    assert str(wrong.value) == (
        "the provider table is wrong: cudnn.AddNormQuantDecode declines the geometry or the form of the row of fe.add_norm_quant_decode at "
        "linear_attention.first: this stand-in serves hidden 5120; got hidden 4096"
    )

    class Older:  # an installed package without the entry, and one whose entry states no launch facts
        __version__ = "1.32.3"

        def __getattr__(self, name):
            raise AttributeError(name)

    monkeypatch.setattr(fe.AddNormQuantDecodeBlock, "_cudnn", lambda self: Older())
    with pytest.raises(NotImplementedError, match=r"needs nvidia-cudnn-frontend with the add \+ RMSNorm \+ quantise decode step \(>= 1.33.0\); this is 1.32.3"):
        fe.AddNormQuantDecodeBlock(row, model, stated, options).check_support(1, 1)
    monkeypatch.undo()
    monkeypatch.setattr(cudnn, "AddNormQuantDecode", type("NoFacts", (), {}), raising=False)
    with pytest.raises(NotImplementedError, match=r"the kernel check needs the launch facts of cudnn.AddNormQuantDecode \(kernel_symbols_for"):
        fe.AddNormQuantDecodeBlock(row, model, stated, options).check_support(1, 1)


def test_a_census_names_the_form_of_every_call_of_a_kernel_with_several_forms():
    """The target calls the norm kernel in another form at each call site, and the census says which: per entry of a section
    the fields of that call (``form``), and what differs at a block's first call in the model's first layer
    (``first_layer``).  A block call is the target's only where the row's form at ITS site is the census's form of THAT
    call; a row that differs at one site is ``not the target`` there and nowhere else, and the reason is said once."""
    import copy

    from model_samples import census
    from model_samples.qwen_dense import QwenDense

    target = QwenDense.target()
    parts = target.data["configurations"]["plain"]["cc10.0/sm148"]["rows"]["1"]
    kinds = QwenDense().geometry.layer_kinds
    calls = [facts for block, _, facts in target.sequence(parts, kinds) if block == "add_norm_quantize"]
    first, later = calls[0]["form"], calls[2]["form"]
    assert first == {"ADD": False, "STORE_RESIDUAL": True, "BF16_OUT": True, "QUANT": "e4m3_static"}, "the first layer's facts, merged field by field"
    assert later == {"ADD": True, "STORE_RESIDUAL": False, "BF16_OUT": True, "QUANT": "e4m3_static"} and calls[-1]["form"]["QUANT"] is None
    assert parts["linear_attention_layer"][0][2]["form"]["ADD"] is True, "the file's own entry is left as it is"
    assert target.sequence(parts, kinds) == target.sequence(parts, kinds) and target.numbers(parts, kinds) == (484, 210, 273, 1)
    attention_first = [facts["form"] for block, _, facts in target.sequence(parts, QwenDense.reduced(1).geometry.layer_kinds) if block == "add_norm_quantize"]
    assert attention_first[0] == {"ADD": False, "STORE_RESIDUAL": True, "BF16_OUT": False, "QUANT": "e4m3_static"}, "whatever kind the first layer is"
    plan1, model1, _ = _target_plan(1)
    assert target.claim(plan1, model1, installed=RECORDED).of("add_norm_quantize") == "target", "and the row has that form at that site"

    plan, model, _ = _target_plan(64)

    def changed(change):
        data = copy.deepcopy(target.data)
        change(data["configurations"]["plain"]["cc10.0/sm148"]["rows"]["1"])
        return census.Target(data)

    def with_normed_rows(rows):  # the census says the norm in front of an MLP also writes its bf16 rows
        for kind in ("linear_attention_layer", "attention_layer"):
            rows[kind][-3][2]["form"]["BF16_OUT"] = True

    claim = changed(with_normed_rows).claim(plan, model, installed=RECORDED)
    numbers = claim.numbers()
    row_form = "[PDL True, PDL_RELEASE True, ADD True, STORE_RESIDUAL False, BF16_OUT False, QUANT nvfp4]"
    reason = f"the row's form at mlp is {row_form}, the target's {row_form.replace('BF16_OUT False', 'BF16_OUT True')}"
    assert numbers["not_the_target"] == {"add_norm_quantize": 64} and (numbers["target_code"], numbers["target_port"]) == (49, 81)
    assert claim.reasons == [f"not the target set: add_norm_quantize ({reason})", "fallback rows present", "the plan has no block for output_head.quantize"][
        1:2
    ] + [
        f"not the target set: add_norm_quantize ({reason})",
        "the plan has no block for output_head.quantize",
    ], "said once for its 64 calls"
    assert (
        claim.of("add_norm_quantize") == f"target; not the target ({reason})" and len(claim.recorded) == 194 - 64
    ), "no recorded shape is held against those calls"
    unmerged = changed(lambda rows: rows.pop(census.FIRST_LAYER)).claim(plan, model, installed=RECORDED)
    assert unmerged.numbers()["not_the_target"] == {"add_norm_quantize": 1} and unmerged.identities[1].reason.startswith(
        "the row's form at linear_attention.first is [PDL True, PDL_RELEASE True, ADD False, STORE_RESIDUAL True, "
    ), "without it the census would say that the first norm of a step adds to a residual"

    with pytest.raises(ValueError, match="^kernel_census.json: first_layer names attention.core, which the first layer of this model does not call$"):
        changed(lambda rows: rows[census.FIRST_LAYER].update({"attention.core": {"form": {}}})).claim(plan, model, installed=RECORDED)
    for change, sentence in (
        (lambda rows: rows["epilogue"][0][2].update(form=["BF16_OUT"]), "1 rows, epilogue: the form of add_norm_quantize is not an object of form fields"),
        (
            lambda rows: rows.update({census.FIRST_LAYER: [["add_norm_quantize", {}]]}),
            "1 rows: first_layer is not an object of block: facts (launch and at and form)",
        ),
        (
            lambda rows: rows[census.FIRST_LAYER]["add_norm_quantize"].update(forms={}),
            "1 rows: first_layer is not an object of block: facts (launch and at and form)",
        ),
        (lambda rows: rows[census.FIRST_LAYER]["add_norm_quantize"].update(form="first"), "first_layer: the form of add_norm_quantize is not an object"),
    ):
        with pytest.raises(ValueError) as refused:
            changed(change)
        assert str(refused.value).startswith("kernel_census.json: plain / cc10.0/sm148, ") and sentence in str(refused.value), sentence


def test_the_norms_twin_and_its_band_on_stated_tensors():
    """The norm at the kernel's rounding points, and the band its entry is judged with: the entry's documented contract, per
    output.  The kernel's chain in fp32 -- the sum, the normalised value rounded once to bf16, the codes of that bf16 value
    -- is inside the band in every form.  Each rule of the band is held on a bracket stated here: a normed element outside
    the two bf16 values, one on the other value beyond the margin, a NaN, an 8-bit code of neither value, 8-bit codes that
    are not those of the rows the same call wrote, a block scale of no admitted maximum, a 4-bit code outside the bracket
    of the two reciprocals, and a residual stream that differs: each fails the block."""
    from types import SimpleNamespace

    from model_samples import tie_out
    from model_samples.blocks import DeviceFacts
    from model_samples.providers import Quant, torch_ref
    from model_samples.qwen_dense import configs
    from model_samples.weights import BLOCK, BlockScaledTensor, _e2m1, _e4m3_values, quantize_fp8

    assert (
        tie_out.NORM_MARGIN_ULP32 == 20.0
    ), "the bound of the entry's page (docs/operations/normalizations/add_norm_quant_decode.md): it moves only with that page"
    page = " ".join((MODELS.parents[1] / "docs" / "operations" / "normalizations" / "add_norm_quant_decode.md").read_text().split())
    assert "the kernel's fp32 value is within `20 x 2^-24` relative of it" in page and "the codes are those of quantizing `out` itself" in page
    assert torch_ref.NVFP4_ENCODE_BOUND == 5 * 2.0**-23 and torch_ref.FP4_SIXTH_BITS == 0x3E2AAAAB
    bf16 = torch.bfloat16

    def judged(out, twin):
        stats = tie_out.BlockStats("add_norm_quant")
        tie_out.BlockTap._judge(stats, tuple(out), tuple(twin.reference), [], [], 0, twin)
        return stats

    # ---- the rules, on a bracket stated here: per element the nearest bf16 value, the other one, the distance from their midpoint
    nearest = torch.tensor([[1.0, -2.0, 0.5, 3.0] * 8, [0.25, 4.0, -1.5, 6.0] * 8]).to(bf16)  # two rows of two blocks of 16
    other = torch.tensor([[1.0078125, -2.015625, 0.50390625, 3.015625] * 8, [0.251953125, 4.03125, -1.5078125, 6.03125] * 8]).to(bf16)  # one unit away
    margin = torch.full((2, 32), 1000.0, dtype=torch.float64)
    margin[0, 5], margin[1, 3] = 3.0, 20.0  # two elements inside the margin: the kernel may write either value there
    residual = torch.ones((2, 32), dtype=bf16)

    def stated(names, quant, reference, bracket=None):
        return SimpleNamespace(bracket=bracket or (nearest, other, margin), outputs=names, quant=quant, reference=reference)

    final = stated(("normed", "residual"), None, (nearest, residual))
    stats = judged((nearest, residual), final)
    assert stats.ok and (stats.elements, stats.other_value, stats.outside, stats.off_nearest, stats.total_bytes) == (64, 0, 0, 0, 0)
    assert stats.line().startswith("residual stream equal; 64 normed elements, 0 of them the other bf16 value") and stats.line().endswith("; 0 outside")
    inside = nearest.clone()
    inside[0, 5], inside[1, 3] = other[0, 5], other[1, 3]
    stats = judged((inside, residual), final)
    assert stats.ok and stats.other_value == 2, "inside the margin the fp32 value may lie on the other side of the midpoint: the kernel's own arithmetic"
    beyond = nearest.clone()
    beyond[0, 6] = other[0, 6]  # the other value where the normalised value is far from the midpoint
    stats = judged((beyond, residual), final)
    assert (
        not stats.ok
        and (stats.off_nearest, stats.outside, stats.other_value) == (1, 0, 1)
        and "; 1 off the nearest value more than 20 fp32 units" in stats.line()
    )
    apart = nearest.clone()
    apart[1, 3] = 6.0625  # two units away: neither value of the bracket, margin or not
    stats = judged((apart, residual), final)
    assert not stats.ok and stats.outside == 1 and stats.line().endswith("; 1 outside")
    signed = nearest.clone()
    signed[0, 0] = -1.0
    assert judged((signed, residual), final).outside == 1, "compared as bit patterns: the sign is part of the value"
    with_nan = nearest.clone()
    with_nan[0, 0] = float("nan")
    stats = judged((with_nan, residual), final)
    assert not stats.ok and not stats.finite and stats.line().endswith("NON-FINITE VALUES")
    stats = judged((nearest, residual + 1), final)
    assert not stats.ok and not stats.state_equal and stats.line().startswith("residual stream DIFFERS; "), "the stream the call updated in place"
    with pytest.raises(
        ValueError, match="the band add_norm_quant is judged against the norm at the kernel's rounding points: the block's twin keeps no bracket"
    ):
        judged((nearest, residual), SimpleNamespace(reference=(nearest, residual)))
    stats = judged((nearest,), final)
    assert not stats.ok and stats.outside == 1, "a call that wrote other outputs than its form names: nothing is compared, and it does not hold"

    # 8-bit codes.  A bf16 value has more digits than an 8-bit code, so two neighbouring bf16 values mostly share their code;
    # 1.0625 and 1.0703125 do not (the first is the tie between the codes of 1.0 and 1.125 and goes to the even one).  The
    # bracket holds that pair twice: once inside the margin, once far from the midpoint
    eight = Quant("fp8", 1.0)
    near8, other8 = nearest.clone(), other.clone()
    for at in ((0, 5), (0, 6)):
        near8[at], other8[at] = 1.0625, 1.0703125
    bracket8 = (near8, other8, margin)
    of_nearest, of_other = quantize_fp8(near8, eight.scale).codes, quantize_fp8(other8, eight.scale).codes
    assert (of_nearest != of_other).nonzero().tolist() == [[0, 5], [0, 6]] and (int(of_nearest[0, 5]), int(of_other[0, 5])) == (0x38, 0x39)
    alone = stated(("codes", "residual"), eight, (of_nearest, residual), bracket8)
    stats = judged((of_nearest, residual), alone)
    assert stats.ok and (stats.total_bytes, stats.equal_bytes, stats.elements) == (64, 64, 0)
    either = of_nearest.clone()
    either[0, 5] = of_other[0, 5]
    stats = judged((either, residual), alone)
    assert stats.ok and stats.total_bytes - stats.equal_bytes == 1, "codes alone: inside the margin either value's code"
    wrong = of_nearest.clone()
    wrong[0, 6] = of_other[0, 6]
    stats = judged((wrong, residual), alone)
    assert not stats.ok and (stats.off_nearest, stats.outside) == (1, 0), "the other value's code beyond the margin"
    wrong = of_nearest.clone()
    wrong[0, 0] = 0x7E
    stats = judged((wrong, residual), alone)
    assert not stats.ok and stats.outside == 1, "the code of neither value"
    wrong[0, 0] = 0x7F  # E4M3 has no infinity: this code is its NaN
    assert not judged((wrong, residual), alone).finite
    both = stated(("codes", "normed", "residual"), eight, (of_nearest, near8, residual), bracket8)
    assert judged((of_nearest, near8, residual), both).ok
    took_other = near8.clone()
    took_other[0, 5] = other8[0, 5]  # the kernel's fp32 value fell on the other side, inside the margin
    assert judged((quantize_fp8(took_other, eight.scale).codes, took_other, residual), both).ok, "the codes are those of the bf16 rows the same call wrote"
    stats = judged((of_nearest, took_other, residual), both)
    assert (
        not stats.ok and stats.not_of_out == 1 and "; 1 8-bit codes not those of the normed rows of their call" in stats.line()
    ), "not the codes of other rows"

    four = Quant("nvfp4", 0.03125)  # the kernel's global scale is 32
    assert float(torch_ref.nvfp4_global_scale(four.scale, "cpu")) == 32.0 and float(torch_ref.nvfp4_global_scale(1 / 3, "cpu")) == 3.0

    def nibbles(packed):
        return torch.stack((packed & 0xF, packed >> 4), dim=-1).flatten(-2)

    def pack(codes):
        return codes[..., 0::2] | (codes[..., 1::2] << 4)

    def kernel4(y, scale):  # the kernel's 4-bit chain on bf16 rows: its block scale, then the codes through the fp32 quotient
        scales = torch_ref.nvfp4_block_scales(y.float().abs().unflatten(-1, (-1, BLOCK)).amax(-1), scale)
        step = _e4m3_values(scales).repeat_interleave(BLOCK, dim=-1)
        encode = torch.where(
            step != 0, torch_ref.nvfp4_global_scale(scale, "cpu") / torch.where(step != 0, step, torch.ones_like(step)), torch.zeros_like(step)
        )
        return pack(_e2m1(y.float() * encode, signed_zero=True)), scales

    codes4, scales4 = kernel4(nearest, four.scale)
    assert scales4.tolist() == [[0x58, 0x58], [0x60, 0x60]], "e4m3(32 x (3 x fp32(1/6))) = 16 and e4m3(32 x (6 x fp32(1/6))) = 32, by their bytes"
    mlp = stated(("codes", "scales", "residual"), four, (codes4, scales4, residual))
    stats = judged((codes4, scales4, residual), mlp)
    assert stats.ok and (stats.total_bytes, stats.equal_bytes) == (64 + 4, 64 + 4), "the codes one nibble per element, and the block scales"
    wrong = scales4.clone()
    wrong[0, 1] += 1
    stats = judged((codes4, wrong, residual), mlp)
    assert not stats.ok and stats.outside >= 1, "a block scale of no maximum the bracket admits (and the codes are then judged under it)"
    wrong = nibbles(codes4)
    wrong[1, 0] = 7  # the code of 6 x the block scale, where the value is the block's smallest
    stats = judged((pack(wrong), scales4, residual), mlp)
    assert not stats.ok and stats.outside == 1, "a code outside the bracket of the two reciprocals"
    wrong = nibbles(codes4)
    wrong[0, 1] ^= 0x8  # the other sign
    assert judged((pack(wrong), scales4, residual), mlp).outside == 1
    # the block maximum is an element inside the margin: the kernel's maximum is one of two values, and so is its block scale
    top, top_other, top_margin = nearest.clone(), other.clone(), margin.clone()
    top[1, 3], top_other[1, 3], top_margin[1, 3] = 6.375, 6.40625, 5.0  # neighbours whose scales 34.0 (a tie: to the even 32) and 34.2 round apart
    edge = SimpleNamespace(bracket=(top, top_other, top_margin), outputs=("codes", "scales", "residual"), quant=four, reference=None)
    low, high = kernel4(top, four.scale), kernel4(torch.where(top_margin <= 20.0, top_other, top), four.scale)
    assert low[1][1, 0] != high[1][1, 0], "two admitted block scales"
    for written in (low, high):
        edge.reference = (*written, residual)
        assert judged((*written, residual), edge).ok, "either: the codes are judged under the block scale the kernel stored"
    firm = top_margin.clone()
    firm[1, 3] = 1000.0  # the same rows with that element far from its midpoint: only the nearest value's maximum is admitted
    edge.bracket = (top, top_other, firm)
    assert judged((*low, residual), edge).ok and not judged((*high, residual), edge).ok

    # ---- the 4-bit models at the kernel's points
    assert torch_ref.nvfp4_block_scales(torch.tensor([6.0, 3.0, 0.0, 448.0 * 6 / 32, 1e9]), four.scale).tolist() == [0x60, 0x58, 0x00, 0x7E, 0x7E]
    y = torch.tensor([[6.0, -6.0, 0.0, -0.0, 1.25, 1.3, -0.25, 0.26, 3.5, 5.0, 2.5, 1.75, 0.75, 9.0, -9.0, 0.1]]).to(bf16)
    low, high = torch_ref.nvfp4_code_bracket(y, torch.tensor([[0x38]], dtype=torch.uint8), 1.0)  # a block scale of 1 at a global scale of 1
    assert low.tolist() == [[7, 15, 0, 8, 2, 3, 8, 1, 5, 6, 4, 3, 1, 7, 15, 0]], "below an exact tie; a zero keeps its sign; saturation at 6"
    assert high.tolist() == [[7, 15, 0, 8, 3, 3, 9, 1, 6, 7, 5, 4, 2, 7, 15, 0]], "above it: two adjacent codes on a tie, one code everywhere else"
    zero_low, zero_high = torch_ref.nvfp4_code_bracket(y, torch.tensor([[0x00]], dtype=torch.uint8), 1.0)
    assert (
        zero_low.tolist() == zero_high.tolist() == [[0, 8, 0, 8, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 8, 0]]
    ), "under a zero block scale: the codes of the signed zero"

    # ---- the twin's arithmetic against the kernel's chain written out in fp32, in every form the row names
    generator = torch.Generator().manual_seed(11)
    x = (3 * torch.randn((3, 5120), generator=generator)).to(bf16)
    stream = (10 * torch.randn((3, 5120), generator=generator)).to(bf16)
    weight = (0.3 * torch.randn(5120, generator=generator)).to(bf16)
    x[0, :64], stream[0, :64] = 0.0, 0.0  # an all-zero block
    stream[1, 64:128] = -x[1, 64:128]  # a stretch whose sum cancels
    weight[200:216] = -1.0  # 1 + w = 0: a block of zero outputs
    eps32 = float(torch.tensor(1e-6, dtype=torch.float32))
    new_residual, near, far, distance = torch_ref.norm_bracket(x, stream, weight, 1e-6)
    assert torch.equal(new_residual, (stream.float() + x.float()).to(bf16)), "the residual: one fp32 addition, one rounding"
    v = stream.double() + x.double()
    exact = v * (1.0 + weight.double()) / torch.sqrt(v.square().mean(-1, keepdim=True) + eps32)
    assert bool(((near.double() - exact).abs() <= (far.double() - exact).abs()).all()), "nearest is the nearer of the two"
    assert bool(((near.double() - exact) * (far.double() - exact) <= 0).all()), "and the two are on either side of the value (or on it)"
    assert torch.equal(near, torch_ref.rms_norm(stream.float() + x.float(), weight, 1e-6, torch.float64)), "on these rows the chain's fp64 norm rounds to it"
    started = torch_ref.norm_bracket(x, None, weight, 1e-6)
    assert started[0] is x and not torch.equal(started[1], near), "the first call of a step: the branch is the stream, and the norm is of the branch alone"

    def kernel_chain(branch, residual, site_form, quant):  # what the entry computes: fp32 throughout, one rounding to bf16
        v32 = branch.float() if residual is None else residual.float() + branch.float()
        y = ((v32 * torch.rsqrt(v32.square().mean(-1, keepdim=True) + 1e-6)) * (1.0 + weight.float())).to(bf16)
        out = []
        if site_form["QUANT"] == "e4m3_static":
            out.append(quantize_fp8(y, quant.scale).codes)
        elif site_form["QUANT"] == "nvfp4":
            out += list(kernel4(y, quant.scale))
        return (*out, *([y] if site_form["BF16_OUT"] else []), branch if residual is None else v32.to(bf16)), y

    row = configs.PROVIDERS["add_norm_quantize"][0]
    twin = torch_ref.AddNormQuantizeBracket(configs.PROVIDERS["add_norm_quantize"][1], SimpleNamespace(norm_eps=1e-6), DeviceFacts((0, 0), "cpu"), {})
    twin.forms = {site: dict(row.site_form(site)) for site, _ in row.sites}
    names = {
        "linear_attention.first": ("codes", "normed", "residual"),
        "full_attention.first": ("codes", "residual"),
        "linear_attention": ("codes", "normed", "residual"),
        "full_attention": ("codes", "residual"),
        "mlp": ("codes", "scales", "residual"),
        "output_head": ("normed", "residual"),
    }
    others = 0
    for site, _ in row.sites:
        quant = four if site in ("mlp", "output_head") else Quant("fp8", 0.02)
        operands = (x, None if site.endswith(".first") else stream, weight, quant, site)
        out, y = kernel_chain(*operands[:2], twin.forms[site], quant)
        reference = twin(*operands)
        assert twin.outputs == names[site], f"{site}: what the form of the site writes"
        assert (reference[0] is None) == (site == "output_head") and (reference[1] is None) == (twin.forms[site]["BF16_OUT"] is False), site
        if site == "mlp":
            assert isinstance(reference[0], BlockScaledTensor) and reference[0].scale == four.scale
        stats = tie_out.BlockStats("add_norm_quant")
        tie_out.BlockTap._judge(stats, out, tie_out._tensors(reference), [], [], 0, twin)
        assert stats.ok and stats.outside == 0 and stats.off_nearest == 0 and stats.not_of_out == 0 and stats.state_equal and stats.finite, (site, stats.line())
        others += int((y != twin.bracket[0]).sum())
        assert int((y != twin.bracket[0]).sum()) <= 0.001 * y.numel(), f"{site}: an fp32 chain leaves the nearest bf16 of the fp64 value on few elements"
        if "scales" in names[site]:
            assert stats.total_bytes == 3 * 5120 + 3 * 320 and stats.equal_bytes >= 0.999 * stats.total_bytes
        # one changed byte of each output of the form fails the call
        for index, name in enumerate(names[site]):
            changed = [t.clone() for t in out]
            flat = changed[index].view(torch.uint8).view(-1)
            flat[7] = flat[7] ^ (0x40 if name == "normed" else 0x44)
            defect = tie_out.BlockStats("add_norm_quant")
            tie_out.BlockTap._judge(defect, changed, tie_out._tensors(reference), [], [], 0, twin)
            assert not defect.ok, f"{site}: one changed byte of {name}"
    assert bool((distance > 0).all()) and float(distance.min()) < 1e6, "the margins of these rows are finite: no element is a bf16 value by construction"


def test_the_norm_row_through_a_session_and_what_fails_it(monkeypatch):
    """The norm row of the ``native`` stack through a real session on the CPU, on the stand-in of its entry.  Clean, the run
    holds: five calls of a step on four forms, each served by the instance of its site, the residual stream in place, the
    consumers on fallback rows reading the kernel's bytes as they are.  Then the negative controls, each of which must
    fail the kernel check with its own sentence -- a call served in the form of another site (only the call log shows
    it), a chain of other kernels under the row's name, a launch more in front of its kernel, a launch less, a launch in
    another shape than the census recorded -- or fail the block in its band while the kernel check holds: a wrong code, a
    wrong block scale, a wrong normed element, a wrong element of the residual stream."""
    from model_samples import census, tie_out
    from model_samples.qwen_dense import configs

    row = configs.PROVIDERS["add_norm_quantize"][0]

    def run(sequences=2, **stand_in):
        session = _recorded_run(monkeypatch, sequences=sequences, stack="native", **stand_in)[0]
        session.trace(4)
        return session, tie_out.run_checks(session, prompt_tokens=3)

    def sentences(checks):
        return [line.strip() for line in checks.lines if line.strip().startswith("kernels: ")]

    def norm_line(checks):
        return next(line for line in checks.lines if line.strip().startswith("add_norm_quantize ") and "calls" in line)

    session, checks = run()
    assert checks.ok and checks.kernels == census.PASS, checks.text()
    assert "8 on exact rows, in the wiring order" in _line(checks, "kernels ") and norm_line(checks).endswith("[band: add_norm_quant]  PASS")
    assert "add_norm_quantize equal (gated)" in _line(checks, "batch invariance"), "each sequence alone writes the bytes it writes among two"
    log = session.traced.call_log[0]
    assert log[1] == census.Call("add_norm_quantize", 0, 2, "cudnn.AddNormQuantDecode #1", row.site_form("linear_attention.first"))
    assert [(call.layer, call.instance[-2:], dict(call.form)["QUANT"]) for call in log if call.block == "add_norm_quantize"] == [
        (0, "#1", "e4m3_static"),
        (0, "#5", "nvfp4"),
        (1, "#4", "e4m3_static"),
        (1, "#5", "nvfp4"),
        (2, "#6", None),
    ], "the instance of the call's own site, call by call"
    counted = _line(checks, "add_norm_quantize, as the call log of a traced step has it (5 calls): ")
    assert counted.count(" x cudnn.AddNormQuantDecode #") == 4 and "2 x cudnn.AddNormQuantDecode #5 " in counted, "four forms, the MLP's twice"
    stated = [line.strip() for line in checks.lines if line.strip().startswith("add_norm_quantize at ")]
    assert [line.split(":")[0] for line in stated] == [
        f"add_norm_quantize at {site}" for site in ("linear_attention.first", "mlp", "full_attention", "output_head")
    ]
    assert "2 launches per step (2 calls x 1 per call); the entry states grid (2, 1, 1), block (640, 1, 1), shared memory 80 B and the kernel " in stated[1]

    # (1) the calls in front of an MLP served by the instance of another form: the launch records are clean (one kernel id, one launch shape)
    def neighbour(site):
        served = "full_attention" if site == "mlp" else site
        return f"cudnn.AddNormQuantDecode #{[name for name, _ in row.sites].index(served) + 1}", tuple(row.site_form(served))

    _, checks = run(norm_serves=neighbour)
    assert not checks.ok and checks.kernels == census.FAIL, checks.text()
    assert sentences(checks) == [
        "kernels: form: add_norm_quantize (layer 0), the call at mlp: the adapter served the call with the form [PDL True, PDL_RELEASE True, ADD True, "
        "STORE_RESIDUAL False, BF16_OUT False, QUANT e4m3_static] (cudnn.AddNormQuantDecode #4); the row says [PDL True, PDL_RELEASE True, ADD True, "
        "STORE_RESIDUAL False, BF16_OUT False, QUANT nvfp4]"
    ], "one sentence for the kind of difference, whichever layer"
    for other in ("replay", "blocks", "tokens", "batch invariance"):
        assert _line(checks, other).endswith("PASS"), checks.text()

    # (2) a fallback under the row's name: the block launches a chain of PyTorch kernels, the row declares the entry's kernel
    _, checks = run(norm_defect="chain")
    missing = "did not launch add_norm_quant: the step has 0 launch(es) of add_norm_quant, the plan 5 = 5 call(s) x 1 per call"
    assert (
        not checks.ok
        and checks.kernels == census.FAIL
        and sentences(checks) == [f"kernels: step 0 (and 3 more of the 4 traced steps): add_norm_quantize (layer {layer}) {missing}" for layer in (0, 1, 2)]
    ), checks.text()
    assert norm_line(checks).endswith("PASS"), "its bytes are the kernel's: only the kernel check sees it"

    # (3) one launch more: in front of the first norm of a step nothing is declared (the gather before it is judged one by one too)
    _, checks = run(norm_defect="extra")
    assert (
        not checks.ok
        and checks.kernels == census.FAIL
        and sentences(checks)
        == [
            f"kernels: step 0 (and 3 more of the 4 traced steps): launched {ELEMENTWISE[:93] + '...'!r} between embedding and add_norm_quantize (layer 0): "
            "a kernel of the fallback family pytorch, which no row there declares; nothing may sit between two exact rows"
        ]
    ), checks.text()

    # (4) one launch less: the final norm does not launch
    _, checks = run(norm_defect="missing")
    assert (
        not checks.ok
        and checks.kernels == census.FAIL
        and sentences(checks)
        == [
            "kernels: step 0 (and 3 more of the 4 traced steps): add_norm_quantize (layer 2) did not launch add_norm_quant: the step has 4 launch(es) of "
            "add_norm_quant, the plan 5 = 5 call(s) x 1 per call"
        ]
    ), checks.text()

    # (5) at one sequence the census holds the run: a launch in another shape than the recorded one, although the entry states that shape too
    _, checks = run(sequences=1, norm_shape=((1, 1, 1), (320, 1, 1), 80))
    assert not checks.ok and checks.kernels == census.FAIL
    assert sentences(checks) == [
        f"kernels: step 0 (and 3 more of the 4 traced steps): add_norm_quantize (layer {layer}) launched add_norm_quant with grid (1, 1, 1), "
        f"block (320, 1, 1), shared memory 80 B; the {CENSUS} has grid (1, 1, 1), block (640, 1, 1), shared memory 80 B"
        for layer in (0, 1, 2)
    ], checks.text()
    _, checks = run(sequences=1)
    assert checks.ok and any(f"{CENSUS}: NOT CLAIMED (" in line and COMPARED_8 in line for line in checks.lines), "held to the recorded shape, and said"

    # (6) wrong bytes: the block fails in its band, the kernel check holds
    for defect, said in (
        ("code", lambda line: not line.split("; ")[-1].startswith("0 outside")),
        ("scale", lambda line: not line.split("; ")[-1].startswith("0 outside")),
        ("bf16", lambda line: not line.split("; ")[-1].startswith("0 outside")),
        ("residual", lambda line: line.split("calls", 1)[1].strip().startswith("residual stream DIFFERS")),
    ):
        _, checks = run(norm_defect=defect)
        line = norm_line(checks)
        assert not checks.ok and checks.kernels == census.PASS and line.endswith("[band: add_norm_quant]  FAIL") and said(line), (defect, line)
        assert _line(checks, "blocks").endswith("FAIL") and "7 of 8 provider blocks inside their bands" in _line(checks, "blocks"), defect

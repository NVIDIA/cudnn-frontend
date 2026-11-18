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
import re
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
#: of the FP8 projection entry at 1 row, per weight shape of the two in-projections: one thread block per 128 weight rows
PROJECTION_SHAPE = {16384: ((128, 1, 1), (256, 1, 1), 139520), 14336: ((112, 1, 1), (256, 1, 1), 139520)}
#: the symbol of that entry's kernel, as a traced run reported it: the rendering follows the kernel's name
PROJECTION_SYMBOL = (
    "cudnn_kernel_frost_fp8_projection_decode_n16384k5120s1r1_m128g8t8c32_w1h1p1q1_tensorptri8gmemalign32o21_tensorptri8gmemalign32o21_"
    "tensorptrbf16gmemalign32o21_tensorptrf32gmemo11_0"
)
#: of the same entry for the out-projection of a linear-attention layer in a step of several tokens per sequence: at 2 to 8 rows
#: two K slices per 64 weight rows (a step of six rows: the recorded launch), at 9 to 16 rows four slices and the 16-row tile
OUT_SHAPE = ((80, 2, 1), (256, 1, 1), 103808)
OUT_SHAPE_SPLIT_4 = ((80, 4, 1), (256, 1, 1), 84352)
#: of the fused verify step at one sequence of six tokens, and a symbol of its kernel (cut in front of its launch knobs and tokens)
VERIFY_SHAPE = ((96, 1, 1), (256, 1, 1), 126512)
VERIFY_SYMBOL = (
    "cudnn_kernel_frost_gdn_verify_fused_CfgTPR4K_RESIDENTTrueNBUF3STATE_STREAMFalseMIN_CTAS_PER_SM1CTAS_PER_SM0BA_PREFETCH_CHUNKS3"
    "NORM_FORMctas2STATE_LOAD_PATHbulkSTATE_STORE_PATHbulkBULK_I_0"
)
CENSUS_VERIFY = "target census verify / cc 10.0 / 148 SMs (recorded 2026-11-12)"
#: the form of the row of that entry in the target stacks, and the further fields of its two call sites
PROJECTION_FORM = "[SPLIT_K 1, MMA_M 128, STAGES 8, TMEM_COLS 32, PREWAIT_WEIGHTS True, HOLD_TMEM True, PDL True, PDL_RELEASE True"
#: of the fused gate / up entry at 1 to 32 rows on a part with at least 148 SMs: one thread block per 128 weight rows up to the cap
SWIGLU_SHAPE = ((1, 1, 148), (192, 1, 1), 231424)
#: the symbol of that entry's kernel, as a traced run reported it: the form's text follows the kernel's name
SWIGLU_SYMBOL = (
    "cudnn_kernel_frost_swiglu_fp4_decode_h5120_i17408_t8_pw2_evf_c148_pdl_SwigluFp4DecodeKernel_TiledMMA_ThrLayoutVMNK11110000_PermutationMNK____"
    "MMAAtom_ThrID10_ShapeMNK128864_TVLayoutA1128_0"
)
#: the form of the row of that entry in the target stacks at 1 to 8 rows
SWIGLU_FORM = "[TOKEN_TILE 8, PREWAIT_STAGES 2, L2_EVICT_FIRST True, MAX_CLUSTERS 148, PDL True]"
RECORDED = {"torch": "2.14.0+cu130", "nvidia-cutlass-dsl": "4.8.0"}  # installed versions equal to the census's recorded ones
CENSUS = "target census plain / cc 10.0 / 148 SMs (recorded 2026-11-12)"
#: what the census line says was compared when the gather, the fused step, the gate, the five norms, the two in-projections
#: and the two gate / up multiplies of the two-layer model held: three numbers
COMPARED_12 = (
    "12 launches of a step compared by launch shape (2 the target's own compiled kernels + 10 cuDNN Frontend entries with byte-equal results "
    "+ 0 declared stand-ins)"
)
COMPARED_2 = (
    "2 launches of a step compared by launch shape (2 the target's own compiled kernels + 0 cuDNN Frontend entries with byte-equal results "
    "+ 0 declared stand-ins)"
)
WHOLE = "the whole target kernel set of a step of this model (what a complete stack launches; this stack's share is the next line): "
GATHER = "void at::native::(anonymous namespace)::indexSelectSmallIndex<BFloat16, int, unsigned int, 2, 2, -2>(at::cuda::detail::TensorInfo<...>)"
# the kernel torch.index_select launched for the embedding from 17 rows (compute capability 10.0, PyTorch 2.13.0), as recorded
GATHER_VECTORIZED = "void at::native::vectorized_gather_kernel<16, int>(char*, char*, int*, int, long, long, long, long, bool)"
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


def _log(sites, steps=1):
    """The call log of ``steps`` traced steps that each made the calls of the wiring, every call served as its row says."""
    from model_samples.census import Call

    return [[Call(site.block, site.layer, 1, "", tuple(site.form)) for site in sites] for _ in range(steps)]


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
    assert kernel(PROJECTION_SYMBOL) == kernel(PROJECTION_SYMBOL.replace("n16384", "n14336").replace("r1_", "r32_")) == "fp8_projection"
    assert kernel(PROJECTION_SYMBOL) != kernel(NVJET), "the entry's symbol is its own, whatever of a library's name it holds further on"
    assert (
        kernel(SWIGLU_SYMBOL) == kernel(SWIGLU_SYMBOL.replace("_t8_", "_t64_").replace("_pdl_", "_nopdl_")) == "swiglu_fp4"
    ), "every form of the fused gate / up kernel"
    assert (
        kernel("cudnn_kernel_frost_gdn_verify_fused_CfgTPR4K_RESIDENTTrue_0") == "gdn_verify_fused"
    ), "the verify step is a kernel of its own, never the decode step's"
    assert kernel("cudnn_kernel_frost_shared_expert_decode_CfgTOKENS4_0") == "cudnn_fe", "a cuDNN Frontend kernel no pattern names is no family's member"
    assert kernel(GATHER) == "embedding_gather" and kernel(GATHER.replace("SmallIndex", "LargeIndex")) == "embedding_gather"
    assert kernel(GATHER_VECTORIZED) == "embedding_gather", "from 17 rows the index-select of the embedding is PyTorch's vectorized gather"
    assert kernel("void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<...> >(long, ...)") == "pytorch", "indexing is not the gather"
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
    assert judged == {
        "gdn_decode_fused",
        "gdn_verify_fused",
        "gate_quant",
        "add_norm_quant",
        "fp8_projection",
        "swiglu_fp4",
        "embedding_gather",
    } and judged <= set(named), "a pattern of its own each"
    for kernel_id in ("gdn_decode_fused", "gdn_verify_fused", "gate_quant", "add_norm_quant", "fp8_projection", "swiglu_fp4"):
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
    traced = census.Traced.of(sites, _log(sites, 2), _step() + _step(), 2, "", rows=1)
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
    # from 17 rows the embedding's index-select is another kernel of PyTorch: the row holds it by name as it holds the other two
    many = [record._replace(name=GATHER_VECTORIZED, grid=(17, 3, 1), block=(256, 1, 1)) if record.name == GATHER else record for record in _step()]
    traced = census.Traced.of(sites, _log(sites, 1), many, 1, "", rows=17)
    assert traced.judge() == (census.PASS, []) and "2 on exact rows, in the wiring order; 7 attributed by family" in traced.summary()
    unnamed = [record._replace(name="void at::native::some_other_gather<16, int>(char*)") if record.name == GATHER_VECTORIZED else record for record in many]
    status, lines = census.Traced.of(sites, _log(sites, 1), unnamed, 1, "", rows=17).judge()
    assert status == census.FAIL and lines[0].startswith("step 0: embedding did not launch embedding_gather: the step has 0 launch(es) of embedding_gather")


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
        _compare([step[:-1] + [at("cudnn_kernel_frost_shared_expert_decode_CfgTOKENS4_0"), step[-1]]]),
        "launched 'cudnn_kernel_frost_shared_expert_decode_CfgTOKENS4_0' after linear_attention.step (layer 0): the kernel cudnn_fe, "
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


def test_a_traced_step_without_its_call_log_fails_the_check():
    """A traced pass keeps one call log per traced step: what the adapters wrote down after each call.  The form check reads
    from it which entry instance served a call, and the comparison whether the step made the calls of the wiring.  A step
    without its log shows neither, so a pass with launch records and no log, or fewer logs than steps, or more, fails --
    whatever the launch records say."""
    from model_samples import census

    sites = _sites()
    two = _step() + _step()
    assert census.Traced.of(sites, _log(sites, 2), two, 2, "", rows=1).judge() == (census.PASS, []), "the control: one log per traced step"
    said = (
        "calls: the call log holds the block calls of {} for 2 traced steps: without one log per traced step nothing shows which entry instance served a call"
    )
    for log, has in (([], "no step"), (_log(sites, 1), "1 step"), (_log(sites, 3), "3 steps")):
        traced = census.Traced.of(sites, log, two, 2, "", rows=1)
        assert traced.judge() == (census.FAIL, [said.format(has)]), has
        assert traced.summary() == "2 traced steps: 9 launches per step; 1 kind of difference between the launches and the plan"
    # ... also where the records carry no launch arguments: it is a failure, not "nothing observed"
    bare = [record._replace(grid=None, block=None, shared_memory=None) for record in two]
    assert census.Traced.of(sites, [], bare, 2, "", rows=1).judge()[0] == census.FAIL
    # a pass without launch records has no traced step to hold a log against: nothing was observed, and that is what it says
    assert census.Traced.of(sites, [], [], 2, "the profiler failed (no tracing library)", rows=1).judge() == (census.NOT_OBSERVED, [])
    # one step of one call: the singular
    one = [sites[-1]]
    at = _Clock()
    status, lines = census.Traced.of(one, [], [at(ARGMAX), at(HOST_READ, kind="copy")], 1, "", rows=1).judge()
    assert status == census.FAIL and lines == [
        "calls: the call log holds the block calls of no step for 1 traced step: without one log per traced step nothing shows which entry instance served a call"
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
    held = [sites[0], sites[1], bare, *sites[3:]]
    status, lines = census.Traced.of(held, _log(held), _step(), 1, "", rows=1).judge()
    assert status == census.FAIL and lines == [f"form: {no_form}"], "it fails a pass whose launch records are clean"
    # a form that does not hold fails a pass whose launch records are clean, and one that observed nothing
    held = [sites[0], sites[1], other, *sites[3:]]
    for records, missing in ((_step(), ""), ([], "the profiler failed (no tracing library)")):
        status, lines = census.Traced.of(held, _log(held), records, 1, missing, rows=1).judge()
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
    shapeless = census.Traced.of(sites, _log(sites), bare, 1, "", rows=1)
    assert shapeless.judge() == (census.NOT_OBSERVED, [])
    assert shapeless.unobserved == "the profiler's kernel records carry no launch arguments (grid, block, shared memory)"
    assert shapeless.summary().endswith("; launch shapes not observed: the profiler's kernel records carry no launch arguments (grid, block, shared memory)")
    # ... and a kernel that is missing is a failure all the same
    missing = census.Traced.of(sites, _log(sites), [record for record in bare if record.name != SYMBOL], 1, "", rows=1)
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


def test_a_row_serves_the_layer_widths_it_names():
    """An entry that serves some layer widths only has them in its row (``widths``: fields of the model's geometry).  A
    model of the family with another width is not that row's: the next row of the block is asked, and where there is none
    the decline is a ``NotImplementedError`` that names both -- a SKIP, never "the provider table is wrong".  The norm
    entry serves one hidden width; its row says so, and a checkpoint with another one plans on the reference chain on a
    real device, with no kernel identity stated: what the family README says of such a checkpoint."""
    from dataclasses import replace
    from types import SimpleNamespace

    import cudnn

    from model_samples.blocks import DeviceFacts, ProviderRow, plan, select_row
    from model_samples.qwen_dense import QwenDense, configs

    facts = DeviceFacts((10, 0), "stated device", 148, 180 * GIB, 80 * GIB)
    wide, narrow = (SimpleNamespace(geometry=SimpleNamespace(hidden=hidden, mlp=96)) for hidden in (5120, 4096))
    rows = [
        ProviderRow(("cc10.x",), (1, None), (1, None), "fe.add_norm_quant_decode", "native", serves="rows 1-8, hidden 5120", widths=(("hidden", 5120),)),
        ProviderRow(("any",), (1, None), (1, None), "torch_ref.add_norm_quantize", "native", serves="any", fallback="reference chain"),
    ]
    assert rows[0].serves_model(wide) and not rows[0].serves_model(narrow) and not rows[0].serves_model(None), "no model: no widths to hold"
    assert rows[1].widths == () and rows[1].serves_model(narrow) and rows[1].serves_model(None), "a row that names no widths serves any model"
    assert rows[0].widths_text() == "hidden 5120" and rows[0].widths_text(narrow) == "hidden 4096" and rows[0].widths_text(wide) == ""
    assert select_row("b", rows, facts, 1, 1, "native", model=wide).key == "fe.add_norm_quant_decode"
    assert select_row("b", rows, facts, 1, 1, "native", model=narrow).key == "torch_ref.add_norm_quantize", "another width: the next row of the block"
    assert select_row("b", rows, facts, 1, 1, "native").key == "torch_ref.add_norm_quantize"
    with pytest.raises(NotImplementedError, match=r"^cudnn.AddNormQuantDecode serves hidden 5120; this model has hidden 4096$"):
        select_row("b", rows[:1], facts, 1, 1, "native", model=narrow)
    with pytest.raises(NotImplementedError, match=r"^cudnn.AddNormQuantDecode serves hidden 5120; this model has hidden 4096$"):
        select_row("b", rows, facts, 1, 1, "native", key="fe.add_norm_quant_decode", model=narrow)  # --provider cannot widen an entry
    with pytest.raises(NotImplementedError, match=r"^cudnn.AddNormQuantDecode serves hidden 5120; this plan names no model$"):
        select_row("b", rows[:1], facts, 1, 1, "native")
    two = replace(rows[0], widths=(("hidden", 5120), ("mlp", 17408)))
    assert not two.serves_model(wide) and two.widths_text(wide) == "mlp 96" and two.widths_text() == "hidden 5120, mlp 17408", "every width it names"
    with pytest.raises(NotImplementedError, match=r"^cudnn.AddNormQuantDecode serves compute capability 10.x; this device is 12.0$"):
        select_row("b", rows[:1], DeviceFacts((12, 0)), 1, 1, "native", model=narrow)  # the class is said first
    elsewhere = ProviderRow(("cc12.x",), (1, None), (1, None), "fe.gdn_decode_fused", "native", serves="rows 1-8")  # a row of another class in front
    with pytest.raises(NotImplementedError, match=r"^cudnn.AddNormQuantDecode serves hidden 5120; this model has hidden 4096$"):
        select_row("b", [elsewhere, rows[0]], facts, 1, 1, "native", model=narrow)  # the width is said of the row that serves the class
    with pytest.raises(NotImplementedError, match=r"^cudnn.AddNormQuantDecode serves rows 1-8, hidden 5120; this run has 9 sequences x 1 tokens per sequence$"):
        select_row("b", [replace(rows[0], rows=(1, 8))], facts, 9, 1, "native", model=wide)

    # the family's table says what the entry serves: the row's width is one the entry builds for, and the entry declines another by name
    row = configs.PROVIDERS["add_norm_quantize"][0]
    assert row.key == "fe.add_norm_quant_decode" and row.widths == (("hidden", 5120),) and row.serves == "rows 1-1,024, hidden 5120"
    narrow_rows = {r.key: r.widths for block, rows in configs.PROVIDERS.items() for r in rows if r.widths and block != "projection.fp8.out"}
    assert set(narrow_rows) == {
        "fe.add_norm_quant_decode",
        "fe.fp8_projection_decode",
        "fe.swiglu_fp4_decode",
    }, "the rows of the table whose entries serve some widths only"
    # the fused gate / up entry serves one pair of widths, the published model's: each of its four rows names both, and the
    # entry built for the host states them and declines another pair by name
    gate_up = [r for r in configs.PROVIDERS["mlp.gate_up"] if r.key == "fe.swiglu_fp4_decode"]
    assert len(gate_up) == 4 and all(r.widths == (("hidden", 5120), ("mlp", 17408)) for r in gate_up)
    assert [r.serves for r in gate_up] == [f"rows {first}-{last} of 1-64, hidden 5120, intermediate 17408" for first, last, _ in configs.GATE_UP_TILES]
    from cudnn.gemm import SwigluGeometry

    assert cudnn.SwigluFp4Decode("cpu", SwigluGeometry(5120, 17408)).form_for(1)["GEOM"] == {"HIDDEN": 5120, "INTERMEDIATE": 17408}
    with pytest.raises(NotImplementedError, match=r"serves hidden 5120, intermediate 17408 or hidden 5120, intermediate 128 \(validated\); got hidden 4096"):
        cudnn.SwigluFp4Decode("cpu", SwigluGeometry(4096, 17408))
    # the FP8 projection entry serves the two weight shapes of the published model's in-projections: its row names every layer
    # width those follow from, and the shapes it names per call site are the model's own at those widths
    shapes = {site: tuple(value for _, value in fields) for site, fields in configs.PROJECTION_SITES}
    published = QwenDense().geometry
    assert dict(narrow_rows["fe.fp8_projection_decode"]) == {
        name: getattr(published, name) for name in ("hidden", "la_key_heads", "la_value_heads", "la_head_dim", "attn_heads", "attn_kv_heads", "attn_head_dim")
    }
    assert shapes == {"linear_attention.qkvz": published.in_projection("linear_attention"), "full_attention.qkv": published.in_projection("full_attention")}
    assert shapes == {"linear_attention.qkvz": (16384, 5120), "full_attention.qkv": (14336, 5120)}
    # its rows for the out-projection of a verify step serve one weight shape, the linear-attention layers': each names the
    # widths that shape follows from, and the shape it names for its one served call site is the model's own
    out_rows = [r for r in configs.PROVIDERS["projection.fp8.out"] if r.widths]
    assert len(out_rows) == len(configs.VERIFY_OUT_ROWS) == 5 and {r.key for r in out_rows} == {"fe.fp8_projection_decode"}
    assert all(dict(r.widths) == {name: getattr(published, name) for name in ("hidden", "la_value_heads", "la_head_dim")} for r in out_rows)
    assert all(tuple(value for _, value in dict(r.sites)["linear_attention.out"]) == (published.hidden, published.la_out) == (5120, 6144) for r in out_rows)
    form = dict(add=True, bf16=False, quant="nvfp4")
    assert cudnn.AddNormQuantDecode("cpu", dict(row.widths)["hidden"], 1e-6, **form).form_for(1)["HIDDEN"] == 5120
    with pytest.raises(NotImplementedError, match=r"serves hidden 5120 \(validated\); got hidden 4096$"):
        cudnn.AddNormQuantDecode("cpu", 4096, 1e-6, **form)

    # a model of the family with other layer widths, on a device with memory (the real entries are asked for their launch): the norm
    # block is the reference chain's in the target stacks, the plan holds, and the run states no kernel identity
    reduced = QwenDense.reduced(2).geometry
    other = QwenDense(replace(reduced, hidden=4096, la_key_heads=8, la_value_heads=32, attn_heads=16), ("other widths",))
    reason = (
        "no recorded section for this model's widths on cc 10.0 / 148 SMs: hidden 4096 (recorded 5120), la_key_heads 8 (recorded 16), "
        "la_value_heads 32 (recorded 48), attn_heads 16 (recorded 24)"
    )
    for stack in configs.TARGET_STACKS:
        planned = plan(other.provider_table, other.regimes, facts, 1, 1, stack, model=other, options={"device": "cpu"})
        assert planned.rows["add_norm_quantize"].key == "torch_ref.add_norm_quantize" and planned.rows["add_norm_quantize"].fallback == "reference chain"
        assert planned.rows["attention.gate_quantize"].key == "fe.attention_gate_quant_decode", "an entry that serves the width keeps its row"
        assert planned.rows["projection.fp8.in"].key == "cublas.bf16_dequantized", "other weight shapes: the in-projection stays on the multiply over a copy"
        assert planned.rows["mlp.gate_up"].key == "cublas.bf16_dequantized+torch_ref.silu_mul_quantize", "another hidden width: so does the gate / up multiply"
        claim = QwenDense.target().claim(planned, other, installed=RECORDED)
        assert claim.coverage() == [f"kernel identity: not stated for this run ({reason})"] and not claim.known
        line = next(line for line in planned.table().splitlines() if line.split()[0] == "add_norm_quantize")
        assert "torch_ref.add_norm_quantize" in line and line.endswith("fallback: reference chain"), line
        assert "by provider: cuDNN FE 2 | fallbacks 7 (cuBLAS over copies 4, PyTorch chains 3) | PyTorch 2" in planned.table()
    with pytest.raises(NotImplementedError, match=r"^cudnn.AddNormQuantDecode serves hidden 5120; this model has hidden 4096$"):
        plan(
            other.provider_table,
            other.regimes,
            facts,
            1,
            1,
            "native",
            {"add_norm_quantize": "fe.add_norm_quant_decode"},
            model=other,
            options={"device": "cpu"},
        )
    # the published widths, the same device: the entry's row, and the table prints the width with what the row serves
    served, _, _ = _target_plan(2)
    assert served.rows["add_norm_quantize"] is row
    line = next(line for line in served.table().splitlines() if line.split()[0] == "add_norm_quantize")
    assert "cudnn.AddNormQuantDecode" in line and "rows 1-1,024, hidden 5120" in line, line


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
    model = QwenDense(geometry, reductions=("tiny geometry",))
    model.provider_table = _table_of(geometry)
    return model


def _table_of(geometry):
    """The family's provider table with the layer widths a row names re-stated for ``geometry``: the stand-in entries of this
    module serve whatever the widths, so the rows of the published widths are the rows of the tiny geometry (as
    ``_census_of`` re-states the census section)."""
    from dataclasses import replace

    from model_samples.qwen_dense import configs

    return {
        block: [replace(row, widths=tuple((name, getattr(geometry, name)) for name, _ in row.widths)) for row in rows]
        for block, rows in configs.PROVIDERS.items()
    }


def _census_of(model):
    """The family's census with its sections re-stated for the layer widths of ``model``: the stand-in providers of this
    module launch the recorded shapes whatever the widths, so a run of the tiny geometry is a run a section holds for."""
    import copy

    from model_samples import census
    from model_samples.qwen_dense import QwenDense

    data = copy.deepcopy(QwenDense.target().data)
    for sections in data["configurations"].values():  # the decode step's and the verify step's
        for section in sections.values():
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


def _projection_symbol(fields, rows=1):
    """The symbol a stand-in instance of the FP8 projection entry states for its weight shape and a row count: the rendering
    behind the kernel's name, as the entry writes it (a row that names no split, weight tile, ring or token tile: the
    in-projection's)."""
    split, tile, stages, token_tile = (fields.get(name, default) for name, default in (("SPLIT_K", 1), ("MMA_M", 128), ("STAGES", 8), ("TOKEN_TILE", 8)))
    return (
        f"cudnn_kernel_frost_fp8_projection_decode_n{fields['OUT_FEATURES']}k{fields['IN_FEATURES']}s{split}r{rows}_m{tile}g{stages}t{token_tile}c32_w1h1p1q1_0"
    )


def _projection_entry(calls, decline=None):
    """A stand-in for ``cudnn.Fp8ProjectionDecode``: it keeps what it was built with, states the entry's facts from it and
    writes its builds and launches into ``calls``.  ``decline``: a one-element list whose text, when set, its build raises."""

    class Projection:
        def __init__(self, device, geometry, split_k=None, form=None, pdl=None, pdl_release=None):
            self.built = dict(device=str(device), geometry=geometry, split_k=split_k, form=form, pdl=pdl, pdl_release=pdl_release)

        def compile(self, rows=None):
            calls.append(("projection compile", self.built["geometry"].OUT_FEATURES, rows))
            if decline and decline[0]:
                raise NotImplementedError(decline[0])

        def __call__(self, weight, activation, scale, out):
            calls.append(("projection launch", tuple(weight.shape), tuple(activation.shape), tuple(out.shape)))
            return out

        def kernel_symbols_for(self, rows):
            return (_projection_symbol(self.form_for(rows)["GEOM"], rows),)

        def form_for(self, rows):
            geometry, form = self.built["geometry"], self.built["form"]
            stated = {name: getattr(form, name) for name in ("MMA_M", "STAGES", "TMEM_COLS", "PREWAIT_WEIGHTS", "HOLD_TMEM")}
            geom = {"OUT_FEATURES": geometry.OUT_FEATURES, "IN_FEATURES": geometry.IN_FEATURES}
            return dict(
                stated,
                GEOM=geom,
                SPLIT_K=self.built["split_k"],
                ROWS=rows,
                TOKEN_TILE=8,
                PDL=bool(self.built["pdl"]),
                PDL_RELEASE=bool(self.built["pdl_release"]),
            )

        def launch_shape_for(self, rows):
            return ((self.built["geometry"].OUT_FEATURES // self.built["form"].MMA_M, self.built["split_k"], 1), (256, 1, 1), 139520)

    return Projection


def _swiglu_entry(calls, decline=None):
    """A stand-in for ``cudnn.SwigluFp4Decode``: it keeps what it was built with, states the entry's facts from it and writes
    its builds and launches into ``calls`` (a launch: the operands it was handed).  ``decline``: a one-element list whose
    text, when set, its build raises."""

    class Swiglu:
        def __init__(self, device, geometry=None, cfg=None, pdl=None):
            self.built = dict(device=str(device), geometry=geometry, cfg=cfg, pdl=pdl)

        def compile(self, rows=None):
            calls.append(("swiglu compile", rows))
            if decline and decline[0]:
                raise NotImplementedError(decline[0])

        def __call__(self, weight, weight_scale, activation, activation_scale, alpha, out_global_scale, out, out_scale):
            calls.append(("swiglu launch", weight, weight_scale, activation, activation_scale, alpha, out_global_scale, out, out_scale))
            return out, out_scale

        def kernel_symbols_for(self, rows):
            return (SWIGLU_SYMBOL.replace("_t8_", f"_t{self.form_for(rows)['TOKEN_TILE']}_"),)

        def form_for(self, rows):
            cfg, geometry = self.built["cfg"], self.built["geometry"]
            tile = max(cfg.TOKEN_TILE, next(t for t in (8, 16, 32, 64) if rows <= t))  # the config's tile is a floor
            return dict(
                TOKEN_TILE=tile,
                PREWAIT_STAGES=cfg.PREWAIT_STAGES,
                L2_EVICT_FIRST=cfg.L2_EVICT_FIRST,
                MAX_CLUSTERS=cfg.MAX_CLUSTERS,
                PDL=bool(self.built["pdl"]),
                GEOM={"HIDDEN": geometry.HIDDEN, "INTERMEDIATE": geometry.INTERMEDIATE},
            )

        def launch_shape_for(self, rows):
            return SWIGLU_SHAPE

    return Swiglu


def _norm_symbol(form):
    """The symbol a stand-in instance of the norm entry states for a form: the form's text behind the kernel's name, as the
    entry writes it."""
    mode = "add" if form["ADD"] else ("store" if form.get("STORE_RESIDUAL") else "none")
    outputs = "_".join(part for part in ("bf16" if form["BF16_OUT"] else None, form["QUANT"]) if part)
    return f"cudnn_kernel_frost_add_rms_norm_quant_h64_{mode}_{outputs}_wait_release_eps358637bd_0"


def _swiglu_symbol(form):
    """The symbol a stand-in instance of the fused gate / up entry states for a form: the form's text behind the kernel's
    name, as the entry writes it."""
    policy, launch = ("evf" if form["L2_EVICT_FIRST"] else "dflt"), ("pdl" if form["PDL"] else "nopdl")
    return f"cudnn_kernel_frost_swiglu_fp4_decode_h64_i96_t{form['TOKEN_TILE']}_pw{form['PREWAIT_STAGES']}_{policy}_c{form['MAX_CLUSTERS']}_{launch}_0"


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
    projection_defect="",
    projection_serves=None,
    projection_shape=None,
    swiglu_defect="",
    swiglu_serves=None,
    swiglu_shape=None,
    on_stored=False,
    stored_defect="",
    out_defect="",
):
    """The provider factories of a device that exists only as a log: every provider of the provider rows is a stand-in that
    runs the reference's arithmetic and writes the launches a real provider of its row would make into the log; a stand-in
    profiler (it replaces ``loop.trace_steps``) hands the log over as the launch records of a traced pass.

    ``step_launches`` replaces what the stand-in of the fused step launches (default: the entry's own symbol in the shape the
    entry states); ``declares`` replaces the census entries it declares; ``serves`` what it says served a call (default: its
    one entry instance, which states the form of the row).  ``profiler``: ``"records"``, or the reason it cannot run -- or a
    list of these, one per traced pass, the last one for every further pass.  The stand-in of the gate entry runs the
    kernel's fp32 chain, states and launches ``gate_shape`` (a shape, or a function of the rows of a call) and is compared
    with the gate's real twin; ``gate_defect`` makes it write one wrong byte per call.

    The stand-in of the norm entry runs that kernel's fp32 chain in the form of the call's site -- one instance per form,
    the residual stream updated in place, the 4-bit block scales as the kernel forms them and in its 128 x 4 layout, the
    bf16 rows handed over unquantised where the form writes no codes -- states and launches one thread block per row (or
    ``norm_shape``) and is compared with the norm's real twin.  ``norm_serves(site)`` replaces what it says served a call; ``norm_defect`` makes
    it ``"extra"`` (launch one more kernel in front of its own), ``"missing"`` (not launch at the output head), ``"chain"`` (launch a
    chain of PyTorch kernels in place of its own), or write per call a wrong ``"code"``, block ``"scale"``, ``"bf16"``
    element or ``"residual"`` element.

    The stand-in of the FP8 projection entry multiplies the stored 8-bit codes of both operands in fp32 and applies the fp32
    product of the two scales, one rounding to bf16 -- on the merged codes the loader kept for it (``stored_weights``) --
    states and launches one thread block per 128 weight rows of the shape its row names for the call's site (or
    ``projection_shape``), one instance per site, and is compared with the reference multiply.  ``projection_serves(site)``
    replaces what it says served a call; ``projection_defect`` makes it ``"extra"`` (launch one more kernel in front of its
    own), ``"behind"`` (one more behind it), ``"missing"`` (not launch in an attention layer), ``"chain"`` (launch the
    kernels of a multiply over a copy in place of its own), write one wrong ``"element"`` per call, or -- ``"writes"`` --
    write into the stored weight it multiplies on (the first half of its rows are zero from its first call on; a test
    switches the same on later through the block's ``writes``).

    The stand-in of the fused gate / up entry runs that kernel's chain in fp32 on the weight the loader kept for it in
    the kernel's layout (``stored_weights``, ``stored_layout``: at the tiny widths tiles of 32 gate + 32 up rows, the
    block scales in the 128 x 4 layout; the entry's own helper needs tiles of 64 + 64) -- both products accumulated in
    fp32 and rounded once to bf16, SiLU x up rounded once more, the block scales as the kernel forms them, the codes under
    them -- hands its output over in the 128 x 4 layout, states and launches one thread block per 128 weight rows up to
    the cap (or ``swiglu_shape``) and is compared with the entry's real twin.  ``swiglu_serves`` replaces what it says
    served a call; ``swiglu_defect`` makes it ``"extra"`` (launch one more kernel in front of its own), ``"behind"`` (one
    more behind it), ``"missing"`` (not launch in the second layer), ``"chain"`` (launch the kernels of a multiply over a
    copy in place of its own), write per call a wrong ``"code"`` or a block ``"scale"`` one binade up, or -- ``"writes"``
    -- write into the stored weight it multiplies on (the first row of its block scales is zero from its first call on).

    ``on_stored`` (a test's own stand-ins: no provider row of the family does this for these blocks) puts every other
    multiply on the STORED FORMAT of its weight too: the stand-ins of the two multiplies over a copy then say
    ``stored_weights`` and decode the weight's code values from what the loader kept -- 8-bit codes, or packed 4-bit
    codes times their block scales -- accumulate in fp32 and round once, while they launch what their row's provider
    launches.  ``stored_defect`` names the stored container (``"ScaledTensor"``, ``"BlockScaledTensor"``) whose
    multiplies write one wrong element per call; or, ``"writes"``, makes each of them write into its stored weight (the
    first row of its last tensor -- the block scales of a 4-bit weight, the codes of an 8-bit one -- is zero from its
    first call on).

    A plan of several tokens per sequence has two more stand-ins.  The fused verify step runs the reference's verify
    step and launches the verify kernel's symbol in the recorded shape of one sequence of six tokens.  The FP8
    projection entry's stand-in serves the out-projection of the linear-attention layers too, on the row of the step's
    rows: it states and launches one thread block per K slice and 64 weight rows, with the shared memory of that row's
    kernel.  ``out_defect`` is a defect of that call: ``"split"`` (the kernel of the other row range: four K slices, its
    symbol, its shape, an instance that states its form), ``"chain"`` (the kernels of a multiply over a copy under the
    row's name), ``"behind"`` (one more launch behind its kernel), ``"missing"`` (no launch), ``"element"`` (one wrong
    element per call), ``"weight"`` (another weight of the same shape under the row's name: its rows in reverse),
    ``"writes"`` (it writes into the stored weight it multiplies on); and ``"attention"`` is a defect of the other call
    site: the attention layers' call, a multiply over a copy in the plan, launches the entry's kernel."""
    from model_samples import loop
    from model_samples.census import Facts, Kernel, Launch
    from model_samples.providers import factories, torch_ref
    from model_samples.timing import Trace
    from model_samples.weights import (
        BLOCK,
        InterleavedBlockScaledTensor,
        ScaledTensor,
        TiledBlockScaledTensor,
        Unquantized,
        _e2m1,
        _e2m1_values,
        _e4m3_values,
        dequantize,
        quantize_fp8,
        tile_scale_index,
        tile_scales,
        untile_scales,
    )

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

        def launch_facts(self, rows, site=""):
            return Facts((self._symbol(),), dict(self.row.form), ((96, rows, 1), (256, 1, 1), 42112))

        def serving(self):  # what the stand-in's own entry instance states
            return serves or ("cudnn.GdnDecodeFused #1", tuple(self.row.form))

        def __call__(self, x, z, hidden, slot_ids, params, slots):
            for name, shape in step_launches or [(self._symbol(), ((96, x.shape[0], 1), (256, 1, 1), 42112))]:
                launch(name, shape)
            return super().__call__(x, z, hidden, slot_ids, params, slots)

    class VerifyStep(torch_ref.GdnDecodeStep):
        """The stand-in of the fused verify step: the reference's verify step on the step's slot table, one launch per call
        under the verify kernel's symbol, in the launch shape recorded for one sequence of six tokens."""

        provider = "cuDNN FE"
        kernels = (Kernel("gdn_verify_fused", "cuDNN FE", "exact", launches=1, shape="entry"),)

        def launch_facts(self, rows, site=""):
            return Facts((VERIFY_SYMBOL,), dict(self.row.form), VERIFY_SHAPE)

        def serving(self):
            return "cudnn.GdnVerifyFused #1", tuple(self.row.form)

        def __call__(self, x, z, hidden, table, params, slots):
            launch(VERIFY_SYMBOL, VERIFY_SHAPE)
            return super().__call__(x, z, hidden, table, params, slots)

    class Gate(torch_ref.GateQuantize):
        """The stand-in of the gate entry: the kernel's own chain in fp32 -- the gated value as a quotient, rounded once to
        bf16, then the static 8-bit quantisation -- one launch per call, and the real twin to be compared with."""

        provider = "cuDNN FE"
        kernels = (Kernel("gate_quant", "cuDNN FE", "exact", launches=1, shape="entry"),)
        _twin = None

        def _shape(self, rows):  # one shape, or -- for a run of several row counts -- the shape of a row count
            return gate_shape(rows) if callable(gate_shape) else gate_shape

        def launch_facts(self, rows, site=""):
            return Facts((GATE_SYMBOL,), dict(self.row.form), self._shape(rows))

        def serving(self):
            return "cudnn.AttentionGateQuantDecode #1", tuple(self.row.form)

        def twin(self, reference):
            if self._twin is None:
                self._twin = torch_ref.GateQuantizeQuotient(reference.row, self.model, self.facts, self.options)
            return self._twin

        def __call__(self, o, gate, quant):
            launch(GATE_SYMBOL, self._shape(o.shape[0]))
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

    class Projection(torch_ref.Linear):
        """The stand-in of the FP8 projection entry: the kernel's own chain -- the code values of both operands multiplied
        and accumulated in fp32, the fp32 product of the two scales, one rounding to bf16 -- one launch per call, from the
        instance of the call's site; compared with the reference multiply."""

        provider = "cuDNN FE"
        kernels = (Kernel("fp8_projection", "cuDNN FE", "exact", launches=1, shape="entry"),)
        stored_weights = True
        writes = projection_defect == "writes"  # into the stored weight it multiplies on
        _site = ""

        #: the shared memory of a thread block by (K slices, ring stages, token tile), for a row that names its token tile
        SHARED = {(2, 11, 8): 103808, (4, 7, 16): 84352, (4, 7, 8): 71040, (4, 7, 32): 110976}

        def _shape(self, site):
            fields = dict(self.row.site_form(site))
            shared = self.SHARED[(fields["SPLIT_K"], fields["STAGES"], fields["TOKEN_TILE"])] if "TOKEN_TILE" in fields else 139520
            if site == "linear_attention.out":  # the out-projection of a verify step: its own launch, never another call's shape
                return (fields["OUT_FEATURES"] // fields["MMA_M"], fields["SPLIT_K"], 1), (256, 1, 1), shared
            return projection_shape or ((fields["OUT_FEATURES"] // fields["MMA_M"], fields["SPLIT_K"], 1), (256, 1, 1), shared)

        def _other_range(self, site):
            """The form of the row of the other row range of the out-projection: four K slices, a ring of 7, the 16-row tile."""
            other = {"SPLIT_K": 4, "STAGES": 7, "TOKEN_TILE": 16}
            return tuple((name, other.get(name, value)) for name, value in self.row.site_form(site))

        def launch_facts(self, rows, site=""):
            fields = dict(self.row.site_form(site))
            return Facts((_projection_symbol(fields, rows),), fields, self._shape(site))

        def serving(self):  # what the stand-in's own entry instance of the last call's site states
            if self._site == "linear_attention.out" and out_defect == "split":  # an instance of the other row range served the call
                return "cudnn.Fp8ProjectionDecode #2", self._other_range(self._site)
            if projection_serves is not None:
                return projection_serves(self._site)
            place = [site for site, _ in self.row.sites].index(self._site) + 1
            return f"cudnn.Fp8ProjectionDecode #{place}", tuple(self.row.site_form(self._site))

        def __call__(self, qx, weight, site=""):
            self._site, rows = site, qx.codes.shape[0]
            # the out-projection of a verify step has its own defects; every other call of the entry the in-projection's
            defect = out_defect if site == "linear_attention.out" else projection_defect
            if defect == "chain":
                for name in (ELEMENTWISE, NVJET, SPLIT_K, ELEMENTWISE):
                    launch(name)
            else:
                if defect == "extra":
                    launch(ELEMENTWISE)  # in front of its own kernel: behind the norm's
                if defect == "split":  # the kernel of the other row range, under its own symbol and in its own shape
                    launch(_projection_symbol(dict(self._other_range(site)), rows), OUT_SHAPE_SPLIT_4)
                elif not (defect == "missing" and site in ("full_attention.qkv", "linear_attention.out")):
                    launch(_projection_symbol(dict(self.row.site_form(site)), rows), self._shape(site))
                if defect == "behind":
                    launch(ELEMENTWISE)  # behind its own kernel: in front of its consumer's
            stored = weight.stored  # the merged codes with their one scale, kept by the loader for this block
            if self.writes or defect == "writes":  # the same bytes after every call: the weight the block multiplies on is no longer the loaded one
                stored.codes[: stored.codes.shape[0] // 2] = 0
            codes = stored.codes.flip(0) if defect == "weight" else stored.codes  # another weight of this shape under the row's name
            scale = torch.tensor(float(qx.scale), dtype=torch.float32) * torch.tensor(float(stored.scale), dtype=torch.float32)
            out = ((_e4m3_values(qx.codes) @ _e4m3_values(codes).T) * scale).to(torch.bfloat16)
            if defect == "element":
                out[0, 0] = out[0, 0].float() * 1.5 + 1.0
            return out

    class Swiglu(torch_ref.GateUp):
        """The stand-in of the fused gate / up entry: the kernel's own chain in fp32, on the stored codes the loader kept
        in the kernel's layout and never on a copy -- both products accumulated in fp32 with the fp32 product of the two
        scales and rounded once to bf16, SiLU x up rounded once more, the block scales and the codes of that value -- one
        launch per call, the output in the 128 x 4 layout, and the entry's real twin."""

        provider = "cuDNN FE"
        kernels = (Kernel("swiglu_fp4", "cuDNN FE", "exact", launches=1, shape="entry"),)
        stored_weights = True
        tile = 32  # gate rows per weight tile at the tiny widths
        writes = swiglu_defect == "writes"  # into the stored weight it multiplies on
        _twin = None
        _layer = -1  # of the last call: the block is called once per layer, in layer order

        def stored_layout(self, merged):
            n = merged.codes.shape[0]

            def interleave(rows):  # [gate rows, up rows] -> tiles of [gate, up]
                return rows.reshape(2, n // (2 * self.tile), self.tile, -1).permute(1, 0, 2, 3).reshape(n, -1).contiguous()

            return InterleavedBlockScaledTensor(interleave(merged.codes), tile_scales(interleave(merged.block_scales)), float(merged.scale), self.tile)

        def _shape(self):
            return swiglu_shape or SWIGLU_SHAPE

        def launch_facts(self, rows, site=""):
            return Facts((_swiglu_symbol(dict(self.row.form)),), dict(self.row.form), self._shape())

        def serving(self):  # what the stand-in's own entry instance states
            return swiglu_serves or ("cudnn.SwigluFp4Decode #1", tuple(self.row.form))

        def twin(self, reference):
            if self._twin is None:
                self._twin = torch_ref.GateUpFp4Bound(reference.row, self.model, self.facts, self.options)
            return self._twin

        def compared(self, out):
            return out

        def __call__(self, qx, weight, quant):
            self._layer = (self._layer + 1) % len(self.model.geometry.layer_kinds)
            if swiglu_defect == "chain":
                for name in (ELEMENTWISE, NVJET, ELEMENTWISE):
                    launch(name)
            else:
                if swiglu_defect == "extra":
                    launch(ELEMENTWISE)  # in front of its own kernel: behind the norm's
                if not (swiglu_defect == "missing" and self._layer == 1):
                    launch(_swiglu_symbol(dict(self.row.form)), self._shape())
                if swiglu_defect == "behind":
                    launch(ELEMENTWISE)  # behind its own kernel: in front of its consumer's
            stored = weight.stored  # the merged weight in the kernel's layout, kept by the loader for this block
            if self.writes:  # the same bytes after every call: the weight the block multiplies on is no longer the loaded one
                stored.tiled_scales[0] = 0
            n, half = stored.codes.shape[0], stored.codes.shape[0] // 2
            scales = untile_scales(stored.tiled_scales, n, stored.codes.shape[1] * 2 // BLOCK)
            nibbles = torch.stack((stored.codes & 0xF, stored.codes >> 4), dim=-1).flatten(-2)
            tiles = (_e2m1_values(nibbles) * _e4m3_values(scales).repeat_interleave(BLOCK, dim=-1)).view(n // (2 * stored.tile), 2, stored.tile, -1)
            x = dequantize(qx)
            alpha = torch_ref.fp32_product(x.scale, stored.scale)
            gate = ((x.data.float() @ tiles[:, 0].reshape(half, -1).T) * alpha).to(torch.bfloat16)
            up = ((x.data.float() @ tiles[:, 1].reshape(half, -1).T) * alpha).to(torch.bfloat16)
            # SiLU x up and the quantisation as the reference pair forms them from these two products: inside the entry's
            # error bound, and on an exact tie the reference's choice -- so the token line of the tiny geometry (rows 64
            # wide, where one 4-bit code moves a logit by several units) is held to the family's band as it is with the
            # reference chain
            q = torch_ref.silu_mul_quantize(torch.cat((gate, up), dim=-1), quant)
            if swiglu_defect == "scale":
                q.block_scales[0, 0] += 8  # one binade up: of no block maximum the error bound admits
            if swiglu_defect == "code":  # the first element's code: the other sign, or the largest code where it is a zero
                q.codes[0, 0] = q.codes[0, 0] ^ 0x8 if q.codes[0, 0] & 0x7 else q.codes[0, 0] | 0x7
            return TiledBlockScaledTensor(q.codes, tile_scales(q.block_scales), float(quant.scale))

    def on_stored_format(qx, weight):
        """A multiply on the stored format of its weight, as a kernel there makes it: the weight's code values decoded from
        the stored tensor -- 8-bit codes, or 4-bit nibbles (the low one first) times the scale of their block of 16 --
        never read from a copy; both operands' code values multiplied and accumulated in fp32, the fp32 product of the two
        scales, one rounding to bf16."""
        stored = weight.stored
        if stored_defect == "writes":
            stored.tensors()[-1][0] = 0
        if isinstance(stored, ScaledTensor):
            values = _e4m3_values(stored.codes)
        else:
            nibbles = torch.stack((stored.codes & 0xF, stored.codes >> 4), dim=-1).flatten(-2)
            values = _e2m1_values(nibbles) * _e4m3_values(stored.block_scales).repeat_interleave(BLOCK, dim=-1)
        x = dequantize(qx)
        scale = torch.tensor(float(x.scale), dtype=torch.float32) * torch.tensor(float(stored.scale), dtype=torch.float32)
        out = ((x.data.float() @ values.T) * scale).to(torch.bfloat16)
        if stored_defect == type(stored).__name__:
            out[0, 0] = out[0, 0].float() * 1.5 + 1.0
        return out

    class StoredLinear(torch_ref.Linear):
        """A test's own multiply on the stored format of its weight, on the row of a multiply over a copy: it launches what
        that row's provider launches and reads no copy."""

        stored_weights = True

        def __call__(self, qx, weight, site=""):
            for name in (ELEMENTWISE, NVJET, SPLIT_K, ELEMENTWISE):
                launch(name)
            return on_stored_format(qx, weight)

    class StoredGateUp(torch_ref.GateUp):
        """The same for the merged gate / up multiply: the product on the packed 4-bit weight, then the chain's SiLU x up."""

        stored_weights = True

        def __call__(self, qx, weight, quant):
            for name in (ELEMENTWISE, NVJET, ELEMENTWISE):
                launch(name)
            self.product = on_stored_format(qx, weight)
            return torch_ref.silu_mul_quantize(self.product, quant)

    class OverCopies(torch_ref.Linear):
        """The stand-in of a multiply over a copy: the kernels such a multiply launches, then the reference's arithmetic.
        ``out_defect == "attention"``: at the attention layers' out-projection it launches the FP8 projection kernel
        instead -- the entry serving a call its row hands on."""

        def __call__(self, qx, weight, site=""):
            if out_defect == "attention" and site == "full_attention.out":
                fields = {"OUT_FEATURES": 5120, "IN_FEATURES": 6144, "SPLIT_K": 2, "MMA_M": 64, "STAGES": 11, "TOKEN_TILE": 8}
                launch(_projection_symbol(fields, qx.codes.shape[0]), OUT_SHAPE)
            else:
                for name in (ELEMENTWISE, NVJET, SPLIT_K, ELEMENTWISE):
                    launch(name)
            return super().__call__(qx, weight, site)

    over_copies = OverCopies
    stand_ins = {
        "fe.gdn_decode_fused": Step,
        "fe.gdn_verify_fused": VerifyStep,
        "fe.attention_gate_quant_decode": Gate,
        "fe.add_norm_quant_decode": Norm,
        "fe.fp8_projection_decode": Projection,
        "fe.swiglu_fp4_decode": Swiglu,
        "cublas.bf16_dequantized": StoredLinear if on_stored else over_copies,
        "cublas.bf16_dequantized+torch_ref.silu_mul_quantize": StoredGateUp if on_stored else logging(torch_ref.GateUp, ELEMENTWISE, NVJET, ELEMENTWISE),
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


def _verify_run(monkeypatch, sequences=1, tokens=6, decode=True, steps=3, stack="native", **stand_in):
    """Verify steps of the tiny geometry on the CPU through the rows of ``stack``, on the stand-in providers and the stand-in
    profiler of :func:`_stand_ins` (``stand_in``: its options; the gate's stand-in launches the shape recorded for the
    rows of each call): ``steps`` steps of ``tokens`` tokens per sequence, each keeping another number of them.  With ``decode`` a
    session of one token per sequence is planned in front of it and loads, as a run with both kinds of step does; without,
    the verify session loads for its own plan alone.  Returns ``(verify session, decode session or None)`` with the run
    recorded and the traced pass not yet made."""
    from model_samples import Session, TextIO, blocks, census

    from model_samples.qwen_dense import QwenDense

    stand_in.setdefault("gate_shape", lambda rows: ((3 * rows, 1, 1), (256, 1, 1), 0))  # three rows per thread block: (3, 1, 1) at one row
    build = _stand_ins(monkeypatch, **stand_in)
    monkeypatch.setattr(census, "installed_versions", lambda packages: dict(RECORDED))
    model = _tiny_model()
    monkeypatch.setattr(QwenDense, "_target", _census_of(model))
    text = TextIO.synthetic(model.geometry.vocab)
    facts = blocks.DeviceFacts((10, 0), "stated device", 148, 0, 0)

    def session_of(per_sequence, share=None):
        session = Session(model, text, sequences=sequences, stack=stack, device="cpu", capacity=40, check_kernels=True, tokens=per_sequence, share=share)
        options = {"reference_precision": "fp64", "device": "cpu", "check_kernels": True}
        session.plan = blocks.plan(model.provider_table, model.regimes, facts, sequences, per_sequence, stack, model=model, options=options, factories=build)
        return session

    first = session_of(1) if decode else None
    verify = session_of(tokens, share=first)
    if first is not None:
        first._followers = (verify,)  # as ``plan_table([verify])`` leaves it: the load keeps what both plans read
        first.load()
    verify.load()
    generator = torch.Generator().manual_seed(0)
    for step in range(steps):
        ids = torch.randint(0, model.geometry.vocab, (sequences, tokens), generator=generator, dtype=torch.int32)
        verify.read(verify.verify(ids, step % tokens))
    return verify, first


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
    out = capsys.readouterr().err
    assert ok and judged == 6 and session.traced.status == census.NOT_OBSERVED and session.traced.unobserved == "needs a recorded step on a CUDA device"
    assert "the tokens); the kernels of the traced steps against the plan" in out, "the line before the checks says that the kernels are checked too"
    assert "  kernels           not observed: needs a recorded step on a CUDA device    NOT OBSERVED" in out
    # whether the kernels are checked is the session's decision, read in one place: a session that does not check them prints no line
    session, text, prompts = _recorded_run(monkeypatch, check_kernels=False, profiler="needs a recorded step on a CUDA device")
    session.reset(fresh=True)
    session.new_report()
    capsys.readouterr()
    ok, judged, _ = sample._decode(session, text, StopRule(text.eos_ids, force_tokens=4), prompts, tie_out, check=True, checked_steps=None)
    assert ok and "  kernels  " not in capsys.readouterr().err, "without the flag the run prints no kernel line"


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
        return module.main(arguments), capsys.readouterr().err
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
    out = capsys.readouterr().err
    served, kernels = out.index("served by (decided before the load"), out.index(
        "kernels, planned (decided before the load from model_samples/qwen_dense/configs.py"
    )
    assert served < kernels and "cudnn.GdnDecodeFused [PDL False, PDL_RELEASE False]" in out[kernels:] and "kernels, as the entries state them" not in out
    with pytest.raises(AssertionError, match="reached a load"):
        _sample_main(monkeypatch, capsys, [])
    assert "kernels, planned" not in capsys.readouterr().err
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
    return code, capsys.readouterr().err.splitlines()


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
    # the norm block and the in-projection are called at named sites in every stack; here no entry serves them, so there is no
    # instance to count and no such line
    assert not any("as the call log of a traced step has it" in line for line in lines)

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
        "projection.fp8.in": "fp8_projection",
        "linear_attention.step": "gdn_decode_fused",
        "mlp.gate_up": "swiglu_fp4",
        "attention.gate_quantize": "gate_quant",
    }
    for block, kernel_id in served.items():
        meta = target.data["kernels"][kernel_id]
        same = "a change of a form the kernel census names updates kernel_census.json in the same commit"
        # the form the kernel has everywhere; a field that follows the row count (the fused gate / up kernel's token tile) is
        # named on the entries of the section's row count, below
        everywhere = {name: value for name, value in plan.rows[block].form if not (block == "mlp.gate_up" and name == "TOKEN_TILE")}
        assert everywhere == meta.get("form", {}), f"{block}: the form of its native row in configs.py and of {kernel_id} in the census differ; {same}"
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
    # the in-projection is one form at two weight shapes: the census names the shape of every call, the row the shape of every
    # call site, and the launch shape follows it (one thread block per 128 weight rows)
    by_call = [(site, facts) for (block, _, facts), site in zip(calls, full.sites()) if block == "projection.fp8.in"]
    assert len(by_call) == 64 and [site for site, _ in by_call[:4]] == ["linear_attention.qkvz"] * 3 + ["full_attention.qkv"]
    for site, facts in by_call:
        wanted = {**target.data["kernels"]["fp8_projection"]["form"], **facts["form"]}
        assert dict(full_plan.rows["projection.fp8.in"].site_form(site)) == wanted, f"projection.fp8.in at {site}: configs.py and the census differ; {same}"
        assert tuple(map(tuple, facts["launch"][:2])) + (facts["launch"][2],) == PROJECTION_SHAPE[facts["form"]["OUT_FEATURES"]], site
        assert facts["launch"][0][0] * 128 == facts["form"]["OUT_FEATURES"], "the grid is the weight's rows over the weight tile"
    # the fused gate / up kernel is one form per row count: the census names the token tile on every call of its 1-row section,
    # the table has one row per tile, and for every call of the 64-layer model the row of 1 row and the census name one form
    by_call = [facts for block, _, facts in calls if block == "mlp.gate_up"]
    assert len(by_call) == 64 and all(facts["form"] == {"TOKEN_TILE": 8} for facts in by_call)
    assert all(tuple(map(tuple, facts["launch"][:2])) + (facts["launch"][2],) == SWIGLU_SHAPE for facts in by_call), "one thread block per SM of a 148-SM part"
    wanted = {**target.data["kernels"]["swiglu_fp4"]["form"], **by_call[0]["form"]}
    assert dict(full_plan.rows["mlp.gate_up"].form) == wanted, f"mlp.gate_up: configs.py and the census differ; {same}"
    assert census.form_text(full_plan.rows["mlp.gate_up"].form).strip() == SWIGLU_FORM
    assert configs.TARGET_STACKS == QwenDense.target_stacks == ("native", "fe-entries")
    # which kernels a stack of this repository calls an entry for is said per kernel, and held here: the three numbers count
    # the complete stack, and a reader of the file must be able to tell what exists today
    kernels = target.data["kernels"]
    from model_samples import blocks

    several = blocks.plan(model.provider_table, model.regimes, plan.facts, 1, 6, "native", model=model)  # the plan of a verify step: six tokens per sequence
    stepped = {kernel.id for block in several.values() for kernel in block.kernels if kernel.relation == "exact"}
    assert stepped - set(served.values()) == {"gdn_verify_fused"}, "the verify step has one kernel the decode step has not"
    assert {kernel_id for kernel_id, meta in kernels.items() if meta["entry"] is not None} == set(served.values()) | stepped
    assert all("entry" in meta for meta in kernels.values()), "null where no stack calls an entry for the kernel yet"
    for kernel_id, meta in kernels.items():
        ported = meta.get("kind") == "port" and meta["entry"] is not None
        assert ("tie_out" in meta) == ("differences" in meta) == ported, f"{kernel_id}: a port a stack calls says where its byte equality is recorded"
        if ported:
            assert isinstance(meta["tie_out"], str) and "is not a test of this repository" in meta["tie_out"] and meta["differences"], kernel_id
            holds = {
                "gate_quant": ("docs/operations/attention/gate_quant_decode.md", "test/python/attention_gate/frost"),
                "add_norm_quant": ("docs/operations/normalizations/add_norm_quant_decode.md", "test/python/norm/frost"),
                "fp8_projection": ("docs/operations/gemm_fusions/fp8_projection_decode.md", "test/python/gemm/frost/projection_decode"),
                "swiglu_fp4": ("docs/operations/gemm_fusions/swiglu_fp4_decode.md", "test/python/gemm/frost/swiglu_decode"),
            }[kernel_id]
            for path in holds:
                assert path in meta["tie_out"] and (MODELS.parents[1] / path).exists(), f"{kernel_id}: what the repository holds for the entry"
    assert "on compute capability 10.0 -- no differing byte" in kernels["add_norm_quant"]["tie_out"], "the class the comparison was made on"
    assert len(kernels["add_norm_quant"]["differences"]) == 5, "what differs from the target kernel besides its results, as the entry lists it"
    assert "on compute capability 10.0 -- no differing bit" in kernels["fp8_projection"]["tie_out"] and len(kernels["fp8_projection"]["differences"]) == 11
    swiglu = kernels["swiglu_fp4"]
    assert "on compute capability 10.0 -- no differing byte" in swiglu["tie_out"] and len(swiglu["differences"]) == 10
    assert sum("were not timed against each other" in item for item in swiglu["differences"]) == 1, "its instruction stream is its own: said, with no timing"
    assert [
        kernel_id for kernel_id, meta in kernels.items() if meta.get("kind") == "port" and meta["entry"] is None
    ] == [], "every port of the set has its entry"
    readme = " ".join((MODELS / "qwen_dense" / "README.md").read_text().split())
    for sentence in (
        "These are the numbers of the complete stack, not of today's. On compute capability 10.0 this stack launches all 273 today: the norm "
        "entry's 129, the projection entry's 64, the fused gate / up entry's 64 and the gate entry's 16.",
        "for the fused gate / up entry it was made on compute capability 10.0",
        'And "byte-equal" is the entry\'s own record, not something this sample measures',
        "for the gate entry it was made on compute capability 10.3, and none is recorded for 10.0",
        "for the norm entry it was made on compute capability 10.0",
        "for the projection entry it was made on compute capability 10.0",
        # two fields of the projection row's form are the port's own: the reader is told what the target kernel has there
        "Two fields of that form are the entry's own, and their values are not the target kernel's: the tensor-memory columns and their release "
        "(`TMEM_COLS 32`, `HOLD_TMEM True`) are at the values the entry's byte comparison with the target kernel ran, while the target kernel "
        "allocates 256 columns and releases them right after the accumulator read. The census lists both under `differences`",
        "32 tensor-memory columns released late (the entry's own two fields: the target kernel allocates 256 and releases them early)",
    ):
        assert sentence in readme, sentence
    projection = kernels["fp8_projection"]
    assert (projection["form"]["TMEM_COLS"], projection["form"]["HOLD_TMEM"]) == (32, True), "the values of the entry's byte comparison"
    assert sum("where the target kernel allocates 256" in item for item in projection["differences"]) == 1, "and what the target kernel has instead"
    assert sum("the target kernel releases it right after the accumulator read" in item for item in projection["differences"]) == 1
    about = " ".join(target.data["about"])
    assert "with an entry comes 'form': what the target passes to it and, for a field only the port has, the value the port's byte comparison ran with" in about
    assert "A port's byte equality is the entry's own record, not a measurement of the sample that reads this file" in about
    assert "for a kernel whose form follows the row count, the field of the section's row count" in about
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
    form: 322 of the 484 launches of a step of the 64-layer model are target kernels, 49 the target's own compiled kernels
    and 273 cuDNN Frontend entries with byte-equal results (the norm entry's 129, in five forms, the projection entry's 64,
    at two weight shapes, the fused gate / up entry's 64 and the gate entry's 16); every other block is ``not yet``.
    Always three numbers."""
    from model_samples import census
    from model_samples.qwen_dense import QwenDense, configs

    plan, model, target = _target_plan(64)
    claim = target.claim(plan, model, installed=RECORDED)
    numbers = claim.numbers()
    assert (numbers["launches"], numbers["code"], numbers["port"], numbers["stand_ins"]) == (484, 210, 273, 1)
    assert (numbers["target_code"], numbers["target_port"]) == (49, 273) and numbers["not_the_target"] == {}
    assert numbers["target_port"] == numbers["port"], "on this class every port of the set is launched through its entry"
    assert numbers["not_yet"] == {
        "projection.fp8.out": 64,
        "mlp.down": 64,
        "attention.prepare": 16,
        "attention.core": 16,
        "output_head": 1,
    }
    assert sum(numbers["not_yet"].values()) + 49 + 273 + 1 == 484 and numbers["stand_ins_not_yet"] == ["output_head.quantize"]
    whole, reached = claim.coverage()
    assert whole == (
        WHOLE + "484 launches per step = 210 the target's own compiled kernels + 273 cuDNN Frontend entries with byte-equal results + 1 declared stand-in"
    ), "the three numbers of the complete stack, said to be that: the 210 are not all launched by a stack today"
    assert reached == (
        "kernel identity, as planned: 322 of 484 launches of a step are target kernels (49 the target's own compiled kernels + 273 cuDNN Frontend entries "
        "with byte-equal results); not yet: projection.fp8.out 64, mlp.down 64, "
        "attention.prepare 16, attention.core 16, output_head 1; 1 declared stand-in (not yet: output_head.quantize)"
    )
    readme = (MODELS / "qwen_dense" / "README.md").read_text()
    assert whole in readme and reached in readme, "the two lines of the family README are the lines a run prints"
    assert configs.readme_table(QwenDense().calls_text(), "native") in readme, "regenerate the native table of the README from configs.readme_table()"
    assert configs.readme_table(QwenDense().calls_text(), "fe-entries") == configs.readme_table(QwenDense().calls_text(), "native"), "the same rows today"
    # the table is rendered for one GPU class: its heading names that class, and the sentence under it is the plan's own
    # by-provider line there; what the other classes plan instead is said with their plans' counts
    flat = " ".join(readme.split())

    def said(planned, tail=True):
        line = planned.by_provider().replace("by provider: ", "By provider: ").replace("cuDNN FE ", "cuDNN Frontend ")
        entries, natural = line.rsplit(" | PyTorch ", 1)
        return entries.replace(" | ", "; ") + (f"; PyTorch as the natural provider {natural}." if tail else "")

    table = configs.readme_table(QwenDense().calls_text(), "native", (10, 0))
    heading = "## Who serves each block (stacks `native` and `fe-entries`, compute capability 10.0)"
    assert (
        " ".join(f"{heading} {table} {said(plan)}".split()) in flat
    ), "the heading, the table and the by-provider sentence are those of compute capability 10.0"
    assert said(plan) == "By provider: cuDNN Frontend 5; fallbacks 5 (cuBLAS over copies 3, PyTorch chains 2); PyTorch as the natural provider 2."
    for cc, sms in (((10, 3), 152), ((10, 7), 212)):
        other = said(_target_plan(64, cc=cc, sms=sms)[0], tail=False).replace("By provider: ", "")
        assert other == "cuDNN Frontend 4; fallbacks 5 (cuBLAS over copies 3, PyTorch chains 2)", cc
        assert f"On the other 10.x classes the in-projection is the multiply over a copy of `fe-cublas`: {other}." in flat, cc
    # the table shows the row of 1 to 8 rows of the fused gate / up entry; the sentence under it names the other three
    tiles = "; ".join(f"`TOKEN_TILE {tile}` from {first} to {last} rows" for first, last, tile in configs.GATE_UP_TILES[1:])
    assert f"The row of the MLP's gate / up multiply is the one of 1 to 8 rows; three more rows of the same key name the other token tiles: {tiles}." in flat
    for sequences, (first, last, tile) in ((n, row) for row in configs.GATE_UP_TILES for n in row[:2]):
        row = _target_plan(2, sequences=sequences)[0].rows["mlp.gate_up"]
        assert row.key == "fe.swiglu_fp4_decode" and dict(row.form)["TOKEN_TILE"] == tile and row.rows == (first, last), sequences
    other = said(_target_plan(64, cc=(12, 0), sms=170)[0], tail=False).replace("By provider: ", "")
    assert f"the reference chains of `fe-cublas` as well: {other}." in flat and other.startswith("cuDNN Frontend 1; fallbacks 8 "), "compute capability 12.x"
    default = _target_plan(64, stack="fe-cublas")[0]
    assert f"34 on the four-layer one. {said(default)}" in flat, "the sentence under the table of the default stack"
    shapes = {"embedding": GATHER_SHAPE, "add_norm_quantize": NORM_SHAPE, "linear_attention.step": SHAPE, "attention.gate_quantize": GATE_SHAPE}
    shapes["mlp.gate_up"] = SWIGLU_SHAPE
    kinds = model.geometry.layer_kinds
    assert claim.recorded == {
        # an in-projection's recorded shape is the one of its layer's weight: grid 128 in a linear-attention layer, 112 in an attention layer
        index: (PROJECTION_SHAPE[model.geometry.in_projection(kinds[layer])[0]] if block == "projection.fp8.in" else shapes[block])
        for index, (block, layer) in enumerate(model.wiring())
        if block in shapes or block == "projection.fp8.in"
    }, "the recorded launch shape of every target launch: 322 of them, as tuples"
    assert len(claim.recorded) == 322
    assert configs.readme_sites(QwenDense(), "native") in readme, "regenerate the call-site table of the README from configs.readme_sites()"
    from collections import Counter

    forms = Counter(site for (block, _), site in zip(model.wiring(), model.sites()) if block == "add_norm_quantize")
    assert forms == {"linear_attention.first": 1, "linear_attention": 47, "full_attention": 16, "mlp": 64, "output_head": 1}, "five forms: 1 / 47 / 16 / 64 / 1"
    shapes_called = Counter(site for (block, _), site in zip(model.wiring(), model.sites()) if block == "projection.fp8.in")
    assert shapes_called == {"linear_attention.qkvz": 48, "full_attention.qkv": 16}, "two weight shapes: 48 / 16"

    plan, model, _ = _target_plan(2)
    claim = target.claim(plan, model, installed=RECORDED)
    assert claim.coverage()[1].startswith(
        "kernel identity, as planned: 12 of 20 launches of a step are target kernels (2 the target's own compiled kernels + 10 cuDNN Frontend entries "
        "with byte-equal results); not yet: projection.fp8.out 2, mlp.down 2, "
    )
    kinds = {block: (identity.relation, identity.kind) for block, identity in zip(claim.blocks, claim.identities)}
    for block in (
        "add_norm_quantize",
        "projection.fp8.in",
        "linear_attention.step",
        "mlp.gate_up",
        "attention.gate_quantize",
    ):  # the class their target forms ran on: 2 or more sequences gate them
        assert plan.rows[block].batch_invariant == ("cc10.0",), f"{block}: batch invariance of its native row is a gate on compute capability 10.0"
    assert kinds["embedding"] == kinds["linear_attention.step"] == ("target", "code") and kinds["attention.gate_quantize"] == ("target", "port")
    assert kinds["token_choice"] == ("host", "") and kinds["mlp.down"] == ("not yet", "code") and kinds["add_norm_quantize"] == ("target", "port")
    assert kinds["mlp.gate_up"] == ("target", "port")
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
    multiply = f"{PROJECTION_FORM}] + one of 2 forms, by call site fp8_projection 2 name and launch shape, as the entry states them target"
    assert row["projection.fp8.in"].split("cudnn.Fp8ProjectionDecode")[1].split() == multiply.split()
    assert table[20:22] == [f"  projection.fp8.in at {site}:{census.form_text(fields)}" for site, fields in configs.PROJECTION_SITES], "and the weight's shape"
    fused = f"{SWIGLU_FORM} swiglu_fp4 2 name and launch shape, as the entry states them target"
    assert row["mlp.gate_up"].split("cudnn.SwigluFp4Decode")[1].split() == fused.split(), "the kernel stated whole, the token tile of the row count in it"
    assert table[-3].startswith("  per step: 12 launches on 6 exact rows") and table[-2:] == [f"  {line}" for line in claim.coverage()]
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
    assert 3 not in differs.recorded and len(differs.recorded) == 11, "no recorded launch shape is held against a call that is not the target's"
    # compute capability 12.x: the gate and the norm entry do not serve there, the blocks stay on their reference chains, and the census has no section
    plan12, model, _ = _target_plan(2, cc=(12, 0), sms=170)
    assert plan12.rows["attention.gate_quantize"].key == "torch_ref.gate_quantize" and plan12.rows["linear_attention.step"].form == (
        ("PDL", True),
        ("PDL_RELEASE", True),
    )
    assert plan12.rows["add_norm_quantize"].key == "torch_ref.add_norm_quantize" and plan12.rows["add_norm_quantize"].site_form("mlp") == ()
    assert plan12.rows["projection.fp8.in"].key == "cublas.bf16_dequantized" and plan12.rows["projection.fp8.in"].site_form("full_attention.qkv") == ()
    assert plan12.rows["mlp.gate_up"].key == "cublas.bf16_dequantized+torch_ref.silu_mul_quantize" and plan12.rows["mlp.gate_up"].form == ()
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
    assert there.numbers()["not_yet"]["add_norm_quantize"] == 5 and there.numbers()["not_yet"]["projection.fp8.in"] == 2
    assert there.numbers()["not_yet"]["mlp.gate_up"] == 2


def test_every_reason_a_class_is_not_claimed():
    """The census line says ``NOT CLAIMED`` with every reason that applies, in a fixed order: a row selected with
    ``--provider``, fallback rows, a kernel of the target the plan has no block for, no recorded section for the GPU class
    and SM count (both are named) or for the row count, versions other than the recorded ones."""
    from model_samples import census

    plan, model, target = _target_plan()
    claim = target.claim(plan, model, installed=RECORDED)
    fallback, absent = "fallback rows present", "the plan has no block for output_head.quantize"
    assert claim.reasons == [fallback, absent] and claim.name == CENSUS and (claim.basis, claim.claimed, claim.rows) == ("recorded", False, 1)
    assert claim.line(census.PASS) == f"{CENSUS}: NOT CLAIMED ({fallback}; {absent}); {COMPARED_12}: each in the recorded one"
    assert claim.compared() == (12, COMPARED_12, 0), "what item 7 compares: a count of target launches is three numbers here too"
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
    clean = census.Traced.of(sites, _log(sites), _step(), 1, "", rows=1, claim=claim)
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
    traced = census.Traced.of(moved, _log(moved), _step(fused=(SYMBOL, other)), 1, "", rows=1, claim=claim)
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
        status, lines = census.Traced.of(moved, _log(moved), _step(fused=(SYMBOL, other)), 1, "", rows=1, claim=claim).judge()
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
    status, lines = census.Traced.of(sites, _log(sites), step, 1, "", rows=1, claim=claim).judge()
    assert status == census.FAIL and lines == [
        f"step 0: embedding launched embedding_gather with grid (80, 1, 1), block (128, 1, 1), shared memory 0 B; the {CENSUS} has grid (40, 1, 1), "
        "block (128, 1, 1), shared memory 0 B"
    ]
    # records without launch arguments: the recorded shapes were not observed
    bare = [record._replace(grid=None, block=None, shared_memory=None) for record in _step()]
    unstated = [replace(site, facts=None) if site.facts else site for site in sites]
    unseen = census.Traced.of(unstated, _log(unstated), bare, 1, "", rows=1, claim=claim)
    assert unseen.judge() == (census.NOT_OBSERVED, []) and unseen.details()[-1].endswith("; the launch shapes of its target launches were not observed")
    # a call that is not the target's (another form, a fallback row) is held to no recorded shape -- whatever its identity carries
    reason = "the row's form is [PDL False, PDL_RELEASE False], the target's [PDL True, PDL_RELEASE True]"
    claim.identities[2] = Identity(census.OTHER, "gdn_decode_fused", "code", reason, SHAPE)
    assert claim.recorded == {0: GATHER_SHAPE}, "only a call whose identity is target (or stand-in) is held to the census"
    assert census.Traced.of(moved, _log(moved), _step(fused=(SYMBOL, other)), 1, "", rows=1, claim=claim).judge() == (census.PASS, [])


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
    assert lines[planned + 20 : planned + 22] == [
        "  projection.fp8.in at linear_attention.qkvz: [OUT_FEATURES 16384, IN_FEATURES 5120]",
        "  projection.fp8.in at full_attention.qkv: [OUT_FEATURES 14336, IN_FEATURES 5120]",
    ], "the weight's shape at each call site of the in-projection"
    assert lines[planned + 23].strip().startswith(WHOLE + "20 launches per step = 9 ")
    reached = (
        "kernel identity, as planned: 12 of 20 launches of a step are target kernels (2 the target's own compiled kernels + 10 cuDNN Frontend entries with "
    )
    assert lines[planned + 24].strip().startswith(reached + "byte-equal results); not yet: projection.fp8.out 2, mlp.down 2, ")
    (kernels,) = _kernel_lines(lines)
    assert kernels.endswith("    PASS") and "12 on exact rows, in the wiring order" in kernels
    not_claimed = f"    {CENSUS}: NOT CLAIMED (fallback rows present; the plan has no block for output_head.quantize)"
    assert not_claimed + f"; {COMPARED_12}: each in the recorded one" in lines
    gate = next(line for line in lines if line.strip().startswith("attention.gate_quantize ") and "calls" in line)
    assert gate.endswith("[band: gate_fp8]  PASS") and "0 the code of neither" in gate, gate
    norm = next(line for line in lines if line.strip().startswith("add_norm_quantize ") and "calls" in line)
    assert norm.endswith("[band: add_norm_quant]  PASS") and "residual stream equal" in norm and norm.split("; ")[-1].startswith("0 outside"), norm
    assert any(line.startswith("  blocks            8 of 8 provider blocks inside their bands") and line.endswith("PASS") for line in lines)
    multiply = next(line for line in lines if line.strip().startswith("projection.fp8.in ") and "calls" in line)
    assert multiply.endswith("[band: bf16_rounding]  PASS"), multiply
    fused = next(line for line in lines if line.strip().startswith("mlp.gate_up ") and "calls" in line)
    assert fused.endswith("[band: swiglu_fp4]  PASS") and "; 0 outside their bounds" in fused, fused
    assert any(
        line.strip().startswith(
            f"mlp.gate_up: cudnn.SwigluFp4Decode {SWIGLU_FORM}, 2 launches per step (2 calls x 1 per call); the entry states grid (1, 1, 148), "
        )
        for line in lines
    ), "the fused gate / up kernel: one launch per MLP, in the shape the entry states"
    assert (
        f"    projection.fp8.in, as the call log of a traced step has it (2 calls): 1 x cudnn.Fp8ProjectionDecode #1 {PROJECTION_FORM}, OUT_FEATURES 16384, "
        f"IN_FEATURES 5120]; 1 x cudnn.Fp8ProjectionDecode #2 {PROJECTION_FORM}, OUT_FEATURES 14336, IN_FEATURES 5120]"
    ) in lines, "one instance per weight shape, each stating the row's form and its own shape"
    stated = [line.strip() for line in lines if line.strip().startswith("projection.fp8.in at ") and "the entry states" in line]
    assert [line.split("; the entry states ")[1].split(" and the kernel ")[0] for line in stated] == [
        "grid (128, 1, 1), block (256, 1, 1), shared memory 139,520 B",
        "grid (112, 1, 1), block (256, 1, 1), shared memory 139,520 B",
    ], "one thread block per 128 weight rows: the launch shape says which weight"
    served = next(line.strip() for line in lines if line.strip().startswith("add_norm_quantize, as the call log of a traced step has it (5 calls): "))
    assert [part.split(" x ")[0] for part in served.split(": ", 1)[1].split("; ")] == ["1", "2", "1", "1"], "per form, in the order of its first call"
    assert "2 x cudnn.AddNormQuantDecode #5 [PDL True, PDL_RELEASE True, ADD True, STORE_RESIDUAL False, BF16_OUT False, QUANT nvfp4]" in served
    written = json.loads(target.read_text())
    (run,) = written["runs"]
    assert written["stack"] == "native" and run["target"]["not_claimed"][0] == "fallback rows present" and run["target"]["numbers"]["target_port"] == 10
    fused = [site for site in run["sites"] if site["block"] == "mlp.gate_up"]
    assert [(site["layer"], site["instance"], site["identity"]) for site in fused] == [
        (0, "cudnn.SwigluFp4Decode #1", "target"),
        (1, "cudnn.SwigluFp4Decode #1", "target"),
    ]
    assert all(site["served"] == site["form"] and site["form"]["TOKEN_TILE"] == 8 and site["form"]["MAX_CLUSTERS"] == 148 for site in fused)
    assert all(site["stated"]["launch"] == [[1, 1, 148], [192, 1, 1], 231424] for site in fused), "one entry instance serves every layer"
    multiplies = [site for site in run["sites"] if site["block"] == "projection.fp8.in"]
    assert [(site["site"], site["instance"], site["identity"]) for site in multiplies] == [
        ("linear_attention.qkvz", "cudnn.Fp8ProjectionDecode #1", "target"),
        ("full_attention.qkv", "cudnn.Fp8ProjectionDecode #2", "target"),
    ]
    assert all(site["served"] == site["form"] and site["form"]["STAGES"] == 8 and site["form"]["HOLD_TMEM"] is True for site in multiplies)
    assert [site["stated"]["launch"] for site in multiplies] == [[[128, 1, 1], [256, 1, 1], 139520], [[112, 1, 1], [256, 1, 1], 139520]]
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
        "output_head.quantize); 11 launches of a step compared by launch shape (1 the target's own compiled kernel + 10 cuDNN Frontend entries with "
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
    assert "projection.fp8.in equal (gated)" in invariance, "and so does the row of the FP8 projection entry"
    assert "mlp.gate_up equal (gated)" in invariance, "and the row of the fused gate / up entry"

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
    out = capsys.readouterr().err
    assert (
        "kernels, planned (decided before the load from model_samples/qwen_dense/configs.py; stack native" in out
        and "kernel identity, as planned: 12 of 20" in out
    )


def test_a_dry_plan_of_the_native_stack_imports_nothing_new():
    """In a fresh interpreter: planning the ``native`` stack on stated device facts, reading the census and stating the claim
    add only ``model_samples`` and standard-library modules -- the versions the census compares are read from the
    distributions' metadata, no package is imported for them."""
    code = f"""
import sys
import cudnn, torch
cudnn.GdnDecodeFused, cudnn.AttentionGateQuantDecode, cudnn.AddNormQuantDecode, cudnn.SwigluFp4Decode
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
print("TABLE", "gate_quant" in table, "add_norm_quant" in table, "fp8_projection" in table and "swiglu_fp4" in table and "12 of 20" in table, claim.reasons[0])
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
    split = re.compile(r"\d+ (?:of )?the target's own compiled kernels? \+ \d+ (?:of a )?cuDNN Frontend entr(?:y|ies)|=\s*49 \+ 273\b")
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
    assert "321 of the 484 launches of a step of the 64-layer model, 48 the target's own compiled kernels + 273 cuDNN Frontend entries" in area
    assert "322 of the 484" not in area, "the five entries are 321 launches; the 322nd is PyTorch's gather"
    assert "(322 = 49 + 273 with PyTorch's gather" in area and "on compute capability 10.0, 321 of the 484" in area, "said for the class it holds on"
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
    assert target.claim(plan, model, installed=RECORDED).known and len(target.claim(plan, model, installed=RECORDED).recorded) == 12

    # stated facts: no entry is built.  The same plan on a device with memory, where the entries are asked for their launch, is
    # held in test_a_row_serves_the_layer_widths_it_names: the norm block of such a model is the reference chain's
    stated = DeviceFacts((10, 0), "stated device", 148, 0, 0)
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
    assert Claim.recorded == {} and census.Traced.of(sites, _log(sites), step, 1, "", rows=1, claim=unheld).judge() == (census.PASS, [])
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
    assert any(f"{CENSUS}: NOT CLAIMED (" in line and COMPARED_12 in line for line in held.lines), "and the census line with what was compared"
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
    traced = census.Traced.of(sites, _log(sites), step, 1, "", rows=1, claim=complete)
    status, lines = traced.judge()
    assert status == census.FAIL and len(lines) == 1 and f"the {CENSUS} has grid (96, 1, 1)" in lines[0], lines
    assert traced.details()[-1] == f"{CENSUS}: NOT EQUAL (the kernel check did not hold)"


def test_the_first_use_of_an_entry_is_part_of_the_plan(monkeypatch):
    """An entry gates the installed toolchain where it builds its kernel.  On a device where a kernel can be built the
    adapter makes that first use while the run is planned, on scratch operands of the run's row count: a decline is then a
    ``NotImplementedError`` -- a SKIP before the load, in the entry's own words -- where the installation does not meet a
    requirement the adapter can name (the part's target by cuDNN Frontend's public probe; on compute capability 10.7 the
    CUDA 13 flavour of the CuTe DSL for the gate entry and for the norm entry), and a ``ValueError`` -- an ERROR -- for
    any other decline: those are the adapter's own operands, and a faulty adapter must not read as an unsupported GPU.
    The norm entry is built once per form the run calls; its build takes no operand, and neither does the projection
    entry's.  The fused gate / up entry is built from the row count alone, which is held in front of the build: every
    typed decline of that build is the entry's own sentence and a SKIP, whatever this adapter can name -- and at its
    first launch, where the operands are the adapter's, the sentence says whose the decline is."""
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

    projection_decline, swiglu_decline = [""], [""]
    monkeypatch.setattr(cudnn, "AttentionGateQuantDecode", Gate, raising=False)
    monkeypatch.setattr(cudnn, "GdnDecodeFused", Step, raising=False)
    monkeypatch.setattr(cudnn, "AddNormQuantDecode", Norm, raising=False)
    monkeypatch.setattr(cudnn, "Fp8ProjectionDecode", _projection_entry(calls, projection_decline), raising=False)
    monkeypatch.setattr(cudnn, "SwigluFp4Decode", _swiglu_entry(calls, swiglu_decline), raising=False)
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
    projection_builds = [("projection compile", 16384, 4), ("projection compile", 14336, 4)]  # one per weight shape, at the run's row count
    assert calls == [
        *norm_builds,  # one build per form the run calls, none for a form of the row no call of this model is made in
        *projection_builds,
        ("step compile", 4, (4, 10240)),
        ("swiglu compile", 4),  # the fused gate / up entry: its build takes the row count and no operand
        ("gate compile", (4, 6144), "cpu", (4, 6144)),
    ], "the first use of each entry, at the run's row count"
    table.first_use()
    assert len(calls) == 9 and table.first_used, "made once per plan"
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
    del calls[:]
    table["projection.fp8.in"].compile(4, 1)
    assert calls == [
        event
        for out_features in (16384, 14336)
        for event in (("projection compile", out_features, 4), ("projection launch", (out_features, 5120), (4, 5120), (4, out_features)))
    ], "and for the FP8 projection entry at both weight shapes: scratch operands in the entry's own layout"
    del calls[:]
    table["mlp.gate_up"].compile(4, 1)
    assert [call[0] for call in calls] == ["swiglu compile", "swiglu launch"] and calls[0][1] == 4
    assert [tuple(operand.shape) for operand in calls[1][1:]] == [
        (34816, 2560),  # the merged gate / up weight, two codes per byte
        (34816, 320),  # one block scale per 16 elements of a row
        (4, 2560),
        (128, 320),  # one band of the 128 x 4 layout for the rows of the run
        (1,),
        (1,),
        (4, 8704),
        (128, 1088),
    ], "and for the fused gate / up entry: scratch operands in the entry's own layout, the outputs the call's own"
    # (0a) the build of the fused gate / up entry is handed the row count and nothing else, and the entry has stated a launch
    # for that row count before it is built: nothing of the adapter's is left for the build to decline.  So every typed decline
    # of it is the entry's answer about the installation and a SKIP in the entry's own words -- the compiler it is qualified
    # with, on every class it serves, and the names its kernel is written on, which no public probe of the package answers:
    # with every requirement met that this adapter can ask about, the entry's sentence is still the SKIP, and never an ERROR
    # about operands this adapter did not build
    qualified = "the SwiGLU FP4 decode kernel is qualified with the CuTe DSL compiling with CUDA 13.4 or later; the DSL in this process compiles with CUDA 12.9"
    written_on = (
        "the SwiGLU FP4 decode kernel is written on cutlass.utils.blackwell_helpers; the installed CuTe DSL (nvidia-cutlass-dsl 4.9.0) does not export "
        "make_smem_layout_epi"
    )
    for decline, compiler, cc in (
        (qualified, (12, 9), (10, 0)),
        (qualified, (13, 0), (10, 3)),
        (qualified, (13, 4), (10, 0)),
        (written_on, (13, 4), (10, 0)),
        (written_on, (13, 4), (10, 7)),
    ):
        swiglu_decline[0] = decline
        with monkeypatch.context() as installed:  # an installation with that compiler and the part's target
            installed.setattr(cudnn.frost, "cutedsl_runtime_available", lambda cc=None: True)
            installed.setattr(dsl_version, "CUDA_VERSION", SimpleNamespace(major=compiler[0], minor=compiler[1]))
            with pytest.raises(NotImplementedError) as declined:
                planned(cc=cc)
        assert str(declined.value) == decline, (compiler, cc)
    # ... and a CuTe DSL that does not say which compiler it bound declines nothing by that: the entry's sentence, not an
    # import error of this adapter's own question
    with monkeypatch.context() as installed:
        installed.setattr(cudnn.frost, "cutedsl_runtime_available", lambda cc=None: True)
        installed.delattr(dsl_version, "CUDA_VERSION")
        with pytest.raises(NotImplementedError) as declined:
            planned()
    assert str(declined.value) == written_on
    swiglu_decline[0] = ""
    # ... the row count is the one thing this adapter hands the build, and it is held in front of it: an entry that states no
    # launch for it is a wrong table (an ERROR), whatever its build would have said
    block = resolved()["mlp.gate_up"]
    stated = cudnn.SwigluFp4Decode.launch_shape_for

    def past_the_ceiling(self, rows):
        if rows > 64:
            raise NotImplementedError("the SwiGLU FP4 decode kernel serves 1 .. 64 rows per call; got 65")
        return stated(self, rows)

    monkeypatch.setattr(cudnn.SwigluFp4Decode, "launch_shape_for", past_the_ceiling)
    swiglu_decline[0] = written_on
    del calls[:]
    with pytest.raises(ValueError) as wrong:
        block.first_use(65, 1)
    assert str(wrong.value) == (
        "the provider table is wrong: the row of fe.swiglu_fp4_decode claims this run (65 rows on compute capability 10.0), and cudnn.SwigluFp4Decode "
        "declines it: the SwiGLU FP4 decode kernel serves 1 .. 64 rows per call; got 65"
    )
    assert calls == [], "before anything is built"
    monkeypatch.setattr(cudnn.SwigluFp4Decode, "launch_shape_for", stated)
    swiglu_decline[0] = ""
    # (0) the FP8 projection entry asks for a CuTe DSL with the layer its kernel is written on: without it the entry's own sentence
    # is the SKIP; with it the same decline is the adapter's -- an ERROR
    import importlib.util

    layer = "the FP8 projection decode kernel is written on cutlass.cute.experimental, which does not import on the installed CuTe DSL"
    found = importlib.util.find_spec
    projection_decline[0] = layer
    with pytest.raises(ValueError) as wrong:
        planned()
    assert str(wrong.value) == f"cudnn.Fp8ProjectionDecode declined the operands this adapter built for its first use: {layer}"
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *rest: None if name == "cutlass.cute.experimental" else found(name, *rest))
    with pytest.raises(NotImplementedError) as skipped:
        planned()
    assert str(skipped.value) == layer, "the entry's own words"
    monkeypatch.setattr(importlib.util, "find_spec", found)
    planned(cc=(10, 3))  # on another class the row is not the entry's: nothing of it is built, and nothing declines
    projection_decline[0] = ""

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
    # ... the fused gate / up entry's first launch does take operands of this adapter's, so there the sentence says whose the
    # decline is: the installation's with an older compiler than the entry is qualified with, the adapter's operands with that
    # compiler -- and with a CuTe DSL that does not say which compiler it bound, which is no unmet requirement (the entry
    # declines nothing on a value it could not read, and neither does the adapter's question raise)
    monkeypatch.setattr(Norm, "decline", "")
    swiglu_decline[0] = qualified
    for compiler, whose in (
        ((12, 9), "declines this installation at"),
        ((13, 4), "declined the operands this adapter built for"),
        (None, "declined the operands this adapter built for"),
    ):
        with monkeypatch.context() as installed:
            if compiler is None:
                installed.delattr(dsl_version, "CUDA_VERSION")
            else:
                installed.setattr(dsl_version, "CUDA_VERSION", SimpleNamespace(major=compiler[0], minor=compiler[1]))
            with pytest.raises(RuntimeError) as declined:
                table["mlp.gate_up"].compile(4, 1)
        assert str(declined.value) == f"cudnn.SwigluFp4Decode {whose} its first launch: {qualified}", compiler
    swiglu_decline[0] = ""
    for block in ("attention.gate_quantize", "add_norm_quantize", "mlp.gate_up"):
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

    class Projection(Entry):
        name = "cudnn.Fp8ProjectionDecode"

    class Swiglu(Entry):
        name = "cudnn.SwigluFp4Decode"

    def scratch(block, sequences, tokens):  # in place of the adapters' scratch operands: what would be allocated, and when
        events.append(("scratch", block.entry, sequences * tokens))
        return block._build(block.options["device"]), ()

    sentence = blocks.Plan.memory

    def memory(plan, need_bytes, what="", checks_bytes=0):
        events.append(("memory sentence", plan.stack))
        return sentence(plan, need_bytes, what, checks_bytes)

    def no_weights(model, *arguments, **options):
        raise AssertionError("reached a load")

    gpu = {"total": 180 * GIB, "free": 1 * GIB}  # a real device, as it reports itself
    monkeypatch.setattr(cudnn, "GdnDecodeFused", Step, raising=False)
    monkeypatch.setattr(cudnn, "AttentionGateQuantDecode", Gate, raising=False)
    monkeypatch.setattr(cudnn, "AddNormQuantDecode", Norm, raising=False)
    monkeypatch.setattr(cudnn, "Fp8ProjectionDecode", Projection, raising=False)
    monkeypatch.setattr(cudnn, "SwigluFp4Decode", Swiglu, raising=False)
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
        """The first use of the norm entry and of the FP8 projection entry in a stack that has their rows: one build per form
        the two-layer model calls (four) and one per weight shape (two), in front of the other entries' -- and no scratch
        operand: their builds take none."""
        return [] if stack == "fe-cublas" else [("build", "cudnn.AddNormQuantDecode")] * 4 + [("build", "cudnn.Fp8ProjectionDecode")] * 2

    def first_uses(stack, *counts):
        """In the order of the blocks: the norm and the projection entry, the fused step on scratch operands, then -- in a stack
        that has their rows -- the fused gate / up entry (its build takes the row count and no operand) and the gate entry."""
        events = []
        for rows in counts:
            events += [*norm_builds(stack), ("scratch", "cudnn.GdnDecodeFused", rows), ("build", "cudnn.GdnDecodeFused")]
            if stack != "fe-cublas":
                events += [("build", "cudnn.SwigluFp4Decode"), ("scratch", "cudnn.AttentionGateQuantDecode", rows), ("build", "cudnn.AttentionGateQuantDecode")]
        return events

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
        # (2) the default run where too little is free: the same, for both sequence counts of the run.  A stack whose in-projection
        # and whose gate / up multiply are on the stored weights at both counts keeps those and no bf16 copy of them (0.15 GiB
        # of 8-bit codes and 0.19 GiB of 4-bit codes where the copies are 0.29 and 0.66 GiB, for the two-layer model); the
        # reference of its checks then makes its copy of one such weight at a time (0.33 GiB the largest: a gate / up weight),
        # and the sentence says that part
        what, gib = " for 1 and 4 sequences, one count after the other", "3.7"
        if stack != "fe-cublas":
            what += ", 0.33 GiB of it for the checks (the reference's bf16 copy of one weight at a time, where the run keeps none)"
            gib = "3.5"
        few = "^" + re.escape(f"stack {stack} needs about {gib} GiB on this GPU{what}; 1.0 GiB are free: free memory on this GPU")
        with pytest.raises(NotImplementedError, match=few):
            run(stack, 1, 4)
        assert events == [("memory sentence", stack)], stack
        del events[:]
        with monkeypatch.context() as scoped:  # the sample's own run on such a device; what it patches ends with it
            code, out = _sample_main(scoped, capsys, ["--stack", stack], total=180 * GIB, free=1 * GIB)
        assert code == 0 and out.splitlines()[-1] == (
            f"[00] SKIP stack {stack} needs about {gib} GiB on this GPU{what}; 1.0 GiB are free: "
            "free memory on this GPU, or select another one with CUDA_VISIBLE_DEVICES"
        ), out
        assert events == [("memory sentence", stack)], f"{stack}: the sample's own run builds nothing either"
        # (3) with room: the memory sentence, then the first use of every entry for every sequence count, then nothing more
        gpu.update(free=80 * GIB)
        first, table = run(stack, 1, 4)
        assert f"this stack needs about {gib} GiB on this GPU{what}; 80.0 GiB are free" in table
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
    each site: six forms, of which the 64-layer model calls five, 1 / 47 / 16 / 64 / 1 times a step.  The in-projection
    is called with another weight shape in each kind of layer: two sites, 48 / 16 times a step, the shape of each in its
    row.  The out-projection is called at a site per kind of layer too, 48 / 16 times a step: one weight shape, and in a
    step of several tokens per sequence two rows.  A site name belongs to one block."""
    from collections import Counter
    from types import SimpleNamespace

    from model_samples import census
    from model_samples.qwen_dense import QwenDense, configs

    full = QwenDense()
    wiring, sites = full.wiring(), full.sites()
    assert len(sites) == len(wiring) == 484 and census.call_sites(full) == sites
    every = [(block, layer, site) for (block, layer), site in zip(wiring, sites) if site]
    assert {block for block, _, _ in every} == {
        "add_norm_quantize",
        "projection.fp8.in",
        "projection.fp8.out",
    }, "the three blocks that are called at named sites"
    named = [call for call in every if call[0] == "add_norm_quantize"]
    multiplies = [call for call in every if call[0] == "projection.fp8.in"]
    outs = [call for call in every if call[0] == "projection.fp8.out"]
    assert len(named) == 129 and len(multiplies) == 64 and len(outs) == 64
    names = [{site for _, _, site in calls} for calls in (named, multiplies, outs)]
    assert not (names[0] & names[1] or names[0] & names[2] or names[1] & names[2]), "a site name belongs to one block: an adapter finds its sites by name"
    assert Counter(site for _, _, site in outs) == {"linear_attention.out": 48, "full_attention.out": 16}
    assert outs[:4] == [("projection.fp8.out", layer, "linear_attention.out") for layer in (0, 1, 2)] + [("projection.fp8.out", 3, "full_attention.out")]
    assert Counter(site for _, _, site in named) == {"linear_attention.first": 1, "linear_attention": 47, "full_attention": 16, "mlp": 64, "output_head": 1}
    assert Counter(site for _, _, site in multiplies) == {"linear_attention.qkvz": 48, "full_attention.qkv": 16}
    assert multiplies[:4] == [("projection.fp8.in", layer, "linear_attention.qkvz") for layer in (0, 1, 2)] + [("projection.fp8.in", 3, "full_attention.qkv")]
    assert named[:3] == [("add_norm_quantize", 0, "linear_attention.first"), ("add_norm_quantize", 0, "mlp"), ("add_norm_quantize", 1, "linear_attention")]
    assert named[6:8] == [("add_norm_quantize", 3, "full_attention"), ("add_norm_quantize", 3, "mlp")] and named[-1] == ("add_norm_quantize", 64, "output_head")

    def norm_sites(model):
        return [site for (block, _), site in zip(model.wiring(), model.sites()) if block == "add_norm_quantize"]

    assert norm_sites(QwenDense.reduced(1)) == ["full_attention.first", "mlp", "output_head"], "a model that starts with an attention layer"
    assert [site for site in QwenDense.reduced(1).sites() if site and site not in norm_sites(QwenDense.reduced(1))] == [
        "full_attention.qkv",
        "full_attention.out",
    ]
    from model_samples.qwen_dense import model as family

    assert {site.removesuffix(family.FIRST) for _, _, site in named} == set(family.NORM_SITES) and family.FIRST == ".first"
    assert {site for _, _, site in multiplies} == set(family.IN_PROJECTION_SITES.values()) and set(family.IN_PROJECTION_SITES) == set(full.geometry.layer_kinds)
    assert {site for _, _, site in outs} == set(family.OUT_PROJECTION_SITES.values()) and set(family.OUT_PROJECTION_SITES) == set(full.geometry.layer_kinds)
    assert family.WEIGHT_SITES == {"projection.fp8.in": family.IN_PROJECTION_SITES, "projection.fp8.out": family.OUT_PROJECTION_SITES}
    assert norm_sites(QwenDense.reduced(2)) == ["linear_attention.first", "mlp", "full_attention", "mlp", "output_head"]
    bare = SimpleNamespace(wiring=lambda: (("embedding", -1), ("token_choice", -1)))
    assert census.call_sites(bare) == ("", ""), "a model that names no call sites: every block in one form"
    with pytest.raises(ValueError, match="^the model names 1 call sites for the 2 calls of its wiring$"):
        census.call_sites(SimpleNamespace(wiring=bare.wiring, sites=lambda: ("",)))

    # what the step hands the block is the site of that call: seen on a step of the reference stack
    session, _ = _tiny_session(1)
    seen = []

    def tap(name, layer, block, operands):
        seen.append((name, layer, operands[4] if name == "add_norm_quantize" else (operands[2] if name.startswith("projection.fp8.") else "")))
        return block(*operands)

    session.tap = tap
    session._advance(torch.zeros(1, dtype=torch.int32))
    assert seen == [(block, layer, site) for (block, layer), site in zip(session.model.wiring(), session.model.sites())]
    assert [site for block, _, site in seen if block == "add_norm_quantize"] == ["linear_attention.first", "mlp", "full_attention", "mlp", "output_head"]
    assert [site for block, _, site in seen if block == "projection.fp8.in"] == ["linear_attention.qkvz", "full_attention.qkv"]
    assert [site for block, _, site in seen if block == "projection.fp8.out"] == ["linear_attention.out", "full_attention.out"]

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
        "| 8-bit in-projection (linear attention QKVZ / attention QKV) | `linear_attention.qkvz` | 48 | [OUT_FEATURES 16384, IN_FEATURES 5120] |",
        "| 8-bit in-projection (linear attention QKVZ / attention QKV) | `full_attention.qkv` | 16 | [OUT_FEATURES 14336, IN_FEATURES 5120] |",
    ]

    # the row of the FP8 projection entry: the whole kernel in its form, and per call site the weight's shape
    row = configs.PROVIDERS["projection.fp8.in"][0]
    assert (
        row.key == "fe.fp8_projection_decode"
        and row.sites == configs.PROJECTION_SITES
        and [site for site, _ in row.sites] == list(family.IN_PROJECTION_SITES.values())
    )
    assert row.form == (
        ("SPLIT_K", 1),
        ("MMA_M", 128),
        ("STAGES", 8),
        ("TMEM_COLS", 32),
        ("PREWAIT_WEIGHTS", True),
        ("HOLD_TMEM", True),
        ("PDL", True),
        ("PDL_RELEASE", True),
    ), "the kernel is stated whole: nothing of it is the entry's pick"
    assert row.classes == ("cc10.0",) and row.rows == (1, 32) and row.sequences == (1, 32) and row.band == "bf16_rounding"
    assert set(row.stacks) == {"native", "fe-entries"} and row.batch_invariant == ("cc10.0",) and not row.fallback and row.ci
    assert row.site_form("full_attention.qkv") == (*row.form, ("OUT_FEATURES", 14336), ("IN_FEATURES", 5120))
    assert (
        configs.PROVIDERS["projection.fp8.in"][1].key == "cublas.bf16_dequantized"
    ), "the multiply over a copy stays the row of every other class and row count"
    out_rows = configs.PROVIDERS["projection.fp8.out"]
    at_one_token = next(r for r in out_rows if "native" in r.stacks and r.tokens[0] <= 1)
    assert (
        at_one_token.key == "cublas.bf16_dequantized" and at_one_token.sites == ()
    ), "a step of one token per sequence: the out-projection is a multiply over a copy at both call sites"
    assert all(r.tokens == (2, None) for r in out_rows if r.key == "fe.fp8_projection_decode"), "the entry's rows of this block are a verify step's"


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
    assert [site.site for site in sites if site.block not in ("add_norm_quantize", "projection.fp8.in", "projection.fp8.out")] == [""] * 227
    outs = [site for site in sites if site.block == "projection.fp8.out"]
    assert Counter(site.site for site in outs) == {"linear_attention.out": 48, "full_attention.out": 16}
    assert all(
        site.form == () and site.facts is None and site.wrong == "" for site in outs
    ), "a multiply over a copy at both sites of a step of one token per sequence"
    assert sites[2].row == "projection.fp8.in at linear_attention.qkvz" and sites[3].row == "linear_attention.step"
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
    # the adapter handed a model of a width its entry declines: a table whose row does not name the width it serves.  (The
    # family's row names it, so a plan never hands this adapter such a model: test_a_row_serves_the_layer_widths_it_names)
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
    assert numbers["not_the_target"] == {"add_norm_quantize": 64} and (numbers["target_code"], numbers["target_port"]) == (49, 209)
    assert claim.reasons == [f"not the target set: add_norm_quantize ({reason})", "fallback rows present", "the plan has no block for output_head.quantize"][
        1:2
    ] + [
        f"not the target set: add_norm_quantize ({reason})",
        "the plan has no block for output_head.quantize",
    ], "said once for its 64 calls"
    assert (
        claim.of("add_norm_quantize") == f"target; not the target ({reason})" and len(claim.recorded) == 322 - 64
    ), "no recorded shape is held against those calls"

    # a kernel whose form follows the row count: the census names the field on the entries of its row count.  A census that names
    # another token tile there -- the same kernel, the same launch shape -- makes the row "not the target" by its form
    def with_another_tile(rows):
        for kind in ("linear_attention_layer", "attention_layer"):
            rows[kind][-2][2]["form"]["TOKEN_TILE"] = 16

    claim = changed(with_another_tile).claim(plan, model, installed=RECORDED)
    everywhere = "PREWAIT_STAGES 2, L2_EVICT_FIRST True, MAX_CLUSTERS 148, PDL True"
    reason = f"the row's form is [TOKEN_TILE 8, {everywhere}], the target's [{everywhere}, TOKEN_TILE 16]"
    assert claim.numbers()["not_the_target"] == {"mlp.gate_up": 64} and claim.numbers()["target_port"] == 209 and len(claim.recorded) == 322 - 64
    assert claim.of("mlp.gate_up") == f"not the target ({reason})" and claim.reasons[1] == f"not the target set: mlp.gate_up ({reason})"
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
    # a code of the OTHER bf16 value: the kernel's own arithmetic inside the margin, a failure beyond it
    apart4 = other.clone()
    apart4[1, 1], apart4[1, 3] = 5.5, 5.5  # stated so that the two values of these elements have different codes (6 and 4; 6 and 6)
    took = nibbles(codes4)
    took[1, 1] = 7  # the code of 5.5 under a block scale of 1, where the nearest value is 4: an element far from its midpoint
    stats = judged((pack(took), scales4, residual), stated(("codes", "scales", "residual"), four, (codes4, scales4, residual), (nearest, apart4, margin)))
    assert not stats.ok and (stats.off_nearest, stats.outside) == (1, 0), "4-bit codes: the other value's code beyond the margin"
    near_tie = margin.clone()
    near_tie[1, 1] = 4.0
    assert judged((pack(took), scales4, residual), stated(("codes", "scales", "residual"), four, (codes4, scales4, residual), (nearest, apart4, near_tie))).ok
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
    # the order of the kernel's two multiplications is part of the byte: at a global scale of 3 these block maxima sit on a rounding
    # tie of the block scale, which a division by 6, or the product regrouped, breaks the other way
    ties, third = torch.tensor([0.048828125, 0.09765625, 0.1953125]), 1.0 / 3.0
    sixth = torch.tensor(torch_ref.FP4_SIXTH_BITS, dtype=torch.int32).view(torch.float32)
    three = torch_ref.nvfp4_global_scale(third, "cpu")
    assert float(three) == 3.0 and torch_ref.nvfp4_block_scales(ties, third).tolist() == [13, 21, 29], "e4m3(global x (largest x fp32(1/6)))"
    from model_samples.weights import _e4m3

    assert _e4m3(three * (ties / 6.0)).tolist() == _e4m3((three * ties) * sixth).tolist() == [12, 20, 28], "neither of these is the kernel's byte"
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
    fail the kernel check with its own sentence -- a call served in the form of another site (the launch records of that
    run are clean), a chain of other kernels under the row's name, a launch more in front of its kernel, a launch less, a
    launch in another shape than the census recorded, a launch under the symbol of another form, a traced step without its
    call log -- or fail the block in its band while the kernel check holds: a wrong code, a wrong block scale, a wrong
    normed element, a wrong element of the residual stream.  And the one thing the check does not see, pinned as that: a
    launch more BEHIND the kernel, in front of a fallback row, is counted as that row's."""
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
    assert "12 on exact rows, in the wiring order" in _line(checks, "kernels ") and norm_line(checks).endswith("[band: add_norm_quant]  PASS")
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
    assert checks.ok and any(f"{CENSUS}: NOT CLAIMED (" in line and COMPARED_12 in line for line in checks.lines), "held to the recorded shape, and said"

    def tampered(change):
        """The clean run at one sequence with the records of its traced pass changed before the checks read them."""
        session = _recorded_run(monkeypatch, sequences=1, stack="native")[0]
        session.trace(4)
        change(session.traced)
        return session, tie_out.run_checks(session, prompt_tokens=3)

    def norms(step):
        return [index for index, record in enumerate(step) if census.canonical(record) == "add_norm_quant"]

    # (5a) a form also shows in a launch record, by the kernel's symbol: the first two norm launches of a step under each other's
    # symbol, with a call log that says every call was served in the form of its site
    def swap_symbols(traced):
        for step in traced.steps:
            first, second = norms(step)[:2]
            step[first], step[second] = step[first]._replace(name=step[second].name), step[second]._replace(name=step[first].name)

    _, checks = tampered(swap_symbols)
    stored, added = (_norm_symbol(dict(row.site_form(site))) for site in ("linear_attention.first", "mlp"))
    assert (
        not checks.ok
        and checks.kernels == census.FAIL
        and sentences(checks)
        == [
            f"kernels: step 0 (and 3 more of the 4 traced steps): add_norm_quantize (layer 0) launched {launched!r}; cudnn.AddNormQuantDecode names {named!r} "
            "for this run"
            for launched, named in ((added, stored), (stored, added))
        ]
    ), checks.text()

    # (5b) what the check sees behind a norm's kernel, and what it does not.  A launch more in FRONT of the first norm's kernel
    # fails (case 3: the gather before it is judged one by one).  The same launch directly BEHIND the kernel of a layer's input
    # norm fails too on this class: its consumer, the in-projection, is an exactly judged row.  So it does behind the norm in
    # front of an MLP: its consumer is the fused gate / up kernel.  Behind the FINAL norm it sits in front of a fallback row
    # (the output head) and is attributed to it: the check holds.  That the adapter launches nothing but its entry is held
    # on the adapter itself (test_the_norm_entrys_adapter_over_a_stand_in_entry: the tensors it returns are the ones the
    # entry wrote)
    def one_more_behind(which):
        def change(traced):
            for step in traced.steps:
                at = norms(step)[which]
                step.insert(at + 1, census.Launch(ELEMENTWISE, "kernel", step[at].end + 0.01, step[at].end + 0.02, (1, 1, 1), (128, 1, 1), 0, 32))

        return change

    _, checks = tampered(one_more_behind(0))
    assert (
        not checks.ok
        and checks.kernels == census.FAIL
        and sentences(checks)
        == [
            f"kernels: step 0 (and 3 more of the 4 traced steps): launched {ELEMENTWISE[:93] + '...'!r} between add_norm_quantize (layer 0) and "
            "projection.fp8.in (layer 0): a kernel of the fallback family pytorch, which no row there declares; nothing may sit between two exact rows"
        ]
    ), checks.text()
    _, checks = tampered(one_more_behind(1))
    assert (
        not checks.ok
        and checks.kernels == census.FAIL
        and sentences(checks)
        == [
            f"kernels: step 0 (and 3 more of the 4 traced steps): launched {ELEMENTWISE[:93] + '...'!r} between add_norm_quantize (layer 0) and "
            "mlp.gate_up (layer 0): a kernel of the fallback family pytorch, which no row there declares; nothing may sit between two exact rows"
        ]
    ), checks.text()
    clean, _ = tampered(lambda traced: None)
    session, checks = tampered(one_more_behind(4))
    assert checks.ok and checks.kernels == census.PASS, checks.text()
    assert session.traced.result.launches == [count + 1 for count in clean.traced.result.launches]
    assert session.traced.result.families["pytorch"] == [count + 1 for count in clean.traced.result.families["pytorch"]], "counted as the fallback row's"

    # (5c) the call log is what item 0 reads: a traced pass that kept none, or the log of one step of four, fails -- the launch
    # records are those of the clean run
    def no_log(traced):
        del traced.call_log[:]

    def one_log(traced):
        del traced.call_log[1:]

    for change, has in ((no_log, "no step"), (one_log, "1 step")):
        session, checks = tampered(change)
        assert (
            not checks.ok
            and checks.kernels == census.FAIL
            and sentences(checks)
            == [
                f"kernels: calls: the call log holds the block calls of {has} for 4 traced steps: without one log per traced step nothing shows which "
                "entry instance served a call"
            ]
        ), checks.text()
        assert not any(COMPARED_12 in line for line in checks.lines), "nothing is said to have been compared"

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


# ----------------------------------------------------------------------------------------------- the row of the FP8 projection entry
def test_the_projection_entrys_adapter_over_a_stand_in_entry(monkeypatch):
    """The adapter of ``cudnn.Fp8ProjectionDecode`` hands the entry what its row says and nothing else: per call site one
    instance, built with the split, the performance form as a whole, the launch knobs and the weight's shape of that site;
    per call the stored 8-bit codes of both operands as they are, the fp32 product of the two scales as one element built
    once per pair of values, and an output buffer of the call's own -- and it returns the tensor its entry wrote.  What it
    states about a launch is the entry's own answer with the fields of the weight's shape beside the others.  A table or
    an operand it cannot serve is named.  And the row's form is the PUBLIC entry's own at every row count it claims."""
    import cudnn
    from cudnn.gemm import ProjectionForm, ProjectionGeometry

    from model_samples.blocks import DeviceFacts
    from model_samples.providers import BlockBase, Quant, fe
    from model_samples.qwen_dense import QwenDense, configs
    from model_samples.weights import Dequantized, ScaledTensor, Unquantized
    from dataclasses import replace

    calls = []
    monkeypatch.setattr(cudnn, "Fp8ProjectionDecode", _projection_entry(calls), raising=False)
    monkeypatch.setattr(fe, "_ENTRIES", {})
    row = configs.PROVIDERS["projection.fp8.in"][0]
    model = QwenDense.reduced(2)
    facts = DeviceFacts((10, 0), "stated device", 148, 0, 0)
    block = fe.Fp8ProjectionDecodeBlock(row, model, facts, {"device": "cpu", "check_kernels": True, "merge_rule": "largest_scale"})
    assert block.stored_weights is True and BlockBase.stored_weights is False, "the loader keeps the stored weight for this block only"
    assert block.entry == "cudnn.Fp8ProjectionDecode" and [(k.id, k.relation, k.launches, k.shape) for k in block.kernels] == [
        ("fp8_projection", "exact", 1, "entry")
    ]
    block.check_support(1, 1)
    assert block._sites() == ("linear_attention.qkvz", "full_attention.qkv"), "the weight shapes the two-layer model calls"
    assert fe.Fp8ProjectionDecodeBlock(row, QwenDense.reduced(1), facts, {"device": "cpu"})._sites() == ("full_attention.qkv",)
    wide, narrow = block._build("cpu", "linear_attention.qkvz"), block._build("cpu", "full_attention.qkv")
    assert wide is not narrow and wide is block._build("cpu", "linear_attention.qkvz"), "one instance per weight shape, built once"
    assert wide.built == dict(
        device="cpu",
        geometry=ProjectionGeometry(16384, 5120),
        split_k=1,
        form=ProjectionForm(MMA_M=128, STAGES=8, TMEM_COLS=32, PREWAIT_WEIGHTS=True, HOLD_TMEM=True),
        pdl=True,
        pdl_release=True,
    ), "the kernel as the row states it: the form is passed as a whole, so the entry never takes its own pick"
    assert narrow.built["geometry"] == ProjectionGeometry(14336, 5120) and narrow.built["form"] == wide.built["form"]
    stated = block.launch_facts(1, "full_attention.qkv")
    assert stated.shape == PROJECTION_SHAPE[14336] and stated.symbols == (_projection_symbol({"OUT_FEATURES": 14336, "IN_FEATURES": 5120}),)
    assert all(stated.form[name] == value for name, value in row.site_form("full_attention.qkv")), "the shape beside the other fields"
    assert stated.form["GEOM"] == {"OUT_FEATURES": 14336, "IN_FEATURES": 5120}, "as the entry states it, one level down"
    assert block.serving() == ("cudnn.Fp8ProjectionDecode", ()), "before the first call nothing has served"

    # a call: the stored codes of both operands as they are, one scale element, an output of the call's own
    codes = torch.arange(48 * 64, dtype=torch.uint8).reshape(48, 64) % 100
    weight = Dequantized(codes.to(torch.bfloat16), 0.013, stored=ScaledTensor(codes, 0.013))
    qx = ScaledTensor((torch.arange(2 * 64, dtype=torch.uint8).reshape(2, 64) % 90), 0.017857142857142856)
    del calls[:]
    out = block(qx, weight, "linear_attention.qkvz")
    again = block(qx, weight, "linear_attention.qkvz")
    assert calls == [("projection launch", (48, 64), (2, 64), (2, 48))] * 2, "one launch per call, no build inside a step"
    assert out.dtype == torch.bfloat16 and out.shape == (2, 48) and out.data_ptr() != again.data_ptr(), "a buffer of the call's own"
    seen = []
    monkeypatch.setattr(type(wide), "__call__", lambda self, w, a, scale, o: (seen.append((w, a, scale, o)), o)[1])
    out = block(qx, weight, "linear_attention.qkvz")
    w, a, scale, o = seen[-1]
    assert o is out, "what the adapter returns is the tensor its entry wrote: it launches nothing else"
    assert w.dtype == a.dtype == torch.float8_e4m3fn and w.data_ptr() == codes.data_ptr() and a.data_ptr() == qx.codes.data_ptr(), "views, no copy"
    product = torch.tensor([0.017857142857142856], dtype=torch.float32) * torch.tensor([0.013], dtype=torch.float32)
    assert scale.dtype == torch.float32 and scale.shape == (1,) and torch.equal(scale, product), "the fp32 product of the two scales"
    assert not torch.equal(
        product, torch.tensor([0.017857142857142856 * 0.013], dtype=torch.float32)
    ), "which these two values tell from a rounded fp64 product"
    block(qx, weight, "linear_attention.qkvz")
    assert seen[-1][2] is scale, "one element per pair of values, built once: never inside a later step"
    halved = Dequantized(codes.to(torch.bfloat16), 0.5, stored=ScaledTensor(codes, 0.5))
    block(qx, halved, "linear_attention.qkvz")
    assert seen[-1][2] is not scale and torch.equal(seen[-1][2], torch.tensor([0.017857142857142856], dtype=torch.float32) * 0.5), "another weight scale"
    assert seen[-1][0].data_ptr() == codes.data_ptr(), "and still that weight's own codes"
    assert block.serving() == ("cudnn.Fp8ProjectionDecode #1", row.site_form("linear_attention.qkvz")), "read from the instance that was called"
    block(qx, weight, "full_attention.qkv")
    assert block.serving() == ("cudnn.Fp8ProjectionDecode #2", row.site_form("full_attention.qkv"))

    # what it cannot serve is named
    with pytest.raises(ValueError, match=r"the weight of this call has no stored 8-bit codes: the weights of this session were loaded for another plan"):
        block(qx, Dequantized(codes.to(torch.bfloat16), 0.013), "linear_attention.qkvz")
    per_row = ScaledTensor(codes, torch.full((48,), 0.013))
    with pytest.raises(ValueError, match=r"the weight of this call has one scale per weight row"):
        block(qx, Dequantized(codes.to(torch.bfloat16), per_row.scale, stored=per_row), "linear_attention.qkvz")
    with pytest.raises(ValueError, match=r"^cudnn.Fp8ProjectionDecode multiplies static 8-bit codes; this block is handed Unquantized$"):
        block(Unquantized(torch.zeros((2, 64), dtype=torch.bfloat16), "fp8", 1.0), weight, "linear_attention.qkvz")
    with pytest.raises(ValueError, match=r"no form for the call site 'mlp'; the row names linear_attention.qkvz, full_attention.qkv$"):
        block(qx, weight, "mlp")
    merged = fe.Fp8ProjectionDecodeBlock(row, model, facts, {"device": "cpu", "merge_rule": "own_scale"})
    with pytest.raises(
        NotImplementedError, match=r"^cudnn.Fp8ProjectionDecode takes one scale per weight; the merge rule \'own_scale\' keeps one per weight row: "
    ):
        merged.check_support(1, 1)
    for wrong, said in (
        (replace(row, form=(*row.form, ("ROWS", 1))), "names ROWS"),
        (replace(row, form=tuple(pair for pair in row.form if pair[0] != "STAGES")), "does not name STAGES"),
        (replace(row, sites=tuple((site, fields[:1]) for site, fields in row.sites)), "does not name IN_FEATURES"),
    ):
        with pytest.raises(ValueError, match=rf"^provider row fe.fp8_projection_decode: its form at linear_attention.qkvz {said}; this adapter passes "):
            fe.Fp8ProjectionDecodeBlock(wrong, model, facts, {"device": "cpu"}).check_support(1, 1)
    assert Quant("fp8", 1.0).fmt == "fp8"

    # the PUBLIC entry, built for the host from the same arguments: it states every field of the row's form at every row count
    # the row claims, and the launch shape the census records at 1 row; a shape or a split it does not serve is a wrong table
    monkeypatch.undo()
    monkeypatch.setattr(fe, "_ENTRIES", {})
    real = fe.Fp8ProjectionDecodeBlock(row, model, facts, {"device": "cpu", "check_kernels": True})
    for site, fields in row.sites:
        entry = real._build("cpu", site)
        assert type(entry) is cudnn.Fp8ProjectionDecode
        for rows in range(row.rows[0], row.rows[1] + 1):
            form = real._stated(entry, rows)
            assert all(form[name] == value for name, value in row.site_form(site)), (site, rows)
            assert form["ROWS"] == rows and form["TOKEN_TILE"] == (8 if rows <= 8 else 16 if rows <= 16 else 32), "the token tile follows the rows"
        assert tuple(real.launch_facts(1, site).shape) == PROJECTION_SHAPE[dict(fields)["OUT_FEATURES"]], site
        assert real.launch_facts(1, site).symbols == (), "nothing is compiled on the host"
        with pytest.raises(NotImplementedError, match=r"serves at most 32 rows per call"):
            entry.form_for(33)
    other = replace(row, form=tuple(("SPLIT_K", 4) if name == "SPLIT_K" else (name, value) for name, value in row.form))
    with pytest.raises(ValueError) as declined:
        fe.Fp8ProjectionDecodeBlock(other, model, facts, {"device": "cpu"})._build("cpu", "linear_attention.qkvz")
    assert str(declined.value).startswith(
        "the provider table is wrong: cudnn.Fp8ProjectionDecode declines the weight shape or the form of the row of fe.fp8_projection_decode at "
        "linear_attention.qkvz: the FP8 projection decode kernel serves split_k 1 at out features 16384, in features 5120"
    )


def _weights_of(weights, block):
    """The weight handles of ``block`` in the weights of one load, in layer order."""
    if block == "output_head":
        return [weights.head]
    name = {"projection.fp8.in": "in_proj", "projection.fp8.out": "out_proj", "mlp.gate_up": "gate_up", "mlp.down": "down"}[block]
    return [getattr(layer, name) for layer in weights.layers]


FIVE = ("projection.fp8.in", "projection.fp8.out", "mlp.gate_up", "mlp.down", "output_head")
#: the blocks whose weights a run of the target stacks keeps on their stored format on compute capability 10.0: the two rows of
#: an entry that multiplies there (8-bit codes; 4-bit codes in the fused kernel's layout)
KEPT = ("projection.fp8.in", "mlp.gate_up")
OTHER_PLAN = "the weights of this session were loaded for another plan (plan every session of a run before the first load: Session.plan_table)"


def test_the_loader_keeps_each_weight_in_the_forms_the_plans_read(monkeypatch):
    """The loader keeps a multiplied weight in the forms it is asked for, block by block: the merged stored tensor for a
    block a provider multiplies on the stored format -- 8-bit codes with their one scale for the two projections, packed
    4-bit codes with their block scales and their global scale for the MLP's two multiplies and the output head, bit for bit
    what the bf16 copy is made from -- and the copy unless no block reads one.  A load that was asked for nothing holds a
    copy of every weight and no stored tensor, as every stack did.  The bytes the family states for each form are the
    bytes a load holds; a block it keeps nothing for, a weight left without any form and a multiply handed a weight
    without the form it reads are each named."""
    from model_samples.providers import library, torch_ref
    from model_samples.qwen_dense import QwenDense
    from model_samples.qwen_dense import model as family
    from model_samples.timing import Stages
    from model_samples.weights import BLOCK, BlockScaledTensor, Checkpoint, Dequantized, ScaledTensor, dequantize, quantize_fp8

    model = _tiny_model()
    stages = Stages(sync=lambda: None)

    def load(**forms):
        return model.load(Checkpoint.synthetic(model, 0), "cpu", stages, **forms)

    plain = load()
    assert family.STORED_WEIGHTS == FIVE and (plain.stored, plain.uncopied) == ((), ())
    assert all(weight.stored is None and weight.data is not None for block in FIVE for weight in _weights_of(plain, block)), "nothing asked: copies only"
    formats = {"projection.fp8.in": "fp8", "projection.fp8.out": "fp8", "mlp.gate_up": "nvfp4", "mlp.down": "nvfp4", "output_head": "nvfp4"}
    for block in FIVE:
        fmt, shapes = model.weight_shapes(block)
        assert fmt == formats[block] and len(shapes) == len(_weights_of(plain, block)), block
        beside, alone = load(stored=(block,)), load(stored=(block,), uncopied=(block,))
        assert (beside.stored, beside.uncopied, alone.stored, alone.uncopied) == ((block,), (), (block,), (block,)), "the weights say what the load kept"
        for other in FIVE:
            for copy, kept, bare in zip(_weights_of(plain, other), _weights_of(beside, other), _weights_of(alone, other)):
                assert torch.equal(kept.data.view(torch.int16), copy.data.view(torch.int16)), "a copy is the copy of every stack"
                if other != block:
                    assert kept.stored is None and bare.stored is None, f"{block}: only what was asked for ({other})"
                    assert torch.equal(bare.data.view(torch.int16), copy.data.view(torch.int16))
        held = 0
        for copy, kept, bare, shape in zip(_weights_of(plain, block), _weights_of(beside, block), _weights_of(alone, block), shapes):
            stored = kept.stored
            assert tuple(copy.data.shape) == shape, f"{block}: the merged weight"
            if fmt == "fp8":  # the merged 8-bit codes with their one scale
                assert isinstance(stored, ScaledTensor) and isinstance(stored.scale, float) and stored.codes.dtype == torch.uint8
                assert tuple(stored.codes.shape) == shape
                held += stored.codes.numel()
            else:  # the packed 4-bit codes (two per byte), one E4M3 block scale per 16 elements, the global scale
                assert isinstance(stored, BlockScaledTensor) and isinstance(stored.scale, float)
                assert stored.codes.dtype == stored.block_scales.dtype == torch.uint8
                assert tuple(stored.codes.shape) == (shape[0], shape[1] // 2) and tuple(stored.block_scales.shape) == (shape[0], shape[1] // BLOCK)
                held += stored.codes.numel() + stored.block_scales.numel()
            again = dequantize(stored)  # exactly what the copy was made from
            assert torch.equal(again.data.view(torch.int16), copy.data.view(torch.int16)) and again.scale == copy.scale == stored.scale == kept.scale
            # on the stored format alone: the same stored tensor, the same scale, no copy
            assert bare.data is None and bare.scale == copy.scale and type(bare.stored) is type(stored)
            assert all(torch.equal(a, b) for a, b in zip(bare.stored.tensors(), stored.tensors()))
            with pytest.raises(
                ValueError,
                match=rf"^this multiply reads a bf16 copy of its weight, and the load kept that weight on its stored format alone: {re.escape(OTHER_PLAN)}$",
            ):
                bare.bf16()
            assert kept.bf16() is kept.data and copy.bf16() is copy.data
        # the family's arithmetic is what a load holds
        assert model.stored_bytes((block,)) == held > 0, block
        copies = sum(weight.data.numel() * weight.data.element_size() for weight in _weights_of(plain, block))
        assert model.copy_bytes() - model.copy_bytes((block,)) == copies == 2 * sum(out * inner for out, inner in shapes), block
        assert alone.copy_bytes == model.copy_bytes((block,)) and beside.copy_bytes == plain.copy_bytes == model.copy_bytes()
        assert model.check_bytes((block,)) == max(2 * out * inner for out, inner in shapes), "the largest copy the checks make of it"
    every = load(stored=FIVE, uncopied=FIVE)
    assert all(weight.data is None and weight.stored is not None for block in FIVE for weight in _weights_of(every, block))
    assert every.embedding.dtype == torch.bfloat16 and every.layers[0].input_norm.dtype == torch.bfloat16, "what is no multiplied weight stays as it is"
    # what the loader cannot keep, and a weight that would be left without any form
    blocks_text = ", ".join(FIVE)
    with pytest.raises(
        ValueError, match=rf"^this family's loader keeps the stored weights of {blocks_text}; a provider of this run multiplies on those of attention.core$"
    ):
        load(stored=("attention.core",))
    with pytest.raises(ValueError, match=r"^a load that keeps no bf16 copy of the weights of mlp.down must keep their stored format, and was not asked to$"):
        load(stored=("projection.fp8.in",), uncopied=("mlp.down",))
    with pytest.raises(ValueError, match=rf"^block attention.core multiplies no weight this family's loader keeps; those are {blocks_text}$"):
        model.weight_shapes("attention.core")
    with pytest.raises(ValueError, match=r"^a weight is held as a bf16 copy, on its stored format, or as both; this one as neither$"):
        Dequantized(None, 1.0)
    # a multiply over a copy and the reference multiply read the copy: handed a weight without one, each says so
    bare = _weights_of(load(stored=("projection.fp8.out",), uncopied=("projection.fp8.out",)), "projection.fp8.out")[0]
    qx = quantize_fp8(torch.ones((1, bare.stored.codes.shape[1]), dtype=torch.bfloat16), 0.25)
    for multiply in (lambda: library.matmul_bf16(qx, bare), lambda: torch_ref.linear(qx, bare, torch.float64)):
        with pytest.raises(ValueError, match=r"^this multiply reads a bf16 copy of its weight, and the load kept that weight on its stored format alone: "):
            multiply()

    # the published model: bytes per block and format, the copies, the bound of the checks, the memory need, the bytes of a step
    full = QwenDense()
    assert full.geometry.in_projection("linear_attention") == (16384, 5120) and full.geometry.in_projection("full_attention") == (14336, 5120)
    assert full.weight_shapes("projection.fp8.out") == ("fp8", ((5120, 6144),) * 64) and full.weight_shapes("mlp.gate_up") == ("nvfp4", ((34816, 5120),) * 64)
    assert full.weight_shapes("mlp.down") == ("nvfp4", ((5120, 17408),) * 64) and full.weight_shapes("output_head") == ("nvfp4", ((248320, 5120),))
    stored_bytes = {
        "projection.fp8.in": 48 * 16384 * 5120 + 16 * 14336 * 5120,  # one byte per element
        "projection.fp8.out": 64 * 5120 * 6144,
        "mlp.gate_up": 64 * 34816 * 5120 * 9 // 16,  # half a byte per element and one block-scale byte per 16
        "mlp.down": 64 * 5120 * 17408 * 9 // 16,
        "output_head": 248320 * 5120 * 9 // 16,
    }
    assert stored_bytes["projection.fp8.in"] == 5_200_936_960 and stored_bytes["mlp.gate_up"] == 6_417_285_120 and sum(stored_bytes.values()) == 17_555_292_160
    assert full.stored_bytes() == 0 and all(full.stored_bytes((block,)) == stored_bytes[block] for block in FIVE)
    assert full.stored_bytes(FIVE) == sum(stored_bytes.values())
    assert full.copy_bytes() - full.copy_bytes(("projection.fp8.in",)) == 2 * stored_bytes["projection.fp8.in"], "two bytes per element of a copy"
    assert full.copy_bytes() - full.copy_bytes(FIVE) == 2 * (5_200_936_960 + 64 * 5120 * 6144 + 64 * 34816 * 5120 + 64 * 5120 * 17408 + 248320 * 5120)
    assert full.check_bytes() == 0 and full.check_bytes(("projection.fp8.in",)) == 2 * 16384 * 5120 and full.check_bytes(FIVE) == 2 * 248320 * 5120
    plain_need = full.memory_bytes(1, 64)
    assert plain_need == full.copy_bytes() + full.state_bytes(1, 64) + full.copy_bytes() // 8, "every weight as a copy: as before"
    codes = stored_bytes["projection.fp8.in"]
    beside = full.memory_bytes(1, 64, ("projection.fp8.in",)) - plain_need
    assert (
        beside == codes + codes // 8 + codes // 4
    ), "the stored form beside the copy: one byte per element, an eighth of it as of everything held, a quarter more"
    alone = full.memory_bytes(1, 64, ("projection.fp8.in",), ("projection.fp8.in",))
    held = full.copy_bytes(("projection.fp8.in",)) + codes
    assert (
        alone == held + full.state_bytes(1, 64) + held // 8 + codes // 4 + 2 * 16384 * 5120
    ), "the stored form alone: what is held, the working memory, the checks' one copy"
    small = _tiny_model()
    assert small.memory_bytes(1, 16, FIVE, FIVE) == small.copy_bytes(FIVE) + small.stored_bytes(FIVE) + small.state_bytes(1, 16) + 2 * GIB + small.stored_bytes(
        FIVE
    ) // 4 + small.check_bytes(FIVE), "a small model: 2 GiB where an eighth of what is held is less"
    needs = [
        full.memory_bytes(1, 90),
        full.memory_bytes(1, 90, ("projection.fp8.in",)),
        full.memory_bytes(1, 90, ("projection.fp8.in",), ("projection.fp8.in",)),
        full.memory_bytes(1, 90, KEPT, KEPT),  # what the plan of the stack ``native`` keeps on compute capability 10.0
        full.memory_bytes(1, 90, ("mlp.gate_up",), ("mlp.gate_up",)),  # ... and on the other 10.x classes
    ]
    assert (
        " ".join(f"{need / GIB:.1f}" for need in needs) == "56.7 63.3 52.6 37.1 41.3"
    ), "the published checkpoint, one sequence: copies; both forms of the in-projections; their codes alone; with the gate / up codes alone; those alone"
    fused = stored_bytes["mlp.gate_up"]
    both = full.copy_bytes(KEPT) + codes + fused
    assert (
        needs[3] == both + full.state_bytes(1, 90) + both // 8 + (codes + fused) // 4 + 2 * 34816 * 5120
    ), "the checks' one copy is the largest: a gate / up weight"
    assert full.check_bytes(KEPT) == 2 * 34816 * 5120 == full.check_bytes(("mlp.gate_up",))
    # ... and the family README and the script's help state these figures, not others
    readme = " ".join((MODELS / "qwen_dense" / "README.md").read_text().split())
    for said in (
        f"holds every weight as a bf16 copy, {full.copy_bytes() / GIB:.1f} GiB for this checkpoint, and its plan asks for about {needs[0] / GIB:.1f} GiB free",
        f"one byte per weight element: {codes / GIB:.1f} GiB on the published model",
        f"holds no copy of the in-projections ({2 * codes / GIB:.1f} GiB on the published model)",
        f"{codes / GIB:.1f} GiB where their copies are {2 * codes / GIB:.1f} GiB, and the gate / up weights on their stored 4-bit codes alone, "
        f"{fused / GIB:.1f} GiB where their copies are {2 * 64 * 34816 * 5120 / GIB:.1f} GiB: it holds {both / GIB:.1f} GiB of weights, and its plan asks for "
        f"about {needs[3] / GIB:.1f} GiB",
        f"and {full.check_bytes(KEPT) / GIB:.2f} GiB for the one copy its checks make at a time",
        f"where the in-projections stay on their copies, its plan asks for about {needs[4] / GIB:.1f} GiB; that is arithmetic, not a run",
        f"and no bf16 copy of it: {fused / GIB:.1f} GiB on the published model where the copies are {2 * 64 * 34816 * 5120 / GIB:.1f} GiB",
    ):
        assert said in readme, said
    script = " ".join((MODELS / "qwen_dense" / "00_decode.py").read_text().split())
    assert (
        f"about {needs[3] / GIB:.0f} GiB on compute capability 10.0, about {needs[4] / GIB:.0f} GiB on the other 10.x classes, "
        f"about {needs[0] / GIB:.0f} GiB elsewhere"
    ) in script
    assert f"{full.copy_bytes() / GIB:.1f} {full.copy_bytes(('projection.fp8.in',)) / GIB:.1f} {stored_bytes['projection.fp8.in'] / GIB:.1f}" == "50.1 40.4 4.8"
    # what a step reads: a stored weight on its format where its copy has two bytes per element; every other figure as it was
    copied, read = full.bytes_per_step(1, 58), full.bytes_per_step(1, 58, stored=("projection.fp8.in",))
    assert copied["stored"] == 0 and read["stored"] == 5_200_936_960 == copied["copies"] - read["copies"], "the copies of these weights are not read by a step"
    assert (read["weights"], read["state"]) == (copied["weights"], copied["state"]) and full.bytes_per_step(1, 58, stored=()) == copied
    four = full.bytes_per_step(1, 58, stored=("mlp.down",))
    assert (
        four["stored"] == stored_bytes["mlp.down"] and copied["copies"] - four["copies"] == 2 * 64 * 5120 * 17408 - stored_bytes["mlp.down"]
    ), "4 bits: 0.5625 for 2"
    whole = full.bytes_per_step(1, 58, stored=FIVE)
    assert (
        whole["stored"] == 17_555_292_160 and whole["copies"] == whole["weights"]
    ), "every multiply on its stored format: the step reads the model's own bytes"
    native = full.bytes_per_step(1, 58, stored=KEPT)
    assert native["stored"] == read["stored"] + fused and copied["copies"] - native["copies"] == read["stored"] + 2 * 64 * 34816 * 5120 - fused
    said = f"reads the in-projections on their stored codes, {read['stored'] / 1e9:.1f} GB where the copies are {2 * read['stored'] / 1e9:.1f} GB, "
    said += f"and the gate / up weights on theirs, {fused / 1e9:.1f} GB where the copies are {2 * 64 * 34816 * 5120 / 1e9:.1f} GB: "
    said += f"{(native['copies'] + native['state']) / 1e9:.1f} GB per step"
    assert said in " ".join((MODELS / "qwen_dense" / "README.md").read_text().split()), said


def test_a_run_keeps_no_copy_of_a_weight_that_every_plan_multiplies_on_its_stored_format(monkeypatch):
    """Which forms a load keeps is decided from the plans of the run, before the first load: the stored format of a weight
    for a block that multiplies on it in SOME plan (this session's or a follower's), and no bf16 copy of it where EVERY plan
    has such a block for it.  One follower that reads the copy -- a stack without the row, a row selected away -- keeps the
    copy.  The report's bytes of a step follow the session's own plan.  A stored weight is kept in ONE row order: plans
    that multiply on it in two -- two providers' layouts, or a layout beside the checkpoint's order -- are refused with
    the plans, in either order."""
    from model_samples import Session, TextIO
    from model_samples.blocks import DeviceFacts, plan

    from model_samples.weights import InterleavedBlockScaledTensor, ScaledTensor

    # a session alone: the stand-ins of the projection entry and of the fused gate / up entry multiply on the stored codes, and
    # nothing else reads those weights
    session = _recorded_run(monkeypatch, sequences=1, stack="native")[0]
    assert session.stored_weights() == session.uncopied_weights() == KEPT
    assert (session.weights.stored, session.weights.uncopied) == (KEPT, KEPT)
    assert all(
        type(layer.in_proj.stored) is ScaledTensor and layer.in_proj.data is None for layer in session.weights.layers
    ), "the stored codes, and no copy of them"
    assert all(
        type(layer.gate_up.stored) is InterleavedBlockScaledTensor and layer.gate_up.data is None for layer in session.weights.layers
    ), "the 4-bit gate / up weight in the layout its provider multiplies on, and no copy of it"
    assert list(session.stored_layouts()) == ["mlp.gate_up"], "the one block whose provider has a layout of its own"
    assert all(layer.out_proj.data is not None and layer.out_proj.stored is None for layer in session.weights.layers), "a multiply over a copy keeps its copy"
    assert all(layer.down.data is not None and layer.down.stored is None for layer in session.weights.layers)
    # ... and the report's bytes of a step count them as they are read: on the stored format, not as copies
    moved = session.report().bytes
    assert moved == session.model.bytes_per_step(1, session.state.position, stored=KEPT) and moved["stored"] > 0
    elements = sum(out * inner for block in KEPT for out, inner in session.model.weight_shapes(block)[1])
    assert (
        moved["copies"] == session.model.bytes_per_step(1, session.state.position)["copies"] - 2 * elements + moved["stored"]
    ), "the two bytes per element of their copies off, their stored bytes on"
    assert "): bf16 copies of the weights, and 0.00 GB of weights in their stored format (arithmetic, not a measurement)" in session.report().text()
    session = _recorded_run(monkeypatch, sequences=1, stack="fe-cublas")[0]
    assert session.stored_weights() == session.uncopied_weights() == () and (session.weights.stored, session.weights.uncopied) == ((), ())
    assert all(layer.in_proj.stored is None and layer.in_proj.data is not None for layer in session.weights.layers), "a stack without such a block: copies"
    assert session.report().bytes == session.model.bytes_per_step(1, session.state.position) and session.report().bytes["stored"] == 0
    assert "): bf16 copies of the weights (arithmetic, not a measurement)" in session.report().text(), "every multiply of this stack reads a copy"

    build = _stand_ins(monkeypatch)
    tiny = _tiny_model()
    text = TextIO.synthetic(tiny.geometry.vocab)
    facts = DeviceFacts((10, 0), "stated device", 148, 0, 0)

    def sessions(first_stack, later_stack, overrides=None):
        first = Session(tiny, text, sequences=1, stack=first_stack, device="cpu", capacity=16)
        later = Session(tiny, text, sequences=1, stack=later_stack, device="cpu", capacity=16, share=first, overrides=overrides)
        for each in (first, later):
            options = {"reference_precision": "fp64", "device": "cpu", "check_kernels": each.check_kernels}
            each.plan = plan(tiny.provider_table, tiny.regimes, facts, 1, 1, each.stack, each.overrides, model=tiny, options=options, factories=build)
        return first, later

    # a session whose own plan has no such block keeps the stored format for the session that will run on its weights, beside the copy
    first, later = sessions("fe-cublas", "native")
    assert first.stored_weights() == first.uncopied_weights() == (), "its own plan asks for none"
    first.plan_table([later])
    assert first.stored_weights() == KEPT and first.uncopied_weights() == (), "the follower multiplies on the codes, this one reads the copies"
    first.load()
    later.load()
    assert later.weights is first.weights and (first.weights.stored, first.weights.uncopied) == (KEPT, ())
    assert all(layer.in_proj.stored is not None and layer.in_proj.data is not None for layer in later.weights.layers), "both forms"
    assert all(type(layer.gate_up.stored) is InterleavedBlockScaledTensor and layer.gate_up.data is not None for layer in later.weights.layers)
    assert first.weights.alone == (), "a weight with its copy has an independent reference: no checksum is taken of it"
    ids = later.prefill([text.encode("ab0")])
    assert later.read(ids) and later.plan["projection.fp8.in"].serving()[0] == "cudnn.Fp8ProjectionDecode #2"
    first.read(first.prefill([text.encode("ab0")]))
    assert first.stored_weights(own=True) == () and first.report().bytes["stored"] == 0, "its own steps read the copies, whatever its load keeps for the next"
    assert later.stored_weights(own=True) == KEPT and later.report().bytes["stored"] == tiny.stored_bytes(KEPT) > 0
    # the other way round: the follower reads the copy, so the copy is made although the first session never reads it
    first, later = sessions("native", "fe-cublas")
    first.plan_table([later])
    assert first.stored_weights() == KEPT and first.uncopied_weights() == ()
    first.load()
    later.load()
    assert all(layer.in_proj.stored is not None and layer.in_proj.data is not None for layer in first.weights.layers)
    assert all(layer.gate_up.stored is not None and layer.gate_up.data is not None for layer in first.weights.layers)
    assert later.read(later.prefill([text.encode("ab0")])), "the multiplies over a copy find their copies"
    # the same stack with one row selected away in the follower: that weight's copy stays, the other's goes; with the rows in both: both go
    first, later = sessions("native", "native", overrides={"projection.fp8.in": "cublas.bf16_dequantized"})
    first.plan_table([later])
    assert first.uncopied_weights() == ("mlp.gate_up",) and first._kept() == {"stored": KEPT, "uncopied": ("mlp.gate_up",)}
    first, later = sessions("native", "native", overrides={"mlp.gate_up": "cublas.bf16_dequantized+torch_ref.silu_mul_quantize"})
    first.plan_table([later])
    assert first.uncopied_weights() == ("projection.fp8.in",) and first._kept() == {"stored": KEPT, "uncopied": ("projection.fp8.in",)}
    first.load()
    later.load()
    assert all(layer.gate_up.data is not None and layer.in_proj.data is None for layer in first.weights.layers)
    assert later.read(later.prefill([text.encode("ab0")])), "the gate / up multiply over a copy reads the copy in the checkpoint's row order"
    first, later = sessions("native", "native")
    first.plan_table([later])
    assert first.uncopied_weights() == KEPT and first._kept() == {"stored": KEPT, "uncopied": KEPT}
    first.load()
    later.load()
    assert all(layer.in_proj.data is None and layer.gate_up.data is None for layer in later.weights.layers) and later.read(later.prefill([text.encode("ab0")]))
    # a load keeps ONE stored form of a weight: two plans whose providers multiply on two layouts of it are a wrong table
    first, later = sessions("native", "native")
    other = type("OtherLayout", (type(later.plan["mlp.gate_up"]),), {})
    later.plan["mlp.gate_up"] = other(later.plan.rows["mlp.gate_up"], tiny, facts, {})
    first._followers = (later,)
    with pytest.raises(
        ValueError,
        match=r"^the provider table is wrong: two plans of this run multiply on two layouts of the stored weight of mlp.gate_up \(Swiglu, OtherLayout\); a load keeps one$",
    ):
        first.stored_layouts()
    # ... and so are a provider's own layout in one plan and the checkpoint's row order in another (a block that multiplies on
    # the merged tensor as the checkpoint has it): one weight in two row orders, refused where the plans are -- whichever
    # plan comes first, with the table of the run and again at the load, before anything is read
    for own_first, orders in ((True, "Swiglu, the checkpoint's row order"), (False, "the checkpoint's row order, Swiglu")):
        first, later = sessions("native", "native")
        plain = later if own_first else first
        merged = type("OnTheMergedTensor", (type(plain.plan["mlp.gate_up"]),), {"stored_layout": None})
        plain.plan["mlp.gate_up"] = merged(plain.plan.rows["mlp.gate_up"], tiny, facts, {})
        said = f"the provider table is wrong: two plans of this run multiply on two layouts of the stored weight of mlp.gate_up ({orders}); a load keeps one"
        with pytest.raises(ValueError) as refused:
            first.plan_table([later])
        assert str(refused.value) == said, "with the table of the run: before the memory sentence, before an entry builds anything"
        first._followers = (later,)
        with pytest.raises(ValueError) as refused:
            first.stored_layouts()
        assert str(refused.value) == said
        with pytest.raises(ValueError) as refused:
            first.load()
        assert str(refused.value) == said and first.weights is None and first.stages.seconds == {}, "at the load: nothing was read for it"
    # ... while a plan whose block reads the bf16 copy is no reader of the stored form: beside a layout it asks for nothing
    first, later = sessions("native", "native", overrides={"mlp.gate_up": "cublas.bf16_dequantized+torch_ref.silu_mul_quantize"})
    first.plan_table([later])
    assert list(first.stored_layouts()) == ["mlp.gate_up"]
    first, later = sessions("fe-cublas", "fe-cublas")
    first.plan_table([later])
    assert first._kept() == {}, "a run whose every multiply reads a copy asks its family's loader for nothing"


def test_a_form_of_a_weight_the_load_did_not_keep_fails_by_name_before_the_first_step(monkeypatch):
    """A session that runs on another session's weights without having been planned with it may read a form that load did
    not keep.  Its own ``load()`` says so before anything is compiled or stepped: a block that multiplies on a stored
    format the load kept none of, a block that reads a bf16 copy of a weight the load kept on its stored format alone, and
    a block that multiplies on a stored weight in another row order than the load kept it in, each by its entry and its
    block.  The reference session of the checks is the one reader that may come without a copy:
    it makes its own."""
    from model_samples import Session, TextIO
    from model_samples.blocks import DeviceFacts, plan
    from model_samples.weights import OnDemand

    build = _stand_ins(monkeypatch)
    tiny = _tiny_model()
    text = TextIO.synthetic(tiny.geometry.vocab)
    facts = DeviceFacts((10, 0), "stated device", 148, 0, 0)

    def session(stack, share=None, **options):
        made = Session(tiny, text, sequences=1, stack=stack, device="cpu", capacity=16, share=share, **options)
        plan_options = {"reference_precision": "fp64", "device": "cpu", "check_kernels": made.check_kernels}
        made.plan = plan(tiny.provider_table, tiny.regimes, facts, 1, 1, stack, model=tiny, options=plan_options, factories=build)
        return made

    # (1) weights loaded for a plan of copies; a session on them whose block multiplies on the stored codes
    first = session("fe-cublas")
    first.load()  # without the table of the run: its plan alone
    later = session("native", share=first)
    with pytest.raises(ValueError) as missing:
        later.load()
    assert (
        str(missing.value)
        == f"cudnn.Fp8ProjectionDecode multiplies on the stored format of the weight of projection.fp8.in, and the load kept none: {OTHER_PLAN}"
    )
    assert later.state is None and later.record is None, "before the first step, and before anything is compiled for it"
    # (2) weights loaded for a plan on the stored codes alone; a session on them whose block reads the copy
    first = session("native")
    first.load()
    assert first.weights.uncopied == KEPT
    for stack, entry in (("fe-cublas", "torch.matmul over a bf16 copy"), ("reference", "torch_ref.linear")):
        later = session(stack, share=first)
        with pytest.raises(ValueError) as missing:
            later.load()
        assert str(missing.value) == f"{entry} reads a bf16 copy of the weight of projection.fp8.in, and the load kept its stored format alone: {OTHER_PLAN}"
        assert later.state is None
    # (3) the reference session of the checks comes with its own maker of copies, and runs
    cost = OnDemand()
    reference = session("reference", share=first, on_demand=cost)
    reference.load()
    assert cost.copies == 4 and reference.read(reference.prefill([text.encode("ab0")])), "the warm-up step: one copy per in-projection and gate / up call"
    assert cost.copies == 4 + 4 * 3 and all(
        layer.in_proj.data is None and layer.gate_up.data is None for layer in first.weights.layers
    ), "one per call; the weights are as they were"
    # (3a) with the in-projection selected back onto its copy the gate / up weight is the one without a copy, and a session on such
    # weights that reads ITS copy is named by that block
    kept = Session(tiny, text, sequences=1, stack="native", device="cpu", capacity=16, overrides={"projection.fp8.in": "cublas.bf16_dequantized"})
    plan_options = {"reference_precision": "fp64", "device": "cpu", "check_kernels": True}
    kept.plan = plan(tiny.provider_table, tiny.regimes, facts, 1, 1, "native", kept.overrides, model=tiny, options=plan_options, factories=build)
    kept.load()
    assert kept.weights.uncopied == ("mlp.gate_up",)
    later = session("fe-cublas", share=kept)
    with pytest.raises(ValueError) as missing:
        later.load()
    said = (
        "torch.matmul over a bf16 copy + torch_ref.silu_mul_quantize reads a bf16 copy of the weight of mlp.gate_up, and the load kept its stored format alone"
    )
    assert str(missing.value) == f"{said}: {OTHER_PLAN}"
    # (3b) a load keeps a stored weight in ONE row order.  A session whose block multiplies on its own layout of a weight the load
    # kept in the checkpoint's order is named before its first step -- not at the adapter's first call -- and so is the reverse
    plain = session("native")
    plain.stored_layouts = lambda: {}  # a load that was not told of the layout
    said = "cudnn.SwigluFp4Decode multiplies on the stored weight of mlp.gate_up in its own row order, and the load kept it in the checkpoint's"
    for made in (plain, session("native", share=plain)):
        with pytest.raises(ValueError) as missing:
            made.load()
        assert str(missing.value) == f"{said}: {OTHER_PLAN}" and made.state is None and made.weights.laid_out == ()
    assert first.weights.laid_out == ("mlp.gate_up",), "the load of the plan itself kept the layout of its provider"
    later = session("native", share=first)
    dict(later.plan.items())["mlp.gate_up"].stored_layout = None  # a block that multiplies on the merged tensor as the checkpoint has it
    with pytest.raises(ValueError) as missing:
        later.load()
    said = "cudnn.SwigluFp4Decode multiplies on the stored weight of mlp.gate_up in the checkpoint's row order, and the load kept it in a provider's own"
    assert str(missing.value) == f"{said}: {OTHER_PLAN}" and later.state is None
    # (4) planned together, nothing is missing in either direction
    first = session("fe-cublas")
    later = session("native", share=first)
    first.plan_table([later])
    first.load()
    later.load()
    assert later.read(later.prefill([text.encode("ab0")]))


def test_the_reference_of_the_checks_makes_its_copy_of_a_weight_for_one_call(monkeypatch):
    """The reference blocks of the checks multiply bf16 copies.  Where the run keeps a weight on its stored format alone,
    the checks make the copy themselves, for one call at a time (``weights.OnDemand``): bit for bit the copy a load would
    have kept, the session's weights untouched, and nothing held after the call.  So every check line of such a run is
    the line of the same run with its copies kept -- shown here by loading the same plan both ways -- and a wrong element
    still fails the band.  What the copies cost is counted and printed under the checks; a run that made none prints no
    such line."""
    from model_samples import Session, tie_out
    from model_samples.weights import BlockScaledTensor, Dequantized, OnDemand, ScaledTensor, dequantize, quantize_fp8, quantize_nvfp4

    # the maker on stated operands: 8-bit, 4-bit, one scale per row; operands that need nothing come back as they are
    values = torch.linspace(-3.0, 3.0, 4 * 32).reshape(4, 32)
    eight, four = quantize_fp8(values, 0.02), quantize_nvfp4(values, 0.01)
    rows = ScaledTensor(eight.codes, torch.tensor([0.02, 0.03, 0.04, 0.05]))
    cost = OnDemand()
    assert (cost.copies, cost.bytes, cost.largest, cost.seconds) == (0, 0, 0, 0.0)
    activation, with_copy = quantize_fp8(torch.ones((1, 32)), 0.25), Dequantized(dequantize(eight).data, eight.scale, eight)
    untouched = (activation, with_copy, "site")
    assert cost.operands(untouched) is untouched and cost.copies == 0, "a weight with its copy is handed on as it is"
    for stored in (eight, four, rows):
        bare = Dequantized(None, stored.scale, stored)
        before = (cost.copies, cost.bytes)
        operand, made, site = cost.operands((activation, bare, "site"))
        assert operand is activation and site == "site" and made is not bare and bare.data is None, "the run's own weight is not changed"
        assert made.stored is stored and made.scale is stored.scale and made.data.dtype == torch.bfloat16 and tuple(made.data.shape) == (4, 32)
        assert torch.equal(made.data.view(torch.int16), dequantize(stored).data.view(torch.int16)), "the copy a load would have kept"
        assert (cost.copies, cost.bytes) == (before[0] + 1, before[1] + 4 * 32 * 2)
    assert isinstance(four, BlockScaledTensor) and cost.largest == 4 * 32 * 2 and cost.seconds >= 0.0

    def run(sequences, copies, **stand_in):
        with monkeypatch.context() as scoped:
            if copies:  # the same plan with its copies kept, as a load that was asked for both forms holds them
                scoped.setattr(Session, "uncopied_weights", lambda self: ())
            session = _recorded_run(scoped, sequences=sequences, stack="native", **stand_in)[0]
            session.trace(4)
            return session, tie_out.run_checks(session, prompt_tokens=3)

    for sequences in (1, 2):
        session, checks = run(sequences, copies=False)
        kept_session, kept = run(sequences, copies=True)
        assert session.weights.uncopied == KEPT and kept_session.weights.uncopied == ()
        assert checks.ok and kept.ok, checks.text() + kept.text()
        assert all(layer.in_proj.data is None and layer.gate_up.data is None for layer in session.weights.layers), "the weights of the run are as they were"
        on_demand = [line for line in checks.lines if line.startswith("  on demand ")]
        assert len(on_demand) == 1 and not any(line.startswith("  on demand ") for line in kept.lines), "said only where a copy was made"
        # ... and the line that holds the stored bytes such a copy is made from to those of the load (the test below)
        held = [line for line in checks.lines if line.startswith("  stored weights ")]
        assert len(held) == 1 and held[0].endswith("    PASS") and not any(line.startswith("  stored weights ") for line in kept.lines)
        assert [line for line in checks.lines if line not in (*on_demand, *held)] == kept.lines, "every check line is the line of the run that kept its copies"
        assert checks.lines.index(on_demand[0]) == checks.lines.index(_line(checks, "kernels ")) - 1, "under the four checks, above the kernel check"
        assert checks.lines.index(held[0]) == checks.lines.index(on_demand[0]) - 1, "the condition of the copies above their cost"
        # one copy per in-projection and per gate / up call of a reference block: the twin's call of the teacher-forced pass and the
        # reference stack's own pass, 4 calls a step over the 7 recorded steps, and the 4 calls of the reference session's warm-up step
        steps = 7
        shapes = [shape for block in KEPT for shape in session.model.weight_shapes(block)[1]]
        per_step = sum(2 * out * inner for out, inner in shapes)
        cost = checks.on_demand
        assert cost.copies == 4 + 2 * 4 * steps and cost.bytes == (1 + 2 * steps) * per_step and cost.largest == max(2 * out * inner for out, inner in shapes)
        assert kept.on_demand.copies == 0 and kept.on_demand.bytes == 0
        assert on_demand[0] == tie_out.on_demand_line(cost, KEPT), "the line of this run's own ledger"
        assert (
            "  blocks " not in on_demand[0] and "  tokens " not in on_demand[0] and not on_demand[0].endswith(("PASS", "FAIL", "REPORTED"))
        ), "it is no verdict"
    # the line on the figures of a run of the two-layer model at the published widths: 460 copies (two in-projections and two gate / up
    # weights per step and reader), the largest and their sum in GiB
    ledger = OnDemand()
    ledger.copies, ledger.bytes, ledger.largest, ledger.seconds = 460, 115 * 2 * (16384 + 14336 + 2 * 34816) * 5120, 2 * 34816 * 5120, 2.2345
    assert tie_out.on_demand_line(ledger, KEPT) == (
        "  on demand         the reference read 460 bf16 copies of weights this run keeps on their stored format alone (projection.fp8.in, mlp.gate_up): "
        "each made for one call and dropped after it; the largest 0.33 GiB, 110.1 GiB in all, 2.2 s of the checks (orientation only)"
    )
    ledger.copies = 3072
    assert "the reference read 3,072 bf16 copies of weights this run keeps on their stored format alone (mlp.down, output_head): " in tie_out.on_demand_line(
        ledger, ("mlp.down", "output_head")
    )
    # a second reference reads its copies the same way: three readers
    session = _recorded_run(monkeypatch, sequences=1, stack="native")[0]
    checks = tie_out.run_checks(session, prompt_tokens=3, second_reference=True)
    assert checks.ok and checks.on_demand.copies == 2 * 4 + 3 * 4 * 7
    # the reference on a copy made on demand is a reference: one wrong element of the block fails its band
    session, checks = run(2, copies=False, projection_defect="element")
    line = next(line for line in checks.lines if line.strip().startswith("projection.fp8.in ") and "calls" in line)
    assert not checks.ok and line.endswith("[band: bf16_rounding]  FAIL") and checks.on_demand.copies > 0, line


def test_the_checks_hold_a_weight_kept_on_its_stored_format_alone_to_the_bytes_of_its_load(monkeypatch):
    """The reference's copy of a weight a run keeps on its stored format alone is made from the bytes the weight's own
    block multiplies on.  A block that writes into them moves its reference with it, and its block line passes: the block
    is compared with a multiply on its own changed weight.  So the loader takes a checksum of every such weight
    (``weights.stored_checksum``), the checks take it again before their first pass and after their last, and a weight
    whose stored bytes are not those of the load fails the run, named by block and layer, whatever the block lines say.
    The same defect in a run that keeps its copies fails the band, as it did before a run could be without them."""
    from model_samples import Session, tie_out
    from model_samples.timing import Stages
    from model_samples.weights import BlockScaledTensor, Checkpoint, ScaledTensor, stored_checksum

    # (1) the checksum on stated tensors: 8-bit codes with one scale, 4-bit codes with their block scales, a scale per row
    codes = torch.arange(4 * 32, dtype=torch.uint8).reshape(4, 32)
    scales = (torch.arange(4 * 2, dtype=torch.uint8) + 100).reshape(4, 2)
    makers = {
        "eight": (lambda: ScaledTensor(codes.clone(), 0.02), 1, 1),
        "four": (lambda: BlockScaledTensor(codes[:, :16].clone(), scales.clone(), 0.01), 2, 1),
        "per row": (lambda: ScaledTensor(codes.clone(), torch.tensor([0.02, 0.03, 0.04, 0.05])), 2, 0),
    }
    for name, (make, tensors, floats) in makers.items():
        stored = make()
        first = stored_checksum(stored)
        assert len(first) == 3 * tensors + floats and all(type(value) is int for value in first[: 3 * tensors]), f"{name}: three 64-bit sums per tensor"
        assert stored_checksum(make()) == first == stored_checksum(stored), f"{name}: the same bytes, the same checksum; taking it changes nothing"
        for index in range(len(stored.tensors())):
            for damage in ("one byte", "two rows exchanged", "two bytes exchanged along a row"):
                changed = make()
                tensor = changed.tensors()[index]
                last = tensor.shape[1] - 1
                if damage == "one byte":
                    tensor[1, last] ^= 1
                elif damage == "two rows exchanged":  # every byte is still there
                    tensor[[0, 3]] = tensor[[3, 0]]
                else:  # at the same place of two 64-bit words where a row has two: the row's sum stays
                    other = 8 if tensor.shape[1] >= 16 else last
                    tensor[2, [0, other]] = tensor[2, [other, 0]]
                assert stored_checksum(changed) != first, f"{name}, tensor {index}: {damage}"
    assert stored_checksum(ScaledTensor(codes.clone(), 0.03)) != stored_checksum(ScaledTensor(codes.clone(), 0.02)), "the one scale is part of it"
    moved = makers["per row"][0]()
    moved.scale[2] = 0.5
    assert stored_checksum(moved) != stored_checksum(makers["per row"][0]()), "and a scale per row, by its bytes"
    # the same bytes give the same sums however the tensor lies in its storage: viewed as 64-bit words, or put together byte by byte
    odd = torch.arange(100, dtype=torch.uint8)[1:97].reshape(4, 24)
    assert odd.storage_offset() == 1 and stored_checksum(ScaledTensor(odd, 0.02)) == stored_checksum(ScaledTensor(odd.clone(), 0.02))
    narrow = ScaledTensor(codes[:, :12].clone(), 0.02)  # rows that are no whole number of words
    other = ScaledTensor(codes[:, :12].clone(), 0.02)
    other.codes[3, 11] ^= 0x80
    assert stored_checksum(narrow) == stored_checksum(ScaledTensor(codes[:, :12].clone(), 0.02)) != stored_checksum(other)

    # (2) the loader: a weight kept on its stored format alone carries the checksum of its stored bytes, and the weights list them
    model = _tiny_model()
    layers = len(model.geometry.layer_kinds)

    def load(**forms):
        stages = Stages(sync=lambda: None)
        return model.load(Checkpoint.synthetic(model, 0), "cpu", stages, **forms), stages

    plain, stages = load()
    assert plain.alone == () and "checksum" not in stages.seconds, "a load of copies takes none"
    assert all(weight.checksum is None and weight.intact() for block in FIVE for weight in _weights_of(plain, block))
    for block in FIVE:
        beside, stages = load(stored=(block,))
        assert beside.alone == () and "checksum" not in stages.seconds, "a weight with its copy has an independent reference: none is taken"
        assert all(weight.checksum is None and weight.intact() for weight in _weights_of(beside, block))
        alone, stages = load(stored=(block,), uncopied=(block,))
        assert "checksum" in stages.seconds, "its own stage of the load"
        assert [(name, layer) for name, layer, _ in alone.alone] == ([(block, None)] if block == "output_head" else [(block, i) for i in range(layers)])
        assert all(listed is weight for (_, _, listed), weight in zip(alone.alone, _weights_of(alone, block))), "the handles of the weights themselves"
        for weight in _weights_of(alone, block):
            assert weight.data is None and weight.checksum == stored_checksum(weight.stored) and weight.intact()
            tensor = weight.stored.tensors()[-1]
            kept = tensor[0, 0].clone()
            tensor[0, 0] ^= 1
            assert not weight.intact(), f"{block}: one changed byte of what the block multiplies on"
            tensor[0, 0] = kept
            assert weight.intact(), "the loaded bytes again"
    every, _ = load(stored=FIVE, uncopied=FIVE)
    assert len(every.alone) == 4 * layers + 1 and {name for name, _, _ in every.alone} == set(FIVE) and tie_out.changed_weights(every) == []
    every.layers[1].down.stored.block_scales[0, 0] ^= 1
    every.head.stored.codes[5, 3] ^= 0x10
    assert tie_out.changed_weights(every) == ["mlp.down of layer 1", "output_head"], "named by block and layer, in the order of the load"
    assert tie_out.changed_weights(plain) == [] and tie_out.changed_weights(object()) == [], "a family whose loader lists none has none to hold"

    # (3) through a session: a clean run says that the stored bytes are those of the load, above what the copies cost
    def run(copies=False, late=False, **stand_in):
        with monkeypatch.context() as scoped:
            if copies:  # the same plan with its copies kept: the reference reads the copy of the load
                scoped.setattr(Session, "uncopied_weights", lambda self: ())
            session = _recorded_run(scoped, sequences=1, stack="native", **stand_in)[0]
            session.trace(4)
            if late:  # the block starts to write when the checks start
                session.plan["projection.fp8.in"].writes = True
            return session, tie_out.run_checks(session, prompt_tokens=3)

    def lines(checks, start):
        return [line for line in checks.lines if line.startswith(start)]

    session, checks = run()
    said = "  stored weights    4 of 4 weights this run keeps on their stored format alone hold the bytes of their load, before the checks and after them: "
    assert checks.ok and lines(checks, "  stored weights ") == [said + "the reference's copies were made from what was loaded    PASS"]
    assert tie_out.stored_weights_line(4, [], []) == lines(checks, "  stored weights ")[0]
    assert [(name, layer) for name, layer, _ in session.weights.alone] == [(name, layer) for layer in (0, 1) for name in ("mlp.gate_up", "projection.fp8.in")]
    from model_samples.weights import InterleavedBlockScaledTensor

    for name, _, weight in session.weights.alone:  # the checksum is of the tensors the weight's block multiplies on, in the layout it has them
        assert weight.checksum == stored_checksum(weight.stored) and (type(weight.stored) is InterleavedBlockScaledTensor) == (name == "mlp.gate_up")

    # (4) the block writes into its stored weight from its first call on: half of the weight's rows are zero for the block AND for
    # the copy the reference makes of it, so the block line is the line of a clean run -- and the run fails on the weight
    def projection_line(checks):
        return next(line for line in checks.lines if line.strip().startswith("projection.fp8.in ") and "calls" in line)

    session, checks = run(projection_defect="writes")
    assert "8 of 8 provider blocks inside their bands" in _line(checks, "blocks") and _line(checks, "blocks").endswith("PASS")
    assert "largest distance 0.00 bf16 units" in projection_line(checks) and projection_line(checks).endswith("[band: bf16_rounding]  PASS")
    assert _line(checks, "replay").endswith("PASS") and _line(checks, "tokens").endswith("PASS") and checks.kernels == census_pass(), "every other line holds"
    failed = lines(checks, "  stored weights ")
    assert not checks.ok and len(failed) == 1, checks.text()
    assert failed[0] == (
        "  stored weights    2 of 4 weights this run keeps on their stored format alone hold the bytes of their load; written to since: "
        "projection.fp8.in of layer 0 (before the checks), projection.fp8.in of layer 1 (before the checks).  The reference's copy of such a weight "
        "was made from the changed bytes: its block was compared with a reference that moved with it    FAIL"
    )
    assert [weight.intact() for _, _, weight in session.weights.alone] == [True, False, True, False], "the gate / up weights are as loaded"
    # ... the same block in the same plan loaded WITH its copies: the reference reads the copy of the load, and the band fails
    session, checks = run(copies=True, projection_defect="writes")
    assert session.weights.alone == () and not lines(checks, "  stored weights ") and not checks.ok
    assert "largest distance 255.00 bf16 units; 50.0% of elements" in projection_line(checks) and projection_line(checks).endswith(
        "[band: bf16_rounding]  FAIL"
    )
    # ... and a block that starts to write during the checks is caught by the second reading, and said so
    session, checks = run(late=True)
    failed = lines(checks, "  stored weights ")
    assert not checks.ok and "projection.fp8.in of layer 0 (during the checks), projection.fp8.in of layer 1 (during the checks)" in failed[0], checks.text()
    assert failed[0].startswith("  stored weights    2 of 4 weights ") and failed[0].endswith("    FAIL")

    # (4a) the same for the fused gate / up entry, whose weight the load keeps in the kernel's own layout: the checksum is of THAT
    # tensor, so a block that writes into what it multiplies on is seen -- its block line holds (the reference's copy is made from
    # the written bytes), the run fails on the weight; with the copies kept the band fails
    def fused_line(checks):
        return next(line for line in checks.lines if line.strip().startswith("mlp.gate_up ") and "calls" in line)

    session, checks = run(swiglu_defect="writes")
    failed = lines(checks, "  stored weights ")
    assert not checks.ok and fused_line(checks).endswith("[band: swiglu_fp4]  PASS") and "8 of 8 provider blocks inside their bands" in _line(checks, "blocks")
    assert failed == [
        "  stored weights    2 of 4 weights this run keeps on their stored format alone hold the bytes of their load; written to since: "
        "mlp.gate_up of layer 0 (before the checks), mlp.gate_up of layer 1 (before the checks).  The reference's copy of such a weight "
        "was made from the changed bytes: its block was compared with a reference that moved with it    FAIL"
    ], checks.text()
    session, checks = run(copies=True, swiglu_defect="writes")
    assert session.weights.alone == () and not lines(checks, "  stored weights ") and not checks.ok
    assert fused_line(checks).endswith("[band: swiglu_fp4]  FAIL") and not fused_line(checks).split("; ")[-1].startswith("0 outside"), fused_line(checks)
    assert tie_out.stored_weights_line(3, ["a of layer 0"], ["a of layer 0", "b"]).startswith(
        "  stored weights    1 of 3 weights this run keeps on their stored format alone hold the bytes of their load; written to since: "
        "a of layer 0 (before the checks), b (during the checks).  "
    )

    # (5) every multiply on its stored format, each of a test's own blocks writing into its weight (4-bit: the block scales): the five
    # weights of those blocks are named, the output head without a layer; the four of the two entries' rows are as loaded
    session = _recorded_run(monkeypatch, sequences=1, stack="native", on_stored=True, stored_defect="writes")[0]
    session.trace(4)
    checks = tie_out.run_checks(session, prompt_tokens=3)
    failed = lines(checks, "  stored weights ")
    assert not checks.ok and len(failed) == 1 and failed[0].startswith("  stored weights    4 of 9 weights ") and failed[0].endswith("    FAIL")
    written = failed[0].split("written to since: ", 1)[1].split(".  The reference's copy", 1)[0].split(", ")
    assert written == [
        *(f"{block} of layer {layer} (before the checks)" for layer in range(2) for block in ("mlp.down", "projection.fp8.out")),
        "output_head (before the checks)",
    ], written


def census_pass():
    from model_samples import census

    return census.PASS


def test_a_copy_made_on_demand_is_gone_when_its_call_returns(monkeypatch):
    """One copy at a time: the bf16 copy the checks make of a weight for a reference call is the caller's for that call and
    is gone when the call returns -- through the twin's call of the teacher-forced pass, and through the passes of the
    reference session and of a second reference.  The memory sentence counts ONE such copy (the largest), so a copy that
    outlived its call would be memory the sentence does not state.  Held on the tensors themselves: when the next copy
    is made, and when the checks return, no earlier one is alive."""
    import gc
    import weakref

    from model_samples import tie_out
    from model_samples.weights import OnDemand

    made = []  # a weak reference to the tensor of every copy, in the order they were made
    alive_at_the_next = []
    with_copy = OnDemand._with_copy

    def watched(self, weight):
        alive_at_the_next.append(sum(reference() is not None for reference in made))
        handle = with_copy(self, weight)
        assert handle.data is not None and weight.data is None, "the copy is on a handle of its own; the run's weight has none"
        made.append(weakref.ref(handle.data))
        return handle

    monkeypatch.setattr(OnDemand, "_with_copy", watched)
    gc.disable()  # the copies must go by reference count, when the call returns: a collector that runs later would hide one that waits for it
    try:
        # two sequences: the tap also runs each block alone; all five multiplies on their stored format: every reference multiply reads a copy
        session = _recorded_run(monkeypatch, sequences=2, stack="native", on_stored=True)[0]
        session.trace(4)
        checks = tie_out.run_checks(session, prompt_tokens=3, second_reference=True)
        alive_after = sum(reference() is not None for reference in made)
    finally:
        gc.enable()
    assert checks.ok, checks.text()
    steps, calls = 7, 9  # nine multiplies a step; the twin's call, the reference pass, the second reference; two warm-up steps
    assert len(made) == checks.on_demand.copies == 2 * calls + 3 * calls * steps
    assert alive_at_the_next == [0] * len(made), "no copy of an earlier call is alive when the next one is made"
    assert alive_after == 0, "and none when the checks return"
    assert all(weight.data is None for block in FIVE for weight in _weights_of(session.weights, block)), "the run's weights hold none"
    # the reference session alone, outside the checks: a call through Session._call drops its copy with its operands
    del made[:], alive_at_the_next[:]
    from model_samples import Session

    cost = OnDemand()
    reference = Session(session.model, session.text, 2, stack="reference", device="cpu", capacity=16, share=session, on_demand=cost)
    reference.load()
    reference.read(reference.prefill([session.text.encode("ab" + str(s)) for s in range(2)]))
    assert len(made) == cost.copies == calls + 3 * calls and all(reference() is None for reference in made) and alive_at_the_next == [0] * len(made)


def test_only_the_reference_session_of_a_runs_checks_makes_copies_per_call(monkeypatch):
    """A copy made per call is a cast and two synchronisations in front of a block.  The reference session of a run's
    checks may carry them; a session whose own steps are timed, traced or checked may not, so ``on_demand`` is refused
    for every other session when it is built, and a session that has it is not traced."""
    from model_samples import Session, TextIO
    from model_samples.blocks import DeviceFacts, plan
    from model_samples.weights import OnDemand

    build = _stand_ins(monkeypatch)
    tiny = _tiny_model()
    text = TextIO.synthetic(tiny.geometry.vocab)
    facts = DeviceFacts((10, 0), "stated device", 148, 0, 0)

    def session(stack, **options):
        made = Session(tiny, text, sequences=1, stack=stack, device="cpu", capacity=16, **options)
        plan_options = {"reference_precision": "fp64", "device": "cpu", "check_kernels": made.check_kernels}
        made.plan = plan(tiny.provider_table, tiny.regimes, facts, 1, 1, stack, model=tiny, options=plan_options, factories=build)
        return made

    first = session("native")
    first.load()
    assert first.weights.uncopied == KEPT
    refused = "on_demand is for the reference session of a run's checks (stack reference, on the weights of the session it checks: share=); a session of stack "
    reads = " reads the forms its load kept and never dequantises a weight per call"
    for stack, options, what in (
        ("native", {}, "native"),  # the checked session itself: its steps would carry the copies
        ("fe-cublas", {}, "fe-cublas"),
        ("reference", {}, "reference"),  # a reference stack on its own load: it holds every copy
        ("native", {"share": first}, "native on shared weights"),
        ("fe-entries", {"share": first}, "fe-entries on shared weights"),
    ):
        with pytest.raises(ValueError) as error:
            Session(tiny, text, sequences=1, stack=stack, device="cpu", capacity=16, on_demand=OnDemand(), **options)
        assert str(error.value) == refused + what + reads, stack
    # the one session that may: the reference stack on the weights of the session it checks -- and it is not traced
    cost = OnDemand()
    reference = session("reference", share=first, on_demand=cost)
    reference.load()
    assert reference.read(reference.prefill([text.encode("ab0")])) and cost.copies == 4 + 4 * 3
    with pytest.raises(
        ValueError,
        match=r"^a session that makes bf16 copies of its weights per call \(on_demand\) is the reference of a run's checks; its steps are not traced$",
    ):
        reference.trace(2)
    assert reference.traced is None and cost.copies == 4 + 4 * 3, "nothing ran"
    # a session without one is built and traced as before
    plain = session("reference", share=first.__class__(tiny, text, sequences=1, stack="fe-cublas", device="cpu", capacity=16))
    assert plain.on_demand is None


def test_every_multiply_on_its_stored_format_through_a_session(monkeypatch):
    """Two provider rows of the family multiply on a stored weight (the in-projection's 8-bit codes, the gate / up weight's
    4-bit codes in the fused kernel's layout); for the other three multiplies a test's own stand-ins do, on the rows of
    the multiplies over a copy.  With them every multiplied weight of the run is on its stored format in every plan: the
    load keeps the 8-bit codes of both projections and the packed 4-bit codes, block scales and global scale of the MLP
    and the output head, and no bf16 copy of any of them; the blocks multiply on what was kept; the reference makes every
    copy on demand and each block is inside its band; the kernel check holds; the bytes of a step are the model's own; and
    the memory sentence states what is held and the checks' one copy.  One wrong element in a multiply on a 4-bit weight
    fails its band."""
    from model_samples import Session, TextIO, blocks, census, tie_out
    from model_samples.qwen_dense import QwenDense
    from model_samples.weights import BlockScaledTensor, InterleavedBlockScaledTensor, ScaledTensor

    session = _recorded_run(monkeypatch, sequences=2, stack="native", on_stored=True)[0]
    model = session.model
    assert session.stored_weights() == session.uncopied_weights() == FIVE and (session.weights.stored, session.weights.uncopied) == (FIVE, FIVE)
    for block in FIVE:
        kind = ScaledTensor if model.weight_shapes(block)[0] == "fp8" else (InterleavedBlockScaledTensor if block == "mlp.gate_up" else BlockScaledTensor)
        assert all(weight.data is None and type(weight.stored) is kind for weight in _weights_of(session.weights, block)), block
    assert session.weights.copy_bytes == model.copy_bytes(FIVE) < model.copy_bytes() // 4, "what is left as bf16: the embedding table, the norms, the conv taps"
    session.trace(4)
    checks = tie_out.run_checks(session, prompt_tokens=3)
    assert checks.ok and checks.kernels == census.PASS, checks.text()
    assert _line(checks, "blocks").endswith("PASS") and "8 of 8 provider blocks inside their bands" in _line(checks, "blocks")
    for block in FIVE:
        line = next(line for line in checks.lines if line.strip().startswith(block + " ") and "calls" in line)
        assert line.endswith("[band: swiglu_fp4]  PASS" if block == "mlp.gate_up" else "[band: bf16_rounding]  PASS"), line
    assert "projection.fp8.in equal (gated)" in _line(checks, "batch invariance") and _line(checks, "tokens").endswith("PASS")
    # nine multiplies a step (two layers of four, and the head), each read by the twin's call and by the reference stack's own pass
    steps, calls = 7, 9
    cost = checks.on_demand
    assert cost.copies == calls + 2 * calls * steps, "and the reference session's warm-up step"
    assert cost.bytes == (1 + 2 * steps) * (model.copy_bytes() - model.copy_bytes(FIVE)) and cost.largest == model.check_bytes(FIVE)
    assert model.check_bytes(FIVE) == 2 * model.geometry.vocab * model.geometry.hidden, "the output head's copy is the largest"
    assert f"keeps on their stored format alone ({', '.join(FIVE)}): each made for one call" in _line(checks, "on demand")
    assert all(weight.data is None for block in FIVE for weight in _weights_of(session.weights, block)), "nothing was kept by the checks"
    # the bytes of a step: every multiplied weight on its stored format, so the stack moves what the model holds
    moved = session.report().bytes
    assert moved == model.bytes_per_step(2, session.state.position, stored=FIVE) and moved["copies"] == moved["weights"]
    assert moved["stored"] == model.stored_bytes(FIVE) and " (1.00x): bf16 copies of the weights, and " in session.report().text()
    # one wrong element in every multiply of a test's own block on a 4-bit weight: down and the head
    session = _recorded_run(monkeypatch, sequences=1, stack="native", on_stored=True, stored_defect="BlockScaledTensor")[0]
    session.trace(4)
    checks = tie_out.run_checks(session, prompt_tokens=3)
    failed = [line.split()[0] for line in checks.lines if line.endswith("[band: bf16_rounding]  FAIL")]
    assert not checks.ok and checks.kernels == census.PASS and failed == ["mlp.down", "output_head"], checks.text()
    # ... and on an 8-bit weight outside the entry's row: the out-projection
    session = _recorded_run(monkeypatch, sequences=1, stack="native", on_stored=True, stored_defect="ScaledTensor")[0]
    session.trace(4)
    checks = tie_out.run_checks(session, prompt_tokens=3)
    failed = [line.split()[0] for line in checks.lines if line.endswith("[band: bf16_rounding]  FAIL")]
    assert not checks.ok and failed == ["projection.fp8.out"], checks.text()

    # the memory sentence of such a plan, on the published widths: what is held and, named, the checks' one copy
    build = _stand_ins(monkeypatch, on_stored=True)
    for model, need, checks_part in ((QwenDense.reduced(2), "3.1", "0.33"), (QwenDense(), "27.9", "2.37")):
        text = TextIO.synthetic(model.geometry.vocab)
        made = Session(model, text, sequences=1, stack="native", device="cuda:0", capacity=90)
        facts = blocks.DeviceFacts((10, 0), "stated device", 148, 180 * GIB, 80 * GIB)
        options = {"reference_precision": "fp64", "check_kernels": True}
        made.plan = blocks.plan(model.provider_table, model.regimes, facts, 1, 1, "native", model=model, options=options, factories=build)
        table = made.plan_table()
        sentence = (
            f"this stack needs about {need} GiB on this GPU, {checks_part} GiB of it for the checks (the reference's bf16 copy of one weight at a time, "
            "where the run keeps none); 80.0 GiB are free"
        )
        assert table.splitlines()[-1].strip() == sentence, table.splitlines()[-1]
        assert (
            made.model.memory_bytes(1, 90, **made._kept()) == model.memory_bytes(1, 90, FIVE, FIVE) and made.weights is None
        ), "from the plan: nothing was loaded"
    plan = blocks.Plan({}, {}, {}, blocks.DeviceFacts((10, 0), "stated device", 148, 180 * GIB, 80 * GIB), 1, 1, "native", None, "a table")
    assert plan.memory(4 * GIB) == "this stack needs about 4.0 GiB on this GPU; 80.0 GiB are free", "no part named where every weight has its copy"
    with pytest.raises(
        NotImplementedError, match=r"^stack native needs about 100.0 GiB on this GPU, 2.37 GiB of it for the checks \(.*\); 80.0 GiB are free: free memory"
    ):
        plan.memory(100 * GIB, checks_bytes=2 * 248320 * 5120)


def test_the_projection_row_on_other_classes_and_row_counts():
    """The row of the FP8 projection entry is the target's on compute capability 10.0 and up to 32 rows: there the target
    launches this kernel for the in-projections.  On every other class, and above 32 rows, the block stays on its multiply
    over a copy and the plan prints the fallback reason; the entry is not selected onto a class or a row count its row does
    not claim; and a row that names another kernel than the census is ``not the target`` by its form."""
    from dataclasses import replace

    from model_samples import census
    from model_samples.blocks import DeviceFacts, plan
    from model_samples.qwen_dense import QwenDense, configs

    target = QwenDense.target()
    for cc, sms in (((10, 3), 152), ((10, 7), 212), ((12, 0), 170)):
        planned, model, _ = _target_plan(2, cc=cc, sms=sms)
        assert planned.rows["projection.fp8.in"].key == "cublas.bf16_dequantized", cc
        line = next(line for line in planned.table().splitlines() if line.split()[0] == "projection.fp8.in")
        assert line.endswith("fallback: bf16 copy of 8-bit weights") and "torch.matmul over a bf16 copy" in line, line
        assert not target.claim(planned, model, installed=RECORDED).known, "no recorded section of the class: no identity is stated"
        with pytest.raises(NotImplementedError, match=rf"^cudnn.Fp8ProjectionDecode serves compute capability 10.0; this device is {cc[0]}.{cc[1]}$"):
            _target_plan(2, cc=cc, sms=sms, overrides={"projection.fp8.in": "fe.fp8_projection_decode"})
    for sequences, key in ((32, "fe.fp8_projection_decode"), (33, "cublas.bf16_dequantized")):
        planned, model, _ = _target_plan(2, sequences=sequences)
        assert planned.rows["projection.fp8.in"].key == key, sequences
    with pytest.raises(
        NotImplementedError,
        match=r"^cudnn.Fp8ProjectionDecode serves rows 1-32, weights 16384 x 5120 and 14336 x 5120; this run has 33 sequences x 1 tokens per sequence$",
    ):
        _target_plan(2, sequences=33, overrides={"projection.fp8.in": "fe.fp8_projection_decode"})
    # on the class and at 1 row: the target; selected back onto the copy with --provider: not the target set, and said
    planned, model, _ = _target_plan(2)
    claim = target.claim(planned, model, installed=RECORDED)
    assert claim.of("projection.fp8.in") == "target" and claim.numbers()["target_port"] == 10
    selected, model, _ = _target_plan(2, overrides={"projection.fp8.in": "cublas.bf16_dequantized"})
    claim = target.claim(selected, model, installed=RECORDED)
    assert (
        claim.reasons[0] == "not the target set: projection.fp8.in selected by --provider"
        and claim.of("projection.fp8.in") == "not yet (bf16 copy of 8-bit weights)"
    )
    assert claim.numbers()["target_port"] == 8 and claim.compared()[0] == 10
    # a row that states another kernel than the census names (a ring of 7 stages): the same entry, not the target, by its form
    table = dict(configs.PROVIDERS)
    row = table["projection.fp8.in"][0]
    other = replace(row, form=tuple(("STAGES", 7) if name == "STAGES" else (name, value) for name, value in row.form))
    table["projection.fp8.in"] = [other, *table["projection.fp8.in"][1:]]
    model = QwenDense.reduced(2)
    facts = DeviceFacts((10, 0), "stated device", 148, 180 * GIB, 80 * GIB)
    planned = plan(table, model.regimes, facts, 1, 1, "native", model=model, options={"device": "cpu"})
    claim = target.claim(planned, model, installed=RECORDED)
    form = PROJECTION_FORM + ", OUT_FEATURES 16384, IN_FEATURES 5120]"
    assert claim.identities[2] == census.Identity(
        census.OTHER, "fp8_projection", "port", f"the row's form at linear_attention.qkvz is {form.replace('STAGES 8', 'STAGES 7')}, the target's {form}"
    )
    assert claim.numbers()["not_the_target"] == {"projection.fp8.in": 2} and claim.numbers()["target_port"] == 8 and len(claim.recorded) == 10
    sites = census.expected(planned, model)
    assert (
        census.check_forms(sites) == [] and sites[2].facts.form["STAGES"] == 7
    ), "the row and its entry agree: the form is the row's, and it is not the census's"


def test_a_session_hands_its_merge_rule_to_the_plan(monkeypatch):
    """A weight merged at one scale per row is not the FP8 projection entry's operand, and the family README says the plan
    declines the row by name.  The adapter can only decline what its session tells it: the session's merge rule is an
    option of its plan, so the decline comes from ``plan_table()``, before anything is loaded -- not as an error of the
    warm-up step behind the load, in a sentence about another plan."""
    from model_samples import Session, TextIO, blocks
    from model_samples.qwen_dense import QwenDense

    model = QwenDense.reduced(2)
    text = TextIO.synthetic(model.geometry.vocab)
    stated = blocks.DeviceFacts((10, 0), "stated device", 148, 0, 0)
    resolve, options = blocks.plan, []

    def on_stated_facts(table, regimes, device, *rest, **named):  # a session on the CPU states no GPU class: the same call on a cc 10.0 part
        options.append(dict(named["options"]))
        return resolve(table, regimes, stated, *rest, **named)

    monkeypatch.setattr(blocks, "plan", on_stated_facts)
    monkeypatch.setattr(model, "load", lambda *args, **kwargs: pytest.fail("a plan that declines loads nothing"))
    session = Session(model, text, sequences=1, stack="native", device="cpu", capacity=16, merge_rule="own_scale")
    with pytest.raises(
        NotImplementedError,
        match=r"^cudnn.Fp8ProjectionDecode takes one scale per weight; the merge rule 'own_scale' keeps one per weight row: load with the merge rule "
        r'"largest_scale", or select another declared row of the block with --provider$',
    ):
        session.plan_table()
    assert options[-1]["merge_rule"] == "own_scale" and session.plan is None and session.weights is None, "declined in the plan, with the session's rule"
    # the same session with the row selected away plans (a multiply over a copy applies a scale per row), and so does the default rule
    selected = Session(
        model, text, sequences=1, stack="native", device="cpu", capacity=16, merge_rule="own_scale", overrides={"projection.fp8.in": "cublas.bf16_dequantized"}
    )
    assert selected.resolve().rows["projection.fp8.in"].key == "cublas.bf16_dequantized" and options[-1]["merge_rule"] == "own_scale"
    default = Session(model, text, sequences=1, stack="native", device="cpu", capacity=16)
    assert default.resolve().rows["projection.fp8.in"].key == "fe.fp8_projection_decode" and options[-1]["merge_rule"] == "largest_scale"
    assert Session(model, text, sequences=1, stack="fe-cublas", device="cpu", capacity=16, merge_rule="own_scale").resolve().rows["projection.fp8.in"].key == (
        "cublas.bf16_dequantized"
    ), "a stack without the row takes either rule"


def test_the_projection_row_through_a_session_and_what_fails_it(monkeypatch):
    """The row of the FP8 projection entry in the ``native`` stack through a real session on the CPU, on the stand-in of its
    entry (which multiplies on the stored codes the loader kept).  Clean, the run holds: two calls of a step, each served by
    the instance of its weight shape, inside the band of the reference multiply, a sequence alone as among two.  Then the
    negative controls, each of which must fail the kernel check with its own sentence -- a call served by the instance of
    the other shape, the kernels of a multiply over a copy under the row's name, a launch more in front of its kernel, a
    launch more behind it where its consumer is judged one by one, a launch less, a launch in another shape than the
    census recorded, the two launches of a step in each other's grid -- or fail the block in its band while the kernel
    check holds: a wrong element.  And what the check does not see, pinned as that: a launch more behind the kernel in an
    attention layer is counted as the fallback row's that follows."""
    from model_samples import census, tie_out
    from model_samples.qwen_dense import configs

    row = configs.PROVIDERS["projection.fp8.in"][0]
    full = {site: PROJECTION_FORM + f", OUT_FEATURES {dict(fields)['OUT_FEATURES']}, IN_FEATURES 5120]" for site, fields in row.sites}

    def run(sequences=2, **stand_in):
        session = _recorded_run(monkeypatch, sequences=sequences, stack="native", **stand_in)[0]
        session.trace(4)
        return session, tie_out.run_checks(session, prompt_tokens=3)

    def sentences(checks):
        return [line.strip() for line in checks.lines if line.strip().startswith("kernels: ")]

    def multiply_line(checks):
        return next(line for line in checks.lines if line.strip().startswith("projection.fp8.in ") and "calls" in line)

    every = "kernels: step 0 (and 3 more of the 4 traced steps): "
    session, checks = run()
    assert checks.ok and checks.kernels == census.PASS, checks.text()
    assert "12 on exact rows, in the wiring order" in _line(checks, "kernels ") and multiply_line(checks).endswith("[band: bf16_rounding]  PASS")
    assert "projection.fp8.in equal (gated)" in _line(checks, "batch invariance"), "each sequence alone writes the bytes it writes among two"
    log = session.traced.call_log[0]
    assert [(call.layer, call.instance, call.form) for call in log if call.block == "projection.fp8.in"] == [
        (0, "cudnn.Fp8ProjectionDecode #1", row.site_form("linear_attention.qkvz")),
        (1, "cudnn.Fp8ProjectionDecode #2", row.site_form("full_attention.qkv")),
    ], "the instance of the call's own weight shape, call by call"
    counted = _line(checks, "projection.fp8.in, as the call log of a traced step has it (2 calls): ")
    assert counted.count(" x cudnn.Fp8ProjectionDecode #") == 2

    # (1) the call of the linear-attention layer served by the instance of the other weight shape
    def neighbour(site):
        return "cudnn.Fp8ProjectionDecode #2", tuple(row.site_form("full_attention.qkv"))

    _, checks = run(projection_serves=neighbour)
    assert not checks.ok and checks.kernels == census.FAIL, checks.text()
    assert sentences(checks) == [
        "kernels: form: projection.fp8.in (layer 0), the call at linear_attention.qkvz: the adapter served the call with the form "
        f"{full['full_attention.qkv']} (cudnn.Fp8ProjectionDecode #2); the row says {full['linear_attention.qkvz']}"
    ]
    for other in ("replay", "blocks", "tokens", "batch invariance"):
        assert _line(checks, other).endswith("PASS"), checks.text()

    # (2) a fallback under the row's name: the block launches the kernels of a multiply over a copy, the row declares the entry's kernel
    _, checks = run(projection_defect="chain")
    missing = "did not launch fp8_projection: the step has 0 launch(es) of fp8_projection, the plan 2 = 2 call(s) x 1 per call"
    assert not checks.ok and checks.kernels == census.FAIL and sentences(checks) == [f"{every}projection.fp8.in (layer {layer}) {missing}" for layer in (0, 1)]
    assert multiply_line(checks).endswith("PASS"), "its bytes are the kernel's: only the kernel check sees it"

    # (3) one launch more in front of its kernel: behind the norm's kernel, which is judged one by one -- in both layers
    _, checks = run(projection_defect="extra")
    nothing = "a kernel of the fallback family pytorch, which no row there declares; nothing may sit between two exact rows"
    launched = f"launched {ELEMENTWISE[:93] + '...'!r} between"
    assert (
        not checks.ok
        and checks.kernels == census.FAIL
        and sentences(checks) == [f"{every}{launched} add_norm_quantize (layer {layer}) and projection.fp8.in (layer {layer}): {nothing}" for layer in (0, 1)]
    ), checks.text()

    # (4) one launch more behind its kernel: in a linear-attention layer its consumer is the fused step, judged one by one; in an
    # attention layer the consumer is a reference chain, and the launch is counted as that row's -- what the check does not see
    _, checks = run(projection_defect="behind")
    assert (
        not checks.ok
        and checks.kernels == census.FAIL
        and sentences(checks) == [f"{every}{launched} projection.fp8.in (layer 0) and linear_attention.step (layer 0): {nothing}"]
    ), checks.text()

    # (5) one launch less: the in-projection of the attention layer does not launch
    _, checks = run(projection_defect="missing")
    assert (
        not checks.ok
        and checks.kernels == census.FAIL
        and sentences(checks)
        == [
            f"{every}projection.fp8.in (layer 1) did not launch fp8_projection: the step has 1 launch(es) of fp8_projection, the plan 2 = 2 call(s) x 1 per call"
        ]
    ), checks.text()

    # (6) at one sequence the census holds the run: a launch in another shape than the recorded one, although the entry states it too
    _, checks = run(sequences=1, projection_shape=((128, 1, 1), (256, 1, 1), 147712))
    assert not checks.ok and checks.kernels == census.FAIL
    assert sentences(checks) == [
        f"{every}projection.fp8.in (layer {layer}) launched fp8_projection with grid (128, 1, 1), block (256, 1, 1), shared memory 147,712 B; "
        f"the {CENSUS} has grid ({grid}, 1, 1), block (256, 1, 1), shared memory 139,520 B"
        for layer, grid in ((0, 128), (1, 112))
    ], checks.text()

    # (6a) the two launches of a step in each other's grid, under their own symbols: the census and the entry both see it
    def swap_grids(traced):
        for step in traced.steps:
            first, second = (index for index, record in enumerate(step) if census.canonical(record) == "fp8_projection")
            step[first], step[second] = step[first]._replace(grid=step[second].grid), step[second]._replace(grid=step[first].grid)

    session = _recorded_run(monkeypatch, sequences=1, stack="native")[0]
    session.trace(4)
    swap_grids(session.traced)
    checks = tie_out.run_checks(session, prompt_tokens=3)
    assert not checks.ok and checks.kernels == census.FAIL
    assert sentences(checks) == [
        f"{every}projection.fp8.in (layer {layer}) launched fp8_projection with grid ({found}, 1, 1), block (256, 1, 1), shared memory 139,520 B; "
        f"{who} grid ({grid}, 1, 1), block (256, 1, 1), shared memory 139,520 B"
        for layer, found, grid in ((0, 112, 128), (1, 128, 112))
        for who in (f"the {CENSUS} has", "cudnn.Fp8ProjectionDecode states")
    ], checks.text()
    assert not any(COMPARED_12 in line for line in checks.lines), "nothing is said to have been compared"

    # (7) a wrong element: the block fails in its band, the kernel check holds
    _, checks = run(projection_defect="element")
    line = multiply_line(checks)
    assert not checks.ok and checks.kernels == census.PASS and line.endswith("[band: bf16_rounding]  FAIL"), line
    assert _line(checks, "blocks").endswith("FAIL") and "7 of 8 provider blocks inside their bands" in _line(checks, "blocks")


# ----------------------------------------------------------------------------------------------- the rows of the fused gate / up entry
def test_the_gate_up_entrys_adapter_over_a_stand_in_entry(monkeypatch):
    """The adapter of ``cudnn.SwigluFp4Decode`` hands the entry what its row says and nothing else: one instance per row, built
    with the row's form as ONE config (token tile, weight stages ahead of the wait, L2 policy, cluster cap) and the launch
    knob, so the entry never takes a default of its own; per call the stored weight in the kernel's layout as it is, the
    4-bit operand as the norm's kernel wrote it, the fp32 product of the two scales and the fp32 reciprocal of the
    consumer's scale as one element each, built once per value, and output buffers of the call's own -- and it returns the
    tensors its entry wrote, in the kernel's layout.  An operand whose block scales come row by row is brought into the
    layout by this block.  The layout of the weight is made by the entry's PUBLIC helper, and a copy made from it is the
    copy of the merged weight.  A table, a weight or an operand it cannot serve is named.  And each row's form is the
    PUBLIC entry's own at every row count the row claims."""
    import cudnn
    from cudnn.gemm import SwigluGeometry
    from cudnn.gemm.frost.swiglu_decode import Cfg
    from dataclasses import replace

    from model_samples.blocks import DeviceFacts
    from model_samples.providers import BlockBase, Quant, fe
    from model_samples.qwen_dense import QwenDense, configs
    from model_samples.weights import (
        BlockScaledTensor,
        Dequantized,
        InterleavedBlockScaledTensor,
        ScaledTensor,
        TiledBlockScaledTensor,
        dequantize,
        quantize_nvfp4,
        stored_checksum,
        tile_scale_index,
        tile_scales,
        untile_scales,
    )

    calls = []
    monkeypatch.setattr(cudnn, "SwigluFp4Decode", _swiglu_entry(calls), raising=False)
    monkeypatch.setattr(fe, "_ENTRIES", {})
    rows = [row for row in configs.PROVIDERS["mlp.gate_up"] if row.key == "fe.swiglu_fp4_decode"]
    assert [(row.sequences, row.rows, dict(row.form)["TOKEN_TILE"]) for row in rows] == [
        ((1, last), (first, last), tile) for first, last, tile in configs.GATE_UP_TILES
    ]
    assert configs.GATE_UP_TILES == ((1, 8, 8), (9, 16, 16), (17, 32, 32), (33, 64, 64)), "the narrowest of the four token tiles that holds the rows"
    row = rows[0]
    model = QwenDense.reduced(2)
    facts = DeviceFacts((10, 0), "stated device", 148, 0, 0)
    options = {"device": "cpu", "check_kernels": True}
    block = fe.SwigluFp4DecodeBlock(row, model, facts, options)
    assert block.stored_weights is True and callable(block.stored_layout) and BlockBase.stored_layout is None, "the one adapter with a layout of its own"
    assert block.entry == "cudnn.SwigluFp4Decode" and [(k.id, k.relation, k.launches, k.shape) for k in block.kernels] == [("swiglu_fp4", "exact", 1, "entry")]
    block.check_support(1, 1)
    geometry = SwigluGeometry(5120, 17408)
    entry = block._build("cpu")
    assert entry is block._build("cpu") and entry.built == dict(
        device="cpu",
        geometry=geometry,
        cfg=Cfg(TOKEN_TILE=8, PREWAIT_STAGES=2, L2_EVICT_FIRST=True, MAX_CLUSTERS=148, GEOM=geometry),
        pdl=True,
    ), "the kernel as the row states it: the config is passed as a whole, so the entry never takes a default of its own"
    others = [fe.SwigluFp4DecodeBlock(other, model, facts, options)._build("cpu") for other in rows[1:]]
    assert [built.built["cfg"].TOKEN_TILE for built in others] == [16, 32, 64] and len({id(built) for built in (entry, *others)}) == 4, "one instance per row"
    stated = block.launch_facts(1)
    assert stated.shape == SWIGLU_SHAPE and stated.symbols == (SWIGLU_SYMBOL,) and all(stated.form[name] == value for name, value in row.form)
    assert block.serving() == ("cudnn.SwigluFp4Decode", ()), "before the first call nothing has served"

    # the weight in the kernel's layout, made by the entry's PUBLIC helper: tiles of 64 gate rows and the 64 up rows of the same
    # channels, the block scales of those rows in the 128 x 4 layout -- a permutation of bytes, and a copy made from it is the
    # copy of the merged weight, in the checkpoint's row order
    torch.manual_seed(3)
    merged = quantize_nvfp4(torch.randn(256, 64), 0.013)  # two tiles: gate channels 0 .. 127, then up channels 0 .. 127; 4 blocks a row
    laid = block.stored_layout(merged)
    assert type(laid) is InterleavedBlockScaledTensor and (laid.tile, laid.scale) == (64, 0.013) and laid.shape == merged.shape == (256, 64)
    assert laid.codes.dtype == laid.tiled_scales.dtype == torch.uint8 and laid.tiled_scales.shape == merged.block_scales.shape
    natural = torch.cat([torch.arange(0, 64), torch.arange(128, 192), torch.arange(64, 128), torch.arange(192, 256)])  # of every row of the layout
    assert torch.equal(laid.codes, merged.codes[natural]) and not torch.equal(laid.codes, merged.codes)
    assert torch.equal(
        laid.tiled_scales.reshape(-1)[tile_scale_index(256, 4, "cpu")], merged.block_scales[natural]
    ), "the scales of the rows in the 128 x 4 layout"
    assert torch.equal(untile_scales(laid.tiled_scales, 256, 4), merged.block_scales[natural])
    assert sorted(laid.codes.reshape(-1).tolist()) == sorted(merged.codes.reshape(-1).tolist()), "the same bytes: nothing is requantised"
    copy, plain = dequantize(laid), dequantize(merged)
    assert torch.equal(copy.data.view(torch.int16), plain.data.view(torch.int16)) and copy.scale == plain.scale == 0.013, "the copy of the merged weight"
    assert laid.tensors()[0] is laid.codes and laid.tensors()[1] is laid.tiled_scales and stored_checksum(laid) != stored_checksum(merged)
    # ... also where the weight is longer than one chunk of the copy, and at the tile of a test's own layout
    long = quantize_nvfp4(torch.randn(2048 + 256, 64), 0.01)
    assert torch.equal(dequantize(block.stored_layout(long), rows_per_chunk=1024).data.view(torch.int16), dequantize(long).data.view(torch.int16))

    # ... and at every chunk length: a chunk of the copy is whole weight tiles AND whole bands of the scale layout, whatever the
    # tile (at 32 gate rows a weight tile is half a band), and a weight that does not fill its last band is read through the index
    def interleaved(merged, tile):
        n = merged.codes.shape[0]

        def rows(tensor):  # [gate rows, up rows] -> tiles of [gate, up]
            return tensor.reshape(2, n // (2 * tile), tile, -1).permute(1, 0, 2, 3).reshape(n, -1).contiguous()

        return InterleavedBlockScaledTensor(rows(merged.codes), tile_scales(rows(merged.block_scales)), float(merged.scale), tile)

    short = quantize_nvfp4(torch.randn(192, 64), 0.01)  # one band and a half
    for whole, tiles in ((long, (32, 64)), (short, (32,))):
        expected = dequantize(whole).data.view(torch.int16)
        for tile in tiles:
            for chunk in (1, 64, 100, 128, 200, 2048):
                made = dequantize(interleaved(whole, tile), rows_per_chunk=chunk)
                assert torch.equal(made.data.view(torch.int16), expected), (tuple(whole.shape), tile, chunk)

    # a call: the stored weight and the operand as they are, two scalars, outputs of the call's own
    weight = Dequantized(None, laid.scale, laid)
    activation = quantize_nvfp4(torch.randn(2, 64), 0.017857142857142856)
    tiled = torch.zeros((128, 4), dtype=torch.uint8)
    tiled.view(-1)[tile_scale_index(2, 4, "cpu")] = activation.block_scales
    qx = TiledBlockScaledTensor(activation.codes, tiled, 0.017857142857142856)  # as the norm's kernel writes it
    quant = Quant("nvfp4", 0.0625)
    del calls[:]
    out = block(qx, weight, quant)
    again = block(qx, weight, quant)
    assert [call[0] for call in calls] == ["swiglu launch"] * 2, "one launch per call, no build inside a step"
    _, w, ws, a, a_scales, alpha, scale, codes, scales = calls[0]
    assert w is laid.codes and ws is laid.tiled_scales and a is qx.codes and a_scales is qx.tiled_scales, "handed over as they are: no copy, no launch"
    assert (
        type(out) is TiledBlockScaledTensor and out.codes is codes and out.tiled_scales is scales and out.scale == 0.0625
    ), "what the entry wrote, in its layout"
    assert codes.dtype == scales.dtype == torch.uint8 and tuple(codes.shape) == (2, 8704) and tuple(scales.shape) == (128, 1088)
    assert again.codes.data_ptr() != out.codes.data_ptr() and again.tiled_scales.data_ptr() != out.tiled_scales.data_ptr(), "buffers of the call's own"
    product = torch.tensor([0.017857142857142856], dtype=torch.float32) * torch.tensor([0.013], dtype=torch.float32)
    assert alpha.dtype == torch.float32 and alpha.shape == (1,) and torch.equal(alpha, product), "the fp32 product of the two static scales"
    assert not torch.equal(
        product, torch.tensor([0.017857142857142856 * 0.013], dtype=torch.float32)
    ), "which these two values tell from a rounded fp64 product"
    assert scale.dtype == torch.float32 and scale.shape == (1,) and float(scale) == 16.0, "the global scale of the output: the reciprocal of the consumer's"
    assert calls[1][5] is alpha and calls[1][6] is scale, "one element per value, built once: never inside a later step"
    third = Quant("nvfp4", 1.0 / 3.0)
    block(qx, weight, third)
    assert torch.equal(calls[2][6], torch.ones(1, dtype=torch.float32) / torch.tensor([1.0 / 3.0], dtype=torch.float32)), "the fp32 reciprocal, rounded once"
    assert block.serving() == ("cudnn.SwigluFp4Decode #1", row.form), "read from the instance that was called"
    # an operand whose block scales come row by row (a reference chain in front of this block; one row of a batch in a check): the
    # codes as they are, the scales brought into the 128 x 4 layout by this block
    del calls[:]
    block(activation.rows(1, 2), weight, quant)
    _, _, _, a, a_scales, _, _, codes, _ = calls[0]
    assert a.data_ptr() == activation.codes[1:2].data_ptr() and tuple(a_scales.shape) == (128, 4) and tuple(codes.shape) == (1, 8704)
    assert torch.equal(a_scales.view(-1)[tile_scale_index(1, 4, "cpu")], activation.block_scales[1:2]) and int(a_scales.sum()) == int(
        activation.block_scales[1].sum()
    )

    # what it cannot serve is named
    other_plan = "the weights of this session were loaded for another plan"
    with pytest.raises(ValueError, match=rf"the weight of this call has it in the checkpoint's row order: {other_plan}"):
        block(qx, Dequantized(None, merged.scale, merged), quant)
    with pytest.raises(
        ValueError,
        match=rf"^cudnn.SwigluFp4Decode multiplies on the stored 4-bit gate / up weight in its own row order; the weight of this call has no stored 4-bit codes: {other_plan}",
    ):
        block(qx, dequantize(merged), quant)
    for asked, said in ((None, "no quantised operand"), (Quant("fp8", 1.0), "'fp8'")):
        with pytest.raises(ValueError, match=rf"^cudnn.SwigluFp4Decode writes 4-bit codes with their block scales; this block is asked for {said}$"):
            block(qx, weight, asked)
    with pytest.raises(ValueError, match=r"^cudnn.SwigluFp4Decode multiplies packed 4-bit codes with their block scales; this block is handed ScaledTensor$"):
        block(ScaledTensor(activation.codes, 1.0), weight, quant)
    with pytest.raises(ValueError, match=r"^cudnn.SwigluFp4Decode multiplies on a merged 4-bit gate / up weight; the loader handed its layout ScaledTensor$"):
        block.stored_layout(ScaledTensor(merged.codes, 1.0))
    with pytest.raises(ValueError) as declined:
        block.stored_layout(quantize_nvfp4(torch.randn(192, 64), 0.01))
    assert str(declined.value).startswith(
        "the provider table is wrong: cudnn.SwigluFp4Decode cannot lay out the weight of the row of fe.swiglu_fp4_decode: the SwiGLU FP4 decode kernel: "
        "interleave_gate_up: the weight rows must be a multiple of 128"
    )
    for wrong, said in (
        (replace(row, form=(*row.form, ("ROWS", 1))), "names ROWS"),
        (replace(row, form=tuple(pair for pair in row.form if pair[0] != "MAX_CLUSTERS")), "does not name MAX_CLUSTERS"),
        (replace(row, form=tuple(pair for pair in row.form if pair[0] != "PDL")), "does not name PDL"),
    ):
        with pytest.raises(ValueError, match=rf"^provider row fe.swiglu_fp4_decode: its form {said}; this adapter passes TOKEN_TILE, PREWAIT_STAGES, "):
            fe.SwigluFp4DecodeBlock(wrong, model, facts, {"device": "cpu"}).check_support(1, 1)

    # the PUBLIC entry, built for the host from the same arguments: it states every field of a row's form at every row count the
    # row claims, the launch shape the census records up to 32 rows, and no symbol before a build.  The config's token tile is a
    # FLOOR -- outside its row's range an instance runs another tile -- which is why the table has a row per tile
    monkeypatch.undo()
    monkeypatch.setattr(fe, "_ENTRIES", {})
    for at in rows:
        real = fe.SwigluFp4DecodeBlock(at, model, facts, options)
        entry = real._build("cpu")
        assert type(entry) is cudnn.SwigluFp4Decode and entry.cfg.GEOM == geometry
        for count in range(at.rows[0], at.rows[1] + 1):
            form = real._stated(entry, count)
            assert all(form[name] == value for name, value in at.form), (at.rows, count)
            assert tuple(real.launch_facts(count).shape) == (SWIGLU_SHAPE if count <= 32 else ((1, 1, 148), (192, 1, 1), 230400)), count
        assert real.launch_facts(at.rows[0]).symbols == (), "nothing is compiled on the host"
        with pytest.raises(NotImplementedError, match=r"serves at most 64 rows per call"):
            entry.form_for(65)
    first = fe.SwigluFp4DecodeBlock(rows[0], model, facts, options)._build("cpu")
    assert first.form_for(8)["TOKEN_TILE"] == 8 and first.form_for(9)["TOKEN_TILE"] == 16, "the first row's instance past its row's range: another kernel"
    narrow = QwenDense(replace(model.geometry, hidden=4096), ("another hidden width",))
    with pytest.raises(ValueError) as declined:
        fe.SwigluFp4DecodeBlock(row, narrow, facts, {"device": "cpu"})._build("cpu")
    assert str(declined.value) == (
        "the provider table is wrong: cudnn.SwigluFp4Decode declines the geometry or the form of the row of fe.swiglu_fp4_decode: the SwiGLU FP4 decode "
        "kernel serves hidden 5120, intermediate 17408 or hidden 5120, intermediate 128 (validated); got hidden 4096, intermediate 17408"
    )
    odd = replace(row, form=tuple(("PREWAIT_STAGES", 1) if name == "PREWAIT_STAGES" else (name, value) for name, value in row.form))
    with pytest.raises(
        ValueError, match=r"^the provider table is wrong: cudnn.SwigluFp4Decode declines the geometry or the form .*Cfg.PREWAIT_STAGES must be one of"
    ):
        fe.SwigluFp4DecodeBlock(odd, model, facts, {"device": "cpu"})._build("cpu")


def test_the_gate_ups_twin_and_its_band_on_stated_tensors():
    """The fused gate / up step evaluated in fp64 with its error bound, and the band its entry is judged with: the entry's
    documented contract, an interval.  The terms of the bound are those of the entry's own suite.  The kernel's chain in
    fp32 -- both products accumulated in fp32 and rounded once to bf16, SiLU x up rounded once more, the block scale of
    that value, the codes under it -- is inside the band, and so are the bytes of the block's reference pair.  Each rule
    is held on its own: a block scale outside the interval its block's bound admits (above; and below, by buffers that
    were never written and by the byte just under the lower end, each with every code inside its bound), a code farther
    from the fp64 value than half a code step plus the bound, a scale byte that is no number, other outputs than codes
    with their scales: each fails the block.  And what the band cannot see is pinned as that: one bf16 unit of the
    product."""
    from types import SimpleNamespace

    from model_samples import tie_out
    from model_samples.providers import Quant, torch_ref
    from model_samples.weights import BLOCK, BlockScaledTensor, _e2m1, _e4m3_values, dequantize, quantize_nvfp4

    assert (
        torch_ref.SWIGLU_ROUNDING,
        torch_ref.SWIGLU_SUM_UNIT,
        torch_ref.SWIGLU_SILU_LIPSCHITZ,
        torch_ref.SWIGLU_APPROXIMATE,
        torch_ref.SWIGLU_FLOOR,
        torch_ref.SWIGLU_SCALE_SLACK,
    ) == (2.0**-8, 2.0**-23, 1.1, 2.0**-19, 2.0**-120, 2.0**-21), "the terms of the entry's own suite: they move only with it"
    suite = (MODELS.parents[1] / "test" / "python" / "gemm" / "frost" / "swiglu_decode" / "swiglu_fp4_decode_utils.py").read_text()
    for term in (
        "d_g = 2.0**-8 * g.abs() + (1 + 2.0**-8) * (k + 1) * 2.0**-23 * sg",
        "big_a = 1.1 * d_g * (u.abs() + d_u) + silu.abs() * d_u",
        "e_v = big_a + (2.0**-8 + 2.0**-19) * (v.abs() + big_a) + 2.0**-120",
        "lo = e4m3_round(gs * (amax - e_a).clamp(min=0) / 6.0 * (1 - 2.0**-21))",
        "hi = e4m3_round(gs * (amax + e_a) / 6.0 * (1 + 2.0**-21))",
        "bound = h * s + (v.abs() + e_v - 6 * s).clamp(min=0) + e_v",
    ):
        assert term in suite, f"the entry's suite no longer derives its bound as this sample states it: {term}"
    page = " ".join((MODELS.parents[1] / "docs" / "operations" / "gemm_fusions" / "swiglu_fp4_decode.md").read_text().split())
    assert (
        "every block scale lies in the interval of bytes its propagated error admits, and every code is within half a code step of its own block scale "
        "plus that error"
    ) in page, "the contract of the entry's page"

    # the E4M3 rounding of an fp64 value, by table: every code's own value, ties to the even code, saturation
    every = torch.arange(0, 0x7F, dtype=torch.uint8)
    values = _e4m3_values(every, torch.float64)
    assert torch.equal(torch_ref.e4m3_nearest(values), every) and bool((values[1:] > values[:-1]).all()), "127 finite codes, ascending"
    middle = (values[1:] + values[:-1]) / 2
    assert torch.equal(torch_ref.e4m3_nearest(middle), every[:-1] + (every[:-1] % 2)), "a tie goes to the even code"
    assert torch.equal(torch_ref.e4m3_nearest(middle * (1 + 2.0**-40)), every[1:]) and torch.equal(torch_ref.e4m3_nearest(middle * (1 - 2.0**-40)), every[:-1])
    assert torch_ref.e4m3_nearest(torch.tensor([0.0, 448.0, 449.0, 1e30], dtype=torch.float64)).tolist() == [0, 0x7E, 0x7E, 0x7E]

    # operands: 3 rows of a 256-wide activation, a merged weight of 64 gate and 64 up rows, the scales of the family's recipe
    torch.manual_seed(7)
    width, half, rows = 256, 64, 3
    stored = quantize_nvfp4(torch.randn(2 * half, width) * width**-0.5, 6.5 * width**-0.5 / (448 * 6))
    weight = dequantize(stored)
    qx = quantize_nvfp4(torch.randn(rows, width), 8.0 / (448 * 6))
    quant = Quant("nvfp4", 32.0 / (448 * 6))
    twin = torch_ref.GateUpFp4Bound(None, None, None, {})
    assert twin.entry == "torch_ref.linear + torch_ref.silu_mul_quantize at the kernel's rounding points" and twin.bound is None
    reference = twin(qx, weight, quant)
    value, error = twin.bound
    assert isinstance(reference, BlockScaledTensor) and twin.quant is quant and value.dtype == error.dtype == torch.float64 and value.shape == (rows, half)
    # ... the value and its bound, written out once more from the operands
    x, w = dequantize(qx).data.double(), weight.data.double()
    alpha = float(torch.tensor(qx.scale, dtype=torch.float32) * torch.tensor(stored.scale, dtype=torch.float32))
    gate, up = alpha * (x @ w[:half].T), alpha * (x @ w[half:].T)
    spread_gate, spread_up = alpha * (x.abs() @ w[:half].abs().T), alpha * (x.abs() @ w[half:].abs().T)
    silu = gate / (1 + torch.exp(-gate))
    d_gate = 2.0**-8 * gate.abs() + (1 + 2.0**-8) * (width + 1) * 2.0**-23 * spread_gate
    d_up = 2.0**-8 * up.abs() + (1 + 2.0**-8) * (width + 1) * 2.0**-23 * spread_up
    a = 1.1 * d_gate * (up.abs() + d_up) + silu.abs() * d_up
    assert torch.allclose(value, silu * up, rtol=1e-12, atol=0.0) and torch.allclose(
        error, a + (2.0**-8 + 2.0**-19) * ((silu * up).abs() + a) + 2.0**-120, rtol=1e-12
    )
    assert bool((error > 0).all()) and float((error / value.abs()).median()) < 0.1, "a bound of a few percent of a typical value"
    with pytest.raises(ValueError, match="the fused gate / up kernel's twin quantises to 4-bit codes"):
        twin(qx, weight, Quant("fp8", 1.0))

    def judged(out, of=twin, ref=None):
        stats = tie_out.BlockStats("swiglu_fp4")
        tie_out.BlockTap._judge(stats, tuple(out), tuple((reference if ref is None else ref).tensors()), [], [], 0, of)
        return stats

    def kernel_chain(move=0.0):
        """The kernel's chain in fp32: ``(packed codes, block scales row by row)``.  ``move``: bf16 units added to every
        element of the product before it is quantised."""
        scale = torch.tensor(qx.scale, dtype=torch.float32) * torch.tensor(stored.scale, dtype=torch.float32)
        g = ((x.float() @ w[:half].float().T) * scale).to(torch.bfloat16)
        u = ((x.float() @ w[half:].float().T) * scale).to(torch.bfloat16)
        v = (torch.nn.functional.silu(g.float()) * u.float()).to(torch.bfloat16)
        if move:
            v = (v.view(torch.int16) + int(move)).view(torch.bfloat16)  # the next bf16 value away from zero
        scales = torch_ref.nvfp4_block_scales(v.float().abs().unflatten(-1, (-1, BLOCK)).amax(-1), quant.scale)
        step = _e4m3_values(scales).repeat_interleave(BLOCK, dim=-1) * torch.tensor(float(quant.scale), dtype=torch.float32)
        codes = _e2m1(v.float() / step, signed_zero=True)
        return codes[..., 0::2] | (codes[..., 1::2] << 4), scales

    packed, scales = kernel_chain()
    stats = judged((packed, scales))
    assert stats.ok and stats.finite and (stats.blocks, stats.total_bytes, stats.outside) == (rows * half // BLOCK, rows * half, 0)
    assert 0.0 < stats.worst_share <= 1.0 and stats.blocks_equal == stats.blocks and stats.equal_bytes >= 0.95 * stats.total_bytes
    line = stats.line()
    assert line.startswith("12 block scales, 0 of them another byte than the fp64 chain's; 192 codes, ") and line.endswith(
        "% of its bound; 0 outside their bounds"
    )
    assert judged(reference.tensors()).ok, "the fp64 chain at the kernel's rounding points"
    pair = torch_ref.silu_mul_quantize(torch_ref.linear(qx, weight, torch.float64), quant)
    assert judged(pair.tensors()).ok, "and the bytes of the block's reference pair, which rounds the SiLU once more"
    # what the band cannot see, pinned as that: the product one bf16 unit off on EVERY element is inside the bound of its two
    # roundings (the entry's own suite holds that with operands whose every intermediate is exact)
    assert judged(kernel_chain(move=1)).ok, "one bf16 unit of the product hides behind the bound"

    # each rule on its own
    for change, why in ((8, "one binade up"), (-8, "one binade down")):
        wrong = scales.clone()
        wrong[1, 2] = int(wrong[1, 2]) + change
        stats = judged((packed, wrong))
        assert not stats.ok and stats.outside >= 1 and not stats.line().endswith("; 0 outside their bounds"), why
    block = scales.clone()
    block[0, 0] ^= 1  # the neighbouring byte: 6 % of the scale, where the bound admits a few tenths of a percent of the block's largest value
    assert not judged((packed, block)).ok, "a block scale of another largest magnitude than the bound admits"
    # ... the LOWER end of the interval is a rule of its own, and the one rule of the band that fails an output nobody wrote.
    # Under a zero block scale every code reads as zero, and what a code's bound admits below a scale that is too small is
    # the whole value (the clip term): buffers of zeros are inside the code rule on every element, and outside by the lower
    # end alone, in every block
    stats = judged((torch.zeros_like(packed), torch.zeros_like(scales)))
    assert not stats.ok and stats.finite and stats.outside == stats.blocks == rows * half // BLOCK, "an output that was never written"
    assert 0.0 < stats.worst_share < 1.0 and stats.blocks_equal == 0 and stats.line().endswith("; 12 outside their bounds"), "no code is outside its bound"
    # ... and the place of each end is the byte the bound admits, written out once more from the value and its bound: the
    # byte below the lower end fails and so does the byte above the upper one, each with the codes of its block those of
    # the fp64 value under THAT scale (below, the largest of them clipped), so that every code is inside its bound and the
    # scale rule alone decides
    global_scale = torch_ref.nvfp4_global_scale(quant.scale, "cpu").double()
    largest, spread = (t.unflatten(-1, (-1, BLOCK)).amax(-1) for t in (value.abs(), error))
    low = torch_ref.e4m3_nearest(global_scale * (largest - spread).clamp(min=0) / 6.0 * (1 - 2.0**-21))
    high = torch_ref.e4m3_nearest(global_scale * (largest + spread) / 6.0 * (1 + 2.0**-21))
    assert bool(((scales >= low) & (scales <= high)).all()) and bool((low > 0).all()), "the fp32 chain's bytes are inside; every interval has a byte below it"
    for end, beyond, why in ((low, -1, "one byte below the lower end"), (high, 1, "one byte above the upper end")):
        at_end = (scales == end).nonzero()
        assert len(at_end), "the operands have a block whose byte is this end of its interval: that byte is admitted (the chain above passed)"
        row, col = (int(i) for i in at_end[0])
        moved = scales.clone()
        moved[row, col] = int(end[row, col]) + beyond
        step_there = _e4m3_values(moved, torch.float64)[row, col] / global_scale
        recoded = torch.stack((packed & 0xF, packed >> 4), dim=-1).flatten(-2)
        span = slice(col * BLOCK, (col + 1) * BLOCK)
        recoded[row, span] = _e2m1((value[row, span] / step_there).float(), signed_zero=True)
        stats = judged((recoded[..., 0::2] | (recoded[..., 1::2] << 4), moved))
        assert not stats.ok and stats.outside == 1 and stats.worst_share <= 1.0, f"{why}: the scale rule alone, every code inside its bound"
    codes = torch.stack((packed & 0xF, packed >> 4), dim=-1).flatten(-2)
    at = int((codes[0, :BLOCK] & 0x7).argmax())  # the largest code of the first block
    flipped = codes.clone()
    flipped[0, at] ^= 0x8  # the other sign
    stats = judged((flipped[..., 0::2] | (flipped[..., 1::2] << 4), scales))
    assert not stats.ok and stats.outside == 1 and stats.worst_share > 1.0 and stats.line().endswith("; 1 outside their bounds")
    stepped = codes.clone()
    stepped[0, at] = (stepped[0, at] & 0x8) | ((stepped[0, at] & 0x7) - 2)  # two codes toward zero: a whole code step and more
    assert not judged((stepped[..., 0::2] | (stepped[..., 1::2] << 4), scales)).ok, "a code more than half a step and the bound from the fp64 value"
    # ... and HALF a code step is the rule: the neighbouring code on the value's side is nearer than a whole step, and outside
    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float64)
    unit = (_e4m3_values(scales, torch.float64) / torch_ref.nvfp4_global_scale(quant.scale, "cpu").double()).repeat_interleave(BLOCK, dim=-1)
    magnitude = (codes & 0x7).long()
    toward = torch.where(value.abs() > grid[magnitude] * unit, magnitude + 1, magnitude - 1)  # the neighbour on the value's side
    fine = (magnitude >= 1) & (magnitude <= 2) & (((codes & 0x8) != 0) == (value < 0))  # both codes below 2: half a step is 0.25
    away = (grid[toward.clamp(0, 7)] * unit - value.abs()).abs()
    between = fine & (away > 0.25 * unit + error) & (away <= 0.5 * unit + error)
    assert bool(between.any()), "the operands have an element whose neighbouring code is between half a step and a whole step away"
    row, column = (int(i) for i in between.nonzero()[0])
    near = codes.clone()
    near[row, column] = (near[row, column] & 0x8) | int(toward[row, column])
    stats = judged((near[..., 0::2] | (near[..., 1::2] << 4), scales))
    assert not stats.ok and stats.outside == 1 and 1.0 < stats.worst_share < 2.5, "half a code step, not a whole one"
    not_a_number = scales.clone()
    not_a_number[2, 1] = 0x7F
    stats = judged((packed, not_a_number))
    assert not stats.ok and not stats.finite and stats.line().endswith("NON-FINITE VALUES")
    stats = judged((packed,), ref=SimpleNamespace(tensors=lambda: (reference.codes,)))
    assert not stats.ok and stats.outside == 1, "a call that wrote other outputs than codes with their block scales: nothing is compared"
    with pytest.raises(ValueError, match="the band swiglu_fp4 is judged against the gate / up step with its error bound: the block's twin keeps no bound"):
        judged((packed, scales), of=SimpleNamespace())
    assert "every 4-bit code within half a code step of the fp64 value under the kernel's own block scale plus that bound" in tie_out.BANDS["swiglu_fp4"]
    # a zero row: zero block scales and zero codes, the kernel's rule for a block whose scale byte is zero
    zero = quantize_nvfp4(torch.zeros(1, width), 8.0 / (448 * 6))
    twin(zero, weight, quant)
    assert judged((torch.zeros((1, half // 2), dtype=torch.uint8), torch.zeros((1, half // BLOCK), dtype=torch.uint8)), ref=twin(zero, weight, quant)).ok
    # ... which isolates the rule of the block scales: under zero codes every scale gives the value zero, so a scale byte that is
    # not the byte of a zero block is outside its interval while every code is inside its bound
    lone = torch.zeros((1, half // BLOCK), dtype=torch.uint8)
    lone[0, 1] = 0x38  # the byte of 1.0
    stats = judged((torch.zeros((1, half // 2), dtype=torch.uint8), lone), ref=twin(zero, weight, quant))
    assert not stats.ok and stats.outside == 1 and stats.worst_share == 0.0 and stats.blocks_equal == stats.blocks - 1, "the scale rule alone"
    assert stats.line().endswith("; 1 outside their bounds")


def test_the_gate_up_row_through_a_session_and_what_fails_it(monkeypatch):
    """The rows of the fused gate / up entry in the ``native`` stack through a real session on the CPU, on the stand-in of the
    entry (which multiplies on the weight the loader kept in the kernel's layout).  Clean, the run holds: one launch per
    MLP from one instance, inside the band of the entry's twin, a sequence alone as among two, the consumer on its fallback
    row reading the kernel's bytes as they are.  Then the negative controls, each of which must fail the kernel check with
    its own sentence -- a call served in another form, the kernels of a multiply over a copy under the row's name, a launch
    more in front of its kernel (behind the norm's, which is judged one by one), a launch less, a launch in another shape
    than the census recorded, a launch in another shape than the entry states -- or fail the block in its band while the
    kernel check holds: a wrong code, a wrong block scale.  And what the check does not see, pinned as that: a launch more
    behind the kernel is counted as the fallback row's that follows."""
    from model_samples import census, tie_out
    from model_samples.qwen_dense import configs
    from model_samples.weights import TiledBlockScaledTensor

    row = configs.PROVIDERS["mlp.gate_up"][0]

    def run(sequences=2, **stand_in):
        session = _recorded_run(monkeypatch, sequences=sequences, stack="native", **stand_in)[0]
        session.trace(4)
        return session, tie_out.run_checks(session, prompt_tokens=3)

    def sentences(checks):
        return [line.strip() for line in checks.lines if line.strip().startswith("kernels: ")]

    def fused_line(checks):
        return next(line for line in checks.lines if line.strip().startswith("mlp.gate_up ") and "calls" in line)

    every = "kernels: step 0 (and 3 more of the 4 traced steps): "
    session, checks = run()
    assert checks.ok and checks.kernels == census.PASS, checks.text()
    assert "12 on exact rows, in the wiring order" in _line(checks, "kernels ") and fused_line(checks).endswith("[band: swiglu_fp4]  PASS")
    assert "mlp.gate_up equal (gated)" in _line(checks, "batch invariance"), "each sequence alone writes the bytes it writes among two"
    log = session.traced.call_log[0]
    assert [(call.layer, call.instance, call.form) for call in log if call.block == "mlp.gate_up"] == [
        (0, "cudnn.SwigluFp4Decode #1", row.form),
        (1, "cudnn.SwigluFp4Decode #1", row.form),
    ], "one instance serves the MLP of every layer"
    stated = _line(checks, "mlp.gate_up: cudnn.SwigluFp4Decode ")
    assert f"{SWIGLU_FORM}, 2 launches per step (2 calls x 1 per call); the entry states grid (1, 1, 148), block (192, 1, 1), shared memory 231,424 B" in stated
    # the consumer on its fallback row is handed the kernel's bytes in the kernel's layout, and gathers when it reads them
    handed = []
    session.tap = lambda name, layer, block, operands: (handed.append(type(operands[0])) if name == "mlp.down" else None, block(*operands))[1]
    session.model.step(session._call, session.weights, session.state, torch.zeros(2, dtype=torch.int32))
    session.tap = None
    assert handed == [TiledBlockScaledTensor] * 2, "what a consumer needs beyond the kernel's bytes is that consumer's to launch"

    # (1) the calls served by an instance of another form: the token tile of the next row -- the same kernel name and launch shape
    other = tuple(("TOKEN_TILE", 16) if name == "TOKEN_TILE" else (name, value) for name, value in row.form)
    _, checks = run(swiglu_serves=("cudnn.SwigluFp4Decode #2", other))
    assert not checks.ok and checks.kernels == census.FAIL, checks.text()
    assert sentences(checks) == [
        f"kernels: form: mlp.gate_up (layer 0): the adapter served the call with the form {SWIGLU_FORM.replace('TOKEN_TILE 8', 'TOKEN_TILE 16')} "
        f"(cudnn.SwigluFp4Decode #2); the row says {SWIGLU_FORM}"
    ], "one sentence for the kind of difference, whichever layer"
    for held in ("replay", "blocks", "tokens", "batch invariance"):
        assert _line(checks, held).endswith("PASS"), checks.text()

    # (2) a fallback under the row's name: the block launches the kernels of a multiply over a copy, the row declares the entry's kernel
    _, checks = run(swiglu_defect="chain")
    missing = "did not launch swiglu_fp4: the step has 0 launch(es) of swiglu_fp4, the plan 2 = 2 call(s) x 1 per call"
    assert not checks.ok and checks.kernels == census.FAIL and sentences(checks) == [f"{every}mlp.gate_up (layer {layer}) {missing}" for layer in (0, 1)]
    assert fused_line(checks).endswith("PASS"), "its bytes are the kernel's: only the kernel check sees it"

    # (3) one launch more in front of its kernel: behind the kernel of the norm in front of the MLP, which is judged one by one
    _, checks = run(swiglu_defect="extra")
    nothing = "a kernel of the fallback family pytorch, which no row there declares; nothing may sit between two exact rows"
    launched = f"launched {ELEMENTWISE[:93] + '...'!r} between"
    assert (
        not checks.ok
        and checks.kernels == census.FAIL
        and sentences(checks) == [f"{every}{launched} add_norm_quantize (layer {layer}) and mlp.gate_up (layer {layer}): {nothing}" for layer in (0, 1)]
    ), checks.text()

    # (4) one launch more behind its kernel: its consumer is a multiply over a copy, and the launch is counted as that row's -- what
    # the check does not see while the consumer is a fallback row
    clean, _ = run()
    session, checks = run(swiglu_defect="behind")
    assert checks.ok and checks.kernels == census.PASS, checks.text()
    assert session.traced.result.families["pytorch"] == [count + 2 for count in clean.traced.result.families["pytorch"]], "counted as the fallback row's"

    # (5) one launch less: the gate / up multiply of the second layer does not launch
    _, checks = run(swiglu_defect="missing")
    assert (
        not checks.ok
        and checks.kernels == census.FAIL
        and sentences(checks)
        == [f"{every}mlp.gate_up (layer 1) did not launch swiglu_fp4: the step has 1 launch(es) of swiglu_fp4, the plan 2 = 2 call(s) x 1 per call"]
    ), checks.text()

    # (6) at one sequence the census holds the run: a launch on another thread-block count than the recorded one (the cap of a part
    # with more SMs), although the entry states that shape too
    _, checks = run(sequences=1, swiglu_shape=((1, 1, 152), (192, 1, 1), 231424))
    assert not checks.ok and checks.kernels == census.FAIL
    assert sentences(checks) == [
        f"{every}mlp.gate_up (layer {layer}) launched swiglu_fp4 with grid (1, 1, 152), block (192, 1, 1), shared memory 231,424 B; "
        f"the {CENSUS} has grid (1, 1, 148), block (192, 1, 1), shared memory 231,424 B"
        for layer in (0, 1)
    ], checks.text()
    _, checks = run(sequences=1)
    assert checks.ok and any(f"{CENSUS}: NOT CLAIMED (" in line and COMPARED_12 in line for line in checks.lines), "held to the recorded shape, and said"

    # (6a) the launch in another shape than the entry states, field by field: the grid, the block, the shared memory
    def tampered(**fields):
        session = _recorded_run(monkeypatch, sequences=2, stack="native")[0]
        session.trace(4)
        for step in session.traced.steps:
            for index, record in enumerate(step):
                if census.canonical(record) == "swiglu_fp4":
                    step[index] = record._replace(**fields)
        return tie_out.run_checks(session, prompt_tokens=3)

    for fields, found in (
        ({"grid": (1, 1, 74)}, "grid (1, 1, 74), block (192, 1, 1), shared memory 231,424 B"),
        ({"block": (256, 1, 1)}, "grid (1, 1, 148), block (256, 1, 1), shared memory 231,424 B"),
        ({"shared_memory": 230400}, "grid (1, 1, 148), block (192, 1, 1), shared memory 230,400 B"),
    ):
        checks = tampered(**fields)
        assert not checks.ok and checks.kernels == census.FAIL
        assert sentences(checks) == [
            f"{every}mlp.gate_up (layer {layer}) launched swiglu_fp4 with {found}; cudnn.SwigluFp4Decode states grid (1, 1, 148), block (192, 1, 1), "
            "shared memory 231,424 B"
            for layer in (0, 1)
        ], fields

    # (7) wrong bytes: the block fails in its band, the kernel check holds
    for defect in ("code", "scale"):
        _, checks = run(swiglu_defect=defect)
        line = fused_line(checks)
        assert not checks.ok and checks.kernels == census.PASS and line.endswith("[band: swiglu_fp4]  FAIL"), (defect, line)
        assert not line.split("; ")[-1].startswith("0 outside") and "7 of 8 provider blocks inside their bands" in _line(checks, "blocks"), defect


def test_the_gate_up_rows_on_other_classes_row_counts_and_widths():
    """The fused gate / up entry serves compute capability 10.x and 1 to 64 rows at the published widths, and its rows say so:
    one row per token tile, selected by the row count; from 65 rows, on compute capability 12.x and for a checkpoint of
    the family with another MLP width the block stays on its multiply over a copy and the plan prints the fallback reason
    -- never an error.  The entry is not selected onto a class or a row count its rows do not claim.  On the other 10.x
    classes the rows are selected and no identity is stated (the census has no section there).  And a row that names
    another cluster cap than the census is ``not the target`` by its form."""
    from dataclasses import replace

    from model_samples import census
    from model_samples.blocks import DeviceFacts, plan
    from model_samples.qwen_dense import QwenDense, configs

    target = QwenDense.target()
    fallback = "cublas.bf16_dequantized+torch_ref.silu_mul_quantize"
    for sequences, tile in ((1, 8), (8, 8), (9, 16), (16, 16), (17, 32), (32, 32), (33, 64), (64, 64)):
        planned, model, _ = _target_plan(2, sequences=sequences)
        row = planned.rows["mlp.gate_up"]
        assert row.key == "fe.swiglu_fp4_decode" and dict(row.form)["TOKEN_TILE"] == tile and row.rows[0] <= sequences <= row.rows[1], sequences
        assert (
            census.check_forms(census.expected(planned, model)) == []
        ), f"{sequences} rows: the row's form is the entry's own (the public entry, built for the host)"
    planned, model, _ = _target_plan(2, sequences=65)
    assert planned.rows["mlp.gate_up"].key == fallback, "from 65 rows the block is the multiply over a copy"
    line = next(line for line in planned.table().splitlines() if line.split()[0] == "mlp.gate_up")
    assert line.endswith("fallback: bf16 copy of 4-bit weights") and "torch.matmul over a bf16 copy + torch_ref.silu_mul_quantize" in line, line
    with pytest.raises(
        NotImplementedError,
        match=r"^cudnn.SwigluFp4Decode serves rows 1-8 of 1-64, hidden 5120, intermediate 17408; this run has 65 sequences x 1 tokens per sequence$",
    ):
        _target_plan(2, sequences=65, overrides={"mlp.gate_up": "fe.swiglu_fp4_decode"})
    assert _target_plan(2, sequences=40, overrides={"mlp.gate_up": "fe.swiglu_fp4_decode"})[0].rows["mlp.gate_up"].rows == (
        33,
        64,
    ), "--provider: the row of the run's rows"
    # the other 10.x classes: the rows are the entry's, and the census has no section of the class -- no identity is stated
    for cc, sms in (((10, 3), 152), ((10, 7), 212)):
        planned, model, _ = _target_plan(64, cc=cc, sms=sms)
        assert planned.rows["mlp.gate_up"].key == "fe.swiglu_fp4_decode" and planned.rows["projection.fp8.in"].key == "cublas.bf16_dequantized", cc
        claim = target.claim(planned, model, installed=RECORDED)
        assert not claim.known and claim.coverage()[0].startswith(
            f"kernel identity: not stated for this run (no recorded section for cc {cc[0]}.{cc[1]} / {sms} SMs"
        ), cc
        # ... what a section of such a class would count: the target runs a library multiply for the in-projections there
        data = __import__("copy").deepcopy(target.data)
        data["configurations"]["plain"][f"cc{cc[0]}.{cc[1]}/sm{sms}"] = __import__("copy").deepcopy(data["configurations"]["plain"]["cc10.0/sm148"])
        there = census.Target(data).claim(planned, model, installed=RECORDED).numbers()
        assert (there["target_code"], there["target_port"]) == (49, 209) and there["not_yet"]["projection.fp8.in"] == 64, cc
    # compute capability 12.x: the multiply over a copy, with its reason; --provider cannot put the entry there
    planned, model, _ = _target_plan(2, cc=(12, 0), sms=170)
    assert planned.rows["mlp.gate_up"].key == fallback
    with pytest.raises(NotImplementedError, match=r"^cudnn.SwigluFp4Decode serves compute capability 10.x; this device is 12.0$"):
        _target_plan(2, cc=(12, 0), sms=170, overrides={"mlp.gate_up": "fe.swiglu_fp4_decode"})
    # a checkpoint of the family with another MLP width: not this row's, on any class -- the next row, and no identity stated
    wider = QwenDense(replace(QwenDense.reduced(2).geometry, mlp=20480), ("another MLP width",))
    facts = DeviceFacts((10, 0), "stated device", 148, 180 * GIB, 80 * GIB)
    planned = plan(wider.provider_table, wider.regimes, facts, 1, 1, "native", model=wider, options={"device": "cpu"})
    assert planned.rows["mlp.gate_up"].key == fallback and planned.rows["add_norm_quantize"].key == "fe.add_norm_quant_decode"
    claim = target.claim(planned, wider, installed=RECORDED)
    assert claim.coverage() == [
        "kernel identity: not stated for this run (no recorded section for this model's widths on cc 10.0 / 148 SMs: mlp 20480 (recorded 17408))"
    ]
    with pytest.raises(NotImplementedError, match=r"^cudnn.SwigluFp4Decode serves hidden 5120, mlp 17408; this model has mlp 20480$"):
        plan(wider.provider_table, wider.regimes, facts, 1, 1, "native", {"mlp.gate_up": "fe.swiglu_fp4_decode"}, model=wider, options={"device": "cpu"})
    # on the class and at 1 row: the target; selected back onto the copy with --provider: not the target set, and said
    planned, model, _ = _target_plan(2)
    claim = target.claim(planned, model, installed=RECORDED)
    assert claim.of("mlp.gate_up") == "target" and claim.numbers()["target_port"] == 10
    selected, model, _ = _target_plan(2, overrides={"mlp.gate_up": fallback})
    claim = target.claim(selected, model, installed=RECORDED)
    assert claim.reasons[0] == "not the target set: mlp.gate_up selected by --provider" and claim.of("mlp.gate_up") == "not yet (bf16 copy of 4-bit weights)"
    assert claim.numbers()["target_port"] == 8 and claim.compared()[0] == 10
    # a row that states another kernel than the census names (a cap of 152 thread blocks): the same entry, not the target, by its form
    table = dict(configs.PROVIDERS)
    rows = table["mlp.gate_up"]
    capped = replace(rows[0], form=tuple(("MAX_CLUSTERS", 152) if name == "MAX_CLUSTERS" else (name, value) for name, value in rows[0].form))
    table["mlp.gate_up"] = [capped, *rows[1:]]
    model = QwenDense.reduced(2)
    planned = plan(table, model.regimes, facts, 1, 1, "native", model=model, options={"device": "cpu"})
    claim = target.claim(planned, model, installed=RECORDED)
    everywhere = "PREWAIT_STAGES 2, L2_EVICT_FIRST True, MAX_CLUSTERS 148, PDL True"
    reason = f"the row's form is {SWIGLU_FORM.replace('MAX_CLUSTERS 148', 'MAX_CLUSTERS 152')}, the target's [{everywhere}, TOKEN_TILE 8]"
    assert claim.identities[claim.blocks.index("mlp.gate_up")] == census.Identity(census.OTHER, "swiglu_fp4", "port", reason)
    assert claim.numbers()["not_the_target"] == {"mlp.gate_up": 2} and claim.numbers()["target_port"] == 8 and len(claim.recorded) == 10
    sites = census.expected(planned, model)
    fused = next(site for site in sites if site.block == "mlp.gate_up")
    assert (
        census.check_forms(sites) == [] and fused.facts.form["MAX_CLUSTERS"] == 152
    ), "the row and its entry agree: the form is the row's, and it is not the census's"
    assert tuple(fused.facts.shape) == ((1, 1, 152), (192, 1, 1), 231424), "the cap is the grid: this field of the form does show in a launch record"


def test_the_loader_keeps_a_weight_in_the_layout_its_provider_multiplies_on(monkeypatch):
    """A provider that multiplies on its own layout of a stored weight hands the loader that layout (``layouts``): the load
    then keeps the weight in it and in no other stored form -- beside the bf16 copy where a block of the run reads one,
    alone where none does -- takes its checksum of the tensors the block multiplies on, and a copy made from them is the
    copy of the merged weight.  A layout without its stored weight is named.  At the published widths the layout holds
    exactly the bytes the family's arithmetic counts for the stored weight."""
    from model_samples.blocks import DeviceFacts
    from model_samples.qwen_dense import QwenDense, configs
    from model_samples.timing import Stages
    from model_samples.weights import BLOCK, BlockScaledTensor, Checkpoint, InterleavedBlockScaledTensor, dequantize, stored_checksum

    model = _tiny_model()
    block = "mlp.gate_up"
    stand_in = _stand_ins(monkeypatch)["fe.swiglu_fp4_decode"](configs.PROVIDERS[block][0], model, DeviceFacts((10, 0)), {})
    layouts = {block: stand_in.stored_layout}

    def load(**forms):
        stages = Stages(sync=lambda: None)
        return model.load(Checkpoint.synthetic(model, 0), "cpu", stages, **forms), stages

    plain, stages = load()
    assert "lay out" not in stages.seconds and all(layer.gate_up.stored is None for layer in plain.layers)
    merged, stages = load(stored=(block,))
    assert "lay out" not in stages.seconds and all(
        type(layer.gate_up.stored) is BlockScaledTensor for layer in merged.layers
    ), "no layout asked: as the checkpoint has it"
    beside, stages = load(stored=(block,), layouts=layouts)
    assert "lay out" in stages.seconds and beside.alone == (), "its own stage of the load; a weight with its copy has no checksum"
    alone, stages = load(stored=(block,), uncopied=(block,), layouts=layouts)
    assert "lay out" in stages.seconds and [(name, layer) for name, layer, _ in alone.alone] == [(block, 0), (block, 1)]
    for copy, checkpoint_order, kept, bare in zip(*(_weights_of(weights, block) for weights in (plain, merged, beside, alone))):
        laid = kept.stored
        assert type(laid) is type(bare.stored) is InterleavedBlockScaledTensor and laid.tile == 32 and laid.scale == copy.scale == bare.scale
        assert torch.equal(
            kept.data.view(torch.int16), copy.data.view(torch.int16)
        ), "the copy of a load is the copy of every stack: the checkpoint's row order"
        assert bare.data is None and all(torch.equal(a, b) for a, b in zip(bare.stored.tensors(), laid.tensors())), "the same tensor, with or without the copy"
        assert not torch.equal(laid.codes, checkpoint_order.stored.codes) and sorted(laid.codes.reshape(-1).tolist()) == sorted(
            checkpoint_order.stored.codes.reshape(-1).tolist()
        ), "the same bytes in the kernel's order: ONE stored form, never the checkpoint's beside it"
        assert torch.equal(dequantize(laid).data.view(torch.int16), copy.data.view(torch.int16)), "a copy made from it is that copy, bit for bit"
        assert bare.checksum == stored_checksum(bare.stored) and bare.intact() and kept.checksum is None
        bare.stored.codes[0, 0] ^= 1
        assert not bare.intact(), "the checksum is of what the block multiplies on"
    with pytest.raises(ValueError, match=r"^a load was handed a layout of the stored weights of mlp.gate_up, and was not asked to keep them$"):
        load(layouts=layouts)
    # the published widths: whole weight tiles, whole bands of whole scale tiles -- the layout is a permutation that adds no byte
    fmt, shapes = QwenDense().weight_shapes(block)
    assert fmt == "nvfp4" and all(out % 128 == 0 and (inner // BLOCK) % 4 == 0 and inner % 64 == 0 for out, inner in shapes)
    assert QwenDense().stored_bytes((block,)) == sum(out * inner // 2 + out * inner // BLOCK for out, inner in shapes) == 6_417_285_120
    assert QwenDense().check_bytes((block,)) == 2 * 34816 * 5120, "the reference's copy is made from it chunk by chunk: one copy, no second packed form"


# ----------------------------------------------------------------------------------------------- the out-projection of a verify step
LA_OUT, FA_OUT = "projection.fp8.out at linear_attention.out", "projection.fp8.out at full_attention.out"
#: the form of the row of the FP8 projection entry for the out-projection of a verify step of 2 to 8 rows, at its one call site
OUT_FORM = "[SPLIT_K 2, MMA_M 64, STAGES 11, TOKEN_TILE 8, TMEM_COLS 32, PREWAIT_WEIGHTS True, HOLD_TMEM True, PDL True, PDL_RELEASE True, OUT_FEATURES 5120, IN_FEATURES 6144]"
OUT_SYMBOL = "cudnn_kernel_frost_fp8_projection_decode_n5120k6144s2r6_m64g11t8c32_w1h1p1q1_0"
COMPARED_13 = (
    "13 launches of a step compared by launch shape (2 the target's own compiled kernels + 11 cuDNN Frontend entries with byte-equal results "
    "+ 0 declared stand-ins)"
)


def test_the_out_projection_row_of_a_verify_step_through_a_session_and_what_fails_it(monkeypatch):
    """The row of the FP8 projection entry for the out-projection of a verify step, through real sessions on the CPU: a
    decode session and a verify session of one sequence x six tokens on one load, on the stand-ins of the entries (the
    projection's multiplies on the stored codes the loader kept).  Clean, the verify run holds: the linear-attention
    layer's call is the entry's, judged against the reference multiply under its own name and served by the instance of
    its row; the attention layer's call is the multiply over a copy; 13 launches of a step are judged one by one, each
    in the recorded shape; and the decode session of the same load is the plan it was.  Then the negative controls on
    the verify plan, each of which must fail with its own sentence: the kernel of the other row range (another split);
    the entry's kernel at the attention layer's call; the kernels of a multiply over a copy under the row's name; a
    launch more in front of the kernel and behind it (it stands between two rows that are judged one by one); a launch
    less; a launch in another shape than the recorded one; and, with the kernel check holding, a wrong element and
    another weight of the same shape, which fail the block in its band.

    The ids of the steps are those of seed 0.  The token line of the tiny geometry moves with them -- its rows are 64
    wide, so one 4-bit code of the gate / up stand-in, another chain than the reference's, moves a logit by units -- and
    is not what a defect of this row is told by: every defect below is asserted by its own line."""
    from model_samples import census, tie_out
    from model_samples.qwen_dense import configs

    small = next(r for r in configs.PROVIDERS["projection.fp8.out"] if r.key == "fe.fp8_projection_decode")
    assert census.form_text(small.site_form("linear_attention.out")).strip() == OUT_FORM

    def run(**stand_in):
        verify, decode = _verify_run(monkeypatch, **stand_in)
        verify.trace(4)
        return verify, decode, tie_out.run_checks(verify, prompt_tokens=0)

    def sentences(checks):
        return [line.strip() for line in checks.lines if line.strip().startswith("kernels: ")]

    def judged(checks, label):
        return next(line for line in checks.lines if line.strip().startswith(label + " ") and " calls " in line)

    every = "kernels: step 0 (and 2 more of the 3 traced steps): "
    verify, decode, checks = run()
    assert checks.ok and checks.kernels == census.PASS, checks.text()
    assert "13 on exact rows, in the wiring order" in _line(checks, "kernels ")
    assert judged(checks, LA_OUT).endswith("[band: bf16_rounding]  PASS") and judged(checks, FA_OUT).endswith("[band: bf16_rounding]  PASS")
    assert not any(
        line.split()[:1] == ["projection.fp8.out"] and line.split()[1].isdigit() for line in checks.lines
    ), "each call site is judged under its own name"
    assert "9 of 9 provider blocks inside their bands" in _line(checks, "blocks")
    log = verify.traced.call_log[0]
    assert [(call.layer, call.instance, call.form) for call in log if call.block == "projection.fp8.out"] == [
        (0, "cudnn.Fp8ProjectionDecode #1", small.site_form("linear_attention.out")),
        (1, "", ()),
    ], "the linear-attention layer's call from the entry's instance, the attention layer's from no entry"
    assert _line(checks, "projection.fp8.out, as the call log of a traced step has it (1 calls): ").endswith(f"1 x cudnn.Fp8ProjectionDecode #1 {OUT_FORM}")
    assert _line(checks, f"{LA_OUT}: cudnn.Fp8ProjectionDecode {OUT_FORM}, 1 launch per step (1 call x 1 per call); ").endswith(
        f"the entry states grid (80, 2, 1), block (256, 1, 1), shared memory 103,808 B and the kernel {OUT_SYMBOL}"
    )
    assert _line(checks, "kernel identity, as planned: ").strip() == (
        "kernel identity, as planned: 13 of 20 launches of a step are target kernels (2 the target's own compiled kernels + 11 cuDNN Frontend entries "
        "with byte-equal results); not yet: mlp.down 2, attention.prepare 1, attention.core 1, projection.fp8.out 1, output_head 1; 1 declared "
        "stand-in (not yet: output_head.quantize)"
    )
    assert _line(checks, CENSUS_VERIFY).endswith(f"{COMPARED_13}: each in the recorded one")
    records = [record for record in verify.traced.steps[0] if census.canonical(record) == "fp8_projection"]
    assert [record.shape for record in records] == [PROJECTION_SHAPE[16384], OUT_SHAPE, PROJECTION_SHAPE[14336]], "in-projection, out-projection, in-projection"

    # the decode session of the same load: one row at both call sites, the plan and the numbers it had
    text = decode.text
    ids = decode.prefill([text.encode("ab0")])
    decode.read(ids)
    for _ in range(4):
        ids = decode.step(ids)
        decode.read(ids)
    decode.trace(4)
    plain = tie_out.run_checks(decode, prompt_tokens=3)
    assert plain.ok and plain.kernels == census.PASS, plain.text()
    assert judged(plain, "projection.fp8.out").endswith("[band: bf16_rounding]  PASS") and " at linear_attention.out" not in plain.text()
    assert "12 on exact rows, in the wiring order" in _line(plain, "kernels ") and _line(plain, CENSUS).endswith(f"{COMPARED_12}: each in the recorded one")

    # (1) the split of the other row range: the kernel of 9 to 16 rows -- four K slices, its own symbol and shape -- from an
    # instance that states that form.  The objects, the census and the entry's own symbol each say it
    _, _, checks = run(out_defect="split")
    other = OUT_FORM.replace("SPLIT_K 2", "SPLIT_K 4").replace("STAGES 11", "STAGES 7").replace("TOKEN_TILE 8", "TOKEN_TILE 16")
    assert (
        not checks.ok
        and checks.kernels == census.FAIL
        and sentences(checks)
        == [
            (
                f"kernels: form: projection.fp8.out (layer 0), the call at linear_attention.out: the adapter served the call with the form {other} "
                f"(cudnn.Fp8ProjectionDecode #2); the row says {OUT_FORM}"
            ),
            (
                f"{every}projection.fp8.out (layer 0) launched fp8_projection with grid (80, 4, 1), block (256, 1, 1), shared memory 84,352 B; "
                f"the {CENSUS_VERIFY} has grid (80, 2, 1), block (256, 1, 1), shared memory 103,808 B"
            ),
            (
                f"{every}projection.fp8.out (layer 0) launched 'cudnn_kernel_frost_fp8_projection_decode_n5120k6144s4r6_m64g7t16c32_w1h1p1q1_0'; "
                f"cudnn.Fp8ProjectionDecode names '{OUT_SYMBOL}' for this run"
            ),
        ]
    ), checks.text()
    assert judged(checks, LA_OUT).endswith("PASS"), "another split is another byte class of the kernel; this stand-in's arithmetic has none"

    # (2) the attention layer's call served by the entry: a launch of its kernel where the plan has the multiply over a copy
    _, _, checks = run(out_defect="attention")
    assert (
        not checks.ok
        and checks.kernels == census.FAIL
        and sentences(checks)
        == [
            f"{every}launched fp8_projection ('{OUT_SYMBOL}') after attention.gate_quantize (layer 1), where the plan has no such launch: "
            "the step has 4 launch(es) of fp8_projection, the plan 3 = 3 call(s) x 1 per call"
        ]
    ), checks.text()

    # (3) a library multiply under the row's name, and (6) a launch less: the entry's launch is missing
    missing = (
        f"{every}projection.fp8.out (layer 0) did not launch fp8_projection: the step has 2 launch(es) of fp8_projection, the plan 3 = 3 call(s) x 1 per call"
    )
    for defect in ("chain", "missing"):
        _, _, checks = run(out_defect=defect)
        assert not checks.ok and checks.kernels == census.FAIL and sentences(checks) == [missing], checks.text()
        assert judged(checks, LA_OUT).endswith("PASS"), "its bytes are the kernel's: only the kernel check sees it"

    # (4) (5) one launch more: in front of the kernel it stands behind the fused verify step, behind the kernel in front of the
    # norm of the MLP -- three rows that are judged one by one, so nothing may sit between them
    nothing = "a kernel of the fallback family pytorch, which no row there declares; nothing may sit between two exact rows"
    launched = f"launched {ELEMENTWISE[:93] + '...'!r} between"
    for defect, where in (
        ("extra", "linear_attention.step (layer 0) and projection.fp8.out (layer 0)"),
        ("behind", "projection.fp8.out (layer 0) and add_norm_quantize (layer 0)"),
    ):
        _, _, checks = run(out_defect=defect)
        assert not checks.ok and checks.kernels == census.FAIL and sentences(checks) == [f"{every}{launched} {where}: {nothing}"], checks.text()

    # (7) a launch in another shape than the recorded one -- the shared memory of the 16-row token tile: the census and the entry
    verify, _ = _verify_run(monkeypatch)
    verify.trace(4)
    for step in verify.traced.steps:
        at = [index for index, record in enumerate(step) if census.canonical(record) == "fp8_projection"][1]
        step[at] = step[at]._replace(shared_memory=117120)
    checks = tie_out.run_checks(verify, prompt_tokens=0)
    found = "projection.fp8.out (layer 0) launched fp8_projection with grid (80, 2, 1), block (256, 1, 1), shared memory 117,120 B; "
    assert (
        not checks.ok
        and checks.kernels == census.FAIL
        and sentences(checks)
        == [
            f"{every}{found}the {CENSUS_VERIFY} has grid (80, 2, 1), block (256, 1, 1), shared memory 103,808 B",
            f"{every}{found}cudnn.Fp8ProjectionDecode states grid (80, 2, 1), block (256, 1, 1), shared memory 103,808 B",
        ]
    ), checks.text()
    assert not any(COMPARED_13 in line for line in checks.lines), "nothing is said to have been compared"

    # (8) a wrong element, and (9) another weight of the same shape under the row's name: the block fails in its band against
    # the reference multiply, the kernel check holds
    for defect in ("element", "weight"):
        _, _, checks = run(out_defect=defect)
        line = judged(checks, LA_OUT)
        assert not checks.ok and checks.kernels == census.PASS and line.endswith("[band: bf16_rounding]  FAIL"), line
        assert _line(checks, "blocks").endswith("FAIL") and "8 of 9 provider blocks inside their bands" in _line(checks, "blocks")
        assert judged(checks, FA_OUT).endswith("PASS"), "the other call site is another row, judged on its own"


def test_the_loader_keeps_the_out_projection_weights_by_call_site(monkeypatch):
    """In a verify plan the out-projection of a linear-attention layer multiplies on the stored 8-bit codes and the
    out-projection of an attention layer on a bf16 copy: two rows of one block, by call site, so a load keeps the forms
    of that block per call site.  A verify plan alone keeps the linear-attention layers' weights on their stored codes
    alone -- no copy; the reference of the checks reads one made from those bytes for the call, and the checks hold the
    bytes to their load -- and the attention layers' as copies.  A run in which a decode session shares the load keeps
    what BOTH plans read: the codes for the verify plan, the copy for the decode plan.  The numbers of the memory
    sentence and of the bytes of a step follow, on the published model."""
    from model_samples import census, tie_out
    from model_samples.qwen_dense import QwenDense, model as family
    from model_samples.timing import Stages
    from model_samples.weights import ScaledTensor, dequantize

    one = "projection.fp8.out at linear_attention.out"

    def checked(**stand_in):
        verify, decode = _verify_run(monkeypatch, **stand_in)
        verify.trace(4)
        return verify, decode, tie_out.run_checks(verify, prompt_tokens=0)

    # the verify plan alone: the stored codes alone for the linear-attention layer, the copy alone for the attention layer
    verify, _, checks = checked(decode=False)
    weights = verify.weights
    assert weights.stored == weights.uncopied == ("projection.fp8.in", one, "mlp.gate_up") == verify.stored_weights() == verify.uncopied_weights()
    la, fa = (layer.out_proj for layer in weights.layers)
    assert (
        la.data is None and isinstance(la.stored, ScaledTensor) and isinstance(la.stored.scale, float) and la.checksum is not None
    ), "no copy: every plan multiplies on the codes"
    assert fa.data is not None and fa.stored is None and fa.checksum is None, "the attention layer's weight is read as a copy and kept as nothing else"
    assert [(block, layer) for block, layer, _ in weights.alone] == [
        ("mlp.gate_up", 0),
        ("projection.fp8.in", 0),
        ("projection.fp8.out", 0),
        ("mlp.gate_up", 1),
        ("projection.fp8.in", 1),
    ], "the weights kept on their stored format alone, in the order the loader makes them"
    assert checks.ok and checks.kernels == census.PASS, checks.text()
    assert (
        _line(checks, "stored weights")
        .strip()
        .startswith("stored weights    5 of 5 weights this run keeps on their stored format alone hold the bytes of their load, ")
    )
    assert f"weights this run keeps on their stored format alone (projection.fp8.in, {one}, mlp.gate_up)" in _line(checks, "on demand")
    assert next(line for line in checks.lines if line.strip().startswith(LA_OUT + " ")).endswith(
        "[band: bf16_rounding]  PASS"
    ), "against a copy made from the stored bytes"
    assert (
        verify.model.copy_bytes(weights.uncopied)
        == weights.copy_bytes
        == verify.model.copy_bytes(("projection.fp8.in", "mlp.gate_up")) - 2 * la.stored.codes.numel()
    )

    # ... and a block that writes into that weight moves its own reference with it: the checks name the weight, by block and layer
    _, _, checks = checked(decode=False, out_defect="writes")
    assert not checks.ok and _line(checks, "stored weights").endswith("FAIL") and "projection.fp8.out of layer 0 (" in _line(checks, "stored weights")

    # a decode session and a verify session on one load: the codes AND the copy of the linear-attention layer's weight
    verify, decode, checks = checked()
    weights = verify.weights
    assert (
        decode.weights is weights and weights.stored == ("projection.fp8.in", "mlp.gate_up", one) and weights.uncopied == ("projection.fp8.in", "mlp.gate_up")
    )
    la, fa = (layer.out_proj for layer in weights.layers)
    assert (
        la.data is not None and isinstance(la.stored, ScaledTensor) and la.checksum is None
    ), "both forms: the copy for the decode plan, the codes for the verify plan"
    assert torch.equal(dequantize(la.stored).data, la.data), "the copy is what the codes dequantise to"
    assert fa.data is not None and fa.stored is None
    assert len(weights.alone) == 4 and checks.ok, checks.text()
    assert "(projection.fp8.in, mlp.gate_up)" in _line(checks, "on demand"), "the reference of the checks reads the load's copy of the out-projection"
    assert decode.stored_weights(own=True) == ("projection.fp8.in", "mlp.gate_up") and verify.stored_weights(own=True) == (
        "projection.fp8.in",
        one,
        "mlp.gate_up",
    )

    # a plan that serves the block with ONE row answers for every call site of it: with a decode plan whose multiplies read the
    # stored format at both call sites (a test's own stand-ins) and a verify plan that is two rows on it, every plan multiplies
    # every out-projection on its codes -- the block is named whole, and no copy of any of its weights is made
    verify, decode, checks = checked(on_stored=True)
    assert decode.plan.at("projection.fp8.out", "linear_attention.out") == decode.plan.at(
        "projection.fp8.out", "full_attention.out"
    ), "one row in the decode plan"
    assert [label for label, name, _, _, _ in verify.plan.parts() if name == "projection.fp8.out"] == [LA_OUT, FA_OUT], "two in the verify plan"
    assert "projection.fp8.out" in verify.weights.stored and "projection.fp8.out" in verify.weights.uncopied and one not in verify.weights.uncopied
    assert all(layer.out_proj.data is None and layer.out_proj.stored is not None for layer in verify.weights.layers) and checks.ok, checks.text()

    # the loader itself: the names it takes, and what it refuses
    model, stages = _tiny_model(), Stages()
    from model_samples.weights import Checkpoint

    checkpoint = Checkpoint.synthetic(model, 0)
    both = model.load(checkpoint, "cpu", stages, stored=("projection.fp8.out",))
    assert all(layer.out_proj.stored is not None and layer.out_proj.data is not None for layer in both.layers), "the block: both call sites"
    other = model.load(checkpoint, "cpu", stages, stored=("projection.fp8.out at full_attention.out",), uncopied=("projection.fp8.out at full_attention.out",))
    assert [(layer.out_proj.data is None, layer.out_proj.stored is None) for layer in other.layers] == [(False, True), (True, False)]
    assert [(block, layer) for block, layer, _ in other.alone] == [("projection.fp8.out", 1)]
    for stored, uncopied, sentence in (
        (("projection.fp8.out at mlp",), (), "a provider of this run multiplies on those of projection.fp8.out at mlp$"),
        (("mlp.down at linear_attention.out",), (), "a provider of this run multiplies on those of mlp.down at linear_attention.out$"),
        ((one,), ("projection.fp8.out",), "keeps no bf16 copy of the weights of projection.fp8.out must keep their stored format, and was not asked to$"),
        ((), (one,), f"keeps no bf16 copy of the weights of {one} must keep their stored format, and was not asked to$"),
    ):
        with pytest.raises(ValueError, match=sentence):
            model.load(checkpoint, "cpu", stages, stored=stored, uncopied=uncopied)
    whole = model.load(checkpoint, "cpu", stages, stored=("projection.fp8.out",), uncopied=(one,))  # the block's codes cover the call site's
    assert [layer.out_proj.data is None for layer in whole.layers] == [True, False] and all(layer.out_proj.stored is not None for layer in whole.layers)
    assert family.weights_of(one) == ("projection.fp8.out", "linear_attention") and family.weights_of("projection.fp8.in at full_attention.qkv") == (
        "projection.fp8.in",
        "full_attention",
    )
    assert family.weights_of("mlp.down") == ("mlp.down", None)
    with pytest.raises(ValueError, match=r"^projection.fp8.out at mlp names no weight this family's loader keeps; "):
        family.weights_of("projection.fp8.out at mlp")

    # the published model: what the row adds to a load, in bytes -- the numbers of the family README
    full = QwenDense()
    codes = 48 * 5120 * 6144  # the out-projection weights of the 48 linear-attention layers, one byte per element
    assert (
        full.weight_shapes(one) == ("fp8", ((5120, 6144),) * 48) and full.weight_shapes("projection.fp8.out at full_attention.out")[1] == ((5120, 6144),) * 16
    )
    assert (
        full.weight_shapes("projection.fp8.out")[1] == ((5120, 6144),) * 64
        and full.weight_shapes("projection.fp8.in at full_attention.qkv")[1] == ((14336, 5120),) * 16
    )
    assert full.stored_bytes((one,)) == codes == 1_509_949_440 and round(codes / GIB, 2) == 1.41
    assert full.copy_bytes() - full.copy_bytes((one,)) == 2 * codes and round(2 * codes / GIB, 2) == 2.81, "the copies a verify plan alone does not make"
    assert full.check_bytes((one,)) == 2 * 5120 * 6144, "the checks' own copy of one such weight at a time"
    plain_kept = ("projection.fp8.in", "mlp.gate_up")
    shared = full.memory_bytes(1, 64, stored=(*plain_kept, one), uncopied=plain_kept, tokens=6) - full.memory_bytes(
        1, 64, stored=plain_kept, uncopied=plain_kept, tokens=6
    )
    assert (
        shared == codes + codes // 8 + codes // 4 and round(shared / GIB, 1) == 1.9
    ), "a decode and a verify session on one load: the codes beside every copy, and their share of working memory"
    step = full.bytes_per_step(1, 0, stored=(*plain_kept, one))
    before = full.bytes_per_step(1, 0, stored=plain_kept)
    assert (
        step["stored"] - before["stored"] == codes and before["copies"] - step["copies"] == codes
    ), "a step reads these weights at one byte per element, not two"

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
``main()``, run to its end on the CPU.  No GPU, no weights, no profiler."""

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
            return (SYMBOL,) if rows == 1 else ()

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


def _tiny_session(sequences, stack="reference"):
    from model_samples import Session, TextIO

    model = _tiny_model()
    text = TextIO.synthetic(model.geometry.vocab)
    session = Session(model, text, sequences=sequences, stack=stack, device="cpu", capacity=32)
    session.plan_table()
    session.load()
    return session, text


def _stand_ins(monkeypatch, step_launches=None, declares=None, profiler="records", serves=None):
    """The provider factories of a device that exists only as a log: every provider of the ``fe-cublas`` rows is a stand-in that
    runs the reference's arithmetic and writes the launches a real provider of its row would make into the log; a stand-in
    profiler (it replaces ``loop.trace_steps``) hands the log over as the launch records of a traced pass.

    ``step_launches`` replaces what the stand-in of the fused step launches (default: the entry's own symbol in the shape the
    entry states); ``declares`` replaces the census entries it declares; ``serves`` what it says served a call (default: its
    one entry instance, which states the form of the row).  ``profiler``: ``"records"``, or the reason it cannot run -- or a
    list of these, one per traced pass, the last one for every further pass."""
    from model_samples import loop
    from model_samples.census import Facts, Kernel, Launch
    from model_samples.providers import factories, torch_ref
    from model_samples.timing import Trace

    log = []
    passes = [profiler] if isinstance(profiler, str) else list(profiler)

    def launch(name, shape=((1, 1, 1), (128, 1, 1), 0), kind="kernel"):
        at = float(len(log))
        log.append(Launch(name, kind, at, at + 0.5, bytes=4) if kind == "copy" else Launch(name, "kernel", at, at + 0.5, *shape, 32))

    def logging(base, *names):
        class StandIn(base):
            def __call__(self, *operands):
                for name in names:
                    launch(name, kind="copy" if name.startswith("Memcpy") else "kernel")
                return super().__call__(*operands)

        return StandIn

    class Step(torch_ref.GdnDecodeStep):
        provider = "cuDNN FE"
        kernels = declares or (Kernel("gdn_decode_fused", "cuDNN FE", "exact", launches=1, shape="entry"),)

        def launch_facts(self, rows):
            return Facts((SYMBOL,), {"PDL": False, "PDL_RELEASE": False}, ((96, rows, 1), (256, 1, 1), 42112))

        def serving(self):  # what the stand-in's own entry instance states, whatever the row says
            return serves or ("cudnn.GdnDecodeFused #1", (("PDL", False), ("PDL_RELEASE", False)))

        def __call__(self, x, z, hidden, slot_ids, params, slots):
            for name, shape in step_launches or [(SYMBOL, ((96, x.shape[0], 1), (256, 1, 1), 42112))]:
                launch(name, shape)
            return super().__call__(x, z, hidden, slot_ids, params, slots)

    stand_ins = {
        "fe.gdn_decode_fused": Step,
        "cublas.bf16_dequantized": logging(torch_ref.Linear, ELEMENTWISE, NVJET, SPLIT_K, ELEMENTWISE),
        "cublas.bf16_dequantized+torch_ref.silu_mul_quantize": logging(torch_ref.GateUp, ELEMENTWISE, NVJET, ELEMENTWISE),
        "torch.index_select": logging(torch_ref.Embedding, GATHER),
        "torch.argmax": logging(torch_ref.TokenChoice, ARGMAX),
        "torch_ref.add_norm_quantize": logging(torch_ref.AddNormQuantize, ELEMENTWISE, COPY_IN, ELEMENTWISE),
        "torch_ref.attention_prepare": logging(torch_ref.AttentionPrepare, ARANGE, "Memcpy DtoD (Device -> Device)"),
        "torch_ref.attention": logging(torch_ref.Attention, GEMV, SOFTMAX, GEMM_FP64),
        "torch_ref.gate_quantize": logging(torch_ref.GateQuantize, ELEMENTWISE),
    }
    for key, stand_in in stand_ins.items():  # the stand-ins keep the census entries of the providers they stand for
        if key != "fe.gdn_decode_fused":
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


def _recorded_run(monkeypatch, sequences=2, **stand_in):
    """A decode of the tiny geometry on the CPU through the ``fe-cublas`` rows, on the stand-in providers and the stand-in
    profiler of :func:`_stand_ins` (``stand_in``: its options).  Returns ``(session, text, prompts)`` with the run recorded
    and the traced pass not yet made."""
    from model_samples import Session, TextIO, blocks

    build = _stand_ins(monkeypatch, **stand_in)
    model = _tiny_model()
    text = TextIO.synthetic(model.geometry.vocab)
    session = Session(model, text, sequences=sequences, stack="fe-cublas", device="cpu", capacity=16, check_kernels=True)
    facts = blocks.DeviceFacts((10, 0), "stated device", 148, 0, 0)
    options = {"reference_precision": "fp64", "device": "cpu", "check_kernels": True}
    session.plan = blocks.plan(model.provider_table, model.regimes, facts, sequences, model=model, options=options, factories=build)
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
    session = _recorded_run(monkeypatch)[0]
    session.trace(4, records=False)
    quiet = tie_out.run_checks(session, prompt_tokens=3)
    assert session.traced is None and quiet.ok and quiet.kernels == "", "a pass that was not asked for launch records keeps none"
    assert not any(line.strip().startswith("kernels") for line in quiet.lines), "and the fifth check is printed only where it was asked for"
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
    ok, judged, _ = sample._decode(session, text, StopRule(text.eos_ids, force_tokens=4), prompts, tie_out, check=True, checked_steps=None, kernels=True)
    out = capsys.readouterr().out
    assert ok and judged == 6 and session.traced.status == census.NOT_OBSERVED and session.traced.unobserved == "needs a recorded step on a CUDA device"
    assert "the tokens); the kernels of the traced steps against the plan" in out, "the line before the checks says that the kernels are checked too"
    assert "  kernels           not observed: needs a recorded step on a CUDA device    NOT OBSERVED" in out
    session.reset(fresh=True)
    session.new_report()
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
                return Path(path).write_text(_trace_text(events(self.steps))[:900])
            if self.fail == "writes nothing":
                return None  # an exporter that logs what it cannot write and raises nothing
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


def _sample_main(monkeypatch, capsys, arguments):
    """``main()`` of the sample script on stated device facts (compute capability 10.0), up to its first load."""
    import importlib.util

    from model_samples import blocks

    model_samples = importlib.import_module("model_samples")  # the package as an object: its names are patched below

    def no_load(session):
        raise AssertionError("reached a load")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(blocks.DeviceFacts, "probe", classmethod(lambda cls, device: cls((10, 0), "stated device", 148, 0, 80 * GIB)))
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

    from model_samples import blocks
    from model_samples.qwen_dense import QwenDense

    model_samples = importlib.import_module("model_samples")  # the package as an object: its names are patched below
    build = _stand_ins(monkeypatch, **stand_in)
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

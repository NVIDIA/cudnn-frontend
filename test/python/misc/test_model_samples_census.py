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

import dataclasses
import hashlib
import json
import math
import re
import struct
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
#: what the census line says was compared when the gather, the fused step, the gate, the attention preparation, the five norms,
#: the two in-projections and the two gate / up multiplies of the two-layer model held: three numbers
COMPARED_13 = (
    "13 launches of a step compared by launch shape (2 the target's own compiled kernels + 11 cuDNN Frontend entries with byte-equal results "
    "+ 0 declared stand-ins)"
)
#: the same on the stack ``native`` of the two-layer model: the two down multiplies of the library are held to their recorded
#: class and shape as well; its two out-projections, the output head of the reduced vocabulary and the head's quantise --
#: for which the census records no launch shape -- by membership alone
COMPARED_15 = (
    "15 launches of a step compared by launch shape (4 the target's own compiled kernels + 11 cuDNN Frontend entries with byte-equal results "
    "+ 0 declared stand-ins)"
)
#: ... and with the attention core on the library's kernel, at the maximum length the census was recorded with
COMPARED_16 = COMPARED_15.replace("15 launches", "16 launches").replace("(4 the target's", "(5 the target's")
OTHER_VOCAB = "no recorded launch shape of output_head for this model: vocab 15520 (recorded 248320)"
COMPARED_2 = (
    "2 launches of a step compared by launch shape (2 the target's own compiled kernels + 0 cuDNN Frontend entries with byte-equal results "
    "+ 0 declared stand-ins)"
)
WHOLE = "the whole target kernel set of a step of this model (what a complete stack launches; this stack's share is the next line): "
GATHER = "void at::native::(anonymous namespace)::indexSelectSmallIndex<BFloat16, int, unsigned int, 2, 2, -2>(at::cuda::detail::TensorInfo<...>)"
# the kernel torch.index_select launched for the embedding from 17 rows (compute capability 10.0, PyTorch 2.13.0), as recorded
GATHER_VECTORIZED = "void at::native::vectorized_gather_kernel<16, int>(char*, char*, int*, int, long, long, long, long, bool)"
ELEMENTWISE = "void at::native::vectorized_elementwise_kernel<2, at::native::CUDAFunctorOnSelf_add<double>, std::array<char*, 2ul> >(int, ...)"
FILL = "void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<unsigned char>, std::array<char*, 1ul> >(int, ...)"
ARGMAX = "void at::native::reduce_kernel<512, 1, at::native::ReduceOp<BFloat16, at::native::ArgMaxOps<BFloat16>, unsigned int, long, 4, 4> >(...)"
SOFTMAX = "void (anonymous namespace)::softmax_warp_forward<double, double, double, 3, false, false, 32>(double*, double const*, int, int, int, ...)"
ARANGE = "void (anonymous namespace)::elementwise_kernel_with_index<int, at::native::arange_cuda_out(...)::{lambda(long)#1}>(int, ...)"
NVJET = "nvjet_sm100_tst_64x8_64x16_2x1_v_bz_splitK_TNT"
SPLIT_K = "void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, float, __nv_bfloat16, __nv_bfloat16, true>(...)"
GEMV = "std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, double, double, double, double, false, true, cublasGemvParamsEx<int, ...> >(...)"
GEMM_FP64 = "void cutlass::Kernel2<cutlass_80_tensorop_d884gemm_32x64_16x4_nn_align1>(cutlass_80_tensorop_d884gemm_32x64_16x4_nn_align1::Params)"
HOST_READ = "Memcpy DtoH (Device -> Pageable)"
COPY_IN = "Memcpy HtoD (Pageable -> Device)"
# the kernels of FlashInfer's multiplies on the stored formats, as recorded on compute capability 10.0 with 148 SMs: the two
# classes of its 4-bit multiply (the down multiply and the output head of the published widths, with their launch shapes), its
# own CUTLASS 8-bit multiply (what the call launches where no tuner has chosen; the profiler leaves the name mangled) and the
# cuBLASLt kernel the census records as the tuner's pick
FI_SPLIT_K = (
    "kernel_cutlass_kernel_flashinfergemmkernelsdense_blockscaled_gemm_sm100_splitkSm100BlockScaledSplitKGemmKernel_object_at__TiledMMA_ThrLayoutVMNK11110000_0"
)
FI_PERSISTENT = "kernel_cutlass_kernel_flashinfergemmkernelsdense_blockscaled_gemm_sm100Sm100BlockScaledPersistentDenseGemmKernel_object_at__TiledMMA_ThrLayoutVMNK11110000_0"
FI_CUTLASS_FP8 = (
    "_ZN7cutlass13device_kernelINS_4gemm6kernel13GemmUniversalIN4cute5tupleIJiiiiEEENS1_10collective13CollectiveMmaINS1_35MainloopSm100TmaUmmaWarpSpecial"
    "izedILi13ELi2ELi4ENS5_IJNS4_1CILi1EEESB_SB_EEENS_4arch5Sm100EEENS5_IJNSA_ILi64EEESG_NSA_ILi128EEEEEENS_12float_e4m3_tENS5_IJlSB_lEEESJ_SK_NS4_8Tiled"
    "MMAINS4_8MMA_AtomIJNS4_10MMA_TraitsINS4_19SM100_MMA_F8F6F4_SSEJSJ_SJ_fSG_SG_NS4_17integral_constantINS4_4UMMA5MajorELSR_0EEESS_NSP_INSQ_7ScaleInELST"
    "_0EEESU_EEEEEENS4_6LayoutISC_NS5_IJNSA_ILi0EEESY_SY_EEEEENS5_IJNS4_10UnderscoreES11_S11_EEEEENS4_13SM90_TMA_LOADENS4_14ComposedLayoutINS4_7SwizzleIL"
    "i3ELi4ELi3EEENS4_18smem_ptr_flag_bitsILi8EEENSX_INS5_IJNSA_ILi8EEESH_EEENS5_IJSH_SB_EEEEEEEvNS4_8identityES14_S1E_vS1F_EENS_8epilogue10collective18C"
    "ollectiveEpilogueINS1H_23Sm100TmaWarpSpecializedILi3ELi2ELi16ELb1ELb0EEEJSI_NS5_IJNSX_ISG_SB_EENSX_INSA_ILi32EEESB_EEEEENS_10bfloat16_tENS5_IJSB_llE"
    "EES1Q_S1R_NS1H_6fusion15Sm90TreeVisitorINS1S_11Sm90ComputeINS_10multipliesES1Q_fLNS_15FloatRoundStyleE2EvEEJNS1S_19Sm90ScalarBroadcastIfSZ_Li1ES1V_E"
    "ENS1T_INS1U_IS1V_ffLS1W_2EvEEJS1Z_NS1S_12Sm90AccFetchEEEEEEENS4_5SM1004TMEM4LOAD26SM100_TMEM_LOAD_16dp256b4xES14_NS15_IS17_NS18_ILi16EEENSX_INS5_IJS"
    "G_S1A_EEENS5_IJSB_SG_EEEEEEENS4_17SM75_U16x8_LDSM_TENS4_14SM90_TMA_STOREES2C_NS4_17SM90_U16x8_STSM_TENS4_39AutoVectorizingCopyWithAssumedAlignmentIL"
    "i128EEEEEEvvEEEEvNT_6ParamsE"
)
NVJET_FP8 = "nvjet_sm100_qqtst_64x8_128x16_4x1_v_bz_TNT"
DOWN_SHAPE = ((40, 1, 2), (192, 1, 1), 231424)
HEAD_SHAPE = ((1, 1, 148), (192, 1, 1), 231424)
FP8_SHAPE = ((1, 80, 1), (256, 1, 1), 227328)
#: the same CUTLASS multiply on 4-bit operands: the name the library's own kernel would carry for another operand type
FI_CUTLASS_FP4 = FI_CUTLASS_FP8.replace("NS_12float_e4m3_tE", "NS_12float_e2m1_tE")
VALIDATED_LIBRARY = "0.7.2.post1"
#: the kernels of FlashInfer's paged decode attention at the published attention widths, as recorded on compute capability 10.0
#: with 148 SMs: the variant the census records for one sequence at a maximum length of 2,051, with its launch shape at one
#: token per sequence, and a variant the library launches at other step shapes
FI_ATTENTION = "fmhaSm100fKernel_QkvBfloat16OBfloat16H256PagedKvCausalP64MultiCtasKvCgaVarSeqQ8Kv256StaticGroupedSwapsAbForGen"
FI_ATTENTION_OTHER = "fmhaSm100fKernel_QkvBfloat16OBfloat16H256PagedKvCausalP64MultiCtasKvVarSeqQ8Kv256StaticGroupedSwapsAbForGen"
ATTENTION_CLASS = "MultiCtasKvCgaVarSeqQ8Kv256StaticGroupedSwapsAbForGen"
ATTENTION_SHAPE = ((5, 4, 1), (512, 1, 1), 178456)
ATTENTION_ENTRY = "flashinfer.decode.trtllm_batch_decode_with_kv_cache"
#: the maximum length the scripts default to and the census was recorded with
RECORDED_LENGTH = 2051
#: the table of the stand-in library's attention kernel set, as (architecture, kernel): the two kernels its call reads at
#: a step, one of the same geometry for compute capability 10.0 alone and one for 10.3 alone, and one of another head width
ATTENTION_CONTEXT = "Kernel_QkvBfloat16OBfloat16H256PagedKvCausalP64VarSeqQ128Kv128PersistentContext"
ATTENTION_SET = (
    ("kSM_100f", FI_ATTENTION),
    ("kSM_100f", FI_ATTENTION_OTHER),
    ("kSM_100", "fmhaSm100a" + ATTENTION_CONTEXT),
    ("kSM_103", "fmhaSm103a" + ATTENTION_CONTEXT),
    ("kSM_100f", FI_ATTENTION.replace("H256", "H128")),
)
#: the kernel-binary directory the stand-in library reads where a test names none: every file of its set (built at its first use)
ATTENTION_BINARIES = []
#: what the stand-in of the library's attention was handed, per call: every argument by name, as the objects it was handed
ATTENTION_CALLS = []
#: what the stand-in of the library was called with, per call: (function, backend, rows of the activation, shape of the weight as it was
#: handed over), in call order; and the shapes of the tiny model's down weight, head and out-projection weight as a step hands them over
LIBRARY_CALLS, LIBRARY_WEIGHTS = [], [(48, 64), (32, 272), (1, 16, 64)]
#: what the stand-in of the attention preparation entry was asked, per compile and per launch (:func:`_prepare_entry`), in call order
PREPARE_CALLS = []
#: the compiles and the launches of the stand-in of the 4-bit row quantise entry, in order
ROW_QUANTIZE_CALLS = []
#: the sentences of the area README that state the dependency policy of the library rows, as they were agreed
AREA_POLICY = (
    "FlashInfer is imported inside the adapters of the rows that call it, on a device, in the stack `native` alone; no other stack and no plan on "
    "stated device facts imports it.",
    "FlashInfer is an optional package of that stack alone; the stack `fe-entries` runs the cuDNN Frontend rows without it.",
    "no requirement file of this repository's tests names the package `native` needs (FlashInfer), which is why `fe-entries` exists. The GPU test "
    "has an L0 cell for `native` that runs where the package is installed at a validated version and is skipped elsewhere: skipped is uncovered, "
    "not passing",
    "Version updates for `samples/models/requirements.txt` come from the `pip` entry for `/samples/models` in `.github/dependabot.yml`, as pull "
    "requests, and are handled by the maintainers of this repository, as for its other requirement files. The packages `constraints-native.txt` "
    "pins are on the ignore list of that entry, each by its name with the three update types of a version update and nothing else: the entry "
    "proposes no version update for them -- not for a pin, and not for the lower bound of `flashinfer-python` in `requirements-flashinfer.txt` "
    "either, which is covered through the exact pin of the same package --, and a security update is not ignored. The pins are the versions a "
    "family's kernel census was recorded with, and a pin changes only together with a recorded run; a security update of one fails the host tests "
    "until that run is made.",
)
#: the two parts of a memory sentence that are the checks' own, as the sentence names them
COPY = "the reference's bf16 copy of one weight at a time, where the run keeps none"
BEYOND = "what they hold for the sequences beyond the first of a count"
#: what a run of 65,535 rows needs: its state, what its checks hold for 65,534 sequences beyond the first, and four tenths of
#: that for PyTorch's pool; on the stack ``native`` the library rows hold their weights on the stored codes at every row count
MOST = "6652.2 GiB on this GPU, 4563.36 GiB of it for the checks (what they hold for the 65534 sequences beyond the first)"
MOST_NATIVE = f"6652.0 GiB on this GPU, 4563.53 GiB of it for the checks ({COPY}, and what they hold for the 65534 sequences beyond the first)"


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
    assert kernel(_row_quantize_symbol(dict(width=5120, pdl=True, pdl_release=True))) == "row_quantize_fp4", "the 4-bit row quantise, by its symbol's head"
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
    # the library's multiplies on the stored formats: the two classes of the 4-bit one and no other kernel of that library;
    # for the 8-bit one a cuBLASLt kernel of 8-bit operands, or the library's own CUTLASS multiply -- the SM100 mainloop on
    # E4M3 operands, as its mangled name carries them -- and no other CUTLASS multiply
    assert kernel(FI_SPLIT_K) == kernel(FI_PERSISTENT) == "fi_mm_fp4" and kernel(NVJET_FP8) == kernel(FI_CUTLASS_FP8) == "cublaslt_fp8_gemm"
    assert len(FI_CUTLASS_FP8) == 1360 and FI_CUTLASS_FP8.count("NS_12float_e4m3_tE") == 1, "the name as the profiler recorded it, whole"
    for name in (
        "kernel_cutlass_kernel_flashinfergemmkernelsdense_bf16_gemm_sm100_splitkSm100DenseSplitKGemmKernel_object_at__TiledMMA_0",
        "kernel_cutlass_kernel_flashinfergemmkernelsdense_blockscaled_gemm_sm120_b12xSm120BlockScaledGemmKernel_object_at__0",
    ):
        assert kernel(name) == "", "another kernel of the library's CuTe DSL modules is not its 4-bit multiply of this class"
    assert FI_CUTLASS_FP4 != FI_CUTLASS_FP8 and FI_CUTLASS_FP4[:259] == FI_CUTLASS_FP8[:259], "the twin differs late in its name"
    for name in (FI_CUTLASS_FP4, FI_CUTLASS_FP8.replace("MainloopSm100TmaUmmaWarpSpecialized", "MainloopSm90TmaGmmaWarpSpecialized"), FI_CUTLASS_FP8[:170]):
        assert kernel(name) == "", "a CUTLASS multiply of other operands, of another mainloop, or a name cut short: no member"
    assert kernel("nvjet_sm100_tst_128x8_64x12_4x1_v_bz_TNT") == "cublas", "a cuBLASLt kernel of other operands is its family's, not the 8-bit multiply's"
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
        "attention_prepare",
        "row_quantize_fp4",
        "embedding_gather",
        "fi_mm_fp4",
        "cublaslt_fp8_gemm",
        "fi_decode_attention",
    } and judged <= set(named), "a pattern of its own each"
    # the library's kernels: both classes of its 4-bit multiply under one id; for its 8-bit multiply the cuBLASLt kernel of
    # 8-bit operands and the library's own CUTLASS multiply (a mangled name), each in front of the family it would fall to
    assert kernel(FI_SPLIT_K) == kernel(FI_PERSISTENT) == "fi_mm_fp4" and kernel(NVJET_FP8) == kernel(FI_CUTLASS_FP8) == "cublaslt_fp8_gemm"
    assert kernel("nvjet_sm103_qqtst_64x8_64x16_2x1_v_bz_splitK_TNT") == "cublaslt_fp8_gemm" and kernel(NVJET) == "cublas", "8-bit operands alone"
    assert named.index("cublaslt_fp8_gemm") < named.index("cublas") and census.FAMILY_OF["cublaslt_fp8_gemm"] == "cublas"
    for kernel_id in ("gdn_decode_fused", "gdn_verify_fused", "gate_quant", "add_norm_quant", "fp8_projection", "swiglu_fp4", "row_quantize_fp4"):
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
    assert len(wiring) == full.calls_per_step() == 485 and Counter(block for block, _ in wiring) == {b: sum(c) for b, c in full.calls().items()}
    assert wiring[0] == ("embedding", -1)
    assert wiring[-4:] == (("add_norm_quantize", 64), ("output_head.quantize", -1), ("output_head", -1), ("token_choice", -1))
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
    assert len(reduced.wiring()) == reduced.calls_per_step() == 21
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
        "fe.attention_prepare_decode",
        "fe.row_quantize_fp4_decode",
        "flashinfer.mm_fp4",
        "flashinfer.mm_fp4_head",
        "flashinfer.decode_attention",
    }, "the rows of the table whose entries serve some widths only"
    assert narrow_rows["flashinfer.mm_fp4"] == (("hidden", 5120), ("mlp", 17408)), "a library row names the widths its operand layout was run with"
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
    out_rows = [r for r in configs.PROVIDERS["projection.fp8.out"] if r.widths and r.key == "fe.fp8_projection_decode"]
    assert len(out_rows) == len(configs.VERIFY_OUT_ROWS) == 5
    # the library's row behind them serves both call sites, whose weights have one shape where the two mixers have one output width
    (library_row,) = [r for r in configs.PROVIDERS["projection.fp8.out"] if r.widths and r.key != "fe.fp8_projection_decode"]
    assert library_row.key == "flashinfer.bmm_fp8" and not library_row.sites and published.la_out == published.attn_out == 6144
    assert dict(library_row.widths) == {name: getattr(published, name) for name in ("hidden", "la_value_heads", "la_head_dim", "attn_heads", "attn_head_dim")}
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
        assert planned.rows["output_head.quantize"].key == "torch_ref.quantize", "another hidden width: the head's quantise stays on its reference chain"
        assert "by provider: cuDNN FE 2 | fallbacks 8 (cuBLAS over copies 4, PyTorch chains 4) | PyTorch 2" in planned.table()
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
        "  per step: 2 launches on 2 exact rows, checked one by one in the wiring order; the launches of 18 fallback calls and of the host path "
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

    def restated(row):  # the widths a row names, and the row width a form names (the row quantise entry's ``WIDTH``: the hidden width)
        form = tuple((name, geometry.hidden if name == "WIDTH" else value) for name, value in row.form)
        return replace(row, widths=tuple((name, getattr(geometry, name)) for name, _ in row.widths), form=form)

    return {block: [restated(row) for row in rows] for block, rows in configs.PROVIDERS.items()}


def _census_of(model):
    """The family's census with its sections re-stated for the layer widths of ``model``: the stand-in providers of this
    module launch the recorded shapes whatever the widths, so a run of the tiny geometry is a run a section holds for."""
    import copy

    from model_samples import census
    from model_samples.qwen_dense import QwenDense

    data = copy.deepcopy(QwenDense.target().data)
    for kernel in data["kernels"].values():  # a form that names the row width names the model's (``_table_of`` re-states the row)
        if "WIDTH" in kernel.get("form", {}):
            kernel["form"]["WIDTH"] = model.geometry.hidden
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


#: the form of the row of the attention preparation entry in the target stacks: the target's
PREPARE_FORM = "[POSITIONS 3d, SECTION (11, 11, 10), PAGE_SIZE 64, POSITION_DTYPE int64, SLOT_DTYPE int64, PDL False, PDL_RELEASE False]"


def _prepare_shape(rows):
    """The launch of the attention preparation entry at ``rows`` tokens, as the census records it at the published widths: one
    thread block per (token, head) over 24 query and 4 key / value heads, 128 threads, one fp32 slot of shared memory per warp."""
    return ((rows, 28, 1), (128, 1, 1), 16)


def _prepare_symbol(built):
    """The symbol a stand-in instance of the attention preparation entry states: the form's text behind the kernel's name, as
    the entry writes it -- the geometry, the position form with its section, the page, the two integer dtypes, the launch arm."""
    g = built["geometry"]
    section = "".join(f"_{part}" for part in built["mrope_section"] or ())
    arm = "plain" if not built["pdl"] else ("wait" if built["pdl_release"] is False else "release_wait")
    return (
        f"cudnn_kernel_frost_attention_prepare_h{g.HEADS}_{g.KV_HEADS}_d{g.HEAD_DIM}_r{g.ROTARY_DIM}_{built['positions']}{section}_p{built['page_size']}"
        f"_i{built['position_dtype'][3:]}_i{built['slot_dtype'][3:]}_{arm}_eps358637bd_0"
    )


def _prepare_entry(calls, launch=None, defect="", decline=None, shape=None):
    """A stand-in for ``cudnn.AttentionPrepareDecode`` on the CPU, at any widths.  It keeps what it was built with, states the
    entry's facts from it and writes its compiles and its launches into ``calls`` (a launch: the operands it was handed, by
    name).  A call runs the kernel's own chain in fp32 -- the mean of squares, a reciprocal root that is a few parts in ten
    million off (the kernel's is approximate), the gain, ONE rounding to bf16; the rotation by the table row each rotary index reads for the token's position, one fp32 product and a fused sum,
    one rounding more -- and writes what the kernel writes: the queries and the gates into the outputs it is handed, the keys
    and the values into the two pools at the row of each token's slot.  A token whose slot is negative or past the pool,
    or that lies past the slot vector, writes no pool row.  ``launch(name, shape)`` records the launch it makes (``shape``:
    another one than the recorded).  ``decline``: a one-element list whose text, when set, its build raises.  ``defect``
    makes a call ``"extra"`` (launch one more kernel in front of its own), ``"missing"`` (not launch), ``"chain"`` (launch a
    chain of PyTorch kernels in place of its own), or write a wrong query ``"element"``, another ``"gate"`` bit, another
    ``"value"`` bit, the key row of its first token at the ``"neighbour"`` of its slot as well, or a ``"nan"``."""

    def rounded(x, weight, cos, sin, eps, order):  # [tokens, heads, D] bf16 -> the same, normalised and rotated
        half = cos.shape[-1]
        xf = x.float()
        root = torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps) * torch.tensor(1.0 + 3e-7, dtype=torch.float32)
        normed = ((xf * root) * (1.0 + weight.float())).to(torch.bfloat16)
        n1, n2 = normed[..., :half].float(), normed[..., half : 2 * half].float()
        first = (n1.double() * cos.double() - (n2 * sin).double()).float().to(torch.bfloat16)
        if order == "3d":
            second = (n2.double() * cos.double() + (n1 * sin).double()).float().to(torch.bfloat16)
        else:
            second = (n1.double() * sin.double() + (n2 * cos).double()).float().to(torch.bfloat16)
        return torch.cat((first, second, normed[..., 2 * half :]), dim=-1)

    class Prepare:
        limit = 1024

        def __init__(self, device, geometry, eps, *, positions, page_size, position_dtype, slot_dtype, mrope_section=None, pdl=None, pdl_release=None):
            if (positions == "3d") != (mrope_section is not None):
                raise ValueError(f"positions={positions!r} with mrope_section={mrope_section!r}")
            self.built = dict(
                device=str(device),
                geometry=geometry,
                eps=eps,
                positions=positions,
                page_size=page_size,
                position_dtype=position_dtype,
                slot_dtype=slot_dtype,
                mrope_section=mrope_section,
                pdl=pdl,
                pdl_release=pdl_release,
            )

        def compile(self, rows=None):
            calls.append(("prepare compile", rows))
            if decline and decline[0]:
                raise NotImplementedError(decline[0])

        def kernel_symbols_for(self, rows):
            return (_prepare_symbol(self.built),)

        def form_for(self, rows):
            b = self.built
            pdl = bool(b["pdl"])
            return dict(
                GEOMETRY={name: getattr(b["geometry"], name) for name in ("HEADS", "KV_HEADS", "HEAD_DIM", "ROTARY_DIM")},
                EPS=b["eps"],
                POSITIONS=b["positions"],
                SECTION=b["mrope_section"],
                PAGE_SIZE=b["page_size"],
                POSITION_DTYPE=b["position_dtype"],
                SLOT_DTYPE=b["slot_dtype"],
                PDL=pdl,
                PDL_RELEASE=pdl if b["pdl_release"] is None else b["pdl_release"],
            )

        def launch_shape_for(self, rows):
            if rows > self.limit:
                raise NotImplementedError(f"this stand-in serves at most {self.limit} rows (tokens) per call; got {rows}")
            return shape or _prepare_shape(rows)

        def __call__(self, q, gate, k, v, q_weight, k_weight, cos_sin, positions, k_pool, v_pool, slot_mapping, q_out=None, gate_out=None):
            b, tokens = self.built, q.shape[0]
            operands = dict(q=q, gate=gate, k=k, v=v, q_weight=q_weight, k_weight=k_weight, cos_sin=cos_sin, positions=positions)
            calls.append(("prepare launch", dict(operands, k_pool=k_pool, v_pool=v_pool, slot_mapping=slot_mapping, q_out=q_out, gate_out=gate_out)))
            if tuple(k_pool.shape[1:]) != (b["geometry"].KV_HEADS, b["page_size"], b["geometry"].HEAD_DIM):
                raise ValueError(f"k_pool must be [num_pages >= 1, H={b['geometry'].KV_HEADS}, page_size={b['page_size']}, D={b['geometry'].HEAD_DIM}]")
            if tuple(positions.shape) != ((3, tokens) if b["positions"] == "3d" else (tokens,)):
                raise ValueError(f"positions must be {[3, tokens] if b['positions'] == '3d' else [tokens]} under positions={b['positions']!r}")
            if launch is not None:
                if defect == "chain":
                    launch(ELEMENTWISE)
                    launch(ELEMENTWISE)
                else:
                    if defect == "extra":
                        launch(ELEMENTWISE)  # in front of its own kernel: behind the in-projection's
                    if defect != "missing":
                        launch(_prepare_symbol(b), shape or _prepare_shape(tokens))
            half = b["geometry"].ROTARY_DIM // 2
            index = torch.arange(half)
            which = torch.zeros(half, dtype=torch.long)  # the position row each rotary index reads
            if b["positions"] == "3d":
                _, tall, wide = b["mrope_section"]
                which = torch.where((index % 3 == 1) & (index < 3 * tall), 1, torch.where((index % 3 == 2) & (index < 3 * wide), 2, 0))
            read = positions.reshape(-1, tokens).long()[which].t()  # [tokens, half]: the position of rotary index r for each token
            cos, sin = cos_sin[read, index][:, None], cos_sin[read, index + half][:, None]
            q_out[:tokens] = rounded(q, q_weight, cos, sin, b["eps"], b["positions"])
            gate_out[:tokens] = gate
            keys = rounded(k, k_weight, cos, sin, b["eps"], b["positions"])
            slots = slot_mapping.tolist()
            for token, slot in enumerate(slots[:tokens]):
                if 0 <= slot < k_pool.shape[0] * b["page_size"]:
                    page, row = divmod(slot, b["page_size"])
                    k_pool[page, :, row] = keys[token]
                    v_pool[page, :, row] = v[token]
                    if defect == "neighbour" and token == 0:
                        other, at = divmod((slot + 1) % (k_pool.shape[0] * b["page_size"]), b["page_size"])
                        k_pool[other, :, at] = keys[token]
            if defect == "element":
                q_out[0, 0, 0] = q_out[0, 0, 0].float() * 1.5 + 1.0
            if defect == "nan":
                q_out[0, 0, -1] = float("nan")
            if defect == "gate":
                gate_out.view(torch.int16)[0, 0, 0] ^= 1
            if defect == "value" and slots and 0 <= slots[0] < v_pool.shape[0] * b["page_size"]:
                v_pool.view(torch.int16)[slots[0] // b["page_size"], 0, slots[0] % b["page_size"], 0] ^= 1
            return q_out, gate_out

    return Prepare


def _row_quantize_shape(rows, width=5120):
    """The launch the 4-bit row quantise entry states at ``rows`` rows: one thread block per row, one thread per block of
    sixteen elements, no shared memory."""
    return ((rows, 1, 1), (width // 16, 1, 1), 0)


def _row_quantize_symbol(built):
    """The symbol a stand-in instance of the 4-bit row quantise entry states: the form's text behind the kernel's name, as the
    entry writes it -- the row width and the launch arm."""
    arm = ("wait_release" if built["pdl_release"] in (None, True) else "wait") if built["pdl"] else ("release" if built["pdl_release"] else "plain")
    return f"cudnn_kernel_frost_row_quantize_fp4_w{built['width']}_{arm}_0"


def _row_quantize_entry(calls, launch=None, defect="", decline=None, shape=None):
    """A stand-in for ``cudnn.RowQuantizeFp4Decode`` on the CPU, at any width, with the entry's signature and defaults.  It
    keeps what it was built with (``built``: a knob that was not passed stays ``None``), states the entry's facts from it and
    writes its compiles and its launches into ``calls`` (a launch: the operands it was handed, by name).  A call runs
    the kernel's rule in fp32 -- the block scale ``e4m3(scale * (largest * fp32(1 / 6)))``, the encode scale as the reciprocal
    of ``value * (1 / scale)`` formed with two IEEE divisions, the codes with the sign kept on a zero -- and writes what the
    kernel writes: the packed codes of the live rows and their block-scale bytes at their places of the 128 x 4 layout,
    into the outputs it is handed, or into its own pair of the row count (scales zero-filled) for an output it is not
    handed.  ``launch(name, shape)`` records the launch it makes: one thread block per row (``shape``: another).
    ``decline``: a one-element list whose text, when set, its build raises.  ``defect`` makes a call ``"extra"`` (launch
    one more kernel in front of its own), ``"behind"`` (one more behind it), ``"fill"`` (PyTorch's fill of the scale buffer
    in front of its kernel), ``"memset"`` (the same fill as a record of the profiler's fill category), ``"bands"`` (launch
    its kernel in a grid of whole bands of 128 rows, which is not the shape it states), ``"missing"`` (not launch), ``"chain"`` (launch a chain of
    PyTorch kernels in place of its own), or write per call a ``"code"`` two steps off, a block ``"scale"`` one step off,
    an ``"unsigned"`` zero, or the block scales ``"row-major"``."""
    from model_samples.weights import BLOCK, _e2m1, _e4m3, _e4m3_values, tile_scale_index

    class RowQuantize:
        limit = 1024

        def __init__(self, device, width, *, pdl=None, pdl_release=None):
            self.built = dict(device=str(device), width=width, pdl=pdl, pdl_release=pdl_release)
            self.owned = {}

        def compile(self, rows=None):
            calls.append(("row quantize compile", rows))
            if decline and decline[0]:
                raise NotImplementedError(decline[0])

        def kernel_symbols_for(self, rows):
            return (_row_quantize_symbol(self.built),)

        def form_for(self, rows):
            b, wait = self.built, bool(self.built["pdl"])
            return dict(WIDTH=b["width"], PDL=wait, PDL_RELEASE=wait if b["pdl_release"] is None else b["pdl_release"])

        def launch_shape_for(self, rows):
            if not 1 <= rows <= self.limit:
                raise NotImplementedError(f"this stand-in serves 1 to {self.limit} rows per call; got {rows}")
            return shape or _row_quantize_shape(rows, self.built["width"])

        def outputs(self, rows):
            width = self.built["width"]
            if rows not in self.owned:
                scales = torch.zeros((-(-rows // 128) * 128, width // BLOCK), dtype=torch.float8_e4m3fn)
                self.owned[rows] = dict(codes=torch.empty((rows, width // 2), dtype=torch.uint8), scales=scales)
            return self.owned[rows]

        def __call__(self, x, scale, codes=None, scales=None):
            b, rows = self.built, x.shape[0]
            calls.append(("row quantize launch", dict(x=x, scale=scale, codes=codes, scales=scales)))
            if tuple(x.shape[1:]) != (b["width"],) or x.dtype != torch.bfloat16 or scale.dtype != torch.float32 or scale.numel() != 1:
                raise ValueError(f"x must be bf16 [rows, {b['width']}] and scale one fp32 element")
            if launch is not None:
                if defect == "chain":
                    launch(ELEMENTWISE)
                    launch(ELEMENTWISE)
                else:
                    if defect in ("extra", "fill"):
                        launch(FILL if defect == "fill" else ELEMENTWISE)  # in front of its own kernel: behind the final norm's
                    if defect == "memset":
                        launch("Memset (Device)", kind="copy")
                    if defect == "bands":  # the grid of whole bands of 128 rows, where the entry states one thread block per row
                        launch(_row_quantize_symbol(b), ((-(-rows // 128) * 128, 1, 1), (b["width"] // 16, 1, 1), 0))
                    elif defect != "missing":
                        launch(_row_quantize_symbol(b), shape or _row_quantize_shape(rows, b["width"]))
                    if defect == "behind":
                        launch(ELEMENTWISE)  # behind its own kernel: in front of the head's
            sixth = torch.tensor(0x3E2AAAAB, dtype=torch.int32).view(torch.float32)
            blocks = x.float().unflatten(-1, (-1, BLOCK))
            stored = _e4m3(scale * (blocks.abs().amax(-1) * sixth))
            if defect == "scale":
                stored[0, 0] += 1  # one step up: not the byte of the kernel's rule
            value = _e4m3_values(stored)
            live = value != 0
            encode = torch.where(live, 1.0 / (torch.where(live, value, torch.ones_like(value)) * (1.0 / scale)), torch.zeros_like(value))
            nibbles = _e2m1(blocks * encode.unsqueeze(-1), signed_zero=defect != "unsigned").flatten(-2)
            if defect == "code":  # two steps off its code, the sign kept: outside every bracket
                nibbles[0, 0] = (nibbles[0, 0] & 0x8) | (((nibbles[0, 0] & 0x7) + 2) % 8)
            owned = self.outputs(rows) if codes is None or scales is None else {}
            codes, scales = owned["codes"] if codes is None else codes, owned["scales"] if scales is None else scales
            codes[:rows] = nibbles[..., 0::2] | (nibbles[..., 1::2] << 4)
            where = scales.view(torch.uint8).view(-1)
            if defect == "row-major":
                where[: stored.numel()] = stored.flatten()
            else:
                where[tile_scale_index(rows, stored.shape[-1], "cpu")] = stored
            return codes, scales

    return RowQuantize


def _kernel_binaries(directory, kernels=ATTENTION_SET, without=(), changed=(), unlisted=(), misnamed=(), manifest=True, header=None):
    """A kernel-binary directory of the stand-in library at ``directory``, laid out as the library's: the folder of the
    attention kernel set, named as the backend, with a manifest (a line per file: SHA-256, name), a header whose table
    names ``kernels`` with their architecture and SHA-256, and a binary per kernel.  ``without``: kernels whose binary is
    left out; ``changed``: files one byte of which differs from what the manifest lists (a kernel, or the header);
    ``unlisted``: kernels the manifest has no line for; ``misnamed``: kernels whose SHA-256 in the header is another
    file's; ``manifest`` false: no manifest; ``header``: another text in the header's place.  Returns ``directory``."""
    from model_samples.providers import flashinfer_calls

    folder = Path(directory) / "2d6a" / "fmha" / flashinfer_calls.DECODE_ATTENTION_BACKEND
    (folder / "include").mkdir(parents=True)
    digests = {name: hashlib.sha256(f"the binary of {name}".encode()).hexdigest() for _, name in kernels}
    named = {name: hashlib.sha256(b"another file").hexdigest() if name in misnamed else digest for name, digest in digests.items()}
    records = "\n".join(f'{{ 8, {architecture}, "{name}", 178456, "{named[name]}"}},' for architecture, name in kernels)
    declared = (
        "struct KernelMetaInfo\n{\n    int mTileSizeQ;\n    int mSM;\n    const char* mFuncName;\n    int mSharedMemBytes;\n    const char* sha256;\n};\n"
    )
    if header is None:
        header = f"#pragma once\n\n{declared}\nstatic const KernelMetaInfo sKernelMetaInfos[] = {{\n{records}\n}};\n"
    (folder / "include" / "flashInferMetaInfo.h").write_bytes(header.encode() + (b"\n" if "include/flashInferMetaInfo.h" in changed else b""))
    for _, name in kernels:
        if name not in without:
            (folder / f"{name}.cubin").write_bytes(f"the binary of {name}".encode() + (b"." if name in changed else b""))
    lines = [f"{digest}  {name}.cubin" for name, digest in digests.items() if name not in unlisted]
    if manifest:
        (folder / "checksums.txt").write_text("\n".join([*lines, f"{hashlib.sha256(header.encode()).hexdigest()}  include/flashInferMetaInfo.h"]) + "\n")
    return Path(directory)


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
    library_defect="",
    library_installed=VALIDATED_LIBRARY,
    library_binaries=(VALIDATED_LIBRARY, True),
    kernel_binaries=None,
    attention_defect="",
    prepare_defect="",
    prepare_serves=None,
    prepare_shape=None,
    quantize_defect="",
    quantize_serves=None,
    quantize_shape=None,
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
    the residual stream updated in place, the 4-bit block scales as the kernel forms them and in its 128 x 4 layout, no
    quantised operand where the form writes no codes -- states and launches one thread block per row (or
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

    The library rows run the REAL adapters (``providers.flashinfer_calls``) over a stand-in of the library: its 4-bit
    multiply decodes the packed codes and the block scales out of the 128 x 4 layout of both operands, accumulates in
    fp32, applies ``alpha`` and rounds once; its 8-bit multiply does the same on the 8-bit codes; each launches the
    kernel the library launched for that call on the recorded GPU -- the split-K class of the 4-bit multiply in the
    recorded shape for a weight of up to 128 rows (the down multiply of the tiny geometry), its persistent class for
    a larger one (the output head), the library's own CUTLASS multiply for the 8-bit call -- and its tuner says
    that nothing was tuned.  ``library_installed`` is the version the distribution's metadata states (``None``: not
    installed).  ``library_defect`` makes it: ``"class"`` (the persistent class at the down multiply, in the recorded
    shape), ``"shape"`` (the split-K class in another grid), ``"tuning"`` (its tuner is in tuning mode at every call),
    ``"has tuned"`` (its tuner has tuned the 4-bit multiply in this process), ``"extra"`` (one more kernel in front of
    the down multiply's), ``"chain"`` (the kernels of a multiply over a copy in place of the 4-bit kernel), ``"element"``
    (one wrong element per 4-bit call), ``"pick"`` (the 8-bit call launches the cuBLASLt kernel the census records),
    ``"split"`` (a split pick of cuBLASLt: its kernel and a reduce launch behind it), ``"mixed"`` (the recorded pick in
    every other step), ``"other"`` (the 8-bit call launches a kernel of neither family), ``"twin"`` (the 8-bit call
    launches the same CUTLASS multiply on 4-bit operands: another kernel, whose name differs from the library's own far
    from its start), ``"refuses"`` (the library refuses the operands of its first use), ``"mistyped"`` (the same, as a
    ``TypeError``), ``"build"`` (it cannot build its kernel at its first use), ``"import"`` (the package is installed
    and cannot be imported) or ``"paths"`` (its artifact paths name no folder of the attention kernel set) or ``"checksums"`` (its checksums name no manifest of that set).  Every call of the stand-in is written to ``LIBRARY_CALLS`` with the backend it was handed
    -- its defaults are the library's own, which are not the rows' --, its row count and the shape of its weight.

    The library's paged decode attention is the same stand-in's ``decode`` call, under the REAL adapter
    (``flashinfer_calls.DecodeAttentionBlock``): it keeps the library's defaults, writes what it is handed to
    ``ATTENTION_CALLS`` by name, reads the page table and the lengths as adjacent 32-bit words, gathers the attended rows
    of every sequence out of the two pool halves through them and returns a kernel-like chain in fp32 (the scores, the
    softmax and the weighted sum accumulated in fp32, one rounding to bf16: some elements another byte than the fp64
    chain's, every one inside the band) in the buffer it was handed; it launches the
    kernel the census records, in its shape for the tokens of the call, for one sequence at a maximum length of 2,051,
    and another variant of that kernel elsewhere.  ``library_binaries`` is the kernel-binary package as the installation
    has it: (the version its metadata states, whether its module can be found).  ``kernel_binaries`` is the kernel-binary
    directory the library resolved (:func:`_kernel_binaries`; left out: one that holds every file); the SHA-256 the
    library names for the manifest of its kernel set is that of the directory's manifest as the stand-in finds it.  The call reads two
    binaries of it, as the library does at the recorded step: first the variant it launches elsewhere, then the kernel
    it launches.  Without the first, or with one that is not the file the header names, it raises the library's
    refusal of a kernel binary; without the second it raises nothing, writes lines of its own with a local path to
    standard error and launches the first.  ``attention_defect`` makes the call:
    ``"variant"`` (the other variant where the recorded one belongs), ``"shape"`` (the recorded variant in another
    grid), ``"fill"`` or ``"multiply"`` (one PyTorch launch in front of the kernel), ``"behind"`` (a second kernel of
    the library behind it), ``"width"`` (the kernel of another head width), ``"page"`` (of another page), ``"element"`` (one wrong element per call),
    ``"writes"`` (it writes a byte of the pool), ``"refuses"`` / ``"mistyped"`` (the library's own check of an operand),
    ``"manifest"`` (the library's refusal of a manifest that is not on the machine), ``"binary"`` (its refusal of a
    kernel binary: the error of its loader's callback goes to the interpreter's hook, the call raises another) or
    ``"build"`` (it cannot build its launcher).

    A plan of several tokens per sequence has two more stand-ins.  The fused verify step runs the reference's verify
    step and launches the verify kernel's symbol in the recorded shape of one sequence of six tokens.  The FP8
    projection entry's stand-in serves the out-projection of the linear-attention layers too, on the row of the step's
    rows: it states and launches one thread block per K slice and 64 weight rows, with the shared memory of that row's
    kernel.  ``out_defect`` is a defect of that call: ``"split"`` (the kernel of the other row range: four K slices, its
    symbol, its shape, an instance that states its form), ``"chain"`` (the kernels of a multiply over a copy under the
    row's name), ``"behind"`` (one more launch behind its kernel), ``"missing"`` (no launch), ``"element"`` (one wrong
    element per call), ``"weight"`` (another weight of the same shape under the row's name: its rows in reverse),
    ``"writes"`` (it writes into the stored weight it multiplies on); and ``"attention"`` is a defect of the other call
    site: the attention layers' call, a multiply over a copy in the plan, launches the entry's kernel.

    The row of the attention preparation runs the REAL adapter (``fe.AttentionPrepareDecodeBlock``) over a stand-in of its
    entry (:func:`_prepare_entry`): the kernel's chain in fp32 on the four views, the table rows and the two pool views the
    adapter hands it, one launch per call in the recorded shape of the call's rows (or ``prepare_shape``), compared with
    the entry's real twin; the session then builds its attention cache as a pool of pages.  ``prepare_serves`` replaces
    what the adapter says served a call; ``prepare_defect`` is a defect of the stand-in entry: ``"extra"``, ``"missing"``,
    ``"chain"``, a wrong ``"element"``, ``"gate"`` bit or ``"value"`` bit, the key row at the ``"neighbour"`` of its
    slot as well, or a ``"nan"``.

    The row of the output head's quantise runs the REAL adapter (``fe.RowQuantizeFp4DecodeBlock``) over a stand-in of its
    entry (:func:`_row_quantize_entry`): the kernel's rule in fp32 on the rows and the one scale element the adapter hands
    it, into the two buffers the adapter hands it, one launch per call in the shape the entry states for the call's rows
    (or ``quantize_shape``), compared with the entry's real twin.  ``quantize_serves`` replaces what the adapter says
    served a call; ``quantize_defect`` is a defect of the stand-in entry: ``"extra"``, ``"behind"``, ``"fill"``,
    ``"memset"``, ``"missing"``, ``"chain"``, a wrong ``"code"`` or block ``"scale"``, the ``"unsigned"`` zero, or the block
    scales ``"row-major"``."""
    from types import SimpleNamespace

    import cudnn

    from model_samples import loop
    from model_samples.census import Facts, Kernel, Launch
    from model_samples.providers import factories, fe, flashinfer_calls, torch_ref
    from model_samples.timing import Trace
    from model_samples.weights import (
        BLOCK,
        BlockScaledTensor,
        InterleavedBlockScaledTensor,
        ScaledTensor,
        TiledBlockScaledTensor,
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
            else:  # the form ends at the bf16 rows: no quantised operand
                q = None
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

    class Tuner:
        """The stand-in of the library's tuner: whether it is tuning, and which operations it has tuned in this process."""

        is_tuning_mode = library_defect == "tuning"

        class stats:
            tuned_op_total_configs = {"fp4_gemm": 1} if library_defect == "has tuned" else {}

        @classmethod
        def get(cls):
            return cls

    def attended_rows(pool, table, sequence, attended):
        """The first ``attended`` positions of ``sequence`` out of one half of a pool, through its page-table row:
        ``[key / value heads, attended, D]``."""
        at = torch.arange(attended)
        return pool[table[sequence, at // pool.shape[2]].long(), :, at % pool.shape[2]].transpose(0, 1)

    def words(tensor, *shape):
        """The memory of ``tensor`` as a kernel reads it: adjacent 32-bit words from its first element, whatever its dtype."""
        count = 1
        for extent in shape:
            count *= extent
        return tensor.contiguous().view(-1).view(torch.int32)[:count].view(*shape)

    kernel_set = f"2d6a/fmha/{flashinfer_calls.DECODE_ATTENTION_BACKEND}"  # the folder of the attention kernel set, as the library's artifact path names it
    if kernel_binaries is None:
        if not ATTENTION_BINARIES:
            import tempfile

            ATTENTION_BINARIES.append(_kernel_binaries(Path(tempfile.mkdtemp(prefix="kernel_binaries_")) / "cubins"))
        kernel_binaries = ATTENTION_BINARIES[0]

    published = Path(kernel_binaries) / kernel_set / "checksums.txt"  # the manifest the stand-in library names: the directory's, as it stands now
    manifest_digest = hashlib.sha256(published.read_bytes()).hexdigest() if published.is_file() else ""

    def on_the_machine(name):
        """Whether the library's loader finds the binary of ``name``: the file is there and is the one the header names."""
        binary = Path(kernel_binaries) / kernel_set / f"{name}.cubin"
        return binary.is_file() and binary.read_bytes() == f"the binary of {name}".encode()

    def loader_callback(name):
        """The library's loader callback for a kernel binary that is not on the machine: it raises where only the
        interpreter's hook for an exception nobody can catch sees it, as a callback of ``ctypes`` does."""
        import ctypes

        def get_cubin_callback():
            raise RuntimeError(
                f"Artifact not found locally: {kernel_set}/{name}.cubin (looked at /a/local/path/cubins/{kernel_set}/{name}.cubin). "
                "FLASHINFER_NO_DOWNLOAD is set \u2014 refusing to download. This means flashinfer-cubin is missing this file."
            )

        get_cubin_callback.__module__ = flashinfer_calls.LOADER
        ctypes.CFUNCTYPE(None)(get_cubin_callback)()

    class Library:
        """The stand-in of the library: the two multiplies and the attention the adapters call, its tuner, the one size
        function the attention's adapter asks, the kernel-binary directory it resolved and its artifact paths."""

        class autotuner:
            AutoTuner = Tuner

        class jit:
            env = SimpleNamespace(FLASHINFER_CUBIN_DIR=Path(kernel_binaries))

        class artifacts:
            @dataclasses.dataclass(frozen=True)
            class ArtifactPath:
                MULTIPLY: str = "7b1f/gemm/"
                ATTENTION: str = ("2d6a/fmha/another" if library_defect == "paths" else kernel_set) + "/"
                ARCHS: tuple = ("sm_100a", "sm_103a")

            class CheckSumHash:
                map_checksums = {} if library_defect == "checksums" else {f"{kernel_set}/checksums.txt": manifest_digest}

        class utils:
            @staticmethod
            def get_trtllm_gen_multi_ctas_kv_counter_bytes(batch_size, num_qo_heads, sm_count):
                return (max(batch_size * num_qo_heads, sm_count) + 7) // 8 * 8 * 4

        class decode:
            @staticmethod
            def trtllm_batch_decode_with_kv_cache(
                query,
                kv_cache,
                workspace_buffer,
                block_tables,
                seq_lens,
                max_seq_len,
                bmm1_scale=1.0,
                bmm2_scale=1.0,
                window_left=-1,
                out=None,
                kv_layout="HND",
                enable_pdl=None,
                backend="auto",
                q_len_per_req=1,
                multi_ctas_kv_counter_buffer=None,
                **more,
            ):
                import os

                handed = dict(locals(), refused=os.environ.get(flashinfer_calls.NO_DOWNLOAD))
                handed.pop("os")
                ATTENTION_CALLS.append(handed)
                if attention_defect in ("refuses", "mistyped"):
                    raise (ValueError if attention_defect == "refuses" else TypeError)("Invalid shape of out\n(the library's own check of its operands)")
                if attention_defect == "build":
                    raise RuntimeError("ninja exited with status 1 in /a/local/path")
                if attention_defect == "manifest":
                    raise RuntimeError(
                        f"Artifact not found locally: {kernel_set}//checksums.txt (looked at /a/local/path/cubins/{kernel_set}/checksums.txt). "
                        "FLASHINFER_NO_DOWNLOAD is set \u2014 refusing to download. This means flashinfer-cubin is missing this file."
                    )
                if attention_defect == "binary" or not on_the_machine(FI_ATTENTION_OTHER):  # the binary the selection reads first
                    loader_callback(FI_ATTENTION_OTHER)
                    raise RuntimeError(f"Failed to load cubin for {FI_ATTENTION_OTHER}")
                tokens = q_len_per_req
                sequences = query.shape[0] // tokens
                recorded = sequences == 1 and max_seq_len == RECORDED_LENGTH
                grid = ((5 * tokens, 4, sequences), (512, 1, 1))
                name, shape = (FI_ATTENTION, (*grid, 178456)) if recorded else (FI_ATTENTION_OTHER, (*grid, 145168))
                if not on_the_machine(FI_ATTENTION):  # the binary it would launch: no error, lines of its own, the other kernel
                    sys.stderr.write("CUDA Error: CUDA_ERROR_NOT_FOUND /a/local/path/site-packages/flashinfer/data/include/fmhaKernels.cuh 1365\n")
                    name, shape = FI_ATTENTION_OTHER, (*grid, 145168)
                if attention_defect == "variant":
                    name = FI_ATTENTION_OTHER
                if attention_defect == "shape":
                    shape = ((4 * tokens, 4, sequences), *shape[1:])
                if attention_defect == "width":
                    name = name.replace("H256", "H128")
                if attention_defect == "page":
                    name = name.replace("P64", "P32")
                if attention_defect in ("fill", "multiply"):
                    launch(ELEMENTWISE if attention_defect == "fill" else NVJET)
                launch(name, shape)
                if attention_defect == "behind":
                    launch("flashinfer::ComputeLSEFromMDKernel(float*, float const*, int, int)")
                keys, values = kv_cache
                table, lengths = words(block_tables, sequences, block_tables.shape[-1]), words(seq_lens, sequences)
                rows = []
                for row in range(query.shape[0]):
                    sequence, token = divmod(row, tokens)
                    attended = int(lengths[sequence]) - (tokens - 1 - token)
                    k, v = (attended_rows(half, table, sequence, attended).float() for half in (keys, values))
                    grouped = query[row].float().view(k.shape[0], -1, query.shape[-1])  # [key / value heads, query heads of a group, D]
                    weights = torch.softmax(torch.einsum("hgd,htd->hgt", grouped, k) * bmm1_scale, dim=-1)
                    rows.append(torch.einsum("hgt,htd->hgd", weights, v))
                result = torch.stack(rows).view(query.shape).to(torch.bfloat16)
                if attention_defect == "element":
                    result[0, 0, 0] = result[0, 0, 0].float() * 1.5 + 4.0
                if attention_defect == "writes":
                    values.view(torch.int16)[0, 0, 0, 0] ^= 1
                if out is None:
                    return result
                out.copy_(result)
                return out

        class gemm:
            @staticmethod
            def mm_fp4(a, b, a_descale, b_descale, alpha, out_dtype, backend="auto"):
                LIBRARY_CALLS.append(("mm_fp4", backend, a.shape[0], tuple(b.shape)))
                if library_defect in ("refuses", "mistyped"):  # the library's own checks of its operands raise either type
                    raise (ValueError if library_defect == "refuses" else TypeError)("mm_fp4 accepts 2d tensors\n(the library's own check of its operands)")
                if library_defect == "build":
                    raise RuntimeError("ninja exited with status 1 in /a/local/path")
                rows, n, blocks = a.shape[0], b.shape[1], a.shape[1] * 2 // BLOCK
                down = n <= 128  # the down multiply of the tiny geometry; the output head has more weight rows
                if library_defect == "chain":
                    for name in (ELEMENTWISE, NVJET, ELEMENTWISE):
                        launch(name)
                else:
                    if library_defect == "extra" and down:
                        launch(ELEMENTWISE)
                    if down:
                        launch(
                            FI_PERSISTENT if library_defect == "class" else FI_SPLIT_K,
                            ((1, 1, 40), (192, 1, 1), 231424) if library_defect == "shape" else DOWN_SHAPE,
                        )
                    else:
                        launch(FI_PERSISTENT, HEAD_SHAPE)
                x = dequantize(BlockScaledTensor(a, untile_scales(a_descale.view(torch.uint8), rows, blocks), 1.0)).data
                weight = dequantize(BlockScaledTensor(b.t(), untile_scales(b_descale.t().view(torch.uint8), n, blocks), 1.0)).data
                out = ((x.float() @ weight.float().T) * alpha).to(out_dtype)
                if library_defect == "element":
                    out[0, 0] = out[0, 0].float() * 1.5 + 1.0
                return out

            @staticmethod
            def bmm_fp8(a, b, scale_a, scale_b, out_dtype, backend="cublas"):
                LIBRARY_CALLS.append(("bmm_fp8", backend, a.shape[1], tuple(b.shape)))
                if library_defect == "pick" or (library_defect == "mixed" and len(LIBRARY_CALLS) // 5 % 2):  # "mixed": in every other step
                    launch(NVJET_FP8, ((80, 1, 1), (256, 1, 1), 156168))
                elif library_defect == "split":
                    launch("nvjet_sm100_qqtst_64x8_64x16_2x1_v_bz_splitK_TNT", ((80, 2, 1), (256, 1, 1), 156168))
                    launch(SPLIT_K)
                elif library_defect == "other":
                    launch("a_kernel_of_another_library")
                elif library_defect == "twin":
                    launch(FI_CUTLASS_FP4, FP8_SHAPE)
                else:
                    launch(FI_CUTLASS_FP8, FP8_SHAPE)
                return ((a.float() @ b.float()) * (scale_a * scale_b)).to(out_dtype)

    def over_library(real):
        """The real adapter of a library row over the stand-in library, on the CPU."""

        class StandIn(real):
            first_use_devices = ("cuda", "cpu")

            def _library(self):
                if library_defect == "import":
                    raise ImportError("libcudart.so.13: cannot open shared object file in /a/local/path")
                return Library

        StandIn.__name__ = real.__name__
        return StandIn

    class Prepared(fe.AttentionPrepareDecodeBlock):
        """The real adapter of the attention preparation over the stand-in of its entry, on the CPU."""

        first_use_devices = ("cuda", "cpu")
        _entry = _prepare_entry(PREPARE_CALLS, launch, prepare_defect, shape=prepare_shape)

        def _cudnn(self):
            return SimpleNamespace(AttentionPrepareDecode=self._entry, __version__=cudnn.__version__)

        def serving(self):
            return prepare_serves or super().serving()

    Prepared.__name__ = fe.AttentionPrepareDecodeBlock.__name__

    class Quantized(fe.RowQuantizeFp4DecodeBlock):
        """The real adapter of the output head's quantise over the stand-in of its entry, on the CPU."""

        first_use_devices = ("cuda", "cpu")
        _entry = _row_quantize_entry(ROW_QUANTIZE_CALLS, launch, quantize_defect, shape=quantize_shape)

        def _cudnn(self):
            return SimpleNamespace(RowQuantizeFp4Decode=self._entry, __version__=cudnn.__version__)

        def serving(self):
            return quantize_serves or super().serving()

    Quantized.__name__ = fe.RowQuantizeFp4DecodeBlock.__name__
    del LIBRARY_CALLS[:], PREPARE_CALLS[:], ATTENTION_CALLS[:], ROW_QUANTIZE_CALLS[:]
    monkeypatch.setattr(fe, "_ENTRIES", {})  # the entries of this device: none of an earlier one's
    monkeypatch.setattr(flashinfer_calls, "installed", lambda: library_installed)
    monkeypatch.setattr(flashinfer_calls, "binaries_package", lambda: tuple(library_binaries))
    monkeypatch.delenv(flashinfer_calls.BINARIES_DIRECTORY, raising=False)  # the library's directory variable: a test that sets it says so
    monkeypatch.setattr(flashinfer_calls, "_WORKSPACES", {})  # the workspaces of this device: none of an earlier one's
    monkeypatch.setattr(flashinfer_calls, "_HELD", {})  # ... and nothing an earlier one held of its kernel binaries
    over_copies = OverCopies
    stand_ins = {
        "flashinfer.mm_fp4": over_library(flashinfer_calls.MmFp4Block),
        "flashinfer.mm_fp4_head": over_library(flashinfer_calls.MmFp4HeadBlock),
        "flashinfer.bmm_fp8": over_library(flashinfer_calls.BmmFp8Block),
        "flashinfer.decode_attention": over_library(flashinfer_calls.DecodeAttentionBlock),
        "fe.gdn_decode_fused": Step,
        "fe.gdn_verify_fused": VerifyStep,
        "fe.attention_gate_quant_decode": Gate,
        "fe.add_norm_quant_decode": Norm,
        "fe.fp8_projection_decode": Projection,
        "fe.swiglu_fp4_decode": Swiglu,
        "fe.attention_prepare_decode": Prepared,
        "fe.row_quantize_fp4_decode": Quantized,
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


def _recorded_run(monkeypatch, sequences=2, stack="fe-cublas", check_kernels=True, max_model_len=RECORDED_LENGTH, **stand_in):
    """A decode of the tiny geometry on the CPU through the rows of ``stack``, on the stand-in providers and the stand-in
    profiler of :func:`_stand_ins` (``stand_in``: its options), at the maximum length ``max_model_len``.  Returns
    ``(session, text, prompts)`` with the run recorded and the traced pass not yet made."""
    from model_samples import Session, TextIO, blocks, census

    from model_samples.qwen_dense import QwenDense

    build = _stand_ins(monkeypatch, **stand_in)
    monkeypatch.setattr(census, "installed_versions", lambda packages: dict(RECORDED))
    model = _tiny_model()
    monkeypatch.setattr(QwenDense, "_target", _census_of(model))
    text = TextIO.synthetic(model.geometry.vocab)
    session = Session(model, text, sequences=sequences, stack=stack, device="cpu", capacity=16, check_kernels=check_kernels, max_model_len=max_model_len)
    facts = blocks.DeviceFacts((10, 0), "stated device", 148, 0, 0)
    options = {"reference_precision": "fp64", "device": "cpu", "check_kernels": session.check_kernels, "max_model_len": session.max_model_len}
    session.plan = blocks.plan(model.provider_table, model.regimes, facts, sequences, 1, stack, model=model, options=options, factories=build)
    session.load()
    prompts = [text.encode("ab" + str(s)) for s in range(sequences)]
    ids = session.prefill(prompts)
    session.read(ids)
    for _ in range(4):
        ids = session.step(ids)
        session.read(ids)
    return session, text, prompts


def _verify_run(monkeypatch, sequences=1, tokens=6, decode=True, steps=3, stack="fe-entries", max_model_len=RECORDED_LENGTH, **stand_in):
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
        common = dict(sequences=sequences, stack=stack, device="cpu", capacity=40, check_kernels=True, max_model_len=max_model_len)
        session = Session(model, text, tokens=per_sequence, share=share, **common)
        options = {"reference_precision": "fp64", "device": "cpu", "check_kernels": True, "max_model_len": session.max_model_len}
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
    assert len(traced.steps) == 4 and len(traced.call_log) == 4 and all(len(calls) == 21 for calls in traced.call_log)
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


def _sample_main(monkeypatch, capsys, arguments, total=0, free=80 * GIB, script="00_decode.py"):
    """``main()`` of a sample script on stated device facts (compute capability 10.0), up to its first load.  ``total``:
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
    spec = importlib.util.spec_from_file_location(f"model_sample_qwen_dense_{script[:2]}_census_main", MODELS / "qwen_dense" / script)
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


def _sample_run(monkeypatch, capsys, arguments, cc=(10, 0), sms=148, **stand_in):
    """``main()`` of the sample script to its LAST line, on the CPU: stated device facts (compute capability ``cc`` with
    ``sms`` SMs), the tiny geometry, and the stand-in providers and stand-in profiler of :func:`_stand_ins` (``stand_in``:
    its options).  Returns ``(exit code, the lines the run printed)``."""
    import importlib.util

    from model_samples import blocks, census
    from model_samples.qwen_dense import QwenDense

    model_samples = importlib.import_module("model_samples")  # the package as an object: its names are patched below
    build = _stand_ins(monkeypatch, **stand_in)
    monkeypatch.setattr(census, "installed_versions", lambda packages: dict(RECORDED))
    stated = blocks.DeviceFacts(cc, "stated device", sms, 180 * GIB, 80 * GIB)
    real_session = getattr(model_samples.Session, "real", model_samples.Session)  # of a second run of one test: not the first run's wrapper

    def session(model, text, **options):  # the sample's session on the CPU, planned on the stated facts and the stand-ins
        made = real_session(model, text, **dict(options, device="cpu"))
        plan_options = {"reference_precision": made.reference_precision, "device": "cpu", "check_kernels": made.check_kernels}
        plan_options["max_model_len"] = made.max_model_len
        table, regimes = model.provider_table, model.regimes

        def resolve():  # where the sample asks the session for its plan: what a plan declines is then a line of the run
            if made.plan is None:
                made.plan = blocks.plan(
                    table, regimes, stated, made.sequences, 1, made.stack, made.overrides, model=model, options=plan_options, factories=build
                )
            return made.plan

        made.resolve = resolve
        return made

    session.real = real_session

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
    assert (written["sample"], written["stack"], written["schema"], written["traced_steps"]) == ("qwen_dense/00_decode", "fe-cublas", 2, 32)
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
    assert "ADDED []" in done.stdout and "TABLE True 21" in done.stdout, done.stdout + done.stderr


# ----------------------------------------------------------------------------------------------- the target kernel set
def _target_plan(layers=2, stack="native", cc=(10, 0), sms=148, sequences=1, overrides=None, max_model_len=RECORDED_LENGTH):
    """A plan of ``stack`` on stated device facts for a run of the maximum length ``max_model_len``, with the family's
    census: ``(plan, model, target)``."""
    from model_samples.blocks import DeviceFacts, plan
    from model_samples.qwen_dense import QwenDense

    model = QwenDense() if layers == 64 else QwenDense.reduced(layers)
    facts = DeviceFacts(cc, "stated device", sms, 180 * GIB, 80 * GIB)
    options = {"device": "cpu", "max_model_len": max_model_len}
    return plan(model.provider_table, model.regimes, facts, sequences, 1, stack, overrides, model=model, options=options), model, QwenDense.target()


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
            assert (launches, code, port, stand_ins) == (484, 194, 290, 0) and code + port + stand_ins == launches
            assert target.numbers(parts, QwenDense.reduced(2).geometry.layer_kinds) == (20, 8, 12, 0)
            assert target.numbers(parts, QwenDense.reduced(4).geometry.layer_kinds) == (34, 14, 20, 0)
    for layers in (2, 4, 64):  # the census's entries of a step, beside the wiring: every call has its place, and every target entry has its block
        plan, model, _ = _target_plan(layers)
        claim = target.claim(plan, model, installed=RECORDED)
        assert len(claim.identities) == len(model.wiring()) and claim.absent == [], layers
    plan, model, _ = _target_plan()
    exact = {name: [kernel.id for kernel in block.kernels if kernel.relation == "exact"] for name, block in plan.items()}
    served = {name: ids[0] for name, ids in exact.items() if ids}
    for block, kernel_id in served.items():
        meta = target.data["kernels"][kernel_id]
        same = "a change of a form the kernel census names updates kernel_census.json in the same commit"
        # the form the kernel has everywhere; a field that follows the row count (the fused gate / up kernel's token tile) is
        # named on the entries of the section's row count, below
        everywhere = {name: value for name, value in plan.rows[block].form if not (block == "mlp.gate_up" and name == "TOKEN_TILE")}
        everywhere = {
            name: list(value) if isinstance(value, tuple) else value for name, value in everywhere.items()
        }  # a tuple of the table is a list of the file
        assert everywhere == meta.get("form", {}), f"{block}: the form of its native row in configs.py and of {kernel_id} in the census differ; {same}"
        assert plan[block].entry == meta["entry"], f"{block}: the census names the entry the row calls"
    # the head's quantise: the entry's grid is its own (one thread block per row), so its two section entries record no launch
    # shape -- neither the entry's nor the target kernel's -- and the launch is held by membership; the row's form is the kernel's
    quantise = target.data["kernels"]["row_quantize_fp4"]
    assert (
        quantise["entry"] == "cudnn.RowQuantizeFp4Decode"
        and quantise["form"] == dict(configs.HEAD_QUANTIZE_FORM)
        and "fi_fp4_quantize" not in target.data["kernels"]
    )
    for configuration, sections_of in target.data["configurations"].items():
        for rows, parts_of in sections_of["cc10.0/sm148"]["rows"].items():
            entries = [
                (kernel_id, facts) for block, kernel_id, facts in target.sequence(parts_of, model.geometry.layer_kinds) if block == "output_head.quantize"
            ]
            assert entries == [("row_quantize_fp4", {})], f"{configuration}, {rows} rows: one entry of the block, and no launch shape on it"
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
    # which kernels a stack of this repository calls an entry for is said per kernel, and held here: the three numbers count
    # the complete stack, and a reader of the file must be able to tell what exists today
    kernels = target.data["kernels"]
    from model_samples import blocks

    several = blocks.plan(model.provider_table, model.regimes, plan.facts, 1, 6, "native", model=model)  # the plan of a verify step: six tokens per sequence
    stepped = {kernel.id for block in several.values() for kernel in block.kernels if kernel.relation == "exact"}
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
                "attention_prepare": ("docs/operations/attention/attention_prepare_decode.md", "test/python/attention_prepare/frost"),
                "row_quantize_fp4": ("docs/operations/block_scaling/row_quantize_fp4_decode.md", "test/python/row_quantize/frost"),
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
        "These are the numbers of the complete stack, not of today's. On compute capability 10.0 this stack launches all 290 today: the norm "
        "entry's 129, the projection entry's 64, the fused gate / up entry's 64, the gate entry's 16, the attention preparation entry's 16 and the "
        "row quantise entry's 1.",
        "for the row quantise entry it was made on compute capability 10.0, for one x86 build of the target kernel",
        "for the fused gate / up entry it was made on compute capability 10.0",
        "for the attention preparation entry it was made on compute capability 10.0",
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
    # every target kernel of the section has its launch shape, but a tuner's pick and the head's quantise -- a port whose launch
    # shape is its own (asserted above: its entries record none).  The reader refuses a section that says claimed for that
    # entry: how such a port is compared with the target's record is not part of this file yet
    for key, section in sections.items():
        for rows, parts in section["rows"].items():
            for block, kernel_id, facts in target.sequence(parts, QwenDense().geometry.layer_kinds):
                free = kernels[kernel_id]["relation"] == "stand-in" or "selection" in kernels[kernel_id] or kernel_id == "row_quantize_fp4"
                assert ("launch" in facts) != free, f"{key}, {rows} rows: {block} ({kernel_id})"
                assert set(facts) <= set(census.ENTRY_FACTS), block


def test_the_identity_of_every_block_call_and_the_three_numbers():
    """How every block call of the ``native`` plan stands to the target, decided against the census from the row's kernel and
    form: 484 of the 484 launches of a step of the 64-layer model are target kernels (194 the target's own compiled kernels +
    290 cuDNN Frontend entries with byte-equal results; no declared stand-in).  The first are the fused
    step's 48, the gather, and the library's calls: 64 out-projections, 64 down multiplies, the output head and the 16
    of the attention core.  The second are the norm entry's 129, in five forms, the projection entry's 64, at two weight
    shapes, the fused gate / up entry's 64, the gate entry's 16, the attention preparation entry's 16 and the row quantise
    entry's 1 -- the quantise in front of the output head, for which the census records no launch shape: it is held by
    membership, with the out-projections.  No block of the stack is ``not yet``.  ``fe-entries`` has the cuDNN Frontend
    rows alone, and its lines say so.  Always three numbers; every number here is the plan's or the census file's."""
    from model_samples import census
    from model_samples.qwen_dense import QwenDense, configs

    plan, model, target = _target_plan(64)
    claim = target.claim(plan, model, installed=RECORDED)
    numbers = claim.numbers()
    section = target.data["configurations"]["plain"]["cc10.0/sm148"]["numbers"]  # the three numbers of the set: the census file's
    assert (numbers["launches"], numbers["code"], numbers["port"], numbers["stand_ins"]) == tuple(
        section[name] for name in ("launches", "code", "port", "stand_in")
    )
    assert (numbers["target_code"], numbers["target_port"]) == (section["code"], section["port"]) and numbers["not_the_target"] == {}
    assert numbers["target_port"] == numbers["port"], "on this class every port of the set is launched through its entry"
    assert numbers["target_code"] == numbers["code"] and numbers["not_yet"] == {}, "and every kernel of the target's own"
    assert section["code"] + section["port"] + section["stand_in"] == section["launches"] and numbers["stand_ins_not_yet"] == [] and claim.absent == []
    (whole,) = claim.coverage()  # one line: the stack launches the whole set, so its share is not a second line
    stand = f"{section['stand_in']} declared stand-in{'s' if section['stand_in'] != 1 else ''}"
    assert whole == (
        f"kernel identity, as planned: {section['launches']} launches per step = {section['code']} the target's own compiled kernels + "
        f"{section['port']} cuDNN Frontend entries with byte-equal results + {stand}"
    ), "the three numbers of the complete stack, which this stack launches"
    readme = (MODELS / "qwen_dense" / "README.md").read_text()
    assert whole in readme, "the line of the family README is the line a run prints"
    assert configs.readme_table(QwenDense().calls_text(), "native") in readme, "regenerate the native table of the README from configs.readme_table()"
    # fe-entries: the cuDNN Frontend rows of native alone, the three blocks a library serves there on their fallback rows
    entries_plan, _, _ = _target_plan(64, "fe-entries")
    entries = target.claim(entries_plan, model, installed=RECORDED)
    served = entries.numbers()
    assert (
        served["target_port"] == section["port"] and served["target_code"] == section["code"] - 64 - 64 - 16 - 1
    ), "every port, and the code less the library's 145"
    assert (
        entries.coverage()[1]
        == (
            f"kernel identity, as planned: {served['target_code'] + served['target_port']} of {section['launches']} launches of a step are target kernels "
            f"({served['target_code']} the target's own compiled kernels + {served['target_port']} cuDNN Frontend entries "
            "with byte-equal results); not yet: projection.fp8.out 64, mlp.down 64, "
            f"attention.core 16, output_head 1; {stand}"
        )
        and entries.coverage()[1] in readme
        and entries.coverage()[0] == WHOLE + whole.split(": ", 1)[1]
        and entries.coverage()[0] in readme
    ), "and its two lines are in the README beside the line of native: the whole set, and this stack's share"
    assert entries.compared()[2] == 1 and entries.line(census.PASS).endswith(
        ": each in the recorded one; 1 more by membership alone"
    ), "the quantise, there too"
    differing = {block for block in plan.rows if plan.rows[block].key != entries_plan.rows[block].key}
    # every count the two READMEs state of the library rows is the plan's: the area README's sentence with what is held to a
    # recorded launch shape and what by membership; the family README's share of the target's own kernels, its census line,
    # the two-layer model's counts, the launches of each library row in the stack's table, the number of differing rows
    spaced, area = " ".join(readme.split()), " ".join((MODELS / "README.md").read_text().split())
    code, port, launches = numbers["target_code"], numbers["target_port"], numbers["launches"]
    split = "{} the target's own compiled kernels + {} cuDNN Frontend entries with byte-equal results"
    shaped, _, by_membership = claim.compared()
    # by launch shape: the launches the census holds a record of the target for.  By membership: the out-projections, whose
    # kernel the library chooses, and the head's quantise, whose census entry records no launch shape (its grid is its own)
    held = [identity for identity in claim.identities if identity.relation == "target"]
    loose = [block for block, identity in zip(claim.blocks, claim.identities) if identity.relation == "target" and not identity.launch]
    shaped_code, shaped_port = (sum(identity.kind == kind and bool(identity.launch) for identity in held) for kind in ("code", "port"))
    assert (code + port, shaped, by_membership) == (launches, 419, 65) and shaped + by_membership == code + port and shaped == shaped_code + shaped_port
    assert sorted(set(loose)) == ["output_head.quantize", "projection.fp8.out"] and loose.count("output_head.quantize") == 1 and len(loose) == by_membership
    assert f"{shaped} launches of a step compared by launch shape" in claim.line(census.PASS) and f"{shaped + 1} launches" not in claim.line(census.PASS)
    assert claim.line(census.PASS) == (
        f"{CENSUS}: NOT CLAIMED (the census does not claim this class yet); {claim.compared()[1]}: each in the recorded one; "
        f"{by_membership} more by membership alone"
    ), "nothing but the unclaimed class keeps this run from EQUAL, and the line counts the quantise by membership"
    # the launch shape of the attention core is held for a run at the maximum length the census was recorded with alone
    longer = target.claim(_target_plan(64, max_model_len=4096)[0], model, installed=RECORDED)
    assert (longer.numbers()["target_code"], longer.compared()[0], longer.compared()[2]) == (code, 403, 81)
    assert "at another maximum length the library launches another variant of its kernel or another grid" in spaced and "as well: 403 and 81." in spaced
    assert (
        f"with them it launches {code + port} of the {launches} ({split.format(code, port)}) on compute capability 10.0 at 148 SMs, the class that ran, "
        "and no launch of the step is a declared stand-in." in area
    )
    assert f"Of those, {shaped} ({split.format(shaped_code, shaped_port)}) are held to the kernel and the launch shape the census records" in area
    calls = {block: sum(site.block == block for site in census.expected(plan, model, rows=1)) for block in sorted(differing)}
    assert calls == {"attention.core": 16, "mlp.down": 64, "output_head": 1, "projection.fp8.out": 64} and by_membership == calls["projection.fp8.out"] + 1
    assert (
        f"the other {by_membership} are held by membership: the {calls['projection.fp8.out']} out-projections, whose kernel the library chooses among its "
        "own, and the quantise, for which the census records no launch shape -- its kernel is held to the launch shape its entry states, not to a record "
        "of the target" in area
    )
    said = "the library, untuned, launched its own CUTLASS multiply for the out-projections, not the cuBLASLt kernel the census records as the target's pick"
    assert said in area
    library = sum(calls.values())
    assert f"this stack launches all {port} today" in spaced and code == numbers["code"] == 48 + 1 + library
    assert f"it launches all {code} of the target's own compiled kernels: the fused step's 48, PyTorch's gather and {library} calls of FlashInfer" in spaced
    assert (
        f"{claim.compared()[1]}: each in the recorded one; {by_membership} more by membership alone` -- the {by_membership} are the "
        f"{calls['projection.fp8.out']} out-projections, whose kernel the library chooses, and the quantise in front of the output head: the census "
        "records no launch shape for it, so its kernel is held to the launch shape its entry states and to no record of the target." in spaced
    )
    for label, block, called in (
        ("8-bit out-projection", "projection.fp8.out", "`flashinfer.gemm.bmm_fp8` on the stored 8-bit weight"),
        ("MLP down", "mlp.down", "`flashinfer.gemm.mm_fp4` on the stored 4-bit weight"),
        ("output head", "output_head", "the same call of `flashinfer.gemm.mm_fp4` on the stored 4-bit weight"),
        ("attention core", "attention.core", "`flashinfer.decode.trtllm_batch_decode_with_kv_cache` on the pool of pages the preparation wrote"),
    ):
        assert f"| {label} | {calls[block]} | {called}" in readme, f"{label}: the launches of a step the stack's table states are the plan's calls"
    assert "The two stacks differ in four rows: the MLP's down multiply, the output head, the 8-bit out-projection and the attention core" in spaced
    assert len(differing) == 4
    small_plan, small, _ = _target_plan(2)
    two = target.claim(small_plan, small, installed=RECORDED).numbers()
    small_entries = target.claim(_target_plan(2, "fe-entries")[0], small, installed=RECORDED).numbers()
    assert (
        f"two-layer model counts {two['target_code'] + two['target_port']} of {two['launches']} ({split.format(two['target_code'], two['target_port'])}; "
        f"no declared stand-in), and `fe-entries` {small_entries['target_code'] + small_entries['target_port']} of {small_entries['launches']} "
        f"there ({split.format(small_entries['target_code'], small_entries['target_port'])})" in spaced
    ), "the counts of the two-layer model on both stacks, each with its split, as the plans count them"
    shapes = {"embedding": GATHER_SHAPE, "add_norm_quantize": NORM_SHAPE, "linear_attention.step": SHAPE, "attention.gate_quantize": GATE_SHAPE}
    shapes["mlp.gate_up"], shapes["attention.prepare"] = SWIGLU_SHAPE, _prepare_shape(1)
    shapes.update({"mlp.down": DOWN_SHAPE, "output_head": HEAD_SHAPE})  # the library's 4-bit multiply: its two kernel classes, each in its recorded shape
    shapes["attention.core"] = ATTENTION_SHAPE  # the library's attention, for a run at the maximum length the census was recorded with
    kinds = model.geometry.layer_kinds
    assert claim.recorded == {
        # an in-projection's recorded shape is the one of its layer's weight: grid 128 in a linear-attention layer, 112 in an attention layer
        index: (PROJECTION_SHAPE[model.geometry.in_projection(kinds[layer])[0]] if block == "projection.fp8.in" else shapes[block])
        for index, (block, layer) in enumerate(model.wiring())
        if block in shapes or block == "projection.fp8.in"
    }, "the recorded launch shape of every target launch but the library's 8-bit multiplies, whose kernel its tuner chooses: 419 of them, as tuples"
    assert (
        len(claim.recorded) == shaped and "output_head.quantize" not in shapes
    ), "the launches less the out-projections and the head's quantise, which are held by membership alone"
    classes = {model.wiring()[index][0]: named for index, named in claim.classes.items()}
    assert classes == {
        "mlp.down": "Sm100BlockScaledSplitKGemmKernel",
        "output_head": "Sm100BlockScaledPersistentDenseGemmKernel",
        "attention.core": ATTENTION_CLASS,
    }, "the class of each 4-bit multiply, and the variant of the attention kernel"
    assert (
        len(claim.classes) == 65 + 16 and set(claim.picks.values()) == {NVJET_FP8} and len(claim.picks) == 64
    ), "and the recorded pick of every out-projection"
    assert configs.readme_sites(QwenDense(), "native") in readme, "regenerate the call-site table of the README from configs.readme_sites()"

    plan, model, _ = _target_plan(2)
    claim = target.claim(plan, model, installed=RECORDED)
    two_layers = target.numbers(target.data["configurations"]["plain"]["cc10.0/sm148"]["rows"]["1"], model.geometry.layer_kinds)
    assert claim.coverage() == [
        f"kernel identity, as planned: {two_layers[0]} launches per step = {two_layers[1]} the target's own compiled kernels + {two_layers[2]} cuDNN "
        f"Frontend entries with byte-equal results + {two_layers[3]} declared stand-in{'s' if two_layers[3] != 1 else ''}"
    ], "the two-layer model: the whole set of its step, launched whole"
    head = claim.identities[claim.blocks.index("output_head")]
    assert (head.relation, head.launch, head.named) == (
        "target",
        None,
        "",
    ) and head.reason == "vocab 15520 (recorded 248320)", "another vocabulary: held by name alone"
    kinds = {block: (identity.relation, identity.kind) for block, identity in zip(claim.blocks, claim.identities)}
    assert kinds["embedding"] == kinds["linear_attention.step"] == ("target", "code") and kinds["attention.gate_quantize"] == ("target", "port")
    assert kinds["token_choice"] == ("host", "") and kinds["add_norm_quantize"] == ("target", "port")
    assert kinds["mlp.gate_up"] == kinds["attention.prepare"] == ("target", "port")
    quantise = target.data["kernels"]["row_quantize_fp4"]  # the head's quantise stands to the target as the census file says its kernel does
    assert kinds["output_head.quantize"] == (quantise["relation"], quantise.get("kind", ""))
    assert (
        kinds["mlp.down"] == kinds["projection.fp8.out"] == kinds["output_head"] == kinds["attention.core"] == ("target", "code")
    ), "the library's own kernels, called as the target calls them"
    table = census.table(plan, claim).splitlines()
    assert table[1].split()[-1] == "identity" and table[0].startswith("kernels, planned (decided before the load from the family's configs.py; stack native,")
    row = {line.split()[0]: line for line in table[2:15]}
    assert row["embedding"].endswith("  name  target") or row["embedding"].rstrip().endswith("target")
    assert (
        row["linear_attention.step"].split("cudnn.GdnDecodeFused")[1].split()
        == "[PDL True, PDL_RELEASE True] gdn_decode_fused 1 name and launch shape, as the entry states them target".split()
    )
    gate = "[THREADS 256, PDL_RELEASE True] gate_quant 1 name and launch shape, as the entry states them target"
    assert row["attention.gate_quantize"].split("cudnn.AttentionGateQuantDecode")[1].split() == gate.split()
    assert row["mlp.down"].split("flashinfer.gemm.mm_fp4")[1].split() == "[BACKEND cute-dsl, TUNED False] fi_mm_fp4 2 name target".split()
    assert row["attention.core"].split(ATTENTION_ENTRY)[1].split() == "[KV_LAYOUT HND] fi_decode_attention 1 name target".split()
    assert row["output_head"].split("flashinfer.gemm.mm_fp4")[1].split() == "[BACKEND cute-dsl, TUNED False] fi_mm_fp4 1 name target".split()
    quantised = "[WIDTH 5120, PDL True, PDL_RELEASE True] row_quantize_fp4 1 name and launch shape, as the entry states them"
    assert row["output_head.quantize"].split("cudnn.RowQuantizeFp4Decode")[1].split()[:-1] == quantised.split(), "the entry's whole config"
    assert row["projection.fp8.out"].split("flashinfer.gemm.bmm_fp8")[1].split() == "[BACKEND auto] cublaslt_fp8_gemm, cublas 2 name target".split()
    prepare = f"{PREPARE_FORM} attention_prepare 1 name and launch shape, as the entry states them target"
    assert row["attention.prepare"].split("cudnn.AttentionPrepareDecode")[1].split() == prepare.split(), "the kernel stated whole"
    assert row["token_choice"].rstrip().endswith("host path, not compared                          --")
    norm = "[PDL True, PDL_RELEASE True] + one of 6 forms, by call site add_norm_quant 5 name and launch shape, as the entry states them target"
    assert row["add_norm_quantize"].split("cudnn.AddNormQuantDecode")[1].split() == norm.split()
    assert table[15:21] == [f"  add_norm_quantize at {site}:{census.form_text(fields)}" for site, fields in configs.NORM_FORMS], "the form of every call site"
    assert table[17] == "  add_norm_quantize at linear_attention: [ADD True, STORE_RESIDUAL False, BF16_OUT True, QUANT e4m3_static]"
    multiply = f"{PROJECTION_FORM}] + one of 2 forms, by call site fp8_projection 2 name and launch shape, as the entry states them target"
    assert row["projection.fp8.in"].split("cudnn.Fp8ProjectionDecode")[1].split() == multiply.split()
    assert table[21:23] == [f"  projection.fp8.in at {site}:{census.form_text(fields)}" for site, fields in configs.PROJECTION_SITES], "and the weight's shape"
    fused = f"{SWIGLU_FORM} swiglu_fp4 2 name and launch shape, as the entry states them target"
    assert row["mlp.gate_up"].split("cudnn.SwigluFp4Decode")[1].split() == fused.split(), "the kernel stated whole, the token tile of the row count in it"
    assert table[-2].startswith("  per step: 20 launches on 12 exact rows") and table[-1:] == [f"  {line}" for line in claim.coverage()]
    assert "identity" not in census.table(_plan(options={"device": "cpu"})[0]).splitlines()[1], "a stack without a target has no such column"

    # fe-entries: the cuDNN Frontend rows of native, so the same identities there; the three blocks a library serves are not yet
    other, model, _ = _target_plan(2, "fe-entries")
    there = target.claim(other, model, installed=RECORDED)
    assert there.coverage()[1].startswith(
        f"kernel identity, as planned: {two_layers[0] - 6} of {two_layers[0]} launches of a step are target kernels ({two_layers[1] - 6} the target's own "
        f"compiled kernels + {two_layers[2]} cuDNN Frontend entries with byte-equal results); not yet: projection.fp8.out 2, mlp.down 2, "
    ), "the set less the six launches of the four library rows: two out-projections, two down multiplies, the attention core, the head"
    for block, mine, theirs in zip(claim.blocks, claim.identities, there.identities):
        assert (theirs == mine) != (block in configs.LIBRARY_ROWS), block
        assert block not in configs.LIBRARY_ROWS or (theirs.relation, theirs.kernel, theirs.kind) == ("not yet", mine.kernel, "code"), block
    assert census.table(other, there).splitlines()[-3].startswith("  per step: 14 launches on 8 exact rows")
    # a row selected with --provider: the fused step in its fe-cublas form is the same kernel in another form
    selected, model, _ = _target_plan(2, overrides={"linear_attention.step": "fe.gdn_decode_fused"})
    differs = target.claim(selected, model, installed=RECORDED)
    step = differs.identities[3]
    assert step.text() == "not the target (the row's form is [PDL False, PDL_RELEASE False], the target's [PDL True, PDL_RELEASE True])" and step.launch is None
    assert differs.numbers()["not_the_target"] == {"linear_attention.step": 1} and "; not the target: linear_attention.step 1; " in differs.coverage()[1]
    assert 3 not in differs.recorded and len(differs.recorded) == 15, "no recorded launch shape is held against a call that is not the target's"
    # compute capability 12.x: the gate and the norm entry do not serve there, the blocks stay on their reference chains, and the census has no section
    plan12, model, _ = _target_plan(2, cc=(12, 0), sms=170)
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
    assert there.numbers()["not_yet"]["mlp.gate_up"] == 2 and there.numbers()["not_yet"]["attention.prepare"] == 1


def test_every_reason_a_class_is_not_claimed():
    """The census line says ``NOT CLAIMED`` with every reason that applies, in a fixed order: a row selected with
    ``--provider``, fallback rows, a launch shape recorded with a width this model does not have, a kernel of the target the
    plan has no block for, no recorded section for the GPU class and SM count (both are named) or for the row count,
    versions other than the recorded ones."""
    from model_samples import census

    plan, model, target = _target_plan()
    claim = target.claim(plan, model, installed=RECORDED)
    fallback, absent = "fallback rows present", "the plan has no block for output_head.quantize"
    assert claim.reasons == [OTHER_VOCAB] and claim.name == CENSUS and (claim.basis, claim.claimed, claim.rows) == ("recorded", False, 1)
    assert claim.line(census.PASS) == f"{CENSUS}: NOT CLAIMED ({OTHER_VOCAB}); {COMPARED_16}: each in the recorded one; 4 more by membership alone"
    assert claim.compared() == (16, COMPARED_16, 4), "what item 7 compares: a count of target launches is three numbers here too"
    assert claim.line(census.NOT_OBSERVED) == f"{CENSUS}: NOT CLAIMED ({OTHER_VOCAB}); the launch shapes of its target launches were not observed"
    assert claim.line(census.FAIL) == f"{CENSUS}: NOT CLAIMED ({OTHER_VOCAB})", "a failed check says nothing about the recorded shapes"
    # the cuDNN Frontend rows alone: one launch is held by membership, the head's quantise, for which the census records no launch shape
    entries = target.claim(_target_plan(stack="fe-entries")[0], model, installed=RECORDED)
    assert entries.reasons == [fallback] and entries.compared() == (13, COMPARED_13, 1)
    assert entries.line(census.PASS) == f"{CENSUS}: NOT CLAIMED ({fallback}); {COMPARED_13}: each in the recorded one; 1 more by membership alone"
    selected, model, _ = _target_plan(overrides={"linear_attention.step": "fe.gdn_decode_fused", "mlp.down": "torch_ref.linear"})
    assert target.claim(selected, model, installed=RECORDED).reasons == [
        "not the target set: linear_attention.step selected by --provider",
        "not the target set: mlp.down selected by --provider",
        fallback,
        OTHER_VOCAB,
    ]
    # a kernel of the target the plan has no block for: a model whose wiring has no block between the final norm and the head,
    # planned from a table without that block.  The census's entry is then kept as absent, and the class is not claimed for it
    from model_samples.blocks import plan as planned_from

    class Without(type(model)):
        def wiring(self):
            return tuple(call for call in super().wiring() if call[0] != "output_head.quantize")

    older = Without(model.geometry, model.reductions)
    table = {block: rows for block, rows in model.provider_table.items() if block != "output_head.quantize"}
    lacking = target.claim(
        planned_from(table, model.regimes, plan.facts, 1, 1, "native", model=older, options=dict(plan["embedding"].options)), older, installed=RECORDED
    )
    assert [block for block, _, _ in lacking.absent] == ["output_head.quantize"] and lacking.reasons == [OTHER_VOCAB, absent]
    assert len(lacking.identities) == len(older.wiring()) == len(model.wiring()) - 1 and lacking.compared() == (16, COMPARED_16, 3)
    other_sms, model, _ = _target_plan(sms=152)
    claim = target.claim(other_sms, model, installed=RECORDED)
    assert claim.reasons == ["no recorded section for cc 10.0 / 152 SMs: recorded is cc 10.0 / 148 SMs"], "both SM counts are named"
    assert (
        claim.name == "target census plain"
        and not claim.known
        and claim.recorded == {}
        and claim.line(census.PASS) == "target census plain: NOT CLAIMED (no recorded section for cc 10.0 / 152 SMs: recorded is cc 10.0 / 148 SMs)"
    )
    four, model, _ = _target_plan(sequences=4)
    claim = target.claim(four, model, installed=RECORDED)
    assert claim.reasons == ["no recorded section for 4 rows on cc 10.0 / 148 SMs: recorded is 1 row"] and claim.name == CENSUS and not claim.known
    assert claim.coverage() == ["kernel identity: not stated for this run (no recorded section for 4 rows on cc 10.0 / 148 SMs: recorded is 1 row)"]
    older = {"torch": "2.13.0+cu130", "nvidia-cutlass-dsl": "4.8.1", "triton": "3.7.1"}
    versions = "versions other than the recorded ones: PyTorch 2.13.0 (recorded with 2.14.0), CuTe DSL 4.8.1 (recorded with 4.8.0)"
    assert target.claim(plan, model, installed=older).reasons == [OTHER_VOCAB, versions], "only the packages a provider of this area imports"
    assert target.claim(plan, model, installed={"torch": None, "nvidia-cutlass-dsl": "4.8.0"}).reasons == [OTHER_VOCAB], "not installed: not compared"
    found = census.installed_versions(("torch", "no-such-distribution-of-this-test"))
    assert found["no-such-distribution-of-this-test"] is None and found["torch"].split("+")[0] == torch.__version__.split("+")[0]
    assert set(target.data["validated"]["recorded"]) >= set(census.PACKAGES), "the census records a version of every package that is compared"
    dumped = target.claim(plan, model, installed=RECORDED).dump()
    assert dumped["claimed"] is False and dumped["not_claimed"] == [OTHER_VOCAB] and dumped["numbers"]["launches"] == 20
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
    assert claimed_wrongly(tampered, "fe-entries") == [
        ("plain", "cc10.0/sm148", "1", ["fallback rows present"])
    ], "the rule sees a census that claims too early"
    # ... and the file itself cannot say claimed for this class yet: the head's quantise is a port whose launch shape is its own,
    # its entry records none, and the reader refuses a claimed section for a target kernel without one
    early = copy.deepcopy(real.data)
    early["configurations"]["plain"]["cc10.0/sm148"]["claimed"] = True
    with pytest.raises(ValueError, match=r"claimed without the launch shape of output_head.quantize"):
        census.Target(early)

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
        (
            "kernel identity, as planned: 2 of 4 launches of a step are target kernels (2 the target's own compiled kernels + 0 cuDNN Frontend entries "
            "with byte-equal results); not yet: add_norm_quantize 1, projection.fp8.out 1; 0 declared stand-ins"
        ),
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
    inside its band, and the attention core, the library's, in the band of its row; both first uses of the library are
    announced once; the census line reads ``NOT CLAIMED (...)`` and, where the launches hold, says that the target
    launches have the recorded shape.  A target launch in another shape than the recorded one fails the run; a wrong
    byte of the gate fails its block, and so does a wrong element of the attention core; kernel binaries that are not
    on the machine are a ``SKIP`` before anything is loaded; ``--provider`` is said; without a profiler the run is
    ``RAN native stack``."""
    from model_samples import tie_out

    target = tmp_path / "census.json"
    code, lines = _sample_run(monkeypatch, capsys, ["--stack", "native", "--sequences", "1", "--dump-census", str(target)])
    assert code == 0 and lines[-1] == "[00] PASS", lines[-4:]
    planned = lines.index(next(line for line in lines if line.startswith("kernels, planned (decided before the load")))
    assert lines[planned + 1].split()[-1] == "identity" and "stack native" in lines[planned]
    assert lines[planned + 15] == "  add_norm_quantize at linear_attention.first: [ADD False, STORE_RESIDUAL True, BF16_OUT True, QUANT e4m3_static]"
    assert lines[planned + 21 : planned + 23] == [
        "  projection.fp8.in at linear_attention.qkvz: [OUT_FEATURES 16384, IN_FEATURES 5120]",
        "  projection.fp8.in at full_attention.qkv: [OUT_FEATURES 14336, IN_FEATURES 5120]",
    ], "the weight's shape at each call site of the in-projection"
    whole = "kernel identity, as planned: 20 launches per step = 8 the target's own compiled kernels + 12 cuDNN Frontend entries with byte-equal results"
    assert lines[planned + 24].strip() == whole + " + 0 declared stand-ins", "one line: the stack launches the whole set of its step"
    assert not any(line.strip().startswith(WHOLE) or "not yet" in line for line in lines[planned : planned + 26]), "no share of a set, no block left"
    (kernels,) = _kernel_lines(lines)
    assert kernels.endswith("    PASS") and "20 on exact rows, in the wiring order" in kernels
    # the head's launch shape was recorded with the published vocabulary: on another one it is held by its name alone, and the line says so;
    # the head's quantise is the fourth launch held by membership: the census records no launch shape for it
    tiny_vocab = "no recorded launch shape of output_head for this model: vocab 272 (recorded 248320)"
    not_claimed = f"    {CENSUS}: NOT CLAIMED ({tiny_vocab})"
    assert not_claimed + f"; {COMPARED_16}: each in the recorded one; 4 more by membership alone" in lines
    # the head's quantise on its entry: judged against the quantisation at the kernel's rounding points, every call inside the band
    quantised = next(line for line in lines if line.strip().startswith("output_head.quantize ") and "calls" in line)
    assert quantised.endswith("; 0 outside  [band: row_quantize_fp4]  PASS") and " of them not the twin's byte; " in quantised, quantised
    assert lines.count(f"    band row_quantize_fp4: {tie_out.BANDS['row_quantize_fp4']}") == 1
    # the attention core on the library: both first uses announced before the tables, the block in the band its row names,
    # the kernel in the class and the shape the census records for this maximum length, and its name printed
    first_uses = [line for line in lines if line.startswith("[00] planning: first use of ")]
    assert [line.split(" FlashInfer ")[0] for line in first_uses] == [
        "[00] planning: first use of",
        "[00] planning: first use of the paged decode attention of",
    ]
    assert all(lines.index(line) < planned and "FlashInfer 0.7.2.post1" in line for line in first_uses) and "FLASHINFER_NO_DOWNLOAD" in first_uses[1]
    core = next(line for line in lines if line.strip().startswith("attention.core ") and "calls" in line)
    assert core.endswith("the cache untouched  [band: attention_bf16_weights]  PASS") and "; 0 outside their bound; the largest distance " in core, core
    assert lines.count(f"    band attention_bf16_weights: {tie_out.BANDS['attention_bf16_weights']}") == 1
    assert (
        f"    attention.core: {ATTENTION_ENTRY} [KV_LAYOUT HND], 1 launch per step (1 call x 1 per call); fi_decode_attention by name, "
        f"and in the kernel class {ATTENTION_CLASS} and the launch shape the census records: grid (5, 4, 1), block (512, 1, 1), shared memory 178,456 B"
    ) in lines
    launched = [line for line in lines if line.startswith("    launched by the library for ")]
    assert len(launched) == 1 and launched[0].startswith(f"    launched by the library for attention.core (printed, never gated): {FI_ATTENTION} x "), launched
    assert launched[0].endswith(" over the traced steps")
    gate = next(line for line in lines if line.strip().startswith("attention.gate_quantize ") and "calls" in line)
    assert gate.endswith("[band: gate_fp8]  PASS") and "0 the code of neither" in gate, gate
    norm = next(line for line in lines if line.strip().startswith("add_norm_quantize ") and "calls" in line)
    assert norm.endswith("[band: add_norm_quant]  PASS") and "residual stream equal" in norm and norm.split("; ")[-1].startswith("0 outside"), norm
    assert any(line.startswith("  blocks            11 of 11 provider blocks inside their bands") and line.endswith("PASS") for line in lines)
    prepared = next(line for line in lines if line.strip().startswith("attention.prepare ") and "calls" in line)
    assert prepared.endswith("[band: attention_prepare]  PASS") and "; 0 query and 0 key elements outside their band; " in prepared, prepared
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
    assert written["stack"] == "native" and run["target"]["not_claimed"] == [tiny_vocab] and run["target"]["numbers"]["target_port"] == 12
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
    assert [site["identity"] for site in run["sites"] if site["block"] in ("embedding", "attention.gate_quantize", "mlp.down", "attention.core")] == [
        "target",
        "target",
        "target",
        "target",
        "target",
    ]
    (core_site,) = [site for site in run["sites"] if site["block"] == "attention.core"]
    assert (core_site["instance"], core_site["served"], core_site["stated"]) == (ATTENTION_ENTRY, {"KV_LAYOUT": "HND"}, None)
    assert core_site["served"] == core_site["form"] and [kernel["id"] for kernel in core_site["kernels"]] == ["fi_decode_attention"]
    (prepare,) = [site for site in run["sites"] if site["block"] == "attention.prepare"]
    assert (prepare["instance"], prepare["identity"], prepare["stated"]["launch"]) == (
        "cudnn.AttentionPrepareDecode #1",
        "target",
        [[1, 28, 1], [128, 1, 1], 16],
    )
    assert prepare["served"] == prepare["form"] and prepare["form"]["SECTION"] == [11, 11, 10] and prepare["form"]["POSITIONS"] == "3d"

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

    # one wrong element of the attention core per call: its block fails in its band, the kernel check holds
    code, lines = _sample_run(monkeypatch, capsys, ["--stack", "native", "--sequences", "1"], attention_defect="element")
    core = next(line for line in lines if line.strip().startswith("attention.core ") and "calls" in line)
    assert code == 1 and lines[-1] == "[00] FAIL" and core.endswith("[band: attention_bf16_weights]  FAIL") and "; 0 outside their bound; " not in core, core
    assert _kernel_lines(lines)[0].endswith("    PASS")

    # the head's quantise.  One code two steps off its own, per call: outside its bracket, and the block fails by its own line
    # while the kernel check holds.  And the scale handed to the entry un-inverted -- the consumer's static scale where the
    # kernel takes its reciprocal: other block scales and other codes, the same line
    code, lines = _sample_run(monkeypatch, capsys, ["--stack", "native", "--sequences", "1"], quantize_defect="code")
    quantised = next(line for line in lines if line.strip().startswith("output_head.quantize ") and "calls" in line)
    assert code == 1 and lines[-1] == "[00] FAIL" and quantised.endswith("[band: row_quantize_fp4]  FAIL") and "; 0 outside" not in quantised, quantised
    assert _kernel_lines(lines)[0].endswith("    PASS")
    from model_samples.providers import fe

    with monkeypatch.context() as wrong:
        wrong.setattr(fe.RowQuantizeFp4DecodeBlock, "_scale", lambda self, value, device: torch.tensor([float(value)], dtype=torch.float32, device=device))
        code, lines = _sample_run(monkeypatch, capsys, ["--stack", "native", "--sequences", "1"])
    quantised = next(line for line in lines if line.strip().startswith("output_head.quantize ") and "calls" in line)
    assert (
        code == 1 and lines[-1] == "[00] FAIL" and quantised.endswith("[band: row_quantize_fp4]  FAIL") and " 0 of them not the twin's byte" not in quantised
    ), quantised

    # kernel binaries that are not on the machine as their manifest lists them -- no manifest; the binary the library would
    # launch, in whose place it would launch another kernel and print a path -- are a SKIP before anything is loaded,
    # in the sample's own words: the library is not called, no line follows, no path is printed
    for defect, binaries, sentence in (
        ({"manifest": False}, (None, False), f"{ATTENTION_ENTRY} needs the kernel-binary package flashinfer-cubin 0.7.2.post1, which is not installed: "),
        (
            {"without": [FI_ATTENTION]},
            (VALIDATED_LIBRARY, True),
            f"{ATTENTION_ENTRY}: the installed flashinfer-cubin 0.7.2.post1 does not hold the kernel binaries the library's table names for this call's "
            "data types, head width, paging and page on this GPU's class ",
        ),
    ):
        directory = _kernel_binaries(tmp_path / f"binaries_{len(defect)}_{binaries[1]}", **defect)
        code, lines = _sample_run(monkeypatch, capsys, ["--stack", "native", "--sequences", "1"], kernel_binaries=directory, library_binaries=binaries)
        assert code == 0 and lines[-1].startswith(f"[00] SKIP {sentence}") and "/a/local/path" not in "\n".join(lines), lines[-1]
        assert ATTENTION_CALLS == [], "the library's attention was not called"
        assert lines[-1].endswith("--stack fe-entries runs the cuDNN Frontend rows of this stack without the package") and not any(
            line.startswith(("checks", "[00] stage")) for line in lines
        ), "nothing was loaded and nothing checked"

    # --provider: the fused step in its fe-cublas form.  The run holds; it is not the target set, and the line says so first
    selected = ["--stack", "native", "--sequences", "1", "--provider", "linear_attention.step=fe.gdn_decode_fused"]
    code, lines = _sample_run(monkeypatch, capsys, selected)
    assert code == 0 and lines[-1] == "[00] PASS", lines[-4:]
    assert (
        f"    {CENSUS}: NOT CLAIMED (not the target set: linear_attention.step selected by --provider; {tiny_vocab}); "
        "15 launches of a step compared by launch shape (4 the target's own compiled kernels + 11 cuDNN Frontend entries "
        "with byte-equal results + 0 declared stand-ins): each in the recorded one; 4 more by membership alone"
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
    assert "output_head.quantize equal (gated)" in invariance, "and the row of the head's quantise"

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
        and "kernel identity, as planned: 20 launches per step = 8 the target's own compiled kernels + 12 cuDNN Frontend entries with byte-equal results + 0 declared stand-ins"
        in out
    )


def test_a_dry_plan_of_the_native_stack_imports_nothing_new():
    """In a fresh interpreter: planning the ``native`` stack on stated device facts, reading the census and stating the claim
    add only ``model_samples`` and standard-library modules -- the versions the census compares are read from the
    distributions' metadata, no package is imported for them."""
    code = f"""
import sys
import cudnn, torch
cudnn.GdnDecodeFused, cudnn.AttentionGateQuantDecode, cudnn.AddNormQuantDecode, cudnn.SwigluFp4Decode, cudnn.AttentionPrepareDecode
import importlib.metadata
before = set(sys.modules)
sys.path.insert(0, {str(MODELS)!r})
from model_samples import census
from model_samples.blocks import DeviceFacts, plan
from model_samples.qwen_dense import QwenDense
model = QwenDense.reduced(2)
planned = plan(model.provider_table, model.regimes, DeviceFacts((10, 0), "stated", 148, 0, 0), 1, 1, "native", model=model, options={{"check_kernels": True, "max_model_len": 2051}})
claim = model.target().claim(planned, model)
table = census.table(planned, claim)
added = sorted({{name.split(".")[0] for name in set(sys.modules) - before}})
print("ADDED", [name for name in added if name not in sys.stdlib_module_names and name not in ("model_samples", "cudnn")])
print("TABLE", "gate_quant" in table, "add_norm_quant" in table, "fp8_projection" in table and "swiglu_fp4" in table and "attention_prepare" in table and "row_quantize_fp4" in table and "20 launches per step = 8 the target's own compiled kernels + 12 cuDNN Frontend entries with byte-equal results + 0 declared stand-ins" in table, claim.reasons[0])
print("LIBRARY", "fi_mm_fp4" in table and "cublaslt_fp8_gemm" in table, sorted(name for name in sys.modules if name.split(".")[0] in ("flashinfer", "tvm_ffi", "triton")))
print("NOTE", planned.announcements())
"""
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "ADDED []" in done.stdout and f"TABLE True True True {OTHER_VOCAB}" in done.stdout, done.stdout + done.stderr
    # the library rows are planned and their package is not imported; a plan on stated device facts makes no first use of
    # the library and announces none
    assert "LIBRARY True []" in done.stdout and "NOTE []" in done.stdout, done.stdout + done.stderr


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
    assert "338 of the 484 launches of a step of the 64-layer model, 48 the target's own compiled kernels + 290 cuDNN Frontend entries" in area
    assert "322 of the 484" not in area, "the five entries are 321 launches; the 322nd is PyTorch's gather"
    assert "(339 = 49 + 290 with PyTorch's gather" in area and "on compute capability 10.0, 338 of the 484" in area, "said for the class it holds on"
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
    widths = ("hidden", "mlp", "la_key_heads", "la_value_heads", "la_head_dim", "attn_heads", "attn_kv_heads", "attn_head_dim", "rotary_dim", "rotary_section")
    recorded = {name: list(value) if isinstance(value, list) else value for name, value in section["geometry"].items()}
    assert recorded == {
        name: list(getattr(full, name)) if name == "rotary_section" else getattr(full, name) for name in widths
    }, "the widths every recorded launch shape of a layer depends on, and the rotary widths the form of the attention preparation names"
    reduced = QwenDense.reduced(2).geometry
    assert all(getattr(reduced, name) == getattr(full, name) for name in widths) and reduced.vocab != full.vocab, "the reduced model keeps the widths"
    plan, model, _ = _target_plan(2)
    assert target.claim(plan, model, installed=RECORDED).known and len(target.claim(plan, model, installed=RECORDED).recorded) == 16

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
    assert any(f"{CENSUS}: NOT CLAIMED (" in line and COMPARED_16 in line for line in held.lines), "and the census line with what was compared"
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
        assert refused(misspelt) == "linear_attention.step states lauch; an entry states launch and at and form and class and pick and run"
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


def test_a_launch_shape_recorded_at_a_fact_of_the_run_is_held_for_a_run_that_states_it(monkeypatch):
    """An entry of a census section can name the facts of the run its launch shape was recorded with (``run``: the maximum
    length a run is started with).  A session at that maximum length is held to the entry's launch shape and to its
    kernel class.  At another one the call is still the target's and nothing of it is held by shape: the launch counts
    under membership, and the reason names the block and both lengths, as a fact of the run.  A plan that states no
    maximum length shares none.  The reader refuses a fact that is no object of whole numbers and a name it does not
    know; and the family's file documents every fact its reader knows, and states this one where its record does."""
    import copy

    from model_samples import Session, TextIO, blocks, census
    from model_samples.qwen_dense import QwenDense

    family = QwenDense.target()
    about = " ".join(family.data["about"])
    assert [fact for fact in census.ENTRY_FACTS if f"'{fact}'" not in about] == [], "the file documents every fact of an entry"
    stating = set()
    for configuration, sections in family.data["configurations"].items():
        for key, section in sections.items():
            for rows, parts in section["rows"].items():
                for block, _, facts in family.sequence(parts, QwenDense().geometry.layer_kinds):
                    if "run" in facts:
                        stating.add((configuration, block))
                        assert facts["run"] == {"max_model_len": section["recorded"]["max_model_len"]} and "launch" in facts, (configuration, key, rows, block)
    assert stating == {(census.PLAIN, "attention.core"), (census.VERIFY, "attention.core")}, "the launch shapes that follow the maximum length"

    # the family's census with the fact on an entry that a row of the stack serves today: the down multiply, a class and a launch shape
    data = copy.deepcopy(family.data)
    section = data["configurations"]["plain"]["cc10.0/sm148"]
    recorded = section["recorded"]["max_model_len"]
    down = [facts for part in ("linear_attention_layer", "attention_layer") for block, _, facts in section["rows"]["1"][part] if block == "mlp.down"]
    assert len(down) == 2 and all(facts["class"] and facts["launch"] for facts in down)
    for facts in down:
        facts["run"] = {"max_model_len": recorded}
    target = census.Target(data)
    model = QwenDense.reduced(2)
    text = TextIO.synthetic(model.geometry.vocab)
    monkeypatch.setattr(blocks.DeviceFacts, "probe", classmethod(lambda cls, device: cls((10, 0), "stated device", 148, 0, 0)))

    def claimed(census_file, **session):
        plan = Session(model, text, stack="native", device="cuda:0", capacity=58, **session).resolve()
        return census_file.claim(plan, model, installed=RECORDED)

    as_it_is = claimed(family, max_model_len=recorded)
    held = claimed(target, max_model_len=recorded)
    calls = [index for index, block in enumerate(held.blocks) if block == "mlp.down"]
    assert len(calls) == 2 and all(held.classes[index] == down[0]["class"] and index in held.recorded for index in calls), "shape and class are held"
    assert (held.reasons, held.recorded, held.classes, held.compared()) == (as_it_is.reasons, as_it_is.recorded, as_it_is.classes, as_it_is.compared())
    # the attention core's row is the one whose entry names the fact in the family's own file: held at the recorded length, with its class
    core = held.blocks.index("attention.core")
    assert core in as_it_is.recorded and as_it_is.classes[core] == ATTENTION_CLASS and as_it_is.identities[core].relation == census.TARGET
    other = claimed(target, max_model_len=recorded + 1)
    said = f"no recorded launch shape of mlp.down for this run: max_model_len {recorded + 1} (recorded {recorded})"
    said_core = said.replace("mlp.down", "attention.core")
    assert other.reasons.count(said) == 1 and sorted(other.reasons) == sorted([*held.reasons, said, said_core]), "said once for its two calls"
    assert [other.identities[index].relation for index in (*calls, core)] == [
        census.TARGET
    ] * 3 and other.numbers() == held.numbers(), "the target's call, as it was"
    assert not any(index in other.recorded or index in other.classes for index in (*calls, core)), "neither the launch shape nor the class is held"
    assert (other.compared()[0], other.compared()[2]) == (held.compared()[0] - 3, held.compared()[2] + 3), "the three launches count under membership"
    assert said in other.line(census.PASS) and f"{held.compared()[2] + 3} more by membership alone" in other.line(census.PASS)
    real = claimed(family, max_model_len=recorded + 1)  # the family's file as it is: the attention core's launch alone
    assert sorted(real.reasons) == sorted([*as_it_is.reasons, said_core]) and real.compared()[2] == as_it_is.compared()[2] + 1
    unnamed = claimed(target)  # a session that names no maximum length: its capacity
    assert f"no recorded launch shape of mlp.down for this run: max_model_len 58 (recorded {recorded})" in unnamed.reasons
    plan, _, _ = _target_plan(2, max_model_len=None)  # a plan that states none
    unstated = target.claim(plan, model, installed=RECORDED)
    assert not any(index in unstated.recorded or index in unstated.classes for index in calls)
    assert f"no recorded launch shape of mlp.down for this run: max_model_len not stated (recorded {recorded})" in unstated.reasons
    # an entry recorded with a width this model does not share AND with the fact: both reasons are said, each in its own words
    doubled = copy.deepcopy(family.data)
    (head,) = [facts for block, _, facts in doubled["configurations"]["plain"]["cc10.0/sm148"]["rows"]["1"]["after_step"] if block == "output_head"]
    head["run"] = {"max_model_len": recorded}
    reasons = claimed(census.Target(doubled), max_model_len=recorded + 1).reasons
    widths = [reason for reason in other.reasons if reason.startswith("no recorded launch shape of output_head for this model: ")]
    assert len(widths) == 1 and widths[0] in reasons, "the reason of the model's widths, as it was"
    assert f"no recorded launch shape of output_head for this run: max_model_len {recorded + 1} (recorded {recorded})" in reasons

    def refused(change, where="layer"):
        wrong = _small_census(claimed=False)
        change(wrong["configurations"]["plain"]["cc10.0/sm148"]["rows"]["1"])
        with pytest.raises(ValueError) as error:
            census.Target(wrong)
        assert str(error.value).startswith(f"kernel_census.json: plain / cc10.0/sm148, 1 rows, {where}"), str(error.value)
        return str(error.value)

    for bad in ({"max_model_len": "2051"}, {"max_model_len": 2051.0}, {"max_model_len": True}, {"max_model_len": 0}, {"max_model_length": 2051}, [2051], 2051):
        assert "run of linear_attention.step" in refused(lambda rows: rows["layer"][0][2].update(run=bad)), bad
        assert "run of linear_attention.step" in refused(lambda rows: rows.update(first_layer={"linear_attention.step": {"run": bad}}), "first_layer"), bad
    assert "linear_attention.step states runs" in refused(lambda rows: rows["layer"][0][2].update(runs={"max_model_len": 2051}))
    stated = _small_census()
    stated["configurations"]["plain"]["cc10.0/sm148"]["rows"]["1"]["layer"][0][2]["run"] = {"max_model_len": 2051}
    assert census.Target(stated), "the fact as the reader takes it"


def test_a_run_fact_is_asked_of_the_part_that_serves_the_call_site(monkeypatch):
    """A block whose calls several rows serve by call site hands each part its own options, and the holder of the parts
    carries none.  The fact of the run that a census entry names is asked of the part that serves the call's site: at
    the recorded maximum length the launch shape of that call is held as the census without the fact holds it, at
    another one it counts under membership and the reason names the block and both lengths."""
    import copy

    from model_samples import Session, TextIO, blocks, census
    from model_samples.qwen_dense import QwenDense

    family = QwenDense.target()
    data = copy.deepcopy(family.data)
    section = data["configurations"][census.VERIFY]["cc10.0/sm148"]
    recorded, tokens = section["recorded"]["max_model_len"], section["tokens"]
    stated = [facts for block, _, facts in section["rows"][str(tokens)]["linear_attention_layer"] if block == "projection.fp8.out" and "launch" in facts]
    assert len(stated) == 1, "one call of the block in a linear-attention layer records a launch shape"
    stated[0]["run"] = {"max_model_len": recorded}
    target = census.Target(data)
    model = QwenDense.reduced(2)
    text = TextIO.synthetic(model.geometry.vocab)
    monkeypatch.setattr(blocks.DeviceFacts, "probe", classmethod(lambda cls, device: cls((10, 0), "stated device", 148, 0, 0)))

    def claimed(census_file, length):
        plan = Session(model, text, stack="native", device="cuda:0", capacity=62, draft_tokens=tokens - 1, max_model_len=length).resolve()
        holder = plan["projection.fp8.out"]
        assert len(holder.parts) > 1 and not hasattr(holder, "options"), "several rows serve the block's calls, and the holder carries no options"
        return census_file.claim(plan, model, configuration=census.VERIFY, installed=RECORDED)

    without, held, other = claimed(family, recorded), claimed(target, recorded), claimed(target, recorded + 1)
    assert (held.recorded, held.reasons, held.compared()) == (without.recorded, without.reasons, without.compared()), "held at the recorded length"
    (call,) = [index for index, block in enumerate(held.blocks) if block == "projection.fp8.out" and index in held.recorded]
    core = held.blocks.index("attention.core")  # its entry names the fact in the family's own file: its one call goes with it
    assert call not in other.recorded and core in held.recorded and core not in other.recorded
    assert other.compared()[0] == held.compared()[0] - 2 and other.compared()[2] == held.compared()[2] + 2
    assert any("projection.fp8.out" in reason and f"{recorded + 1}" in reason and f"{recorded}" in reason for reason in other.reasons)


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
    prepared = []  # the attention preparation's entry: its first use has a case of its own (the test of its adapter), its log here is its own
    monkeypatch.setattr(cudnn, "AttentionPrepareDecode", _prepare_entry(prepared), raising=False)
    quantize_decline = [""]
    monkeypatch.setattr(cudnn, "RowQuantizeFp4Decode", _row_quantize_entry(calls, decline=quantize_decline), raising=False)
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
        ("row quantize compile", None),  # the head's quantise: its build takes no operand and no row count
    ], "the first use of each entry, at the run's row count"
    assert [kind for kind, *_ in prepared] == ["prepare compile", "prepare launch"], "the attention preparation: its build, then one call on scratch operands"
    table.first_use()
    assert len(calls) == 10 and len(prepared) == 2 and table.first_used, "made once per plan"
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
    del calls[:]
    table["output_head.quantize"].compile(4, 1)
    assert [call[0] for call in calls] == ["row quantize compile", "row quantize launch"] and calls[0][1] is None
    assert {name: tuple(operand.shape) for name, operand in calls[1][1].items()} == {
        "x": (4, 5120),
        "scale": (1,),
        "codes": (4, 2560),
        "scales": (128, 320),  # one band of the 128 x 4 layout for the rows of the run
    }, "and for the head's quantise: the build again without an operand, then one launch on scratch rows into outputs of the call's own"
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
    # (2c) the row quantise entry asks the same of the installation on compute capability 10.7.  Its build takes no operand: its
    # own sentence is the SKIP where that requirement is unmet; with the CUDA 13.4 flavour, or on another class, the same
    # decline is no answer about the installation this adapter can name -- an ERROR
    quantize_flavour = "the FP4 row quantization decode step on compute capability 10.7 requires a CuTe DSL whose compiler is CUDA 13.4 or later"
    quantize_decline[0] = quantize_flavour
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
            quantize_flavour
            if outcome is NotImplementedError
            else f"cudnn.RowQuantizeFp4Decode declined the operands this adapter built for its first use: {quantize_flavour}"
        )
        assert str(declined.value) == own, (compiler, cc)
    quantize_decline[0] = ""
    monkeypatch.setattr(dsl_version, "CUDA_VERSION", SimpleNamespace(major=12, minor=9))
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
    ``main()``, and nothing was built for it -- also a run of many sequences whose state would fit and whose checks
    would not.  A toolchain an entry declines stays a SKIP before the weights, also for a
    session that is loaded without its table.  The library rows of ``native`` make their first use in the same place:
    behind the memory sentence, in the order of the blocks."""
    import cudnn
    import cudnn.frost

    from model_samples import Session, TextIO, blocks
    from model_samples.providers import fe, flashinfer_calls
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

    class Prepare(Entry):
        name = "cudnn.AttentionPrepareDecode"

        def __call__(self, *operands):  # its first use makes one call behind the build
            events.append(("launch", self.name))

    class RowQuantize(Entry):
        name = "cudnn.RowQuantizeFp4Decode"

    def scratch(block, sequences, tokens):  # in place of the adapters' scratch operands: what would be allocated, and when
        events.append(("scratch", block.entry, sequences * tokens))
        return block._build(block.options["device"]), ()

    def library_first_use(block, sequences, tokens):  # in place of a library row's first use: which call would be made, and when
        events.append(("first use", block.entry, sequences * tokens))

    sentence = blocks.Plan.memory

    def memory(plan, *parts):  # the need, what it covers, and the two parts of it that are the checks'
        events.append(("memory sentence", plan.stack))
        return sentence(plan, *parts)

    def no_weights(model, *arguments, **options):
        raise AssertionError("reached a load")

    gpu = {"total": 180 * GIB, "free": 1 * GIB}  # a real device, as it reports itself
    monkeypatch.setattr(cudnn, "GdnDecodeFused", Step, raising=False)
    monkeypatch.setattr(cudnn, "AttentionGateQuantDecode", Gate, raising=False)
    monkeypatch.setattr(cudnn, "AddNormQuantDecode", Norm, raising=False)
    monkeypatch.setattr(cudnn, "Fp8ProjectionDecode", Projection, raising=False)
    monkeypatch.setattr(cudnn, "SwigluFp4Decode", Swiglu, raising=False)
    monkeypatch.setattr(cudnn, "AttentionPrepareDecode", Prepare, raising=False)
    monkeypatch.setattr(cudnn, "RowQuantizeFp4Decode", RowQuantize, raising=False)
    monkeypatch.setattr(fe, "_ENTRIES", {})
    monkeypatch.setattr(fe, "_FIRST_LAUNCH", {})
    monkeypatch.setattr(fe.GdnDecodeFusedBlock, "_scratch", scratch)
    monkeypatch.setattr(fe.AttentionGateQuantDecodeBlock, "_scratch", scratch)
    monkeypatch.setattr(fe.AttentionPrepareDecodeBlock, "_scratch", scratch)
    monkeypatch.setattr(flashinfer_calls, "installed", lambda: VALIDATED_LIBRARY)  # the library's package as the validated version: its rows plan
    monkeypatch.setattr(flashinfer_calls, "binaries_package", lambda: (VALIDATED_LIBRARY, True))  # ... and its kernel-binary package beside it
    monkeypatch.delenv(flashinfer_calls.BINARIES_DIRECTORY, raising=False)
    monkeypatch.setattr(flashinfer_calls.LibraryBlock, "first_use", library_first_use)
    monkeypatch.setattr(blocks.Plan, "memory", memory)
    monkeypatch.setattr(blocks.DeviceFacts, "probe", classmethod(lambda cls, device: cls((10, 0), "stated device", 148, gpu["total"], gpu["free"])))
    monkeypatch.setattr(QwenDense, "load", no_weights)
    model = QwenDense.reduced(2)
    text = TextIO.synthetic(model.geometry.vocab)
    state, conv = 65536 * 48 * 128 * 128 * 4, 65536 * 3 * 10240 * 2
    assert state == 192 * GIB and model.state_bytes(65535, 256) > state + conv, "the scratch state of a first use at 65,535 rows: one slot pool of the run"
    assert configs.PROVIDERS["linear_attention.step"][0].sequences == (1, 65535), "the row count the fused step's rows admit"

    def run(stack, *counts, **request):  # ``capacity=58``: the positions of the sample's default request
        del events[:]
        first = Session(model, text, sequences=counts[0], stack=stack, device="cuda:0", **request)
        later = [Session(model, text, sequences=count, stack=stack, device="cuda:0", share=first, **request) for count in counts[1:]]
        return first, first.plan_table(later)

    def norm_builds(stack):
        """The first use of the norm entry and of the FP8 projection entry in a stack that has their rows: one build per form
        the two-layer model calls (four) and one per weight shape (two), in front of the other entries' -- and no scratch
        operand: their builds take none."""
        return [] if stack == "fe-cublas" else [("build", "cudnn.AddNormQuantDecode")] * 4 + [("build", "cudnn.Fp8ProjectionDecode")] * 2

    def first_uses(stack, *counts):
        """In the order of the blocks: the norm and the projection entry, the fused step on scratch operands, then -- in a stack
        that has their rows -- the fused gate / up entry (its build takes the row count and no operand), the attention
        preparation entry (its build takes no operand: the scratch operands are made behind it, for its one call), the
        gate entry and the entry of the head's quantise (its build takes no operand either); and in the stack ``native``
        the library's four calls, each at the place of its block: the out-projection, the down multiply, the attention
        core, the output head."""
        events = []
        for rows in counts:
            events += [*norm_builds(stack), ("scratch", "cudnn.GdnDecodeFused", rows), ("build", "cudnn.GdnDecodeFused")]
            library = (lambda entry: [("first use", entry, rows)]) if stack == "native" else (lambda entry: [])
            events += library("flashinfer.gemm.bmm_fp8")
            if stack != "fe-cublas":
                events += [("build", "cudnn.SwigluFp4Decode"), *library("flashinfer.gemm.mm_fp4")]
                prepare = "cudnn.AttentionPrepareDecode"
                events += [("build", prepare), ("scratch", prepare, rows), ("launch", prepare), *library(ATTENTION_ENTRY)]
                events += [("scratch", "cudnn.AttentionGateQuantDecode", rows), ("build", "cudnn.AttentionGateQuantDecode")]
                events += [("build", "cudnn.RowQuantizeFp4Decode")]
            events += library("flashinfer.gemm.mm_fp4")
        return events

    target = "SM100 requires a CuTe DSL build supporting sm_100a; found 4.8.0 without that target"
    assert configs.PROVIDER_STACKS == ("fe-cublas", "native", "fe-entries")
    for stack in configs.PROVIDER_STACKS:
        # (1) the run needs more than the GPU has in total: the estimate, and not one scratch operand
        gpu.update(total=180 * GIB, free=1 * GIB)
        with pytest.raises(NotImplementedError) as skipped:
            run(stack, 65535)
        most = {"native": MOST_NATIVE}.get(stack, MOST)
        assert str(skipped.value) == (
            f"stack {stack} needs about {most}; this GPU has 180.0 GiB in total: the run does not fit on it (CUDA_VISIBLE_DEVICES selects another GPU)"
        )
        assert events == [("memory sentence", stack)], f"{stack}: nothing is built for a run that does not fit"
        # (2) the default run where too little is free: the same, for both sequence counts of the run.  A stack whose in-projection
        # and whose gate / up multiply are on the stored weights at both counts keeps those and no bf16 copy of them (0.15 GiB
        # of 8-bit codes and 0.19 GiB of 4-bit codes where the copies are 0.29 and 0.66 GiB, for the two-layer model); the
        # reference of its checks then makes its copy of one such weight at a time (0.33 GiB the largest: a gate / up weight),
        # and the sentence says that part beside what the checks hold for the three sequences beyond the first (0.14 GiB)
        what, gib = f" for 1 and 4 sequences, one count after the other, 0.14 GiB of it for the checks ({BEYOND})", "3.9"
        if stack != "fe-cublas":
            what = f" for 1 and 4 sequences, one count after the other, 0.47 GiB of it for the checks ({COPY}, and {BEYOND})"
            gib = "3.7"  # native keeps the out-projections, the down multiplies and the head on their stored codes too, and holds the library's workspace
        few = "^" + re.escape(f"stack {stack} needs about {gib} GiB on this GPU{what}; 1.0 GiB are free: free memory on this GPU")
        with pytest.raises(NotImplementedError, match=few):
            run(stack, 1, 4, capacity=58)
        assert events == [("memory sentence", stack)], stack
        # ... and a run of 1,024 sequences on a GPU with 30 GiB free: its state is 3.3 GiB and would fit, what its checks hold
        # for 1,023 sequences beyond the first does not -- declined here, and not by an allocation inside the checks
        gpu.update(free=30 * GIB)
        many = {"fe-cublas": "72.8", "native": "73.1"}.get(stack, "72.9")  # the weights each stack holds at that count; on native the library's workspace
        with pytest.raises(NotImplementedError) as skipped:
            run(stack, 1024, capacity=58)
        assert str(skipped.value).startswith(f"stack {stack} needs about {many} GiB on this GPU, ") and str(skipped.value).endswith(
            "what they hold for the 1023 sequences beyond the first); 30.0 GiB are free: free memory on this GPU, or select another one with CUDA_VISIBLE_DEVICES"
        ), str(skipped.value)
        assert events == [("memory sentence", stack)], f"{stack}: nothing is built for a run whose checks do not fit"
        gpu.update(free=1 * GIB)
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
        first, table = run(stack, 1, 4, capacity=58)
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
        ], "the decline of the first entry that builds ends the plan: nothing behind it is built, a library row included"
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

    # (5) the library's package on such a device: missing, or another version than the validated one.  A SKIP that names the
    # version, the command and the stack that runs without the package -- before the memory sentence, nothing built, nothing
    # loaded; the sample's last line is that sentence and its exit code 0
    fix = "pip install -c samples/models/constraints-native.txt -r samples/models/requirements-flashinfer.txt"
    without = "--stack fe-entries runs the cuDNN Frontend rows of this stack without the package"
    validated = "a validated flashinfer-python (0.7.2.post1: the launch records of its kernels are data of that version)"
    for version, declined in (
        (None, f"flashinfer.gemm.bmm_fp8 needs flashinfer-python 0.7.2.post1, which is not installed: {fix}; {without}"),
        ("0.7.3", f"flashinfer.gemm.bmm_fp8 needs {validated}; this is 0.7.3: {fix}; {without}"),
    ):
        monkeypatch.setattr(flashinfer_calls, "installed", lambda version=version: version)
        del events[:]
        with pytest.raises(NotImplementedError) as skipped:
            run("native", 1, 4)
        assert str(skipped.value) == declined and events == [], "the first library row of the table names what is missing: nothing was asked of the GPU"
        with monkeypatch.context() as scoped:
            code, out = _sample_main(scoped, capsys, ["--stack", "native"], total=180 * GIB, free=80 * GIB)
        assert code == 0 and out.splitlines()[-1] == f"[00] SKIP {declined}" and "first use of FlashInfer" not in out, out
        del events[:]
        _, table = run("fe-entries", 1, 4)
        assert "FlashInfer" not in table and events == [
            ("memory sentence", "fe-entries"),
            *first_uses("fe-entries", 1, 4),
        ], "the stack the sentence names plans"
        # on stated device facts the plan of native needs no package: the table states the rows (the adapter's own first use
        # makes no call there: the test of the first use of a library row)
        gpu.update(total=0, free=80 * GIB)
        _, table = run("native", 1, 4)
        assert table.count("none; unverified in CI") == 4, table
        gpu.update(total=180 * GIB, free=80 * GIB)
    # ... and with the validated version a sample announces the library's first use while it plans, in one line: the plain
    # script, and the speculative script, whose plans hold the library's rows three times (one sentence, said once)
    monkeypatch.setattr(flashinfer_calls, "installed", lambda: VALIDATED_LIBRARY)
    monkeypatch.setattr(Step, "decline", target)
    monkeypatch.setattr(cudnn.frost, "cutedsl_runtime_available", lambda cc=None: False)
    for script in ("00_decode.py", "02_speculative_decode.py"):
        for stack, said in (("native", 1), ("fe-entries", 0)):
            with monkeypatch.context() as scoped:
                code, out = _sample_main(scoped, capsys, ["--stack", stack], total=180 * GIB, free=80 * GIB, script=script)
            line = f"[{script[:2]}] planning: first use of FlashInfer 0.7.2.post1: it builds the kernels"
            assert code == 0 and sum(each.startswith(line) for each in out.splitlines()) == said, (script, stack, out)


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
    assert len(sites) == len(wiring) == 485 and census.call_sites(full) == sites
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
    at_one_token = next(r for r in out_rows if "fe-entries" in r.stacks and r.tokens[0] <= 1)
    assert (
        at_one_token.key == "cublas.bf16_dequantized" and at_one_token.sites == ()
    ), "a step of one token per sequence: the out-projection is a multiply over a copy at both call sites"
    library = next(r for r in out_rows if "native" in r.stacks and r.tokens[0] <= 1)
    assert (
        library.key == "flashinfer.bmm_fp8" and library.sites == () and library.stacks == ("native",)
    ), "... and in the stack native the library's 8-bit multiply on the stored codes, at both call sites as well"
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
    assert [site.site for site in sites if site.block not in ("add_norm_quantize", "projection.fp8.in", "projection.fp8.out")] == [""] * 228
    outs = [site for site in sites if site.block == "projection.fp8.out"]
    assert Counter(site.site for site in outs) == {"linear_attention.out": 48, "full_attention.out": 16}
    assert all(
        site.form == (("BACKEND", "auto"),) and site.facts is None and site.wrong == "" for site in outs
    ), "the library's multiply at both sites of a step of one token per sequence: a form, and no entry that states a launch"
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
    from model_samples.weights import ScaledTensor, TiledBlockScaledTensor

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
    # the final norm: the form ends at the bf16 rows and the call returns no quantised operand; a step that asks for one there
    # is refused before the launch (the consumer's operand is a block of its own)
    with pytest.raises(ValueError, match=r"its form at output_head writes no codes; the step asks for a nvfp4 operand there$"):
        block(x, residual, weight, four, "output_head")
    q4, normed4, _ = block(x, residual, weight, None, "output_head")
    head = built[2]
    assert head.arguments["bf16"] is True and head.arguments["quant"] is None and head.calls[-1][3] is None and head.calls[-1][5] is None
    assert len([call for call in head.calls if call[0] == "launch"]) == 1, "the refused call launched nothing"
    assert q4 is None and block.compared((q4, normed4, residual)) == (None, normed4, residual), "what the tie-out compares is what the call returned"
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
        plan(dict(model.provider_table), model.regimes, real, 4, 1, "fe-entries", model=model, options=options)
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
    assert target.sequence(parts, kinds) == target.sequence(parts, kinds) and target.numbers(parts, kinds) == (484, 194, 290, 0)
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
    assert numbers["not_the_target"] == {"add_norm_quantize": 64} and (numbers["target_code"], numbers["target_port"]) == (194, 226)
    assert claim.reasons == [f"not the target set: add_norm_quantize ({reason})"], "said once for its 64 calls"
    assert (
        claim.of("add_norm_quantize") == f"target; not the target ({reason})" and len(claim.recorded) == 419 - 64
    ), "no recorded shape is held against those calls"

    # a kernel whose form follows the row count: the census names the field on the entries of its row count.  A census that names
    # another token tile there -- the same kernel, the same launch shape -- makes the row "not the target" by its form
    def with_another_tile(rows):
        for kind in ("linear_attention_layer", "attention_layer"):
            rows[kind][-2][2]["form"]["TOKEN_TILE"] = 16

    claim = changed(with_another_tile).claim(plan, model, installed=RECORDED)
    everywhere = "PREWAIT_STAGES 2, L2_EVICT_FIRST True, MAX_CLUSTERS 148, PDL True"
    reason = f"the row's form is [TOKEN_TILE 8, {everywhere}], the target's [{everywhere}, TOKEN_TILE 16]"
    assert claim.numbers()["not_the_target"] == {"mlp.gate_up": 64} and claim.numbers()["target_port"] == 226 and len(claim.recorded) == 419 - 64
    assert claim.of("mlp.gate_up") == f"not the target ({reason})" and claim.reasons[0] == f"not the target set: mlp.gate_up ({reason})"
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
            "1 rows: first_layer is not an object of block: facts (launch and at and form and class and pick and run)",
        ),
        (
            lambda rows: rows[census.FIRST_LAYER]["add_norm_quantize"].update(forms={}),
            "1 rows: first_layer is not an object of block: facts (launch and at and form and class and pick and run)",
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
        session = _recorded_run(monkeypatch, sequences=sequences, stack="fe-entries", **stand_in)[0]
        session.trace(4)
        return session, tie_out.run_checks(session, prompt_tokens=3)

    def sentences(checks):
        return [line.strip() for line in checks.lines if line.strip().startswith("kernels: ")]

    def norm_line(checks):
        return next(line for line in checks.lines if line.strip().startswith("add_norm_quantize ") and "calls" in line)

    session, checks = run()
    assert checks.ok and checks.kernels == census.PASS, checks.text()
    assert "14 on exact rows, in the wiring order" in _line(checks, "kernels ") and norm_line(checks).endswith("[band: add_norm_quant]  PASS")
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
    assert checks.ok and any(f"{CENSUS}: NOT CLAIMED (" in line and COMPARED_13 in line for line in checks.lines), "held to the recorded shape, and said"

    def tampered(change):
        """The clean run at one sequence with the records of its traced pass changed before the checks read them."""
        session = _recorded_run(monkeypatch, sequences=1, stack="fe-entries")[0]
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
    # front of an MLP: its consumer is the fused gate / up kernel.  And behind the FINAL norm: its consumer is the quantise
    # of the head's operand, on its entry an exactly judged row as well.  That the adapter launches nothing but its entry is
    # also held on the adapter itself (test_the_norm_entrys_adapter_over_a_stand_in_entry: the tensors it returns are the
    # ones the entry wrote)
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
    _, checks = tampered(one_more_behind(4))
    assert (
        not checks.ok
        and checks.kernels == census.FAIL
        and sentences(checks)
        == [
            f"kernels: step 0 (and 3 more of the 4 traced steps): launched {ELEMENTWISE[:93] + '...'!r} between add_norm_quantize (layer 2) and "
            "output_head.quantize: a kernel of the fallback family pytorch, which no row there declares; nothing may sit between two exact rows"
        ]
    ), checks.text()

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
        assert not any(COMPARED_13 in line for line in checks.lines), "nothing is said to have been compared"

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
        assert _line(checks, "blocks").endswith("FAIL") and "9 of 10 provider blocks inside their bands" in _line(checks, "blocks"), defect


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
    from model_samples.weights import BlockScaledTensor, Dequantized, ScaledTensor
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
    with pytest.raises(ValueError, match=r"^cudnn.Fp8ProjectionDecode multiplies static 8-bit codes; this block is handed BlockScaledTensor$"):
        block(BlockScaledTensor(torch.zeros((2, 32), dtype=torch.uint8), torch.zeros((2, 4), dtype=torch.uint8), 1.0), weight, "linear_attention.qkvz")
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
    from model_samples.providers import flashinfer_calls, library, torch_ref
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
    # the stack native with its library rows keeps the weight of every multiply on its stored codes on compute capability 10.0,
    # and all but the in-projections on the other 10.x classes: what its plan asks for, what it holds, what a step reads
    others = FIVE[1:]
    # ... with the one workspace the library's attention is handed
    asked = [full.memory_bytes(1, 90, kept, kept) + flashinfer_calls.WORKSPACE_BYTES for kept in (FIVE, others)]
    assert (
        " ".join(f"{need / GIB:.1f}" for need in asked) == "28.2 32.5" and full.check_bytes(FIVE) == 2 * 248320 * 5120
    ), "the checks' one copy: the output head"
    for said in (
        f"it holds {(full.copy_bytes(FIVE) + full.stored_bytes(FIVE)) / GIB:.1f} GiB of weights, and its plan asks for about {asked[0] / GIB:.1f} GiB, "
        f"{full.check_bytes(FIVE) / GIB:.2f} GiB of it for the one copy its checks make at a time (the output head's is the largest)",
        f"On the other 10.x classes its plan asks for about {asked[1] / GIB:.1f} GiB; that is arithmetic as well.",
        f"weight of a multiply on its stored codes there: {(whole['copies'] + whole['state']) / 1e9:.1f} GB per step, the model's own bytes",
    ):
        assert said in readme, said
    assert f"about {asked[0] / GIB:.0f} GiB and about {asked[1] / GIB:.0f} GiB there" in script, "the help's figures for native"


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
    session = _recorded_run(monkeypatch, sequences=1, stack="fe-entries")[0]
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
    first, later = sessions("fe-cublas", "fe-entries")
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
    first, later = sessions("fe-entries", "fe-cublas")
    first.plan_table([later])
    assert first.stored_weights() == KEPT and first.uncopied_weights() == ()
    first.load()
    later.load()
    assert all(layer.in_proj.stored is not None and layer.in_proj.data is not None for layer in first.weights.layers)
    assert all(layer.gate_up.stored is not None and layer.gate_up.data is not None for layer in first.weights.layers)
    assert later.read(later.prefill([text.encode("ab0")])), "the multiplies over a copy find their copies"
    # the same stack with one row selected away in the follower: that weight's copy stays, the other's goes; with the rows in both: both go
    first, later = sessions("fe-entries", "fe-entries", overrides={"projection.fp8.in": "cublas.bf16_dequantized"})
    first.plan_table([later])
    assert first.uncopied_weights() == ("mlp.gate_up",) and first._kept() == {"stored": KEPT, "uncopied": ("mlp.gate_up",)}
    first, later = sessions("fe-entries", "fe-entries", overrides={"mlp.gate_up": "cublas.bf16_dequantized+torch_ref.silu_mul_quantize"})
    first.plan_table([later])
    assert first.uncopied_weights() == ("projection.fp8.in",) and first._kept() == {"stored": KEPT, "uncopied": ("projection.fp8.in",)}
    first.load()
    later.load()
    assert all(layer.gate_up.data is not None and layer.in_proj.data is None for layer in first.weights.layers)
    assert later.read(later.prefill([text.encode("ab0")])), "the gate / up multiply over a copy reads the copy in the checkpoint's row order"
    first, later = sessions("fe-entries", "fe-entries")
    first.plan_table([later])
    assert first.uncopied_weights() == KEPT and first._kept() == {"stored": KEPT, "uncopied": KEPT}
    first.load()
    later.load()
    assert all(layer.in_proj.data is None and layer.gate_up.data is None for layer in later.weights.layers) and later.read(later.prefill([text.encode("ab0")]))
    # a load keeps ONE stored form of a weight: two plans whose providers multiply on two layouts of it are a wrong table
    first, later = sessions("fe-entries", "fe-entries")
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
        first, later = sessions("fe-entries", "fe-entries")
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
    first, later = sessions("fe-entries", "fe-entries", overrides={"mlp.gate_up": "cublas.bf16_dequantized+torch_ref.silu_mul_quantize"})
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
    later = session("fe-entries", share=first)
    with pytest.raises(ValueError) as missing:
        later.load()
    assert (
        str(missing.value)
        == f"cudnn.Fp8ProjectionDecode multiplies on the stored format of the weight of projection.fp8.in, and the load kept none: {OTHER_PLAN}"
    )
    assert later.state is None and later.record is None, "before the first step, and before anything is compiled for it"
    # (2) weights loaded for a plan on the stored codes alone; a session on them whose block reads the copy
    first = session("fe-entries")
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
    kept = Session(tiny, text, sequences=1, stack="fe-entries", device="cpu", capacity=16, overrides={"projection.fp8.in": "cublas.bf16_dequantized"})
    plan_options = {"reference_precision": "fp64", "device": "cpu", "check_kernels": True}
    kept.plan = plan(tiny.provider_table, tiny.regimes, facts, 1, 1, "fe-entries", kept.overrides, model=tiny, options=plan_options, factories=build)
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
    plain = session("fe-entries")
    plain.stored_layouts = lambda: {}  # a load that was not told of the layout
    said = "cudnn.SwigluFp4Decode multiplies on the stored weight of mlp.gate_up in its own row order, and the load kept it in the checkpoint's"
    for made in (plain, session("fe-entries", share=plain)):
        with pytest.raises(ValueError) as missing:
            made.load()
        assert str(missing.value) == f"{said}: {OTHER_PLAN}" and made.state is None and made.weights.laid_out == ()
    assert first.weights.laid_out == ("mlp.gate_up",), "the load of the plan itself kept the layout of its provider"
    later = session("fe-entries", share=first)
    dict(later.plan.items())["mlp.gate_up"].stored_layout = None  # a block that multiplies on the merged tensor as the checkpoint has it
    with pytest.raises(ValueError) as missing:
        later.load()
    said = "cudnn.SwigluFp4Decode multiplies on the stored weight of mlp.gate_up in the checkpoint's row order, and the load kept it in a provider's own"
    assert str(missing.value) == f"{said}: {OTHER_PLAN}" and later.state is None
    # (4) planned together, nothing is missing in either direction
    first = session("fe-cublas")
    later = session("fe-entries", share=first)
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
            session = _recorded_run(scoped, sequences=sequences, stack="fe-entries", **stand_in)[0]
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
    session = _recorded_run(monkeypatch, sequences=1, stack="fe-entries")[0]
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
            session = _recorded_run(scoped, sequences=1, stack="fe-entries", **stand_in)[0]
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
    assert "10 of 10 provider blocks inside their bands" in _line(checks, "blocks") and _line(checks, "blocks").endswith("PASS")
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
    assert (
        not checks.ok and fused_line(checks).endswith("[band: swiglu_fp4]  PASS") and "10 of 10 provider blocks inside their bands" in _line(checks, "blocks")
    )
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
    session = _recorded_run(monkeypatch, sequences=1, stack="fe-entries", on_stored=True, stored_defect="writes")[0]
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
        session = _recorded_run(monkeypatch, sequences=2, stack="fe-entries", on_stored=True)[0]
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

    first = session("fe-entries")
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

    session = _recorded_run(monkeypatch, sequences=2, stack="fe-entries", on_stored=True)[0]
    model = session.model
    assert session.stored_weights() == session.uncopied_weights() == FIVE and (session.weights.stored, session.weights.uncopied) == (FIVE, FIVE)
    for block in FIVE:
        kind = ScaledTensor if model.weight_shapes(block)[0] == "fp8" else (InterleavedBlockScaledTensor if block == "mlp.gate_up" else BlockScaledTensor)
        assert all(weight.data is None and type(weight.stored) is kind for weight in _weights_of(session.weights, block)), block
    assert session.weights.copy_bytes == model.copy_bytes(FIVE) < model.copy_bytes() // 4, "what is left as bf16: the embedding table, the norms, the conv taps"
    session.trace(4)
    checks = tie_out.run_checks(session, prompt_tokens=3)
    assert checks.ok and checks.kernels == census.PASS, checks.text()
    assert _line(checks, "blocks").endswith("PASS") and "10 of 10 provider blocks inside their bands" in _line(checks, "blocks")
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
    session = _recorded_run(monkeypatch, sequences=1, stack="fe-entries", on_stored=True, stored_defect="BlockScaledTensor")[0]
    session.trace(4)
    checks = tie_out.run_checks(session, prompt_tokens=3)
    failed = [line.split()[0] for line in checks.lines if line.endswith("[band: bf16_rounding]  FAIL")]
    assert not checks.ok and checks.kernels == census.PASS and failed == ["mlp.down", "output_head"], checks.text()
    # ... and on an 8-bit weight outside the entry's row: the out-projection
    session = _recorded_run(monkeypatch, sequences=1, stack="fe-entries", on_stored=True, stored_defect="ScaledTensor")[0]
    session.trace(4)
    checks = tie_out.run_checks(session, prompt_tokens=3)
    failed = [line.split()[0] for line in checks.lines if line.endswith("[band: bf16_rounding]  FAIL")]
    assert not checks.ok and failed == ["projection.fp8.out"], checks.text()

    # the memory sentence of such a plan, on the published widths: what is held and, named, the checks' one copy
    build = _stand_ins(monkeypatch, on_stored=True)
    for model, need, checks_part in ((QwenDense.reduced(2), "3.1", "0.33"), (QwenDense(), "27.9", "2.37")):
        text = TextIO.synthetic(model.geometry.vocab)
        made = Session(model, text, sequences=1, stack="fe-entries", device="cuda:0", capacity=90)
        facts = blocks.DeviceFacts((10, 0), "stated device", 148, 180 * GIB, 80 * GIB)
        options = {"reference_precision": "fp64", "check_kernels": True}
        made.plan = blocks.plan(model.provider_table, model.regimes, facts, 1, 1, "fe-entries", model=model, options=options, factories=build)
        table = made.plan_table()
        sentence = (
            f"this stack needs about {need} GiB on this GPU, {checks_part} GiB of it for the checks (the reference's bf16 copy of one weight at a time, "
            "where the run keeps none); 80.0 GiB are free"
        )
        assert table.splitlines()[-1].strip() == sentence, table.splitlines()[-1]
        assert (
            made.model.memory_bytes(1, 90, **made._kept()) == model.memory_bytes(1, 90, FIVE, FIVE) and made.weights is None
        ), "from the plan: nothing was loaded"
    plan = blocks.Plan({}, {}, {}, blocks.DeviceFacts((10, 0), "stated device", 148, 180 * GIB, 80 * GIB), 1, 1, "fe-entries", None, "a table")
    assert plan.memory(4 * GIB) == "this stack needs about 4.0 GiB on this GPU; 80.0 GiB are free", "no part named where every weight has its copy"
    with pytest.raises(
        NotImplementedError, match=r"^stack fe-entries needs about 100.0 GiB on this GPU, 2.37 GiB of it for the checks \(.*\); 80.0 GiB are free: free memory"
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
    assert claim.of("projection.fp8.in") == "target" and claim.numbers()["target_port"] == 12
    selected, model, _ = _target_plan(2, overrides={"projection.fp8.in": "cublas.bf16_dequantized"})
    claim = target.claim(selected, model, installed=RECORDED)
    assert (
        claim.reasons[0] == "not the target set: projection.fp8.in selected by --provider"
        and claim.of("projection.fp8.in") == "not yet (bf16 copy of 8-bit weights)"
    )
    assert (
        claim.numbers()["target_port"] == 10 and claim.compared()[0] == 14
    ), "twelve ports less two in-projections; by launch shape the sixteen less those two"
    # a row that states another kernel than the census names (a ring of 7 stages): the same entry, not the target, by its form
    table = dict(configs.PROVIDERS)
    row = table["projection.fp8.in"][0]
    other = replace(row, form=tuple(("STAGES", 7) if name == "STAGES" else (name, value) for name, value in row.form))
    table["projection.fp8.in"] = [other, *table["projection.fp8.in"][1:]]
    model = QwenDense.reduced(2)
    facts = DeviceFacts((10, 0), "stated device", 148, 180 * GIB, 80 * GIB)
    planned = plan(table, model.regimes, facts, 1, 1, "fe-entries", model=model, options={"device": "cpu"})
    claim = target.claim(planned, model, installed=RECORDED)
    form = PROJECTION_FORM + ", OUT_FEATURES 16384, IN_FEATURES 5120]"
    assert claim.identities[2] == census.Identity(
        census.OTHER, "fp8_projection", "port", f"the row's form at linear_attention.qkvz is {form.replace('STAGES 8', 'STAGES 7')}, the target's {form}"
    )
    assert claim.numbers()["not_the_target"] == {"projection.fp8.in": 2} and claim.numbers()["target_port"] == 10 and len(claim.recorded) == 11
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
    session = Session(model, text, sequences=1, stack="fe-entries", device="cpu", capacity=16, merge_rule="own_scale")
    with pytest.raises(
        NotImplementedError,
        match=r"^cudnn.Fp8ProjectionDecode takes one scale per weight; the merge rule 'own_scale' keeps one per weight row: load with the merge rule "
        r'"largest_scale", or select another declared row of the block with --provider$',
    ):
        session.plan_table()
    assert options[-1]["merge_rule"] == "own_scale" and session.plan is None and session.weights is None, "declined in the plan, with the session's rule"
    # the same session with the row selected away plans (a multiply over a copy applies a scale per row), and so does the default rule
    selected = Session(
        model,
        text,
        sequences=1,
        stack="fe-entries",
        device="cpu",
        capacity=16,
        merge_rule="own_scale",
        overrides={"projection.fp8.in": "cublas.bf16_dequantized"},
    )
    assert selected.resolve().rows["projection.fp8.in"].key == "cublas.bf16_dequantized" and options[-1]["merge_rule"] == "own_scale"
    default = Session(model, text, sequences=1, stack="fe-entries", device="cpu", capacity=16)
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
        session = _recorded_run(monkeypatch, sequences=sequences, stack="fe-entries", **stand_in)[0]
        session.trace(4)
        return session, tie_out.run_checks(session, prompt_tokens=3)

    def sentences(checks):
        return [line.strip() for line in checks.lines if line.strip().startswith("kernels: ")]

    def multiply_line(checks):
        return next(line for line in checks.lines if line.strip().startswith("projection.fp8.in ") and "calls" in line)

    every = "kernels: step 0 (and 3 more of the 4 traced steps): "
    session, checks = run()
    assert checks.ok and checks.kernels == census.PASS, checks.text()
    assert "14 on exact rows, in the wiring order" in _line(checks, "kernels ") and multiply_line(checks).endswith("[band: bf16_rounding]  PASS")
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

    # (4) one launch more behind its kernel: in a linear-attention layer its consumer is the fused step, in an attention layer the
    # attention preparation's entry -- both judged one by one, so the launch fails in both layers (while that consumer was a
    # reference chain, the launch of the attention layer was counted as the chain's)
    _, checks = run(projection_defect="behind")
    assert (
        not checks.ok
        and checks.kernels == census.FAIL
        and sentences(checks)
        == [
            f"{every}{launched} projection.fp8.in (layer 0) and linear_attention.step (layer 0): {nothing}",
            f"{every}{launched} projection.fp8.in (layer 1) and attention.prepare (layer 1): {nothing}",
        ]
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

    session = _recorded_run(monkeypatch, sequences=1, stack="fe-entries")[0]
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
    assert not any(COMPARED_13 in line for line in checks.lines), "nothing is said to have been compared"

    # (7) a wrong element: the block fails in its band, the kernel check holds
    _, checks = run(projection_defect="element")
    line = multiply_line(checks)
    assert not checks.ok and checks.kernels == census.PASS and line.endswith("[band: bf16_rounding]  FAIL"), line
    assert _line(checks, "blocks").endswith("FAIL") and "9 of 10 provider blocks inside their bands" in _line(checks, "blocks")


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
        session = _recorded_run(monkeypatch, sequences=sequences, stack="fe-entries", **stand_in)[0]
        session.trace(4)
        return session, tie_out.run_checks(session, prompt_tokens=3)

    def sentences(checks):
        return [line.strip() for line in checks.lines if line.strip().startswith("kernels: ")]

    def fused_line(checks):
        return next(line for line in checks.lines if line.strip().startswith("mlp.gate_up ") and "calls" in line)

    every = "kernels: step 0 (and 3 more of the 4 traced steps): "
    session, checks = run()
    assert checks.ok and checks.kernels == census.PASS, checks.text()
    assert "14 on exact rows, in the wiring order" in _line(checks, "kernels ") and fused_line(checks).endswith("[band: swiglu_fp4]  PASS")
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
    assert checks.ok and any(f"{CENSUS}: NOT CLAIMED (" in line and COMPARED_13 in line for line in checks.lines), "held to the recorded shape, and said"

    # (6a) the launch in another shape than the entry states, field by field: the grid, the block, the shared memory
    def tampered(**fields):
        session = _recorded_run(monkeypatch, sequences=2, stack="fe-entries")[0]
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
        assert not line.split("; ")[-1].startswith("0 outside") and "9 of 10 provider blocks inside their bands" in _line(checks, "blocks"), defect


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
        assert (there["target_code"], there["target_port"]) == (194, 226) and there["not_yet"]["projection.fp8.in"] == 64, cc
    # compute capability 12.x: the multiply over a copy, with its reason; --provider cannot put the entry there
    planned, model, _ = _target_plan(2, cc=(12, 0), sms=170)
    assert planned.rows["mlp.gate_up"].key == fallback
    with pytest.raises(NotImplementedError, match=r"^cudnn.SwigluFp4Decode serves compute capability 10.x; this device is 12.0$"):
        _target_plan(2, cc=(12, 0), sms=170, overrides={"mlp.gate_up": "fe.swiglu_fp4_decode"})
    # a checkpoint of the family with another MLP width: not this row's, on any class -- the next row, and no identity stated
    wider = QwenDense(replace(QwenDense.reduced(2).geometry, mlp=20480), ("another MLP width",))
    facts = DeviceFacts((10, 0), "stated device", 148, 180 * GIB, 80 * GIB)
    planned = plan(wider.provider_table, wider.regimes, facts, 1, 1, "fe-entries", model=wider, options={"device": "cpu"})
    assert planned.rows["mlp.gate_up"].key == fallback and planned.rows["add_norm_quantize"].key == "fe.add_norm_quant_decode"
    claim = target.claim(planned, wider, installed=RECORDED)
    assert claim.coverage() == [
        "kernel identity: not stated for this run (no recorded section for this model's widths on cc 10.0 / 148 SMs: mlp 20480 (recorded 17408))"
    ]
    with pytest.raises(NotImplementedError, match=r"^cudnn.SwigluFp4Decode serves hidden 5120, mlp 17408; this model has mlp 20480$"):
        plan(wider.provider_table, wider.regimes, facts, 1, 1, "fe-entries", {"mlp.gate_up": "fe.swiglu_fp4_decode"}, model=wider, options={"device": "cpu"})
    # on the class and at 1 row: the target; selected back onto the copy with --provider: not the target set, and said
    planned, model, _ = _target_plan(2)
    claim = target.claim(planned, model, installed=RECORDED)
    assert claim.of("mlp.gate_up") == "target" and claim.numbers()["target_port"] == 12
    selected, model, _ = _target_plan(2, overrides={"mlp.gate_up": fallback})
    claim = target.claim(selected, model, installed=RECORDED)
    assert claim.reasons[0] == "not the target set: mlp.gate_up selected by --provider" and claim.of("mlp.gate_up") == "not yet (bf16 copy of 4-bit weights)"
    assert (
        claim.numbers()["target_port"] == 10 and claim.compared()[0] == 14
    ), "less the two gate / up launches, with the library's down multiplies and attention core"
    # a row that states another kernel than the census names (a cap of 152 thread blocks): the same entry, not the target, by its form
    table = dict(configs.PROVIDERS)
    rows = table["mlp.gate_up"]
    capped = replace(rows[0], form=tuple(("MAX_CLUSTERS", 152) if name == "MAX_CLUSTERS" else (name, value) for name, value in rows[0].form))
    table["mlp.gate_up"] = [capped, *rows[1:]]
    model = QwenDense.reduced(2)
    planned = plan(table, model.regimes, facts, 1, 1, "fe-entries", model=model, options={"device": "cpu"})
    claim = target.claim(planned, model, installed=RECORDED)
    everywhere = "PREWAIT_STAGES 2, L2_EVICT_FIRST True, MAX_CLUSTERS 148, PDL True"
    reason = f"the row's form is {SWIGLU_FORM.replace('MAX_CLUSTERS 148', 'MAX_CLUSTERS 152')}, the target's [{everywhere}, TOKEN_TILE 8]"
    assert claim.identities[claim.blocks.index("mlp.gate_up")] == census.Identity(census.OTHER, "swiglu_fp4", "port", reason)
    assert claim.numbers()["not_the_target"] == {"mlp.gate_up": 2} and claim.numbers()["target_port"] == 10 and len(claim.recorded) == 11
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
COMPARED_14 = (
    "14 launches of a step compared by launch shape (2 the target's own compiled kernels + 12 cuDNN Frontend entries with byte-equal results "
    "+ 0 declared stand-ins)"
)


def test_the_out_projection_row_of_a_verify_step_through_a_session_and_what_fails_it(monkeypatch):
    """The row of the FP8 projection entry for the out-projection of a verify step, through real sessions on the CPU: a
    decode session and a verify session of one sequence x six tokens on one load, on the stand-ins of the entries (the
    projection's multiplies on the stored codes the loader kept).  Clean, the verify run holds: the linear-attention
    layer's call is the entry's, judged against the reference multiply under its own name and served by the instance of
    its row; the attention layer's call is the multiply over a copy; 14 launches of a step are judged one by one, each
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
    assert "15 on exact rows, in the wiring order" in _line(checks, "kernels ")
    assert judged(checks, LA_OUT).endswith("[band: bf16_rounding]  PASS") and judged(checks, FA_OUT).endswith("[band: bf16_rounding]  PASS")
    assert not any(
        line.split()[:1] == ["projection.fp8.out"] and line.split()[1].isdigit() for line in checks.lines
    ), "each call site is judged under its own name"
    assert "11 of 11 provider blocks inside their bands" in _line(checks, "blocks")
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
        "kernel identity, as planned: 15 of 20 launches of a step are target kernels (2 the target's own compiled kernels + 13 cuDNN Frontend entries "
        "with byte-equal results); not yet: mlp.down 2, attention.core 1, projection.fp8.out 1, output_head 1; 0 declared stand-ins"
    )
    assert _line(checks, CENSUS_VERIFY).endswith(f"{COMPARED_14}: each in the recorded one; 1 more by membership alone")
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
    assert "14 on exact rows, in the wiring order" in _line(plain, "kernels ") and _line(plain, CENSUS).endswith(
        f"{COMPARED_13}: each in the recorded one; 1 more by membership alone"
    )

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
    assert not any(COMPARED_14 in line for line in checks.lines), "nothing is said to have been compared"

    # (8) a wrong element, and (9) another weight of the same shape under the row's name: the block fails in its band against
    # the reference multiply, the kernel check holds
    for defect in ("element", "weight"):
        _, _, checks = run(out_defect=defect)
        line = judged(checks, LA_OUT)
        assert not checks.ok and checks.kernels == census.PASS and line.endswith("[band: bf16_rounding]  FAIL"), line
        assert _line(checks, "blocks").endswith("FAIL") and "10 of 11 provider blocks inside their bands" in _line(checks, "blocks")
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


def _draft_run(monkeypatch, drafted=5, steps=3, stack="fe-entries", verify=True, drafter=False, total=0, max_model_len=RECORDED_LENGTH, **stand_in):
    """A session built with ``draft_tokens=`` on the stand-in providers and the stand-in profiler of :func:`_stand_ins`,
    behind a plain session on the same load: a prompt through ``prefill``, then ``steps`` verify steps of seeded ids, each
    keeping another count (``verify=False``: the prompt alone).  Returns the session, its run recorded and no traced pass
    made yet.  The ids are those of seed 1: with some ids the fp32 chains of the stand-ins leave the logits band on this
    tiny geometry, which says nothing about the session; what a test of a defect asserts is that defect's own line.
    ``drafter``: the session is handed the seeded drafter of the tiny model, whose plan is resolved on the same stand-ins
    (its embedding lookup writes its gather into the log; its chains launch nothing there), and every step is the step
    ``speculate`` runs, on seeded drafted tokens (``force=``; the ids of seed 0, for the reason above: a window of such a
    session starts with the session's own greedy token, and behind the ids of seed 1 the stand-ins leave the band in
    every stack).  ``total``: the memory of the device the plans are resolved for; 0 is a plan on stated device facts,
    which asks nothing of an installation."""
    from model_samples import Session, TextIO, blocks, census

    from model_samples.qwen_dense import QwenDense, QwenDenseDrafter

    stand_in.setdefault("gate_shape", lambda rows: ((3 * rows, 1, 1), (256, 1, 1), 0))
    build = _stand_ins(monkeypatch, **stand_in)
    monkeypatch.setattr(census, "installed_versions", lambda packages: dict(RECORDED))
    model = _tiny_model()
    monkeypatch.setattr(QwenDense, "_target", _census_of(model))
    text = TextIO.synthetic(model.geometry.vocab)
    facts = blocks.DeviceFacts((10, 0), "stated device", 148, total, 80 * GIB if total else 0)
    tokens = drafted + 1

    def plan(per_sequence, **more):
        options = {"reference_precision": "fp64", "device": "cpu", "check_kernels": True, "max_model_len": max_model_len, **more}
        return blocks.plan(model.provider_table, model.regimes, facts, 1, per_sequence, stack, model=model, options=options, factories=build)

    common = dict(sequences=1, stack=stack, device="cpu", capacity=40, check_kernels=True, max_model_len=max_model_len)
    first = Session(model, text, **common)
    session = Session(model, text, share=first, draft_tokens=drafted, drafter=QwenDenseDrafter.seeded(model) if drafter else None, **common)
    first.plan, session.plan, session.prefill_plan = plan(1), plan(tokens), plan(1, state_tokens=tokens)
    if drafter:  # the drafter's own table, at one row per sequence, on the same stand-ins
        options = {"reference_precision": "fp64", "device": "cpu", "check_kernels": True}
        its = session.drafter
        session.draft_plan = blocks.plan(its.provider_table, its.regimes, facts, 1, 1, stack, model=its, options=options, factories=build)
    first._followers = (session,)  # as ``plan_table([session])`` leaves it: the load keeps what all three plans read
    first.load()
    session.load()
    session.read(session.prefill([text.encode("abc")]))
    generator = torch.Generator().manual_seed(0 if drafter else 1)
    for step in range(steps if verify else 0):
        ids = torch.randint(0, model.geometry.vocab, (1, tokens), generator=generator, dtype=torch.int32)
        if drafter:  # the window starts with the session's own greedy token; the drafted tokens behind it are these
            ((window, greedy),) = session.read(session.speculate(ids[:, 1:].contiguous()))
            assert window[1:] == ids[0, 1:].tolist() and len(greedy) == tokens
        else:
            greedy = session.read(session.verify(ids))
            assert len(greedy[0]) == tokens
        session.accept(step % tokens)
    return session


def test_a_traced_pass_of_engine_steps_replays_the_prompt_untraced(monkeypatch):
    """The traced pass of a session that prefills and verifies behind its prompt: the prompt is replayed from a zero state
    in front of the profiler and ended by its host read, the traced window holds the verify steps alone -- one call log
    per traced step, the records of the verify plan's launches -- and the kernel check holds them against the verify
    plan, saying beside its verdict that the prompt's decode steps were not observed.  A launch more inside a verify
    step fails it; a session that has recorded no verify step says why it has no traced pass."""
    from model_samples import census, tie_out

    session = _draft_run(monkeypatch)
    position, counts = session.state.position, list(session.record.accepted)
    assert counts == [0, 1, 2] and session.prompt_record.count == 3
    from model_samples import loop

    fed, order, feed, window = [], [], type(session)._feed, loop.trace_steps

    class Chosen:  # the ids a prompt step chose, as the traced pass is handed them: a host read of them is seen
        def __init__(self, ids):
            self.ids = ids

        def tolist(self):
            order.append(f"host read behind prompt token {len(fed)}")
            return self.ids.tolist()

    monkeypatch.setattr(type(session), "_feed", lambda self, ids: Chosen(feed(self, ids)) if fed.append(int(ids[0])) is None else None)
    monkeypatch.setattr(loop, "trace_steps", lambda *operands, **more: window(*operands, **more) if order.append("the traced window") is None else None)
    session.trace(8)
    assert fed == session.prompt_record.prompt[0].tolist() and len(fed) == 3, "the prompt was replayed in front of the traced steps, token by token"
    # ... and ended as the loop ends a prompt, by the one host read of its last ids: nothing of it is in flight when the window opens
    assert order == ["host read behind prompt token 3", "the traced window"], order
    traced = session.traced
    assert len(traced.steps) == len(traced.call_log) == 3, "three verify steps were recorded: one call log and one cut of the records per traced step"
    assert all(len(calls) == 21 and {call.rows for call in calls} == {6} for calls in traced.call_log), "the calls of a verify step, at its six rows"
    names = [[record.name for record in step] for step in traced.steps]
    assert all(sum(name == VERIFY_SYMBOL for name in step) == 1 for step in names), "the fused verify step of the linear-attention layer"
    assert not any("gdn_decode_fused" in name for step in names for name in step), "no decode step of the prompt is in the traced window"
    assert all(step[-1] == HOST_READ and step.count(HOST_READ) == 1 for step in names), "a traced step ends in its one host read"
    assert (session.state.position, session.record.accepted, session.prompt_record.count) == (position, counts, 3), "the run's state and records are put back"
    assert traced.rows == 6 and traced.claim is not None and census.configuration_of(session.plan) == "verify"
    report = next(line for line in session.report().text().splitlines() if line.startswith("traced pass (3)"))
    assert report.endswith(" | behind the prompt's 3 decode steps, replayed untraced")
    checks = tie_out.run_checks(session, prompt_tokens=3)
    assert checks.ok and checks.kernels == census.PASS, checks.text()
    assert (
        checks.lines[0]
        == "checks (1 sequence x 6 tokens per sequence, 3 verify steps; tokens kept behind the first: 0, 1, 2; behind a prompt of 3 decode steps)"
    )
    blocks_lines = [line for line in checks.lines if line.startswith("  blocks ")]
    assert len(blocks_lines) == 2 and all(line.endswith("    PASS") for line in blocks_lines), "the stand-ins of both plans are judged against the reference"
    # the kernel check looked at the verify steps: beside its verdict the run says what it did not look at
    unseen = "    the prompt's decode steps: 3 replayed in front of the traced window, their kernels not observed (the check is of the verify steps)"
    assert checks.lines[[line.startswith("  kernels ") for line in checks.lines].index(True) + 1 :].count(unseen) == 1, checks.text()
    # the block check of the prompt covers the steps it was asked for: one call of the fused decode step where all three have three
    step = next(line for line in checks.lines if line.split()[:1] == ["linear_attention.step"])
    assert "   3 calls " in step and "[band: gdn_step]  PASS" in step, step
    some = _draft_run(monkeypatch)
    some.trace(8)
    first = tie_out.run_checks(some, prompt_tokens=3, max_prompt_steps=1)
    assert first.ok and first.lines[0].endswith("behind a prompt of 3 decode steps, the first 1 of them judged)")
    assert "   1 calls " in next(line for line in first.lines if line.split()[:1] == ["linear_attention.step"])
    # ... and a prompt of which no step is judged says that, not that its provider blocks are the reference chain
    unjudged = _draft_run(monkeypatch)
    unjudged.trace(8)
    nothing = tie_out.run_checks(unjudged, prompt_tokens=3, max_prompt_steps=0)
    assert nothing.ok and "  blocks            the prompt's decode steps: no step of this kind was judged    REPORTED" in nothing.lines
    assert not any("no provider block to judge" in line for line in nothing.lines), nothing.text()
    # a launch more inside a verify step: the kernel check of the verify plan fails with its sentence, in every traced step
    wrong = _draft_run(monkeypatch, projection_defect="extra")
    wrong.trace(8)
    checks = tie_out.run_checks(wrong, prompt_tokens=3)
    kernels = [line.strip() for line in checks.lines if line.strip().startswith("kernels: ")]
    assert not checks.ok and checks.kernels == census.FAIL and kernels, checks.text()
    assert all("which no row there declares; nothing may sit between two exact rows" in line for line in kernels), kernels
    assert (
        kernels[0].startswith("kernels: step 0 (and 2 more of the 3 traced steps): launched ")
        and "add_norm_quantize (layer 0) and projection.fp8.in (layer 0)" in kernels[0]
    )
    # a session that has fed its prompt and verified nothing: why there is no traced pass, and the kernels are not observed
    idle = _draft_run(monkeypatch, verify=False)
    idle.trace(8)
    why = "needs a recorded verify step behind the prompt on a CUDA device"
    assert idle.traced.steps is None and idle.traced.missing == why and idle.report().trace_missing == why
    status, unobserved, lines = tie_out.kernel_check(idle)
    assert (status, unobserved) == (census.NOT_OBSERVED, why) and lines[0].endswith(f"not observed: {why}    {census.NOT_OBSERVED}")


# ---------------------------------------------------------------- the speculative script on a target stack, to its last line
def _speculative_run(monkeypatch, capsys, arguments, **stand_in):
    """``main()`` of ``02_speculative_decode.py`` to its LAST line, on the CPU: stated device facts (compute capability
    10.0), the tiny geometry with its seeded drafter, and the stand-in providers and stand-in profiler of
    :func:`_stand_ins` under every plan of the run -- the plain plan, the verify plan, the prompt's decode plan and the
    drafter's (its lookup is the stand-in of PyTorch's gather, which writes a launch into the log; its chains are the
    reference's).  Returns ``(exit code, the lines the run printed, what it wrote to standard output)``."""
    import importlib.util

    from model_samples import blocks, census
    from model_samples.qwen_dense import QwenDense

    model_samples = importlib.import_module("model_samples")  # the package as an object: its names are patched below
    stand_in.setdefault("gate_shape", lambda rows: ((3 * rows, 1, 1), (256, 1, 1), 0))  # three rows per thread block, as the tiny census records it
    build = _stand_ins(monkeypatch, **stand_in)
    monkeypatch.setattr(census, "installed_versions", lambda packages: dict(RECORDED))
    stated = blocks.DeviceFacts((10, 0), "stated device", 148, 180 * GIB, 80 * GIB)
    real_session = getattr(model_samples.Session, "real", model_samples.Session)  # of a second run of one test: not the first run's wrapper

    def session(model, text, **options):  # a session of the script on the CPU, every one of its plans on the stated facts and the stand-ins
        made = real_session(model, text, **dict(options, device="cpu"))

        def plan(of, tokens, overrides, **more):
            plan_options = {"reference_precision": made.reference_precision, "device": "cpu", "check_kernels": made.check_kernels, **more}
            plan_options["max_model_len"] = made.max_model_len
            table, regimes = of.provider_table, of.regimes
            return blocks.plan(
                table, regimes, stated, 1, tokens, made.stack, overrides, model=of, options=plan_options, source=of.table_source, factories=build
            )

        made.plan = plan(model, made.tokens, made.overrides)
        if made.draft_tokens is not None:
            made.prefill_plan = plan(model, 1, made.overrides, state_tokens=made.tokens)
        if made.drafter is not None:
            made.draft_plan = plan(made.drafter, 1, {})
        return made

    session.real = real_session
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (80 * GIB, 180 * GIB))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device=None: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device=None: 0)
    monkeypatch.setattr(blocks.DeviceFacts, "probe", classmethod(lambda cls, device: stated))
    monkeypatch.setattr(model_samples, "check_environment", lambda: model_samples.Environment("1.33.0", "1.33.0", True, "0123456789ab", "2.13.0", "4.8.0"))
    monkeypatch.setattr(model_samples, "Session", session)
    monkeypatch.setattr(QwenDense, "reduced", classmethod(lambda cls, layers=2, vocab=15520: _tiny_model()))
    monkeypatch.setattr(QwenDense, "_target", _census_of(_tiny_model()))
    paths = list(sys.path)
    spec = importlib.util.spec_from_file_location(
        "model_sample_qwen_dense_02_speculative_decode_census_run", MODELS / "qwen_dense" / "02_speculative_decode.py"
    )
    module = importlib.util.module_from_spec(spec)
    capsys.readouterr()
    try:
        spec.loader.exec_module(module)
        code = module.main(["--force-tokens", "24", *arguments])
    finally:
        sys.path[:] = paths
    captured = capsys.readouterr()
    return code, captured.err.splitlines(), captured.out


def test_the_speculative_sample_on_a_target_stack_to_its_last_line(monkeypatch, capsys):
    """``main()`` of ``02_speculative_decode.py`` with ``--stack native``, on stand-in providers and the stand-in profiler,
    to its last line.  The kernel check runs without a flag, on whole steps -- the lookups of the drafter's passes and
    the verify plan's rows, one by one --, and under it the run says what it states of the drafter's kernels: nothing.
    Launch records that hold: ``[02] PASS``.  A launch more between two exact rows of the verify step: ``[02] FAIL``,
    exit code 1.  The profiler out: ``[02] RAN native stack (kernels not observed: <reason>)``, never ``PASS``.  A count
    of drafted tokens the census holds no section for states no identity; the default count states the verify plan's
    three numbers.  Standard output stays empty: a seeded run has no language."""
    drafter = "    kernel identity of the drafter's passes: not stated (the kernel census holds the verify step and the output head behind it; "
    drafter += "no section holds a draft pass)"
    unseen = "    the prompt's decode steps: 26 replayed in front of the traced window, their kernels not observed (the check is of the verify steps)"
    three = [
        "--stack",
        "native",
        "--force-tokens",
        "11",
        "--speculative-config",
        '{"method": "mtp", "num_speculative_tokens": 3, "use_local_argmax_reduction": true}',
    ]
    code, lines, out = _speculative_run(monkeypatch, capsys, three)
    assert (code, lines[-1], out) == (0, "[02] PASS", ""), lines[-6:]
    (kernels,) = _kernel_lines(lines)
    # ... 13 of them the cuDNN Frontend rows' and the lookup's, five the library rows': two down multiplies, the attention layer's
    # out-projection, the attention core, the head
    assert kernels.endswith("    PASS") and "23 on exact rows, in the wiring order" in kernels, "three lookups of the passes and the verify plan's 19"
    assert lines.count(drafter) == 1 and lines.count(unseen) == 1 and lines.index(kernels.join(("  ", ""))) < lines.index(unseen) < lines.index(drafter)
    unstated = "kernel identity: not stated for this run (no recorded section for 4 tokens per sequence on cc 10.0 / 148 SMs: "
    assert any(line.strip().startswith(unstated) for line in lines), "three drafted tokens: a step shape the census holds no section for"
    # 11 new tokens at three drafted tokens: the first from the prompt, then steps that keep 1, 2, 3 and 4 tokens -- four steps, every count once
    assert "  acceptance        4 of 4 steps kept the count they were scheduled to keep; every count from 0 to 3 was kept at least once    PASS" in lines
    blocks_lines = [line for line in lines if line.startswith("  blocks  ")]
    assert len(blocks_lines) == 2 and all(line.endswith("    PASS") for line in blocks_lines), "the stand-ins of both kinds of step against the reference"
    assert lines[-2] == "[02] the drafter ran on reference blocks: none of its blocks was compared with a provider (no provider serves them yet)"
    assert not any(line.startswith("kernels, planned") and "1 sequence, cc" in line for line in lines), "the kernel table is the verify plan's alone"

    # one launch more in front of the in-projection of the verify step, between two exact rows: the kernel check fails the run
    code, lines, out = _speculative_run(monkeypatch, capsys, three, projection_defect="extra")
    assert (code, lines[-1], out) == (1, "[02] FAIL", ""), lines[-6:]
    assert _kernel_lines(lines)[0].endswith("    FAIL") and any("nothing may sit between two exact rows" in line for line in lines)
    assert lines.count(drafter) == 1, "what is not stated is said beside a failed check too"

    # the profiler out: every other line holds, the kernels were not observed -- RAN, with the stack's name and the reason
    reason = "the profiler failed (no tracing library)"
    code, lines, out = _speculative_run(monkeypatch, capsys, three, profiler=reason)
    assert (code, lines[-1], out) == (0, f"[02] RAN native stack (kernels not observed: {reason})", "") and "[02] PASS" not in lines
    assert _kernel_lines(lines) == [f"kernels           not observed: {reason}    NOT OBSERVED"]
    assert f"traced pass           not available: {reason}" in lines and lines.count(drafter) == 1

    # the default configuration on a stack without a target, with its kernels checked: five lookups of the passes and the model's
    code, lines, out = _speculative_run(monkeypatch, capsys, ["--stack", "fe-cublas", "--check-kernels"])
    assert (code, lines[-1], out) == (0, "[02] PASS", ""), lines[-6:]
    assert _kernel_lines(lines)[0].endswith("    PASS") and lines.count(drafter) == 1 and not any("kernel identity, as planned" in line for line in lines)
    code, lines, out = _speculative_run(monkeypatch, capsys, ["--stack", "fe-cublas"])  # ... and without the flag nothing is traced for a check
    assert (code, lines[-1]) == (0, "[02] PASS") and _kernel_lines(lines) == [] and drafter not in lines

    # the default configuration on native: six tokens per sequence, the step shape the census records -- the three numbers, before the load
    # and under the check, and every launch in its recorded shape
    code, lines, out = _speculative_run(monkeypatch, capsys, ["--stack", "native"])
    planned = (
        "kernel identity, as planned: 20 launches per step = 7 the target's own compiled kernels + 13 cuDNN Frontend entries with byte-equal results "
        "+ 0 declared stand-ins"
    )
    load = next(index for index, line in enumerate(lines) if line.startswith("[02] stage "))
    stated = [index for index, line in enumerate(lines) if line.strip().startswith(planned)]
    assert len(stated) == 2 and stated[0] < load < stated[1], "with the plan, and again under the kernel check"
    (kernels,) = _kernel_lines(lines)
    assert kernels.endswith("    PASS") and "25 on exact rows, in the wiring order" in kernels, "five lookups of the passes and the verify plan's 20"
    assert any(
        line.strip().startswith("target census verify / cc 10.0 / 148 SMs") and ": NOT CLAIMED (no recorded launch shape of output_head" in line
        for line in lines
    ), "no fallback row is left in the verify plan of native"
    assert lines.count(drafter) == 1 and out == ""
    # On this tiny geometry the fp32 chains of the stand-ins leave the seeded model's declared logits band on a row of a rejected draft
    # at six tokens per sequence.  The band is the model's and no stand-in widens it: the token line of the run's own checks fails,
    # every line of the comparison with decode steps holds, and a failed line of the checks ends the run in FAIL.
    failed = [line.split()[0] for line in lines if line.endswith("    FAIL")]
    tokens = [line for line in lines if line.startswith("  tokens  ")]
    assert (
        failed == ["tokens"]
        and tokens[0].endswith("    PASS")
        and "greedy tokens equal to the reference stack's" in tokens[1]
        and tokens[1].endswith("    FAIL")
    )
    assert (code, lines[-1]) == (1, "[02] FAIL")
    # ... and on fe-entries, the cuDNN Frontend rows alone: the numbers of a verify plan without the rows a library serves
    code, lines, out = _speculative_run(monkeypatch, capsys, ["--stack", "fe-entries"])
    alone = "kernel identity, as planned: 15 of 20 launches of a step are target kernels (2 the target's own compiled kernels + 13 cuDNN Frontend entries "
    alone += "with byte-equal results); not yet: mlp.down 2, attention.core 1, projection.fp8.out 1, output_head 1"
    assert sum(line.strip().startswith(alone) for line in lines) == 2 and "20 on exact rows, in the wiring order" in _kernel_lines(lines)[0]
    assert (code, lines[-1]) == (1, "[02] FAIL"), "the token line of the stand-ins, as on native"


# ----------------------------------------------------------------------------------------------- the rows a library serves
def test_the_library_rows_through_a_session_and_what_fails_them(monkeypatch):
    """The three rows of FlashInfer in the ``native`` stack through a real session on the CPU: the real adapters over a
    stand-in of the library (its multiplies decode the operands the adapters hand it, in the layout they hand them in).
    Clean, the run holds: every multiply inside the band of the reference multiply, the down multiply in the kernel class
    and the launch shape the census records, the out-projection held by membership with the kernel the library chose
    printed beside the recorded pick, every weight on its stored format alone.  Then the controls.  The 4-bit multiply
    is held exactly: another kernel class, another grid, a launch more, the kernels of a multiply over a copy -- and the
    same call inside a tuning context of the library, or after one has tuned the operation in the process -- each fail
    the kernel check with its own sentence; a wrong element fails the block in its band while the kernel check holds.
    The 8-bit multiply is the library's choice: the recorded pick, a split pick with its reduce launch, the pick in every
    other step and the library's own kernel all pass, each printed and compared; a kernel of neither family fails."""
    from model_samples import census, tie_out
    from model_samples.qwen_dense import configs

    def run(sequences=1, **stand_in):
        session = _recorded_run(monkeypatch, sequences=sequences, stack="native", **stand_in)[0]
        session.trace(4)
        return session, tie_out.run_checks(session, prompt_tokens=3)

    def sentences(checks):
        return [line.strip() for line in checks.lines if line.strip().startswith("kernels: ")]

    def chosen(checks):
        return [line.strip().split(": ", 1)[1] for line in checks.lines if line.strip().startswith("chosen by the library for projection.fp8.out")]

    every = "kernels: step 0 (and 3 more of the 4 traced steps): "
    recorded_in = "the target census plain / cc 10.0 / 148 SMs (recorded 2026-11-12)"
    rows = {block: configs.PROVIDERS[block][0] for block in ("mlp.down", "output_head")}
    assert [(row.key, row.form, row.stacks, row.ci) for row in rows.values()] == [
        ("flashinfer.mm_fp4", configs.MM_FP4_FORM, ("native",), False),
        ("flashinfer.mm_fp4_head", configs.MM_FP4_FORM, ("native",), False),
    ] and configs.MM_FP4_FORM == (("BACKEND", "cute-dsl"), ("TUNED", False)), "the rows name the library's untuned pick"

    session, checks = run()
    assert checks.ok and checks.kernels == census.PASS, checks.text()
    for block, calls in (("projection.fp8.out", 14), ("mlp.down", 14), ("output_head", 7)):
        line = next(line for line in checks.lines if line.split()[:1] == [block])
        assert f"{calls} calls  largest distance 0.00 bf16 units" in line and line.endswith("[band: bf16_rounding]  PASS"), line
    assert "20 on exact rows, in the wiring order" in _line(checks, "kernels ")
    assert (
        _line(checks, "stored weights")
        .strip()
        .startswith("stored weights    9 of 9 weights this run keeps on their stored format alone hold the bytes of their load")
    )
    held = "(projection.fp8.in, projection.fp8.out, mlp.gate_up, mlp.down, output_head)"
    assert held in _line(checks, "on demand"), "the reference's copies of every multiply's weight are made per call: the run keeps none"
    assert _line(checks, "mlp.down: ").strip() == (
        "mlp.down: flashinfer.gemm.mm_fp4 [BACKEND cute-dsl, TUNED False], 2 launches per step (2 calls x 1 per call); fi_mm_fp4 by name, and in the "
        "kernel class Sm100BlockScaledSplitKGemmKernel and the launch shape the census records: grid (40, 1, 2), block (192, 1, 1), shared memory 231,424 B"
    )
    assert _line(checks, "output_head: ").endswith("1 launch per step (1 call x 1 per call); fi_mm_fp4 by name (no launch shape is declared for it)")
    assert OTHER_VOCAB.replace("15520", "272") in _line(
        checks, "target census"
    ), "the head of another vocabulary is another weight: no shape is held, and the line says so"
    assert "4 more by membership alone" in _line(checks, "target census") and COMPARED_16 in _line(checks, "target census")
    digest = hashlib.sha256(FI_CUTLASS_FP8.encode()).hexdigest()[:12]
    named = f"{FI_CUTLASS_FP8[:69]}... (1,360 characters, sha256 {digest})"
    theirs = f"{named} x 4 over the traced steps; not the pick the census records ({NVJET_FP8}): another kernel of the same entry"
    assert chosen(checks) == [theirs, theirs], "the library's own kernel at both call sites: printed, compared, not gated"
    assert census._named(FI_CUTLASS_FP4) != census._named(FI_CUTLASS_FP8) and census._named(NVJET_FP8) == NVJET_FP8, "a long name is told from its twin"
    # what the library was handed: the backend each row names, never the library's own default, at every call of the run
    assert {call[:2] for call in LIBRARY_CALLS} == {("mm_fp4", "cute-dsl"), ("bmm_fp8", "auto")}, "the BACKEND of the rows reaches the library"
    assert {call[3] for call in LIBRARY_CALLS} == set(LIBRARY_WEIGHTS), "the weights of a step, as the loader keeps them: what a first use must be made on"
    assert configs.BMM_FP8_FORM == (("BACKEND", "auto"),), "the library's own default for the 8-bit multiply is another backend"
    log = session.traced.call_log[0]
    assert [(call.block, call.layer, call.instance, call.form) for call in log if call.block in ("projection.fp8.out", "mlp.down", "output_head")] == [
        ("projection.fp8.out", 0, "flashinfer.gemm.bmm_fp8", configs.BMM_FP8_FORM),
        ("mlp.down", 0, "flashinfer.gemm.mm_fp4", configs.MM_FP4_FORM),
        ("projection.fp8.out", 1, "flashinfer.gemm.bmm_fp8", configs.BMM_FP8_FORM),
        ("mlp.down", 1, "flashinfer.gemm.mm_fp4", configs.MM_FP4_FORM),
        ("output_head", -1, "flashinfer.gemm.mm_fp4", configs.MM_FP4_FORM),
    ], "the call log reads the form back from the call: what was passed, and how the library's tuner stood"
    # at another row count no section is recorded: the class is not held there, and the run says that no identity is stated
    _, two = run(sequences=2)
    assert two.ok and "mlp.down equal" not in _line(two, "batch invariance") and "13 of 13 blocks equal" in _line(two, "batch invariance")
    stated = "    attention.core: the library states no batch invariance for this call: "
    assert sum(line.startswith(stated) for line in two.lines) == 1, "what the library states of the attention core, once beside the batch line"
    assert _line(two, "mlp.down: ").endswith("fi_mm_fp4 by name (no launch shape is declared for it)") and chosen(two) == []

    # (1) the other kernel class of the library in the recorded launch shape; (2) the recorded class in another grid
    _, checks = run(library_defect="class")
    assert not checks.ok and checks.kernels == census.FAIL, checks.text()
    assert sentences(checks) == [
        f"{every}mlp.down (layer {layer}) launched {FI_PERSISTENT!r}; {recorded_in} has the kernel class Sm100BlockScaledSplitKGemmKernel of fi_mm_fp4 there"
        for layer in (0, 1)
    ], checks.text()
    _, checks = run(library_defect="shape")
    other = "launched fi_mm_fp4 with grid (1, 1, 40), block (192, 1, 1), shared memory 231,424 B"
    assert not checks.ok and checks.kernels == census.FAIL, checks.text()
    assert sentences(checks) == [
        f"{every}mlp.down (layer {layer}) {other}; {recorded_in} has grid (40, 1, 2), block (192, 1, 1), shared memory 231,424 B" for layer in (0, 1)
    ], checks.text()
    # (3), (4) the same call inside a tuning context, and after one has tuned the operation in this process: not the row's form
    for defect in ("tuning", "has tuned"):
        _, checks = run(library_defect=defect)
        served = "the adapter served the call with the form [BACKEND cute-dsl, TUNED True]"
        assert not checks.ok and checks.kernels == census.FAIL, defect
        assert sentences(checks) == [
            f"kernels: form: mlp.down (layer 0): {served} (flashinfer.gemm.mm_fp4); the row says [BACKEND cute-dsl, TUNED False]",
            f"kernels: form: output_head: {served} (flashinfer.gemm.mm_fp4); the row says [BACKEND cute-dsl, TUNED False]",
        ], defect
        for kept in ("replay", "blocks", "tokens"):
            assert _line(checks, kept).endswith("PASS"), "its bytes are the multiply's: only the kernel check sees a tuned call"
    # (5) a launch more in front of its kernel; (6) the kernels of a multiply over a copy under the row's name
    _, checks = run(library_defect="extra")
    nothing = "a kernel of the fallback family pytorch, which no row there declares; nothing may sit between two exact rows"
    assert not checks.ok and sentences(checks) == [
        f"{every}launched {ELEMENTWISE[:93] + '...'!r} between mlp.gate_up (layer {layer}) and mlp.down (layer {layer}): {nothing}" for layer in (0, 1)
    ], checks.text()
    _, checks = run(library_defect="chain")
    missing = "did not launch fi_mm_fp4: the step has 0 launch(es) of fi_mm_fp4, the plan 3 = 3 call(s) x 1 per call"
    assert not checks.ok and checks.kernels == census.FAIL
    assert sentences(checks) == [f"{every}mlp.down (layer 0) {missing}", f"{every}mlp.down (layer 1) {missing}", f"{every}output_head {missing}"]
    # (7) one wrong element per call: the block fails in its band, the kernel check holds
    _, checks = run(library_defect="element")
    failed = [line.split()[0] for line in checks.lines if line.endswith("[band: bf16_rounding]  FAIL")]
    assert not checks.ok and checks.kernels == census.PASS and failed == ["mlp.down", "output_head"], checks.text()

    # the 8-bit multiply: whichever of its kernels the library chose passes, and the line says which it was
    _, checks = run(library_defect="pick")
    assert checks.ok and chosen(checks) == [f"{NVJET_FP8} x 4 over the traced steps; the pick the census records"] * 2
    _, checks = run(library_defect="split")
    split = "nvjet_sm100_qqtst_64x8_64x16_2x1_v_bz_splitK_TNT x 4 over the traced steps; not the pick"
    assert checks.ok and checks.kernels == census.PASS and all(line.startswith(split) for line in chosen(checks)) and len(chosen(checks)) == 2
    _, checks = run(library_defect="mixed")  # the recorded pick in every other step, the library's own kernel between: not that pick
    mixed = [line for line in chosen(checks) if line.count(" x 2") == 2 and f"{NVJET_FP8} x 2" in line and "; not the pick" in line]
    assert checks.ok and checks.kernels == census.PASS and len(mixed) == len(chosen(checks)) == 2, chosen(checks)
    absent = "did not launch cublaslt_fp8_gemm: the step has 0 launch(es) of cublaslt_fp8_gemm, the plan 2 = 2 call(s) x 1 per call"
    # ... a kernel of neither family, and the library's CUTLASS multiply of other operands: no member, each fails by name
    for defect in ("other", "twin"):
        _, checks = run(library_defect=defect)
        assert not checks.ok and checks.kernels == census.FAIL, defect
        assert sentences(checks) == [f"{every}projection.fp8.out (layer {layer}) {absent}" for layer in (0, 1)], checks.text()

    # a step of several tokens per sequence: two rows serve the out-projection by call site.  What the census records for
    # the projection entry's site is that site's: the library's row at the attention layers' site has no recorded shape,
    # and its line says so
    verify, _ = _verify_run(monkeypatch, stack="native")
    verify.trace(3)
    checks = tie_out.run_checks(verify, prompt_tokens=0)
    assert checks.ok and checks.kernels == census.PASS, checks.text()
    assert _line(checks, "projection.fp8.out at full_attention.out: ").strip() == (
        "projection.fp8.out at full_attention.out: flashinfer.gemm.bmm_fp8 [BACKEND auto], 1 launch per step (1 call x 1 per call); "
        "cublaslt_fp8_gemm by name (no launch shape is declared for it)"
    )
    entry_line = _line(checks, "projection.fp8.out at linear_attention.out: ")
    assert "cudnn.Fp8ProjectionDecode" in entry_line and "the entry states grid (80, 2, 1), block (256, 1, 1), shared memory 103,808 B" in entry_line
    site = next(index for index, site in enumerate(verify.traced.sites) if (site.block, site.site) == ("projection.fp8.out", "full_attention.out"))
    assert site not in verify.traced.claim.recorded and site not in verify.traced.claim.classes, "the census records no launch shape for that call"
    # ... and the kernel class is looked up like the shape: were a shape recorded for the library's call and a class for the
    # block's other call site alone, the library's line would state its own shape and no class
    entry_site = next(index for index, site in enumerate(verify.traced.sites) if (site.block, site.site) == ("projection.fp8.out", "linear_attention.out"))
    recorded, classes = dict(verify.traced.claim.recorded), dict(verify.traced.claim.classes)
    with monkeypatch.context() as scoped:
        scoped.setattr(census.Claim, "recorded", property(lambda self: {**recorded, site: FP8_SHAPE}))
        scoped.setattr(census.Claim, "classes", property(lambda self: {**classes, entry_site: "TheClassOfTheOtherCallSite"}))
        stated = [line for line in verify.traced.details() if line.startswith("projection.fp8.out at full_attention.out: ")]
    assert stated == [
        "projection.fp8.out at full_attention.out: flashinfer.gemm.bmm_fp8 [BACKEND auto], 1 launch per step (1 call x 1 per call); "
        "cublaslt_fp8_gemm by name, and in the launch shape the census records: grid (1, 80, 1), block (256, 1, 1), shared memory 227,328 B"
    ]
    assert _line(checks, "mlp.down: ").endswith(
        "fi_mm_fp4 by name, and in the kernel class Sm100BlockScaledSplitKGemmKernel and the launch shape the census records: "
        "grid (40, 1, 2), block (192, 1, 1), shared memory 231,424 B"
    )

    # the census facts of a library kernel are names: a class or a pick that is not one is a file this reader refuses
    data = json.loads((MODELS / "model_samples" / "qwen_dense" / "kernel_census.json").read_text())
    layer = data["configurations"]["plain"]["cc10.0/sm148"]["rows"]["1"]["linear_attention_layer"]
    for name, block, recorded in (("class", "mlp.down", "Sm100BlockScaledSplitKGemmKernel"), ("pick", "projection.fp8.out", NVJET_FP8)):
        (facts,) = [call[2] for call in layer if len(call) > 2 and name in call[2]]  # the layer's down multiply; its out-projection
        assert facts[name] == recorded
        for facts[name] in (7, ""):  # a number, and a name of no characters
            with pytest.raises(ValueError, match=f"class and pick of {block} are names: a kernel class, a kernel"):
                census.Target(data)
        facts[name] = recorded  # the file as it is again: the other name is refused by itself
    census.Target(data)


def test_the_library_rows_under_the_passes_of_a_speculative_step(monkeypatch):
    """The rows of FlashInfer in a step of speculative decoding on ``native``: a session with a drafter on the CPU, the real
    adapters over the stand-in of the library.  The library's rows serve the prompt's decode steps and the verify step as
    they serve a session without a drafter, and the passes of the drafter call none of them.  The sites of a traced step
    are the passes' and then the model's, so what the census records of a library kernel -- the kernel class of the 4-bit
    multiply with its launch shape, the pick of the 8-bit one -- is held and printed at the MODEL's sites, behind the
    passes: the other kernel class and another grid at the down multiply each fail the kernel check by name there, and
    the kernel the library chose for the attention layers' out-projection is compared with the recorded pick of that
    call site."""
    from model_samples import census, tie_out
    from model_samples.qwen_dense import configs

    def run(**stand_in):
        session = _draft_run(monkeypatch, stack="native", drafter=True, **stand_in)
        session.trace(8)
        return session, tie_out.run_checks(session, prompt_tokens=3)

    def sentences(checks):
        return [line.strip() for line in checks.lines if line.strip().startswith("kernels: ")]

    def chosen(checks):
        return [line.strip() for line in checks.lines if line.strip().startswith("chosen by the library for ")]

    session, checks = run()
    traced, passes = session.traced, 5 * session.drafter.calls_per_pass()
    assert checks.ok and checks.kernels == census.PASS, checks.text()
    # every plan of the session made its first use before the load, and the library was called at the rows of both kinds of step
    assert session.plan.first_used and session.prefill_plan.first_used and session.draft_plan.first_used
    assert {call[:3] for call in LIBRARY_CALLS} == {("mm_fp4", "cute-dsl", 1), ("mm_fp4", "cute-dsl", 6), ("bmm_fp8", "auto", 1), ("bmm_fp8", "auto", 6)}
    assert all(row.key.startswith(("torch_ref.", "torch.")) for row in session.draft_plan.rows.values()), "no row of the drafter's table is a library's"
    # the three tables: the library's rows in the verify plan and in the prompt's decode plan, a chain in every row of the drafter's but its lookup
    tables = session._tables()
    assert tables.count("  by provider: cuDNN FE 7 | FlashInfer 4 | PyTorch 2 | unverified in CI: 4\n") == 2
    assert tables.count("  by provider: fallbacks 12 (PyTorch chains 12) | PyTorch 1\n") == 1 and tables.count("served by (") == 3
    # a step of this session reads every weight of a multiply on its stored codes: the model's own bytes, no copy among them
    moved = session.report().bytes
    assert moved["copies"] == moved["weights"] and 0 < moved["stored"] < moved["weights"], moved
    # the sites: five passes of the drafter, then the model's; the claim is the verify plan's and counts from the model's first site
    assert (traced.first, len(traced.sites)) == (passes, passes + 21) and passes == 90
    assert all(site.block.startswith("draft.") for site in traced.sites[:passes]) and traced.sites[passes].block == "embedding"
    claim = traced.claim
    split_k = "Sm100BlockScaledSplitKGemmKernel"
    assert [(index, claim.classes[index]) for index in sorted(claim.classes)] == [
        (7, split_k),
        (11, ATTENTION_CLASS),
        (16, split_k),
    ], "the two down multiplies of the model and, between them, its attention core"
    assert traced._classes() == {index + passes: named for index, named in claim.classes.items()}
    assert [traced.sites[index].block for index in sorted(traced._classes())] == ["mlp.down", "attention.core", "mlp.down"]
    assert all(traced.sites[index].block.startswith("draft.") for index in claim.classes), "uncounted, the class would stand at a site of a pass"
    assert len(claim.picks) == 1 and set(claim.picks.values()) == {NVJET_FP8}
    assert traced._picks() == {index + passes: pick for index, pick in claim.picks.items()}
    assert [(traced.sites[index].block, traced.sites[index].site) for index in traced._picks()] == [("projection.fp8.out", "full_attention.out")]
    assert all(index + passes in traced._recorded() for index in claim.classes), "a class is recorded with a launch shape, and stands where that stands"
    # the lines: the down multiply in the recorded class and shape, the out-projection's kernel beside the recorded pick of ITS call site
    assert _line(checks, "mlp.down: ").strip() == (
        "mlp.down: flashinfer.gemm.mm_fp4 [BACKEND cute-dsl, TUNED False], 2 launches per step (2 calls x 1 per call); fi_mm_fp4 by name, and in the "
        "kernel class Sm100BlockScaledSplitKGemmKernel and the launch shape the census records: grid (40, 1, 2), block (192, 1, 1), shared memory 231,424 B"
    )
    assert _line(checks, "projection.fp8.out at full_attention.out: ").strip() == (
        "projection.fp8.out at full_attention.out: flashinfer.gemm.bmm_fp8 [BACKEND auto], 1 launch per step (1 call x 1 per call); "
        "cublaslt_fp8_gemm by name (no launch shape is declared for it)"
    )
    digest = hashlib.sha256(FI_CUTLASS_FP8.encode()).hexdigest()[:12]
    assert chosen(checks) == [
        "chosen by the library for projection.fp8.out at full_attention.out (printed and compared, never gated): "
        f"{FI_CUTLASS_FP8[:69]}... (1,360 characters, sha256 {digest}) x 3 over the traced steps; "
        f"not the pick the census records ({NVJET_FP8}): another kernel of the same entry"
    ]
    assert (
        _line(checks, "draft.embedding: ")
        .strip()
        .endswith("5 launches per step (5 calls x 1 per call); embedding_gather by name (no launch shape is declared for it)")
    )
    assert "25 on exact rows, in the wiring order" in _line(checks, "kernels "), "the five lookups of the passes and the 20 of the model"
    planned = "kernel identity, as planned: 20 launches per step = 7 the target's own compiled kernels + 13 cuDNN Frontend entries"
    assert _line(checks, "kernel identity, as planned").strip().startswith(planned)
    line = _line(checks, "target census verify")
    assert "17 launches of a step compared by launch shape (5 the target's own compiled kernels + 12 cuDNN Frontend entries with byte-equal results" in line
    assert "each in the recorded one; 3 more by membership alone" in line
    assert f"    {tie_out.DRAFTER_KERNELS}" in checks.lines, "nothing is stated about the kernels of the passes"
    blocks_lines = [line for line in checks.lines if line.startswith("  blocks ")]
    assert ["11 of 11 provider blocks" in blocks_lines[0], "12 of 12 provider blocks" in blocks_lines[1]] == [True, True] and len(blocks_lines) == 2
    for block in ("projection.fp8.out at full_attention.out", "mlp.down", "output_head"):
        judged = [line for line in checks.lines if line.strip().startswith(f"{block}  ") and "[band: bf16_rounding]" in line]
        assert judged and all(line.endswith("[band: bf16_rounding]  PASS") for line in judged), block
    held = "(projection.fp8.in, projection.fp8.out, mlp.gate_up, mlp.down, output_head)"
    assert held in _line(checks, "on demand") and _line(checks, "stored weights").strip().startswith("stored weights    9 of 9 weights")
    # were a shape recorded for the library's call and a class for the block's other call site alone, the library's line would
    # state its own shape and no class: the lookup is per call site, behind the passes as without them
    library_site = next(index for index, site in enumerate(traced.sites) if (site.block, site.site) == ("projection.fp8.out", "full_attention.out"))
    entry_site = next(index for index, site in enumerate(traced.sites) if (site.block, site.site) == ("projection.fp8.out", "linear_attention.out"))
    assert library_site - passes not in claim.recorded and min(library_site, entry_site) >= passes
    recorded, classes = dict(claim.recorded), dict(claim.classes)
    with monkeypatch.context() as scoped:
        scoped.setattr(census.Claim, "recorded", property(lambda self: {**recorded, library_site - passes: FP8_SHAPE}))
        scoped.setattr(census.Claim, "classes", property(lambda self: {**classes, entry_site - passes: "TheClassOfTheOtherCallSite"}))
        stated = [line for line in traced.details() if line.startswith("projection.fp8.out at full_attention.out: ")]
    assert stated == [
        "projection.fp8.out at full_attention.out: flashinfer.gemm.bmm_fp8 [BACKEND auto], 1 launch per step (1 call x 1 per call); "
        "cublaslt_fp8_gemm by name, and in the launch shape the census records: grid (1, 80, 1), block (256, 1, 1), shared memory 227,328 B"
    ]

    # the controls, at the model's sites of the traced step: (1) the other kernel class in the recorded shape, (2) the recorded
    # class in another grid, (3) the same call after a tuning of the operation in this process
    every = "kernels: step 0 (and 2 more of the 3 traced steps): "
    recorded_in = "the target census verify / cc 10.0 / 148 SMs (recorded 2026-11-12)"
    _, checks = run(library_defect="class")
    assert not checks.ok and checks.kernels == census.FAIL, checks.text()
    assert sentences(checks) == [
        f"{every}mlp.down (layer {layer}) launched {FI_PERSISTENT!r}; {recorded_in} has the kernel class Sm100BlockScaledSplitKGemmKernel of fi_mm_fp4 there"
        for layer in (0, 1)
    ]
    _, checks = run(library_defect="shape")
    other = "launched fi_mm_fp4 with grid (1, 1, 40), block (192, 1, 1), shared memory 231,424 B"
    assert not checks.ok and sentences(checks) == [
        f"{every}mlp.down (layer {layer}) {other}; {recorded_in} has grid (40, 1, 2), block (192, 1, 1), shared memory 231,424 B" for layer in (0, 1)
    ], checks.text()
    _, checks = run(library_defect="has tuned")
    served = "the adapter served the call with the form [BACKEND cute-dsl, TUNED True]"
    assert not checks.ok and sentences(checks) == [
        f"kernels: form: mlp.down (layer 0): {served} (flashinfer.gemm.mm_fp4); the row says [BACKEND cute-dsl, TUNED False]",
        f"kernels: form: output_head: {served} (flashinfer.gemm.mm_fp4); the row says [BACKEND cute-dsl, TUNED False]",
    ], checks.text()
    # such a session needs the library's package as a plain run does: on a device its plans decline before anything is loaded,
    # and the stack of the cuDNN Frontend rows runs the same session without it
    with pytest.raises(NotImplementedError, match="needs flashinfer-python 0.7.2.post1, which is not installed: pip install -c "):
        _draft_run(monkeypatch, stack="native", drafter=True, total=180 * GIB, library_installed=None)
    without = _draft_run(monkeypatch, stack="fe-entries", drafter=True, total=180 * GIB, library_installed=None)
    assert without.record.count == 3 and LIBRARY_CALLS == [], "no row of fe-entries calls the library"
    # the 8-bit multiply stays the library's choice: the recorded pick passes and the line says that it is that pick
    _, checks = run(library_defect="pick")
    assert checks.ok and chosen(checks) == [
        "chosen by the library for projection.fp8.out at full_attention.out (printed and compared, never gated): "
        f"{NVJET_FP8} x 3 over the traced steps; the pick the census records"
    ], checks.text()


def test_the_first_use_of_a_library_row_and_the_weight_it_multiplies_on(monkeypatch):
    """A library row's adapter on its own: what it plans, when it calls the library, what it keeps of a weight.  A plan on
    stated device facts asks nothing of the installation and makes no call.  On a device, the first use is the call of the
    run's row count on scratch operands of the shapes a step multiplies on: operands the library refuses are a
    ``ValueError`` (the adapter's fault: an ERROR), a library that cannot build its kernel there is a
    ``NotImplementedError`` (a SKIP) that names the exception type and no local path.  The announcement is one sentence per
    plan, whichever of the three rows it has.  The stored 4-bit weight is kept in the layout the library reads -- the same
    bytes, the scale bytes in another order -- and a weight or an operand in another layout is refused by name."""
    from dataclasses import replace

    from model_samples import blocks
    from model_samples.providers import flashinfer_calls
    from model_samples.weights import BlockScaledTensor, Dequantized, ScaledTensor, TiledBlockScaledWeight, dequantize, quantize_nvfp4, tile_scales

    model = _tiny_model()

    def planned(stack="native", total=180 * GIB, table=None, **stand_in):
        build = _stand_ins(monkeypatch, **stand_in)
        facts = blocks.DeviceFacts((10, 0), "stated device", 148, total, 80 * GIB if total else 0)
        options = {"reference_precision": "fp64", "device": "cpu"}
        return blocks.plan(table or model.provider_table, model.regimes, facts, 1, 1, stack, model=model, options=options, factories=build)

    plan = planned()
    down, head, out = plan["mlp.down"], plan["output_head"], plan["projection.fp8.out"]
    assert [type(block).__name__ for block in (down, head, out)] == ["MmFp4Block", "MmFp4HeadBlock", "BmmFp8Block"]
    assert {block.provider for block in (down, head, out)} == {"FlashInfer"} and all(block.stored_weights for block in (down, head, out))
    note = (
        "first use of FlashInfer 0.7.2.post1: it builds the kernels of its multiplies now, in this process and with the installed CUDA compiler and "
        'ninja (minutes on the first run of a machine, nothing on a later one; the family README, "What the first run does")'
    )
    notes = plan.announcements()
    assert (
        notes[:1] == [note] and len(notes) == 2 and planned("fe-entries").announcements() == []
    ), "one sentence for the three multiplies; none where no row is the library's"
    assert notes[1].startswith(
        "first use of the paged decode attention of FlashInfer 0.7.2.post1: this sample switches the library's download of kernel binaries off for the "
        "process (FLASHINFER_NO_DOWNLOAD), so they are read from its kernel-binary package or from a directory filled beforehand and never fetched; "
        "it holds the kernel binaries of this GPU to their manifest, and the library then builds its launcher"
    ), "and one for the attention, which says what the sample does to the library's download and to its kernel binaries"
    assert "### What the first run does" in (MODELS / "qwen_dense" / "README.md").read_text(), "the section the sentence names"
    for block in (down, head, out):
        block.first_use(1, 1)
        assert block.serving() == (block.entry, ()), "a first use is no call of a step: the call log starts empty"
    assert [call[1:3] for call in LIBRARY_CALLS] == [("cute-dsl", 1), ("cute-dsl", 1), ("auto", 1)], "one call each, at the run's row count"
    assert [call[3] for call in LIBRARY_CALLS] == LIBRARY_WEIGHTS, "each on a scratch weight of the shape a step multiplies on: the kernel built is the step's"
    # a run of several sequences: the checks call the block again at the rows of ONE sequence (batch invariance), and the
    # library builds the kernel of the 4-bit multiply per range of row counts -- so the plan makes that call as well
    for sequences, tokens, rows in ((16, 1, [16, 1]), (2, 6, [12, 6]), (1, 6, [6])):
        del LIBRARY_CALLS[:]
        down.first_use(sequences, tokens)
        assert [call[2] for call in LIBRARY_CALLS] == rows and down.serving() == (down.entry, ()), (sequences, tokens)
    # operands the library refuses, with either exception of its checks: the adapter's fault; a build that fails: the installation's
    for defect in ("refuses", "mistyped"):
        with pytest.raises(ValueError) as error:
            planned(library_defect=defect)["mlp.down"].first_use(1, 1)
        assert str(error.value) == "flashinfer.gemm.mm_fp4 declined the operands this adapter built for its first use: mm_fp4 accepts 2d tensors", defect
    unbuilt = planned(library_defect="build")["output_head"]
    with pytest.raises(NotImplementedError) as skipped:
        unbuilt.first_use(1, 1)
    assert str(skipped.value) == (
        "flashinfer.gemm.mm_fp4: FlashInfer could not make its first use on this installation (RuntimeError); it builds its "
        'kernels there and needs the CUDA compiler and ninja (the family README, "What the first run does"); '
        "--stack fe-entries runs the cuDNN Frontend rows of this stack without the package"
    ), "the type of the exception, never its text: that may hold a local path"
    unimportable = planned(library_defect="import")["projection.fp8.out"]
    with pytest.raises(NotImplementedError, match=r"flashinfer.gemm.bmm_fp8: FlashInfer could not make its first use on this installation \(ImportError\); it"):
        unimportable.first_use(1, 1)
    # ... and a fault of the adapter's own code is neither: it is raised as it is, never read as the installation's (the
    # scratch operands, the operand's layout, the call's arguments, the tuner's reading -- everything but the library call)
    for method in ("_scratch", "_operand", "_tuned"):
        faulty = planned()["mlp.down"]
        with monkeypatch.context() as scoped:
            scoped.setattr(type(faulty), method, lambda self, *operands: [].missing)
            with pytest.raises(AttributeError, match="'list' object has no attribute 'missing'"):
                faulty.first_use(1, 1)
        assert faulty.serving() == (faulty.entry, ()) and not faulty._first, method
    # outside a first use the library's own exception is the caller's to see: nothing is renamed in a step
    with pytest.raises(RuntimeError, match="ninja exited with status 1"):
        unbuilt(*unbuilt._scratch(torch.device("cpu"), 1))
    # on stated device facts the same plan asks nothing: no package check, no call
    monkeypatch.setattr(flashinfer_calls, "installed", lambda: None)
    stated = planned(total=0, library_defect="refuses")
    stated["mlp.down"].first_use(1, 1)
    assert stated.announcements() == [] and LIBRARY_CALLS == [], "no first use is made there: nothing is announced, nothing called"
    assert "FlashInfer  flashinfer.gemm.mm_fp4" in stated.table() and stated.table().count("none; unverified in CI") == 4
    with pytest.raises(NotImplementedError, match="needs flashinfer-python 0.7.2.post1, which is not installed"):
        planned(library_installed=None)
    # a form the adapter cannot pass is a wrong table, whatever the device
    table = dict(model.provider_table)
    table["mlp.down"] = [replace(table["mlp.down"][0], form=(("BACKEND", "cute-dsl"),)), *table["mlp.down"][1:]]
    with pytest.raises(ValueError, match="provider row flashinfer.mm_fp4: its form does not name TUNED; this adapter's rows name BACKEND, TUNED"):
        planned(total=0, table=table)

    # the weight: the loader's 4-bit weight in the layout the library reads -- built once, the same bytes
    generator = torch.Generator().manual_seed(7)
    merged = quantize_nvfp4(torch.randn((model.geometry.hidden, model.geometry.mlp), generator=generator), 0.25)
    kept = down.stored_layout(merged)
    assert type(kept) is TiledBlockScaledWeight and kept.codes is merged.codes and kept.scale == merged.scale
    assert torch.equal(kept.tiled_scales, tile_scales(merged.block_scales)) and [id(part) for part in kept.tensors()] == [id(kept.codes), id(kept.tiled_scales)]
    back = kept.row_major()
    assert torch.equal(back.codes, merged.codes) and torch.equal(back.block_scales, merged.block_scales) and back.scale == merged.scale
    assert torch.equal(dequantize(kept).data, dequantize(merged).data), "the copy the reference reads is bit for bit the copy of the checkpoint's order"
    with pytest.raises(ValueError, match="multiplies on a stored 4-bit weight; the loader handed its layout ScaledTensor"):
        down.stored_layout(ScaledTensor(torch.zeros((2, 2), dtype=torch.uint8), 1.0))
    # ... and a call on a weight the load kept in another form, or on an operand without codes, is refused by name
    qx = quantize_nvfp4(torch.randn((1, model.geometry.mlp), generator=generator), 1.0)
    product = down(qx, Dequantized(None, 1.0, kept))
    exact = (dequantize(qx).data.float() @ dequantize(merged).data.float().T * (qx.scale * merged.scale)).to(torch.bfloat16)
    assert product.dtype == torch.bfloat16 and torch.equal(product, exact)
    assert down.serving() == ("flashinfer.gemm.mm_fp4", (("BACKEND", "cute-dsl"), ("TUNED", False)))
    with pytest.raises(ValueError, match="the weight of this call has it in the checkpoint's layout: the weights of this session were loaded for another plan"):
        down(qx, Dequantized(None, 1.0, merged))
    with pytest.raises(ValueError, match="multiplies packed 4-bit codes with their block scales; this block is handed ScaledTensor"):
        down(ScaledTensor(torch.zeros((1, model.geometry.mlp), dtype=torch.uint8), 1.0), Dequantized(None, 1.0, kept))
    with pytest.raises(ValueError, match="multiplies on the stored 8-bit weight with its one scale; the weight of this call has no stored 8-bit codes"):
        out(ScaledTensor(torch.zeros((1, model.geometry.la_out), dtype=torch.uint8), 1.0), Dequantized(None, 1.0, kept))
    per_row = ScaledTensor(torch.zeros((model.geometry.hidden, model.geometry.la_out), dtype=torch.uint8), torch.ones(model.geometry.hidden))
    with pytest.raises(ValueError, match="multiplies on the stored 8-bit weight with its one scale; the weight of this call has one scale per weight row"):
        out(ScaledTensor(torch.zeros((1, model.geometry.la_out), dtype=torch.uint8), 1.0), Dequantized(None, 1.0, per_row))
    assert isinstance(qx, BlockScaledTensor)

    # the loader holds a layout to the bytes it was made from.  The reference's copy of such a weight is made through the
    # layout's own way back to the checkpoint's order, so a wrong layout would agree with its reference in every later
    # check: block scales left row by row in the buffer, or codes that are not the stored ones, end the load by name
    def unswizzled(self, merged):
        plain = torch.zeros_like(tile_scales(merged.block_scales))
        plain[: merged.block_scales.shape[0], : merged.block_scales.shape[1]] = merged.block_scales
        return TiledBlockScaledWeight(merged.codes, plain, float(merged.scale))

    def recoded(self, merged):
        return TiledBlockScaledWeight(merged.codes & 0xF0, tile_scales(merged.block_scales), float(merged.scale))

    for wrong, adapter, at in [(w, *a) for w in (unswizzled, recoded) for a in (("MmFp4Block", "mlp.down (layer 0)"), ("MmFp4HeadBlock", "output_head"))]:
        with monkeypatch.context() as scoped:
            scoped.setattr(getattr(flashinfer_calls, adapter), "stored_layout", wrong)  # the head's adapter alone: the load ends at the head
            with pytest.raises(ValueError) as error:
                _recorded_run(scoped, sequences=1, stack="native")
        assert str(error.value) == (
            f"the layout of the stored weight of {at} does not hold the bytes of the checkpoint's tensor: TiledBlockScaledWeight, "
            "read back in the checkpoint's order, is another weight than the one it was made from"
        ), (wrong.__name__, adapter)


def test_the_librarys_package_its_requirement_files_and_the_policy_sentences(monkeypatch, tmp_path):
    """FlashInfer is an optional package of the stack ``native`` alone, at validated versions.  The version check reads the
    distribution's metadata and names the version, the command and the stack that runs without the package.  The
    requirement file is a lower bound at the validated version; the constraints file pins what the census was recorded
    with; the adapter's list is the same; none of them names Triton.  The library's kernel-binary package has a
    requirement file of its own, a lower bound without an index line, and is the one pin that is no version of the
    record: the census names it under a key of its own, at the library's version.  A run never lets the library fetch a
    kernel binary: every adapter reaches the library through one import, and the library's download is switched off
    before it -- where the environment does not hold the variable and where it holds it empty; a text that is set
    stays.  The level of the library's own lines is set before that import the same way, to ``warning``: where the
    environment names none, and where it names an empty one or blanks alone, on which the library's import raises; a
    level that is set stays; and a process the run starts finds both variables.  And the sentences that state this policy are in
    the area's guide and READMEs as they were agreed."""
    from model_samples import providers
    from model_samples.providers import ENTRY_NAMES, KEYS, flashinfer_calls
    from model_samples.qwen_dense import QwenDense

    assert (flashinfer_calls.DISTRIBUTION, flashinfer_calls.VALIDATED) == ("flashinfer-python", (VALIDATED_LIBRARY,))
    # the installed version is what the distribution's metadata states, whatever is importable: nothing is imported for
    # it, and a distribution the metadata does not know is not installed
    asked = []

    def stated(name):
        asked.append(name)
        return "9.9.9"

    def unknown(name):
        raise flashinfer_calls.metadata.PackageNotFoundError(name)

    with monkeypatch.context() as scoped:
        scoped.setattr(flashinfer_calls.metadata, "version", stated)
        version = flashinfer_calls.installed()
        assert version == "9.9.9" and asked == ["flashinfer-python"], "read from the metadata of the distribution, by its name"
        with pytest.raises(NotImplementedError, match=r"needs a validated flashinfer-python \(0.7.2.post1: .*\); this is 9.9.9: pip install -c "):
            flashinfer_calls.check_package("flashinfer.gemm.mm_fp4")
        scoped.setattr(flashinfer_calls.metadata, "version", unknown)
        version = flashinfer_calls.installed()
        assert version is None, "a distribution the metadata does not know"
        with pytest.raises(NotImplementedError, match="flashinfer.gemm.mm_fp4 needs flashinfer-python 0.7.2.post1, which is not installed: pip install -c "):
            flashinfer_calls.check_package("flashinfer.gemm.mm_fp4")
    fix = "pip install -c samples/models/constraints-native.txt -r samples/models/requirements-flashinfer.txt"
    assert fix == f"pip install -c {flashinfer_calls.CONSTRAINTS} -r {flashinfer_calls.REQUIREMENTS}"
    root = MODELS.parents[1]
    requirements = (root / flashinfer_calls.REQUIREMENTS).read_text().splitlines()
    constraints = (root / flashinfer_calls.CONSTRAINTS).read_text().splitlines()
    assert [line for line in requirements if not line.startswith("#")] == [f"flashinfer-python>={VALIDATED_LIBRARY}"], "a lower bound, never a pin"
    assert f"# Validated with: flashinfer-python {VALIDATED_LIBRARY}" in requirements
    pins = dict(line.split("==") for line in constraints if not line.startswith("#"))
    census_versions = QwenDense.target().data["validated"]["recorded"]
    recorded = {name: version for name, version in census_versions.items() if name != "triton"}
    # ... and the one pin that is no version of that record: the library's kernel-binary package, under a key of its own
    binaries = QwenDense.target().data["validated"]["kernel_binaries"]
    package = {name: version for name, version in binaries.items() if name != "joined"}
    assert package == {flashinfer_calls.BINARIES_DISTRIBUTION: VALIDATED_LIBRARY} and flashinfer_calls.BINARIES_DISTRIBUTION not in census_versions
    assert set(binaries["joined"]) == {"date", "gpu", "files", "run"}, "the key names what the package joined the list on"
    assert "triton" in census_versions and pins == {
        **recorded,
        **package,
    }, "the constraints: the recorded versions without Triton, which nothing calls, and the package"
    assert pins["flashinfer-python"] == VALIDATED_LIBRARY and not any("triton" in line.lower() for line in requirements + constraints)
    binaries_file = (root / flashinfer_calls.BINARIES_REQUIREMENTS).read_text().splitlines()
    assert [line for line in binaries_file if not line.startswith("#")] == [
        f"flashinfer-cubin>={VALIDATED_LIBRARY}"
    ], "a lower bound, never a pin, and no index line"
    assert f"# Validated with: flashinfer-cubin {VALIDATED_LIBRARY}" in binaries_file and f"#   {flashinfer_calls.BINARIES_INSTALL}" in binaries_file
    assert flashinfer_calls.BINARIES_INSTALL == (
        "pip install --no-deps --index-url https://flashinfer.ai/whl -c samples/models/constraints-native.txt -r samples/models/requirements-flashinfer-cubin.txt"
    ), "the index is named on the command, for this one package"
    # the refusal and the level of the library's own lines, through every library adapter's import of a package of the
    # library's name: it notes what it finds at its import and makes a logger at the level the variable names, as the
    # library does -- "info" where the variable names none, and an error where its text is no level name
    refusal, level = flashinfer_calls.NO_DOWNLOAD, "FLASHINFER_LOGGING_LEVEL"
    package = tmp_path / "flashinfer"
    package.mkdir()
    (package / "__init__.py").write_text(
        f"import logging\nimport os\nFOUND = os.environ.get({refusal!r})\nLEVEL = os.getenv({level!r}, 'info')\nlogging.Logger('flashinfer.jit').setLevel(LEVEL.upper())\n"
    )
    (package / "autotuner.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))
    adapters = {key: factory for key, factory in providers.factories().items() if "flashinfer" in key}
    child = [sys.executable, "-c", f"import os; print(os.environ.get({refusal!r}), os.environ.get({level!r}))"]
    for held, found in (((None, None), ("1", "warning")), (("", ""), ("1", "warning")), (("0", "  "), ("0", "warning")), (("0", "debug"), ("0", "debug"))):
        for key, adapter in adapters.items():
            for name in ("flashinfer", "flashinfer.autotuner"):
                monkeypatch.delitem(sys.modules, name, raising=False)
            for variable, text in zip((refusal, level), held):
                monkeypatch.setenv(variable, "")  # what the adapter sets is then taken back behind the case
                if text is None:
                    monkeypatch.delenv(variable)
                else:
                    monkeypatch.setenv(variable, text)
            library = adapter._library(None)
            assert (library.FOUND, library.LEVEL) == found, f"{key}, {held!r} in the environment: the library finds {found!r} at its import"
        started = subprocess.run(child, capture_output=True, text=True, timeout=120)
        assert tuple(started.stdout.split()) == found, f"{held!r} in the environment: a process this one starts finds {found!r}"
    # the keys of the library's rows, and what the list of keys says about itself
    assert set(ENTRY_NAMES) == set(KEYS) and {key: entry for key, entry in ENTRY_NAMES.items() if "flashinfer" in key} == {
        "flashinfer.mm_fp4": "flashinfer.gemm.mm_fp4",
        "flashinfer.mm_fp4_head": "flashinfer.gemm.mm_fp4",
        "flashinfer.bmm_fp8": "flashinfer.gemm.bmm_fp8",
        "flashinfer.decode_attention": ATTENTION_ENTRY,
    }
    guide = " ".join((MODELS / "AGENTS.md").read_text().split())
    area = " ".join((MODELS / "README.md").read_text().split())
    family = " ".join((MODELS / "qwen_dense" / "README.md").read_text().split())
    for sentence in (
        "A library's own tuner may choose among that library's kernels for a call a row names; the choice is printed with the kernel check and is "
        "never a row.",
        "The list of provider keys is closed; it changes only together with a kernel, with the reference chain of a block that a kernel is to serve, or "
        "with a declared library entry.",
        "A third-party provider may build its own kernels at its first use; a sample makes that first use while the run is planned, before anything "
        "is loaded, and says so in one line.",
        "A third-party provider is imported inside its adapter, on a device; its version is read from the distribution's metadata there and held to "
        "the validated list, and a missing or unvalidated version is a `SKIP` that names it, the requirement file and the nearest command that runs "
        "without it. A plan on stated device facts imports no third-party package.",
        'A library whose kernels a stack calls has a requirement file of its own, a lower bound with its "Validated with" line '
        "(`requirements-flashinfer.txt`), and `constraints-native.txt` holds the validated versions as exact pins: the versions the launch records of "
        "the family's census were recorded with. The maintainers of this area own that list; a version joins it with a recorded run, never with an "
        "update of the pin alone.",
    ):
        assert sentence in guide, sentence
    # the two exceptions as they were agreed, each directly behind the sentence it is the exception to and once
    built, fetched = (
        "A third-party provider may build its own kernels at its first use; a sample makes that first use while the run is planned, before anything "
        "is loaded, and says so in one line.",
        "A provider's precompiled kernel binaries are not built: they come from a package of that provider at a pinned version "
        "(`constraints-native.txt`) or from a directory the user filled beforehand, never from a fetch inside a run -- a sample switches the "
        "provider's own download off before its first use, and a binary that is not on the machine is a `SKIP` that names what is missing and the fix.",
    )
    joined, exception = (
        "The maintainers of this area own that list; a version joins it with a recorded run, never with an update of the pin alone.",
        "The one pin that is not a version of that record is the library's kernel-binary package (`requirements-flashinfer-cubin.txt`): the "
        "record's runs read kernel binaries the library had fetched, and the package joined the list when its files were held byte for byte to "
        "those files and a run on it launched the recorded kernels in the recorded launch shapes (the census names that run: "
        "`validated.kernel_binaries`).",
    )
    assert f"{built} {fetched} 6. **The stack is explicit.**" in guide and guide.count(fetched) == 1, "rule 5 and its exception, in front of rule 6"
    assert f"{joined} {exception} 13. **The report's GPU side" in guide and guide.count(exception) == 1, "rule 12 and its exception, in front of rule 13"
    assert (
        "FlashInfer is imported inside the adapters of the rows that call it, on a device, in the stack `native` alone; no other stack and no plan on "
        "stated device facts imports it." in area
    )
    for sentence in AREA_POLICY:
        assert sentence in area, sentence
    # the list of keys says of itself what the tripwire says, and what no provider module does
    stated = " ".join((providers.__doc__ or "").split())
    assert "it changes only together with a kernel, with the reference chain of a block that a kernel is to serve, or with a declared library entry." in stated
    assert "no provider module imports a third-party package: an adapter to one imports it inside the functions that run on a device." in stated
    # where the area README's first paragraph goes on from the library rows to speculative decoding, and where its table of modules has the adapter
    assert (
        "The kernels of the attention core are precompiled binaries of the library's kernel-binary package: a run reads them and never fetches "
        "one. A captured decode step (a step recorded once and replayed without Python between its launches) follows in that stack. Speculative "
        "decoding has its first pieces: the loader of the "
        "checkpoint's drafter (its tensors, its rows of the output head, its 8-bit weights); the drafter's draft pass on reference blocks,"
    ) in area
    # ... and what its table of tests says of the library rows in a step of speculative decoding, and of the Dependabot entry
    assert (
        "the same rows in a step of speculative decoding, behind the drafter's passes: the kernel class and the launch shape of the 4-bit multiply and "
        "the recorded pick of the 8-bit one held and printed at the model's call sites;" in area
        and "the ignore list of the Dependabot entry against the packages the constraints pin, and the form of its items (a name and the three "
        "update types of a version update, nothing else);" in area
        and "a launch in another shape than the recorded one at the model's site, a kernel class and a recorded pick at the model's site);" in area
    )
    modules = [line.split("|")[1].strip() for line in (MODELS / "README.md").read_text().splitlines() if line.startswith("| `providers/")]
    assert modules == ["`providers/fe.py`", "`providers/library.py`", "`providers/flashinfer_calls.py`", "`providers/torch_ref.py`"]
    assert (
        "| `providers/flashinfer_calls.py` | adapters to library kernels of FlashInfer on the stored weights: the 4-bit multiply (`flashinfer.gemm.mm_fp4`,"
        in area
    )
    assert "FlashInfer is an optional package of this stack alone." in family
    script = " ".join((MODELS / "qwen_dense" / "00_decode.py").read_text().split())
    assert fix in area and fix in family and fix in script
    assert "### The rows a library serves" in (MODELS / "qwen_dense" / "README.md").read_text()
    skipped = "`--stack native` on compute capability 10.x ends in `[00] SKIP` before anything is loaded, exit code 0"
    assert f"Without either package, or with another version of one, {skipped}" in family
    assert "on compute capability 10.x ``--stack native`` is a ``SKIP`` before anything is loaded" in script
    assert "# without the two packages, on compute capability 10.x: SKIP before anything is loaded" in (MODELS / "README.md").read_text()
    assert (
        flashinfer_calls.BINARIES_INSTALL in area and flashinfer_calls.BINARIES_INSTALL in family
    ), "the install line of the kernel-binary package, as it is typed"
    assert "--dry-plan" not in family + area + script, "the script has no such flag: a plan on stated device facts is made through the helper layer"
    assert "the call log reads `TUNED True`, and the kernel check fails with the form it found beside the form the row names" in family
    assert "prints the kernel that was chosen beside the pick the census records, says whether it is that pick, and never fails on it" in family
    for sentence in (  # what the library rows are, where they serve and where they have run (a comment mark of a command block left out)
        "Three more blocks are multiplies the target runs on a library, FlashInfer, on the stored weights",
        "so give native an environment of its own",
        "the out-projection FlashInfer's (a multiply over a copy in `fe-entries`)",
        "On compute capability 12.x no row of the library serves, so the three multiplies are over copies again",
        "have run on 10.0 alone: on 10.3 and 10.7 the rows are unrun, the census has no section",
        "8-bit multiply the same way: in `native` it starts on the release of the fused step or of the gate, and waits.",
        "Version updates come through the `pip` entry for `/samples/models` in `.github/dependabot.yml` and are handled by the repository's maintainers.",
    ):
        assert sentence in (area + family + guide).replace(" # ", " "), sentence


def test_the_level_of_the_librarys_own_lines_through_a_run_and_a_text_that_names_no_level(monkeypatch, capsys, tmp_path):
    """The level of the library's own lines through ``main()`` on the stack ``native``: the stand-in adapters reach a
    package of the library's name through the adapters' one import, and that package does at its import what the
    library does with the variable -- it upper-cases the text and hands it to ``setLevel`` of a logger.  Where the
    environment names no level -- the variable unset, empty, or blanks alone -- the run sets ``warning`` before the
    import and ends in ``PASS``; a level that is named stays as it was written.  A text that names no level ends the
    run in ``[00] ERROR`` while it is planned, before the library is imported, in a sentence of its own that names the
    variable and not its text -- never as operands an adapter built and the library declined.  What is declined is
    what the library's import raises on and nothing it takes: a level name in either case passes, a name with a blank
    beside it does not.  A plan on stated device facts asks nothing of the environment, and where the library is not
    installed the run is the ``SKIP`` of the missing package whatever the text.  The check sets no logger of the
    process."""
    import importlib
    import logging
    import os

    from model_samples import blocks
    from model_samples.providers import flashinfer_calls

    refusal, level = "FLASHINFER_NO_DOWNLOAD", "FLASHINFER_LOGGING_LEVEL"
    package = tmp_path / "flashinfer"
    package.mkdir()
    (package / "__init__.py").write_text(
        f"import logging\nimport os\nLEVEL = os.getenv({level!r}, 'info')\nlogging.Logger('flashinfer.jit').setLevel(LEVEL.upper())\n"
    )
    (package / "autotuner.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))
    found = []  # the level the package found at every import an adapter made of it
    root = logging.getLogger().level
    built_by_the_area = _stand_ins

    def stand_ins(patch, **options):
        built = built_by_the_area(patch, **options)
        for key, stand_in in list(built.items()):
            if "flashinfer" in key:  # the stand-in library, behind the adapters' own import of the package

                def _library(self, stand_in=stand_in):
                    found.append(flashinfer_calls.LibraryBlock._library(self).LEVEL)
                    return stand_in._library(self)

                built[key] = type(stand_in.__name__, (stand_in,), {"_library": _library})
        return built

    def fresh(text):
        """The environment names ``text`` (``None``: nothing), and the package is not imported yet."""
        for name in ("flashinfer", "flashinfer.autotuner"):
            sys.modules.pop(name, None)
        del found[:]
        monkeypatch.setenv(level, "" if text is None else text)
        if text is None:
            monkeypatch.delenv(level)

    monkeypatch.setattr(sys.modules[__name__], "_stand_ins", stand_ins)
    for variable in (refusal, level):  # what an adapter sets in the environment is taken back behind this case
        monkeypatch.setenv(variable, "")
        monkeypatch.delenv(variable)
    for name in ("flashinfer", "flashinfer.autotuner"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    run = ["--stack", "native", "--sequences", "1"]
    try:
        for text, expected in ((None, "warning"), ("", "warning"), (" \t", "warning"), ("debug", "debug")):  # blanks: a blank and a tab
            fresh(text)
            code, lines = _sample_run(monkeypatch, capsys, run)
            assert code == 0 and lines[-1] == "[00] PASS", (text, lines[-3:])
            assert (
                found and set(found) == {expected} and os.environ[level] == expected
            ), f"{text!r} in the environment: the library finds {expected!r} at its import"
        # a text that names no level: the run's ERROR, in a sentence of its own, while it is planned -- the library not imported
        fresh("loud")
        code, lines = _sample_run(monkeypatch, capsys, run)
        sentence = (
            f": the environment's {level} names no level of Python's logging module, and FlashInfer takes its text for one when it is imported; "
            "name a level there (debug, info, warning, error or critical), or unset the variable and this sample sets it to warning"
        )
        assert code == 1 and lines[-1].startswith("[00] ERROR flashinfer.") and lines[-1].endswith(sentence), lines[-1]
        assert "loud" not in lines[-1].lower() and not any("declined the operands" in line for line in lines), "the variable is named, its text is not"
        assert found == [] and "flashinfer" not in sys.modules and os.environ[level] == "loud", "the library was not imported, and the text stays"
        assert (
            LIBRARY_CALLS == [] and ATTENTION_CALLS == [] and not any(line.startswith(("checks", "[00] stage")) for line in lines)
        ), "nothing was called or loaded"
        # every row of the library asks, not the attention's alone: with the attention core on the chain a multiply's row declines it
        code, lines = _sample_run(monkeypatch, capsys, [*run, "--provider", "attention.core=torch_ref.attention"])
        assert code == 1 and lines[-1].endswith(sentence) and not any("declined the operands" in line for line in lines), lines[-1]
        assert "flashinfer" not in sys.modules and LIBRARY_CALLS == []
        # where the library is not installed nothing would import it, so its level is not asked: the SKIP of the missing package
        fresh("loud")
        code, lines = _sample_run(monkeypatch, capsys, run, library_installed=None)
        assert code == 0 and lines[-1].startswith("[00] SKIP flashinfer.") and "which is not installed" in lines[-1], lines[-1]
        # a plan on stated device facts makes no first use and asks nothing of the environment: the same text, and the row is planned
        model, stated = _tiny_model(), blocks.DeviceFacts((10, 0), "stated device", 148, 0, 0)
        rows = blocks.plan(model.provider_table, model.regimes, stated, 1, 1, "native", model=model, options={"device": "cpu"}).rows
        assert rows["attention.core"].key == "flashinfer.decode_attention" and "flashinfer" not in sys.modules, "the table is stated, nothing is imported"
        # what is declined is what the library's import raises on, and nothing it takes
        for text, taken in (("WARN", True), ("Critical", True), ("notset", True), (" info", False), ("info ", False), ("10", False)):
            fresh(text)
            if taken:
                flashinfer_calls.check_level("an entry")
                assert logging.getLogger().level == root, "the check decides on a logger of its own: the root logger is where it was"
                assert importlib.import_module("flashinfer").LEVEL == text
            else:
                with pytest.raises(ValueError, match=f"an entry: the environment's {level} names no level of Python's logging module"):
                    flashinfer_calls.check_level("an entry")
                with pytest.raises(ValueError, match="Unknown level"):
                    importlib.import_module("flashinfer")
    finally:
        for name in ("flashinfer", "flashinfer.autotuner"):
            sys.modules.pop(name, None)


def _ignored_by_dependabot(text: str, ecosystem: str, directory: str) -> list:
    """The items of the ``ignore`` list of the one ``updates`` entry of a Dependabot file for ``ecosystem`` and ``directory``,
    in file order: per item its ``dependency-name`` and its other keys with their values as lists.  Read as lines, as the
    file is written (block style, one key a line, the values of a key in that line)."""

    def value(line):
        return line.split(":", 1)[1].split(" #", 1)[0].strip().strip("\"'")

    entries = re.split(r"(?m)^  - (?=package-ecosystem:)", text)[1:]
    found = [
        entry for entry in entries if value(entry.splitlines()[0]) == ecosystem and re.search(rf'(?m)^    directory: "?{re.escape(directory)}"?\s*$', entry)
    ]
    assert len(found) == 1, f"one updates entry for {ecosystem} in {directory}; found {len(found)}"
    items, inside = [], False
    for line in found[0].splitlines()[1:]:
        if re.match(r"^    [a-z-]+:", line):  # a key of the entry: the list ends at the next one
            inside = line.startswith("    ignore:")
            assert not inside or line.rstrip() == "    ignore:", "the ignore list is written one item a line"
        elif inside and line.strip() and not line.strip().startswith("#"):
            if line.startswith("      - dependency-name:"):
                items.append((value(line), {}))
                continue
            key = re.match(r"^        ([a-z-]+): *(\[.*\])\s*$", line)
            assert key and items, f"an item of the ignore list is a name and, under it, keys with their values in one line: {line!r}"
            assert key.group(1) not in items[-1][1], f"a key stands once in an item: {line!r}"
            items[-1][1][key.group(1)] = [word.strip().strip("\"'") for word in key.group(2)[1:-1].split(",") if word.strip()]
    return items


def test_the_dependabot_entry_ignores_exactly_the_packages_the_native_constraints_pin():
    """The exact pins of ``constraints-native.txt`` are the versions the kernel census was recorded with, so an update of
    one fails this module by design, and the ``pip`` entry for ``/samples/models`` of ``.github/dependabot.yml`` must not
    propose one as a version update: its ignore list names the packages the file pins -- each by its own name, none by a
    pattern, none twice, no package more and none less.  A pin that is added without its ignore line fails here, and so
    does an ignore line whose pin is gone.  The form of an item is held too: the name and the three update types of a
    version update, nothing else.  By GitHub's documentation ``update-types`` affects version updates only, while an
    ignore item with ``versions`` affects security updates as well (the options page as published for GitHub Enterprise
    Server 3.9; today's page says the first under ``allow``).  (Which versions the file pins is held by the test of the
    requirement files, against the census.)"""
    from model_samples.providers import flashinfer_calls

    root = MODELS.parents[1]
    constraints = [line for line in (root / flashinfer_calls.CONSTRAINTS).read_text().splitlines() if line and not line.startswith("#")]
    assert constraints and all(line.count("==") == 1 for line in constraints), "every line of the constraints is one exact pin"
    pinned = [line.split("==")[0].strip() for line in constraints]
    dependabot = (root / ".github" / "dependabot.yml").read_text()
    items = _ignored_by_dependabot(dependabot, "pip", "/samples/models")
    ignored = [name for name, _ in items]
    assert sorted(ignored) == sorted(pinned), f"the ignore list names {sorted(ignored)}; the constraints pin {sorted(pinned)}"
    assert len(set(ignored)) == len(ignored) and not any(set(name) & set("*?[]") for name in ignored), "one line a package, by its name"
    levels = [f"version-update:semver-{level}" for level in ("major", "minor", "patch")]
    for name, keys in items:
        assert list(keys) == ["update-types"], f"{name}: an item ignores version updates and nothing else; its keys are {['dependency-name', *keys]}"
        assert sorted(keys["update-types"]) == levels, f"{name}: every level of a version update, each once; found {keys['update-types']}"
    # the reader, on what it must tell apart: another directory's entry, a list that ends at the next key, the keys of an item
    other = "updates:\n  - package-ecosystem: pip\n    directory: /\n    ignore:\n      - dependency-name: torch\n"
    mine = (
        '  - package-ecosystem: "pip"\n    directory: "/samples/models"\n    ignore:\n      - dependency-name: "a"  # why\n'
        '        update-types: ["x:y", z]\n      - dependency-name: b\n    labels:\n      - c\n'
    )
    assert _ignored_by_dependabot(other + mine, "pip", "/samples/models") == [("a", {"update-types": ["x:y", "z"]}), ("b", {})]
    assert _ignored_by_dependabot(other + mine, "pip", "/") == [("torch", {})]
    assert _ignored_by_dependabot(mine.replace("    ignore:\n", ""), "pip", "/samples/models") == []
    assert _ignored_by_dependabot(mine.replace(": b\n", ': b\n        versions: ["2.x"]\n'), "pip", "/samples/models")[1] == ("b", {"versions": ["2.x"]})
    with pytest.raises(AssertionError, match="one updates entry for pip in /samples/models; found 0"):
        _ignored_by_dependabot(other, "pip", "/samples/models")
    with pytest.raises(AssertionError, match="a name and, under it, keys with their values in one line"):
        _ignored_by_dependabot(mine.replace('        update-types: ["x:y", z]\n', "        update-types:\n          - x\n"), "pip", "/samples/models")
    with pytest.raises(AssertionError, match="a key stands once in an item"):
        _ignored_by_dependabot(mine.replace(": b\n", ": b\n        versions: [1]\n        versions: [2]\n"), "pip", "/samples/models")


# ---------------------------------------------------------------------------------------------------------------------------
# The attention preparation on its entry (``fe.attention_prepare_decode`` -> ``cudnn.AttentionPrepareDecode``): the row, the
# adapter as it is over a stand-in entry, its operands through the entry's real host check, its twin and its band, the
# kernel check on constructed records, the counts, and the control arms through ``main()``.
# ---------------------------------------------------------------------------------------------------------------------------
def _prepare_block(monkeypatch, calls, sequences=1, tokens=1, capacity=130, model=None, row=None, **entry):
    """The real adapter of the attention preparation over a stand-in entry (:func:`_prepare_entry`; ``entry``: its
    options) on the CPU, at the published widths, with a paged state of ``sequences`` x ``tokens`` and operands of one
    call: ``(block, cache of the model's attention layer, in-projection rows, the layer's parameters, state, model)``."""
    import cudnn

    from model_samples.blocks import DeviceFacts
    from model_samples.providers import AttentionParams, fe
    from model_samples.qwen_dense import QwenDense, configs
    from model_samples.state import DecodeState

    monkeypatch.setattr(cudnn, "AttentionPrepareDecode", _prepare_entry(calls, **entry), raising=False)
    monkeypatch.setattr(fe, "_ENTRIES", {})
    model = QwenDense.reduced(2) if model is None else model
    g = model.geometry
    row = configs.PROVIDERS["attention.prepare"][0] if row is None else row
    block = fe.AttentionPrepareDecodeBlock(row, model, DeviceFacts((10, 0), "stated device", 148, 0, 0), {"device": "cpu", "check_kernels": True})
    state = DecodeState(g.layer_kinds, sequences, capacity, model.state_shapes(), "cpu", tokens, page=64)
    generator = torch.Generator().manual_seed(7)
    rows = sequences * tokens
    qkv = (torch.randn(rows, 2 * g.attn_out + 2 * g.attn_kv_heads * g.attn_head_dim, generator=generator) * 3).bfloat16()
    weights = [(torch.randn(g.attn_head_dim, generator=generator) * 0.3).bfloat16() for _ in range(2)]
    params = AttentionParams(g.attn_heads, g.attn_kv_heads, g.attn_head_dim, g.rotary_dim, g.rope_theta, *weights)
    return block, state.layers[g.layer_kinds.index("full_attention")], qkv, params, state, model


def test_the_attention_preparation_row_and_what_falls_to_the_chain(tmp_path):
    """The row of ``cudnn.AttentionPrepareDecode``: both target stacks, compute capability 10.x, 1 to 1,024 rows of a step of
    one token per sequence or of several, in the target's form, at the published attention widths and the published
    rotary section.  Past 1,024 rows, on compute capability 12.x, on the two other stacks and for a checkpoint of the
    family with another head width, rotary width or rotary section the block is the reference chain -- never an error --
    and a run of another class or another width says that no kernel identity is stated.  The section is read from a
    checkpoint's configuration."""
    from dataclasses import replace
    from types import SimpleNamespace

    from model_samples import blocks, census
    from model_samples.blocks import DeviceFacts
    from model_samples.qwen_dense import Geometry, QwenDense, configs

    row, chain = configs.PROVIDERS["attention.prepare"]
    assert row.key == "fe.attention_prepare_decode" and row.classes == ("cc10.x",) and row.sequences == (1, 1024) and row.tokens == (1, None)
    assert row.rows == (1, 1024) and set(row.stacks) == {"native", "fe-entries"} and row.band == "attention_prepare" and row.batch_invariant == ("cc10.0",)
    assert (
        not row.fallback and row.ci and row.serves == "rows 1-1,024, 24 heads of width 256 on 4 key / value heads, rotary width 64 in the sections 11 + 11 + 10"
    )
    form = {"POSITIONS": "3d", "SECTION": (11, 11, 10), "PAGE_SIZE": 64, "POSITION_DTYPE": "int64", "SLOT_DTYPE": "int64", "PDL": False, "PDL_RELEASE": False}
    assert dict(row.form) == form, "the target's form, stated whole: three position rows in the published sections, pages of 64, an ordinary launch"
    assert census.form_text(row.form).strip() == PREPARE_FORM
    assert row.widths == (("attn_heads", 24), ("attn_kv_heads", 4), ("attn_head_dim", 256), ("rotary_dim", 64), ("rotary_section", (11, 11, 10)))
    assert chain.key == "torch_ref.attention_prepare" and chain.fallback == "reference chain" and set(chain.stacks) == set(configs.EVERY)
    assert Geometry().rotary_section == (11, 11, 10) and sum(Geometry().rotary_section) == Geometry().rotary_dim // 2
    target = QwenDense.target()
    for stack in ("native", "fe-entries"):
        for sequences in (1, 85, 1024):  # the first row count, one past a scheduling wave of 84 tokens, the last
            planned, model, _ = _target_plan(2, stack, sequences=sequences)
            assert planned.rows["attention.prepare"] is row and planned["attention.prepare"].cache_page == 64, (stack, sequences)
            assert (
                census.check_forms(census.expected(planned, model)) == []
            ), f"{sequences} rows: the row's form is the entry's own (the public entry, built for the host)"
        planned, model, _ = _target_plan(2, stack, sequences=1025)
        assert planned.rows["attention.prepare"] is chain and planned["attention.prepare"].cache_page is None, "past 1,024 rows: the chain, on a plain cache"
        line = next(line for line in planned.table().splitlines() if line.split()[0] == "attention.prepare")
        assert line.endswith("fallback: reference chain") and "torch_ref.attention_prepare" in line, line
        with pytest.raises(
            NotImplementedError,
            match=r"^cudnn.AttentionPrepareDecode serves rows 1-1,024, 24 heads of width 256 on 4 key / value heads, rotary width 64 in the sections "
            r"11 \+ 11 \+ 10; this run has 1025 sequences x 1 tokens per sequence$",
        ):
            _target_plan(2, stack, sequences=1025, overrides={"attention.prepare": "fe.attention_prepare_decode"})
        # a step of several tokens per sequence: one launch for its rows -- 1 x 6, and the largest verify step the rows serve, 16 x 5
        model = QwenDense.reduced(2)
        facts = DeviceFacts((10, 0), "stated device", 148, 0, 0)
        for sequences, tokens in ((1, 6), (16, 5), (4, 8)):
            several = blocks.plan(model.provider_table, model.regimes, facts, sequences, tokens, stack, model=model, options={"device": "cpu"})
            assert several.rows["attention.prepare"] is row, (stack, sequences, tokens)
    for stack in ("fe-cublas", "reference"):
        assert _target_plan(2, stack)[0].rows["attention.prepare"] is chain, f"{stack}: the reference chain, as before"
    # the other 10.x classes: the row is the entry's, and the census has no section of the class -- no identity is stated
    for cc, sms in (((10, 3), 152), ((10, 7), 212)):
        planned, model, _ = _target_plan(2, cc=cc, sms=sms)
        claim = target.claim(planned, model, installed=RECORDED)
        assert planned.rows["attention.prepare"] is row and not claim.known, cc
        assert claim.coverage()[0].startswith(f"kernel identity: not stated for this run (no recorded section for cc {cc[0]}.{cc[1]} / {sms} SMs"), cc
    # compute capability 12.x: the chain, and no identity is stated; --provider cannot put the entry there
    planned, model, _ = _target_plan(2, cc=(12, 0), sms=170)
    assert planned.rows["attention.prepare"] is chain and planned["attention.prepare"].cache_page is None
    assert target.claim(planned, model, installed=RECORDED).coverage() == [
        "kernel identity: not stated for this run (no recorded section for cc 12.0 / 170 SMs: recorded is cc 10.0 / 148 SMs)"
    ]
    with pytest.raises(NotImplementedError, match=r"^cudnn.AttentionPrepareDecode serves compute capability 10.x; this device is 12.0$"):
        _target_plan(2, cc=(12, 0), sms=170, overrides={"attention.prepare": "fe.attention_prepare_decode"})
    # a checkpoint of the family with another head width, rotary width or rotary section: not this row's -- the chain, and no identity stated
    facts = DeviceFacts((10, 0), "stated device", 148, 180 * GIB, 80 * GIB)
    for field, value, said in (
        ("attn_head_dim", 128, "attn_head_dim 128 (recorded 256)"),
        ("rotary_dim", 32, "rotary_dim 32 (recorded 64)"),
        ("rotary_section", (16, 8, 8), "rotary_section (16, 8, 8) (recorded (11, 11, 10))"),
    ):
        other = QwenDense(replace(QwenDense.reduced(2).geometry, **{field: value}), (f"another {field}",))
        planned = blocks.plan(other.provider_table, other.regimes, facts, 1, 1, "fe-entries", model=other, options={"device": "cpu"})
        assert planned.rows["attention.prepare"] is chain and planned.rows["add_norm_quantize"].key == "fe.add_norm_quant_decode", field
        assert target.claim(planned, other, installed=RECORDED).coverage() == [
            f"kernel identity: not stated for this run (no recorded section for this model's widths on cc 10.0 / 148 SMs: {said})"
        ], field
        serves = "attn_heads 24, attn_kv_heads 4, attn_head_dim 256, rotary_dim 64, rotary_section (11, 11, 10)"
        with pytest.raises(
            NotImplementedError, match=rf"^cudnn.AttentionPrepareDecode serves {re.escape(serves)}; this model has {re.escape(said.split(' (recorded')[0])}$"
        ):
            blocks.plan(other.provider_table, other.regimes, facts, 1, 1, "fe-entries", {"attention.prepare": "fe.attention_prepare_decode"}, model=other)
    # the section of a checkpoint is what its configuration names under its rotary parameters; the target's where it names none
    config = dict(model_type="qwen3_5_text", layer_types=["linear_attention", "full_attention"], head_dim=256, hidden_size=5120, vocab_size=272)
    config.update(intermediate_size=17408, linear_num_key_heads=16, linear_num_value_heads=48, linear_value_head_dim=128, linear_conv_kernel_dim=4)
    config.update(num_attention_heads=24, num_key_value_heads=4, rms_norm_eps=1e-6, rope_parameters={"partial_rotary_factor": 0.25, "rope_theta": 1.0e7})
    tensors = {"embed_tokens.weight": None, "layers.0.input_layernorm.weight": None}

    def described(**rope):
        made = dict(config, rope_parameters=dict(config["rope_parameters"], **rope))
        return QwenDense.describe(SimpleNamespace(config=made, label="a checkpoint", tensors=tensors, bind=lambda specs, keys: None)).geometry

    assert described().rotary_section == (11, 11, 10) and described(mrope_section=[11, 11, 10]).rotary_section == (11, 11, 10)
    assert described(mrope_section=[16, 8, 8]).rotary_section == (16, 8, 8) and described().rotary_dim == 64
    for wrong in ([11, 11], "11, 11, 10", [11, 11, 10.5], [11, True, 10], 32):
        with pytest.raises(ValueError, match=r"^a checkpoint: config.json has mrope_section = .* where three whole numbers are expected$"):
            described(mrope_section=wrong)
    # the tables of the family README are the table's own rendering, with this row in them
    readme = (MODELS / "qwen_dense" / "README.md").read_text()
    native = configs.readme_table(QwenDense().calls_text(), "native")
    assert f"| attention preparation | 16 | `fe.attention_prepare_decode` | cudnn.AttentionPrepareDecode {PREPARE_FORM} |" in native and native in readme
    default = configs.readme_table(QwenDense().calls_text())
    assert (
        "| attention preparation | 16 | `torch_ref.attention_prepare` | torch_ref.attention_prepare, fallback: reference chain |" in default
        and default in readme
    )


def test_the_attention_preparation_entrys_adapter_over_a_stand_in_entry(monkeypatch):
    """``fe.attention_prepare_decode`` as it is, over a stand-in entry that records what it is handed.  The seven fields of
    the row's form reach the constructor as the entry's own arguments, one instance per (device, geometry, epsilon,
    form).  A call hands the entry four VIEWS of the in-projection's row (nothing copied), the layer's two norm weights,
    the cos / sin table and the position and slot rows of ``cache.at(position)``, the two halves of the cache's pool,
    and outputs of its own; what it returns are the tensors the entry wrote.  Handed a second state, it hands its entry
    that state's pool and table rows: nothing of a call is kept.  A row that leaves a field of the form out, a cache
    that is no pool of pages, a pool of pages of another size and a decline inside a step are errors.  The first use,
    while the run is planned: the entry's build, whose every typed decline is the installation's, then one call on
    scratch operands, whose decline is of the adapter's operands -- unless the installation cannot import what the
    call needs."""
    from dataclasses import replace

    import cudnn
    from cudnn.attention_prepare import AttentionPrepareGeometry

    from model_samples import census
    from model_samples.blocks import DeviceFacts
    from model_samples.providers import fe, torch_ref
    from model_samples.qwen_dense import configs
    from model_samples.state import DecodeState

    calls = []
    block, cache, qkv, params, state, model = _prepare_block(monkeypatch, calls, sequences=3)
    row = block.row
    block.check_support(3, 1)
    assert fe._ENTRIES == {} and block.serving() == ("cudnn.AttentionPrepareDecode", ()), "on stated facts nothing is built; before a call nothing has served"
    assert block.kernels == (census.Kernel("attention_prepare", "cuDNN FE", "exact", launches=1, shape="entry"),)
    assert (block.per_row, block.state_args, block.cache_page, block.provider, block.entry) == ((0, 1), (1,), 64, "cuDNN FE", "cudnn.AttentionPrepareDecode")
    facts = block.launch_facts(3)
    (entry,) = fe._ENTRIES.values()
    seven = dict(positions="3d", page_size=64, position_dtype="int64", slot_dtype="int64", mrope_section=(11, 11, 10), pdl=False, pdl_release=False)
    built = dict(seven, device="cpu", geometry=AttentionPrepareGeometry(24, 4, 256, 64), eps=1e-6)
    assert entry.built == built, "the row's seven fields, as the entry's own arguments, beside the model's geometry and its epsilon"
    assert facts.shape == ((3, 28, 1), (128, 1, 1), 16) and facts.symbols == (
        "cudnn_kernel_frost_attention_prepare_h24_4_d256_r64_3d_11_11_10_p64_i64_i64_plain_eps358637bd_0",
    )
    assert {name: facts.form[name] for name, _ in row.form} == dict(row.form)

    first = block(qkv, cache, 63, params)
    second = block(qkv, cache, 64, params)
    assert [kind for kind, _ in calls] == ["prepare launch", "prepare launch"], "a call launches its entry and nothing else: no compile inside a step"
    (_, one), (_, two) = calls
    base, row_bytes = qkv.data_ptr(), 2 * qkv.shape[1]
    for name, offset, heads, stride in (("q", 0, 24, 512), ("gate", 256, 24, 512), ("k", 12288, 4, 256), ("v", 13312, 4, 256)):
        view = one[name]
        assert tuple(view.shape) == (3, heads, 256) and view.stride() == (14336, stride, 1) and view.data_ptr() == base + 2 * offset, name
        assert base <= view.data_ptr() < base + 3 * row_bytes and torch.equal(
            view, qkv[:, offset : offset + heads * stride].view(3, heads, stride)[..., :256]
        ), name
    assert one["q_weight"] is params.q_norm and one["k_weight"] is params.k_norm, "the layer's two norm weights, as the loader holds them"
    for at, call in ((63, one), (64, two)):
        cos_sin, positions, slots = cache.at(at)
        assert call["cos_sin"] is state.table.cos_sin and call["cos_sin"] is cos_sin
        assert call["positions"].data_ptr() == positions.data_ptr() == state.table.positions[at].data_ptr() and tuple(call["positions"].shape) == (3, 3)
        assert call["slot_mapping"].data_ptr() == slots.data_ptr() == state.table.slots[at].data_ptr() and call["slot_mapping"].tolist() == [
            sequence * 3 * 64 + at for sequence in range(3)
        ], "the slots of the step: sequence x pages x 64 + position, read from the cache's table and from nowhere else"
        assert call["positions"].tolist() == [[at] * 3] * 3 and call["positions"].dtype == call["slot_mapping"].dtype == torch.int64
        assert call["k_pool"].data_ptr() == cache.pool.data_ptr() and call["v_pool"].data_ptr() == cache.pool.data_ptr() + 2 * 256
        assert tuple(call["k_pool"].shape) == tuple(call["v_pool"].shape) == (9, 4, 64, 256), "the two halves of the whole pool: views"
    assert one["q_out"] is not two["q_out"] and one["gate_out"] is not two["gate_out"], "outputs of the call's own: never the entry's pair"
    assert first[0] is one["q_out"] and second[0] is two["q_out"], "what a call returns are the tensors its entry wrote"
    assert first[1].data_ptr() == one["gate_out"].data_ptr() and tuple(first[1].shape) == (3, 6144) and tuple(first[0].shape) == (3, 24, 256)
    assert first[0].dtype == first[1].dtype == torch.bfloat16 and one["q_out"].is_contiguous() and one["gate_out"].is_contiguous()
    assert torch.equal(first[1].view(torch.int16), qkv[:, :12288].view(3, 24, 512)[..., 256:].reshape(3, 6144).view(torch.int16)), "the gates: a bit copy"
    assert block.serving() == ("cudnn.AttentionPrepareDecode #1", row.form), "the call log reads the instance that was called"
    # one sequence alone, as the batch-invariance check runs it: the same pool, the window's columns of the same table rows
    alone = cache.rows(1, 2)
    block(qkv[1:2], alone, 64, params)
    call = calls[-1][1]
    assert call["slot_mapping"].tolist() == [3 * 64 + 64] and call["positions"].tolist() == [[64]] * 3 and call["k_pool"].data_ptr() == cache.pool.data_ptr()
    assert call["q"].data_ptr() == base + row_bytes and call["slot_mapping"].data_ptr() == state.table.slots[64].data_ptr() + 8
    # a second state under the same block, as the checks of a run hand it one: that state's pool and table rows, nothing of the first
    again = DecodeState(model.geometry.layer_kinds, 3, 130, model.state_shapes(), "cpu", page=64)
    second_cache = again.layers[model.geometry.layer_kinds.index("full_attention")]
    assert second_cache.pool.data_ptr() != cache.pool.data_ptr() and not second_cache.pool.any()
    block(qkv, second_cache, 63, params)
    call = calls[-1][1]
    assert call["k_pool"].data_ptr() == second_cache.pool.data_ptr() and call["v_pool"].data_ptr() == second_cache.pool.data_ptr() + 2 * 256, "its pool"
    assert call["positions"].data_ptr() == again.table.positions[63].data_ptr() and call["slot_mapping"].data_ptr() == again.table.slots[63].data_ptr()
    assert (
        call["cos_sin"] is again.table.cos_sin and second_cache.pool[:, :, 63].any() and not second_cache.pool[:, :, :63].any()
    ), "and its tables: its row 63 written"
    # the form and the instances
    with pytest.raises(ValueError) as wrong:
        fe.AttentionPrepareDecodeBlock(replace(row, form=row.form[:-1]), model, block.facts, block.options).check_support(1, 1)
    assert str(wrong.value) == (
        "provider row fe.attention_prepare_decode: its form does not name PDL_RELEASE; this adapter passes POSITIONS, SECTION, PAGE_SIZE, POSITION_DTYPE, "
        "SLOT_DTYPE, PDL, PDL_RELEASE to cudnn.AttentionPrepareDecode, and never leaves one of them to the entry's own default"
    )
    with pytest.raises(ValueError, match=r"^provider row fe.attention_prepare_decode: its form names THREADS; this adapter passes POSITIONS, "):
        fe.AttentionPrepareDecodeBlock(replace(row, form=(*row.form, ("THREADS", 128))), model, block.facts, block.options).check_support(1, 1)
    assert fe.AttentionPrepareDecodeBlock(row, model, block.facts, block.options).launch_facts(1).symbols == facts.symbols and len(fe._ENTRIES) == 1
    other = type(model)(replace(model.geometry, norm_eps=1e-5), model.reductions)
    fe.AttentionPrepareDecodeBlock(row, other, block.facts, block.options).launch_facts(1)
    assert [built.built["eps"] for built in fe._ENTRIES.values()] == [1e-6, 1e-5], "the epsilon is a compile key of the entry: two values, two instances"
    one_row = fe.AttentionPrepareDecodeBlock(replace(row, form=(("POSITIONS", "1d"), ("SECTION", None), *row.form[2:])), model, block.facts, block.options)
    one_row(qkv, cache, 5, params)
    assert tuple(calls[-1][1]["positions"].shape) == (3,) and calls[-1][1]["positions"].data_ptr() == state.table.positions[5].data_ptr()
    assert one_row.serving() == ("cudnn.AttentionPrepareDecode #3", (("POSITIONS", "1d"), ("SECTION", None), *row.form[2:])), "one position row: another form"
    # what is not a call of this row
    launched = len(calls)
    plain = DecodeState(model.geometry.layer_kinds, 3, 130, model.state_shapes(), "cpu").layers[1]
    with pytest.raises(ValueError) as wrong:
        block(qkv, plain, 0, params)
    assert str(wrong.value) == (
        "cudnn.AttentionPrepareDecode writes an attention cache that is a pool of pages of 64 positions; this block is handed KvCache: "
        "the session's state was built for another plan"
    )
    halved = DecodeState(model.geometry.layer_kinds, 3, 130, model.state_shapes(), "cpu", page=32).layers[1]
    with pytest.raises(ValueError) as wrong:
        block(qkv, halved, 0, params)
    assert str(wrong.value) == (
        "cudnn.AttentionPrepareDecode writes an attention cache that is a pool of pages of 64 positions; this block is handed pages of 32 positions: "
        "the session's state was built for another plan"
    ), "a pool of pages of another size is not this adapter's either"
    doubled = DecodeState(model.geometry.layer_kinds, 3, 130, model.state_shapes(), "cpu", page=128).layers[1]
    with pytest.raises(ValueError) as wrong:
        block(qkv, doubled, 0, params)
    assert str(wrong.value) == (
        "cudnn.AttentionPrepareDecode writes an attention cache that is a pool of pages of 64 positions; this block is handed pages of 128 positions: "
        "the session's state was built for another plan"
    ), "nor is a pool of larger pages"
    assert len(calls) == launched, "neither cache was handed to the entry"
    declined = "the attention preparation decode step serves at most 1024 rows (tokens) per call; got 1025"
    with monkeypatch.context() as patched:
        patched.setattr(type(entry), "__call__", lambda self, *operands: (_ for _ in ()).throw(NotImplementedError(declined)))
        with pytest.raises(RuntimeError, match=rf"^cudnn.AttentionPrepareDecode declined the operands of this step: {re.escape(declined)}$"):
            block(qkv, cache, 0, params)
    for name in (
        "attention_prepare/frost/prepare_decode.py",
        "attention_prepare/frost/kernel/prepare_decode_config.py",
        "attention_prepare/frost/kernel/prepare_decode_f16.py",
        "gated_attention_block/cache_write_contract.py",
        "frost/init_order.py",  # what a call of this entry runs in front of its launch: the order behind the initialisation of entry-owned outputs
    ):
        assert name in fe.SOURCE_FILES and (Path(cudnn.__file__).parent / name).is_file(), f"the source hash of a run covers {name}"
    importers = [
        name for name in fe.SOURCE_FILES if re.search(r"^from cudnn\.frost import init_order\b", (Path(cudnn.__file__).parent / name).read_text(), re.M)
    ]
    assert (
        "attention_prepare/frost/prepare_decode.py" in importers and fe.SOURCE_FILES.count("frost/init_order.py") == 1
    ), "this entry's file imports it: hashed once"
    reference = torch_ref.AttentionPrepare(configs.PROVIDERS["attention.prepare"][1], model, block.facts, block.options)
    twin = block.twin(reference)
    assert type(twin) is torch_ref.AttentionPrepareRounded and block.twin(reference) is twin and twin.order == "3d" and reference.twin(reference) is reference
    assert one_row.twin(reference).order == "1d" and one_row.twin(reference) is not twin, "the twin of a row of one position row rotates in that row's order"

    # the first use, while the run is planned: on a real device alone
    real = DeviceFacts((10, 0), "stated device", 148, 180 * GIB, 80 * GIB)

    class OnTheCpu(fe.AttentionPrepareDecodeBlock):
        first_use_devices = ("cpu",)

    def planned(count, decline=None, **more):
        del calls[:]
        monkeypatch.setattr(cudnn, "AttentionPrepareDecode", _prepare_entry(calls, decline=decline, **more), raising=False)
        monkeypatch.setattr(fe, "_ENTRIES", {})
        made = OnTheCpu(row, model, real, {"device": "cpu", "check_kernels": True})
        made.check_support(count, 1)
        return made

    block.first_use(4, 1)
    assert calls[-1][0] == "prepare launch" and len(fe._ENTRIES) == 3, "on stated device facts a plan builds nothing and launches nothing"
    made = planned(5)
    made.first_use(5, 1)
    assert [kind for kind, *_ in calls] == ["prepare compile", "prepare launch"] and calls[0] == ("prepare compile", None), "the build takes no operand"
    scratch = calls[1][1]
    assert tuple(scratch["q"].shape) == (5, 24, 256) and tuple(scratch["k_pool"].shape) == (1, 4, 64, 256) and tuple(scratch["positions"].shape) == (3, 5)
    assert scratch["slot_mapping"].tolist() == [0, 1, 2, 3, 4] and tuple(scratch["cos_sin"].shape) == (1, 64) and not scratch["positions"].any()
    assert scratch["q"].stride() == (14336, 512, 1) and scratch["k"].stride() == (14336, 256, 1), "scratch operands in the layout a step hands the entry"
    made.compile(5, 1)
    assert [kind for kind, *_ in calls[2:]] == ["prepare compile", "prepare launch"], "at load: the entry finds its kernel built, and launches once"
    needs = ["the attention preparation decode step on compute capability 10.7 requires a CuTe DSL whose compiler is CUDA 13.4 or later"]
    made = planned(1, decline=needs)
    with pytest.raises(NotImplementedError, match=rf"^{re.escape(needs[0])}$"):
        made.first_use(1, 1)  # the entry's own words: the SKIP
    assert [kind for kind, *_ in calls] == ["prepare compile"], "declined by the build: nothing was launched"
    made = planned(2)
    for raised, text in ((ValueError, "k_pool must be [num_pages >= 1, H=4, page_size=64, D=256]"), (TypeError, "positions must be torch.int64")):
        monkeypatch.setattr(
            type(fe._ENTRIES and list(fe._ENTRIES.values())[0] or made._build("cpu")),
            "__call__",
            lambda self, *o, e=raised, t=text: (_ for _ in ()).throw(e(t)),
        )
        with pytest.raises(ValueError) as wrong:
            made.first_use(2, 1)
        assert str(wrong.value) == f"cudnn.AttentionPrepareDecode declined the operands this adapter built for its first use: {text}", "an ERROR: the adapter's"

    def from_import(self, *operands):
        try:
            raise ImportError("No module named 'cutlass'")
        except ImportError as error:
            raise NotImplementedError("the attention preparation decode step requires nvidia-cutlass-dsl >= 4.8.0; the CuTe DSL is not installed") from error

    monkeypatch.setattr(type(made._build("cpu")), "__call__", from_import)
    with pytest.raises(NotImplementedError, match=r"requires nvidia-cutlass-dsl >= 4.8.0; the CuTe DSL is not installed$"):
        made.first_use(2, 1)  # what the call needs of the installation cannot be imported: the installation's, a SKIP
    monkeypatch.setattr(
        type(made._build("cpu")), "__call__", lambda self, *o: (_ for _ in ()).throw(NotImplementedError("serves a torch.bfloat16 cos_sin table"))
    )
    with pytest.raises(ValueError, match=r"^cudnn.AttentionPrepareDecode declined the operands this adapter built for its first use: serves a torch.bfloat16"):
        made.first_use(2, 1)  # any other decline of that call is of the operands this adapter built
    # a page-16 entry on the pool of 64-row pages: the entry declines the pool by name, and the adapter says whose operands they are
    made = planned(1)
    small = OnTheCpu(
        replace(row, form=tuple(("PAGE_SIZE", 16) if name == "PAGE_SIZE" else (name, value) for name, value in row.form)), model, real, made.options
    )
    small.check_support(1, 1)
    with pytest.raises(ValueError) as wrong:
        small.first_use(1, 1)
    assert str(wrong.value) == (
        "cudnn.AttentionPrepareDecode declined the operands this adapter built for its first use: k_pool must be [num_pages >= 1, H=4, page_size=16, D=256]"
    )
    # on a real device the adapter asks the entry, at plan time, for the launch of the run's row count: a decline is a wrong table
    made = planned(1024)
    monkeypatch.setattr(type(made._build("cpu")), "limit", 1000)
    with pytest.raises(ValueError) as wrong:
        OnTheCpu(row, model, real, made.options).check_support(1024, 1)
    assert str(wrong.value) == (
        "the provider table is wrong: the row of fe.attention_prepare_decode claims this run (1024 rows on compute capability 10.0), and "
        "cudnn.AttentionPrepareDecode declines it: this stand-in serves at most 1000 rows (tokens) per call; got 1024"
    )


def test_a_decline_of_the_attention_preparations_entry_at_load_is_an_error_that_says_whose_it_is(monkeypatch):
    """After the plan nothing skips.  At load the adapter of ``cudnn.AttentionPrepareDecode`` asks its entry for its
    kernel once more and launches it once, on scratch operands of the run's row count.  A typed decline there -- of the
    build or of that launch -- is a ``RuntimeError`` raised from the entry's own: never a return, never a SKIP.  Its
    sentence says whose the decline is: the installation's where the adapter can name a requirement that is not met (a
    CuTe DSL without the part's target, by cuDNN Frontend's public probe; on compute capability 10.7 a CuTe DSL whose
    compiler is older than CUDA 13.4, which the entry's page asks for there), the operands this adapter built where it
    can name none."""
    from types import SimpleNamespace

    import cudnn
    import cudnn.frost
    import cutlass.base_dsl.version_info as dsl_version

    from model_samples.blocks import DeviceFacts
    from model_samples.providers import fe
    from model_samples.qwen_dense import QwenDense, configs

    calls, declines = [], [""]
    row, model = configs.PROVIDERS["attention.prepare"][0], QwenDense.reduced(2)

    class OnTheCpu(fe.AttentionPrepareDecodeBlock):
        first_use_devices = ("cpu",)

    def planned(cc, sms):  # the block of a run on a real device of that class, behind its plan: the entry built, nothing declined
        del calls[:]
        declines[0] = ""
        monkeypatch.setattr(cudnn, "AttentionPrepareDecode", _prepare_entry(calls, decline=declines), raising=False)
        monkeypatch.setattr(fe, "_ENTRIES", {})
        made = OnTheCpu(row, model, DeviceFacts(cc, "stated device", sms, 180 * GIB, 80 * GIB), {"device": "cpu", "check_kernels": True})
        made.check_support(2, 1)
        made.first_use(2, 1)
        assert [kind for kind, *_ in calls] == ["prepare compile", "prepare launch"]
        return made

    needs = "the attention preparation decode step on compute capability 10.7 requires a CuTe DSL whose compiler is CUDA 13.4 or later"
    pool = "the attention preparation decode step serves at most 1024 rows (tokens) per call; got 1025"
    installation = "cudnn.AttentionPrepareDecode declines this installation at its first launch: "
    operands = "cudnn.AttentionPrepareDecode declined the operands this adapter built for its first launch: "
    for cc, sms, target, compiler, whose in (
        ((10, 0), 148, True, (12, 9), operands),  # the part's target is there, and either flavour of the CuTe DSL serves this class
        ((10, 0), 148, False, (13, 4), installation),  # a CuTe DSL without the part's target
        ((10, 3), 152, True, (12, 9), operands),
        ((10, 7), 212, True, (12, 9), installation),  # the entry's page: on this class a compiler of CUDA 13.4 or later
        ((10, 7), 212, True, (13, 3), installation),
        ((10, 7), 212, True, (13, 4), operands),
        ((10, 7), 212, True, (14, 0), operands),
    ):
        monkeypatch.setattr(cudnn.frost, "cutedsl_runtime_available", lambda cc=None, target=target: target)
        monkeypatch.setattr(dsl_version, "CUDA_VERSION", SimpleNamespace(major=compiler[0], minor=compiler[1]))
        made = planned(cc, sms)
        made.compile(2, 1)
        assert [kind for kind, *_ in calls[2:]] == ["prepare compile", "prepare launch"], "a load nothing declines: the kernel found, one launch"
        # the build declines at load
        declines[0] = needs
        del calls[:]
        with pytest.raises(RuntimeError) as declined:
            made.compile(2, 1)
        assert str(declined.value) == whose + needs, f"compute capability {cc}, compiler {compiler}: the sentence says whose the decline of the build is"
        assert type(declined.value.__cause__) is NotImplementedError and str(declined.value.__cause__) == needs, "raised from the entry's own decline"
        assert [kind for kind, *_ in calls] == ["prepare compile"], "declined by the build: nothing was launched"
        # the launch on the scratch operands declines at load
        declines[0] = ""
        with monkeypatch.context() as patched:
            patched.setattr(type(made._build("cpu")), "__call__", lambda self, *handed: (_ for _ in ()).throw(NotImplementedError(pool)))
            with pytest.raises(RuntimeError) as declined:
                made.compile(2, 1)
        assert str(declined.value) == whose + pool, f"compute capability {cc}, compiler {compiler}: the sentence says whose the decline of the launch is"
        assert type(declined.value.__cause__) is NotImplementedError and str(declined.value.__cause__) == pool
    # what is no typed decline of the entry is not reworded: an operand the entry refuses by type or value is the adapter's defect as it is raised
    made = planned((10, 0), 148)
    for raised in (TypeError, ValueError):
        with monkeypatch.context() as patched:
            patched.setattr(
                type(made._build("cpu")), "__call__", lambda self, *handed, error=raised: (_ for _ in ()).throw(error("positions must be torch.int64"))
            )
            with pytest.raises(raised, match=r"^positions must be torch.int64$"):
                made.compile(2, 1)


@pytest.mark.parametrize("sequences, tokens, position", [(1, 1, 0), (1, 1, 63), (3, 1, 64), (2, 3, 62), (16, 5, 125), (1, 6, 0)])
def test_the_adapters_operands_pass_the_entrys_own_host_check(monkeypatch, sequences, tokens, position):
    """What the adapter hands its entry, through the entry's REAL binding check (``check_binding``: types, dtypes, shapes,
    strides, alignment, overlap, the paged-pool contract) on CPU tensors: at the first position, on both sides of a page
    boundary, at several sequences, at several tokens per sequence across the boundary, at the largest verify step the
    rows serve, and for every sequence alone (the window of the pool a check cuts out)."""
    from cudnn.attention_prepare.frost import prepare_decode

    calls = []
    block, cache, qkv, params, state, model = _prepare_block(monkeypatch, calls, sequences=sequences, tokens=tokens)
    form = dict(positions="3d", mrope_section=(11, 11, 10), page_size=64, position_dtype="int64", slot_dtype="int64")
    cfg = prepare_decode.make_cfg(prepare_decode.AttentionPrepareGeometry(24, 4, 256, 64), 1e-6, **form)

    def checked(operands):
        try:
            return prepare_decode.check_binding(cfg, *(operands[name] for name in prepare_decode.OPERAND_NAMES))
        except NotImplementedError as error:
            if isinstance(error.__cause__, ImportError):  # the paged-pool contract's package imports the CuTe DSL
                pytest.skip(f"the entry's binding check needs the CuTe DSL: {error}")
            raise

    block(qkv, cache, position, params)
    assert checked(calls[-1][1]) == sequences * tokens
    for sequence in range(sequences):  # one sequence alone: its rows, and its window of the pool
        first, last = sequence * tokens, (sequence + 1) * tokens
        block(qkv[first:last], cache.rows(first, last), position, params)
        assert checked(calls[-1][1]) == tokens, sequence
    block.options["device"] = "cpu"
    entry, operands = block._scratch(sequences, tokens)
    entry(*operands)
    assert checked(calls[-1][1]) == sequences * tokens, "and the scratch operands of its first use"


def _suite_twin():
    """The helper module of the entry's own suite (``test/python/attention_prepare/frost/attention_prepare_utils.py``), loaded by
    its path: the twin the entry's device cells are held to."""
    import importlib.util

    path = MODELS.parents[1] / "test" / "python" / "attention_prepare" / "frost" / "attention_prepare_utils.py"
    spec = importlib.util.spec_from_file_location("attention_prepare_utils_of_the_entrys_suite", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_attention_preparations_twin_and_its_band_on_stated_tensors(monkeypatch):
    """The attention preparation at the kernel's rounding points, and the band its entry is judged with.  The twin is the
    twin of the entry's own suite, line by line: equal to it bit for bit, and its band's terms are that file's.  It is
    held against a version written out from the operation's definition, row by row.  Its position is the integer the
    step hands it: a state whose tables are shifted leaves its bytes as they were.  The kernel's chain in fp32 is inside
    the band; outside, each alone and by its own rule: a rotated element and a pass-through element two units off, of
    a query row and of a key row the call wrote, a gate bit, a value bit, the key rows of two neighbouring slots
    exchanged, a pool row no position of the call names, and a value that is not finite.  The epsilon the twin adds is
    the fp32 value the kernel adds: one searched row, stated here, is another byte with the fp64 value."""
    import math

    from cudnn.attention_prepare.frost import prepare_decode

    from model_samples import tie_out
    from model_samples.providers import torch_ref
    from model_samples.qwen_dense import configs

    # the twin's rows against the suite's twin, on that suite's own draws: the same bytes and the same band
    suite = _suite_twin()
    cfg = prepare_decode.make_cfg(
        suite.GEOMETRY, suite.EPS, positions="3d", mrope_section=suite.SECTION, page_size=64, position_dtype="int64", slot_dtype="int64"
    )
    # (1,024 random tokens: among millions of elements a root taken in fp32 moves some to the other bf16 value; and operands whose
    # rotary products underflow, where the sign of a zero says which product of the second half is the fused one)
    draws = [(kind, suite.make_operands(tokens, seed=seed, kind=kind)) for kind, tokens, seed in (("random", 7, 0), ("structured", 9, 1), ("edge", 6, 2))]
    draws += [("random, 1,024 tokens", suite.make_operands(1024, seed=3, kind="random")), ("underflow", suite.underflow_operands(8, seed=4))]
    for kind, ops in draws:
        tokens = ops["positions"].shape[-1]
        at = ops["positions"][0].clone()
        ops["positions"] = at.expand(3, tokens).contiguous()  # one position per token: three equal rows, as a text position is
        table = ops["cos_sin"][at]
        cos, sin = table[:, None, :32], table[:, None, 32:]
        for name, weight in (("q", "q_weight"), ("k", "k_weight")):
            want, want_band = suite.twin_rows(ops[name], ops[weight], ops["positions"], ops["cos_sin"], cfg)
            got, band = torch_ref.prepare_rows(ops[name], ops[weight], cos, sin, suite.EPS)
            assert torch.equal(suite.bits(got), suite.bits(want)), f"{kind}, {name}: the twin is the suite's twin, bit for bit"
            assert band.dtype == torch.float64 and torch.equal(band, want_band), f"{kind}, {name}: and its band is the suite's"
    one_row = prepare_decode.make_cfg(suite.GEOMETRY, suite.EPS, positions="1d", page_size=64, position_dtype="int64", slot_dtype="int64")
    three, _ = torch_ref.prepare_rows(ops["q"], ops["q_weight"], cos, sin, suite.EPS)  # the last draw: every rotary product underflows
    got, _ = torch_ref.prepare_rows(ops["q"], ops["q_weight"], cos, sin, suite.EPS, order="1d")
    want, _ = suite.twin_rows(ops["q"], ops["q_weight"], at.reshape(1, -1), ops["cos_sin"], one_row)
    assert torch.equal(suite.bits(got), suite.bits(want)) and not torch.equal(suite.bits(got), suite.bits(three)), "the order of one position row: other zeros"
    values = torch.tensor([0.0, 1.0, 1.5, -3.0, 2.0**-130, 255.0, 256.0, 1e30], dtype=torch.float64)
    assert torch.equal(torch_ref.bf16_unit(values), suite.unit(values)), "one bf16 unit at a magnitude: the suite's"
    assert torch_ref.bf16_unit(torch.tensor([1.0, 1.99, 2.0], dtype=torch.float64)).tolist() == [2.0**-7, 2.0**-7, 2.0**-6]
    # the epsilon under the root is the fp32 value of the model's, as the kernel adds it.  The suite's draws are of ordinary magnitude,
    # where either value gives the same root; a row whose mean of squares is of the epsilon's order tells them apart.  ONE searched
    # row, stated whole: its first 128 elements are 37 * 2**-18 and its last 128 are 167 * 2**-18, under the weights i / 256 -- with
    # the fp32 value the bytes are the suite's twin's, with the fp64 value the root is another fp32 value and element 100 another byte
    small = suite.make_operands(1, seed=5, kind="random")
    where = small["positions"][0].clone()
    small["positions"] = where.expand(3, 1).contiguous()
    small["q"][:] = torch.cat((torch.full((128,), 37 * 2.0**-18), torch.full((128,), 167 * 2.0**-18))).bfloat16()
    small["q_weight"][:] = (torch.arange(256) / 256).bfloat16()
    table = small["cos_sin"][where]
    want, _ = suite.twin_rows(small["q"], small["q_weight"], small["positions"], small["cos_sin"], cfg)
    got, _ = torch_ref.prepare_rows(small["q"], small["q_weight"], table[:, None, :32], table[:, None, 32:], suite.EPS)
    assert torch.equal(suite.bits(got), suite.bits(want)), "the searched row: the twin adds the fp32 value of the epsilon, as the suite's twin does"
    searched, gain = small["q"][0, 0].float(), small["q_weight"].float() + 1.0
    assert searched.tolist() == [37 * 2.0**-18] * 128 + [167 * 2.0**-18] * 128 and gain.tolist() == [1.0 + i / 256 for i in range(256)], "as stated"
    mean = sum(element**2 for element in searched.tolist()) / 256  # exact: every square and the sum are whole multiples of 2**-36
    roots = []
    for epsilon, moved in ((float(torch.tensor(suite.EPS, dtype=torch.float32)), []), (suite.EPS, [100])):
        roots.append(torch.tensor(1.0 / math.sqrt(mean + epsilon), dtype=torch.float32))
        normed = ((searched * roots[-1]) * gain).bfloat16()  # the elements behind the rotary part are the normalised values themselves
        other = (suite.bits(normed[64:]) != suite.bits(got[0, 0, 64:])).nonzero().flatten()
        assert (other + 64).tolist() == moved, f"the epsilon {epsilon!r}: the fp32 value gives the twin's bytes, the fp64 value another byte at element 100"
    assert float(roots[0]) != float(roots[1]), "two epsilons, two roots in fp32"

    # one call of the adapter over the kernel's fp32 chain, and of its twin: two sequences x three tokens across a page boundary
    calls = []
    block, cache, qkv, params, state, model = _prepare_block(monkeypatch, calls, sequences=2, tokens=3)
    g, position, rows = model.geometry, 62, 6
    reference = torch_ref.AttentionPrepare(configs.PROVIDERS["attention.prepare"][1], model, block.facts, block.options)
    twin = block.twin(reference)
    before = cache.pool.clone()
    out = tuple(t.clone() for t in block(qkv, cache, position, params))
    after = cache.pool.clone()
    cache.pool.copy_(before)
    ref = tuple(t.clone() for t in twin(qkv, cache, position, params))
    ref_after = cache.pool.clone()
    assert tuple(ref[0].shape) == (rows, 24, 256) and tuple(ref[1].shape) == (rows, 6144) and ref[0].dtype == ref[1].dtype == torch.bfloat16
    # the bands the twin hands the judge are the bands of its rows for THESE rows, element for element, and no multiple of them
    turns = [torch_ref.rope_cos_sin(position + token, g.rotary_dim, g.rope_theta, "cpu") for token in range(3)]
    cos, sin = (torch.stack([turn[part] for turn in turns]).repeat(2, 1)[:, None] for part in (0, 1))  # row (sequence, token): its token's position
    for name, x, weight, kept in (
        ("query", qkv[:, :12288].view(rows, 24, 512)[..., :256], params.q_norm, twin.terms.q_band),
        ("key", qkv[:, 12288:13312].view(rows, 4, 256), params.k_norm, twin.terms.k_band),
    ):
        band = torch_ref.prepare_rows(x, weight, cos, sin, model.norm_eps)[1]
        assert band[..., :64].amin() > 0 and torch.equal(kept, band), f"the band of the {name} rows that the judge is handed is the band of the twin's rows"

    # ... the twin against the operation written out row by row from its definition (mean and root in fp64, rounded to fp32 once)
    eps32 = float(torch.tensor(1e-6, dtype=torch.float32).double())

    def defined(x, weight, at):  # one head row [D] at one position
        mean = sum(float(element) ** 2 for element in x.float().tolist()) / 256
        normed = ((x.float() * torch.tensor(1.0 / math.sqrt(mean + eps32), dtype=torch.float32)) * (weight.float() + 1.0)).bfloat16()
        cos, sin = torch_ref.rope_cos_sin(at, g.rotary_dim, g.rope_theta, "cpu")
        n1, n2 = normed[:32].float(), normed[32:64].float()
        first = (n1.double() * cos.double() - (n2 * sin.float()).double()).float().bfloat16()
        second = (n2.double() * cos.double() + (n1 * sin.float()).double()).float().bfloat16()
        return torch.cat((first, second, normed[64:]))

    pairs = qkv[:, :12288].view(rows, 24, 512)
    keys = qkv[:, 12288:13312].view(rows, 4, 256)
    paged = ref_after.view(2, 3, 4, 64, 512)
    for row in range(rows):
        sequence, token = divmod(row, 3)
        at = position + token
        for head in (0, 11, 23):
            assert torch.equal(ref[0][row, head].view(torch.int16), defined(pairs[row, head, :256], params.q_norm, at).view(torch.int16)), (row, head)
        for head in range(4):
            written = paged[sequence, at // 64, head, at % 64]
            assert torch.equal(written[:256].view(torch.int16), defined(keys[row, head], params.k_norm, at).view(torch.int16)), (row, head)
            assert torch.equal(written[256:].view(torch.int16), qkv[row, 13312 + 256 * head : 13568 + 256 * head].view(torch.int16)), "the value: a bit copy"
    assert torch.equal(ref[1].view(torch.int16), pairs[..., 256:].reshape(rows, 6144).view(torch.int16)), "the gates: a bit copy"
    live = torch.zeros(6, 64, dtype=torch.bool)
    for sequence in range(2):
        for at in (62, 63, 64):
            live[sequence * 3 + at // 64, at % 64] = True
    assert int(live.sum()) == 6 and not ref_after.permute(0, 2, 1, 3)[~live].any(), "the twin wrote the rows of the call's positions and no other"

    # ... a state whose tables are shifted: the twin never reads them
    shifted_block, shifted, _, _, shifted_state, _ = _prepare_block(monkeypatch, [], sequences=2, tokens=3)
    shifted_state.table.positions += 1
    shifted_state.table.slots += 1
    moved = tuple(t.clone() for t in shifted_block.twin(reference)(qkv, shifted, position, params))
    assert all(torch.equal(a.view(torch.int16), b.view(torch.int16)) for a, b in zip(moved, ref)) and torch.equal(
        shifted.pool.view(torch.int16), ref_after.view(torch.int16)
    )

    def judged(q=out[0], gate=out[1], pool=after, against=twin):
        stats = tie_out.BlockStats("attention_prepare")
        tie_out.BlockTap._judge(stats, (q, gate), ref, [(pool,)], [(ref_after,)], 1, against)
        return stats

    clean = judged()
    assert clean.ok and clean.finite and clean.elements == rows * 28 * 256 and (clean.outside, clean.keys_outside) == (0, 0), clean.line()
    assert (
        clean.gates_equal and clean.values_equal and clean.state_equal and 0 < clean.other_value < clean.elements // 100
    ), "the fp32 root moves a few elements by a unit"
    assert clean.line().startswith(
        f"{clean.elements:,} query and key elements, {clean.other_value:,} of them not the twin's byte; 0 query and 0 key elements outside"
    )
    exact = judged(q=ref[0], pool=ref_after)
    assert exact.ok and exact.other_value == 0, "the twin against itself"

    def altered(tensor, at, units):  # one element moved by so many of its own bf16 units
        changed = tensor.clone()
        value = changed[at].double()
        changed[at] = float(value + units * torch_ref.bf16_unit(value))
        return changed

    one_off = judged(q=altered(ref[0], (0, 0, 200), 1), pool=ref_after)
    assert one_off.ok and one_off.other_value == 1, "a pass-through element one unit off: inside"
    toward_zero = -2 if float(ref[0][0, 0, 200]) > 0 else 2  # the unit is taken at the larger magnitude: the twin's
    two_off = judged(q=altered(ref[0], (0, 0, 200), toward_zero), pool=ref_after)
    assert not two_off.ok and two_off.outside == 1, "a pass-through element two units off: outside, its band is one unit"
    for name, at in (("a pass-through element", (0, 0, 200)), ("a rotated element", (4, 23, 40))):
        far = ref[0][at].double() + 3 * (torch_ref.bf16_unit(ref[0][at].double()) + twin.terms.q_band[at])
        bad = ref[0].clone()
        bad[at] = float(far)
        stats = judged(q=bad, pool=ref_after)
        assert not stats.ok and (stats.outside, stats.keys_outside) == (1, 0) and stats.gates_equal and stats.values_equal and stats.state_equal, name
        assert stats.line().split("; ")[1] == "1 query and 0 key elements outside their band", name
    rotated = judged(q=altered(ref[0], (4, 23, 40), 2), pool=ref_after)
    assert rotated.outside == 1 - int(
        float(twin.terms.q_band[4, 23, 40]) >= torch_ref.bf16_unit(ref[0][4, 23, 40].double())
    ), "two units: outside unless the band of that element admits them"

    # the key rows are held element by element, as the queries are: ONE key element of a row the call wrote, moved in the twin's own pool
    def keyed(at, value):  # the verdict on the twin's own outputs and pool with the key element ``at`` = (row of the step, head, element) set to ``value``
        (row, head, element), paged = at, ref_after.clone().view(2, 3, 4, 64, 512)
        sequence, token = divmod(row, 3)
        slot = position + token
        assert float(paged[sequence, slot // 64, head, slot % 64, element]) == float(twin.terms.k[at]), "the twin's key element of that row of the step"
        paged[sequence, slot // 64, head, slot % 64, element] = float(value)
        return judged(q=ref[0], pool=paged.view(6, 4, 64, 512))

    plain_key, rotated_key = (0, 0, 200), (4, 3, 40)
    assert float(twin.terms.k_band[plain_key]) == 0.0 < float(twin.terms.k_band[rotated_key]) < float(twin.terms.k_band.amax()), "a band per key element"
    value = twin.terms.k[plain_key].double()
    one_off = keyed(plain_key, value + torch_ref.bf16_unit(value))
    assert one_off.ok and one_off.other_value == 1, "a key element that is not rotated one unit off: inside"
    toward_zero = -2 if float(value) > 0 else 2  # the unit is taken at the larger magnitude: the twin's
    two_off = keyed(plain_key, value + toward_zero * torch_ref.bf16_unit(value))
    outside = (two_off.outside, two_off.keys_outside)
    assert not two_off.ok and outside == (0, 1), "a key element that is not rotated two units off: outside, its band is one unit"
    for name, at in (("a key element that is not rotated", plain_key), ("a rotated key element", rotated_key)):
        value = twin.terms.k[at].double()
        stats = keyed(at, value + 3 * (torch_ref.bf16_unit(value) + twin.terms.k_band[at]))
        assert not stats.ok and (stats.outside, stats.keys_outside) == (0, 1) and stats.gates_equal and stats.values_equal and stats.state_equal, name
        assert stats.line().split("; ")[1] == "0 query and 1 key elements outside their band", name
    # a rotated key element is held to ITS OWN band and to no multiple of it: the first that tells one band from two (a band of half its unit or more),
    # moved toward zero by whole bf16 units (the unit is then the twin's) -- the most units inside `unit + band` are inside, one more is outside
    units = torch_ref.bf16_unit(twin.terms.k.double())
    tells = (twin.terms.k_band >= units / 2) & (twin.terms.k.double().abs() > (3 + torch.floor(twin.terms.k_band / units)) * units)
    assert tells[..., 64:].sum() == 0 and tells.any(), "a rotated key element whose band tells one band from two"
    at = tuple(int(i) for i in tells.nonzero()[0])
    value, unit, band = twin.terms.k[at].double(), float(units[at]), float(twin.terms.k_band[at])
    most = 1 + int(band // unit)  # the most whole units that stay inside unit + band
    sign = -1.0 if float(value) > 0 else 1.0
    inside = keyed(at, value + sign * most * unit)
    assert inside.ok and inside.keys_outside == 0, "a rotated key element moved by the most units its band admits: inside"
    one_more = keyed(at, value + sign * (most + 1) * unit)
    outside = (one_more.outside, one_more.keys_outside)
    assert not one_more.ok and outside == (0, 1), "a rotated key element one unit past its own band: outside, the band is not doubled"
    # ... and the same of a rotated QUERY element, the first whose band is a WHOLE unit or more: the most units it admits are then two
    # or more, so that a band which is narrower than the twin's admits fewer and a band which is wider admits one more -- held on both sides
    units = torch_ref.bf16_unit(ref[0].double())
    tells = (twin.terms.q_band >= units) & (ref[0].double().abs() > (3 + torch.floor(twin.terms.q_band / units)) * units)
    assert tells[..., 64:].sum() == 0 and tells.any(), "a rotated query element whose band is a whole unit or more"
    at = tuple(int(i) for i in tells.nonzero()[0])
    value, unit, band = ref[0][at].double(), float(units[at]), float(twin.terms.q_band[at])
    most, sign = 1 + int(band // unit), -1.0 if float(ref[0][at]) > 0 else 1.0
    assert most >= 2 and 1 + int(band / 2 // unit) < most, "half that band admits fewer units: a judge that halves it fails the next line"
    moved = ref[0].clone()
    moved[at] = float(value + sign * most * unit)
    assert judged(q=moved, pool=ref_after).ok, "a rotated query element moved by the most units its band admits: inside"
    moved[at] = float(value + sign * (most + 1) * unit)
    one_more = judged(q=moved, pool=ref_after)
    outside = (one_more.outside, one_more.keys_outside)
    assert not one_more.ok and outside == (1, 0), "a rotated query element one unit past its own band: outside, the band is not doubled"
    bit = out[1].clone()
    bit.view(torch.int16)[5, 6143] ^= 1
    stats = judged(gate=bit)
    assert not stats.ok and not stats.gates_equal and (stats.outside, stats.keys_outside, stats.values_equal, stats.state_equal) == (0, 0, True, True)
    assert "gate rows DIFFER" in stats.line()
    paged = after.clone().view(2, 3, 4, 64, 512)
    paged.view(torch.int16)[1, 1, 3, 0, 511] ^= 1  # a value bit of sequence 1 at position 64
    stats = judged(pool=paged.view(6, 4, 64, 512))
    assert not stats.ok and not stats.values_equal and (stats.outside, stats.keys_outside, stats.gates_equal, stats.state_equal) == (0, 0, True, True)
    assert "value rows DIFFER" in stats.line()
    paged = after.clone().view(2, 3, 4, 64, 512)
    held = paged[0, 0, :, 62, :256].clone()  # the key rows of positions 62 and 63 of sequence 0: two neighbouring slots, exchanged
    paged[0, 0, :, 62, :256] = paged[0, 0, :, 63, :256]
    paged[0, 0, :, 63, :256] = held
    stats = judged(pool=paged.view(6, 4, 64, 512))
    assert not stats.ok and stats.keys_outside > 1000 and (stats.outside, stats.gates_equal, stats.values_equal, stats.state_equal) == (0, True, True, True)
    for half in (0, 256):  # a row no position of the call names: its key half, its value half
        paged = after.clone().view(2, 3, 4, 64, 512)
        paged[1, 2, 0, 5, half] = 1.0
        stats = judged(pool=paged.view(6, 4, 64, 512))
        assert not stats.ok and not stats.state_equal and (stats.outside, stats.keys_outside, stats.gates_equal, stats.values_equal) == (0, 0, True, True), half
        assert "the rest of the pool WRITTEN" in stats.line()
    nan = out[0].clone()
    nan[2, 3, 100] = float("nan")
    stats = judged(q=nan)
    assert not stats.ok and not stats.finite and stats.outside == 1 and "NON-FINITE VALUES" in stats.line()
    paged = after.clone().view(2, 3, 4, 64, 512)
    paged[0, 1, 2, 0, 17] = float("inf")  # in a key row of the call
    stats = judged(pool=paged.view(6, 4, 64, 512))
    assert not stats.ok and not stats.finite and stats.keys_outside == 1
    # a gate element and a written value element that are not finite on the entry's side alone: another byte than the twin's, and not finite
    nan = out[1].clone()
    nan[5, 6143] = float("nan")
    stats = judged(gate=nan)
    assert not stats.ok and not stats.gates_equal and not stats.finite, stats.line()
    assert stats.line().endswith("; gate rows DIFFER; value rows equal; the rest of the pool untouched; NON-FINITE VALUES")
    paged = after.clone().view(2, 3, 4, 64, 512)
    paged[1, 1, 3, 0, 511] = float("inf")  # the last value element of sequence 1 at position 64
    stats = judged(pool=paged.view(6, 4, 64, 512))
    assert not stats.ok and not stats.values_equal and not stats.finite, stats.line()
    assert stats.line().endswith("; gate rows equal; value rows DIFFER; the rest of the pool untouched; NON-FINITE VALUES")
    with pytest.raises(ValueError, match="the band attention_prepare is judged against the preparation at the kernel's rounding points"):
        judged(against=reference)
    assert "attention_prepare" in tie_out.BANDS and "one bf16 unit" in tie_out.BANDS["attention_prepare"] and "bitwise" in tie_out.BANDS["attention_prepare"]


@pytest.mark.parametrize(
    "where, value",
    [("gate", float("nan")), ("value", float("inf")), ("value", float("nan")), ("unwritten", float("inf"))],
    ids=[
        "the same NaN in a gate row",
        "the same infinity in a written value row",
        "the same NaN in a written value row",
        "an infinity in a row no position names",
    ],
)
def test_a_copied_value_that_is_not_finite_is_outside_the_attention_preparations_band(monkeypatch, where, value):
    """The gate rows and the value rows of the attention preparation are bit copies of the call's operand, and the entry
    and its twin copy the same operand: a NaN or an infinity in it is the same byte on both sides, and equal bytes do
    not say that a value is finite.  The judge holds the gate rows and the value rows a call wrote finite, as it holds
    the queries and the key rows: such a call is not ok and not finite, on a line that still reads the rows equal and
    ends in ``NON-FINITE VALUES``.  What a call wrote are the rows of the pool at its positions -- for every row of the
    step the page and the slot of its position, formed from host integers.  A value that is not finite in a row no
    position of the call names, there before the call on both sides, is no value of this call: the call that wrote
    that row was judged with it.  One token on a page of 64 slots, and two sequences of three tokens across a page
    boundary; the value stands in the last row of the step."""
    from model_samples import tie_out
    from model_samples.providers import torch_ref
    from model_samples.qwen_dense import configs

    for sequences, tokens, position in ((1, 1, 5), (2, 3, 62)):
        block, cache, qkv, params, _, model = _prepare_block(monkeypatch, [], sequences=sequences, tokens=tokens)
        g, row = model.geometry, sequences * tokens - 1  # the last row of the step: the last token of the last sequence
        heads, kv_heads, width = g.attn_heads, g.attn_kv_heads, g.attn_head_dim
        twin = block.twin(torch_ref.AttentionPrepare(configs.PROVIDERS["attention.prepare"][1], model, block.facts, block.options))
        own = (sequences - 1) * (cache.pool.shape[0] // sequences)  # the first page of the last sequence
        page, slot = own + (position + tokens - 1) // 64, (position + tokens - 1) % 64  # the row of the pool that row's position names
        beyond = (own + (position + tokens) // 64, 2, (position + tokens) % 64)  # value head 2 at the next position: no row of this step
        operand, before = qkv.clone(), cache.pool.clone()
        if where == "gate":  # head 3, element 17 of its gate row, which stands behind the head's query elements
            operand[row, 3 * 2 * width + width + 17] = value
        elif where == "value":  # value head 2, element 40: behind the queries with their gates and the keys
            operand[row, (2 * heads + kv_heads) * width + 2 * width + 40] = value
        else:  # in the pool before the call: a key element and a value element of that row
            before[beyond + (40,)] = before[beyond + (width + 40,)] = value
        results = []
        for call in (block, twin):  # the entry's stand-in and the twin, each from the same pool
            cache.pool.copy_(before)
            results.append((tuple(t.clone() for t in call(operand, cache, position, params)), cache.pool.clone()))
        (out, after), (ref, ref_after) = results
        stats = tie_out.BlockStats("attention_prepare")
        tie_out.BlockTap._judge(stats, out, ref, [(after,)], [(ref_after,)], 1, twin)
        # both sides hold the value in the same bytes, and nothing else of the call is not finite
        strays = [int((~torch.isfinite(t)).sum()) for t in (out[0], out[1], after[page, :, slot], after)]  # queries, gates, the row written, the pool
        assert strays == {"gate": [0, 1, 0, 0], "value": [0, 0, 1, 1], "unwritten": [0, 0, 0, 2]}[where], (sequences, strays)
        assert tie_out.bytes_equal(out[1], ref[1]) and tie_out.bytes_equal(after[..., width:], ref_after[..., width:]), sequences
        assert (stats.outside, stats.keys_outside, stats.gates_equal, stats.values_equal, stats.state_equal) == (0, 0, True, True, True), stats.line()
        equal = "; 0 query and 0 key elements outside their band; gate rows equal; value rows equal; the rest of the pool untouched"
        if where == "unwritten":
            assert stats.ok and stats.finite and stats.line().endswith(equal), (sequences, stats.line())
        else:
            assert not stats.ok and not stats.finite and stats.line().endswith(equal + "; NON-FINITE VALUES"), (sequences, stats.line())


def _attention_layers(count=16, form=None, symbol=None):
    """The sites of a step of ``count`` attention layers between a gather and the host path, as a target stack plans them: the
    in-projection and the attention preparation on their entries (exact, each with its entry's stated symbol and launch
    shape), the attention core on its reference chain, the gate on its entry.  ``(sites, the preparation's symbol)``."""
    from cudnn.attention_prepare import AttentionPrepareGeometry

    from model_samples.census import Facts, Kernel, Site
    from model_samples.qwen_dense import configs

    row = configs.PROVIDERS["attention.prepare"][0]
    form = row.form if form is None else form
    built = dict(
        geometry=AttentionPrepareGeometry(24, 4, 256, 64), positions="3d", mrope_section=(11, 11, 10), page_size=64, position_dtype="int64", slot_dtype="int64"
    )
    symbol = symbol or _prepare_symbol(dict(built, pdl=False, pdl_release=False))
    stated = {**dict(row.form), "EPS": 1e-6}
    projection = Facts((_projection_symbol({"OUT_FEATURES": 14336, "IN_FEATURES": 5120}),), {"SPLIT_K": 1}, PROJECTION_SHAPE[14336])
    sites = [Site("embedding", -1, (Kernel("embedding_gather", "PyTorch", "exact", launches=1),), "torch.index_select")]
    prepared, gate = Facts((symbol,), stated, _prepare_shape(1)), Facts((GATE_SYMBOL,), {"THREADS": 256}, GATE_SHAPE)

    def exact(name):
        return (Kernel(name, "cuDNN FE", "exact", launches=1, shape="entry"),)

    for layer in range(count):
        sites.append(Site("projection.fp8.in", layer, exact("fp8_projection"), "cudnn.Fp8ProjectionDecode", (("SPLIT_K", 1),), projection))
        sites.append(Site("attention.prepare", layer, exact("attention_prepare"), "cudnn.AttentionPrepareDecode", tuple(form), prepared))
        sites.append(Site("attention.core", layer, (Kernel("pytorch", "PyTorch"), Kernel("cublas", "cuBLAS")), "torch_ref.attention"))
        sites.append(Site("attention.gate_quantize", layer, exact("gate_quant"), "cudnn.AttentionGateQuantDecode", (("THREADS", 256),), gate))
    sites.append(Site("token_choice", -1, (Kernel("pytorch", "PyTorch", "host"),), "torch.argmax"))
    return sites, symbol


def test_the_kernel_check_of_the_attention_preparation_on_constructed_records():
    """The 16 launches of the attention preparation's entry in a step of 16 attention layers: in the wiring order, under the
    entry's symbol and in its launch shape, with nothing between an in-projection and the entry, the check holds, and
    the launches of the chain that reads the pool behind the entry are its families'.  A launch between the
    in-projection and the entry, a missing launch, another launch shape, a call served in another form and a gather
    launched by the chain behind the entry each fail with their own sentence.  The entry's kernel has a pattern of its
    own, in front of the one for any other cuDNN Frontend kernel."""
    from model_samples import census
    from model_samples.census import Call, Launch
    from model_samples.providers import factories

    sites, symbol = _attention_layers()
    record = Launch(symbol, "kernel", 0.0, 0.5, *_prepare_shape(1), 32)
    assert census.canonical(record) == "attention_prepare" and [kernel_id for kernel_id, _ in census.FAMILIES].index("attention_prepare") < [
        kernel_id for kernel_id, _ in census.FAMILIES
    ].index("cudnn_fe"), "a pattern of its own, in front of the pattern for any other kernel of cuDNN Frontend"
    assert census.canonical(record._replace(name="cudnn_kernel_frost_attention_gate_quant_x")) == "gate_quant"
    exact = {kernel.id for factory in factories().values() for kernel in factory.kernels if kernel.relation == "exact"}
    assert "attention_prepare" in exact and exact <= {kernel_id for kernel_id, _ in census.FAMILIES}, "every exactly judged kernel id has a pattern of its own"
    for kernel_id in exact:
        named = [name for name, _ in census.FAMILIES]
        assert named.count(kernel_id) == 1, kernel_id

    def step(defect="", layer=5):
        layer = 15 if defect in ("missing", "gather") else layer  # a launch more or less of a kernel is placed by aligning two repeating sequences
        at = _Clock()
        records = [at(GATHER, GATHER_SHAPE)]
        for index in range(16):
            here = defect if index == layer else ""
            records.append(at(_projection_symbol({"OUT_FEATURES": 14336, "IN_FEATURES": 5120}), PROJECTION_SHAPE[14336]))
            if here == "between":
                records.append(at(ELEMENTWISE))
            if here != "missing":
                records.append(at(symbol, ((1, 28, 1), (256, 1, 1), 16) if here == "shape" else _prepare_shape(1)))
            records += [at(ELEMENTWISE), at(GEMV), at(SOFTMAX), at(GEMM_FP64)]  # the chain that reads the pool: a cast, the two multiplies, the softmax
            if here == "gather":
                records.append(at(GATHER, GATHER_SHAPE))
            records.append(at(GATE_SYMBOL, GATE_SHAPE))
        return [*records, at(ARGMAX), at(HOST_READ, kind="copy")]

    clean = census.compare([step(), step()], sites)
    assert clean.ok and clean.exact == 49 and clean.launches == [114, 114], clean.sentences()
    assert [(row.blocks, row.relation, len(row.sites)) for row in clean.table if "attention.prepare" in row.blocks or "attention.core" in row.blocks] == [
        (("attention.prepare",), "exact", 16),
        (("attention.core",), "fallback", 16),
    ], "an exact row of its own, and the chain behind it as the row of its place in the wiring"
    assert census.check_forms(sites, _log(sites, 2)) == []
    sentences = {
        "between": f"launched {census._short(ELEMENTWISE)} between projection.fp8.in (layer 5) and attention.prepare (layer 5): a kernel of the fallback "
        "family pytorch, which no row there declares; nothing may sit between two exact rows",
        "missing": "attention.prepare (layer 15) did not launch attention_prepare: the step has 15 launch(es) of attention_prepare, the plan 16 = 16 call(s) "
        "x 1 per call",
        "shape": "attention.prepare (layer 5) launched attention_prepare with grid (1, 28, 1), block (256, 1, 1), shared memory 16 B; "
        "cudnn.AttentionPrepareDecode states grid (1, 28, 1), block (128, 1, 1), shared memory 16 B",
        "gather": f"launched embedding_gather ({census._short(GATHER)}) after attention.prepare (layer 15), where the plan has no such launch: the step has "
        "2 launch(es) of embedding_gather, the plan 1 = 1 call(s) x 1 per call",
    }
    for defect, sentence in sentences.items():
        found = census.compare([step(), step(defect)], sites)
        assert [text for _, text in found.failures] == [sentence] and found.failures[0][0] == 1, defect
    # another form in the call log: the same symbol and the same launch shape would not show it
    log = _log(sites, 2)
    one_row = (("POSITIONS", "1d"), ("SECTION", None), *sites[2].form[2:])
    log[1][2] = Call("attention.prepare", 0, 1, "cudnn.AttentionPrepareDecode #2", one_row)
    assert census.check_forms(sites, log) == [
        f"attention.prepare (layer 0): the adapter served the call with the form{census.form_text(one_row)} (cudnn.AttentionPrepareDecode #2); "
        f"the row says {PREPARE_FORM}"
    ]
    # a row that names another form than its entry states fails on the objects, field by field
    other, _ = _attention_layers(form=one_row)
    assert census.check_forms(other)[:2] == [
        "attention.prepare: the row's form says POSITIONS '1d', cudnn.AttentionPrepareDecode states POSITIONS '3d' for this run",
        "attention.prepare: the row's form says SECTION None, cudnn.AttentionPrepareDecode states SECTION (11, 11, 10) for this run",
    ]


def test_the_counts_with_the_attention_preparation_on_its_entry():
    """The census names the entry as a PORT of the kernel it replaces, in the form the row names: the three numbers of the
    set move by the 16 launches of that kernel -- 194 + 289 + 1 for the plain step, 146 + 337 + 1 for the verify step
    -- and the recorded launch shapes stay.  The plans of both target stacks follow, each count with its split (the cases
    below: 64 layers and two, a step of one token and of six), with what is held to a recorded launch shape and what by
    membership, and each beside the same plan without the row: the block on its reference chain reaches the launches of
    the entry fewer, one per attention layer.  The attention core is what is left ``not yet``."""
    from model_samples import blocks, census
    from model_samples.qwen_dense import QwenDense, configs

    target = QwenDense.target()
    meta = target.data["kernels"]["attention_prepare"]
    assert (meta["origin"], meta["relation"], meta["kind"], meta["entry"]) == ("cuDNN Frontend", "target", "port", "cudnn.AttentionPrepareDecode")
    row = configs.PROVIDERS["attention.prepare"][0]
    assert meta["form"] == {name: list(value) if isinstance(value, tuple) else value for name, value in row.form}, "the census names the row's form"
    assert isinstance(meta["form"]["SECTION"], list) and isinstance(dict(row.form)["SECTION"], tuple), "a list in the file, a tuple in the table: one form"
    assert "on compute capability 10.0 -- no differing byte" in meta["tie_out"] and "is not a test of this repository" in meta["tie_out"]
    for path in ("docs/operations/attention/attention_prepare_decode.md", "test/python/attention_prepare/frost"):
        assert path in meta["tie_out"] and (MODELS.parents[1] / path).exists(), path
    assert len(meta["differences"]) == 7 and sum("a slot at or past the pool's end writes nothing" in item for item in meta["differences"]) == 1
    assert sum("were not timed against each other" in item for item in meta["differences"]) == 1, "its instruction stream is its own: said, with no timing"
    plain, verify = target.data["configurations"]["plain"]["cc10.0/sm148"], target.data["configurations"]["verify"]["cc10.0/sm148"]
    assert plain["numbers"] == {"launches": 484, "code": 194, "port": 290, "stand_in": 0}
    assert verify["numbers"] == {"launches": 484, "code": 146, "port": 338, "stand_in": 0}
    full, two, four = (QwenDense().geometry.layer_kinds, QwenDense.reduced(2).geometry.layer_kinds, QwenDense.reduced(4).geometry.layer_kinds)
    assert target.numbers(plain["rows"]["1"], full) == (484, 194, 290, 0) and target.numbers(verify["rows"]["6"], full) == (484, 146, 338, 0)
    assert target.numbers(plain["rows"]["1"], two) == (20, 8, 12, 0) and target.numbers(plain["rows"]["1"], four) == (34, 14, 20, 0)
    assert target.numbers(verify["rows"]["6"], two) == (20, 7, 13, 0)
    # the recorded launch shapes stay as they were recorded: they are the entry's own
    for section, rows in ((plain, 1), (verify, 6)):
        (entry,) = [facts for block, _, facts in section["rows"][str(rows)]["attention_layer"] if block == "attention.prepare"]
        assert entry == {"launch": [[rows, 28, 1], [128, 1, 1], 16]} and tuple(map(tuple, entry["launch"][:2])) + (16,) == _prepare_shape(rows)
        planned, model, _ = _target_plan(2)
        assert tuple(planned["attention.prepare"].launch_facts(rows).shape) == _prepare_shape(rows), "what the public entry states for that row count"
        for name in ("rotary_dim", "rotary_section"):
            assert section["geometry"][name] == (64 if name == "rotary_dim" else [11, 11, 10]), "a section names the rotary widths it was recorded with"
    facts = blocks.DeviceFacts((10, 0), "stated device", 148, 0, 0)
    split = "{} the target's own compiled kernels + {} cuDNN Frontend entries with byte-equal results"
    decode_left = {"projection.fp8.out": 64, "mlp.down": 64, "attention.core": 16, "output_head": 1}
    options = {"device": "cpu", "check_kernels": True, "max_model_len": RECORDED_LENGTH}  # a run at the maximum length of the record
    chain_table = {**configs.PROVIDERS, "attention.prepare": configs.PROVIDERS["attention.prepare"][1:]}  # the block on its reference chain
    counted = {}  # per plan: the numbers of its claim, and those of the same plan without the row
    for layers, stack, tokens, code, port, shaped, loose, left in (
        (64, "native", 1, 194, 290, 419, 65, {}),
        (64, "native", 6, 146, 338, 467, 17, {}),
        (64, "fe-entries", 1, 49, 290, 338, 1, decode_left),
        (64, "fe-entries", 6, 49, 338, 386, 1, {"mlp.down": 64, "attention.core": 16, "projection.fp8.out": 16, "output_head": 1}),
        (2, "native", 1, 8, 12, 16, 4, {}),
        (2, "native", 6, 7, 13, 17, 3, {}),
        (2, "fe-entries", 1, 2, 12, 13, 1, {"projection.fp8.out": 2, "mlp.down": 2, "attention.core": 1, "output_head": 1}),
        (2, "fe-entries", 6, 2, 13, 14, 1, {"mlp.down": 2, "attention.core": 1, "projection.fp8.out": 1, "output_head": 1}),
    ):
        model = QwenDense() if layers == 64 else QwenDense.reduced(layers)
        planned = blocks.plan(model.provider_table, model.regimes, facts, 1, tokens, stack, model=model, options=options)
        claim = target.claim(planned, model, census.configuration_of(planned), installed=RECORDED)
        numbers, where = claim.numbers(), (layers, stack, tokens)
        assert (numbers["target_code"], numbers["target_port"], numbers["not_yet"], numbers["not_the_target"]) == (code, port, left, {}), where
        assert (claim.compared()[0], claim.compared()[2]) == (shaped, loose) and shaped + loose == code + port, where
        gone = "; not yet: " + ", ".join(f"{block} {count}" for block, count in left.items()) if left else ""  # no clause where no block is left
        if left:  # a stack that leaves blocks on fallback rows: the whole set, and its share on a second line
            assert claim.coverage()[1] == (
                f"kernel identity, as planned: {code + port} of {numbers['launches']} launches of a step are target kernels ({split.format(code, port)})"
                f"{gone}; 0 declared stand-ins"
            ), where
        else:  # a stack that launches the whole set: one line
            assert claim.coverage() == [
                f"kernel identity, as planned: {numbers['launches']} launches per step = {split.format(code, port)} + 0 declared stand-ins"
            ], where
        identity = claim.identities[claim.blocks.index("attention.prepare")]
        assert (identity.relation, identity.kernel, identity.kind, identity.launch) == ("target", "attention_prepare", "port", _prepare_shape(tokens)), where
        core = claim.identities[claim.blocks.index("attention.core")]
        standing = "target" if stack == "native" else "not yet (reference chain over the attention cache)"  # the library's row, or the chain
        assert core.text() == standing and (core.kernel, core.kind) == ("fi_decode_attention", "code"), where
        line = next(line for line in census.table(planned, claim).splitlines() if line.split()[0] == "attention.prepare")
        assert (
            line.split("cudnn.AttentionPrepareDecode")[1].split()
            == f"{PREPARE_FORM} attention_prepare 1 name and launch shape, as the entry states them target".replace(
                " 1 name", f" {16 if layers == 64 else 1} name"
            ).split()
        ), where
        # the row moves what a stack reaches by the launches of its entry and by nothing else: the same plan without the row has
        # them not yet, and reaches that many ports fewer
        chain = blocks.plan(chain_table, model.regimes, facts, 1, tokens, stack, model=model, options=options)
        before = target.claim(chain, model, census.configuration_of(chain), installed=RECORDED).numbers()
        prepared = claim.blocks.count("attention.prepare")  # the launches of the entry in a step: one per attention layer
        assert prepared == model.geometry.layer_kinds.count("full_attention") and before["not_yet"] == {**left, "attention.prepare": prepared}, where
        assert (before["target_code"], before["target_port"], before["not_the_target"]) == (code, port - prepared, {}), where
        counted[where] = (numbers, before)
    # 484 = the 468 of the stack without this row and the 16 launches of the entry; the same 16 on the stack of the entries alone
    published = [(64, stack, tokens) for tokens in (1, 6) for stack in ("native", "fe-entries")]
    now, then = ([sum(counted[where][side][kind] for kind in ("target_code", "target_port")) for where in published] for side in (0, 1))
    assert (now, then) == ([484, 339, 484, 387], [468, 323, 468, 371]) and {with_row - without for with_row, without in zip(now, then)} == {16}
    # the family README's record of the runs with this row is of the plans of its day: on `native` the 16 launches of the
    # attention core, which the library's row has brought since, are not in its counts, and on neither stack is the launch of
    # the head's quantise, which its entry's row has brought since
    said = " ".join((MODELS / "qwen_dense" / "README.md").read_text().split())
    core, quantise = 16, 1
    of_step, of_step_entries, of_verify, of_verify_entries = (
        f"{count - quantise - (core if where[1] == 'native' else 0)} of {counted[where][0]['launches']}" for count, where in zip(now, published)
    )
    for count in (
        f"{of_step} launches of a step target kernels on `native` ({split.format(194 - core, 289)}; {419 - core} of them in the recorded launch shape, 64 by membership)",
        f"{of_step_entries} on `fe-entries` ({split.format(49, 289)})",
        f"{of_verify} launches of the verify step target kernels on `native` ({split.format(146 - core, 337)}; {467 - core} of them in the launch shape the census records, 16",
        f"{of_verify_entries} on `fe-entries` ({split.format(49, 337)})",
        f"18 of 20 launches of a step target kernels on `native` ({split.format(7, 11)}) and 13 of 20 on `fe-entries` ({split.format(2, 11)})",
        f"of a verify step 18 of 20 ({split.format(6, 12)}) and 14 of 20 ({split.format(2, 12)})",
    ):
        assert count in said, count
    # a row that names one position row: the same entry in another form -- not the target's, by its form, and nothing recorded is held against it
    table = dict(configs.PROVIDERS)
    from dataclasses import replace

    one_row = replace(row, form=(("POSITIONS", "1d"), ("SECTION", None), *row.form[2:]))
    table["attention.prepare"] = [one_row, *table["attention.prepare"][1:]]
    model = QwenDense.reduced(2)
    planned = blocks.plan(table, model.regimes, facts, 1, 1, "fe-entries", model=model, options={"device": "cpu"})
    claim = target.claim(planned, model, installed=RECORDED)
    reason = f"the row's form is{census.form_text(one_row.form)}, the target's {PREPARE_FORM}"
    assert claim.identities[claim.blocks.index("attention.prepare")] == census.Identity(census.OTHER, "attention_prepare", "port", reason)
    assert (
        claim.numbers()["not_the_target"] == {"attention.prepare": 1}
        and claim.numbers()["target_port"] == 11
        and "not the target set: attention.prepare" in "; ".join(claim.reasons)
    )


def _shifted_tables(monkeypatch, which):
    """Every page table a state builds from now on with one of its tables a row late: row ``p`` of ``which`` (``"slots"`` or
    ``"positions"``) holds what belongs to the step that starts at ``p - 1`` (row 0: the last row's)."""
    from model_samples import state

    build = state.PageTable.build.__func__

    def late(cls, *arguments):
        table = build(cls, *arguments)
        for each in (table, *(() if table.single is None else (table.single,))):
            setattr(each, which, torch.roll(getattr(each, which), 1, 0))
        return table

    monkeypatch.setattr(state.PageTable, "build", classmethod(late))


def test_a_wrong_table_a_wrong_twin_and_a_wrong_accessor_each_fail_the_run(monkeypatch, capsys):
    """Control arms through the sample's ``main()`` on the stack ``fe-entries``, on the stand-in entry of the attention
    preparation: the clean run passes with the block judged in its band and its launch held; a slot table that is one
    row late, a position table that is one row late, a twin whose two rotary halves are exchanged and an accessor that
    exchanges two pages each end in ``FAIL``, by the line that sees them."""
    from model_samples import state
    from model_samples.providers import torch_ref

    arguments = ["--stack", "fe-entries", "--sequences", "1"]

    def run(*more, **stand_in):
        code, lines = _sample_run(monkeypatch, capsys, [*arguments, *more], **stand_in)
        block = next((line for line in lines if line.strip().startswith("attention.prepare ") and "calls" in line), "")
        return code, lines, block

    code, lines, block = run()
    assert code == 0 and lines[-1] == "[00] PASS" and block.endswith("[band: attention_prepare]  PASS") and _kernel_lines(lines)[0].endswith("    PASS"), lines[
        -6:
    ]
    assert "; 0 query and 0 key elements outside their band; gate rows equal; value rows equal; the rest of the pool untouched" in block
    stated = [line.strip() for line in lines if line.strip().startswith("attention.prepare: ")]
    assert any(
        line.startswith(
            f"attention.prepare: cudnn.AttentionPrepareDecode {PREPARE_FORM}, 1 launch per step (1 call x 1 per call); the entry states grid (1, 28, 1), "
        )
        for line in stated
    ), "\n".join(stated)
    for which in ("slots", "positions"):
        with monkeypatch.context() as patched:
            _shifted_tables(patched, which)
            code, lines, block = run()
        assert code == 1 and lines[-1] == "[00] FAIL" and block.endswith("[band: attention_prepare]  FAIL"), (which, block)
        outside = block.split("; ")[1]
        if which == "slots":  # the keys and the values stand one row away: the queries and the gates are what they must be
            assert (
                outside.startswith("0 query and ")
                and not outside.startswith("0 query and 0 key")
                and "value rows DIFFER; the rest of the pool WRITTEN" in block
            )
        else:  # rotated by the angle of another position, and written where they belong
            assert not outside.startswith("0 query") and " and 0 key" not in outside and "value rows equal; the rest of the pool untouched" in block, block
        assert "gate rows equal" in block and _kernel_lines(lines)[0].endswith("    PASS"), "the launch is the planned one: the kernel check cannot see a table"
    with monkeypatch.context() as patched:  # the twin's two rotary halves exchanged
        rows = torch_ref.prepare_rows

        def exchanged(x, weight, cos, sin, eps, order="3d"):
            out, band = rows(x, weight, cos, sin, eps, order)
            half = cos.shape[-1]
            return torch.cat((out[..., half : 2 * half], out[..., :half], out[..., 2 * half :]), dim=-1), band

        patched.setattr(torch_ref, "prepare_rows", exchanged)
        code, lines, block = run()
    assert code == 1 and lines[-1] == "[00] FAIL" and block.endswith("[band: attention_prepare]  FAIL"), block
    assert not block.split("; ")[1].startswith("0 query") and "gate rows equal; value rows equal; the rest of the pool untouched" in block, block
    # an accessor that exchanges two pages: the write of the twin, in a run whose state has two pages -- then the read of the chain.
    # 26 prompt tokens and 45 new ones: 70 steps, across position 64 (the logits band of the tiny geometry holds that long on these stand-ins)
    longer = ["--force-tokens", "45"]
    code, lines, block = run(*longer)
    assert code == 0 and lines[-1] == "[00] PASS" and block.endswith("[band: attention_prepare]  PASS"), "the clean run across position 64"
    assert block.split()[1:3] == ["70", "calls"], block
    with monkeypatch.context() as patched:
        write = state.PagedKvCache.write

        def other_page(self, position, k, v):
            page, row = divmod(position, self.page)
            return write(self, (page ^ 1 if self.pages > 1 else page) * self.page + row, k, v)

        patched.setattr(state.PagedKvCache, "write", other_page)
        code, lines, block = run(*longer)
    assert code == 1 and lines[-1] == "[00] FAIL" and block.endswith("[band: attention_prepare]  FAIL") and "the rest of the pool WRITTEN" in block, block
    with monkeypatch.context() as patched:
        read = state.PagedKvCache._read

        def exchanged_pages(self, half, count, dtype):  # every pair of pages of a sequence the other way round
            if self.pages < 2:
                return read(self, half, count, dtype)
            whole = read(self, half, -(-count // (2 * self.page)) * 2 * self.page, dtype)
            return whole.unflatten(2, (-1, 2, self.page)).flip(3).flatten(2, 4)[:, :, :count]

        patched.setattr(state.PagedKvCache, "_read", exchanged_pages)
        code, lines, block = run(*longer)
    tokens = next(line for line in lines if line.startswith("  tokens "))
    assert code == 1 and lines[-1] == "[00] FAIL" and tokens.endswith("FAIL") and block.endswith("[band: attention_prepare]  PASS"), (tokens, block)


def test_the_source_hash_covers_what_the_hashed_files_import():
    """The list of hashed files is closed under its rule, to the fixed point: every module under ``frost/``,
    ``_torch_stream.py`` and ``_cutlass_*`` that a file of the list imports at module level is in the list, and so is
    the ``__init__.py`` of every package such an import runs.  The fixed point is formed again here, from the files and
    with ``ast``: a module a listed file starts to import turns this red until it is hashed, so the hash of a run
    covers the modules its kernels are built from.  The two modules that ``frost/init_order.py`` imports are among
    them, once each."""
    import ast

    import cudnn

    from model_samples.providers import fe

    package = Path(cudnn.__file__).parent

    def module_file(dotted):  # cudnn.x.y -> x/y.py or x/y/__init__.py of the package, or None
        parts = dotted.split(".")
        if parts[0] != "cudnn":
            return None
        inner = "/".join(parts[1:])
        return next((name for name in (f"{inner}.py" if inner else "__init__.py", f"{inner}/__init__.py") if (package / name).is_file()), None)

    def run_at_import(body):  # every statement an import of the module runs: not the body of a function
        for node in body:
            yield node
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for field in ("body", "orelse", "finalbody", "handlers"):
                    yield from run_at_import([inner for inner in getattr(node, field, []) if isinstance(inner, ast.AST)])

    def imported(name):
        """The files of the package the module ``name`` imports at module level, the ``__init__.py`` of every package on the way included."""
        found, inside = set(), ["cudnn", *name.split("/")[:-1]]

        def add(dotted):
            parts = dotted.split(".")
            found.update(filter(None, (module_file(".".join(parts[:depth])) for depth in range(2, len(parts) + 1))))

        for node in run_at_import(ast.parse((package / name).read_text()).body):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                base = inside[: len(inside) - (node.level - 1)] if node.level else []
                dotted = ".".join([*base, *(node.module.split(".") if node.module else [])])
                add(dotted)
                for alias in node.names:  # ``from package import module``
                    add(f"{dotted}.{alias.name}")
        return found

    def under(name):
        return name.startswith("frost/") or name == "_torch_stream.py" or name.startswith("_cutlass")

    def fixed_point(listed):
        closed, new = set(listed), list(listed)
        while new:
            new = sorted({name for each in new for name in imported(each) if under(name)} - closed)
            closed.update(new)
        return closed

    listed = fe.SOURCE_FILES
    assert len(listed) == len(set(listed)) == 45 and all((package / name).is_file() for name in listed), "every file once, and a file of the package"
    assert fixed_point(listed) == set(listed), f"imported by a hashed file and not hashed: {sorted(fixed_point(listed) - set(listed))}"
    # the rule finds a module that is left out, at the first level and behind it, and a package's own file
    for name in ("frost/capture.py", "_torch_stream.py", "frost/tile_dsl/swizzle.py", "frost/buffers.py", "frost/__init__.py", "_cutlass_helpers/__init__.py"):
        assert name in fixed_point([each for each in listed if each != name]), f"{name}: a listed file imports it"
    assert not under("__init__.py") and "_device.py" not in listed, "outside the rule: the package's entry point, and a module imported inside a function"
    source = (package / "frost/init_order.py").read_text()
    assert re.search(r"^from cudnn\._torch_stream import ", source, re.M) and re.search(r"^from \. import capture\b", source, re.M)
    at = listed.index("frost/init_order.py")
    assert listed[at + 1 : at + 3] == ("frost/capture.py", "_torch_stream.py"), "what a call of an entry runs in front of its launch, with what that imports"


# ---------------------------------------------------------------------------------------------------------------------------
# The twin of a library attention kernel that documents no numerics contract (``torch_ref.AttentionBound``): the reference
# chain itself, which also keeps the operand terms of a bound.  No row of a table names it, so nothing here runs a session:
# the twin is called as the tie-out calls it, and its terms are held to the definition written out here.
# ---------------------------------------------------------------------------------------------------------------------------
#: query heads, key / value heads and head width of the published model's attention layers
ATTENTION_WIDTHS = (24, 4, 256)


def _attention_weights(query, keys, scale=None):
    """The softmax weights of one row's attended rows, written out from the definition, head by head: per query head the
    weights BEFORE they are normalised (``exp`` of every scaled score less the largest) and their sum, in fp64.  ``query``
    ``[heads, D]``, ``keys`` ``[kv heads, attended rows, D]``; a query head reads the key / value head of its group; the
    scores are ``(q . k_j) * D**-0.5`` unless ``scale`` says otherwise."""
    heads, width = query.shape
    group = heads // keys.shape[0]
    scale = width**-0.5 if scale is None else scale
    weights = []
    for head in range(heads):
        scores = (keys[head // group].double() @ query[head].double()) * scale
        raw = torch.exp(scores - scores.max())
        weights.append((raw, raw.sum()))
    return weights


def _attention_terms(query, keys, values, scale=None):
    """``(chain, spread)`` of one row, ``[heads * D]`` each in fp64, from the definition: per query head and element the sum
    of the attended values, and of their magnitudes, under the softmax weights of :func:`_attention_weights`."""
    group = query.shape[0] // keys.shape[0]
    chain, spread = [], []
    for head, (raw, total) in enumerate(_attention_weights(query, keys, scale)):
        held = values[head // group].double()
        chain.append((raw / total) @ held)
        spread.append((raw / total) @ held.abs())
    return torch.stack(chain).flatten(), torch.stack(spread).flatten()


def test_the_twin_of_a_library_attention_kernel_is_the_reference_chain_and_keeps_the_terms_of_its_bound():
    """The twin of an attention kernel that documents no numerics contract is the reference chain itself: its bytes are the
    chain's on a plain and on a paged cache, at one token per sequence and at several, in both precisions a run can ask of
    its reference.  And it keeps the two operand terms of the band -- the chain before its rounding to bf16 and the
    weighted magnitude of the attended values -- always in fp64, equal to the terms written out from the definition row by
    row and head by head: token ``t`` of a sequence attends to the positions up to its own, never to a later token's or to a
    row behind the step.  It reads the cache through its accessors with the host integer it is handed: a state whose
    tables are shifted -- the positions and the slots, the page table and the lengths -- leaves its bytes and its terms
    as they were, and it writes nothing.  The terms it keeps are those of its LAST call: one twin serves every call here."""
    from model_samples.providers import torch_ref
    from model_samples.qwen_dense import QwenDense
    from model_samples.state import DecodeState, KvCache, PagedKvCache

    model = QwenDense.reduced(2)
    g = model.geometry
    heads, kv_heads, width = g.attn_heads, g.attn_kv_heads, g.attn_head_dim
    assert (heads, kv_heads, width) == ATTENTION_WIDTHS
    layer = g.layer_kinds.index("full_attention")
    fresh = torch_ref.AttentionBound(None, None, None, {})
    assert fresh.entry == "torch_ref.attention with the operand terms of its bound" and fresh.bound is None and fresh.per_row == (0, 1)
    assert fresh.state_args == () and fresh.cache_page is None, "it writes nothing and reaches the cache through its accessors, in either layout"

    def bits(tensor):
        return tensor.view(torch.int16)

    twins = {precision: torch_ref.AttentionBound(None, None, None, {"reference_precision": precision}) for precision in ("fp64", "ieee-fp32")}
    generator = torch.Generator().manual_seed(11)
    capacity, moved = 130, {False: 0, True: 0}  # elements that are another byte in fp32 than in fp64: at one token per sequence, at several
    for sequences, tokens, position in ((1, 1, 0), (1, 1, 36), (3, 1, 64), (2, 3, 62), (1, 6, 0), (2, 2, 128)):
        rows = sequences * tokens
        keys = torch.randn(sequences, kv_heads, capacity, width, generator=generator).bfloat16()
        values = torch.randn(sequences, kv_heads, capacity, width, generator=generator).bfloat16()
        query = torch.randn(rows, heads, width, generator=generator).bfloat16()
        # the terms from the definition: row (sequence, token) attends to the positions 0 .. position + token of its sequence
        defined = [
            _attention_terms(query[row], keys[row // tokens, :, : position + row % tokens + 1], values[row // tokens, :, : position + row % tokens + 1])
            for row in range(rows)
        ]
        chain, spread = (torch.stack([terms[part] for terms in defined]) for part in (0, 1))
        kept = {}
        for layout, page in (("plain", None), ("paged", 64)):
            state = DecodeState(g.layer_kinds, sequences, capacity, model.state_shapes(), "cpu", tokens, page=page, max_model_len=capacity if page else None)
            cache = state.layers[layer]
            assert isinstance(cache, PagedKvCache if page else KvCache) and cache.tokens == tokens
            for at in range(capacity):  # every position holds a drawn row: also the later tokens' of the step and the rows behind it
                cache.write(at, keys[:, :, at], values[:, :, at])
            before = [tensor.clone() for tensor in cache.tensors()]
            for precision, acc in (("fp64", torch.float64), ("ieee-fp32", torch.float32)):
                twin = twins[precision]
                out = twin(query, cache, position)
                where = f"{sequences} x {tokens} rows at position {position}, a {layout} cache, {precision}"
                want = torch_ref.attention_tokens(query, cache, position, tokens, acc) if tokens > 1 else torch_ref.attention(query, cache, position, acc)
                assert (
                    out.dtype == torch.bfloat16 and tuple(out.shape) == (rows, heads * width) and torch.equal(bits(out), bits(want))
                ), f"{where}: the chain's bytes"
                got_chain, got_spread = twin.bound
                assert got_chain.dtype == got_spread.dtype == torch.float64 and got_chain.shape == got_spread.shape == out.shape, f"{where}: both terms in fp64"
                # held to the definition far below any rounding of the run: a term in fp32 is 1e-7 of its magnitude off, a rounded chain 1e-3
                assert bool(((got_chain - chain).abs() <= 1e-12 * spread).all()) and bool(((got_spread - spread).abs() <= 1e-12 * spread).all()), where
                if position + tokens > 1:  # a row that attends to two rows or more: the spread is of magnitudes, never the chain's own
                    assert bool((got_spread >= got_chain.abs() * (1 - 1e-12)).all()) and bool((got_spread > got_chain.abs() * 1.5).any()), where
                else:  # one attended row: its weight is 1
                    assert torch.equal(got_spread, got_chain.abs()) and torch.equal(
                        got_chain, values[:, :, 0].repeat_interleave(heads // kv_heads, 1).flatten(1).double()
                    )
                kept[layout, precision] = (out, got_chain, got_spread)
                assert all(torch.equal(bits(a), bits(b)) for a, b in zip(cache.tensors(), before)), f"{where}: the twin writes nothing"
            for term in (1, 2):  # the terms do not follow the run's precision: bit for bit the same in both
                assert torch.equal(kept[layout, "fp64"][term], kept[layout, "ieee-fp32"][term]), (sequences, tokens, position, layout)
            assert torch.equal(bits(kept[layout, "fp64"][0]), bits(kept[layout, "fp64"][1].bfloat16())), "in fp64 the bytes are the kept chain, rounded once"
            moved[tokens > 1] += int((bits(kept[layout, "fp64"][0]) != bits(kept[layout, "ieee-fp32"][0])).sum())
            if page:  # a state whose tables are shifted, its page table and its lengths among them: the twin never reads them
                state.table.positions += 1
                state.table.slots += 1
                state.table.blocks += 1  # every entry names the next page
                state.table.lengths -= 1  # every sequence one position short
                shifted = torch_ref.AttentionBound(None, None, None, {})
                again = shifted(query, cache, position)
                assert torch.equal(bits(again), bits(kept[layout, "fp64"][0])) and all(
                    torch.equal(a, b) for a, b in zip(shifted.bound, kept[layout, "fp64"][1:])
                )
        for part in (0, 1, 2):  # either layout hands the chain the same rows
            assert torch.equal(kept["plain", "fp64"][part].double(), kept["paged", "fp64"][part].double()), (sequences, tokens, position, part)
    assert moved[False] > 0 and moved[True] > 0, "the bytes DO follow the run's precision, at one token and at several: some element is another byte in fp32"


# ---------------------------------------------------------------------------------------------------------------------------
# The band of such a kernel (``attention_bf16_weights``), declared before any row names it: the three constants and the
# band's text, the judge on the model's own cases and on each defect alone, and the control arms.  The judge is called as
# the tie-out calls it, on built operands (whose figures follow from plain arithmetic) and on drawn ones.
# ---------------------------------------------------------------------------------------------------------------------------
#: the largest relative error of one rounding to nearest at eight significant bits: what the band's model allows a weight and the divisor
ROUNDING = 2.0**-8 / (1 + 2.0**-8)


def _drawn_attention(pages, pool_seed, rows, query_seed, query_scale=1.0):
    """Drawn operands of an attention core at the published widths, bf16: ``(queries [rows, 24, 256], keys, values)`` -- a
    pool of ``pages`` pages of 64 rows drawn as ONE tensor ``[pages, 4, 64, 512]`` (the keys in the first half of the last
    axis, the values in the second) and read page by page as one run of rows, ``[4, pages * 64, 256]`` each; the queries
    times ``query_scale``.  Each from a generator of its own seed, on the CPU."""
    heads, kv_heads, width = ATTENTION_WIDTHS
    pool = torch.randn(pages, kv_heads, 64, 2 * width, generator=torch.Generator("cpu").manual_seed(pool_seed)).bfloat16()
    queries = (torch.randn(rows, heads, width, generator=torch.Generator("cpu").manual_seed(query_seed)) * query_scale).bfloat16()
    run = pool.permute(1, 0, 2, 3).reshape(kv_heads, pages * 64, 2 * width)
    return queries, run[..., :width].contiguous(), run[..., width:].contiguous()


def _kernel_like_attention(query, keys, values, scale=None, divisor=False):
    """One row of a kernel-like chain of the band's model, ``[heads * D]`` bf16: every softmax weight carries one rounding to
    bf16 and the output one.  ``divisor=False``: the NORMALISED weights are each rounded and the divisor is exact;
    ``divisor=True``: the weights are rounded before they are normalised and their sum is rounded apart from them."""
    group = query.shape[0] // keys.shape[0]
    rows = []
    for head, (raw, total) in enumerate(_attention_weights(query, keys, scale)):
        held = values[head // group].double()
        if divisor:
            rows.append((raw.bfloat16().double() @ held) / total.bfloat16().double())
        else:
            rows.append((raw / total).bfloat16().double() @ held)
    return torch.stack(rows).flatten().bfloat16()


def _judged_attention(out, twin, ref=None, state=(), ref_state=()):
    """One call judged in the band ``attention_bf16_weights`` as the tie-out judges it: the output ``out`` against the twin
    ``twin`` (anything that keeps ``bound``), ``ref`` the reference chain's bytes (the output's own where none is given)."""
    from model_samples import tie_out

    stats = tie_out.BlockStats("attention_bf16_weights")
    tie_out.BlockTap._judge(stats, (out,), (out if ref is None else ref,), list(state), list(ref_state), 0, twin)
    return stats


def _built_attention(first_key, first_value, second_value):
    """BUILT operands of one row that attends to two rows, at two query heads on one key / value head of width 256, and
    their twin after its call: ``(twin, the chain's bytes)``.  The query is 4 in its first element and zero elsewhere, the
    first key ``first_key`` there and zero elsewhere, the second key zero: the scaled scores are ``first_key / 4`` and 0.
    Every element of the first value row is ``first_value``, of the second ``second_value``: every element of the output is
    the same judged element."""
    from model_samples.providers import torch_ref
    from model_samples.state import KvCache

    query, keys, values = torch.zeros(1, 2, 256), torch.zeros(1, 1, 2, 256), torch.zeros(1, 1, 2, 256)
    query[0, :, 0], keys[0, 0, 0, 0], values[0, 0, 0], values[0, 0, 1] = 4.0, first_key, first_value, second_value
    twin = torch_ref.AttentionBound(None, None, None, {})
    return twin, twin(query.bfloat16(), KvCache(keys.bfloat16(), values.bfloat16()), 1)


def _model_extreme(share):
    """The output of the band's model at its extreme, ``share`` of the largest rounding error it allows, on the built operands
    of equal scores and the values +1 and -1 (chain 0, spread 1): the weight of +1 rounded up, the weight of -1 rounded
    down, the divisor rounded down; then the output's own rounding.  ``[1, 512]`` bf16."""
    error = share * ROUNDING
    return torch.full((1, 512), (0.5 * (1 + error) * 1.0 + 0.5 * (1 - error) * -1.0) / (1 - error), dtype=torch.float64).bfloat16()


def test_the_band_of_a_library_attention_kernel_is_the_bound_of_a_stated_model_and_its_judge_holds_it():
    """The band ``attention_bf16_weights``: three constants and a text that a run prints, declared before any row names them,
    and a docstring that states the model the band is the bound of -- and does not say that the band cannot be left.  The
    judge, on the model's own cases, INSIDE: the chain itself; on drawn operands a chain whose normalised weights are each
    rounded to bf16 with the divisor exact, and one whose weights are rounded before they are normalised with the
    divisor rounded apart from them; on built operands the model's extreme at 0.9 of its largest rounding error (0.898 of
    the bound) and at all of it (the bound itself).  OUTSIDE, each alone and by its own count: the extreme at 1.5 of that
    error; the bf16 value next above the bound of a chain of zero, which says that the spacing is the chain's and not the
    output's; one element one spacing beyond its bound among elements of larger chains, which says that the spacing is
    not the tensor's largest value's; a NaN; an infinite value; a byte of the cache the call changed.  A twin that keeps
    no terms is refused, other outputs than the one tensor are not compared, and the same output is NOT within one bf16
    unit of the chain: the precedent band could not hold it."""
    import math
    from types import SimpleNamespace

    from model_samples import tie_out
    from model_samples.providers import torch_ref
    from model_samples.state import KvCache

    assert (tie_out.ATTENTION_BAND_COEFFICIENT, tie_out.ATTENTION_WEIGHT_TERM, tie_out.ATTENTION_SPACINGS) == (1.0, 2.0**-8, 1.0), "as declared"
    assert tie_out.BANDS["attention_bf16_weights"] == (
        "against the reference chain before its rounding to bf16 (chain = sum p v over the attended rows under their softmax weights p, spread = sum p |v|, "
        "both in fp64): every element finite and within 1 x [2**-8 x (spread + |chain|) + 1 spacing of bf16 values at |chain|] -- the bound of a stated model "
        "of the kernel's arithmetic (every weight one rounding at eight significant bits, the divisor one of its own, the output one), declared before any "
        "run and never taken from a measured distance; the cache bitwise"
    ), "the text a run prints for the band"
    stated = " ".join((tie_out.__doc__ or "").split())
    for sentence in (
        "it is the bound of a stated model of the kernel's arithmetic, declared here before any run of a row that names it.",
        "Which values such a kernel rounds is not established, so no twin can be written at its rounding points.",
        "The model: every softmax weight carries one rounding to nearest at eight significant bits, ``p_j (1 + d_j)`` with ``|d_j| <= u`` and "
        "``u = 2**-8 / (1 + 2**-8)``; the number the weighted sum is divided by carries one of its own, ``1 + e`` with ``|e| <= u``; and the output is "
        "rounded to bf16 once.",
        "-- and ``u / (1 - u)`` is exactly ``2**-8``.",
        "out is finite and |out - chain| <= 1 x [ 2**-8 x (spread + |chain|) + 1 x spacing ]",
        "the chain's alone, neither the output's nor the tensor's largest value's.",
        "It is derived from a model and from no measured distance, and it claims nothing about what the kernel rounds.",
        "there an output of the model itself can use up to ``1 + 2**-8`` of its bound.",
        "So an element outside is a finding, read with its share, its chain and its spread; it is never a reason for another coefficient, a further "
        "term or a shape left out.",
        "A block in this band is not within one bf16 unit of the chain",
        "one wrong, missing or extra attended row whose softmax weight is below the band's term.",
    ):
        assert sentence in stated, sentence
    for never in ("cannot leave", "cannot be left", "never leaves its band", "rounds its weights to bf16", "derived from its arithmetic"):
        assert never not in stated, f"the docstring claims more than the model gives: {never!r}"
    assert ROUNDING / (1 - ROUNDING) == 2.0**-8, "the weight term is the model's own figure: u / (1 - u)"

    # ---- BUILT operands, the model's extreme: equal scores, the values +1 and -1: chain 0, spread 1 in every element
    twin, ref = _built_attention(0.0, 1.0, -1.0)
    chain, spread = twin.bound
    assert not chain.any() and bool((spread == 1.0).all()) and not ref.any() and tuple(chain.shape) == (1, 512)
    inside = _judged_attention(_model_extreme(0.9), twin, ref)
    assert inside.ok and inside.finite and (inside.elements, inside.outside, inside.other_value) == (512, 0, 512)
    assert inside.worst_share == 0.8984375 and inside.worst_at == (0.0, 1.0), "0.9 of the model's largest rounding error: 0.898 of the bound"
    assert inside.line() == (
        "512 elements, 512 of them not the reference chain's byte; 0 outside their bound; the largest distance 0.898 of its bound, at a chain of 0 with a "
        "spread of 1 (printed, not gated); the cache untouched"
    )
    whole = _judged_attention(_model_extreme(1.0), twin, ref)
    assert whole.ok and whole.worst_share == 1.0 and float(_model_extreme(1.0)[0, 0]) == 2.0**-8, "all of it: the bound itself, and inside"
    beyond = _judged_attention(_model_extreme(1.5), twin, ref)
    assert not beyond.ok and beyond.finite and beyond.outside == 512 and beyond.worst_share == 1.5, "1.5 of it: outside"
    assert beyond.line().split("; ")[1:3] == [
        "512 outside their bound",
        "the largest distance 1.500 of its bound, at a chain of 0 with a spread of 1 (printed, not gated)",
    ]
    # ... the spacing is the CHAIN's: at a chain of zero it is that of the smallest normal fp32 value, far below fp64's own
    # spacing at the bound.  The bf16 value next above the bound is outside; with one spacing at the OUTPUT it would be the bound
    above = torch.full((1, 512), 2.0**-8 * (1 + 2.0**-7), dtype=torch.float64).bfloat16()
    assert float(above[0, 0]) == 2.0**-8 + 2.0**-15, "the next bf16 value above 2**-8"
    next_value = _judged_attention(above, twin, ref)
    assert (
        not next_value.ok and next_value.outside == 512 and next_value.worst_share == 1.0078125
    ), "one spacing of the output beyond a bound whose chain is zero"

    # ---- BUILT operands, the chain's own term: the values +1 and +1 (chain 1, spread 1): the bound is 2**-8 x (1 + 1) + one
    # spacing at 1 = 2**-6, on either side of the chain -- above 1 two bf16 values, below it four
    twin_one, ref_one = _built_attention(0.0, 1.0, 1.0)
    assert bool((twin_one.bound[0] == 1.0).all()) and bool((twin_one.bound[1] == 1.0).all()) and bool((ref_one == 1.0).all())
    for value, share in ((1 + 2.0**-6, 1.0), (1 + 2.0**-6 + 2.0**-7, 1.5), (1 - 2.0**-6, 1.0), (1 - 2.0**-6 - 2.0**-8, 1.25)):
        stats = _judged_attention(torch.full((1, 512), value, dtype=torch.float64).bfloat16(), twin_one, ref_one)
        assert (stats.ok, stats.outside, stats.worst_share, stats.worst_at) == (share == 1.0, 0 if share == 1.0 else 512, share, (1.0, 1.0)), value
    # ... and an element whose chain is not finite holds nothing: it is outside, whatever the output is
    twin_inf, ref_inf = _built_attention(0.0, float("inf"), 1.0)
    assert bool(torch.isinf(twin_inf.bound[0]).all()) and bool(torch.isinf(ref_inf).all())
    stats = _judged_attention(torch.ones(1, 512).bfloat16(), twin_inf, ref_inf)
    assert not stats.ok and stats.finite and stats.outside == 512, "a finite output against a chain that is not finite"

    # ---- BUILT operands, weights rounded to bf16: the scaled scores 1 and 0, the values 0.375 and -1
    twin_w, ref_w = _built_attention(4.0, 0.375, -1.0)
    weight = 1.0 / (1.0 + math.exp(-1.0))  # of the first row; the second has the rest
    chain_w, spread_w = twin_w.bound
    assert abs(float(chain_w[0, 0]) - (weight * 0.375 - (1 - weight))) < 1e-15 and abs(float(spread_w[0, 0]) - (weight * 0.375 + (1 - weight))) < 1e-15
    rounded = [float(torch.tensor(value, dtype=torch.float64).bfloat16()) for value in (weight, 1 - weight)]
    assert rounded == [0.73046875, 0.26953125], "the two weights, each rounded to bf16"
    out_w = torch.full((1, 512), rounded[0] * 0.375 + rounded[1] * -1.0, dtype=torch.float64).bfloat16()
    held = _judged_attention(out_w, twin_w, ref_w)
    assert held.ok and held.outside == 0 and abs(held.worst_share - 0.3733452) < 1e-6, "inside: 0.373 of the bound"

    # ---- DRAWN operands at the published widths: a pool of three pages, one sequence, one row per step
    queries, keys, values = _drawn_attention(3, 7001, 1, 9001)
    cache = KvCache(keys[None], values[None])
    for live in (2, 37, 100):
        twin = torch_ref.AttentionBound(None, None, None, {})
        ref = twin(queries, cache, live - 1)
        chain, spread = twin.bound
        itself = _judged_attention(ref, twin, ref)
        assert (
            itself.ok and (itself.elements, itself.outside, itself.other_value) == (6144, 0, 0) and 0.0 < itself.worst_share < 0.5
        ), "the chain itself: its one rounding"
        for divisor in (False, True):
            out = _kernel_like_attention(queries[0], keys[:, :live], values[:, :live], divisor=divisor)[None]
            stats = _judged_attention(out, twin, ref)
            assert stats.ok and stats.finite and stats.outside == 0 and 0.25 < stats.worst_share < 0.75, (live, divisor, stats.line())
            assert 1000 < stats.other_value < 6144, "a kernel-like chain: a share of its elements is another byte than the chain's, every one inside"
            units = tie_out.BlockStats("bf16_unit")
            tie_out.BlockTap._judge(units, (out,), (ref,), [], [], 0, twin)
            assert (
                not units.ok and units.worst_units > 4.0
            ), "and it is NOT within one bf16 unit of the chain: the band's term is the attended values', not the output's"
    # (the twin of the last pass, at 100 live rows, and its chain's bytes)
    spacing = torch_ref.bf16_unit(chain.abs())
    bound = 2.0**-8 * (spread + chain.abs()) + spacing
    assert float(chain.abs().amin()) > float(torch.finfo(torch.float32).tiny), "no chain of zero among the drawn elements"
    # one element one spacing beyond its bound -- an element whose chain is two binades or more below the tensor's largest, and
    # whose bound ends inside its own binade, so that one spacing is one bf16 step there; one spacing INSIDE the bound first
    eligible = (4 * spacing <= torch_ref.bf16_unit(chain.abs().amax())) & (torch_ref.bf16_unit(chain.abs() + bound + 2 * spacing) == spacing)
    assert int(eligible.sum()) > 100, "the drawn operands have such elements"
    at = tuple(int(i) for i in eligible.nonzero()[0])
    away = 1.0 if float(chain[at]) > 0 else -1.0
    for step, outside in ((-1.0, 0), (1.0, 1)):
        moved = ref.clone()
        moved[at] = float(chain[at] + away * (bound[at] + step * spacing[at]))
        stats = _judged_attention(moved, twin, ref)
        assert (stats.ok, stats.outside, stats.other_value) == (
            not outside,
            outside,
            1,
        ) and stats.finite, f"one spacing {'beyond' if outside else 'inside'} the bound"
        assert (stats.worst_share > 1.0) == bool(outside) and abs(stats.worst_share - 1.0) < 2 * float(spacing[at] / bound[at]), stats.line()
        assert stats.worst_at == (float(chain[at]), float(spread[at])), "the line names the chain and the spread of the element with the largest share"
    for value, name in ((float("nan"), "a NaN"), (float("inf"), "an infinite value"), (float("-inf"), "an infinite value")):
        not_finite = ref.clone()
        not_finite[0, 4321] = value
        stats = _judged_attention(not_finite, twin, ref)
        assert not stats.ok and not stats.finite and stats.outside == 1 and stats.line().endswith("the cache untouched; NON-FINITE VALUES"), name
        assert math.isnan(stats.worst_share) if name == "a NaN" else stats.worst_share == float("inf"), name
        stats_after = tie_out.BlockStats("attention_bf16_weights")  # ... and a later clean call does not take it back
        for out in (not_finite, ref):
            tie_out.BlockTap._judge(stats_after, (out,), (ref,), [], [], 0, twin)
        assert not stats_after.ok and not stats_after.finite and (stats_after.elements, stats_after.outside) == (12288, 1), name
        assert math.isnan(stats_after.worst_share) if name == "a NaN" else stats_after.worst_share == float("inf"), name
    # the call writes nothing: the cache it read is held byte for byte
    pool = torch.cat((keys, values), dim=-1)
    written = pool.clone()
    written.view(torch.int16)[3, 150, 300] ^= 1  # one bit of a value row behind the live rows
    untouched = _judged_attention(ref, twin, ref, state=[(pool,)], ref_state=[(pool.clone(),)])
    assert untouched.ok and untouched.state_equal
    stats = _judged_attention(ref, twin, ref, state=[(written,)], ref_state=[(pool,)])
    assert (
        not stats.ok and not stats.state_equal and stats.finite and stats.outside == 0 and stats.line().endswith("the cache WRITTEN")
    ), "a byte of the cache the call changed"
    # over several calls the counts add up and the largest share stays, with the element it is of
    both = tie_out.BlockStats("attention_bf16_weights")
    worse = _kernel_like_attention(queries[0], keys[:, :100], values[:, :100])[None]
    for out in (worse, ref):
        tie_out.BlockTap._judge(both, (out,), (ref,), [], [], 0, twin)
    alone = _judged_attention(worse, twin, ref)
    assert both.ok and (both.elements, both.other_value, both.worst_share, both.worst_at) == (12288, alone.other_value, alone.worst_share, alone.worst_at)
    # what cannot be compared, and a twin that keeps no terms
    for name, out in (("two outputs", (ref, ref)), ("another shape", (ref[:, :100],)), ("another dtype", (ref.float(),))):
        stats = tie_out.BlockStats("attention_bf16_weights")
        tie_out.BlockTap._judge(stats, out, (ref,), [], [], 0, twin)
        assert not stats.ok and (stats.outside, stats.elements) == (1, 0), f"{name}: nothing is compared"
    chain_row = SimpleNamespace(key="torch_ref.attention")  # the reference chain's own block: it keeps nothing
    for other in (SimpleNamespace(), torch_ref.Attention(chain_row, None, None, {}), torch_ref.AttentionBound(None, None, None, {})):
        with pytest.raises(ValueError, match="the band attention_bf16_weights is judged against the reference chain with the operand terms of its bound"):
            _judged_attention(ref, other, ref)


def test_a_cache_a_library_attention_kernel_wrote_in_one_call_is_not_taken_back_by_a_later_clean_call():
    """The band ``attention_bf16_weights`` holds the cache a call reads byte for byte in EVERY call of a run: a block whose
    cache was written in one call stays failed, with ``the cache WRITTEN`` in its line, whatever a later clean call finds --
    as a NaN and an element outside stay.  The output is the chain's own bytes in both calls: the cache alone fails it."""
    from model_samples import tie_out

    twin, ref = _built_attention(4.0, 0.375, -1.0)
    pool = torch.zeros(2, 4).bfloat16()
    written = pool.clone()
    written.view(torch.int16)[1, 2] ^= 1  # one bit of the cache the first call read
    clean = _judged_attention(ref, twin, ref, state=[(pool,)], ref_state=[(pool.clone(),)])
    assert clean.ok and clean.state_equal and clean.line().endswith("the cache untouched"), "the clean call alone passes"
    stats = tie_out.BlockStats("attention_bf16_weights")
    for state, ref_state in (([(written,)], [(pool,)]), ([(pool,)], [(pool.clone(),)])):
        tie_out.BlockTap._judge(stats, (ref,), (ref,), state, ref_state, 0, twin)
    assert (
        not stats.ok and not stats.state_equal and stats.finite and (stats.elements, stats.outside) == (1024, 0) and stats.line().endswith("the cache WRITTEN")
    ), "a later clean call does not take it back"


def test_the_line_of_the_library_attention_band_is_pinned_whole_on_figures_that_are_not_round():
    """The line a block in the band ``attention_bf16_weights`` prints, whole, on the built operands of weights rounded to
    bf16 (scaled scores 1 and 0, the values 0.375 and -1): the share at three decimals, the chain and the spread of that
    element at four significant digits each -- figures that are not 0 and 1, so that a digit less in either shows."""
    twin, ref = _built_attention(4.0, 0.375, -1.0)
    out = torch.full((1, 512), 0.73046875 * 0.375 + 0.26953125 * -1.0, dtype=torch.float64).bfloat16()
    assert _judged_attention(out, twin, ref).line() == (
        "512 elements, 512 of them not the reference chain's byte; 0 outside their bound; the largest distance 0.373 of its bound, at a chain of 0.005206 with "
        "a spread of 0.5431 (printed, not gated); the cache untouched"
    )


def test_the_line_of_the_library_attention_band_prints_its_counts_with_the_thousands_separator():
    """The three counts of the band's line above 999, each with its separator: twelve rows of the built operands and an
    output of ones, every element of which is another byte than the chain's and outside its bound."""
    from types import SimpleNamespace

    twin, ref = _built_attention(4.0, 0.375, -1.0)
    twelve = SimpleNamespace(bound=tuple(term.repeat(12, 1) for term in twin.bound))
    line = _judged_attention(torch.ones(12, 512).bfloat16(), twelve, ref.repeat(12, 1)).line()
    assert line.startswith("6,144 elements, 6,144 of them not the reference chain's byte; 6,144 outside their bound; "), line


def test_terms_of_another_shape_than_the_output_are_not_compared_in_the_library_attention_band():
    """Terms of another shape than the output are not compared in the band ``attention_bf16_weights``, also where they would
    broadcast against it: one column of the terms against the 512 elements of the chain's own bytes is one thing outside
    and no element judged, never 512 elements inside.  The same output against its own terms is compared, and inside."""
    from types import SimpleNamespace

    twin, ref = _built_attention(4.0, 0.375, -1.0)
    other = SimpleNamespace(bound=tuple(term[:, :1] for term in twin.bound))  # terms that would broadcast against the output
    stats = _judged_attention(ref, other, ref)
    assert not stats.ok and (stats.outside, stats.elements) == (1, 0), "terms of another shape: nothing is compared"
    whole = _judged_attention(ref, twin, ref)
    assert whole.ok and (whole.outside, whole.elements) == (0, 512), "its own terms: every element compared"


def test_the_line_of_the_library_attention_band_names_an_element_where_every_element_is_on_its_chain():
    """Where every element of a step is exactly on its chain, no element has used any of its bound: a step with ONE
    attended row, whose chain is that row's value and whose output is the chain's bytes.  The line of the band
    ``attention_bf16_weights`` then names the chain and the spread of an element of the first judged call -- its first --
    and never figures that belong to no element.  A later call that is on its chain too leaves them; the first call that
    is off its chain takes their place, and a later call on its chain does not take that back.  A call that cannot be
    compared is no judged call: it keeps nothing, and the call behind it is the first.  An element whose chain and
    spread are both zero is an element like any other: kept as the first judged call's, and kept with the largest share."""
    from model_samples import tie_out
    from model_samples.providers import torch_ref
    from model_samples.state import KvCache

    def one_row(first):
        """One row that attends to ONE row, at two query heads on one key / value head of width 256, and its twin after
        its call: ``(twin, the chain's bytes)``.  The attended row's value is ``first`` in its first element and 0.5 in
        every other: its weight is 1, so every element of the chain is that value, and of the spread its magnitude."""
        values = torch.full((1, 1, 1, 256), 0.5)
        values[0, 0, 0, 0] = first
        twin = torch_ref.AttentionBound(None, None, None, {})
        return twin, twin(torch.ones(1, 2, 256).bfloat16(), KvCache(torch.zeros(1, 1, 1, 256).bfloat16(), values.bfloat16()), 0)

    twin, ref = one_row(-0.75)
    chain, spread = twin.bound
    assert torch.equal(ref.double(), chain) and torch.equal(spread, chain.abs()), "one attended row: the output is exactly its chain"
    assert (float(chain[0, 0]), float(spread[0, 0]), float(chain[0, 1]), float(chain[0, 256])) == (-0.75, 0.75, 0.5, -0.75)
    stats = _judged_attention(ref, twin, ref)
    assert stats.ok and (stats.elements, stats.outside, stats.other_value, stats.worst_share) == (512, 0, 0, 0.0)
    assert stats.worst_at == (-0.75, 0.75), "no element is off its chain: the first element of the first judged call, never the figures of no element"
    assert stats.line() == (
        "512 elements, 0 of them not the reference chain's byte; 0 outside their bound; the largest distance 0.000 of its bound, at a chain of -0.75 with a "
        "spread of 0.75 (printed, not gated); the cache untouched"
    )
    # over the calls of a run: a second call that is on its chain too leaves the kept element ...
    other_twin, other_ref = one_row(0.25)
    assert float(other_twin.bound[0][0, 0]) == 0.25
    tie_out.BlockTap._judge(stats, (other_ref,), (other_ref,), [], [], 0, other_twin)
    assert stats.ok and (stats.elements, stats.worst_share, stats.worst_at) == (1024, 0.0, (-0.75, 0.75)), "a later call on its chain: the kept element stays"
    # ... the first call that is off its chain takes its place (the model's extreme at 0.9 of its largest rounding error) ...
    extreme_twin, extreme_ref = _built_attention(0.0, 1.0, -1.0)
    tie_out.BlockTap._judge(stats, (_model_extreme(0.9),), (extreme_ref,), [], [], 0, extreme_twin)
    assert stats.ok and (stats.elements, stats.worst_share, stats.worst_at) == (1536, 0.8984375, (0.0, 1.0)), "a larger share: its element"
    # ... and a later call on its chain does not take that back
    tie_out.BlockTap._judge(stats, (ref,), (ref,), [], [], 0, twin)
    assert stats.ok and (stats.elements, stats.worst_share, stats.worst_at) == (2048, 0.8984375, (0.0, 1.0)), "the largest share stays with its element"
    # a call that cannot be compared is no judged call: nothing is kept, and the call behind it is the first judged one
    behind = tie_out.BlockStats("attention_bf16_weights")
    tie_out.BlockTap._judge(behind, (other_ref, other_ref), (other_ref,), [], [], 0, other_twin)
    assert not behind.ok and (behind.elements, behind.outside, behind.worst_at) == (0, 1, (0.0, 0.0)), "two outputs: nothing is compared and nothing kept"
    tie_out.BlockTap._judge(behind, (ref,), (ref,), [], [], 0, twin)
    assert not behind.ok and (behind.elements, behind.outside, behind.worst_share, behind.worst_at) == (512, 1, 0.0, (-0.75, 0.75)), "the first judged call"
    # an element whose chain and spread are both zero (its attended value is zero) is kept like any other: as the first judged call's ...
    null_twin, null_ref = one_row(0.0)
    stats = _judged_attention(null_ref, null_twin, null_ref)
    assert stats.ok and (stats.elements, stats.worst_share, stats.worst_at) == (512, 0.0, (0.0, 0.0)), "the first judged call's first element: a chain of zero"
    tie_out.BlockTap._judge(stats, (ref,), (ref,), [], [], 0, twin)
    assert stats.ok and (stats.elements, stats.worst_share, stats.worst_at) == (1024, 0.0, (0.0, 0.0)), "a later call on its chain: a chain of zero stays too"
    # ... and with the largest share: a later call with a smaller share does not take its place
    off = null_ref.clone()
    off[0, 0] = 2.0**-10  # an output that is not zero where every attended value is: its bound is one spacing at the smallest normal fp32 value
    stats = _judged_attention(off, null_twin, null_ref)
    assert not stats.ok and (stats.outside, stats.worst_share, stats.worst_at) == (1, 2.0**123, (0.0, 0.0)), "outside, with its share, at a chain of zero"
    tie_out.BlockTap._judge(stats, (_model_extreme(0.9),), (extreme_ref,), [], [], 0, extreme_twin)
    kept = (stats.elements, stats.outside, stats.worst_share, stats.worst_at)
    assert not stats.ok and kept == (1024, 1, 2.0**123, (0.0, 0.0)), "a later call with a smaller share does not take its place"
    # the count beside the share is of BYTES: an output of -0 where the chain's byte is +0 is on its chain, and another byte
    signed = null_ref.clone()
    signed[0, 0] = -0.0
    stats = _judged_attention(signed, null_twin, null_ref)
    assert stats.ok and (stats.other_value, stats.worst_share) == (1, 0.0), "an output of -0 at a chain of +0: inside, and another byte than the chain's"


def test_every_control_arm_of_the_library_attention_band_fails_and_what_the_band_cannot_see_is_pinned(monkeypatch):
    """The control arms of the band ``attention_bf16_weights`` through the judge; each must FAIL.  Two say that the band is
    not slack, and are derived on built operands: at a coefficient of 0.5 the model's own extreme at 0.9 of its largest
    rounding error is outside, and with the weight term of weights held at fp32 (``2**-24``) a chain whose weights are
    rounded to bf16 is outside.  The others are defects of an operand, on drawn operands at 100 live rows -- one full page
    and 36 rows of a second -- with a kernel-like chain of the model in the kernel's place.  In the CALL, judged against the
    correct twin: the lengths one too long (a masked row let in), one too short (the newest attended row dropped), the
    page-table entry of the last attended page exchanged with the entry of a full page, the scores scaled by 1.  In the
    TWIN, with the call's bytes correct: the value row of the newest attended position read from the next slot, of
    position 0 from slot 1, and in a step of six rows every row attending to all rows after the step.  What an arm needs
    is pinned with it: two full pages exchanged are the same rows.  And what the band cannot see is pinned as that: a
    value row off by one slot among 2,051 drawn rows, where the softmax weight of that row is below the band's term."""
    from types import SimpleNamespace

    from model_samples import tie_out
    from model_samples.providers import torch_ref
    from model_samples.state import KvCache

    # ---- derived, on the built operands: the band is not slack by a factor of two, and its weight term is what the model needs
    twin, ref = _built_attention(0.0, 1.0, -1.0)
    twin_w, ref_w = _built_attention(4.0, 0.375, -1.0)
    out_w = torch.full((1, 512), 0.73046875 * 0.375 + 0.26953125 * -1.0, dtype=torch.float64).bfloat16()
    assert _judged_attention(_model_extreme(0.9), twin, ref).ok and _judged_attention(out_w, twin_w, ref_w).ok, "both inside the band as declared"
    with monkeypatch.context() as arm:
        arm.setattr(tie_out, "ATTENTION_BAND_COEFFICIENT", 0.5)
        half = _judged_attention(_model_extreme(0.9), twin, ref)
    assert not half.ok and half.outside == 512 and half.worst_share == 2 * 0.8984375, "at a coefficient of 0.5 the model's extreme at 0.9 is outside"
    with monkeypatch.context() as arm:
        arm.setattr(tie_out, "ATTENTION_WEIGHT_TERM", 2.0**-24)
        narrow = _judged_attention(out_w, twin_w, ref_w)
    assert (
        not narrow.ok and narrow.outside == 512 and abs(narrow.worst_share - 26.5469) < 1e-3
    ), "with the weight term 2**-24, weights rounded to bf16 are outside"

    # ---- defects of an operand, on drawn operands: a pool of three pages of one sequence, 100 live rows
    queries, keys, values = _drawn_attention(3, 7001, 6, 9001)
    live = 100

    def call(rows, lengths=None, table=(0, 1, 2), scale=None):
        """What a kernel-like chain of the model returns for a step of ``rows`` rows that it is handed as a kernel is: the
        length of the sequence AFTER the step (the live rows, where none is given), a page table and a scale.  Row ``t``
        attends to ``lengths - (rows - 1 - t)`` positions; position ``j`` is the row ``j % 64`` of the page ``table[j // 64]``."""
        out, lengths = [], live if lengths is None else lengths
        for row in range(rows):
            read = [table[at // 64] * 64 + at % 64 for at in range(lengths - (rows - 1 - row))]
            out.append(_kernel_like_attention(queries[row], keys[:, read], values[:, read], scale))
        return torch.stack(out)

    def twin_with(rows, every=False, newest=False, oldest=False):
        """A twin written from the definition for a step of ``rows`` rows that ends at ``live`` live rows, with one thing
        wrong: every row attends to all rows after the step; the value row of the newest attended position is read from
        the next slot; the value row of position 0 from slot 1."""
        terms = []
        for row in range(rows):
            attended = live if every else live - (rows - 1 - row)
            read = values[:, :attended].clone()
            if newest:
                read[:, attended - 1] = values[:, attended]
            if oldest:
                read[:, 0] = values[:, 1]
            terms.append(_attention_terms(queries[row], keys[:, :attended], read))
        return SimpleNamespace(bound=tuple(torch.stack([term[part] for term in terms]) for part in (0, 1)))

    def correct(rows):
        twin = torch_ref.AttentionBound(None, None, None, {})
        return twin, twin(queries[:rows], KvCache(keys[None], values[None], rows), live - rows)

    twin, ref = correct(1)
    clean = _judged_attention(call(1), twin, ref)
    assert clean.ok and clean.outside == 0 and 0.25 < clean.worst_share < 0.75, clean.line()
    assert torch.equal(twin_with(1).bound[0], twin.bound[0]) or bool(((twin_with(1).bound[0] - twin.bound[0]).abs() <= 1e-12 * twin.bound[1]).all())
    in_the_call = (
        ("the lengths one too long: a masked row let in", dict(lengths=live + 1)),
        ("the lengths one too short: the newest attended row dropped", dict(lengths=live - 1)),
        ("the entry of the last attended page exchanged with the entry of a full page", dict(table=(1, 0, 2))),
        ("the scores scaled by 1, a library's default, in place of the chain's scale", dict(scale=1.0)),
    )
    for name, defect in in_the_call:
        stats = _judged_attention(call(1, **defect), twin, ref)
        assert not stats.ok and stats.finite and stats.outside > 6144 // 10 and stats.worst_share > 10, f"{name}: {stats.line()}"
    in_the_twin = (
        ("the value row of the newest attended position read from the next slot", dict(newest=True)),
        ("the value row of position 0 read from slot 1", dict(oldest=True)),
    )
    for name, defect in in_the_twin:
        stats = _judged_attention(call(1), twin_with(1, **defect), ref)
        assert not stats.ok and stats.finite and stats.outside > 6144 // 10 and stats.worst_share > 10, f"{name}: {stats.line()}"
    twin6, ref6 = correct(6)
    step = call(6)
    clean = _judged_attention(step, twin6, ref6)
    assert clean.ok and clean.elements == 6 * 6144 and clean.outside == 0, "a step of six rows, each attending to the rows up to its own"
    stats = _judged_attention(step, twin_with(6, every=True), ref6)
    assert not stats.ok and stats.outside > 36864 // 10, f"every row of the step attends to all rows after the step: {stats.line()}"
    last = _judged_attention(step[5:], SimpleNamespace(bound=tuple(term[5:] for term in twin_with(6, every=True).bound)), ref6[5:])
    assert last.ok, "the last row of the step attends to all of them either way: the arm is the five rows in front of it"
    # what the page-table arm needs: a last page that is not full.  Two wholly live pages exchanged are the same attended rows
    assert torch.equal(call(1, lengths=128, table=(1, 0, 2)).view(torch.int16), call(1, lengths=128).view(torch.int16)), "two full pages: no byte changes"
    # with the weight term of fp32 weights the drawn kernel-like chain is outside too
    with monkeypatch.context() as arm:
        arm.setattr(tie_out, "ATTENTION_WEIGHT_TERM", 2.0**-24)
        assert _judged_attention(call(1), twin, ref).outside > 100

    # ---- what the band cannot see, pinned as that: one value row off by one slot among 2,051 attended rows
    queries, keys, values = _drawn_attention(39, 5051, 1, 8051, query_scale=4.0)
    live = 2051
    twin, ref = correct(1)
    out = call(1, table=tuple(range(39)))
    assert _judged_attention(out, twin, ref).ok
    for name, defect, at in (("the newest attended position", dict(newest=True), live - 1), ("position 0", dict(oldest=True), 0)):
        stats = _judged_attention(out, twin_with(1, **defect), ref)
        assert stats.ok and stats.outside == 0 and stats.worst_share < 1.0, f"the value row of {name} off by one slot, among 2,051: inside"
        weight = max(float((raw / total)[at]) for raw, total in _attention_weights(queries[0], keys[:, :live]))
        assert weight < tie_out.ATTENTION_WEIGHT_TERM / 4, f"{name}: the softmax weight of that row is below the band's term in every head"


def test_rule_8_states_its_one_exception_word_for_word_and_so_do_the_places_that_state_the_rule():
    """Rule 8 of the area's guide says that a band is the kernel's own contract against a twin at the kernel's rounding points.
    The band of a library kernel that documents no contract is the one exception, and the guide states it in one
    sentence, word for word and once, directly behind the sentence it is the exception to and in front of the bullet
    that follows it.  The two places of the helper layer that state the rule state the exception too, and each says
    what the band is derived from: a stated model of the kernel's arithmetic, never the kernel's arithmetic itself."""
    from model_samples import providers, tie_out

    guide = " ".join((MODELS / "AGENTS.md").read_text().split())
    rule = (
        "8. **Bands are never widened in place.** The band of a block against the reference is named in its provider row (`tie_out.BANDS`). - A provider "
        "that rounds at other points than the reference chain is compared with its twin (`Block.twin`): a reference written at the kernel's rounding "
        "points, in the reference's accumulation precision. The band is the kernel's documented contract, never a width covering the distance between "
        "two chains."
    )
    exception = (
        "The one exception is a library kernel that documents no numerics contract and whose rounding points are not established: its twin is the "
        "reference chain itself, which also keeps the operand terms of a bound, and its band is that bound -- derived from a stated model of the kernel's "
        "arithmetic, declared with that model before any run of the row, held by control arms that must fail, and never taken from a measured distance; "
        "an element outside it is a finding, and the family README says beside the row that the row is not within one bf16 unit of the chain."
    )
    behind = "- A band names the outputs it applies to; an unnamed output is judged by its dtype, and updated state byte for byte."
    assert f"{rule} {exception} {behind}" in guide, "rule 8 up to the sentence of the twin, its one exception word for word, and the bullet behind it"
    assert guide.count("The one exception is a library kernel") == 1 and guide.count("never a width covering the distance between two chains.") == 1
    for name, text in (("tie_out", tie_out.__doc__), ("BlockBase.twin", providers.BlockBase.twin.__doc__)):
        stated = " ".join((text or "").split())
        assert "a band is never widened to cover a chain that rounds elsewhere." in stated, f"{name}: the rule"
        assert (
            "a library kernel that documents no numerics contract and whose rounding points are not established: its twin is the reference chain itself, "
            in stated
        ), name
        assert (
            "which also keeps the operand terms of a bound, and its band is that bound" in stated
            and "derived from a stated model of the kernel's arithmetic" in stated
        ), name
    assert "(``attention_bf16_weights`` below)" in " ".join(tie_out.__doc__.split()) and "(``torch_ref.AttentionBound``)" in " ".join(
        providers.BlockBase.twin.__doc__.split()
    )
    assert (
        "attention_bf16_weights" in tie_out.BANDS and providers.BlockBase.twin(providers.BlockBase, "the chain") == "the chain"
    ), "the default twin is the reference block"


# ---------------------------------------------------------------------------------------------------------------------------
# The row of the attention core on a library: the real adapter of the paged decode attention over the stand-in of the
# library, on the CPU -- what it hands the library, what it declines, its first use, and the kernel check on its kernel.
# ---------------------------------------------------------------------------------------------------------------------------
def _attention_plans(monkeypatch, **stand_in):
    """``planned(sequences, tokens, max_model_len, table, cc)``: the attention core's block in a plan of the stack ``native``
    on compute capability ``cc`` (10.0 where none is named), for the tiny geometry and ONE set of stand-ins (``stand_in``:
    its options): every block it returns is an adapter of the same process."""
    from model_samples import blocks

    model, build = _tiny_model(), _stand_ins(monkeypatch, **stand_in)

    def planned(sequences=1, tokens=1, max_model_len=RECORDED_LENGTH, table=None, cc=(10, 0)):
        facts = blocks.DeviceFacts(cc, "stated device", 148, 180 * GIB, 80 * GIB)
        options = {"reference_precision": "fp64", "device": "cpu", "max_model_len": max_model_len}
        plan = blocks.plan(table or model.provider_table, model.regimes, facts, sequences, tokens, "native", model=model, options=options, factories=build)
        return plan["attention.core"]

    return planned


def _attention_step(sequences=1, tokens=1, live=100, max_model_len=RECORDED_LENGTH):
    """Drawn operands of one step of an attention core at the published widths, on the CPU: ``(queries [rows, 24, 256], a
    paged cache of ``live`` positions per sequence whose every row is drawn, the position the step starts at)``."""
    from model_samples.state import DecodeState

    heads, kv_heads, width = ATTENTION_WIDTHS
    shapes = {"conv": (1, 1), "kv": (kv_heads, width), "rotary": (4, 10000.0)}
    cache = DecodeState(("full_attention",), sequences, live, shapes, torch.device("cpu"), tokens=tokens, page=64, max_model_len=max_model_len).layers[0]
    generator = torch.Generator().manual_seed(11)
    cache.pool.copy_(torch.randn(cache.pool.shape, generator=generator).bfloat16())
    return torch.randn(sequences * tokens, heads, width, generator=generator).bfloat16(), cache, live - tokens


def test_the_attention_adapter_hands_the_library_the_state_as_it_is(monkeypatch):
    """What the library's paged decode attention is handed for a step of two sequences of six tokens: the query as the
    block got it; the two halves of the pool and the rows of the state's page table and lengths, each the state's own
    memory; the run's maximum length, the chain's scale and the tokens per sequence as Python numbers; the field of
    the row's form, and the library's own backend for these GPU classes named in every call; one workspace per device,
    in every call and every adapter; a zero counter buffer of the library's size per sequence count; an output buffer
    of the call's own, which is what the block returns.  The library's launch option, its window and its other outputs
    are left at its defaults.  A row that leaves out the field of the form is a wrong table, and a checkpoint with
    another count of query heads, of key / value heads or another head width keeps the reference chain."""
    from dataclasses import replace

    from model_samples.providers import flashinfer_calls

    planned = _attention_plans(monkeypatch)
    block = planned(2, 6)
    assert type(block).__name__ == "DecodeAttentionBlock" and (block.provider, block.stored_weights) == ("FlashInfer", False), "the block has no weight"
    queries, cache, position = _attention_step(2, 6)
    assert block.reserve_bytes() == 394 * 2**20, "the memory sentence of a plan counts the workspace while the device does not hold it"
    del ATTENTION_CALLS[:]
    first, second = block(queries, cache, position), block(queries, cache, position)
    assert block.reserve_bytes() == planned(1, 1).reserve_bytes() == 0, "and no plan of the device counts it once it is built"
    handed, again = ATTENTION_CALLS
    width = ATTENTION_WIDTHS[2]
    keys, values = handed["kv_cache"]
    assert handed["query"] is queries and type(handed["kv_cache"]) is tuple and keys.shape == values.shape == (*cache.pool.shape[:3], width)
    assert keys.data_ptr() == cache.pool.data_ptr() and values.data_ptr() == cache.pool[..., width:].data_ptr(), "the two halves of the pool, its own memory"
    blocks, lengths = cache.attended(position)
    assert handed["block_tables"].data_ptr() == blocks.data_ptr() == cache.table.blocks.data_ptr() and tuple(handed["block_tables"].shape) == (2, 36)
    assert handed["seq_lens"].data_ptr() == lengths.data_ptr() and handed["seq_lens"].tolist() == [100, 100], "what each sequence holds after the step"
    numbers = [handed[name] for name in ("max_seq_len", "bmm1_scale", "bmm2_scale", "q_len_per_req")]
    assert numbers == [RECORDED_LENGTH, 0.0625, 1.0, 6] and [type(number) for number in numbers] == [int, float, float, int], "numbers of the host"
    assert handed["kv_layout"] == "HND" and block.serving() == (ATTENTION_ENTRY, (("KV_LAYOUT", "HND"),))
    assert handed["backend"] == flashinfer_calls.DECODE_ATTENTION_BACKEND != "auto", "the backend is named in the call: not the library's choice by the GPU"
    assert handed["enable_pdl"] is None and handed["window_left"] == -1 and handed["more"] == {}, "no launch option, no window, no further output"
    planned(1, 1)(*_attention_step(1, 1))  # another adapter of the same device, at another sequence count
    workspace, counter = handed["workspace_buffer"], handed["multi_ctas_kv_counter_buffer"]
    assert workspace is again["workspace_buffer"] is ATTENTION_CALLS[2]["workspace_buffer"] and (workspace.dtype, workspace.numel()) == (
        torch.uint8,
        394 * 2**20,
    )
    assert counter is again["multi_ctas_kv_counter_buffer"] and counter is not ATTENTION_CALLS[2]["multi_ctas_kv_counter_buffer"]
    assert (counter.dtype, counter.numel()) == (torch.uint8, 608) and not counter.any(), "of the library's size for this count on 148 SMs, zero"
    assert first.shape == (12, 24 * 256) and first.data_ptr() == handed["out"].data_ptr() != second.data_ptr() == again["out"].data_ptr()
    assert handed["out"].shape == queries.shape and handed["out"].is_contiguous() and torch.equal(first.view(torch.int16), second.view(torch.int16))
    assert handed["out"].untyped_storage().nbytes() == 2 * queries.numel(), "the whole of a buffer, not a view inside a larger one"
    table = dict(_tiny_model().provider_table)
    table["attention.core"] = [replace(row, form=()) if row.key == "flashinfer.decode_attention" else row for row in table["attention.core"]]
    with pytest.raises(ValueError, match="provider row flashinfer.decode_attention: its form does not name KV_LAYOUT; this adapter's rows name KV_LAYOUT for"):
        planned(table=table)
    # the row names the three attention widths its call was run with: a checkpoint with another of any of them keeps the chain
    from model_samples import blocks
    from model_samples.qwen_dense import QwenDense

    stated = blocks.DeviceFacts((10, 0), "stated device", 148, 0, 0)
    for field, value in (("attn_heads", 16), ("attn_kv_heads", 2), ("attn_head_dim", 128)):
        other = QwenDense(replace(QwenDense.reduced(2).geometry, **{field: value}), (f"another {field}",))
        rows = blocks.plan(other.provider_table, other.regimes, stated, 1, 1, "native", model=other, options={"device": "cpu"}).rows
        assert rows["attention.core"].key == "torch_ref.attention", f"{field}: the chain's row"


def test_the_attention_adapter_declines_what_the_library_would_misread(monkeypatch):
    """The library checks neither the dtype, the layout nor the place of the tables it reads, so the adapter does, and each
    is a ``ValueError`` that names the operand: lengths or a page table of 64-bit words, a strided view of either, a
    page table narrower than the width of the run's maximum length, one off a 16-byte boundary, one on another device;
    a strided output; a pool half, either one, whose last axis is strided; a step past the maximum length, where the step
    that ends at it is made; a cache that is no pool of pages; a plan that names no maximum length.  Why the tables are held at two sequences of two pages is pinned
    with them: read as 32-bit words, tables of 64-bit words give other bytes there and the same bytes at one sequence
    of one page."""
    from model_samples.state import KvCache, PagedKvCache

    planned = _attention_plans(monkeypatch)
    block = planned(2, 1)
    queries, cache, position = _attention_step(2, 1)
    table = cache.table
    good = block(queries, cache, position).clone()

    def declined(match, **wrong):
        kept = {name: getattr(table, name) for name in wrong}
        vars(table).update(wrong)
        with pytest.raises(ValueError, match=f"{ATTENTION_ENTRY}: {match}"):
            block(queries, cache, position)
        vars(table).update(kept)

    declined("the lengths of this step is of dtype torch.int64: the kernel reads 32-bit words", lengths=table.lengths.long())
    declined("the page table of this step is of dtype torch.int64: the kernel reads 32-bit words", blocks=table.blocks.long())
    declined("the lengths of this step is a strided view: the kernel reads adjacent words", lengths=table.lengths.repeat_interleave(2, dim=1)[:, ::2])
    declined("the page table of this step is a strided view: the kernel reads adjacent words", blocks=table.blocks.repeat_interleave(2, dim=1)[:, ::2])
    declined(r"the page table of this step is of shape \(2, 2\): this call reads \(2, 36\)", blocks=table.blocks[:, :2].contiguous())
    shifted = torch.zeros(table.blocks.numel() + 1, dtype=torch.int32)[1:].view(table.blocks.shape).copy_(table.blocks)
    declined("the page table of this step is not 16 bytes aligned", blocks=shifted)
    declined("the page table of this step is on meta, the query on cpu", blocks=table.blocks.to("meta"))
    with monkeypatch.context() as scoped:
        scoped.setattr(type(block), "_output", lambda self, q: torch.empty((*q.shape[:2], 2 * q.shape[2]), dtype=torch.bfloat16)[..., ::2])
        with pytest.raises(ValueError, match="the output buffer of this call is a strided view: the kernel writes adjacent rows"):
            block(queries, cache, position)
    with monkeypatch.context() as scoped:
        scoped.setattr(PagedKvCache, "k_pool", property(lambda self: self.pool[..., ::2]))
        with pytest.raises(ValueError, match="the key half of the pool has stride 2 on its last axis: the kernel reads adjacent elements"):
            block(queries, cache, position)
    with monkeypatch.context() as scoped:
        scoped.setattr(PagedKvCache, "v_pool", property(lambda self: self.pool[..., 1::2]))
        with pytest.raises(ValueError, match="the value half of the pool has stride 2 on its last axis: the kernel reads adjacent elements"):
            block(queries, cache, position)
    with pytest.raises(ValueError, match="a step of 1 token at position 99 passes the run's maximum length 99"):
        planned(2, 1, max_model_len=99)(queries, cache, position)
    full = _attention_step(2, 1, max_model_len=100)  # the same step in a run of 100 positions ends at the maximum length: the last step of that run
    assert torch.equal(planned(2, 1, max_model_len=100)(*full).view(torch.int16), good.view(torch.int16)), "a step that ends at the maximum length is made"
    with pytest.raises(ValueError, match="reads an attention cache that is a pool of pages of 64 positions; this block is handed KvCache"):
        block(queries, KvCache(cache.keys(100, torch.bfloat16), cache.values(100, torch.bfloat16)), position)
    with pytest.raises(ValueError, match="a whole number of positions; the plan of this block names max_model_len=None"):
        planned(2, 1, max_model_len=None)(queries, cache, position)
    # with the adapter's own check out of the way, a kernel reads the first 32-bit words of whatever it is handed
    monkeypatch.setattr(type(block), "_held", lambda self, *operands: None)
    vars(table).update(lengths=table.lengths.long(), blocks=table.blocks.long())
    assert not torch.equal(block(queries, cache, position).view(torch.int16), good.view(torch.int16)), "two sequences of two pages: other bytes"
    one, small, start = _attention_step(1, 1, live=37)
    right = planned(1, 1)(one, small, start).clone()
    vars(small.table).update(lengths=small.table.lengths.long(), blocks=small.table.blocks.long())
    assert torch.equal(planned(1, 1)(one, small, start).view(torch.int16), right.view(torch.int16)), "one sequence of one page: the defect, and the right bytes"


def test_the_first_use_of_the_librarys_attention_and_the_kernel_binaries_it_never_fetches(monkeypatch, tmp_path, capsys):
    """The first use of the paged decode attention: one scratch call per form the run calls -- its sequences and one
    sequence alone, at each plan's tokens per sequence -- at the run's maximum length.  In front of the first call the
    adapter holds every kernel binary the library's table names for the call's data types, head width, paging and page
    on the GPU's class to the manifest of the kernel set: a
    binary that is missing -- the one a step launches or the one its selection reads first --, one changed byte, a
    line the manifest lacks, a header that is not the manifest's, no manifest or a manifest that is not the file the
    library names is a ``SKIP`` with the library not called, and a library that would print a path and launch another kernel is never reached in that state.  The set
    is the class's own and is held once per process.  A file the library itself does not
    find with its download refused is a ``SKIP`` in the sample's words, by what the installation holds, with no path
    of the library's text; the states of the package in
    which the library's import would print a path are declined at plan time, before the library is reached.  A check
    of the library on the operands is the adapter's fault, any other failure the installation's by its type alone, and
    a fault of the adapter's own code is raised as it is.  The interpreter's report of the missing file inside the
    library's loader stays off the streams during the first use; every other report reaches the hook that was there."""
    from types import SimpleNamespace

    from model_samples import census
    from model_samples.providers import flashinfer_calls

    planned = _attention_plans(monkeypatch)
    del ATTENTION_CALLS[:]
    for tokens in (1, 6):  # the plan of a prompt's decode steps and the plan of the verify steps of one session
        block = planned(2, tokens)
        block.first_use(2, tokens)
        assert block.serving() == (ATTENTION_ENTRY, ()), "a first use is no call of a step"
    made = [(call["query"].shape[0], call["q_len_per_req"], call["max_seq_len"], tuple(call["block_tables"].shape)) for call in ATTENTION_CALLS]
    assert made == [(2, 1, 2051, (2, 36)), (1, 1, 2051, (1, 36)), (12, 6, 2051, (2, 36)), (6, 6, 2051, (1, 36))], "four forms, each at the maximum length"

    # the kernel binaries, held before the first call: what the set's header names for the GPU's class, each as the manifest lists it
    install = "pip install --no-deps --index-url https://flashinfer.ai/whl -c samples/models/constraints-native.txt -r samples/models/requirements-flashinfer-cubin.txt"
    without = "--stack fe-entries runs the cuDNN Frontend rows of this stack without the package"
    filled = 'fill it (the family README, "What the first run does") or install the package'
    anew = 'install it anew (the family README, "What the first run does")'
    lacks = (
        "does not hold the kernel binaries the library's table names for this call's data types, head width, paging and page on this GPU's class "
        "as their manifest lists them"
    )
    one, header = "1 of 3 missing or changed", "include/flashInferMetaInfo.h"
    held = (
        ("the binary a step launches is missing", {"without": [FI_ATTENTION]}, one),
        ("the binary the selection reads first is missing", {"without": [FI_ATTENTION_OTHER]}, one),
        ("one byte of a binary differs", {"changed": [FI_ATTENTION]}, one),
        ("the manifest has no line for a binary", {"unlisted": [FI_ATTENTION]}, one),
        ("the header names another SHA-256 than the binary's", {"misnamed": [FI_ATTENTION]}, one),
        ("a binary of this class that no step of the run reads is missing", {"without": [ATTENTION_SET[2][1]]}, one),
        ("no manifest", {"manifest": False}, "their manifest is not there"),
        ("one byte of the header differs", {"changed": [header]}, "the header of the kernel set is not the file their manifest lists"),
        ("a header without the table", {"header": "#pragma once\n"}, "the header of the kernel set is not the table of kernels this adapter reads"),
        ("a table without a kernel of this call", {"kernels": ATTENTION_SET[4:]}, "the header of the kernel set names no kernel of this call for this GPU"),
    )
    reports = []
    monkeypatch.setattr(sys, "unraisablehook", lambda unraisable: reports.append(type(unraisable.exc_value).__name__))
    for number, (case, defect, found) in enumerate(held):
        directory = _kernel_binaries(tmp_path / f"held_{number}", **defect)
        block = _attention_plans(monkeypatch, kernel_binaries=directory)(2, 6)
        capsys.readouterr()
        with pytest.raises(NotImplementedError) as skipped:
            block.first_use(2, 6)
        assert str(skipped.value) == f"{ATTENTION_ENTRY}: the installed flashinfer-cubin 0.7.2.post1 {lacks} ({found}); {anew}; {without}", case
        assert ATTENTION_CALLS == [] and capsys.readouterr() == ("", ""), f"{case}: the library was not called and nothing was printed"
        if FI_ATTENTION_OTHER in defect.get("without", ()):  # behind a hold that skipped, a step is a step: the library's error is raised as it is
            with pytest.raises(RuntimeError, match="Failed to load cubin for"):
                block(*next(block._scratches(torch.device("cpu"), 1, 1)))
            assert reports == ["RuntimeError"], "and the interpreter's report is the interpreter's"
            del reports[:], ATTENTION_CALLS[:]
    # the set is the class's own, and it is held once per process: a kernel of another 10.x class alone is not held on
    # this one and is on that one; what a process held of a class it does not read again
    other_class = _kernel_binaries(tmp_path / "other_class", without=[ATTENTION_SET[3][1]])
    planned = _attention_plans(monkeypatch, kernel_binaries=other_class)
    planned(2, 6).first_use(2, 6)
    with pytest.raises(NotImplementedError, match=re.escape(f"{lacks} ({one})")):
        planned(2, 6, cc=(10, 3)).first_use(2, 6)
    (other_class / "2d6a" / "fmha" / flashinfer_calls.DECODE_ATTENTION_BACKEND / "checksums.txt").unlink()
    planned(1, 1).first_use(1, 1)
    with pytest.raises(NotImplementedError, match="their manifest is not there"):
        _attention_plans(monkeypatch, kernel_binaries=other_class)(1, 1).first_use(1, 1)
    # the manifest is held as well, to the SHA-256 the library names for it: one that lists every binary of the set as it
    # is and is another file than the library's -- a line more, of a file of another kernel set -- is the SKIP, with the
    # library not called (the library would say so itself, at its first read, in a line that holds a local path)
    other_manifest = _kernel_binaries(tmp_path / "other_manifest")
    block = _attention_plans(monkeypatch, kernel_binaries=other_manifest)(2, 6)
    with (other_manifest / "2d6a" / "fmha" / flashinfer_calls.DECODE_ATTENTION_BACKEND / "checksums.txt").open("a") as manifest:
        manifest.write(f"{'0' * 64}  a_binary_of_another_kernel_set.cubin\n")
    capsys.readouterr()
    with pytest.raises(NotImplementedError, match=re.escape(f"{lacks} (their manifest is not the file the library names); {anew}; {without}")):
        block.first_use(2, 6)
    assert ATTENTION_CALLS == [] and capsys.readouterr() == ("", ""), "the library was not called and nothing was printed"
    # the sentence by what the installation holds: (the package's metadata and module, the directory variable)
    launched_missing = _kernel_binaries(tmp_path / "by_installation", without=[FI_ATTENTION])
    for binaries, variable, sentence in (
        ((None, False), False, f" needs the kernel-binary package flashinfer-cubin 0.7.2.post1, which is not installed: {install}; {without}"),
        ((None, False), True, f": the directory FLASHINFER_CUBIN_DIR names {lacks} ({one}); {filled}: {install}; {without}"),
    ):
        planned = _attention_plans(monkeypatch, library_binaries=binaries, kernel_binaries=launched_missing)
        if variable:
            monkeypatch.setenv(flashinfer_calls.BINARIES_DIRECTORY, "/a/local/path/cubins")
        block = planned(2, 6)  # without the package the directory variable declines nothing at the plan: the directory is the library's to read
        with pytest.raises(NotImplementedError) as skipped:
            block.first_use(2, 6)
        assert str(skipped.value) == ATTENTION_ENTRY + sentence and ATTENTION_CALLS == []
    # one verdict per folder of a process: a folder that holds is not the answer for one that does not
    folders = [
        str(directory / "2d6a" / "fmha" / flashinfer_calls.DECODE_ATTENTION_BACKEND) for directory in (_kernel_binaries(tmp_path / "whole"), launched_missing)
    ]
    family = dict(census.FAMILIES)["fi_decode_attention"]
    assert [flashinfer_calls.hold_binaries(folder, family, (10, 0)) for folder in folders] == ["", one]
    # what the hold stands in front of: with it out of the way, a library that lacks the binary it would launch raises
    # nothing, writes a line of its own with a local path and launches the kernel its selection read first
    block = _attention_plans(monkeypatch, kernel_binaries=launched_missing)(1, 1)
    monkeypatch.setattr(type(block), "_hold", lambda self: None)
    capsys.readouterr()
    block.first_use(1, 1)
    assert len(ATTENTION_CALLS) == 1 and "/a/local/path" in capsys.readouterr().err, "no error: a path on standard error, and another kernel"
    # a library the hold cannot ask -- it cannot be imported, its artifact paths name no such kernel set, or its checksums no
    # manifest of that set -- is the installation's
    for defect in ("import", "paths", "checksums"):
        with pytest.raises(
            NotImplementedError, match=rf"{ATTENTION_ENTRY}: FlashInfer could not make its first use on this installation \((Import|Lookup)Error\); "
        ):
            _attention_plans(monkeypatch, library_defect=defect)().first_use(1, 1)
        assert ATTENTION_CALLS == []
    # the table of a header: a record that has not the fields of its type is no such table
    family = re.compile("^a_kernel")
    table = 'struct Meta\n{\n    int mSM;\n    const char* mFuncName;\n    const char* sha256;\n};\nstatic const Meta all[] = {\n{ kSM_100f, "a_kernel", "00"},\n%s};\n'
    assert flashinfer_calls.kernel_set(table % "", family, (10, 0)) == {"a_kernel": "00"} and flashinfer_calls.kernel_set(table % "", family, (12, 0)) == {}
    with pytest.raises(ValueError, match="a record of the header's table has 2 values, its type 3 fields"):
        flashinfer_calls.kernel_set(table % '{ kSM_100f, "a_kernel_of_two_values"},\n', family, (10, 0))

    # a file the library itself did not find, by what the installation holds
    missing = (
        ((None, False), False, "manifest", f" needs the kernel-binary package flashinfer-cubin 0.7.2.post1, which is not installed: {install}; {without}"),
        (
            (VALIDATED_LIBRARY, True),
            False,
            "binary",
            f": the installed flashinfer-cubin 0.7.2.post1 {lacks} (the library found a file missing at 2 x 6 rows at maximum length 2051); {anew}; {without}",
        ),
    )
    for binaries, directory, defect, sentence in missing:
        block = _attention_plans(monkeypatch, attention_defect=defect, library_binaries=binaries)(2, 6)
        if directory:
            monkeypatch.setenv(flashinfer_calls.BINARIES_DIRECTORY, "/a/local/path/cubins")
        with pytest.raises(NotImplementedError) as skipped:
            block.first_use(2, 6)
        assert str(skipped.value) == ATTENTION_ENTRY + sentence and reports == [], defect
    with pytest.raises(RuntimeError, match="Failed to load cubin for"):  # in a step nothing is renamed, and the report is the interpreter's
        block(*next(block._scratches(torch.device("cpu"), 1, 1)))
    assert reports == ["RuntimeError"]
    # during a first use every other report reaches the hook that was there: another module, another type, another text
    hook, loader = sys.unraisablehook, flashinfer_calls.LOADER
    with flashinfer_calls._first_use_reports():
        for error, module in ((RuntimeError("Artifact not found locally: a file"), "elsewhere"), (OSError("Artifact not found locally: a file"), loader)):
            sys.unraisablehook(SimpleNamespace(exc_value=error, object=SimpleNamespace(__module__=module)))
        sys.unraisablehook(SimpleNamespace(exc_value=RuntimeError("another failure"), object=SimpleNamespace(__module__=loader)))
    assert reports == ["RuntimeError", "RuntimeError", "OSError", "RuntimeError"] and sys.unraisablehook is hook

    # at plan time: the states of the package in which the library's import would print a path.  The library is not reached
    declined = (
        (
            (VALIDATED_LIBRARY, True),
            True,
            ": FLASHINFER_CUBIN_DIR is set and the kernel-binary package flashinfer-cubin is installed, and the library would read the directory",
        ),
        (
            (None, True),
            False,
            ": the module flashinfer_cubin can be imported and no distribution flashinfer-cubin is installed, so the version of the kernel binaries",
        ),
        (
            (VALIDATED_LIBRARY, False),
            False,
            ": the distribution flashinfer-cubin 0.7.2.post1 is installed and its module flashinfer_cubin cannot be found; install the package again: ",
        ),
        (
            ("0.7.1", True),
            False,
            " needs the kernel-binary package flashinfer-cubin at the library's version, 0.7.2.post1; this is 0.7.1, which the library ignores: ",
        ),
    )
    for binaries, directory, sentence in declined:
        planned = _attention_plans(monkeypatch, library_binaries=binaries, library_defect="import")
        if directory:
            monkeypatch.setenv(flashinfer_calls.BINARIES_DIRECTORY, "/a/local/path/cubins")
        with pytest.raises(NotImplementedError) as skipped:
            planned()
        assert str(skipped.value).startswith(ATTENTION_ENTRY + sentence) and str(skipped.value).endswith(without) and "/a/local" not in str(skipped.value)
    assert (
        _attention_plans(monkeypatch, library_binaries=(None, False))().entry == ATTENTION_ENTRY
    ), "neither metadata nor module: nothing is declined at the plan"

    # the library's own check of the operands, any other failure, and a fault of the adapter's own code
    for defect in ("refuses", "mistyped"):
        with pytest.raises(ValueError, match=f"{ATTENTION_ENTRY} declined the operands this adapter built for its first use: Invalid shape of out$"):
            _attention_plans(monkeypatch, attention_defect=defect)().first_use(1, 1)
    with pytest.raises(
        NotImplementedError, match=rf"{ATTENTION_ENTRY}: FlashInfer could not make its first use on this installation \(RuntimeError\); it builds"
    ):
        _attention_plans(monkeypatch, attention_defect="build")().first_use(1, 1)
    faulty = _attention_plans(monkeypatch)()
    monkeypatch.setattr(type(faulty), "_workspace", lambda self, device: [].missing)
    with pytest.raises(AttributeError, match="'list' object has no attribute 'missing'"):
        faulty.first_use(1, 1)


def test_the_kernel_check_holds_the_librarys_attention_kernel_and_every_control_arm_of_a_run_fails(monkeypatch, capsys):
    """The attention core's kernel through a session on constructed launch records.  At the maximum length the census was
    recorded with, another variant of the kernel fails by its class and the recorded kernel in another grid by its
    launch shape; a kernel of another head width or of another page is no member of the family; a fill or a multiply in front of the
    kernel and a kernel behind it each fail with their own sentence; a pool byte the call changed fails the block.  At
    another maximum length the row is held by membership: both variants pass and the lines say what is held.  On a
    GPU class the census has no section for, the run judges the block, holds the kernel by its family's name and
    states no identity.  Then the
    control arms of a run, through ``main()``, in a run of more than one page per sequence: the lengths one too long
    and one too short, the first two page-table columns exchanged, the scores at the library's default scale -- each
    must FAIL; in a run of one page per sequence the two columns name the same page and that arm cannot fail."""
    from model_samples import census, tie_out
    from model_samples.providers import flashinfer_calls
    from model_samples.state import PagedKvCache

    def run(max_model_len=RECORDED_LENGTH, **stand_in):
        session = _recorded_run(monkeypatch, sequences=1, stack="native", max_model_len=max_model_len, **stand_in)[0]
        session.trace(4)
        return tie_out.run_checks(session, prompt_tokens=3)

    recorded_in = "the target census plain / cc 10.0 / 148 SMs (recorded 2026-11-12)"
    in_front = "between attention.prepare (layer 1) and attention.core (layer 1): a kernel of the fallback family"
    between = "which no row there declares; nothing may sit between two exact rows"
    failing = {
        "variant": f"attention.core (layer 1) launched {FI_ATTENTION_OTHER!r}; {recorded_in} has the kernel class {ATTENTION_CLASS} of fi_decode_attention there",
        "shape": "attention.core (layer 1) launched fi_decode_attention with grid (4, 4, 1), block (512, 1, 1), shared memory 178,456 B; "
        f"{recorded_in} has grid (5, 4, 1), block (512, 1, 1), shared memory 178,456 B",
        "width": "attention.core (layer 1) did not launch fi_decode_attention: the step has 0 launch(es) of fi_decode_attention, the plan 1 = 1 call(s) x 1 per call",
        "page": "attention.core (layer 1) did not launch fi_decode_attention: the step has 0 launch(es) of fi_decode_attention, the plan 1 = 1 call(s) x 1 per call",
        "fill": f"launched {ELEMENTWISE[:93] + '...'!r} {in_front} pytorch, {between}",
        "multiply": f"launched {NVJET!r} {in_front} cublas, {between}",
        "behind": "launched 'flashinfer::ComputeLSEFromMDKernel(float*, float const*, int, int)' between attention.core (layer 1) and "
        "attention.gate_quantize (layer 1); no block of the plan launches it",
    }
    for defect, sentence in failing.items():
        checks = run(attention_defect=defect)
        assert not checks.ok and checks.kernels == census.FAIL, defect
        assert [line.strip().split(": ", 2)[2] for line in checks.lines if line.strip().startswith("kernels: ")] == [sentence], defect
        assert _line(checks, "attention.core ").endswith("[band: attention_bf16_weights]  PASS"), "its bytes are the chain's: the kernel check alone sees it"
    checks = run(attention_defect="writes")
    assert (
        not checks.ok and checks.kernels == census.PASS and _line(checks, "attention.core ").endswith("the cache WRITTEN  [band: attention_bf16_weights]  FAIL")
    )
    for defect in ("", "variant", "shape"):
        checks = run(max_model_len=4096, attention_defect=defect)
        assert checks.ok and checks.kernels == census.PASS, defect
        assert _line(checks, "attention.core: ").endswith(
            "1 launch per step (1 call x 1 per call); fi_decode_attention by name (no launch shape is declared for it)"
        )
        claim = _line(checks, "target census")
        assert "no recorded launch shape of attention.core for this run: max_model_len 4096 (recorded 2051)" in claim and "5 more by membership alone" in claim
    # the row's class says where the table selects it, not where the census has a record: on another 10.x class a run
    # judges the block in its band, holds the kernel by its family's name -- either variant -- and claims no identity
    unstated = "kernel identity: not stated for this run (no recorded section for cc 10.3 / 148 SMs: recorded is cc 10.0 / 148 SMs)"
    for defect in ("", "variant"):
        code, lines = _sample_run(monkeypatch, capsys, ["--stack", "native", "--sequences", "1"], cc=(10, 3), attention_defect=defect)
        assert code == 0 and lines[-1] == "[00] PASS" and unstated in [line.strip() for line in lines], defect
        assert any(line.strip().startswith("attention.core ") and line.endswith("[band: attention_bf16_weights]  PASS") for line in lines), defect
        assert any(line.strip().endswith("fi_decode_attention by name (no launch shape is declared for it)") for line in lines), defect

    # the control arms of a run: 26 prompt tokens and 48 new ones, 74 positions -- a full page and ten rows of a second
    attended, call = PagedKvCache.attended, flashinfer_calls.LibraryBlock._call

    def lengths(by):
        return lambda self, position: (attended(self, position)[0], attended(self, position)[1] + by)

    def exchanged(self, position):
        blocks, held = attended(self, position)
        return blocks[:, [1, 0, *range(2, blocks.shape[1])]].contiguous(), held

    def unscaled(self, name, *operands, **passed):
        return call(self, name, *operands, **{field: value for field, value in passed.items() if field != "bmm1_scale"})

    arms = (
        ("the lengths one too long", PagedKvCache, "attended", lengths(1)),
        ("the lengths one too short", PagedKvCache, "attended", lengths(-1)),
        ("the first two columns of the page table exchanged", PagedKvCache, "attended", exchanged),
        ("the scores at the library's default scale", flashinfer_calls.LibraryBlock, "_call", unscaled),
    )
    long_run = ["--stack", "native", "--sequences", "1", "--force-tokens", "48"]
    for name, owner, method, wrong in arms:
        with monkeypatch.context() as arm:
            arm.setattr(owner, method, wrong)
            code, lines = _sample_run(arm, capsys, long_run)
        core = next(line for line in lines if line.strip().startswith("attention.core ") and "calls" in line)
        assert code == 1 and lines[-1] == "[00] FAIL" and core.endswith("[band: attention_bf16_weights]  FAIL"), f"{name}: {core}"
    with monkeypatch.context() as arm:
        arm.setattr(PagedKvCache, "attended", exchanged)
        code, lines = _sample_run(arm, capsys, ["--stack", "native", "--sequences", "1"])
    assert code == 0 and lines[-1] == "[00] PASS", "one page per sequence: both columns name it, and the arm cannot fail"
    code, lines = _sample_run(monkeypatch, capsys, long_run)
    assert code == 0 and lines[-1] == "[00] PASS", "the same run with nothing wrong"


# --------------------------------------------------------------------------- the output head's quantise on its entry


def test_the_row_quantise_entrys_adapter_over_a_stand_in_entry(monkeypatch):
    """``fe.row_quantize_fp4_decode`` over a stand-in entry.  The row's form is the entry's whole config: the constructor is
    handed the device, the row width and the two launch knobs, each knob by name, and a row that leaves one out, names
    another field or names a width that is not the model's hidden width is a wrong table.  A call converts nothing: the
    rows are the tensor it was handed, the scale is one fp32 element -- the fp32 reciprocal of the consumer's static
    scale, one tensor per value -- and the outputs are buffers of the call's own, whole bands of 128 rows for the block
    scales, handed on as the entry wrote them; nothing but the entry is launched.  A quantisation it does not write, an
    entry that declines the row count its row claims, and a package without the entry are each named."""
    from dataclasses import replace

    import cudnn

    from model_samples import census
    from model_samples.blocks import DeviceFacts
    from model_samples.providers import Quant, fe
    from model_samples.qwen_dense import QwenDense, configs
    from model_samples.weights import TiledBlockScaledTensor

    calls, launched = [], []
    entry = _row_quantize_entry(calls, lambda name, shape=None, kind="kernel": launched.append((name, shape, kind)))
    monkeypatch.setattr(cudnn, "RowQuantizeFp4Decode", entry, raising=False)
    monkeypatch.setattr(fe, "_ENTRIES", {})
    model = QwenDense.reduced(2)
    row = configs.PROVIDERS["output_head.quantize"][0]
    stated = DeviceFacts((10, 0), "stated device", 148, 0, 0)
    options = {"device": "cpu", "check_kernels": True}
    block = fe.RowQuantizeFp4DecodeBlock(row, model, stated, options)
    block.check_support(1, 1)
    assert fe._ENTRIES == {} and block.serving() == ("cudnn.RowQuantizeFp4Decode", ()), "on stated facts nothing is built; before a call nothing has served"
    assert block.kernels == (census.Kernel("row_quantize_fp4", "cuDNN FE", "exact", launches=1, shape="entry"),)
    assert block.per_row == (0,) and block.state_args == () and not block.stored_weights and block.entry == "cudnn.RowQuantizeFp4Decode"
    facts = block.launch_facts(1)
    (built,) = fe._ENTRIES.values()
    assert built.built == dict(device="cpu", width=5120, pdl=True, pdl_release=True), "the device, the row width and both knobs: nothing left to the entry"
    assert facts.form == dict(row.form) == {"WIDTH": 5120, "PDL": True, "PDL_RELEASE": True} and facts.shape == _row_quantize_shape(1)
    assert facts.symbols == ("cudnn_kernel_frost_row_quantize_fp4_w5120_wait_release_0",), "the form is in the symbol"
    assert block.launch_facts(1024).shape == ((1024, 1, 1), (320, 1, 1), 0) and len(fe._ENTRIES) == 1, "one instance, whatever the row count"

    generator = torch.Generator().manual_seed(11)
    x = torch.randn((3, 5120), generator=generator).to(torch.bfloat16)
    seeded, other = Quant("nvfp4", 8.0 / (448.0 * 6.0)), Quant("nvfp4", 0.0205544)  # the seeded model's head scale; one of no such round reciprocal
    out = block(x, seeded)
    kind, handed = calls[-1]
    assert kind == "row quantize launch" and handed["x"] is x, "the rows are the tensor it was handed: no copy, no cast"
    for quant in (seeded, other):
        block(x, quant)
        scale = calls[-1][1]["scale"]
        divided = torch.ones(1, dtype=torch.float32) / torch.tensor([quant.scale], dtype=torch.float32)
        assert scale.dtype == torch.float32 and scale.numel() == 1 and scale.view(torch.int32).item() == divided.view(torch.int32).item(), quant
        block(x, quant)
        assert calls[-1][1]["scale"] is scale, "one tensor per value: formed once, never inside a later step"
    assert float(calls[-1][1]["scale"]) * float(torch.tensor(other.scale, dtype=torch.float32)) != 1.0, "the second reciprocal is rounded: not exact in fp32"
    first, second = block(x, seeded), block(x, seeded)
    one, two = calls[-2][1], calls[-1][1]
    assert isinstance(first, TiledBlockScaledTensor) and first.scale == seeded.scale and isinstance(out, TiledBlockScaledTensor)
    assert first.codes is one["codes"] and first.codes.dtype == torch.uint8 and tuple(first.codes.shape) == (3, 2560)
    assert one["scales"].dtype == torch.float8_e4m3fn and tuple(one["scales"].shape) == (128, 320), "one band of 128 rows for three rows"
    assert (
        first.tiled_scales.dtype == torch.uint8
        and first.tiled_scales.data_ptr() == one["scales"].data_ptr()
        and first.tiled_scales.shape == one["scales"].shape
    )
    pointers = {tensor.data_ptr() for tensor in (first.codes, second.codes, one["scales"], two["scales"])}
    own = built.outputs(3)
    assert len(pointers) == 4 and not pointers & {own["codes"].data_ptr(), own["scales"].data_ptr()}, "buffers of each call's own, never the entry's pair"
    block.compile(129, 1)
    assert [call[0] for call in calls[-2:]] == ["row quantize compile", "row quantize launch"] and calls[-2][1] is None
    assert tuple(calls[-1][1]["codes"].shape) == (129, 2560) and tuple(calls[-1][1]["scales"].shape) == (256, 320), "129 rows: two whole bands"
    block.compile(128, 1)
    assert tuple(calls[-1][1]["codes"].shape) == (128, 2560) and tuple(calls[-1][1]["scales"].shape) == (128, 320), "128 rows: one whole band, none behind it"
    symbol = "cudnn_kernel_frost_row_quantize_fp4_w5120_wait_release_0"
    assert launched == [(symbol, _row_quantize_shape(3), "kernel")] * 7 + [
        (symbol, _row_quantize_shape(rows), "kernel") for rows in (129, 128)
    ], "one kernel per call, nothing else"
    assert block.serving() == ("cudnn.RowQuantizeFp4Decode #1", row.form), "the instance that was called, and the row's fields as it states them"

    # what it does not write is named: another quantisation, and none
    with pytest.raises(ValueError, match=r"^cudnn.RowQuantizeFp4Decode writes 4-bit codes under a static global scale; this block is asked for 'fp8'$"):
        block(x, Quant("fp8", 0.0625))
    with pytest.raises(ValueError, match=r"this block is asked for no quantised operand$"):
        block(x, None)
    # a wrong table: a field left to the entry, a field this adapter cannot pass, a row width that is not the model's
    for form, said in (
        (tuple(field for field in row.form if field[0] != "PDL_RELEASE"), "does not name PDL_RELEASE"),
        (tuple(field for field in row.form if field[0] != "PDL"), "does not name PDL"),
        (tuple(field for field in row.form if field[0] != "WIDTH"), "does not name WIDTH"),
        ((*row.form, ("THREADS", 320)), "names THREADS"),
        ((("WIDTH", 4096), *row.form[1:]), "names WIDTH 4096; the rows this block quantises have the model's hidden width, 5120"),
    ):
        with pytest.raises(ValueError) as wrong:
            fe.RowQuantizeFp4DecodeBlock(replace(row, form=form), model, stated, options).check_support(1, 1)
        assert str(wrong.value).startswith("provider row fe.row_quantize_fp4_decode: its form ") and said in str(wrong.value), form
    # on a real device the row's claim is held while the run is planned: an entry that declines the row count is a wrong table
    real = DeviceFacts((10, 0), "stated device", 148, 180 * GIB, 80 * GIB)
    with pytest.raises(
        ValueError, match=r"^the provider table is wrong: the row of fe.row_quantize_fp4_decode claims this run \(1025 rows on compute capability 10.0\)"
    ):
        fe.RowQuantizeFp4DecodeBlock(replace(row, rows=(1, 1025)), model, real, options).check_support(1025, 1)
    # the row of the family's table claims the row counts the entry serves and none beyond them: 1,024 rows of a step are the
    # entry's at one token per sequence and at several, 1,025 and more the reference chain's however the step reaches them
    from model_samples.blocks import select_row

    for stack in configs.TARGET_STACKS:
        for sequences, tokens, key in (
            (1024, 1, row.key),
            (512, 2, row.key),
            (170, 6, row.key),
            (1025, 1, "torch_ref.quantize"),
            (205, 5, "torch_ref.quantize"),
            (171, 6, "torch_ref.quantize"),
        ):
            served = select_row("output_head.quantize", configs.PROVIDERS["output_head.quantize"], real, sequences, tokens, stack, model=model)
            assert served.key == key, (stack, sequences, tokens)
    # a package without the entry: a SKIP that names the first version that carries it
    from types import SimpleNamespace

    class Older(fe.RowQuantizeFp4DecodeBlock):
        def _cudnn(self):
            return SimpleNamespace(__version__="1.33.0")

    with pytest.raises(NotImplementedError, match=r"^needs nvidia-cudnn-frontend with the 4-bit row quantise decode step \(>= 1\.34\.0\); this is 1\.33\.0"):
        Older(row, model, stated, options).check_support(1, 1)


def _e4m3_byte(value: float) -> int:
    """The E4M3 code byte of a non-negative fp64 value, from the format's definition: the nearest of its 127 finite codes,
    a tie to the even byte, saturating at 448."""
    codes = [(m / 8.0) * 2.0**-6 if e == 0 else (1.0 + m / 8.0) * 2.0 ** (e - 7) for e in range(16) for m in range(8)][:127]
    if value >= codes[-1]:
        return 126
    high = next(index for index, code in enumerate(codes) if code >= value)
    low = max(high - 1, 0)
    below, above = value - codes[low], codes[high] - value
    return high if above < below or (above == below and high % 2 == 0) else low


def _e2m1_nibble(value: float, negative: bool) -> int:
    """The E2M1 code of an fp64 magnitude with its sign bit, from the format's definition: the nearest of 0, 0.5, 1, 1.5, 2, 3,
    4, 6, a tie to the even code, saturating at 6, the sign kept on a zero."""
    grid = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    index = 0
    for i in range(7):
        middle = (grid[i] + grid[i + 1]) / 2
        index += (value >= middle) if i % 2 else (value > middle)
    return index + 8 * negative


def _row_quantise_judged(codes, scales, twin, expected):
    """One call judged in the band ``row_quantize_fp4`` as the tie-out judges it: the packed ``codes`` and the row-major block
    ``scales`` against the twin's output ``expected`` and the bracket the twin kept of that call."""
    from model_samples import tie_out

    stats = tie_out.BlockStats("row_quantize_fp4")
    tie_out.BlockTap._judge(stats, (codes, scales), expected.tensors(), [], [], 0, twin)
    return stats


def test_the_twin_and_the_band_of_the_row_quantise_entry():
    """The twin of the head's quantise writes the block-scale bytes of the kernel's rule -- two fp32 multiplications in the
    stated association, the constant by its bits, one nearest-even rounding to E4M3 -- and keeps per element the two
    adjacent codes the reciprocals' bound admits; the band holds every block scale to the twin's byte and every code to
    its bracket.  Inside: the twin's own bytes; a chain in fp32 whose reciprocals are IEEE divisions, some of whose codes
    are the second code of their bracket; and, under a power-of-two scale and power-of-two block scales, the code of
    the exact quotient with ties to the even code -- on rows that hold every rounding tie, where that code is the high
    one of its bracket at three of the seven midpoints.  Outside, each by its own count: a block scale one step off, a
    code two steps off, the reference quantiser's own codes (by the sign of a zero, and by nothing else), a block scale
    formed in another association, the two nibbles of a byte exchanged, the block scales at their row-major places of
    a buffer that is read as the 128 x 4 layout, and a scale byte that is no finite value."""
    from model_samples import tie_out
    from model_samples.blocks import DeviceFacts
    from model_samples.providers import Quant, torch_ref
    from model_samples.qwen_dense import QwenDense, configs
    from model_samples.weights import BLOCK, TiledBlockScaledTensor, quantize_nvfp4, tile_scale_index

    model = QwenDense.reduced(2)
    row = configs.PROVIDERS["output_head.quantize"][-1]
    twin = torch_ref.QuantizeBracket(row, model, DeviceFacts((10, 0), "stated device", 148, 0, 0), {"device": "cpu"})
    quant = Quant("nvfp4", 0.0205544)
    generator = torch.Generator().manual_seed(5)
    x = (torch.randn((6, 5120), generator=generator) * 1.7).to(torch.bfloat16)
    x[0, :16] = 0.0  # a zero block
    x[0, 16:32] = -0.0  # a block of negative zeros
    x[1, :16] = torch.tensor(1.0e-6)  # a block whose scale rounds to the zero code though the block is not zero
    x[1, 16:32] = torch.tensor(-3.0e4)  # a block that saturates
    expected = twin(x, quant)
    low, high = twin.bracket
    reciprocal = float(torch.ones(1, dtype=torch.float32) / torch.tensor([quant.scale], dtype=torch.float32))
    sixth = struct.unpack("<f", struct.pack("<I", 0x3E2AAAAB))[0]
    fp32 = lambda value: struct.unpack("<f", struct.pack("<f", value))[0]  # noqa: E731 -- one rounding to fp32
    bound = 5 * 2.0**-23
    for at in [(0, 0), (0, 1), (1, 0), (1, 1), *((r, b) for r in range(6) for b in (7, 160, 319))]:  # the built blocks, and drawn ones
        values = x[at[0], at[1] * BLOCK : (at[1] + 1) * BLOCK].double().tolist()
        byte = _e4m3_byte(fp32(reciprocal * fp32(max(abs(v) for v in values) * sixth)))
        assert int(expected.block_scales[at]) == byte, f"block {at}: e4m3(global x (largest x fp32(1/6))), written out"
        stored = [(m / 8.0) * 2.0**-6 if e == 0 else (1.0 + m / 8.0) * 2.0 ** (e - 7) for e in range(16) for m in range(8)][byte]
        for j, value in enumerate(values):
            exact = abs(value) * reciprocal / stored if stored else 0.0
            sign = math.copysign(1.0, value) < 0
            element = (at[0], at[1] * BLOCK + j)
            assert (int(low[element]), int(high[element])) == (_e2m1_nibble(exact * (1 - bound), sign), _e2m1_nibble(exact * (1 + bound), sign)), element
    assert expected.block_scales[0, 0] == 0 and expected.block_scales[0, 1] == 0 and expected.block_scales[1, 0] == 0 and expected.block_scales[1, 1] == 126
    assert set(low[0, :16].tolist()) == {0} and set(low[0, 16:32].tolist()) == {8} and set(low[1, :16].tolist()) == {0} and set(low[1, 16:32].tolist()) == {15}
    packed = low[..., 0::2] | (low[..., 1::2] << 4)
    assert torch.equal(expected.codes, packed) and expected.scale == quant.scale, "the twin writes the low code of every bracket"

    # inside: the twin's own bytes; and a chain in fp32 with IEEE divisions for the kernel's two reciprocals
    own = _row_quantise_judged(expected.codes, expected.block_scales, twin, expected)
    assert own.ok and (own.blocks, own.blocks_equal, own.total_bytes, own.second_codes, own.outside) == (6 * 320, 6 * 320, 6 * 5120, 0, 0)
    chain = _row_quantize_entry([])("cpu", 5120, pdl=True, pdl_release=True)
    codes, scales = chain(x, torch.ones(1, dtype=torch.float32) / torch.tensor([quant.scale], dtype=torch.float32))
    kernel_like = TiledBlockScaledTensor(codes, scales.view(torch.uint8), quant.scale)
    inside = _row_quantise_judged(*kernel_like.tensors(), twin, expected)
    assert inside.ok and inside.outside == 0 and inside.blocks_equal == inside.blocks, inside.line()
    assert (
        inside.line()
        == f"1,920 block scales, 0 of them not the twin's byte; 30,720 codes, {inside.second_codes:,} of them the second code of their bracket (printed, not gated); 0 outside"
    )
    assert "%" not in inside.line() and "%" not in tie_out.BANDS["row_quantize_fp4"], "a contract: no share"

    # the exact cells: a power-of-two scale, every block scale a power of two, and elements on every rounding tie of the code grid
    exact_quant = Quant("nvfp4", 2.0**-5)
    midpoints = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
    even = {0.25: 0, 0.75: 2, 1.25: 2, 1.75: 4, 2.5: 4, 3.5: 6, 5.0: 6}  # the even code of the two around each midpoint
    rows = torch.zeros((4, 5120), dtype=torch.float32)
    want = torch.zeros((4, 5120), dtype=torch.uint8)
    ties = []
    for r in range(4):
        for block in range(320):
            unit = 2.0 ** ((block + r) % 9 - 6)  # the step of the block: its scale times the model's scale, a power of two
            first = block * BLOCK
            rows[r, first], want[r, first] = 6.0 * unit, 7  # the block's largest magnitude: its scale is exactly unit x 32
            rows[r, first + 1], want[r, first + 1] = 3.0 * unit, 5  # no tie: one code
            for j, middle in enumerate(midpoints):
                for sign, offset in ((1.0, 2 + 2 * j), (-1.0, 3 + 2 * j)):
                    rows[r, first + offset], want[r, first + offset] = sign * middle * unit, even[middle] + (8 if sign < 0 else 0)
                    ties.append((r, first + offset, middle))
    rows = rows.to(torch.bfloat16)
    assert torch.equal(rows.float(), rows.to(torch.bfloat16).float()) and len(ties) == 4 * 320 * 14, "every built value is a bf16 value"
    tied = twin(rows, exact_quant)
    low, high = twin.bracket
    assert torch.equal(tied.codes, low[..., 0::2] | (low[..., 1::2] << 4)), "on rows that hold ties too, the twin's own bytes are the low codes"
    assert all(int(tied.block_scales[r, b]) == _e4m3_byte(2.0 ** ((b + r) % 9 - 6) * 32.0) for r in range(4) for b in range(320)), "powers of two"
    assert torch.equal(low[:, 0::16], high[:, 0::16]) and torch.equal(low[:, 1::16], high[:, 1::16]), "off a midpoint the bracket holds one code"
    assert bool(((want == low) | (want == high)).all()), "the code of the exact quotient, ties to even, is inside every bracket"
    for r, column, middle in ties:
        assert int(high[r, column]) == int(low[r, column]) + 1, "on a tie the bracket holds the two codes around the midpoint"
        assert (int(want[r, column]) == int(high[r, column])) == (middle in (0.75, 1.75, 3.5)), "the even code is the high one at three midpoints"
    judged = _row_quantise_judged(want[..., 0::2] | (want[..., 1::2] << 4), tied.block_scales, twin, tied)
    assert (
        judged.ok and judged.second_codes == 4 * 320 * 2 * 3
    ), "the exact codes are inside; at three midpoints of seven, both signs, they are not the twin's byte"
    # ... and they are the codes of the fp32 chain with IEEE divisions, whose reciprocals are exact under these scales
    codes, scales = chain(rows, torch.ones(1, dtype=torch.float32) / torch.tensor([exact_quant.scale], dtype=torch.float32))
    exact_chain = TiledBlockScaledTensor(codes, scales.view(torch.uint8), exact_quant.scale)
    assert torch.equal(exact_chain.tensors()[0], want[..., 0::2] | (want[..., 1::2] << 4)) and torch.equal(exact_chain.tensors()[1], tied.block_scales)
    assert (
        _row_quantise_judged(*exact_chain.tensors(), twin, tied).second_codes == judged.second_codes > 0
    ), "some codes of a correct chain are the second of their bracket"

    # outside, each alone and by its own count (judged against the twin of the drawn rows)
    expected = twin(x, quant)
    low, high = twin.bracket

    def outside(codes=None, scales=None):
        return _row_quantise_judged(expected.codes if codes is None else codes, expected.block_scales if scales is None else scales, twin, expected)

    step = expected.block_scales.clone()
    step[2, 5] += 1
    assert (outside(scales=step).outside, outside(scales=step).ok) == (1, False), "a block scale one step off"
    two = low.clone()
    at = next((r, c) for r in range(2, 6) for c in range(5120) if 1 <= int(low[r, c]) <= 4 and int(low[r, c]) == int(high[r, c]))
    two[at] += 2
    assert (outside(codes=two[..., 0::2] | (two[..., 1::2] << 4)).outside, outside(codes=two[..., 0::2] | (two[..., 1::2] << 4)).ok) == (
        1,
        False,
    ), "a code two steps off"
    reference = quantize_nvfp4(x, quant.scale)  # the reference chain's own bytes
    nibbles = torch.stack((reference.codes & 0xF, reference.codes >> 4), dim=-1).flatten(-2)
    off = (nibbles != low) & (nibbles != high)
    chain_judged = outside(codes=reference.codes, scales=reference.block_scales)
    assert torch.equal(reference.block_scales, expected.block_scales) and chain_judged.blocks_equal == chain_judged.blocks, "its block scales are the rule's"
    assert chain_judged.outside == int(off.sum()) > 0 and not chain_judged.ok, "its codes are not inside the entry's contract"
    assert set(nibbles[off].tolist()) == {0} and set(low[off].tolist()) == {8}, "by the sign of a zero, and by nothing else"
    # the block scale in another association -- (global x largest) x fp32(1/6) -- on a (largest, scale) pair found by search
    maxima = torch.arange(0, 0x7F80, dtype=torch.int16).view(torch.bfloat16)  # every finite non-negative bf16 value
    sixth32 = torch.tensor(0x3E2AAAAB, dtype=torch.int32).view(torch.float32)
    found = None
    for bits in (0x41BFFFFF, 0x4219999A, 0x41DB6DB6, 0x42429B39):
        scale = float(torch.ones(1, dtype=torch.float32) / torch.tensor([bits], dtype=torch.int32).view(torch.float32))
        global_scale = torch_ref.nvfp4_global_scale(scale, "cpu")
        other = (global_scale * maxima.float()) * sixth32
        differing = (torch_ref.nvfp4_block_scales(maxima.float(), scale) != other.clamp(max=448.0).to(torch.float8_e4m3fn).view(torch.uint8)).nonzero()
        if len(differing):
            found = (scale, maxima[int(differing[0])], other.clamp(max=448.0).to(torch.float8_e4m3fn).view(torch.uint8)[int(differing[0])])
            break
    assert found is not None, "a block maximum and a scale under which the two associations round to two bytes"
    scale, largest, other_byte = found
    one_block = torch.zeros((1, 5120), dtype=torch.bfloat16)
    one_block[0, 3] = largest
    associated = twin(one_block, Quant("nvfp4", scale))
    wrong = associated.block_scales.clone()
    wrong[0, 0] = other_byte
    assert int(associated.block_scales[0, 0]) != int(other_byte)
    assert _row_quantise_judged(associated.codes, wrong, twin, associated).outside == 1, "the other association's byte is another byte: outside"
    expected = twin(x, quant)
    low, high = twin.bracket
    swapped = expected.codes.clone()
    pair = next((r, c) for r in range(2, 6) for c in range(2560) if abs((int(low[r, 2 * c]) & 7) - (int(low[r, 2 * c + 1]) & 7)) >= 3)
    swapped[pair] = (swapped[pair] >> 4) | ((swapped[pair] & 0xF) << 4)
    assert outside(codes=swapped).outside == 2, "the two nibbles of a byte exchanged: both elements outside"
    buffer = torch.zeros((128, 320), dtype=torch.uint8)
    buffer[:6] = expected.block_scales  # row-major, in a buffer that is read as the 128 x 4 layout
    misplaced = TiledBlockScaledTensor(expected.codes, buffer, quant.scale)
    laid_out = torch.zeros((128, 320), dtype=torch.uint8)
    laid_out.view(-1)[tile_scale_index(6, 320, "cpu")] = expected.block_scales
    assert _row_quantise_judged(*TiledBlockScaledTensor(expected.codes, laid_out, quant.scale).tensors(), twin, expected).ok, "in the layout: inside"
    read = _row_quantise_judged(*misplaced.tensors(), twin, expected)
    assert not read.ok and read.outside == read.blocks - read.blocks_equal > 1000, "another place of the layout is another byte"
    nan = expected.block_scales.clone()
    nan[3, 9] = 0x7F
    assert (outside(scales=nan).finite, outside(scales=nan).ok, outside(scales=nan).outside) == (False, False, 1), "a scale byte that is no finite value"
    with pytest.raises(ValueError, match="the block's twin keeps no bracket"):
        _row_quantise_judged(
            expected.codes, expected.block_scales, torch_ref.Quantize(row, model, DeviceFacts((10, 0), "stated device", 148, 0, 0), {}), expected
        )


def test_the_head_quantise_row_through_a_session_and_what_fails_its_kernel_check(monkeypatch):
    """The head's quantise on its entry through a session, on both target stacks: the final norm, the quantise and the head
    are three calls in wiring order, the quantise is served by its one entry instance in the row's form, its block is
    inside its band, and the census line counts its launch by membership -- the census records no launch shape for it,
    so no sentence of the census names it.  What fails the kernel check: the entry's kernel in a grid of whole bands,
    by ONE sentence, the entry's own; an entry instance in another launch arm, by the form on the objects; a launch of
    PyTorch between the final norm and the quantise, and between the quantise and the library's head -- nothing may
    sit between two exact rows; a fill of the scale buffer in front of the entry, as a kernel of PyTorch's and as a
    record of the fill category; a missing launch; a chain in the entry's place.  A wrong code and a wrong block scale
    fail the block and leave the kernel check alone."""
    from model_samples import census, tie_out

    def run(stack="native", **stand_in):
        session, _, _ = _recorded_run(monkeypatch, sequences=1, stack=stack, **stand_in)
        session.trace(4)
        return session, tie_out.run_checks(session, prompt_tokens=3)

    def sentences(checks):
        return [line.strip() for line in checks.lines if line.strip().startswith("kernels: ")]

    def block_line(checks):
        return next(line for line in checks.lines if line.strip().startswith("output_head.quantize ") and "calls" in line)

    every = "kernels: step 0 (and 3 more of the 4 traced steps): "
    form = (("WIDTH", 64), ("PDL", True), ("PDL_RELEASE", True))
    for stack, compared, loose in (("native", COMPARED_16, 4), ("fe-entries", COMPARED_13, 1)):
        session, checks = run(stack)
        assert checks.ok and checks.kernels == census.PASS, checks.text()
        calls = session.traced.call_log[0]
        at = [call.block for call in calls].index("output_head.quantize")
        assert [(call.block, call.layer) for call in calls[at - 1 : at + 2]] == [("add_norm_quantize", 2), ("output_head.quantize", -1), ("output_head", -1)]
        assert (calls[at].instance, calls[at].form, calls[at].rows) == ("cudnn.RowQuantizeFp4Decode #1", form, 1), "the one instance, in the row's form"
        assert block_line(checks).endswith("; 0 outside  [band: row_quantize_fp4]  PASS"), block_line(checks)
        held = next(line.strip() for line in checks.lines if line.strip().startswith(CENSUS))
        assert held.endswith(f"; {compared}: each in the recorded one; {loose} more by membership alone"), held
        stated = next(line.strip() for line in checks.lines if line.strip().startswith("output_head.quantize: cudnn.RowQuantizeFp4Decode "))
        assert "the entry states grid (1, 1, 1), block (4, 1, 1), shared memory 0 B" in stated and "the census records" not in stated, stated
    quantise, final, head = "output_head.quantize", "add_norm_quantize (layer 2)", "output_head"

    # the entry's kernel in the grid of whole bands, where the entry states one thread block per row: one sentence, the entry's
    _, checks = run(quantize_defect="bands")
    assert (
        not checks.ok
        and checks.kernels == census.FAIL
        and sentences(checks)
        == [
            f"{every}{quantise} launched row_quantize_fp4 with grid (128, 1, 1), block (4, 1, 1), shared memory 0 B; cudnn.RowQuantizeFp4Decode states "
            "grid (1, 1, 1), block (4, 1, 1), shared memory 0 B"
        ]
    ), checks.text()
    assert block_line(checks).endswith("PASS"), "its bytes are the kernel's: only the kernel check sees it"
    # an entry instance built in another launch arm: the form on the objects
    _, checks = run(quantize_serves=("cudnn.RowQuantizeFp4Decode #1", (("WIDTH", 64), ("PDL", True), ("PDL_RELEASE", False))))
    assert (
        not checks.ok
        and checks.kernels == census.FAIL
        and sentences(checks)
        == [
            f"kernels: form: {quantise}: the adapter served the call with the form [WIDTH 64, PDL True, PDL_RELEASE False] (cudnn.RowQuantizeFp4Decode #1); "
            "the row says [WIDTH 64, PDL True, PDL_RELEASE True]"
        ]
    ), checks.text()
    # no fallback row stands between the final norm, the quantise and the library's head: a launch of PyTorch there is undeclared
    undeclared = "a kernel of the fallback family pytorch, which no row there declares; nothing may sit between two exact rows"
    for defect, where in (
        ("extra", f"between {final} and {quantise}"),
        ("behind", f"between {quantise} and {head}"),
        ("fill", f"between {final} and {quantise}"),
    ):
        _, checks = run(quantize_defect=defect)
        (sentence,) = sentences(checks)
        assert (
            not checks.ok and checks.kernels == census.FAIL and sentence.startswith(f"{every}launched ") and sentence.endswith(f" {where}: {undeclared}")
        ), sentence
        assert ("FillFunctor" in sentence) == (defect == "fill")
    # ... and the same fill as a record of the profiler's fill category: a copy between two exact rows
    _, checks = run(quantize_defect="memset")
    assert not checks.ok and sentences(checks) == [
        f"{every}an undeclared copy 'Memset (Device)' between {final} and {quantise}: declared are the one host read of a step and the copies of a fallback row"
    ], checks.text()
    _, checks = run(quantize_defect="missing")
    assert not checks.ok and sentences(checks) == [
        f"{every}{quantise} did not launch row_quantize_fp4: the step has 0 launch(es) of row_quantize_fp4, the plan 1 = 1 call(s) x 1 per call"
    ], checks.text()
    _, checks = run(quantize_defect="chain")
    assert not checks.ok and checks.kernels == census.FAIL and f"{every}{quantise} did not launch row_quantize_fp4" in sentences(checks)[0], checks.text()
    # on fe-entries the head's row is a multiply over a copy: its launches behind the quantise are its row's own, and a fill in front is not
    _, checks = run("fe-entries", quantize_defect="behind")
    assert checks.kernels == census.PASS, "behind the quantise the fallback row of the head declares PyTorch's family"
    _, checks = run("fe-entries", quantize_defect="fill")
    assert checks.kernels == census.FAIL and sentences(checks)[0].endswith(f"between {final} and {quantise}: {undeclared}")
    # a wrong byte fails the block by its own line, and the kernel check holds
    for defect in ("code", "scale", "unsigned"):  # (block scales at their row-major places: the judge's own case; one row of four blocks cannot show it)
        _, checks = run(quantize_defect=defect)
        assert not checks.ok and checks.kernels == census.PASS and block_line(checks).endswith("[band: row_quantize_fp4]  FAIL"), (defect, block_line(checks))
        assert "; 0 outside" not in block_line(checks), defect


def test_the_head_reads_what_the_quantise_wrote_on_both_target_stacks(monkeypatch):
    """What the head's multiply is handed behind the entry.  On ``native`` the library is handed the two tensors of the entry
    as they are -- the same memory, nothing gathered, nothing copied.  On ``fe-entries`` the multiply over a copy reads the
    entry's container and gathers the block scales out of the layout: its result is the result on the same codes with
    row-major scales.  And a load of every stack's plan keeps the forms of the weights it kept: the block multiplies no
    weight."""
    from model_samples.providers import library
    from model_samples.weights import BlockScaledTensor, TiledBlockScaledTensor

    session, _, _ = _recorded_run(monkeypatch, sequences=1, stack="native")
    seen = {}

    def tap(name, layer, block, operands):
        out = block(*operands)
        if name == "output_head.quantize":
            seen["quantised"] = out
        elif name == "output_head":
            seen["head read"], seen["weight"], seen["logits"] = operands[0], operands[1], out
        return out

    before = len(LIBRARY_CALLS)
    session.tap = tap
    ids = session.step(torch.zeros(1, dtype=torch.int32))
    session.read(ids)
    quantised = seen["quantised"]
    assert isinstance(quantised, TiledBlockScaledTensor) and seen["head read"] is quantised
    codes, scales = ROW_QUANTIZE_CALLS[-1][1]["codes"], ROW_QUANTIZE_CALLS[-1][1]["scales"]
    assert quantised.codes is codes and quantised.tiled_scales.data_ptr() == scales.data_ptr(), "what the block returned is what the entry wrote"
    head_calls = [call for call in LIBRARY_CALLS[before:] if call[0] == "mm_fp4" and call[3][1] == session.model.geometry.vocab]
    assert len(head_calls) == 1, "one call of the library for the head"
    handed = session.plan["output_head"]._operand(quantised)
    assert handed[0] is quantised.codes and handed[1] is quantised.tiled_scales and handed[2] == quantised.scale, "the two tensors as they are"
    # the multiply over a copy (the head's row of fe-entries): the gather out of the layout is its own, and gives the row-major result
    row_major = BlockScaledTensor(quantised.codes, quantised.row_major().block_scales, quantised.scale)
    other, _, _ = _recorded_run(monkeypatch, sequences=1, stack="fe-entries")
    weight = other.weights.head
    assert torch.equal(library.matmul_bf16(quantised, weight), library.matmul_bf16(row_major, weight)), "the same codes with row-major scales"
    # the forms a load keeps: the block multiplies no weight, so both loads keep the forms they kept without its row
    assert set(session.weights.stored) == set(session.weights.uncopied) == set(FIVE), "native: every multiplied weight on its stored codes, no copy"
    assert set(other.weights.stored) == set(KEPT) == set(other.weights.uncopied), "fe-entries: the two weights its entries multiply on"
    assert not session.plan["output_head.quantize"].stored_weights and not other.plan["output_head.quantize"].stored_weights


def test_the_reference_stack_writes_the_bytes_of_the_wiring_that_quantised_inside_the_norm():
    """The quantise of the head's operand as a block of its own moves no byte of the reference stack: the chain quantises the
    same bf16 rows with the same function, one call later.  Held against the wiring written out as it was -- the final
    norm handed the head's ``Quant``, the head its first result -- through the session's tap: the ids and the logits of
    a plain step, of the prompt's decode step of a session with draft tokens, and of a verify step are byte for byte the
    same."""
    from model_samples import Session, TextIO
    from model_samples.tie_out import bytes_equal

    def run(old_wiring, draft_tokens=None):
        model = _tiny_model()
        text = TextIO.synthetic(model.geometry.vocab)
        more = {} if draft_tokens is None else {"draft_tokens": draft_tokens}
        session = Session(model, text, sequences=2, stack="reference", device="cpu", capacity=32, **more)
        session.plan_table()
        session.load()
        held = {}

        def tap(name, layer, block, operands):
            if name == "add_norm_quantize" and operands[-1] == "output_head":  # the norm handed the head's Quant, as the wiring was
                held["quantised"], normed, residual = block(*operands[:3], session.weights.head_quant, operands[-1])
                return None, normed, residual
            if name == "output_head.quantize":  # no call of its own: the head reads the norm's first result
                return held.pop("quantised")
            return block(*operands)

        if old_wiring:
            session.tap = tap
        ids = session.prefill([text.encode("ab" + str(s)) for s in range(2)])
        session.read(ids)
        if draft_tokens is None:
            for _ in range(3):
                ids = session.step(ids)
                session.read(ids)
        else:
            drafted = torch.arange(2 * draft_tokens, dtype=torch.int32).view(2, draft_tokens) % 7 + 1
            session.read(session.verify(torch.cat((ids.view(2, 1), drafted), dim=1)))
        record = session.record
        return record.outputs[: record.count].clone(), record.logits[: record.count].clone(), record.count

    for draft_tokens in (None, 3):
        new, old = run(False, draft_tokens), run(True, draft_tokens)
        assert new[2] == old[2] > 0 and bytes_equal(new[0], old[0]) and bytes_equal(new[1], old[1]), f"draft_tokens={draft_tokens}: the ids and the logits"

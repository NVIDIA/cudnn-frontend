# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests of the model samples' helper layer (``samples/models/model_samples``): the provider and regime tables against
the family README, the plan's row selection and its typed declines, the memory sentence on stated device facts and for a
run of several sequence counts, the precision setting of the fallback multiply, the environment check on stated metadata,
token choice, the stop rules, the call counts of a decode step, the import boundary of a dry plan, one decode on the CPU (a
tiny geometry on the reference stack, replayed byte for byte), and the tie-out itself: what the band of the fused step and
of a multiply accept and refuse on stated tensors, the token check against its declared band, and the checks of a whole run
through stand-in providers with deliberate defects.  No GPU, no weights."""

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


@pytest.fixture(scope="module", autouse=True)
def helper_layer():
    """``samples/models`` on ``sys.path`` for this module only: the path and every ``model_samples`` module leave with it."""
    sys.path.insert(0, str(MODELS))
    yield
    sys.path.remove(str(MODELS))
    for name in [n for n in sys.modules if n == "model_samples" or n.startswith("model_samples.")]:
        del sys.modules[name]


def _facts(cc=(10, 0), free=80 * GIB):
    from model_samples.blocks import DeviceFacts

    return DeviceFacts(cc, "stated device", 148, 180 * GIB, free)


def _plan(cc=(10, 0), sequences=1, stack="fe-cublas", overrides=None, layers=2, **kwargs):
    from model_samples.blocks import plan
    from model_samples.qwen_dense import QwenDense

    model = QwenDense.reduced(layers)
    return plan(model.provider_table, model.regimes, _facts(cc), sequences, 1, stack, overrides, model=model, **kwargs)


def test_tables_agree_with_the_family_readme():
    from model_samples.qwen_dense import QwenDense, configs

    readme = (MODELS / "qwen_dense" / "README.md").read_text()
    assert configs.readme_table(QwenDense().calls_text()) in readme, "regenerate the block table of the README from configs.readme_table()"
    assert configs.readme_regimes() in readme, "regenerate the regime table of the README from configs.readme_regimes()"
    assert set(configs.PROVIDERS) == set(configs.BLOCK_TITLES) == set(QwenDense().calls())


def test_call_counts_of_a_decode_step():
    from model_samples.qwen_dense import QwenDense

    assert QwenDense().calls_per_step() == 484
    assert QwenDense.reduced(4).calls_per_step() == 34 and QwenDense.reduced(4).geometry.layer_kinds.count("linear_attention") == 3
    assert QwenDense.reduced(2).calls_per_step() == 20 and QwenDense.reduced(2).geometry.layer_kinds == ("linear_attention", "full_attention")
    assert QwenDense().calls_text()["projection.fp8.in"] == "48 / 16" and QwenDense().calls_text()["add_norm_quantize"] == "129"
    with pytest.raises(ValueError, match="layers"):
        QwenDense.reduced(0)


def test_plan_selects_the_first_matching_row_of_the_stack_and_prints_it():
    plan = _plan()
    assert plan.rows["linear_attention.step"].key == "fe.gdn_decode_fused" and plan["linear_attention.step"].provider == "cuDNN FE"
    assert plan.rows["mlp.down"].key == "cublas.bf16_dequantized" and plan.rows["token_choice"].key == "torch.argmax"
    table = plan.table()
    assert "by provider: cuDNN FE 1 | fallbacks 8 (cuBLAS over copies 4, PyTorch chains 4) | PyTorch 2" in table
    assert "cudnn.GdnDecodeFused" in table and "fallback: bf16 copy of 8-bit weights" in table and "declared regime: launch-bound" in table
    reference = _plan(cc=(8, 0), stack="reference")
    assert {row.key.split(".")[0].split("+")[0] for row in reference.rows.values()} <= {"torch_ref", "torch"}


def test_a_row_no_test_runs_is_printed_as_unverified():
    from dataclasses import replace

    from model_samples.blocks import plan
    from model_samples.qwen_dense import QwenDense

    model = QwenDense.reduced(2)
    table = dict(model.provider_table, token_choice=[replace(model.provider_table["token_choice"][0], ci=False)])
    text = plan(table, model.regimes, _facts(), 1, model=model).table()
    assert "  none; unverified in CI" in text and text.count("unverified in CI") == 2 and "| unverified in CI: 1" in text
    assert "published record" in text.splitlines()[1] and "cc 10.0, 1 row" in text, "the last column is the entry's published record, not this run's"
    assert "unverified in CI" not in _plan().table()


def test_a_row_outside_the_stack_is_never_selected():
    from model_samples.blocks import ProviderRow, select_row

    rows = [
        ProviderRow(("any",), (1, None), (1, None), "torch_ref.linear", "selectable", serves="any"),
        ProviderRow(("cc12.x",), (1, 8), (1, 1), "fe.gdn_decode_fused", "native", serves="rows 1-8"),
        ProviderRow(("cc10.x",), (1, 2), (1, 1), "cublas.bf16_dequantized", "fe-cublas", serves="rows 1-2"),
        ProviderRow(("any",), (1, None), (1, None), "torch_ref.gdn_decode_step", "fe-cublas", serves="any"),
    ]
    assert select_row("b", rows, _facts(), 1, 1, "fe-cublas").key == "cublas.bf16_dequantized"
    assert select_row("b", rows, _facts(), 3, 1, "fe-cublas").key == "torch_ref.gdn_decode_step", "past the first row's sequence range"
    assert select_row("b", rows, _facts((12, 0)), 1, 1, "fe-cublas").key == "torch_ref.gdn_decode_step", "the native row serves cc 12.x, another stack"
    assert select_row("b", rows, _facts(), 1, 1, "fe-cublas", key="torch_ref.linear").key == "torch_ref.linear", "selectable: only through --provider"
    with pytest.raises(NotImplementedError, match="serves compute capability 12.x; this device is 10.0"):
        select_row("b", rows, _facts(), 1, 1, "native")
    with pytest.raises(NotImplementedError, match="no provider row in stack reference"):
        select_row("b", rows, _facts(), 1, 1, "reference")
    with pytest.raises(ValueError, match="not a declared row"):
        select_row("b", rows, _facts(), 1, 1, "fe-cublas", key="torch.argmax")


def test_a_declined_row_is_a_skip_and_a_contract_violation_a_value_error():
    from model_samples.blocks import parse_overrides

    with pytest.raises(NotImplementedError, match=r"cudnn.GdnDecodeFused serves compute capability 10.x or 12.x; this device is 9.0"):
        _plan(cc=(9, 0))
    assert _plan(cc=(12, 0)).rows["linear_attention.step"].key == "fe.gdn_decode_fused"
    with pytest.raises(ValueError, match="unknown stack 'captured'; this family declares: fe-cublas, fe-entries, native, reference"):
        _plan(stack="captured")
    with pytest.raises(ValueError, match="no such block"):
        _plan(overrides={"nothing": "torch.argmax"})
    with pytest.raises(ValueError, match="not a declared row"):
        _plan(overrides={"mlp.down": "fe.gdn_decode_fused"})
    assert _plan(overrides=parse_overrides(["mlp.down=torch_ref.linear"])).rows["mlp.down"].key == "torch_ref.linear"
    with pytest.raises(ValueError, match="BLOCK=KEY"):
        parse_overrides(["mlp.down"])


def test_an_older_package_without_the_decode_step_is_a_skip_that_names_the_fix(monkeypatch):
    import cudnn
    from model_samples.providers import fe

    class Older:
        __version__ = "1.32.3"

        def __getattr__(self, name):
            raise AttributeError(name)

    monkeypatch.setattr(fe.GdnDecodeFusedBlock, "_cudnn", lambda self: Older())
    with pytest.raises(NotImplementedError, match=r">= 1.33.0\); this is 1.32.3: install from the checkout"):
        _plan()
    monkeypatch.undo()
    assert hasattr(cudnn, "GdnDecodeFused")


def test_a_pre_release_of_the_dsl_is_read_by_its_numbers(monkeypatch):
    """The fused step states a floor for the CuTe DSL.  A version is compared by the leading digits of its first three
    fields: a release candidate of a later version serves, one of the floor itself does not, and the SKIP names what it read."""
    import importlib.metadata as metadata

    from model_samples.providers import fe

    assert fe.DSL_FLOOR == (4, 8, 0)
    installed = metadata.version
    for version, serves in (
        ("4.8.0", True),
        ("4.8.1rc1", True),  # was read as 4.8, below the floor: a SKIP on a GPU the sample serves
        ("4.8.1", True),
        ("4.9", True),
        ("4.10.0", True),
        ("5.0.0.dev1", True),
        ("4.8.0.post1", True),
        ("4.8.0+local", True),
        ("4.8.0rc1", False),
        ("4.8.0.dev5", False),
        ("4.7.9", False),
        ("4.7", False),
        ("3.99.99", False),
    ):
        assert fe.serves_floor(version, fe.DSL_FLOOR) is serves, version
        monkeypatch.setattr(metadata, "version", lambda name, version=version: version if name == "nvidia-cutlass-dsl" else installed(name))
        if serves:
            assert _plan().rows["linear_attention.step"].key == "fe.gdn_decode_fused"
        else:
            with pytest.raises(NotImplementedError, match=rf"cudnn.GdnDecodeFused needs nvidia-cutlass-dsl >= 4.8.0; this is {re.escape(version)}$"):
                _plan()

    def absent(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "version", absent)
    with pytest.raises(NotImplementedError, match="needs nvidia-cutlass-dsl >= 4.8.0; this is not installed"):
        _plan()


def test_memory_sentence_on_stated_device_facts():
    from model_samples.blocks import DeviceFacts
    from model_samples.qwen_dense import QwenDense, configs

    model = QwenDense.reduced(2)
    need = model.memory_bytes(sequences=4, capacity=64)
    assert 3.5 * GIB < need < 4.5 * GIB, "1.70 GiB of bf16 copies, the state of 4 sequences and 2 GiB of working memory"
    assert _plan().memory(need).startswith("this stack needs about 3.") and "80.0 GiB are free" in _plan().memory(need)
    reserve = configs.FIRST_LAUNCH_RESERVE["cc12.x"]
    assert _plan(cc=(12, 0)).reserve_bytes() == reserve and _plan().reserve_bytes() == 0
    assert _plan(cc=(12, 0))["linear_attention.step"].first_launch() == {}, "nothing is launched by a plan: no measurement yet"
    assert _plan()["mlp.down"].first_launch() == {}, "a block that measures no launch reports none"
    small = _plan(cc=(12, 0))
    small.facts = _facts((12, 0), free=5 * GIB)
    with pytest.raises(NotImplementedError, match=r"stack fe-cublas needs about 6\.\d GiB on this GPU; 5.0 GiB are free: free memory"):
        small.memory(need)
    small.facts = DeviceFacts((12, 0), "stated device", 170, 4 * GIB, 3 * GIB)  # a GPU that has less in total than the run needs
    with pytest.raises(NotImplementedError, match=r"needs about 6\.\d GiB on this GPU; this GPU has 4.0 GiB in total: the run does not fit on it") as skip:
        small.memory(need)
    assert "free memory on this GPU" not in str(skip.value), "freeing memory cannot help there, and the sentence does not ask for it"


def test_every_count_of_a_run_is_planned_before_the_first_load(monkeypatch):
    """The first session plans the sessions that follow it on its weights, and its memory sentence covers the whole run
    against what is free before anything is loaded: a later count can decline only there."""
    import model_samples
    from model_samples import blocks
    from model_samples.qwen_dense import QwenDense, configs

    model = QwenDense.reduced(2)
    text = model_samples.TextIO.synthetic(model.geometry.vocab)
    free = [8 * GIB]
    monkeypatch.setattr(blocks.DeviceFacts, "probe", classmethod(lambda cls, device: cls((12, 0), "stated device", 170, 0, free[0])))

    def run(*counts):
        first = model_samples.Session(model, text, sequences=counts[0], device="cuda:0", capacity=58)
        later = [model_samples.Session(model, text, sequences=n, device="cuda:0", capacity=58, share=first) for n in counts[1:]]
        return later, first.plan_table(later)

    later, table = run(1, 4)
    assert later[0].plan is not None and later[0].plan.sequences == 4, "the later count is planned with the first one"
    need = model.memory_bytes(1, 58) + model.state_bytes(4, 58) + configs.FIRST_LAUNCH_RESERVE["cc12.x"]
    assert need < 7 * GIB, "the copies and the working memory are counted once for the run, the first launch once"
    assert f"this stack needs about {need / GIB:.1f} GiB on this GPU for 1 and 4 sequences, one count after the other; 8.0 GiB are free" in table
    assert "one count after the other" not in run(4)[1], "one count: the sentence of that count"
    with pytest.raises(NotImplementedError, match="serves rows 1-65,535; this run has 70000 sequences"):
        run(1, 70000)
    free[0] = 6 * GIB
    with pytest.raises(
        NotImplementedError, match=r"needs about 6\.\d GiB on this GPU for 1 and 4 sequences, one count after the other; 6.0 GiB are free: free memory"
    ):
        run(1, 4)


def _sample_main(monkeypatch, capsys, arguments, cc=(10, 0), free=80 * GIB, environment=None):
    """``main()`` of the sample script on stated device facts, up to its first load: no GPU is touched and nothing is loaded.
    Returns ``(exit code, printed text)``; raises ``AssertionError("reached a load")`` when the run gets that far."""
    import importlib.util

    import model_samples
    from model_samples import blocks

    def no_load(session):
        raise AssertionError("reached a load")

    def matching():
        return model_samples.Environment("1.33.0", "1.33.0", True, "0123456789ab", "2.13.0", "4.8.0")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(blocks.DeviceFacts, "probe", classmethod(lambda cls, device: cls(cc, "stated device", 148, 0, free)))
    monkeypatch.setattr(model_samples, "check_environment", environment or matching)
    monkeypatch.setattr(model_samples.Session, "load", no_load)
    paths = list(sys.path)
    spec = importlib.util.spec_from_file_location("model_sample_qwen_dense_00_decode_on_stated_facts", MODELS / "qwen_dense" / "00_decode.py")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        return module.main(arguments), capsys.readouterr().out
    finally:
        sys.path[:] = paths


def test_the_sample_declines_before_its_first_load_and_checks_the_gpu_before_the_environment(monkeypatch, capsys):
    """Every sequence count is planned before anything is loaded, so a SKIP cannot follow a check that ran; and the
    environment check comes after the plan, so a GPU the sample does not serve skips whatever is installed."""
    import model_samples

    def stale():
        raise model_samples.EnvironmentMismatch(
            'cudnn 1.33.0 was imported from an editable checkout, but the installed distribution says 1.29.0: run "pip install -e ." again'
        )

    code, out = _sample_main(monkeypatch, capsys, [], cc=(9, 0), environment=stale)
    assert code == 0 and out.splitlines()[-1] == "[00] SKIP cudnn.GdnDecodeFused serves compute capability 10.x or 12.x; this device is 9.0", out
    code, out = _sample_main(monkeypatch, capsys, [], environment=stale)
    assert code == 1 and out.splitlines()[-1].startswith("[00] ERROR cudnn 1.33.0 was imported from an editable checkout"), out
    code, out = _sample_main(monkeypatch, capsys, ["--sequences", "1,70000"])
    assert (
        code == 0 and out.splitlines()[-1] == "[00] SKIP cudnn.GdnDecodeFused serves rows 1-65,535; this run has 70000 sequences x 1 tokens per sequence"
    ), out
    code, out = _sample_main(monkeypatch, capsys, [], cc=(12, 0), free=6 * GIB)
    skip = (
        "[00] SKIP stack fe-cublas needs about 6.7 GiB on this GPU for 1 and 4 sequences, one count after the other; 6.0 GiB are free: free memory on this GPU"
    )
    assert code == 0 and out.splitlines()[-1].startswith(skip), out
    code, out = _sample_main(monkeypatch, capsys, ["--provider", "mlp.down=nothing"])
    assert code == 1 and out.splitlines()[-1].startswith("[00] ERROR --provider mlp.down=nothing: not a declared row"), out
    with pytest.raises(AssertionError, match="reached a load"):
        _sample_main(monkeypatch, capsys, [], cc=(12, 0), free=8 * GIB)
    out = capsys.readouterr().out
    assert "this stack needs about 6.7 GiB on this GPU for 1 and 4 sequences, one count after the other; 8.0 GiB are free" in out
    assert "[00] cuDNN Frontend 1.33.0 (editable checkout, source hash 0123456789ab; distribution 1.33.0)" in out and "SKIP" not in out


def test_the_fallback_multiply_asks_for_the_fp32_accumulator_and_restores_the_setting(monkeypatch):
    """PyTorch lets cuBLAS reduce the partial sums of a bf16 multiply in bf16 by default; the fallback multiply switches that
    off for its own call and leaves the caller's setting as it found it."""
    from model_samples.providers import library
    from model_samples.weights import Dequantized, quantize_fp8

    flags = torch.backends.cuda.matmul
    seen = []

    def matmul(a, b):
        seen.append(flags.allow_bf16_reduced_precision_reduction)
        return (a.float() @ b.float()).to(torch.bfloat16)

    monkeypatch.setattr(library.torch, "matmul", matmul)
    qx = quantize_fp8(torch.tensor([[1.0, -2.0, 0.5, 4.0]]), 0.5)
    weight = Dequantized(torch.tensor([[2.0, 1.0, 0.0, -1.0], [1.0, 1.0, 1.0, 1.0]], dtype=torch.bfloat16), 0.25)
    found = flags.allow_bf16_reduced_precision_reduction  # set and put back by hand: the flag is no attribute a patch can delete
    try:
        for allowed in (True, False):
            flags.allow_bf16_reduced_precision_reduction = allowed
            out = library.matmul_bf16(qx, weight)
            assert flags.allow_bf16_reduced_precision_reduction is allowed, "the caller's setting is restored"
            assert out.dtype == torch.bfloat16 and out.tolist() == [[-1.0, 0.875]]
    finally:
        flags.allow_bf16_reduced_precision_reduction = found
    assert seen == [False, False], "the partial sums stay in the fp32 accumulator during the call"


def test_check_environment_on_stated_metadata():
    import model_samples

    facts = dict(cudnn_version="1.33.0", distribution_version="1.33.0", editable=True, source_hash="0123456789ab", torch_version="2.13.0", dsl_version="4.8.0")
    env = model_samples.check_environment(sources=(), probe=lambda sources: dict(facts))
    assert env.header() == "cuDNN Frontend 1.33.0 (editable checkout, source hash 0123456789ab; distribution 1.33.0) | CuTe DSL 4.8.0 | PyTorch 2.13.0"
    assert "(wheel, " in model_samples.check_environment(sources=(), probe=lambda sources: dict(facts, editable=False)).header()
    stale = 'cudnn 1.33.0 was imported from an editable checkout, but the installed distribution says 1.29.0: run "pip install -e ." again'
    with pytest.raises(model_samples.EnvironmentMismatch) as error:
        model_samples.check_environment(sources=(), probe=lambda sources: dict(facts, distribution_version="1.29.0"))
    assert str(error.value) == stale
    with pytest.raises(model_samples.EnvironmentMismatch, match="no nvidia-cudnn-frontend distribution is installed"):
        model_samples.check_environment(sources=(), probe=lambda sources: dict(facts, distribution_version=None))
    with pytest.raises(model_samples.EnvironmentMismatch, match="imported from an installed wheel.*says 1.32.3: reinstall"):
        model_samples.check_environment(sources=(), probe=lambda sources: dict(facts, editable=False, distribution_version="1.32.3"))


def test_token_choice_and_stop_rules():
    from model_samples.loop import StopRule
    from model_samples.sampling import greedy, margin

    logits = torch.tensor([[0.5, 4.0, -1.0, 3.5], [2.0, -8.0, 2.25, 0.0]], dtype=torch.bfloat16)
    assert greedy(logits).tolist() == [1, 2] and greedy(logits).dtype == torch.int32
    assert margin(logits).tolist() == [0.5, 0.25]
    rule = StopRule(eos_ids=(7, 9), max_new_tokens=5)
    assert rule.limit == 5 and rule.ends(9) and not rule.ends(8)
    forced = StopRule(eos_ids=(7,), force_tokens=32)
    assert forced.limit == 32 and not forced.ends(7), "a forced run ignores the end token"
    with pytest.raises(ValueError):
        StopRule(eos_ids=(), max_new_tokens=0)


def test_a_dry_plan_imports_nothing_beyond_the_package_pytorch_and_the_standard_library():
    """In a fresh interpreter: after ``cudnn`` (with its decode entry resolved) and PyTorch are loaded, importing the helper layer
    and planning on stated device facts adds only ``model_samples`` and standard-library modules."""
    code = f"""
import sys
import cudnn, torch
cudnn.GdnDecodeFused
import importlib.metadata
before = set(sys.modules)
sys.path.insert(0, {str(MODELS)!r})
import model_samples
from model_samples.blocks import DeviceFacts, plan
from model_samples.qwen_dense import QwenDense
model = QwenDense.reduced(2)
table = plan(model.provider_table, model.regimes, DeviceFacts((10, 0), "stated", 148, 1 << 37, 1 << 37), 1, model=model).table()
model_samples.TextIO.synthetic(model.geometry.vocab).encode_chat([{{"role": "user", "content": "x"}}])
added = sorted({{name.split(".")[0] for name in set(sys.modules) - before}})
print("ADDED", [name for name in added if name not in sys.stdlib_module_names and name != "model_samples"])
print("TABLE", "cudnn.GdnDecodeFused" in table)
"""
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "ADDED []" in done.stdout and "TABLE True" in done.stdout, done.stdout + done.stderr


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
    import model_samples

    model = _tiny_model()
    text = model_samples.TextIO.synthetic(model.geometry.vocab)
    session = model_samples.Session(model, text, sequences=sequences, stack=stack, device="cpu", capacity=32)
    assert "torch_ref.gdn_decode_step" in session.plan_table()
    session.load()
    return session, text


def test_reference_stack_decodes_a_tiny_geometry_on_the_cpu_and_replays_byte_for_byte():
    from model_samples.tie_out import bytes_equal

    session, text = _tiny_session(sequences=2)
    assert session.report().calls_per_step == session.model.calls_per_step() == 20
    prompts = [text.encode_chat([{"role": "user", "content": "a"}, {"role": "assistant", "content": str(s)}], add_generation_prompt=False) for s in range(2)]
    assert prompts[0].dtype == torch.int32 and prompts[0].numel() == prompts[1].numel() and int(prompts[1][-1]) == ord("1")

    def run():
        ids = [session.read(session.prefill(prompts))]
        for _ in range(7):
            ids.append(session.read(session.step(torch.tensor(ids[-1], dtype=torch.int32))))
        return ids, session.record.logits[: session.record.count].clone()

    first, logits = run()
    assert session.state.position == prompts[0].numel() + 7 and session.record.count == session.state.position
    assert len({tuple(step) for step in first}) == len(first), "the seeded head walks through a permutation: no step repeats"
    assert first[0][0] != first[0][1], "two sequences whose prompts end in different tokens choose different ids"
    session.reset(fresh=True)
    again, logits_again = run()
    assert again == first and bytes_equal(logits, logits_again), "a second run from a fresh state is byte-identical"
    stream = text.decoder(prompts[0])
    streamed = "".join(stream.push(step[0]) for step in first) + stream.close()
    assert streamed == text.decode([step[0] for step in first])
    with pytest.raises(ValueError, match="2 sequences; got 1 prompts"):
        session.prefill(prompts[:1])
    with pytest.raises(ValueError, match="one non-zero length"):
        session.prefill([prompts[0], prompts[1][:-1]])


def _judged_step(calls, rows=2, defect=None):
    """``calls`` judged calls of a fused step at the published head layout (48 value heads of 128): the provider side is the
    reference side plus what ``defect(call, codes [rows, heads, dim], state [slots, heads, dim, dim])`` changes."""
    from model_samples import tie_out

    heads, dim = 48, 128
    generator = torch.Generator().manual_seed(0)
    ref_state = torch.randn((rows + 1, heads, dim, dim), generator=generator) * 0.05
    ref_state[0].zero_()  # the null slot
    ref_state[1, 0, 0, 0] = 1.0  # the state's largest value: distances are relative to it
    ref_codes = torch.randint(0, 0x7E, (rows, heads * dim), generator=generator, dtype=torch.uint8)  # finite E4M3 codes, also with bit 0 flipped
    conv = torch.zeros((rows + 1, 3, 16), dtype=torch.bfloat16)
    stats = tie_out.BlockStats("gdn_step")
    for call in range(calls):
        codes, state = ref_codes.clone(), ref_state.clone()
        if defect is not None:
            defect(call, codes.view(rows, heads, dim), state)
        tie_out.BlockTap._judge(stats, (codes,), (ref_codes,), [(state, conv)], [(ref_state, conv)], layer=0)
    return stats


def test_the_fused_step_is_judged_per_sequence_and_value_head():
    """One value head of one sequence is one block of the step's launch grid.  A defect confined to such a cell is a small
    share of a run's bytes and state, so the band also holds every cell to a share of its own."""
    from model_samples import tie_out

    assert _judged_step(4).ok

    def one_head_moves_once(call, codes, state):  # what the rounding of a correct kernel looks like against the fp64 reference
        if call == 5:
            codes[0, 7, :24] ^= 1
            state[1, 7].view(-1)[:300] += 2.0e-3

    tolerated = _judged_step(57, rows=1, defect=one_head_moves_once)
    assert tolerated.ok and "worst cell 99.671%: layer 0, sequence 0, value head 7" in tolerated.line(), tolerated.line()

    def sixteen_columns(call, codes, state):
        codes[-1, 47, -16:] ^= 1

    stats = _judged_step(8, rows=4, defect=sixteen_columns)
    assert stats.equal_bytes > tie_out.GDN_EQUAL_BYTES * stats.total_bytes, "over the run the share of equal bytes holds"
    assert not stats.ok and "worst cell 87.500%: layer 0, sequence 3, value head 47" in stats.line(), stats.line()

    def two_bytes(call, codes, state):
        codes[0, 3, :2] ^= 1

    def one_byte(call, codes, state):
        codes[0, 3, :1] ^= 1

    assert not _judged_step(8, defect=two_bytes).ok, "from two wrong bytes of a head at every step on"
    assert _judged_step(8, defect=one_byte).ok, "one wrong byte of a head at every step is inside the band (stated in the README)"

    def a_tenth_of_one_state(call, codes, state):
        state[-1, 47].view(-1)[:1638] += 1.0e-2  # inside the cap

    stats = _judged_step(8, rows=4, defect=a_tenth_of_one_state)
    assert stats.state_inside > tie_out.GDN_STATE_SHARE * stats.state_elements, "over the run the share of the state holds"
    assert not stats.ok and "(worst cell 90.0024%: layer 0, sequence 3, value head 47)" in stats.line(), stats.line()

    def null_slot(call, codes, state):
        state[0, 0, 0, 0] = 1.0e-6

    stats = _judged_step(2, defect=null_slot)
    assert not stats.ok and "null slot WRITTEN" in stats.line() and stats.state_inside == stats.state_elements, "the null slot is in no share"

    def beyond_the_cap_once(call, codes, state):
        if call == 1:
            state[1, 3, 0, 1] += 0.05

    assert not _judged_step(4, defect=beyond_the_cap_once).ok


def test_a_value_that_is_not_finite_fails_its_block():
    """A NaN is not a tie: Python's ``max(x, nan)`` keeps ``x`` and a comparison with a NaN is false, so the bands ask for
    finite values and judge element by element."""
    from model_samples import tie_out

    def nan_once(call, codes, state):
        if call == 1:
            state[-1, 47, 5, 7] = float("nan")

    stats = _judged_step(4, defect=nan_once)
    assert not stats.ok and "largest distance nan" in stats.line() and "NON-FINITE VALUES" in stats.line(), stats.line()

    def nan_code(call, codes, state):
        codes[0, 0, 0] = 0x7F  # E4M3's NaN

    stats = _judged_step(4, rows=1, defect=nan_code)
    assert stats.worst_cell(stats.cell_bytes)[0] > tie_out.GDN_CELL_BYTES, "one byte of a head per call is inside both byte shares"
    assert not stats.ok and not stats.finite and "NON-FINITE VALUES" in stats.line()

    def multiply(out, ref):
        stats = tie_out.BlockStats("bf16_rounding")
        tie_out.BlockTap._judge(stats, (out,), (ref,), [], [])
        return stats

    ref = torch.tensor([[1.0, -2.0, 0.5, 100.0]], dtype=torch.bfloat16)
    one_unit, two_units, nan, infinite = (ref.clone() for _ in range(4))
    one_unit[0, 0], two_units[0, 0], nan[0, 2], infinite[0, 3] = 1.0 + 2.0**-7, 1.0 + 2.0**-6, float("nan"), float("inf")
    assert multiply(ref.clone(), ref).ok and multiply(one_unit, ref).ok and multiply(one_unit, ref).worst_units == 1.0
    assert not multiply(two_units, ref).ok and multiply(two_units, ref).worst_units == 2.0
    for out in (nan, infinite):
        stats = multiply(out, ref)
        assert not stats.ok and not stats.finite and "NON-FINITE VALUES" in stats.line(), stats.line()
    assert "largest distance nan" in multiply(nan, ref).line()


def test_the_token_check_has_a_declared_band_and_can_fail():
    """The logits band is declared in bf16 units of the run's largest reference logit, not taken from the run's own distance:
    a band that follows the distance forces the tokens to be equal at every decidable step and can never fail."""
    from model_samples import tie_out
    from model_samples.qwen_dense import QwenDense, configs
    from model_samples.sampling import greedy

    assert QwenDense.reduced(2).logits_band_units == configs.LOGITS_BAND_UNITS == 4.0
    logits = torch.tensor([[[64.0, 1.0, 0.0, -3.0]], [[2.0, 70.0, 69.5, 0.0]]], dtype=torch.bfloat16)  # 2 steps of 1 sequence
    ids = torch.stack([greedy(step) for step in logits])
    held, text = tie_out.token_check(logits, logits.clone(), ids, ids.clone(), 2.0)
    # the largest logit is 70: one bf16 unit there is 0.5, the band is 1, and a margin must exceed 2 to decide a step
    assert held and "1 of 1 decidable" in text and "1 near-ties" in text and "[band: 2 units = 1, declared]" in text, text

    def moved(by, step=0, at=1):
        out = logits.clone()
        out[step, 0, at] += by
        return out

    assert tie_out.token_check(moved(1.0), logits, ids, ids, 2.0)[0], "two units: inside"
    held, text = tie_out.token_check(moved(1.5), logits, ids, ids, 2.0)
    assert not held and "1 of 1 decidable" in text and "largest logit distance 1.5 = 3.00 bf16 units" in text, text
    other = ids.clone()
    other[1, 0] = 2
    assert tie_out.token_check(logits, logits, other, ids, 2.0)[0], "another token at a near-tie is counted, not failed"
    other[0, 0] = 3
    held, text = tie_out.token_check(logits, logits, other, ids, 2.0)
    assert not held and "0 of 1 decidable" in text, text
    held, text = tie_out.token_check(moved(float("nan"), step=1, at=3), logits, ids, ids, 2.0)
    assert not held and "NON-FINITE LOGITS" in text, text


def test_a_model_without_a_declared_band_reports_its_token_line_and_fails_only_on_logits_that_are_not_finite():
    """A checkpoint read from local disk declares no logits band (``configs.CHECKPOINT_LOGITS_BAND_UNITS``): no band both
    passes two correct stacks and fails a wrong one there.  Its token line is reported: a finite distance of any size holds
    and is printed, a logit that is not finite fails."""
    from model_samples import tie_out
    from model_samples.qwen_dense import configs
    from model_samples.sampling import greedy

    assert configs.CHECKPOINT_LOGITS_BAND_UNITS is None
    logits = torch.tensor([[[12.0, 1.0, 0.0, -3.0]], [[2.0, 11.0, 10.5, 0.0]]], dtype=torch.bfloat16)  # the largest logit is 12: one unit is 1/16
    ids = torch.stack([greedy(step) for step in logits])

    def moved(by, step=0, at=1):
        out = logits.clone()
        out[step, 0, at] += by
        return out

    held, text = tie_out.token_check(moved(8.75), logits, ids, ids, None)
    assert held and "largest logit distance 8.75 = 140.00 bf16 units of the largest logit" in text, text
    assert "[no band is declared for this model: reported, gated on finite logits]" in text and "decidable" not in text, text
    # why there is no band: 128 units are 8 logits at this largest logit and 16 one binade higher, the distance of two correct stacks follows neither
    higher, apart = 2 * logits, 2 * logits
    apart[0, 0, 1] += 8.75
    assert not tie_out.token_check(moved(8.75), logits, ids, ids, 128.0)[0] and tie_out.token_check(apart, higher, ids, ids, 128.0)[0]
    assert tie_out.token_check(torch.zeros_like(logits), logits, ids, ids, None)[0], "reported: a finite stack is not judged here, the block check judges it"
    for unusable in (moved(float("nan")), moved(float("inf"), step=1)):
        held, text = tie_out.token_check(unusable, logits, ids, ids, None)
        assert not held and "NON-FINITE LOGITS" in text, text
    held, text = tie_out.token_check(logits, moved(float("nan")), ids, ids, None)
    assert not held and "NON-FINITE LOGITS" in text, "a reference that is not finite fails the line too"

    clean = _checked_run(2, band=None)
    line = _line(clean, "tokens")
    assert clean.ok and line.endswith("REPORTED") and "[no band is declared for this model: reported, gated on finite logits]" in line, clean.text()
    assert "greedy tokens equal to the reference stack's" in line and "decidable" not in line

    def nan_in_the_down_projection(product, weight):
        if weight.data.shape == (64, 96):
            product = product.clone()
            product[-1, 3] = float("nan")
        return product

    checks = _checked_run(2, multiply_defect=nan_in_the_down_projection, band=None)
    assert not checks.ok and "NON-FINITE LOGITS" in _line(checks, "tokens") and _line(checks, "tokens").endswith("FAIL"), checks.text()


def test_the_traced_pass_tells_a_failure_of_the_profiler_from_a_failure_of_a_step(monkeypatch):
    """Where the profiler cannot start or stop, the report says why and the run goes on.  An error raised by a step under
    the profiler is the step's: it leaves the traced pass as it is, and is never printed as a missing trace."""
    import torch.profiler
    from model_samples import timing

    events = []

    class Profile:
        def __init__(self, activities=None, fail=""):
            self.fail = fail

        def __enter__(self):
            if self.fail == "start":
                raise RuntimeError("the tracing library could not start\nsecond line")
            events.append("start")
            return self

        def __exit__(self, *exc):
            events.append("stop")
            if self.fail == "stop":
                raise RuntimeError("the tracing library could not stop")
            return False

        def events(self):
            return []

    monkeypatch.setattr(torch.profiler, "supported_activities", lambda: {torch.profiler.ProfilerActivity.CUDA})
    monkeypatch.setattr(torch.profiler, "profile", Profile)
    ran = []
    assert timing.trace_steps(ran.append, 3) == (None, "the profiler recorded no kernel activity on this GPU") and ran == [0, 1, 2]
    assert events == ["start", "stop"]

    def out_of_memory(i):
        raise RuntimeError("CUDA out of memory in step %d" % i)

    del events[:]
    with pytest.raises(RuntimeError, match="CUDA out of memory in step 0"):
        timing.trace_steps(out_of_memory, 3)
    assert events == ["start", "stop"], "the profiler is stopped, and the step's error is the one that is raised"
    for fail in ("start", "stop"):
        monkeypatch.setattr(torch.profiler, "profile", lambda activities=None, fail=fail: Profile(fail=fail))
        trace, reason = timing.trace_steps(lambda i: None, 2)
        assert trace is None and reason == f"the profiler failed (the tracing library could not {fail})", reason
    monkeypatch.setattr(torch.profiler, "supported_activities", set)
    assert timing.trace_steps(out_of_memory, 1) == (None, "this PyTorch build records no kernel activity")


def _checked_run(sequences, step_defect=None, multiply_defect=None, band="declared"):
    """One decode of the tiny geometry on the CPU through the ``fe-cublas`` rows and the sample's checks of it.  Every provider
    of that stack is replaced by a stand-in that runs the reference's own arithmetic under another type -- the tie-out then
    judges it with the row's band, and a clean run is exact on any host -- plus a deliberate defect.  ``band`` replaces the
    model's declared logits band (``None``: a model that declares none)."""
    import model_samples
    from model_samples import blocks, tie_out
    from model_samples.providers import factories, torch_ref

    class Step(torch_ref.GdnDecodeStep):
        def __call__(self, x, z, hidden, slot_ids, params, slots):
            out = super().__call__(x, z, hidden, slot_ids, params, slots)
            if step_defect is not None:
                step_defect(out.codes.view(x.shape[0], params.value_heads, params.head_dim), slot_ids, slots)
            return out

    class Multiply(torch_ref.Linear):
        def __call__(self, qx, weight):
            product = super().__call__(qx, weight)
            return product if multiply_defect is None else multiply_defect(product, weight)

    class GateUp(torch_ref.GateUp):
        pass

    stand_ins = {"fe.gdn_decode_fused": Step, "cublas.bf16_dequantized": Multiply, "cublas.bf16_dequantized+torch_ref.silu_mul_quantize": GateUp}
    model = _tiny_model()
    if band != "declared":
        model.logits_band_units = band
    text = model_samples.TextIO.synthetic(model.geometry.vocab)
    session = model_samples.Session(model, text, sequences=sequences, stack="fe-cublas", device="cpu", capacity=16)
    facts = blocks.DeviceFacts((10, 0), "stated device", 148, 0, 0)  # a class on which the fused step's batch invariance is a gate
    options = {"reference_precision": "fp64", "device": "cpu"}
    session.plan = blocks.plan(model.provider_table, model.regimes, facts, sequences, model=model, options=options, factories=dict(factories(), **stand_ins))
    session.load()
    prompts = [text.encode("ab" + str(s)) for s in range(sequences)]
    ids = session.prefill(prompts)
    session.read(ids)
    for _ in range(4):
        ids = session.step(ids)
        session.read(ids)
    return tie_out.run_checks(session, prompt_tokens=3)


def _line(checks, start):
    return next(line for line in checks.lines if line.strip().startswith(start))


def _sample_script(monkeypatch):
    """The sample script as a module, for its own functions; what loading it changes in the process leaves with the test."""
    import importlib.util
    import os

    monkeypatch.setattr(sys, "path", list(sys.path))  # the script puts samples/models on the path
    monkeypatch.setenv("KINETO_LOG_LEVEL", os.environ.get("KINETO_LOG_LEVEL", "6"))  # and sets a default for the profiler's log level
    spec = importlib.util.spec_from_file_location("model_sample_qwen_dense_00_decode_functions", MODELS / "qwen_dense" / "00_decode.py")
    sample = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sample)
    return sample


def test_the_checks_of_a_run_hold_on_exact_stand_ins_and_fail_on_a_nan():
    from model_samples import tie_out

    clean = _checked_run(2)
    assert clean.ok, clean.text()
    assert "6 of 6 provider blocks inside their bands" in clean.text() and "linear_attention.step equal (gated)" in clean.text()
    assert _line(clean, "linear_attention.step").endswith("[band: gdn_step]  PASS") and _line(clean, "mlp.down").endswith("[band: bf16_rounding]  PASS")
    bands = [line.strip() for line in clean.lines if line.strip().startswith("band ")]
    assert bands == [f"band {name}: {tie_out.BANDS[name]}" for name in ("bf16_rounding", "gdn_step")], "each band's text once, on its own line"
    assert max(len(line) for line in clean.lines) < 420, "no line of the checks carries a band's whole text beside its measurement"
    assert "largest logit distance 0 = 0.00 bf16 units" in _line(clean, "tokens") and _line(clean, "tokens").endswith("PASS")

    def nan_in_the_down_projection(product, weight):  # one element of one multiply, at every call
        if weight.data.shape == (64, 96):
            product = product.clone()
            product[-1, 3] = float("nan")
        return product

    checks = _checked_run(2, multiply_defect=nan_in_the_down_projection)
    assert not checks.ok, checks.text()
    assert "NON-FINITE VALUES" in _line(checks, "mlp.down") and _line(checks, "mlp.down").endswith("FAIL")
    assert "NON-FINITE LOGITS" in _line(checks, "tokens") and _line(checks, "tokens").endswith("FAIL")
    assert _line(checks, "replay").endswith("PASS"), "the bytes of a NaN replay: only a finite check sees it"

    def nan_in_one_head_of_the_state(codes, slot_ids, slots):
        slots.state[int(slot_ids[-1]), 1, 2, 3] = float("nan")

    checks = _checked_run(2, step_defect=nan_in_one_head_of_the_state)
    assert not checks.ok and "NON-FINITE VALUES" in _line(checks, "linear_attention.step"), checks.text()


def test_a_run_that_judged_no_provider_block_is_reported_and_never_a_pass(monkeypatch):
    """``--provider`` can put every judged block on its reference row.  Such a run compares the reference with itself: its
    blocks line is ``REPORTED`` and its last line ``RAN``, as for ``--stack reference``.  ``PASS`` is kept for a run in
    which provider blocks were judged and held."""
    from model_samples import Session, TextIO, blocks, tie_out

    reference_rows = blocks.parse_overrides(
        [
            "projection.fp8.in=torch_ref.linear",
            "linear_attention.step=torch_ref.gdn_decode_step",
            "projection.fp8.out=torch_ref.linear",
            "mlp.gate_up=torch_ref.linear+torch_ref.silu_mul_quantize",
            "mlp.down=torch_ref.linear",
            "output_head=torch_ref.linear",
        ]
    )
    model = _tiny_model()
    text = TextIO.synthetic(model.geometry.vocab)
    for sequences, declares_a_band in ((1, True), (2, False)):  # the seeded model's logits band, and a checkpoint's reported token line
        if not declares_a_band:
            model.logits_band_units = None
        session = Session(model, text, sequences=sequences, stack="fe-cublas", device="cpu", capacity=16, overrides=reference_rows)
        session.plan_table()
        session.load()
        ids = session.prefill([text.encode("ab" + str(s)) for s in range(sequences)])
        for _ in range(3):
            ids = session.step(ids)
        checks = tie_out.run_checks(session, prompt_tokens=3)
        line = _line(checks, "blocks")
        assert checks.ok and checks.judged == 0, checks.text()
        assert "no provider block to judge" in line and line.endswith("REPORTED") and "PASS" not in line, line
    assert _checked_run(2).judged == 6, "the stack's own rows: six provider blocks are judged"

    sample = _sample_script(monkeypatch)
    assert sample._verdict("fe-cublas", True, 6) == "PASS"
    assert sample._verdict("fe-cublas", True, 0) == "RAN (no provider block was judged)", "nothing was compared: not a pass"
    assert sample._verdict("fe-cublas", False, 0) == sample._verdict("fe-cublas", False, 6) == "FAIL", "a failed replay or token line fails either way"
    assert sample._verdict("reference", True, 0) == "RAN reference stack (no provider checked)"


def test_the_samples_loop_says_what_comes_next_before_the_phases_that_print_nothing(monkeypatch, capsys):
    """The sample's own loop function on a CPU session of the tiny geometry: one line before the prompt and the decode loop,
    one before the checks; the report and the text stand between them, so a reader of a long run knows what it waits for."""
    from model_samples import StopRule, tie_out

    sample = _sample_script(monkeypatch)
    session, text = _tiny_session(sequences=2)
    prompts = [text.encode("ab" + str(s)) for s in range(2)]
    capsys.readouterr()
    forced = StopRule(text.eos_ids, max_new_tokens=256, force_tokens=5)
    ok, judged, produced = sample._decode(session, text, forced, prompts, tie_out, check=True, checked_steps=None)
    out = capsys.readouterr().out.splitlines()
    assert ok and judged == 0 and [len(ids) for ids in produced] == [5, 5], "the reference stack: nothing to judge; 5 forced tokens per sequence"
    first = (
        "[00] decoding: 3 prompt tokens through the decode step, then 5 new tokens and a traced pass; the report and the text follow, the text is not streamed"
    )
    assert out[0] == first, out[0]
    at = out.index("[00] checks on every step against the fp64-accumulated reference (replay, every provider block, the tokens)")
    assert any(line.startswith("first token ") for line in out[1:at]) and out[at - 1].startswith("text out "), "the report and the text come before the checks"
    assert out[at + 1] == "checks (2 sequences, 7 steps: 3 prompt + 4 decode)", out[at + 1]
    session.reset(fresh=True)
    session.new_report()
    open_end = StopRule(text.eos_ids, max_new_tokens=4)
    ok, judged, produced = sample._decode(session, text, open_end, prompts, tie_out, check=False, checked_steps=16)
    out = capsys.readouterr().out
    assert ok and judged == 0 and max(len(ids) for ids in produced) == 4
    assert out.startswith("[00] decoding: 3 prompt tokens through the decode step, then up to 4 new tokens and a traced pass; ") and "[00] checks on" not in out


def test_the_samples_flags_say_what_they_apply_to_and_none_is_ignored_in_silence(monkeypatch, capsys):
    """The defaults of the two kinds of run, the flags that belong to one kind only, and the help text: it names the
    requirement file, what a checkpoint run needs and where the rest is written."""
    sample = _sample_script(monkeypatch)
    seeded = sample._arguments([])
    assert (seeded.model, seeded.sequences, seeded.layers, seeded.force_tokens, seeded.max_new_tokens, seeded.stack) == (None, [1, 4], 2, 32, 256, "fe-cublas")
    real = sample._arguments(["--model", "some-directory"])
    assert (real.sequences, real.force_tokens, real.max_new_tokens, real.reference_precision) == ([1], None, 256, "fp64")
    assert sample._arguments(["--model", "d", "--max-new-tokens", "8"]).max_new_tokens == 8
    assert sample._arguments(["--force-tokens", "8", "--sequences", "2,3"]).force_tokens == 8
    for wrong, sentence in (
        (["--max-new-tokens", "8"], "--max-new-tokens is for --model; the seeded run decodes --force-tokens N (default 32)"),
        (["--model", "d", "--force-tokens", "4", "--max-new-tokens", "8"], "--force-tokens N decodes exactly N tokens; --max-new-tokens does not apply"),
        (["--prompt", "hi"], "--prompt and --messages need --model"),
        (["--model", "d", "--layers", "4"], "--layers is for the reduced model"),
        (["--sequences", "0"], "take positive integers"),
        (["--sequences", "two"], "--sequences takes N or N,N,..."),
    ):
        with pytest.raises(SystemExit) as stop:
            sample._arguments(wrong)
        assert stop.value.code == 2 and sentence in capsys.readouterr().err, wrong
    with pytest.raises(SystemExit) as stop:
        sample._arguments(["--help"])
    shown = " ".join(capsys.readouterr().out.split())  # argparse wraps its lines by the terminal's width
    assert stop.value.code == 0
    for part in (
        "pip install -r samples/models/requirements.txt",
        "The checkpoint is your own download (nothing is fetched)",
        "compute capability 10.x or 12.x",
        "about 57 GiB of free GPU memory",
        "samples/models/qwen_dense/README.md",
        "--sequences N[,N...]",
        "--layers N",
        '(default "Why is the sky blue?")',
        "fe-cublas (default): cuDNN Frontend + cuBLAS + PyTorch",
        "not with --model",
        "an unknown block or key prints the declared ones",
    ):
        assert part in shown, part


def test_batch_invariance_covers_every_sequence_of_the_batch():
    """The sequence that runs alone rotates with the step, so a row other than the first that reads its neighbour is seen."""

    def last_row_reads_its_neighbour(codes, slot_ids, slots):
        if codes.shape[0] > 1:
            codes[-1, 0, 0] ^= 0x01 | (codes[-2, 0, 0] & 0x02)

    checks = _checked_run(3, step_defect=last_row_reads_its_neighbour)
    line = _line(checks, "batch invariance")
    assert not checks.ok and "one sequence alone vs among 3 (step t: sequence t mod 3)" in line, checks.text()
    assert "linear_attention.step DIFFERS (gated)" in line and "linear_attention.step (2 of 7 calls)" in line and line.endswith("FAIL"), line


def test_the_traced_pass_is_read_from_kernel_activity_records():
    """A step of the loop ends in its one host read, which the records show as a device-to-host copy: the records up to
    and including it are one step.  GPU busy is the union of the intervals, idle the rest of the step's span."""
    from model_samples.timing import Report, Trace, step_facts

    records = [
        ("kernel_a", 0.0, 2.0),
        ("kernel_b", 1.0, 3.0),  # overlaps kernel_a: the union counts the overlap once
        ("Memcpy HtoD (Pageable -> Device)", 5.0, 5.5),
        ("kernel_c", 5.25, 5.4),  # inside the copy
        ("Memcpy DtoH (Device -> Pageable)", 9.0, 10.0),  # the host read: the step ends here
        ("Memset (Device)", 20.0, 20.5),
        ("kernel_a", 21.0, 22.0),
        ("Memcpy DtoH (Device -> Pinned)", 22.0, 22.25),
        ("kernel_a", 30.0, 31.0),  # after the last host read: belongs to no step
    ]
    assert step_facts(list(reversed(records))) == [
        (4.5, 5.5, 3, 2),
        (1.75, 0.5, 1, 2),
    ], "(GPU busy, idle inside the step, kernels, copies) per step, in start order"
    assert step_facts([("kernel_a", 0.0, 1.0)]) == [] and step_facts([]) == []

    report = Report("qwen_dense", reduced=False)
    report.prompt_tokens, report.calls_per_step = 58, 484
    for t in range(4):
        report.clock.launch.append(0.1 * t)
        report.clock.ids.append(0.1 * t + 0.09)
    report.trace = Trace(steps=32, busy=40.0, idle=55.0, kernels=11750, copies=1016, period=125.0)
    report.bytes = {"weights": 17_610_000_000, "copies": 51_250_000_000, "state": 310_000_000}
    text = report.text()
    assert "steps (4)            period 100.000 ms median / 100.000 mean | in the step 90.000 ms | turnaround 10.000 ms | 484 block calls per step" in text
    traced = "traced pass (32)      period under the profiler 125.000 ms (+25 % against the untraced period) | GPU busy 40.000 ms | "
    traced += "idle inside the traced step 55.000 ms | 11750 kernel launches + 1016 copies per step | "
    assert traced + "host-exposed 60.000 ms = untraced period - GPU busy (60% of it)" in text, "busy + idle are parts of the traced period, which comes first"
    launch_bound = "note                  host-exposed 60% of the period: the launch path from Python bounds this step, not the GPU; no floor and no"
    assert launch_bound in text, "what bounds the step is said from the measured share"
    report.trace = Trace(steps=32, busy=83.0, idle=5.0, kernels=599, copies=60, period=101.0)
    assert "note                  GPU busy 83% of the period: the GPU work of this stack bounds this step, not the launch path; no floor" in report.text()
    assert "rate                  10.3 tokens per second per sequence = 4 tokens / 0.390 s of the decode loop" in text
    assert "bytes per step        model 17.92 GB (weights 17.61 in their stored formats, state and cache 0.31); this stack moves 51.56 GB (2.88x)" in text
    assert "first token           prefill route: decode step, 58 tokens x one step; not a time to first token of record" in text
    report.trace, report.trace_missing = None, "the profiler recorded no kernel activity on this GPU"
    assert "traced pass           not available: the profiler recorded no kernel activity on this GPU" in report.text()
    assert "note                  what bounds this step was not measured (no traced pass); no floor and no speed-of-light figure is printed" in report.text()
    reduced = Report("qwen_dense", reduced=True)
    reduced.bytes = report.bytes
    assert "rate                  not printed for a reduced model" in reduced.text() and "bytes per step        reduced model: model 17.92 GB" in reduced.text()


def test_a_session_traces_only_what_it_recorded_and_keeps_its_run(monkeypatch):
    """The traced pass replays the recorded inputs from a zero state on a record of its own and puts the run's state and
    record back, also when a step under the profiler raises.  A stand-in for the profiler runs the steps here, on the CPU."""
    from model_samples import loop
    from model_samples.timing import Trace

    session, text = _tiny_session(sequences=1)
    session.trace()
    assert session.report().trace is None and session.report().trace_missing == "needs a recorded step on a CUDA device", "nothing recorded, and no GPU here"
    assert "traced pass           not available: needs a recorded step on a CUDA device" in session.report().text()
    ids = session.prefill([text.encode("abc")])
    state, record, position, calls = session.state, session.record, session.state.position, session._calls
    session.trace(2)
    assert session.report().trace is None and session.report().trace_missing == "needs a recorded step on a CUDA device", "three steps recorded, no GPU"
    assert session.state is state and session.record is record and session.state.position == position == 3 and session.record.count == 3

    seen = []

    def profiled(step, count, device=None):
        seen.append((session.state is state, session.record is record, session.state.position, session.record.count))
        for i in range(count):
            step(i)
        seen.append((session.state.position, session.record.count, session.record.inputs.shape[0]))
        return Trace(count, 1.0, 2.0, 3, 4, 5.0), ""

    monkeypatch.setattr(loop, "trace_steps", profiled)
    session.trace(2)
    assert seen == [(False, False, 0, 0), (2, 2, 0)], "the traced steps start from a zero state at position 0, on a record that keeps nothing"
    assert session.state is state and session.record is record and session.state.position == 3 and session.record.count == 3, "the run is put back"
    assert session._calls == calls and session.report().trace.kernels == 3 and session.report().trace_missing == ""

    def failing(step, count, device=None):
        step(0)
        raise RuntimeError("a step failed under the profiler")

    monkeypatch.setattr(loop, "trace_steps", failing)
    with pytest.raises(RuntimeError, match="a step failed under the profiler"):
        session.trace(2)
    assert session.state is state and session.record is record and session.state.position == 3 and session.record.count == 3, "also when a step raises"
    assert session._calls == calls
    assert session.report().bytes == session.model.bytes_per_step(1, 3), "the report carries the bytes of a step at the position the run reached"
    session.new_report()
    assert session.report().trace_missing == "" and session.report().calls_per_step == 20 and not session.report().text_path, "a fresh report per prompt set"
    assert ids.shape == (1,)


def test_the_published_model_states_its_bytes_and_its_memory_need(monkeypatch):
    """The published geometry by arithmetic on its tensor list: what a step reads in the stored formats, what this stack
    moves instead, and the memory sentence of a run with every weight held as a bf16 copy -- on a GPU that has less, a
    SKIP before anything is loaded."""
    from model_samples import Session, TextIO, blocks
    from model_samples.qwen_dense import QwenDense

    model = QwenDense()
    step = model.bytes_per_step(sequences=1, positions=343)
    assert 17.60e9 < step["weights"] < 17.62e9, "8-bit projections, a 4-bit MLP and a 4-bit output head, as stored"
    assert 51.2e9 < step["copies"] < 51.3e9 and 2.90 < step["copies"] / step["weights"] < 2.92, "the bf16 copies: 2 bytes per weight element"
    assert 0.30e9 < step["state"] < 0.36e9, "48 recurrent states read and written, 16 key/value caches read"
    assert model.bytes_per_step(4, 343)["state"] == 4 * step["state"] and model.bytes_per_step(4, 343)["copies"] - step["copies"] == 3 * 2 * 5120
    need = model.memory_bytes(sequences=1, capacity=343)
    copies = model.copy_bytes()
    assert 50.0 * GIB < copies < 50.2 * GIB, "the bf16 copies incl. the embedding table"
    assert need == copies + model.state_bytes(1, 343) + copies // 8, "working memory: an eighth of the copies where that is more than 2 GiB"
    assert 56.5 * GIB < need < 57 * GIB, "above the 55.4 GiB a run with this checkpoint took from the device: a GPU with less is a SKIP, not an exception"
    small = QwenDense.reduced(2)
    assert small.memory_bytes(4, 64) == small.copy_bytes() + small.state_bytes(4, 64) + 2 * GIB, "the default run: 2 GiB, as before"
    text = TextIO.synthetic(model.geometry.vocab)
    free = [31 * GIB]
    monkeypatch.setattr(blocks.DeviceFacts, "probe", classmethod(lambda cls, device: cls((12, 0), "stated device", 170, 0, free[0])))
    with pytest.raises(NotImplementedError, match=r"stack fe-cublas needs about 59\.\d GiB on this GPU; 31.0 GiB are free: free memory on this GPU"):
        Session(model, text, sequences=1, device="cuda:0", capacity=343).plan_table()
    card = blocks.plan(model.provider_table, model.regimes, blocks.DeviceFacts((12, 0), "stated device", 170, int(31.84 * GIB), 31 * GIB), 1, model=model)
    with pytest.raises(NotImplementedError, match=r"needs about 59\.\d GiB on this GPU; this GPU has 31.8 GiB in total: the run does not fit on it"):
        card.memory(need)  # the same card with its total stated: the sentence says that no freeing helps
    free[0] = 80 * GIB
    table = Session(model, text, sequences=1, device="cuda:0", capacity=343).plan_table()
    assert "this stack needs about 59." in table and "80.0 GiB are free" in table and "add_norm_quantize        129" in table and "48 / 16" in table


def test_the_checks_of_a_long_run_cover_its_first_steps_and_print_a_second_reference():
    from model_samples import Session, tie_out

    session, text = _tiny_session(sequences=1)
    ids = session.prefill([text.encode("abcd")])
    for _ in range(4):
        ids = session.step(ids)
    assert session.record.count == 8
    checks = tie_out.run_checks(session, prompt_tokens=4, max_steps=3, second_reference=True)
    assert checks.ok and checks.lines[0] == "checks (1 sequence, the first 3 steps: 3 prompt + 0 decode)", checks.text()
    tokens = _line(checks, "tokens").strip()
    assert tokens.startswith("tokens            3 of 3 greedy tokens equal to the reference stack's; ") and tokens.endswith("PASS")
    assert "for scale, the reference stack with ieee-fp32 accumulation is " in tokens and "agrees with it on " in tokens and "(printed, not gated)" in tokens
    bounded = Session(session.model, text, sequences=1, stack="reference", device="cpu", capacity=32, record_steps=5)
    bounded.load()
    assert bounded.record.inputs.shape[0] == 5 and bounded.record.logits.shape[0] == 5, "a long run keeps the ids and logits of its first steps only"
    bounded.prefill([text.encode("abcdefg")])
    assert bounded.record.count == 7 and tie_out.run_checks(bounded, prompt_tokens=7).lines[0] == "checks (1 sequence, the first 5 steps: 5 prompt + 0 decode)"

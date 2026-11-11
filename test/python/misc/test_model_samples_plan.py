# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests of the model samples' helper layer (``samples/models/model_samples``): the provider and regime tables against
the family README, the plan's row selection and its typed declines, the memory sentence on stated device facts, the
environment check on stated metadata, token choice, the stop rules, the call counts of a decode step, the import boundary of a
dry plan, and one decode on the CPU: a tiny geometry on the reference stack, replayed byte for byte.  No GPU, no weights."""

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
    assert "cudnn.GdnDecodeFused" in table and "fallback: bf16 copy of 8-bit weights" in table and "regime: launch-bound" in table
    reference = _plan(cc=(8, 0), stack="reference")
    assert {row.key.split(".")[0].split("+")[0] for row in reference.rows.values()} <= {"torch_ref", "torch"}


def test_a_row_no_test_runs_is_printed_as_unverified():
    from dataclasses import replace

    from model_samples.blocks import plan
    from model_samples.qwen_dense import QwenDense

    model = QwenDense.reduced(2)
    table = dict(model.provider_table, token_choice=[replace(model.provider_table["token_choice"][0], ci=False)])
    text = plan(table, model.regimes, _facts(), 1, model=model).table()
    assert "served, not measured here; unverified in CI" in text and text.count("unverified in CI") == 2 and "| unverified in CI: 1" in text
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
    with pytest.raises(ValueError, match="unknown stack"):
        _plan(stack="native")
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


def test_memory_sentence_on_stated_device_facts():
    from model_samples.qwen_dense import QwenDense, configs

    model = QwenDense.reduced(2)
    need = model.memory_bytes(sequences=4, capacity=64)
    assert 3.5 * GIB < need < 4.5 * GIB, "1.70 GiB of bf16 copies, the state of 4 sequences and 2 GiB of working memory"
    assert _plan().memory(need).startswith("this stack needs about 3.") and "80.0 GB are free" in _plan().memory(need)
    reserve = configs.FIRST_LAUNCH_RESERVE["cc12.x"]
    assert _plan(cc=(12, 0)).reserve_bytes() == reserve and _plan().reserve_bytes() == 0
    small = _plan(cc=(12, 0))
    small.facts = _facts((12, 0), free=5 * GIB)
    with pytest.raises(NotImplementedError, match=r"stack fe-cublas needs about 6\.\d GB on this GPU; 5.0 GB are free: free memory"):
        small.memory(need)


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


def _tiny_session(sequences, stack="reference"):
    import model_samples
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
    text = model_samples.TextIO.synthetic(geometry.vocab)
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
    assert "".join(stream.push(step[0]) for step in first) + stream.close() == text.decode([step[0] for step in first])
    with pytest.raises(ValueError, match="2 sequences; got 1 prompts"):
        session.prefill(prompts[:1])
    with pytest.raises(ValueError, match="one non-zero length"):
        session.prefill([prompts[0], prompts[1][:-1]])

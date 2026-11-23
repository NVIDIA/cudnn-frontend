# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests of the model samples' helper layer (``samples/models/model_samples``): the provider and regime tables against
the family README, the plan's row selection and its typed declines, the memory sentence on stated device facts and for a
run of several sequence counts, the precision setting of the fallback multiply, the environment check on stated metadata,
token choice, the stop rules, the call counts of a decode step, the import boundary of a dry plan, one decode on the CPU (a
tiny geometry on the reference stack, replayed byte for byte), and the tie-out itself: what the band of the fused step and
of a multiply accept and refuse on stated tensors, the token check against its declared band, and the checks of a whole run
through stand-in providers with deliberate defects.  No GPU, no weights."""

import contextlib
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


def test_the_drafters_plan_its_table_and_its_keys():
    """The drafter of a checkpoint as a model of a plan: its provider table resolved on stated facts in every stack, for any
    GPU class and any row count; every block a reference chain but the embedding lookup; the five provider keys its pass
    added to the closed list; its call sites; and the two tables of the family README rendered from the data."""
    from model_samples import providers
    from model_samples.blocks import DeviceFacts, plan
    from model_samples.providers import torch_ref
    from model_samples.qwen_dense import QwenDense, QwenDenseDrafter, configs
    from model_samples.qwen_dense.drafter import BODY, HEAD, MULTIPLIES

    drafter = QwenDenseDrafter.seeded(QwenDense.reduced())
    table = configs.DRAFTER_PROVIDERS
    assert drafter.provider_table is table and drafter.regimes is configs.REGIMES and drafter.table_source == "model_samples/qwen_dense/configs.py"
    assert not hasattr(drafter, "target_stacks"), "the drafter declares no target kernel set: no census section holds a draft pass"
    assert drafter.geometry.layer_kinds == ("full_attention",) and drafter.norm_eps == QwenDense().norm_eps == 1.0e-6

    # one pass: 18 block calls, 15 of the body and 3 of the head; the multiply at five sites, the norm + quantise at two
    wiring, sites = drafter.wiring(), drafter.sites()
    assert len(wiring) == len(sites) == drafter.calls_per_pass() == len(BODY) + len(HEAD) == 18 and (len(BODY), len(HEAD)) == (15, 3)
    assert wiring == tuple((block, 0) for block, _ in (*BODY, *HEAD)) and sites == tuple(site for _, site in (*BODY, *HEAD))
    assert [block for block, _ in wiring] == [
        "draft.embedding",
        "draft.pre_fc_norm",
        "draft.projection",
        "draft.norm_quantize",
        "draft.projection",
        "draft.attention.prepare",
        "draft.attention.core",
        "draft.gate_quantize",
        "draft.projection",
        "draft.norm_quantize",
        "draft.projection",
        "draft.silu_mul",
        "draft.activation_quantize",
        "draft.projection",
        "draft.final_norm",
        "draft.head.quantize",
        "draft.head",
        "draft.token_choice",
    ]
    assert sites == ("", "", "fc", "input", "qkv", "", "", "", "o", "mlp", "gate_up", "", "", "down", "", "", "", "")
    assert drafter.calls() == {block: (5 if block == "draft.projection" else 2 if block == "draft.norm_quantize" else 1) for block in table}
    assert drafter.calls_text() == {block: str(count) for block, count in drafter.calls().items()} and sum(drafter.calls().values()) == 18
    assert list(table) == list(dict.fromkeys(block for block, _ in wiring)) == list(configs.DRAFTER_BLOCK_TITLES), "one row per block, in the order of the pass"
    assert len(table) == 13 and set(table).isdisjoint(configs.PROVIDERS) and set(configs.DRAFTER_BLOCK_TITLES).isdisjoint(configs.BLOCK_TITLES)

    # the call sites are data: the five weights by the loader's names with their shapes at the published widths, the two norms
    assert (
        [site for site, _ in configs.DRAFT_PROJECTION_SITES]
        == [name for name, _ in MULTIPLIES]
        == [s for (b, _), s in zip(wiring, sites) if b == "draft.projection"]
    )
    assert [site for site, _ in configs.DRAFT_NORM_SITES] == ["input", "mlp"] == [s for (b, _), s in zip(wiring, sites) if b == "draft.norm_quantize"]
    shapes = {spec.name[len("drafter.") : -len(".weight")]: spec.shape for spec in drafter.tensor_specs() if spec.name.startswith("drafter.")}
    merged = {name: (sum(shapes[part][0] for part in parts), shapes[parts[0]][1]) for name, parts in MULTIPLIES}
    assert {site: (dict(fields)["OUT_FEATURES"], dict(fields)["IN_FEATURES"]) for site, fields in configs.DRAFT_PROJECTION_SITES} == merged
    assert merged == {"fc": (5120, 10240), "qkv": (14336, 5120), "o": (5120, 6144), "gate_up": (34816, 5120), "down": (5120, 17408)}
    assert dict(configs.DRAFT_NORM_SITES) == {"input": (("ADD", False),), "mlp": (("ADD", True),)}

    # every row: every stack, any GPU class, any row count; a reference chain but the gather, and no form named for a chain
    chain = "reference chain"
    for block, rows in table.items():
        assert len(rows) == 1, block
        row = rows[0]
        assert row.stacks == ("fe-cublas", "reference", "native", "fe-entries") and row.classes == ("any",) and row.serves == "any" and row.band == "same"
        assert all(row.serves_run(sequences, tokens) for sequences, tokens in ((1, 1), (1, 6), (1, 8), (4, 6), (64, 1), (1000, 100))), block
        assert all(row.serves_class(cc) for cc in ((8, 0), (10, 0), (10, 3), (12, 1))) and row.serves_model(drafter) and row.serves_model(None)
        assert (row.form, row.sites, row.widths) == ((), (), ()), "a reference chain is called in no form: the kernel check has nothing to hold one to"
        assert row.fallback == (
            "" if block == "draft.embedding" else chain + " over a plain cache of the drafter's own" if block == "draft.attention.core" else chain
        )
    keys = {block: rows[0].key for block, rows in table.items()}
    assert keys == {
        "draft.embedding": "torch.index_select",
        "draft.pre_fc_norm": "torch_ref.pre_fc_norm_quantize",
        "draft.projection": "torch_ref.linear",
        "draft.norm_quantize": "torch_ref.add_norm_quantize",
        "draft.attention.prepare": "torch_ref.attention_prepare",
        "draft.attention.core": "torch_ref.attention",
        "draft.gate_quantize": "torch_ref.gate_row_quantize",
        "draft.silu_mul": "torch_ref.silu_mul",
        "draft.activation_quantize": "torch_ref.quantize",
        "draft.final_norm": "torch_ref.add_norm_quantize",
        "draft.head.quantize": "torch_ref.quantize",
        "draft.head": "torch_ref.linear",
        "draft.token_choice": "torch.argmax+lookup",
    }

    # the five keys of the pass in the closed list, each with its factory; the list is the model's keys and these
    new = ("torch.argmax+lookup", "torch_ref.pre_fc_norm_quantize", "torch_ref.gate_row_quantize", "torch_ref.silu_mul", "torch_ref.quantize")
    classes = (torch_ref.ListedTokenChoice, torch_ref.PreFcNormQuantize, torch_ref.GateRowQuantize, torch_ref.SiluMul, torch_ref.Quantize)
    assert all(key in providers.KEYS for key in new) and [providers.factories()[key] for key in new] == list(classes)
    assert len(providers.KEYS) == len(set(providers.KEYS)) == len(providers.factories()) == 28 and set(providers.KEYS) == set(providers.factories())
    used_by_the_model = {row.key for rows in configs.PROVIDERS.values() for row in rows}
    assert used_by_the_model & set(new) == {"torch_ref.quantize"}, "one key of the pass is a row of the model too: the reference chain of the head's quantise"
    assert set(providers.KEYS) == used_by_the_model | set(new), "no key that no table names"
    assert all(providers.ENTRY_NAMES[key] == key for key in new[1:]) and providers.ENTRY_NAMES[new[0]] == "torch.argmax + the token id of the chosen row"

    # a dry plan in every stack: the same rows, the same table but for the stack's name
    tables = {}
    for stack in ("fe-cublas", "native", "fe-entries", "reference"):
        for cc, sequences, tokens in (((10, 0), 1, 1), ((12, 0), 4, 6), ((8, 0), 1, 1)):
            resolved = plan(table, drafter.regimes, DeviceFacts(cc), sequences, tokens, stack, model=drafter, source=drafter.table_source)
            assert {block: row.key for block, row in resolved.rows.items()} == keys and list(resolved) == list(table), (stack, cc)
            assert all(block.provider == "PyTorch" and block.row is resolved.rows[name] for name, block in resolved.items())
            assert [kernel.id for kernel in resolved["draft.embedding"].kernels] == ["embedding_gather"]
            assert all(
                [kernel.relation for kernel in block.kernels] in (["fallback"], ["fallback", "fallback"])
                for name, block in resolved.items()
                if name != "draft.embedding"
            )
        text = plan(table, drafter.regimes, DeviceFacts((10, 0)), 1, 1, stack, model=drafter, source=drafter.table_source).table().splitlines()
        assert text[0] == f"served by (decided before the load from model_samples/qwen_dense/configs.py; stack {stack}, 1 sequence, cc 10.0)"
        assert text[-2] == "  by provider: fallbacks 12 (PyTorch chains 12) | PyTorch 1" and len(text) == 1 + 1 + 13 + 2
        tables[stack] = text[1:]
    assert tables["fe-cublas"] == tables["native"] == tables["fe-entries"] == tables["reference"]
    lines = {line.split()[0]: line for line in tables["native"][1:14]}
    assert lines["draft.projection"].split()[1:4] == ["5", "PyTorch", "torch_ref.linear"] and lines["draft.norm_quantize"].split()[1] == "2"
    assert lines["draft.attention.core"].endswith("fallback: reference chain over a plain cache of the drafter's own")
    assert lines["draft.embedding"].split() == ["draft.embedding", "1", "PyTorch", "torch.index_select", "any", "none"]
    with pytest.raises(ValueError, match="unknown stack 'fast'"):
        plan(table, drafter.regimes, DeviceFacts((10, 0)), 1, 1, "fast", model=drafter)

    # the family README: the drafter's block table and its call-site table are rendered from this data; the model's table is untouched
    readme = (MODELS / "qwen_dense" / "README.md").read_text()
    rendered = configs.readme_table(drafter.calls_text(), "native", table=table, titles=configs.DRAFTER_BLOCK_TITLES, per="pass")
    assert rendered.splitlines()[0] == "| block | calls per pass | provider key | entry [form] |" and len(rendered.splitlines()) == 2 + 13
    assert rendered in readme, "regenerate the drafter's block table of the README from configs.readme_table(.., table=configs.DRAFTER_PROVIDERS, ..)"
    assert all(
        rendered == configs.readme_table(drafter.calls_text(), stack, cc, table, configs.DRAFTER_BLOCK_TITLES, "pass")
        for stack in configs.EVERY
        for cc in ((10, 0), (12, 0))
    )
    assert configs.readme_draft_sites() in readme and len(configs.readme_draft_sites().splitlines()) == 2 + 5 + 2
    assert configs.readme_table(QwenDense().calls_text()).splitlines()[0] == "| block | calls per step (64 layers) | provider key | entry [form] |"
    assert configs.readme_table(QwenDense().calls_text()) == configs.readme_table(
        QwenDense().calls_text(), table=configs.PROVIDERS, titles=configs.BLOCK_TITLES
    )


def test_call_counts_of_a_decode_step():
    from model_samples.qwen_dense import QwenDense

    assert QwenDense().calls_per_step() == 485
    assert QwenDense.reduced(4).calls_per_step() == 35 and QwenDense.reduced(4).geometry.layer_kinds.count("linear_attention") == 3
    assert QwenDense.reduced(2).calls_per_step() == 21 and QwenDense.reduced(2).geometry.layer_kinds == ("linear_attention", "full_attention")
    assert QwenDense().calls_text()["projection.fp8.in"] == "48 / 16" and QwenDense().calls_text()["add_norm_quantize"] == "129"
    with pytest.raises(ValueError, match="layers"):
        QwenDense.reduced(0)


def test_plan_selects_the_first_matching_row_of_the_stack_and_prints_it():
    plan = _plan()
    assert plan.rows["linear_attention.step"].key == "fe.gdn_decode_fused" and plan["linear_attention.step"].provider == "cuDNN FE"
    assert plan.rows["mlp.down"].key == "cublas.bf16_dequantized" and plan.rows["token_choice"].key == "torch.argmax"
    table = plan.table()
    assert "by provider: cuDNN FE 1 | fallbacks 9 (cuBLAS over copies 4, PyTorch chains 5) | PyTorch 2" in table
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
    with pytest.raises(ValueError, match="not a declared row"):  # the key that named the quantiser inside the output head's row: no key any more, no alias
        _plan(stack="native", overrides={"output_head": "torch_ref.quantize+flashinfer.mm_fp4"})
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
    from model_samples import blocks, loop
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
    # ... and what the checks hold for the three sequences beyond the first of the larger count, with the share of it for PyTorch's pool
    rows = model.check_row_bytes(4, 58) - model.check_row_bytes(1, 58)
    need = model.memory_bytes(1, 58) + model.state_bytes(4, 58) + configs.FIRST_LAUNCH_RESERVE["cc12.x"] + rows + int(rows * loop.CHECK_POOL)
    assert need < 7 * GIB, "the copies and the working memory are counted once for the run, the first launch once"
    checks = f"{rows / GIB:.2f} GiB of it for the checks (what they hold for the sequences beyond the first of a count)"
    assert f"this stack needs about {need / GIB:.1f} GiB on this GPU for 1 and 4 sequences, one count after the other, {checks}; 8.0 GiB are free" in table
    assert f"{need / GIB:.1f}" == "6.9" and f"{rows / GIB:.2f}" == "0.14"
    assert "one count after the other" not in run(4)[1], "one count: the sentence of that count"
    with pytest.raises(NotImplementedError, match="serves rows 1-65,535; this run has 70000 sequences"):
        run(1, 70000)
    free[0] = 6 * GIB
    with pytest.raises(
        NotImplementedError,
        match=r"needs about 6\.9 GiB on this GPU for 1 and 4 sequences, one count after the other, 0\.14 GiB of it for the checks \(.*\); 6.0 GiB are free: free memory",
    ):
        run(1, 4)


def _sample_main(monkeypatch, capsys, arguments, cc=(10, 0), free=80 * GIB, environment=None, script="00_decode.py"):
    """``main()`` of a sample script on stated device facts, up to its first load: no GPU is touched and nothing is loaded.
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
    spec = importlib.util.spec_from_file_location(f"model_sample_qwen_dense_{script[:2]}_on_stated_facts", MODELS / "qwen_dense" / script)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        return module.main(arguments), capsys.readouterr().err
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
    checks = "0.14 GiB of it for the checks (what they hold for the sequences beyond the first of a count)"
    skip = (
        f"[00] SKIP stack fe-cublas needs about 6.9 GiB on this GPU for 1 and 4 sequences, one count after the other, {checks}; 6.0 GiB are free: "
        "free memory on this GPU"
    )
    assert code == 0 and out.splitlines()[-1].startswith(skip), out
    code, out = _sample_main(monkeypatch, capsys, ["--provider", "mlp.down=nothing"])
    assert code == 1 and out.splitlines()[-1].startswith("[00] ERROR --provider mlp.down=nothing: not a declared row"), out
    with pytest.raises(AssertionError, match="reached a load"):
        _sample_main(monkeypatch, capsys, [], cc=(12, 0), free=8 * GIB)
    out = capsys.readouterr().err
    assert f"this stack needs about 6.9 GiB on this GPU for 1 and 4 sequences, one count after the other, {checks}; 8.0 GiB are free" in out
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
    assert session.report().calls_per_step == session.model.calls_per_step() == 21
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
        def __call__(self, qx, weight, site=""):  # the in-projection is handed its call site; a multiply over a copy ignores it
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


def _sample_script(monkeypatch, script="00_decode.py"):
    """A sample script as a module, for its own functions; what loading it changes in the process leaves with the test."""
    import importlib.util
    import os

    monkeypatch.setattr(sys, "path", list(sys.path))  # the script puts samples/models on the path
    monkeypatch.setenv("KINETO_LOG_LEVEL", os.environ.get("KINETO_LOG_LEVEL", "6"))  # and sets a default for the profiler's log level
    spec = importlib.util.spec_from_file_location(f"model_sample_qwen_dense_{script[:2]}_functions", MODELS / "qwen_dense" / script)
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
    out = capsys.readouterr().err.splitlines()
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
    out = capsys.readouterr().err
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
    assert "this stack moves 51.56 GB (2.88x): bf16 copies of the weights (arithmetic, not a measurement)" in text, "every weight read as a copy"
    kept = Report("qwen_dense", reduced=False)  # a stack with a block that multiplies on the stored format: those weights are read there
    kept.bytes = {"weights": 17_610_000_000, "copies": 46_050_000_000, "stored": 5_200_000_000, "state": 310_000_000}
    moves = "this stack moves 46.36 GB (2.59x): bf16 copies of the weights, and 5.20 GB of weights in their stored format (arithmetic, not a measurement)"
    assert moves in kept.text()
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
    assert session.report().trace_missing == "" and session.report().calls_per_step == 21 and not session.report().text_path, "a fresh report per prompt set"
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
    assert step["stored"] == 0 and set(step) == {"weights", "copies", "stored", "state"}, "no weight of this stack is read on its stored format"
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


def test_one_session_with_two_plans_keeps_the_forms_both_read_and_counts_one_state(monkeypatch):
    """A session built with ``draft_tokens=`` has a verify plan and a decode plan, on ONE state.  On stated facts (compute
    capability 10.0, 148 SMs, stack ``native``, the published geometry): its load keeps the forms both plans read -- as a
    run of a decode session and a verify session does -- and the memory sentence counts one state of ``draft_tokens + 1``
    tokens per sequence, where two sessions count two."""
    from model_samples import Session, TextIO, blocks, loop
    from model_samples.qwen_dense import QwenDense

    monkeypatch.setattr(blocks.DeviceFacts, "probe", classmethod(lambda cls, device: cls((10, 0), "stated device", 148, 0, 80 * GIB)))
    model = QwenDense()
    text = TextIO.synthetic(model.geometry.vocab)
    common = dict(sequences=1, stack="native", device="cuda:0", capacity=2051)

    def new(**more):
        return Session(model, text, **common, **more)

    session = new(draft_tokens=5)
    table = session.plan_table()
    assert (session.plan.tokens, session.prefill_plan.tokens) == (6, 1) and session.plan.first_used and session.prefill_plan.first_used
    numbers = session.claim().numbers()  # what the session says about the target kernel set is its verify plan's
    assert (numbers["launches"], numbers["target_code"], numbers["target_port"]) == (484, 146, 338)
    # in the stack native the library multiplies the out-projection on its stored codes at both call sites of both plans
    stored = alone = ("projection.fp8.in", "projection.fp8.out", "mlp.gate_up", "mlp.down", "output_head")
    assert session.stored_weights() == stored and session.uncopied_weights() == alone
    # the same forms as a decode session and a verify session on one load: the linear-attention out-projections in both forms
    decode, verify = new(), new(tokens=6)
    verify._share = decode
    two = decode.plan_table([verify])
    assert set(decode.stored_weights()) == set(stored) and set(decode.uncopied_weights()) == set(alone)
    assert new(tokens=6).uncopied_weights() == stored, "a verify plan alone keeps the same forms: every multiplied weight on its stored codes, no copy"
    entries = Session(model, text, **{**common, "stack": "fe-entries"}, draft_tokens=5)  # the cuDNN Frontend rows alone: the two plans differ there
    entries.resolve()
    assert entries.stored_weights() == ("projection.fp8.in", "projection.fp8.out at linear_attention.out", "mlp.gate_up")
    assert entries.uncopied_weights() == ("projection.fp8.in", "mlp.gate_up"), "the decode plan of its prompt reads a copy of the out-projections"
    # the tables: the verify steps' rows, one line, the prompt's rows, then ONE memory sentence
    head, line, rest = table.partition(f"\n{loop.PROMPT_ROWS}\n")
    assert line and head == session.plan.table() and rest.startswith(session.prefill_plan.table() + "\n  this stack needs about ")
    assert "stack native, 1 sequence x 6 tokens per sequence, cc 10.0" in head.splitlines()[0] and "stack native, 1 sequence, cc 10.0" in rest.splitlines()[0]
    assert "cudnn.GdnVerifyFused" in head and "cudnn.GdnDecodeFused" not in head and "cudnn.GdnDecodeFused" in rest
    # the state is a pool of pages with its tables, and the workspace of the library's attention is counted once
    from model_samples.providers import flashinfer_calls

    paged, workspace = session._paged(), flashinfer_calls.WORKSPACE_BYTES
    need = model.memory_bytes(1, 2051, stored=stored, uncopied=alone, tokens=6, **paged) + workspace
    checks = model.check_bytes(alone)
    sentence = (
        f"this stack needs about {need / GIB:.1f} GiB on this GPU for a session of 1 sequence x 6 tokens behind its prompt, "
        f"{checks / GIB:.2f} GiB of it for the checks"
    )
    assert table.count("this stack needs about") == 1 and sentence in table, table.splitlines()[-1]
    # one state: two sessions count the decode session's state of one token beside the verify session's
    both = model.memory_bytes(1, 2051, stored=stored, uncopied=alone, **paged) + model.state_bytes(1, 2051, tokens=6, **paged) + workspace
    assert (
        both - need == model.state_bytes(1, 2051, **paged)
        and f"needs about {both / GIB:.1f} GiB on this GPU for 1 and 1 sequences, one count after the other" in two
    )
    # a run of a plain session and such a session names the kinds of step; a run without one keeps its words
    plain = new()
    behind = Session(model, text, share=plain, draft_tokens=5, **common)
    run = plain.plan_table([behind])
    assert plain.stored_weights() == decode.stored_weights() and plain.uncopied_weights() == decode.uncopied_weights()
    assert (
        f"needs about {both / GIB:.1f} GiB on this GPU for a plain session of 1 sequence and a session of 1 sequence x 6 tokens behind its prompt, one after the other"
        in run
    )
    assert loop.PROMPT_ROWS not in run, "the first session's table; a later session prints its own"
    four = Session(model, text, share=plain, **{**common, "sequences": 4})
    counts = new().plan_table([four])
    assert "for 1 and 4 sequences, one count after the other" in counts
    # an override selects its row in both plans; a key whose rows do not serve one of the two kinds of step declines there
    chain = new(draft_tokens=5, overrides={"linear_attention.step": "torch_ref.gdn_decode_step"})
    chain.resolve()
    assert chain.plan.rows["linear_attention.step"].key == chain.prefill_plan.rows["linear_attention.step"].key == "torch_ref.gdn_decode_step"
    with pytest.raises(NotImplementedError, match="cudnn.GdnDecodeFused serves .*; this run has 1 sequences x 6 tokens per sequence"):
        new(draft_tokens=5, overrides={"linear_attention.step": "fe.gdn_decode_fused"}).resolve()
    with pytest.raises(NotImplementedError, match="cudnn.GdnVerifyFused serves .*; this run has 1 sequences x 1 tokens per sequence"):
        new(draft_tokens=5, overrides={"linear_attention.step": "fe.gdn_verify_fused"}).resolve()
    # a count no verify row of the stack holds is declined by that row's sentence, before anything is loaded
    with pytest.raises(NotImplementedError) as declined:
        new(draft_tokens=8).plan_table()
    assert str(declined.value) == "cudnn.GdnVerifyFused serves 8 tokens per sequence of 2-8, sequences 1-4; this run has 1 sequences x 9 tokens per sequence"
    with pytest.raises(NotImplementedError) as declined:
        Session(model, text, draft_tokens=5, **{**common, "sequences": 5}).plan_table()
    assert str(declined.value) == "cudnn.GdnVerifyFused serves 6 tokens per sequence of 2-8, sequences 1-4; this run has 5 sequences x 6 tokens per sequence"
    # what the first launch of the fused step reserves outside the pool is reserved once in a process, so the sentence carries it
    # once -- not once more for the second plan of such a session.  On a class that has such a reserve (compute capability
    # 12.x) a run with such a session asks what a decode session and a verify session on one load ask, and a lone one what
    # its forms and its one state need plus one reserve
    from model_samples.providers import fe

    monkeypatch.setattr(fe, "_FIRST_LAUNCH", {})
    monkeypatch.setattr(fe, "_VERIFY_LAUNCHED", set())
    monkeypatch.setattr(blocks.DeviceFacts, "probe", classmethod(lambda cls, device: cls((12, 0), "stated device", 170, 0, 80 * GIB)))
    reserve = model.first_launch_reserve["cc12.x"]
    assert reserve == 3 * GIB

    def asked(table):
        return table.splitlines()[-1].split("needs about ")[1].split(" GiB on this GPU")[0]

    lone = new(draft_tokens=5)
    alone_table = lone.plan_table()
    assert lone.plan.reserve_bytes() == lone.prefill_plan.reserve_bytes() == reserve, "each plan has a fused step that would make the first launch"
    need = model.memory_bytes(1, 2051, **lone._kept(), tokens=6)
    assert asked(alone_table) == f"{(need + reserve) / GIB:.1f}" != f"{(need + 2 * reserve) / GIB:.1f}", alone_table.splitlines()[-1]
    decode, verify, plain = new(), new(tokens=6), new()
    verify._share = decode
    behind = Session(model, text, share=plain, draft_tokens=5, **common)
    with_prompt, two_sessions = plain.plan_table([behind]), decode.plan_table([verify])
    assert asked(with_prompt) == asked(two_sessions), "one reserve and one more state in both runs"


# --------------------------------------------------------------------------- the speculative script: its plan, its verdict, its help

SPECULATIVE_SCRIPT = "02_speculative_decode.py"


def test_the_speculative_script_plans_every_kind_of_step_before_its_first_load(monkeypatch, capsys):
    """``main()`` of ``02_speculative_decode.py`` on stated device facts, up to its first load.  The plain plan, the verify
    plan, the prompt's decode plan and the drafter's are printed with ONE memory sentence, the drafter in it, behind the
    line that says what a prompt and a step are; a count of drafted tokens no verify row of the stack holds is declined
    by that row's sentence -- a ``SKIP``, before any load; so is a GPU with less free memory than the sentence asks for."""
    from model_samples import EnvironmentMismatch, loop

    def run(*arguments, **facts):
        return _sample_main(monkeypatch, capsys, list(arguments), script=SPECULATIVE_SCRIPT, **facts)

    with pytest.raises(AssertionError, match="reached a load"):
        run()
    said = capsys.readouterr().err
    lines = said.splitlines()
    before = "[02] prompt of 26 tokens = 26 decode steps and 25 passes of the drafter before the first token; then every step is 5 draft passes and "
    before += "one verify step of 6 tokens, and keeps 1 to 6 of them"
    first = next(index for index, line in enumerate(lines) if line.startswith("served by (decided before the load"))
    assert lines.index(before) < first, "the rule and the counts, before the tables and so before the load"
    assert said.count("\nserved by (decided before the load") == 4, "the plain plan, the verify plan, the prompt's decode plan, the drafter's"
    assert loop.PROMPT_ROWS in lines and loop.DRAFTER_ROWS.format(calls=18) in lines
    kinds = "a plain session of 1 sequence and a session of 1 sequence x 6 tokens behind its prompt with its drafter, one after the other"
    assert said.count("this stack needs about ") == 1 and f" GiB on this GPU for {kinds}; 80.0 GiB are free" in said, "one memory sentence for the run"
    assert "[02] --speculative-config (the default): the speculative configuration of a serving engine. Read: method" in said
    assert "[02] draft vocabulary seeded list (seed 0): 3,880 ids of the model's 15,520, in ascending order" in said
    reductions = "acceptance is scheduled, not measured: the drafted tokens of every step are replaced by the plain run's tokens for 0, 1, .., 5 "
    assert reductions + "positions in turn and by another token behind them | the drafter runs on reference blocks" in lines[0]
    # eight drafted tokens are nine tokens per sequence: no verify row of a provider stack holds them
    eight = '{"method": "mtp", "num_speculative_tokens": 8, "use_local_argmax_reduction": true}'
    code, out = run("--speculative-config", eight)  # the seeded default of new tokens follows the schedule: 38 here, said in the stop rule
    last = out.splitlines()[-1]
    assert code == 0 and last.startswith("[02] SKIP cudnn.GdnVerifyFused serves ") and last.endswith("this run has 1 sequences x 9 tokens per sequence"), last
    with pytest.raises(AssertionError, match="reached a load"):  # seven are a row of the stack; and the reference stack holds any count
        run("--speculative-config", eight.replace("8", "7"))
    assert "then every step is 7 draft passes and one verify step of 8 tokens, and keeps 1 to 8 of them" in capsys.readouterr().err
    with pytest.raises(AssertionError, match="reached a load"):
        run("--speculative-config", eight, "--stack", "reference")
    capsys.readouterr()
    code, out = run("--speculative-config", eight.replace("8", "64"), "--stack", "reference")  # ... as far as its schedule fits the context
    last = out.splitlines()[-1]
    assert code == 1 and last.startswith("[02] ERROR the seeded prompt: ") and "(--force-tokens 2082)" in last and "--max-model-len is 2,051" in last, last
    code, out = run("--speculative-config", eight, "--stack", "reference", "--force-tokens", "37")  # a count that was typed is held, before the plan
    short = "[02] ERROR --force-tokens 37: the seeded run schedules its steps to keep 0 to 8 drafted tokens in turn and needs 38 new tokens to keep every count once"
    assert code == 1 and out.splitlines() == [short]
    code, out = run(free=5 * GIB)
    skip = rf"^\[02\] SKIP stack fe-cublas needs about \d\.\d GiB on this GPU for {kinds}; 5\.0 GiB are free: free memory on this GPU"
    assert code == 0 and re.match(skip, out.splitlines()[-1]), out.splitlines()[-1]
    code, out = run(cc=(9, 0))
    assert code == 0 and out.splitlines()[-1] == "[02] SKIP cudnn.GdnDecodeFused serves compute capability 10.x or 12.x; this device is 9.0", out
    code, out = run("--provider", "mlp.down=nothing")
    assert code == 1 and out.splitlines()[-1].startswith("[02] ERROR --provider mlp.down=nothing: not a declared row"), out

    def stale():  # an environment that is not the tree's ends the run in one ERROR line, behind the plan: a GPU that is not served skips first
        raise EnvironmentMismatch("cudnn 1.33.0 was imported from an editable checkout, but the installed distribution says 1.29.0")

    code, out = run(environment=stale)
    assert code == 1 and out.splitlines()[-1] == "[02] ERROR cudnn 1.33.0 was imported from an editable checkout, but the installed distribution says 1.29.0"
    code, out = run(cc=(9, 0), environment=stale)
    assert code == 0 and out.splitlines()[-1].startswith("[02] SKIP cudnn.GdnDecodeFused serves compute capability 10.x or 12.x"), out
    # --provider selects its row in both kinds of step: an entry's key of one kind is declined by that row's sentence, a fallback key serves both
    code, out = run("--provider", "linear_attention.step=fe.gdn_decode_fused")
    assert code == 0 and out.splitlines()[-1] == "[02] SKIP cudnn.GdnDecodeFused serves rows 1-65,535; this run has 1 sequences x 6 tokens per sequence"
    code, out = run("--provider", "linear_attention.step=fe.gdn_verify_fused")
    assert code == 0 and out.splitlines()[-1].endswith("this run has 1 sequences x 1 tokens per sequence") and "[02] SKIP cudnn.GdnVerifyFused serves " in out
    with pytest.raises(AssertionError, match="reached a load"):
        run("--provider", "mlp.down=torch_ref.linear")
    capsys.readouterr()


def test_both_scripts_end_in_the_verdict_the_readme_states(monkeypatch):
    """The last line of a run is one function per script, and the two answer alike over every input: a failed check is
    ``FAIL`` on every stack -- the speculative script compares on the reference stack too --, then the reference stack's
    ``RAN``, kernels that could not be observed, and ``PASS`` only where provider blocks were judged.  The area README's
    table of last lines is those answers, with ``[NN]`` for a script's tag."""
    plain, speculative = _sample_script(monkeypatch), _sample_script(monkeypatch, SPECULATIVE_SCRIPT)
    assert (plain.TAG, speculative.TAG) == ("[00]", "[02]")
    reason = "the profiler failed (no tracing library)"
    said = {}
    for stack in plain.STACKS:
        for passed in (True, False):
            for judged in (0, 6):
                for unobserved in ("", reason):
                    answer = plain._verdict(stack, passed, judged, unobserved)
                    assert speculative._verdict(stack, passed, judged, unobserved) == answer, (stack, passed, judged, unobserved)
                    said[(stack, passed, judged, bool(unobserved))] = answer
    assert all(answer == "FAIL" for (_, passed, _, _), answer in said.items() if not passed), "a failed check, whatever else is true"
    assert {answer for (stack, passed, _, _), answer in said.items() if passed and stack == "reference"} == {"RAN reference stack (no provider checked)"}
    assert said[("native", True, 6, True)] == f"RAN native stack (kernels not observed: {reason})" == said[("native", True, 0, True)]
    assert (said[("fe-cublas", True, 6, False)], said[("fe-cublas", True, 0, False)]) == ("PASS", "RAN (no provider block was judged)")
    readme = (MODELS / "README.md").read_text()
    table = readme.split("| last line | exit code | meaning |\n|---|---|---|\n", 1)[1].split("\n\n", 1)[0].splitlines()
    rows = [(row.split(" | ")[0].strip("| `"), row.split(" | ")[1]) for row in table]
    assert rows == [
        ("[NN] PASS", "0"),
        ("[NN] SKIP <requirement>", "0"),
        ("[NN] RAN reference stack (no provider checked)", "0"),
        ("[NN] RAN (no provider block was judged)", "0"),
        ("[NN] RAN <stack> stack (kernels not observed: <reason>)", "0"),
        ("[NN] ERROR <reason>", "1"),
        ("[NN] FAIL", "1"),
    ]
    stated = {row.replace("<reason>", reason)[5:] for row, _ in rows}
    answers = {answer.replace(f"RAN {stack} stack (kernels", "RAN <stack> stack (kernels") for (stack, _, _, _), answer in said.items()}
    assert answers <= stated and len(answers) == 5, "every answer of the two functions is a row of the table: all but SKIP and ERROR, which main() prints"
    assert "`[NN]` is the script's tag: `[00]` for `00_decode.py`, `[02]` for `02_speculative_decode.py`." in " ".join(readme.split())


def test_the_speculative_scripts_help_states_the_memory_the_tree_computes(monkeypatch):
    """The memory figures of the help text of ``02_speculative_decode.py``, per stack and GPU class, are the one memory
    sentence of its run at the published widths on stated facts: a plain session and a session with the drafter (65,536
    listed ids) at the default context length."""
    from model_samples import Session, TextIO, blocks
    from model_samples.qwen_dense import QwenDense, QwenDenseDrafter

    sample = _sample_script(monkeypatch, SPECULATIVE_SCRIPT)
    model = QwenDense()
    text, drafter = TextIO.synthetic(model.geometry.vocab), QwenDenseDrafter.seeded(model, rows=65536)

    def asked(stack, cc):
        monkeypatch.setattr(blocks.DeviceFacts, "probe", classmethod(lambda cls, device: cls(cc, "stated device", 148, 0, 200 * GIB)))
        common = dict(sequences=1, stack=stack, device="cuda:0", capacity=sample.serving_options.MAX_MODEL_LEN)
        plain = Session(model, text, **common)
        sentence = plain.plan_table([Session(model, text, draft_tokens=5, share=plain, drafter=drafter, **common)]).splitlines()[-1]
        return round(float(sentence.split("needs about ", 1)[1].split(" GiB", 1)[0]))

    figures = {(stack, cc): asked(stack, cc) for stack in ("fe-cublas", "native", "fe-entries") for cc in ((10, 0), (10, 3), (10, 7), (12, 0))}
    cublas, entries, native = ([figures[(stack, cc)] for cc in ((10, 0), (10, 3), (10, 7), (12, 0))] for stack in ("fe-cublas", "fe-entries", "native"))
    # native keeps the weights of its three library rows on their stored codes on compute capability 10.x; elsewhere it plans as fe-entries
    assert (cublas, entries, native) == ([61, 61, 61, 64], [44, 48, 48, 64], [33, 37, 37, 64])
    said = (
        f"about {cublas[0]} GiB of free GPU memory on the stack fe-cublas (about {cublas[3]} GiB on compute capability 12.x), on the stack fe-entries for "
        f"about {entries[0]} GiB on compute capability 10.0, about {entries[1]} GiB on the other 10.x classes and about {entries[3]} GiB elsewhere, and on "
        f"the stack native, whose library rows need FlashInfer and its kernel-binary package (00_decode.py --help has the command), for about {native[0]} GiB and about "
        f"{native[1]} GiB there"
    )
    assert said in " ".join(sample.EPILOG.split()) and "a list of 65,536 ids and the default context length" in " ".join(sample.EPILOG.split())
    # ... and the family README's section on the script states these figures, not others
    readme = " ".join((MODELS / "qwen_dense" / "README.md").read_text().split("\n## Speculative decoding\n", 1)[1].split("\n## ", 1)[0].split())
    assert (
        f"with a list of 65,536 ids and the default context length a run asks for about {cublas[0]} GiB free on the stack `fe-cublas` (about "
        f"{cublas[3]} GiB on compute capability 12.x); on the stack `fe-entries` for about {entries[0]} GiB on compute capability 10.0, about "
        f"{entries[1]} GiB on the other 10.x classes and about {entries[3]} GiB elsewhere; on the stack `native`, whose library rows need "
        f'FlashInfer and its kernel-binary package (the section "The rows a library serves"), for about {native[0]} GiB and about {native[1]} GiB there.'
    ) in readme


# ---------------------------------------------------------------------------------------------------------------------
# The paged attention cache of a state (``state.PagedKvCache``, ``state.PageTable``): a pool of pages of 64 positions per
# sequence, with the position and slot tables a kernel that writes a paged pool is handed.  The row of the attention
# preparation's entry asks for it; a session builds it where a block of its plans does (``Block.cache_page``), and the reference chains reach either
# layout through ``write`` / ``keys`` / ``values``.  Every index of the pool below is arithmetic written in the test.

PAGE = 64


def _bits(tensor):
    """The bytes of a tensor as integers: 0.0 and -0.0 are two values, and a NaN compares equal to itself."""
    whole = tensor.contiguous()
    return whole.view({1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[whole.element_size()])


def _same(a, b) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(_bits(a), _bits(b))


#: a geometry of the tiny kind with TWO key / value heads and four query heads: an axis of one head would hide a pool read
#: in the wrong head order
KV_SHAPES = {"state": (2, 8, 8), "conv": (3, 32), "kv": (2, 8), "rotary": (4, 1.0e7)}


def _paged_model(kinds=("linear_attention", "full_attention")):
    from model_samples.qwen_dense import Geometry, QwenDense

    geometry = Geometry(
        layer_kinds=kinds,
        hidden=64,
        vocab=272,
        mlp=96,
        la_key_heads=1,
        la_value_heads=2,
        la_head_dim=8,
        attn_heads=4,
        attn_kv_heads=2,
        attn_head_dim=8,
        rotary_dim=4,
    )
    return QwenDense(geometry, reductions=("tiny geometry",))


def _two_states(sequences=3, capacity=130, tokens=1, pages=None):
    """A plain state and a paged one of the same sequences, positions and tokens per sequence, one attention layer each."""
    from model_samples.state import DecodeState

    kinds = ("full_attention",)
    plain = DecodeState(kinds, sequences, capacity, KV_SHAPES, "cpu", tokens)
    paged = DecodeState(kinds, sequences, capacity, KV_SHAPES, "cpu", tokens, page=PAGE, pages=pages)
    return plain, paged


def _written(plain, paged, positions, seed=0):
    """Write the same drawn key and value rows at ``positions`` into both caches; returns what was written, ``(keys, values)``
    ``[sequences, kv heads, positions, head width]``, kept by the test."""
    generator = torch.Generator().manual_seed(seed)
    sequences, (heads, width) = plain.sequences, KV_SHAPES["kv"]
    keys = torch.zeros((sequences, heads, len(positions), width), dtype=torch.bfloat16)
    values = torch.zeros_like(keys)
    for index, position in enumerate(positions):
        k, v = (torch.randn((sequences, heads, width), generator=generator).to(torch.bfloat16) for _ in range(2))
        keys[:, :, index], values[:, :, index] = k, v
        for state in (plain, paged):
            state.layers[0].write(position, k, v)
    return keys, values


def test_a_state_builds_the_plain_cache_or_a_pool_of_pages():
    """``page=None`` builds exactly the tensors a state had: two per attention layer, ``[sequences, kv heads, capacity, head
    width]``.  ``page=64`` builds per attention layer ONE zero pool ``[sequences x pages, kv heads, 64, 2 x head width]``
    -- keys in the first half of the last axis, values in the second -- with ``ceil(capacity / 64)`` pages per sequence
    unless ``pages`` says more; and the tables of the state, once for all its layers."""
    from model_samples.state import DecodeState, KvCache, PagedKvCache, RecurrentSlots

    kinds = ("linear_attention", "full_attention", "full_attention")
    plain = DecodeState(kinds, 3, 130, KV_SHAPES, "cpu")
    assert [type(layer) for layer in plain.layers] == [RecurrentSlots, KvCache, KvCache] and plain.page is None and plain.table is None
    for cache in plain.layers[1:]:
        assert [(tuple(t.shape), t.dtype) for t in cache.tensors()] == [((3, 2, 130, 8), torch.bfloat16)] * 2, "today's two tensors"
    assert (
        DecodeState(kinds, 3, 130, {name: shape for name, shape in KV_SHAPES.items() if name != "rotary"}, "cpu").bytes() == plain.bytes()
    ), "a plain state reads no rotary fact"
    for capacity, pages in ((1, 1), (63, 1), (64, 1), (65, 2), (128, 2), (129, 3), (130, 3)):
        state = DecodeState(kinds, 3, capacity, KV_SHAPES, "cpu", page=PAGE)
        assert (state.page, state.pages) == (PAGE, pages), f"capacity {capacity}: ceil(capacity / 64) pages per sequence"
        assert isinstance(state.layers[0], RecurrentSlots) and tuple(state.layers[0].state.shape) == tuple(plain.layers[0].state.shape)
        for cache in state.layers[1:]:
            assert isinstance(cache, PagedKvCache) and (cache.page, cache.pages, cache.tokens) == (PAGE, pages, 1)
            (pool,) = cache.tensors()
            assert tuple(pool.shape) == (3 * pages, 2, PAGE, 16) and pool.dtype == torch.bfloat16 and not bool(pool.any()) and pool.is_contiguous()
            assert pool.data_ptr() == cache.pool.data_ptr() and cache.table is state.table, "one pool per layer, the state's one table"
            # the two halves a kernel is handed: views of the whole pool, the last axis contiguous, every other stride whole 16-byte vectors
            for half, offset in ((cache.k_pool, 0), (cache.v_pool, 8)):
                assert tuple(half.shape) == (3 * pages, 2, PAGE, 8) and half.stride() == (2 * PAGE * 16, PAGE * 16, 16, 1)
                assert half.data_ptr() - cache.pool.data_ptr() == offset * 2 and all((stride * 2) % 16 == 0 for stride in half.stride()[:3])
        assert state.layers[1].pool.data_ptr() != state.layers[2].pool.data_ptr(), "a pool per attention layer"
    more = DecodeState(kinds, 3, 130, KV_SHAPES, "cpu", page=PAGE, pages=13)  # pages per sequence are an argument: more than the capacity needs
    assert more.pages == 13 and tuple(more.layers[1].pool.shape) == (39, 2, PAGE, 16)
    small = DecodeState(kinds, 2, 40, KV_SHAPES, "cpu", page=16)  # the page is the state's argument, not a constant of the class
    assert (small.page, small.pages) == (16, 3) and tuple(small.layers[1].pool.shape) == (6, 2, 16, 16)
    with pytest.raises(ValueError, match="130 positions per sequence need 3 pages of 64; got pages=2"):
        DecodeState(kinds, 3, 130, KV_SHAPES, "cpu", page=PAGE, pages=2)
    for wrong in (0, -64, 64.0, True, "64"):
        with pytest.raises(ValueError, match="the page of an attention cache is a whole number of positions from 1"):
            DecodeState(kinds, 3, 130, KV_SHAPES, "cpu", page=wrong)
        with pytest.raises(ValueError, match=f"and a sequence owns a whole number of pages; got page=64, pages={wrong!r}"):
            DecodeState(kinds, 3, 130, KV_SHAPES, "cpu", page=PAGE, pages=wrong)
    with pytest.raises(ValueError, match="pages= says how many pages a sequence of a PAGED cache owns"):
        DecodeState(kinds, 3, 130, KV_SHAPES, "cpu", pages=3)
    without = DecodeState(("linear_attention",), 3, 130, KV_SHAPES, "cpu", page=PAGE)  # no attention layer: no pool, and no table to build
    assert without.table is None and without.bytes() == DecodeState(("linear_attention",), 3, 130, KV_SHAPES, "cpu").bytes()


def test_a_write_lands_on_the_page_and_the_row_of_its_position_and_nowhere_else():
    """``write(position, k, v)`` of sequence ``i`` lands at ``pool[i x pages + position // 64, :, position % 64]``, keys in the
    first half of the last axis and values in the second -- computed here, not by the class -- and every other byte of
    the pool keeps what it held: three sequences, positions 0 to 129 (two full pages and two rows of a third)."""
    plain, paged = _two_states(sequences=3, capacity=130)
    cache, pages, width = paged.layers[0], 3, 8
    generator = torch.Generator().manual_seed(3)
    canary = torch.randn(cache.pool.shape, generator=generator).to(torch.bfloat16)
    cache.pool.copy_(canary)
    expected = canary.clone()
    for position in range(130):
        k, v = (torch.randn((3, 2, width), generator=generator).to(torch.bfloat16) for _ in range(2))
        cache.write(position, k, v)
        for sequence in range(3):
            row = expected[sequence * pages + position // PAGE, :, position % PAGE]
            row[:, :width], row[:, width:] = k[sequence], v[sequence]
        assert _same(cache.pool, expected), f"position {position}: its row of every sequence, and no other byte"
    # the rows no position of a sequence names: the 62 last rows of every sequence's third page still hold the canary
    for sequence in range(3):
        assert _same(cache.pool[sequence * pages + 2, :, 2:], canary[sequence * pages + 2, :, 2:])
    # ... and the plain cache's write is the two statements it was: [sequences, kv heads, position, head width] of each tensor
    k, v = (torch.randn((3, 2, width), generator=generator).to(torch.bfloat16) for _ in range(2))
    before = [t.clone() for t in plain.layers[0].tensors()]
    plain.layers[0].write(77, k, v)
    for tensor, was, row in zip(plain.layers[0].tensors(), before, (k, v)):
        was[:, :, 77] = row
        assert _same(tensor, was)


@pytest.mark.parametrize("pages", [None, 5])
def test_the_keys_and_values_of_a_pool_are_those_of_the_plain_cache(pages):
    """``keys(n, dtype)`` / ``values(n, dtype)``: the first ``n`` positions of every sequence, ``[sequences, kv heads, n, head
    width]`` in ``dtype`` -- what the reference attention reads.  After the same writes the pool gives the plain cache's
    bytes, and both give the rows the test wrote, for a count inside the first page, at both sides of every page
    boundary and at the capacity, in bf16 (what ``standing()`` asks for) and in fp64 (what the chain accumulates in)."""
    plain, paged = _two_states(sequences=3, capacity=130, pages=pages)
    keys, values = _written(plain, paged, range(130))
    for n in (0, 1, 2, 63, 64, 65, 127, 128, 129, 130):
        for dtype in (torch.bfloat16, torch.float64):
            for name, wrote in (("keys", keys), ("values", values)):
                mine, theirs = getattr(paged.layers[0], name)(n, dtype), getattr(plain.layers[0], name)(n, dtype)
                assert tuple(mine.shape) == (3, 2, n, 8) and mine.dtype == dtype, f"{name}({n}, {dtype})"
                assert _same(mine, theirs) and _same(mine, wrote[:, :, :n].to(dtype)), f"{name}({n}, {dtype}): the rows written at positions 0 .. {n - 1}"
    # the plain cache's read of bf16 is a view of its tensor, as it was; the pool's is a copy (the pages of a sequence are not one run of positions)
    assert plain.layers[0].keys(70, torch.bfloat16).data_ptr() == plain.layers[0].k.data_ptr()
    assert plain.layers[0].values(70, torch.bfloat16).data_ptr() == plain.layers[0].v.data_ptr()
    copy = paged.layers[0].keys(70, torch.bfloat16)
    copy.zero_()
    assert _same(paged.layers[0].keys(70, torch.bfloat16), keys[:, :, :70]), "what a read returns is not the pool's memory"
    # ... also where the rows read ARE one run of the pool: one sequence, one head, a page of one position, one position read
    from model_samples.state import DecodeState

    least = DecodeState(("full_attention",), 1, 4, dict(KV_SHAPES, kv=(1, 8)), "cpu", page=1).layers[0]
    least.write(0, torch.ones((1, 1, 8), dtype=torch.bfloat16), torch.ones((1, 1, 8), dtype=torch.bfloat16))
    least.keys(1, torch.bfloat16).zero_()
    assert bool((least.pool[0] == 1).all()), "always a copy"


def test_the_two_reference_chains_write_and_read_either_layout():
    """The attention preparation and the attention core of the reference chain on a plain cache and on a paged one, positions
    0 to 66 at three sequences: the same queries, gates and attention outputs, byte for byte.  And what the preparation
    writes at its position is held against the in-projection's row itself -- the value rows are its value part, bit for
    bit, and the key rows are not (they are normed and rotated) -- so a chain that hands its cache the two the other way
    round fails here, on both layouts alike."""
    from model_samples.providers import AttentionParams, torch_ref

    heads, (kv_heads, width), (rotary, theta) = 4, KV_SHAPES["kv"], KV_SHAPES["rotary"]
    generator = torch.Generator().manual_seed(4)
    params = AttentionParams(heads, kv_heads, width, rotary, theta, *(0.02 * torch.randn(width, generator=generator).to(torch.bfloat16) for _ in range(2)))
    plain, paged = _two_states(sequences=3, capacity=67)
    for position in range(67):
        row = torch.randn((3, (2 * heads + 2 * kv_heads) * width), generator=generator).to(torch.bfloat16)  # [q | gate] per query head, the keys, the values
        outs = []
        for state in (plain, paged):
            cache = state.layers[0]
            q, gate = torch_ref.attention_prepare(row, cache, position, params, 1e-6, torch.float64)
            outs.append((q, gate, torch_ref.attention(q, cache, position, torch.float64)))
            written = cache.values(position + 1, torch.bfloat16)[:, :, position], cache.keys(position + 1, torch.bfloat16)[:, :, position]
            assert _same(written[0], row[:, (2 * heads + kv_heads) * width :].view(3, kv_heads, width)), f"position {position}: the value rows, a bit copy"
            assert not _same(written[1], row[:, 2 * heads * width : (2 * heads + kv_heads) * width].view(3, kv_heads, width)) and not _same(*written)
        assert all(_same(mine, theirs) for mine, theirs in zip(outs[1], outs[0])), f"position {position}: queries, gates and the attention output"
    assert _same(paged.layers[0].keys(67, torch.float64), plain.layers[0].keys(67, torch.float64))


def test_a_paged_cache_answers_what_the_plain_cache_answers():
    """Everything a check asks of a cache -- the sequences that own some rows of a step (``rows``), where a sequence lives
    (``of``), the cache as a decode step takes it (``first_token``), ``tensors``, and of the state ``snapshot``,
    ``restore``, ``clear``, ``standing`` and ``bytes`` -- against the plain cache on the same writes, at four tokens per
    sequence."""
    from model_samples.state import PagedKvCache

    tokens, pages, width = 4, 3, 8
    plain, paged = _two_states(sequences=3, capacity=130, tokens=tokens)
    keys, values = _written(plain, paged, range(100))
    flat, cache = plain.layers[0], paged.layers[0]
    assert cache.tokens == flat.tokens == tokens
    # rows: the rows 4 to 7 of a step are sequence 1's; rows 4 to 11 those of sequences 1 and 2
    for start, stop, sequences in ((4, 8, [1]), (4, 12, [1, 2]), (0, 4, [0]), (0, 12, [0, 1, 2])):
        window, theirs = cache.rows(start, stop), flat.rows(start, stop)
        assert isinstance(window, PagedKvCache) and window.pool is cache.pool and window.tokens == tokens, "the same pool: the window is of its sequences"
        for n in (1, 64, 65, 100):
            assert _same(window.keys(n, torch.float64), theirs.keys(n, torch.float64)) and _same(window.values(n, torch.bfloat16), theirs.v[:, :, :n])
            assert _same(window.keys(n, torch.bfloat16), keys[sequences, :, :n])
        (mine,) = window.tensors()  # the pages of its sequences, as views of the pool
        assert tuple(mine.shape) == (len(sequences) * pages, 2, PAGE, 16) and mine.data_ptr() == cache.pool[sequences[0] * pages].data_ptr()
        whole = (3 * pages, 2, PAGE, width)
        assert window.k_pool.data_ptr() == cache.pool.data_ptr() and tuple(window.v_pool.shape) == whole, "a kernel is handed the WHOLE pool"
    # a write through the window of sequence 1 is in the pool, at sequence 1's row of that position, and nowhere else
    k, v = (torch.full((1, 2, width), fill, dtype=torch.bfloat16) for fill in (2.5, -3.5))
    before = cache.pool.clone()
    cache.rows(4, 8).write(101, k, v)
    before[1 * pages + 101 // PAGE, :, 101 % PAGE, :width], before[1 * pages + 101 // PAGE, :, 101 % PAGE, width:] = k[0], v[0]
    assert _same(cache.pool, before), "the same memory"
    flat.rows(4, 8).write(101, k, v)
    # of: where sequence i lives in each tensor of tensors() -- its pages; in the plain cache its row of both tensors
    for sequence in range(3):
        (where,) = cache.of(sequence)
        mine = cache.tensors()[0][where]
        assert tuple(mine.shape) == (pages, 2, PAGE, 16) and mine.data_ptr() == cache.pool[sequence * pages].data_ptr()
        theirs = [tensor[at] for tensor, at in zip(flat.tensors(), flat.of(sequence))]
        for half, plain_rows in zip((mine[..., :width], mine[..., width:]), theirs):
            assert _same(half.transpose(0, 1).reshape(2, pages * PAGE, width)[:, :130], plain_rows), f"sequence {sequence}: its positions, page after page"
    # first_token: the same pool as a decode step takes it, with the table of one token per sequence
    first = cache.first_token()
    assert isinstance(first, PagedKvCache) and first.pool is cache.pool and (first.tokens, first.pages) == (1, pages)
    assert first.table is paged.table.first_token() and first.table.tokens == 1 and cache.table.tokens == tokens
    assert paged.decode_layers[0].pool is cache.pool and paged.decode_layers[0].tokens == 1
    alone = cache.rows(4, 8).first_token()  # of a window: the window's sequences, at one token each
    assert (alone.first, alone.sequences, alone.tokens) == (1, 1, 1) and _same(alone.keys(100, torch.bfloat16), keys[1:2, :, :100])
    assert cache.first_token().rows(1, 3).at(7)[2].tolist() == [1 * pages * PAGE + 7, 2 * pages * PAGE + 7], "rows of a decode step are sequences"
    assert _same(first.keys(102, torch.bfloat16), flat.first_token().keys(102, torch.bfloat16))
    # standing: the cached keys and values up to the position, whatever the layout
    for position in (0, 1, 63, 64, 65, 102, 130):
        plain.position = paged.position = position
        for mine, theirs in zip(paged.standing()[0], plain.standing()[0]):
            assert _same(mine, theirs) and tuple(mine.shape) == (3, 2, position, width)
    # snapshot, restore and clear go through tensors(): the pool is saved, written back and zeroed in place; the tables are constants
    table = paged.table
    constants = [t.clone() for t in (table.cos_sin, table.positions, table.slots, table.first_token().positions, table.first_token().slots)]
    plain.position = paged.position = 102
    snaps = plain.snapshot(), paged.snapshot()
    assert len(snaps[1]["layers"][0]) == 1 and _same(snaps[1]["layers"][0][0], cache.pool) and snaps[1]["layers"][0][0].data_ptr() != cache.pool.data_ptr()
    held = cache.pool.clone()
    _written(plain, paged, range(90, 130), seed=5)
    plain.position = paged.position = 130
    assert not _same(cache.pool, held)
    pool = cache.pool
    for state, snap in zip((plain, paged), snaps):
        state.restore(snap)
    assert paged.position == 102 and cache.pool is pool and _same(cache.pool, held), "written back into the existing allocation"
    for mine, theirs in zip(paged.standing()[0], plain.standing()[0]):
        assert _same(mine, theirs)
    paged.clear()
    assert paged.position == 0 and cache.pool is pool and not bool(cache.pool.any()) and paged.table is table
    now = (table.cos_sin, table.positions, table.slots, table.first_token().positions, table.first_token().slots)
    assert all(_same(tensor, was) for tensor, was in zip(now, constants)), "no call writes a table: not a write, not a restore, not a clear"
    assert table.tensors() == () and table.first_token().tensors() == ()
    # bytes: the pool and the tables
    tables = sum(t.numel() * t.element_size() for t in now)
    assert paged.bytes() == cache.pool.numel() * 2 + tables and plain.bytes() == 2 * flat.k.numel() * 2
    assert (
        cache.pool.numel() * 2 - plain.bytes() == 3 * 2 * (pages * PAGE - 130) * 2 * width * 2
    ), "the capacity rounded up to a page: 62 positions more per sequence"


def test_the_position_and_slot_tables_of_a_paged_state():
    """``PageTable``: what a kernel that prepares the operands of a paged attention reads beside the pool.  ``cos_sin``
    ``[capacity, rotary width]`` bf16, per position the cosines and then the sines, each row what the reference chain
    forms for that position (``torch_ref.rope_cos_sin``), bit for bit; ``positions`` int64 ``[capacity - tokens + 1, 3,
    rows]``, row ``p`` holding ``p + t`` for token ``t`` of every sequence, as three real rows; ``slots`` int64 ``[capacity
    - tokens + 1, rows]``: ``sequence x pages x 64 + p + t``, the flat index of that position's row in the pool.  Every value
    against that arithmetic written out here; ``at(position)`` hands out views of the tables."""
    import math

    from model_samples.providers import torch_ref
    from model_samples.state import PageTable

    sequences, tokens, capacity, pages = 3, 4, 130, 3
    plain, paged = _two_states(sequences, capacity, tokens)
    table, cache, rows = paged.table, paged.layers[0], sequences * tokens
    steps = capacity - tokens + 1  # a step starts at 0 .. capacity - tokens: no row beyond, whose last token would be past the capacity
    assert (table.tokens, tuple(table.positions.shape), tuple(table.slots.shape)) == (tokens, (steps, 3, rows), (steps, rows))
    assert table.positions.dtype == table.slots.dtype == torch.int64 and table.positions.is_contiguous() and table.slots.is_contiguous()
    assert table.positions.stride() == (3 * rows, rows, 1), "three REAL rows a position: no expanded axis"
    positions, slots = table.positions.tolist(), table.slots.tolist()
    for start in range(steps):
        for sequence in range(sequences):
            for token in range(tokens):
                row = sequence * tokens + token  # the rows of a step stand sequence by sequence
                assert [positions[start][axis][row] for axis in range(3)] == [start + token] * 3, f"step at {start}, sequence {sequence}, token {token}"
                assert slots[start][row] == sequence * pages * PAGE + start + token
        assert len(set(slots[start])) == rows, "the slots of a step are distinct: no two rows of a launch write one pool row"
    assert max(max(row) for step in positions for row in step) == capacity - 1, "every position is a row of the cos / sin table"
    assert max(max(step) for step in slots) == (sequences - 1) * pages * PAGE + capacity - 1 < cache.pool.shape[0] * PAGE
    # the slot of a position is the row the accessor writes: read back by the slot's own arithmetic, page x 64 + row
    keys, values = _written(plain, paged, range(capacity))
    for start in (0, 60, 63, 64, 126):
        for row, slot in enumerate(slots[start]):
            sequence, token = divmod(row, tokens)
            page, line = divmod(slot, PAGE)
            assert _same(cache.k_pool[page, :, line], keys[sequence, :, start + token]) and _same(
                cache.v_pool[page, :, line], values[sequence, :, start + token]
            )
    # the cos / sin table: the chain's own function at every position, bit for bit; cosines first, then sines
    width, theta = KV_SHAPES["rotary"]
    assert tuple(table.cos_sin.shape) == (capacity, width) and table.cos_sin.dtype == torch.bfloat16 and table.cos_sin.is_contiguous()
    for position in range(capacity):
        cos, sin = torch_ref.rope_cos_sin(position, width, theta, "cpu")
        assert _same(table.cos_sin[position], torch.cat((cos, sin))), f"position {position}"
        for index in range(width // 2):
            angle = position * theta ** (-2 * index / width)
            assert (
                abs(float(table.cos_sin[position, index]) - math.cos(angle)) < 2.0**-7
                and abs(float(table.cos_sin[position, width // 2 + index]) - math.sin(angle)) < 2.0**-7
            )
    # at(position): the rows of a step that starts there, as views inside the tables -- nothing is copied, nothing launched
    for start in (0, 63, 64, steps - 1):
        cos_sin, at, slot = cache.at(start)
        assert cos_sin is table.cos_sin and tuple(at.shape) == (3, rows) and tuple(slot.shape) == (rows,)
        assert at.data_ptr() == table.positions[start].data_ptr() and slot.data_ptr() == table.slots[start].data_ptr()
        assert at.dtype == slot.dtype == torch.int64 and at.stride(-1) == 1 and slot.is_contiguous() and at.data_ptr() % 8 == slot.data_ptr() % 8 == 0
    # a position no step starts at is refused: one past the last row, and a negative one, which would index the tables from their end
    for outside in (-1, -steps, steps, capacity):
        with pytest.raises(
            IndexError, match=f"position {outside} is not one a step of 4 tokens per sequence starts at: the tables hold {steps} such positions, from 0"
        ):
            cache.at(outside)
    with pytest.raises(IndexError, match=f"position -1 is not one a step of 1 token per sequence starts at: the tables hold {capacity} such positions, from 0"):
        cache.first_token().at(-1)
    # one sequence out of several (a check runs a sequence alone): the same table rows cut to its columns, the same slots, the whole pool
    for sequence in range(sequences):
        alone = cache.rows(sequence * tokens, (sequence + 1) * tokens)
        cos_sin, at, slot = alone.at(64)
        assert cos_sin is table.cos_sin and at.tolist() == [[64 + token for token in range(tokens)]] * 3
        assert slot.tolist() == [sequence * pages * PAGE + 64 + token for token in range(tokens)], "its own slots: the pool is the whole pool"
        assert at.data_ptr() == table.positions[64, 0, sequence * tokens :].data_ptr() and slot.data_ptr() == table.slots[64, sequence * tokens :].data_ptr()
        assert (
            at.stride(-1) == 1 and slot.is_contiguous() and at.data_ptr() % 8 == slot.data_ptr() % 8 == 0
        ), "aligned to their element, the token axis contiguous"
    # the decode steps of a prompt on a state of several tokens: the table of ONE token per sequence, on the same cos / sin table
    single = table.first_token()
    assert single.tokens == 1 and single.first_token() is single and single.cos_sin is table.cos_sin
    assert (tuple(single.positions.shape), tuple(single.slots.shape)) == ((capacity, 3, sequences), (capacity, sequences))
    for position in (0, 63, 64, capacity - 1):
        _, at, slot = cache.first_token().at(position)
        assert at.tolist() == [[position] * sequences] * 3 and slot.tolist() == [sequence * pages * PAGE + position for sequence in range(sequences)]
    one_token = _two_states(sequences, capacity)[1].table
    assert plain.table is None and one_token.first_token() is one_token, "at one token per sequence the table is its own table of one token"
    # a state too short for one step of its tokens has no row to start from
    assert tuple(_two_states(2, 3, tokens=4)[1].table.positions.shape) == (0, 3, 8)
    # the two things a kernel cannot check are asked of the integers a table is built from: every position has a row of the
    # cos / sin table, and a sequence's pages hold its positions (else two sequences would share a slot)
    with pytest.raises(ValueError, match="130 positions per sequence need a cos / sin table of as many rows; got 129"):
        PageTable.build(sequences, tokens, capacity, PAGE, pages, table.cos_sin[:129], "cpu")
    with pytest.raises(ValueError, match="130 positions per sequence need 3 pages of 64; got pages=2"):
        PageTable.build(sequences, tokens, capacity, PAGE, 2, table.cos_sin, "cpu")


def test_a_paged_cache_takes_the_tables_of_its_own_pool_and_no_others():
    """A slot is an index into ONE pool, and a row of the tables the rows of ONE step shape.  Tables built for other
    sequences, other pages per sequence, another page or other tokens per sequence would hand a kernel slots past the
    pool's end or inside another sequence's pages, and a kernel that is told a slot reads nothing back.  So a table
    states the integers it was built from and a cache refuses one that was not built for its pool, as it refuses a
    window that is not of the pool's sequences.  Everything a state builds itself is accepted."""
    from model_samples.state import DecodeState, PagedKvCache

    state = DecodeState(("full_attention",), 3, 130, KV_SHAPES, "cpu", tokens=4, page=PAGE)
    cache, table = state.layers[0], state.table
    assert (table.sequences, table.page, table.pages, table.tokens) == (3, PAGE, 3, 4), "what the tables were built from, as host integers"
    assert (table.first_token().sequences, table.first_token().page, table.first_token().pages, table.first_token().tokens) == (3, PAGE, 3, 1)
    # what a state builds: its cache, the cache of its decode steps, a window of some sequences and of none, a window's decode cache
    for built in (cache, cache.first_token(), cache.rows(4, 12), cache.rows(8, 8), cache.rows(12, 12), cache.rows(4, 8).first_token()):
        again = PagedKvCache(built.pool, built.pages, built.table, built.tokens, built.first, built.sequences)
        assert (again.first, again.sequences) == (built.first, built.sequences) and again.pool is cache.pool
    assert (
        PagedKvCache(cache.pool, 3, table, 4).sequences == 3 and PagedKvCache(cache.pool, 3, table, 4, 1).sequences == 2
    ), "left out: the pool's sequences from the first"

    def pool(pages, page=PAGE):
        return torch.zeros((pages, 2, page, 16), dtype=torch.bfloat16)

    mine = "these are for a pool of 9 pages (3 a sequence) of 64 positions, tokens per sequence 4; the cache has a pool of "
    planted = {  # each differs from the state's own cache in ONE thing; the last is what a cache built for two pages a sequence would be
        "the pool of two sequences": (lambda: PagedKvCache(pool(6), 3, table, 4), "6 pages (3 a sequence) of 64 positions, tokens per sequence 4"),
        "other pages per sequence": (lambda: PagedKvCache(cache.pool, 9, table, 4), "9 pages (9 a sequence) of 64 positions, tokens per sequence 4"),
        "another page": (lambda: PagedKvCache(pool(9, 16), 3, table, 4), "9 pages (3 a sequence) of 16 positions, tokens per sequence 4"),
        "the table of four tokens at one": (lambda: PagedKvCache(cache.pool, 3, table, 1), "9 pages (3 a sequence) of 64 positions, tokens per sequence 1"),
        "a pool of two pages a sequence": (lambda: PagedKvCache(pool(6), 2, table, 4), "6 pages (2 a sequence) of 64 positions, tokens per sequence 4"),
    }
    for what, (build, theirs) in planted.items():
        with pytest.raises(ValueError) as error:
            build()
        assert str(error.value) == "the tables of a paged cache are built for its pool: " + mine + theirs, what
    with pytest.raises(ValueError, match="tokens per sequence 1; the cache has a pool of 9 pages .3 a sequence. of 64 positions, tokens per sequence 4"):
        PagedKvCache(cache.pool, 3, table.first_token(), 4)  # ... and the table of one token is not the table of a step of four
    # the window: sequences of the pool, and no others
    for first, sequences in ((2, 2), (4, 0), (-1, 1), (1, -1)):
        with pytest.raises(ValueError, match=f"a paged cache stands for sequences of its pool, which holds 3; got first={first}, sequences={sequences}"):
            PagedKvCache(cache.pool, 3, table, 4, first, sequences)
    with pytest.raises(ValueError, match="which holds 3; got first=2, sequences=2"):
        cache.rows(8, 16)  # the rows 8 to 15 of a step would be sequences 2 and 3


def test_one_provider_asks_for_a_paged_cache_and_every_other_plan_builds_the_plain_one():
    """Two adapters of the closed list name a page, and the same one: the attention preparation's entry, whose kernel writes a
    pool of pages, and the library's paged decode attention, whose kernel reads that pool.  So a plan builds a pool where
    the preparation's row serves -- the two target stacks on compute capability 10.x -- and no plan of another stack, GPU
    class or row count does; the area README says so.  The next adapter that names a page fails this case: its row
    changes what the runs of its stacks hold, and the README's sentence.  Two more sentences of
    that README are held here to what they are about: the page of the target kernel set, which is the family's constant
    and the page of every case of this group; and where the GPU cell of the paged cache runs."""
    from model_samples import Session, TextIO, providers
    from model_samples.qwen_dense import QwenDense, configs

    asking = {key: factory.cache_page for key, factory in providers.factories().items() if getattr(factory, "cache_page", None) is not None}
    assert providers.BlockBase.cache_page is None and asking == {
        "fe.attention_prepare_decode": PAGE,
        "flashinfer.decode_attention": PAGE,
    }, "the adapter whose kernel writes a paged pool, and the adapter whose kernel reads it"
    model = _tiny_model()
    for more in ({}, {"tokens": 4}, {"draft_tokens": 3}):
        assert Session(model, TextIO.synthetic(model.geometry.vocab), sequences=2, stack="reference", device="cpu", capacity=16, **more).cache_page() is None
    for stack, cc, sequences, page in (
        ("fe-entries", (10, 0), 1, PAGE),
        ("fe-entries", (10, 7), 4, PAGE),
        ("fe-entries", (10, 3), 1024, PAGE),
        ("fe-entries", (10, 0), 1025, None),  # past the rows of the entry: the chain, on the cache it had
        ("fe-entries", (12, 0), 1, None),
        ("fe-cublas", (10, 0), 1, None),
        ("reference", (10, 0), 1, None),
    ):
        planned = _plan(cc, sequences, stack, layers=2)
        assert {getattr(block, "cache_page", None) for block in planned.values()} - {None} == ({page} if page else set()), (stack, cc, sequences)
    assert QwenDense.attention_page == configs.ATTENTION_PAGE == dict(configs.PREPARE_FORM)["PAGE_SIZE"], "the page of the row is the family's constant"
    readme = " ".join((MODELS / "README.md").read_text(encoding="utf-8").split())
    assert (
        "a session builds the pool where a block of its plans asks for one: the row of the attention preparation's entry does, so a run of the stacks "
        "`native` and `fe-entries` holds the pool on compute capability 10.x, up to 1,024 rows, and every other run the first layout"
    ) in readme
    assert PAGE == configs.ATTENTION_PAGE and f"(a fixed number of positions each, {PAGE} in the target kernel set; a run of pages per sequence)" in readme
    # the cell is gated: the classes the family serves and compute capability 8.0 (the GPU module holds the condition on stated capabilities)
    assert (
        "one GPU the family serves; L0 runs two layers, L1 four; the cell of the paged cache runs there and on compute capability 8.0, "
        "and is skipped by name on any other class |"
    ) in readme


@contextlib.contextmanager
def _a_block_asks_for_a_page(page=PAGE, key="torch_ref.attention_prepare", tables=False):
    """Inside: a plan that is resolved serves the block of ``key`` with a stand-in that runs the reference chain under another
    type and asks for an attention cache in pages of ``page`` positions (``Block.cache_page``) -- what the adapter of a
    kernel that writes a paged pool states.  ``tables``: the stand-in also says that it reads the state's page table and
    lengths (``Block.attended_tables``), as the adapter of a kernel that reads such a pool does.  Yields the stand-in's class."""
    from model_samples import providers

    registry = providers.factories
    served = registry()[key]

    class AsksForAPage(served):
        cache_page = page
        attended_tables = tables

    providers.factories = lambda: dict(registry(), **{key: AsksForAPage})
    try:
        yield AsksForAPage
    finally:
        providers.factories = registry


def _decoded(session, prompts, steps):
    """A prompt and ``steps`` decode steps: ``(the ids of every step, the logits of every step, what the state stands on)``."""
    ids = session.prefill(prompts)
    chosen = [session.read(ids)]
    for _ in range(steps):
        ids = session.step(ids)
        chosen.append(session.read(ids))
    standing = [tuple(t.clone() for t in layer) for layer in session.state.standing()]
    return chosen, session.record.logits[: session.record.count].clone(), standing


@pytest.mark.parametrize("sequences", [1, 3])
def test_a_session_on_a_paged_cache_is_the_session_on_the_plain_one_byte_for_byte(sequences):
    """Two sessions of the reference stack on one load: in one the attention preparation is a stand-in that asks for a cache
    in pages of 64 positions, so its state is paged; the other's is plain.  A prompt longer than a page and decode steps
    up to the last position of the last page: equal ids, equal logits bytes, equal ``standing()`` bytes.  And a state
    zeroed in place gives the run again."""
    import model_samples
    from model_samples.providers import torch_ref
    from model_samples.state import KvCache, PagedKvCache

    model = _paged_model()
    text = model_samples.TextIO.synthetic(model.geometry.vocab)
    common = dict(sequences=sequences, stack="reference", device="cpu", capacity=128)
    plain = model_samples.Session(model, text, **common)
    paged = model_samples.Session(model, text, share=plain, **common)
    with _a_block_asks_for_a_page() as stand_in:
        paged.resolve()
    assert type(paged.plan["attention.prepare"]) is stand_in and type(plain.resolve()["attention.prepare"]) is not stand_in
    assert (plain.cache_page(), paged.cache_page()) == (None, PAGE), "a session asks its own plans: no block of the plain one names a page"
    assert plain.plan_table([paged]) == plain.plan.table(), "no memory sentence on the CPU"
    plain.load()
    paged.load()
    assert isinstance(plain.state.layers[1], KvCache) and isinstance(paged.state.layers[1], PagedKvCache) and paged.state.pages == 2
    g, table = model.geometry, paged.state.table  # the cos / sin table is built from the model's rotary width and theta, by the chain's function
    assert all(_same(table.cos_sin[p], torch.cat(torch_ref.rope_cos_sin(p, g.rotary_dim, g.rope_theta, "cpu"))) for p in (0, 1, 63, 64, 127))
    assert paged.weights is plain.weights and paged.state is not plain.state, "one load, two states: a pool is never shared"
    generator = torch.Generator().manual_seed(1)
    prompts = [torch.randint(0, model.geometry.vocab, (70,), generator=generator) for _ in range(sequences)]  # a prompt longer than one page
    theirs = _decoded(plain, prompts, 58)
    mine = _decoded(paged, prompts, 58)
    assert paged.state.position == plain.state.position == 128, "the last step wrote position 127: the last row of the last page of every sequence"
    assert mine[0] == theirs[0] and _same(mine[1], theirs[1]), "equal ids and equal logits bytes at every step, also across positions 63 and 64"
    for layer, (own, other) in enumerate(zip(mine[2], theirs[2])):
        assert all(_same(a, b) for a, b in zip(own, other)), f"layer {layer}: what the state stands on"
    pool = paged.state.layers[1].pool
    assert bool(pool[-1, :, -1].any()) and bool(pool[0, :, 0].any()), "the pool's first row and its very last one were written"
    with pytest.raises(ValueError, match="built for 128 positions per sequence"):  # a step past the capacity, as on the plain cache
        paged.step(torch.tensor(mine[0][-1], dtype=torch.int32))
    # zeroed in place: the same pool, the same tables, and the same run
    table = paged.state.table
    paged.reset(fresh=False)
    assert paged.state.layers[1].pool is pool and paged.state.table is table and not bool(pool.any()) and paged.state.position == 0
    again = _decoded(paged, prompts, 58)
    assert again[0] == mine[0] and _same(again[1], mine[1])
    paged.reset(fresh=True)
    assert paged.state.layers[1].pool is not pool and isinstance(paged.state.layers[1], PagedKvCache), "a fresh state is built in the session's layout"


def _judged_run(defect=None):
    """Three sequences of the reference stack whose attention preparation asks for a paged cache and stands in a row with
    a band (``bitwise``), a prompt of 60 tokens and 12 decode steps -- across positions 63 / 64 -- and the checks of that
    run.  ``defect(cache, position)`` runs behind every call of the stand-in."""
    import dataclasses

    import model_samples
    from model_samples import tie_out
    from model_samples.state import PagedKvCache

    model = _paged_model()
    *_, row = model.provider_table["attention.prepare"]  # the reference chain's row: the last one of the block
    model.provider_table = dict(model.provider_table, **{"attention.prepare": [dataclasses.replace(row, band="bitwise")]})
    text = model_samples.TextIO.synthetic(model.geometry.vocab)
    session = model_samples.Session(model, text, sequences=3, stack="reference", device="cpu", capacity=80)
    with _a_block_asks_for_a_page() as stand_in:
        session.plan_table()
    if defect is not None:
        chain = stand_in.__call__

        def wrong(self, qkv, cache, position, params):
            out = chain(self, qkv, cache, position, params)
            defect(cache, position)
            return out

        stand_in.__call__ = wrong
    session.load()
    generator = torch.Generator().manual_seed(2)
    _decoded(session, [torch.randint(0, model.geometry.vocab, (60,), generator=generator) for _ in range(3)], 12)
    assert session.state.position == 72 and isinstance(session.state.layers[1], PagedKvCache) and session.state.pages == 2
    return tie_out.run_checks(session, prompt_tokens=60)


def test_the_checks_of_a_run_on_a_paged_cache_judge_the_block_that_asked_for_it():
    """The checks of a run whose attention preparation asks for a paged cache, over steps that cross positions 63 / 64 at
    three sequences.  The stand-in is another type than the reference chain and its row names a band, so it is judged:
    the reference block runs on the provider's own cache object, through the accessors, and the pool it leaves is
    compared byte for byte.  Every sequence is also run alone, on its window of the pool (``rows``, ``of``).  A stand-in
    that leaves the key rows of a position one row late in their page fails by the state."""
    checks = _judged_run()
    line = next(line for line in checks.lines if line.split()[:1] == ["attention.prepare"])
    assert checks.ok and "1 of 1 provider blocks inside their bands" in checks.text(), checks.text()
    assert "  72 calls  " in line and "STATE DIFFERS" not in line and line.endswith("[band: bitwise]  PASS"), line
    assert "one sequence alone vs among 3 (step t: sequence t mod 3)" in checks.text() and "13 of 13 blocks equal" in checks.text(), checks.text()
    assert checks.lines[1].endswith("byte-identical on a second run from a fresh state    PASS")

    def one_row_late(cache, position):
        keys = cache.k_pool.view(-1, cache.pages, *cache.k_pool.shape[1:])[:, position // PAGE]  # the page of the position, of every sequence
        keys[:, :, (position + 1) % PAGE] = keys[:, :, position % PAGE].clone()
        keys[:, :, position % PAGE] = 0

    failed = _judged_run(one_row_late)
    line = next(line for line in failed.lines if line.split()[:1] == ["attention.prepare"])
    assert not failed.ok and "STATE DIFFERS" in line and line.endswith("[band: bitwise]  FAIL"), failed.text()


def test_two_layouts_in_one_session_are_a_wrong_table_and_the_memory_sentence_counts_the_pool_and_the_tables(monkeypatch):
    """A session has one state, so the blocks of its plans ask for one layout of its attention cache: two pages -- inside a
    plan, or in the two plans of a session that verifies behind its prompt -- are a ``ValueError`` that names the blocks
    and their plans, before anything is built; so is a page that is no whole number of positions from 1; and of a block
    that several rows serve by call site every part is asked.  And what a paged state holds is what the memory sentence
    counts: the pool (the capacity rounded up to whole pages) and the tables, megabytes at the published widths."""
    import copy

    import model_samples
    from model_samples import blocks
    from model_samples.qwen_dense import QwenDense
    from model_samples.state import DecodeState

    tiny = _paged_model()
    text = model_samples.TextIO.synthetic(tiny.geometry.vocab)
    common = dict(sequences=2, stack="reference", device="cpu", capacity=40)
    both = model_samples.Session(tiny, text, **common)
    with _a_block_asks_for_a_page(64), _a_block_asks_for_a_page(16, "torch_ref.attention"):
        both.resolve()
    sentence = (
        "the provider table is wrong: the blocks of one session ask for two layouts of its attention cache -- attention.prepare in the plan of 2 "
        "sequences asks for pages of 64 positions, attention.core in the plan of 2 sequences for pages of 16; a session has one state, in one layout"
    )
    for refused in (both.cache_page, both.plan_table, both.load):
        with pytest.raises(ValueError) as error:
            refused()
        assert str(error.value) == sentence
    assert both.state is None and both.weights is None, "named before anything is read or built"
    first = model_samples.Session(tiny, text, **common)
    with pytest.raises(ValueError) as error:  # ... also for a session that runs behind another one on its weights: the first plans it
        first.plan_table([both])
    assert str(error.value) == sentence and first.cache_page() is None
    behind = model_samples.Session(tiny, text, draft_tokens=3, **common)
    with _a_block_asks_for_a_page(64):
        behind.plan = behind._resolve(behind.tokens)
    with _a_block_asks_for_a_page(16):
        behind.prefill_plan = behind._resolve(1, state_tokens=behind.tokens)
    with pytest.raises(ValueError) as error:
        behind.plan_table()
    assert "attention.prepare in the plan of 2 sequences asks for pages of 16 positions" in str(error.value), "the prompt's decode plan"
    assert "attention.prepare in the plan of 2 sequences x 4 tokens per sequence for pages of 64" in str(error.value), "the verify plan"
    same = model_samples.Session(tiny, text, draft_tokens=3, **common)
    with _a_block_asks_for_a_page(64):
        same.plan = same._resolve(same.tokens)
    assert same.resolve() is same.plan and same.cache_page() == 64, "a block that names no page reads either layout: the prompt's plan asks for nothing"
    with _a_block_asks_for_a_page(64), _a_block_asks_for_a_page(64, "torch_ref.attention"):
        assert model_samples.Session(tiny, text, **common).cache_page() == 64, "two blocks, one page"
    # a page that is no whole number of positions from 1 is a wrong table too, named at the same places: before anything is read
    for wrong in (0, -64, True, 64.0, "64"):
        named = model_samples.Session(tiny, text, **common)
        with _a_block_asks_for_a_page(wrong):
            named.resolve()
        for refused in (named.cache_page, named.plan_table, named.load):
            with pytest.raises(ValueError) as error:
                refused()
            assert str(error.value) == (
                f"the provider table is wrong: attention.prepare in the plan of 2 sequences asks for pages of {wrong!r} positions; "
                "the page of an attention cache is a whole number of positions from 1"
            )
        assert named.state is None and named.weights is None, f"cache_page = {wrong!r}: named before anything is read or built"
    # a block whose calls two rows serve by call site (``blocks.BySite``) names no page itself: every part of it is asked
    split = model_samples.Session(tiny, text, **common)
    with _a_block_asks_for_a_page(64):
        plan = split.resolve()
    row, chain = plan.rows["attention.core"], plan["attention.core"]
    part = copy.copy(chain)
    part.cache_page = 16
    plan["attention.core"] = blocks.BySite("attention.core", [("a", row, chain), ("b", row, part)])
    assert plan["attention.core"].cache_page is None and chain.cache_page is None, "the holder and the part of the call site a name none"
    with pytest.raises(
        ValueError, match="attention.prepare in the plan of 2 sequences asks for pages of 64 positions, attention.core at b in the plan of 2 sequences for"
    ):
        split.cache_page()
    part.cache_page = 64
    assert split.cache_page() == 64, "the part of the call site b asks for the page the preparation asks for"
    # the arithmetic of the memory sentence is the state's own size, at every tokens count, capacity and pages per sequence
    for sequences, tokens, capacity, pages in ((1, 1, 40, None), (3, 1, 130, None), (2, 4, 130, None), (3, 6, 64, None), (2, 1, 130, 13), (1, 5, 3, None)):
        state = DecodeState(tiny.geometry.layer_kinds, sequences, capacity, tiny.state_shapes(), "cpu", tokens, page=PAGE, pages=pages)
        assert state.bytes() == tiny.state_bytes(sequences, capacity, tokens, page=PAGE, pages=pages), (sequences, tokens, capacity, pages)
        assert (
            tiny.state_bytes(sequences, capacity, tokens)
            == DecodeState(tiny.geometry.layer_kinds, sequences, capacity, tiny.state_shapes(), "cpu", tokens).bytes()
        )
    assert tiny.state_shapes()["rotary"] == (tiny.geometry.rotary_dim, tiny.geometry.rope_theta)
    linear = _paged_model(("linear_attention", "linear_attention"))  # no attention layer: no pool, and no table to count
    built = DecodeState(linear.geometry.layer_kinds, 3, 130, linear.state_shapes(), "cpu", page=PAGE)
    assert built.bytes() == linear.state_bytes(3, 130, page=PAGE) == linear.state_bytes(3, 130)
    # at the published widths: 16 attention layers, 4 key / value heads of 256, a context of 2,051 positions = 33 pages of 64
    model = QwenDense()
    plain, paged = model.state_bytes(16, 2051, 6), model.state_bytes(16, 2051, 6, page=PAGE)
    pool = (
        16 * 16 * 33 * 4 * PAGE * 512 * 2 - 16 * 2 * 16 * 4 * 2051 * 256 * 2
    )  # per layer one pool of 33 pages a sequence, where two tensors of 2,051 positions were
    tables = 2051 * 64 * 2 + (2051 - 6 + 1) * 96 * 32 + 2051 * 16 * 32  # cos / sin; positions and slots of a step of 6 tokens; and of the prompt's decode steps
    assert (
        paged - plain == pool + tables and pool == 16 * 16 * 4 * 61 * 512 * 2 and 7.0e6 < tables < 8.0e6
    ), "61 positions more a sequence and layer; megabytes of tables"
    assert model.memory_bytes(16, 2051, tokens=6, page=PAGE) - model.memory_bytes(16, 2051, tokens=6) == pool + tables
    assert (
        model.state_bytes(16, 2051, 6, page=PAGE, pages=39) - paged == 16 * 16 * 6 * 4 * PAGE * 512 * 2
    ), "39 pages a sequence where 33 hold the capacity: six pages more a sequence"
    # ... and the sentence of a session whose plan asks for a page counts it, for the session and for the sessions behind it
    monkeypatch.setattr(blocks.DeviceFacts, "probe", classmethod(lambda cls, device: cls((10, 0), "stated device", 148, 0, 200 * GIB)))
    text = model_samples.TextIO.synthetic(model.geometry.vocab)
    stated = dict(sequences=64, stack="reference", device="cuda:0", capacity=2051)

    def need(first_paged, later_paged):
        first = model_samples.Session(model, text, **stated)
        later = model_samples.Session(model, text, share=first, tokens=2, **stated)
        for session, asked in ((first, first_paged), (later, later_paged)):
            if asked:
                with _a_block_asks_for_a_page():
                    session.resolve()
        return first.plan_table([later]).splitlines()[-1]

    flat = model.memory_bytes(64, 2051) + model.state_bytes(64, 2051, 2)
    assert (
        need(False, False)
        == f"  this stack needs about {flat / GIB:.1f} GiB on this GPU for 64 and 64 sequences, one count after the other; 200.0 GiB are free"
    )
    first = model.memory_bytes(64, 2051, page=PAGE) + model.state_bytes(64, 2051, 2)
    assert (
        need(True, False)
        == f"  this stack needs about {first / GIB:.1f} GiB on this GPU for 64 and 64 sequences, one count after the other; 200.0 GiB are free"
    )
    both_paged = model.memory_bytes(64, 2051, page=PAGE) + model.state_bytes(64, 2051, 2, page=PAGE)
    assert (
        need(True, True)
        == f"  this stack needs about {both_paged / GIB:.1f} GiB on this GPU for 64 and 64 sequences, one count after the other; 200.0 GiB are free"
    )
    assert f"{flat / GIB:.1f}" != f"{first / GIB:.1f}" != f"{both_paged / GIB:.1f}", "the three sentences state three figures"


# The run's maximum length (``Session(max_model_len=)``) and the two tables a kernel that READS a paged attention cache is
# handed beside the pool: the page table of every sequence and the positions it holds after a step (``state.PageTable.blocks``
# / ``lengths``, ``PagedKvCache.attended``).  Every value of a table below is arithmetic written in the test.


def _whole(value, unit):
    """``value`` in whole units, rounded up."""
    return -(-value // unit)


#: (sequences, tokens per sequence, capacity, pages per sequence or None for as many as hold the capacity, the run's maximum
#: length, the entries of one sequence's row of the page table)
TABLE_CASES = (
    (1, 6, 62, None, 2051, 36),  # one own page under a maximum of 33 pages: every column of the row names that page
    (3, 1, 100, None, 2051, 36),  # two own pages, and 34 columns that name the second
    (1, 1, 2051, None, 2051, 36),  # the capacity is the maximum: 33 own pages, and three columns up to a multiple of four
    (2, 4, 130, 40, 2051, 40),  # more pages a sequence than the maximum length needs: the pages are the width
    (3, 4, 64, None, 64, 4),  # a maximum of one page: a row of four entries
)


def test_a_session_holds_the_runs_maximum_length_and_hands_it_to_every_block_of_its_plans():
    """``Session(max_model_len=)`` is the positions a sequence of the run may reach at most.  Left out it is the session's
    capacity; below the capacity it is a ``ValueError`` that names both; so is a value that is no whole number from 1.
    Every block of every plan of the session is handed it in its options."""
    from model_samples import Session, TextIO

    model = _tiny_model()
    text = TextIO.synthetic(model.geometry.vocab)
    common = dict(stack="reference", device="cpu", capacity=40)
    assert Session(model, text, **common).max_model_len == 40, "left out: the capacity"
    assert Session(model, text, max_model_len=40, **common).max_model_len == 40, "a maximum that is the capacity"
    session = Session(model, text, draft_tokens=3, max_model_len=2051, **common)
    session.resolve()
    handed = [block.options["max_model_len"] for plan in (session.plan, session.prefill_plan) for *_, block in plan.parts()]
    assert session.max_model_len == 2051 and handed and set(handed) == {2051}, "the blocks of the verify plan and of the prompt's"
    with pytest.raises(ValueError, match="max_model_len") as below:
        Session(model, text, max_model_len=39, **common)
    assert "39" in str(below.value) and "40" in str(below.value), "the maximum and the capacity are both named"
    for wrong in (0, -2051, 2051.0, True, "2051"):
        with pytest.raises(ValueError, match="max_model_len"):
            Session(model, text, max_model_len=wrong, **common)


def test_the_page_table_and_the_lengths_of_a_paged_state():
    """What a kernel that reads a paged cache is handed beside the pool, built with a state that is given the run's
    maximum length (``DecodeState(max_model_len=)``).  ``blocks`` int32 ``[sequences + 1, width]``: row ``i`` names the pages
    of sequence ``i``, ``i x pages + j`` in column ``j``, every column past its own pages its LAST own page, and the last
    row page 0 in every column; the width is the pages of the maximum length, or the sequence's own where those are
    more, rounded up to four.  ``lengths`` int32 ``[capacity - tokens + 1, sequences rounded up to four]``: row ``p`` holds
    ``p + tokens``, the positions a sequence holds AFTER the step that starts at ``p``.  A state of several tokens per
    sequence also holds the lengths of one token, for the decode steps of its prompt, on the same page table.  Every
    row of either table starts at a 16-byte boundary."""
    from model_samples.state import DecodeState

    for sequences, tokens, capacity, pages, longest, width in TABLE_CASES:
        state = DecodeState(("full_attention",), sequences, capacity, KV_SHAPES, "cpu", tokens, page=PAGE, pages=pages, max_model_len=longest)
        case = f"{sequences} x {tokens}, capacity {capacity}, pages {pages}, maximum length {longest}"
        own = _whole(capacity, PAGE) if pages is None else pages
        assert state.pages == own and width == 4 * _whole(max(own, _whole(longest, PAGE)), 4), case
        table = state.table
        assert tuple(table.blocks.shape) == (sequences + 1, width) and table.blocks.dtype == torch.int32 and table.blocks.is_contiguous(), case
        entries = table.blocks.tolist()
        for sequence in range(sequences):
            for column in range(width):
                assert entries[sequence][column] == sequence * own + min(column, own - 1), f"{case}: sequence {sequence}, column {column}"
        assert entries[sequences] == [0] * width, f"{case}: the row behind the last sequence"
        assert max(max(row) for row in entries) < state.layers[0].pool.shape[0], f"{case}: every entry names a page of the pool"
        assert all(table.blocks[row].data_ptr() % 16 == 0 for row in range(sequences + 1)), case
        for each, lengths in ((tokens, table.lengths), (1, table.first_token().lengths)):
            steps, columns = capacity - each + 1, 4 * _whole(sequences, 4)  # no row for a step whose last token would lie past the capacity
            assert tuple(lengths.shape) == (steps, columns) and lengths.dtype == torch.int32 and lengths.is_contiguous(), case
            assert lengths.tolist() == [[start + each] * columns for start in range(steps)], f"{case}: the positions held after a step of {each}"
            assert all(lengths[start].data_ptr() % 16 == 0 for start in range(steps)), case
        assert table.first_token().blocks is table.blocks, f"{case}: one page table for both kinds of step"


def test_a_paged_cache_hands_out_its_rows_of_the_two_tables_as_views():
    """``PagedKvCache.attended(position)`` -> the page-table rows of this object's sequences and the positions each of them
    holds after the step that starts at ``position``: views inside the state's two tables -- the same memory, contiguous,
    at a 16-byte boundary -- for every sequence of the pool and for a window of them (a check runs some sequences
    alone), on a state of several tokens per sequence and for the decode steps of its prompt.  A position no step starts
    at is the ``IndexError`` of ``at``.  A state that was not given a maximum length has no such table to hand out, and
    one is given only to a paged state, as a whole number that holds the capacity."""
    from model_samples.state import DecodeState

    sequences, tokens, capacity, pages, width = 3, 4, 130, 3, 36
    state = DecodeState(("full_attention",), sequences, capacity, KV_SHAPES, "cpu", tokens, page=PAGE, max_model_len=2051)
    cache, table = state.layers[0], state.table
    for held, first, count in ((cache, 0, sequences), (cache.rows(1 * tokens, 3 * tokens), 1, 2)):  # every sequence; sequences 1 and 2 alone
        for each, view, source in ((tokens, held, table.lengths), (1, held.first_token(), table.first_token().lengths)):
            for position in (0, 63, 64, capacity - each):
                blocks, lengths = view.attended(position)
                where = f"sequences {first} to {first + count - 1}, a step of {each} at {position}"
                assert blocks.tolist() == [[i * pages + min(j, pages - 1) for j in range(width)] for i in range(first, first + count)], where
                assert lengths.tolist() == [position + each] * count, where
                assert (tuple(blocks.shape), tuple(lengths.shape)) == ((count, width), (count,)), where
                for got, inside in ((blocks, table.blocks[first]), (lengths, source[position])):
                    assert got.data_ptr() == inside.data_ptr() and got.untyped_storage().data_ptr() == inside.untyped_storage().data_ptr(), where
                    assert got.dtype == torch.int32 and got.is_contiguous() and got.data_ptr() % 16 == 0, where
    for outside in (-1, capacity - tokens + 1, capacity):  # a position no step starts at: the message ``at`` raises for it
        with pytest.raises(IndexError) as of_at:
            cache.at(outside)
        with pytest.raises(IndexError) as refused:
            cache.attended(outside)
        assert str(refused.value) == str(of_at.value) and str(outside) in str(refused.value), outside
    without = DecodeState(("full_attention",), sequences, capacity, KV_SHAPES, "cpu", tokens, page=PAGE)
    assert without.table.blocks is None and without.table.lengths is None and without.table.first_token().lengths is None
    with pytest.raises(ValueError, match="max_model_len"):
        without.layers[0].attended(0)
    with pytest.raises(ValueError, match="max_model_len"):  # the two tables are tables of a pool of pages
        DecodeState(("full_attention",), sequences, capacity, KV_SHAPES, "cpu", tokens, max_model_len=2051)
    with pytest.raises(ValueError, match="max_model_len") as short:  # a length past the maximum would be handed to a kernel that was told the maximum
        DecodeState(("full_attention",), sequences, capacity, KV_SHAPES, "cpu", tokens, page=PAGE, max_model_len=capacity - 1)
    assert str(capacity) in str(short.value) and str(capacity - 1) in str(short.value)
    for wrong in (0, -64, 2051.0, True, "2051"):
        with pytest.raises(ValueError, match="max_model_len"):
            DecodeState(("full_attention",), sequences, capacity, KV_SHAPES, "cpu", tokens, page=PAGE, max_model_len=wrong)


def test_a_state_holds_the_two_tables_only_where_a_block_of_its_sessions_plans_reads_them(monkeypatch):
    """One adapter of the closed list reads the page table and the lengths (``Block.attended_tables``): the library's paged
    decode attention.  A state of a run without such a block holds neither: its tables, its bytes and the memory
    sentence of its run do not follow the run's maximum length.  A session whose plans have a block that says so -- in its step's plan or in its prompt's, a part of a block that
    several rows serve included -- builds its state with both, for the session's maximum length; the family's
    arithmetic counts them to the byte and the memory sentence holds that count.  Such a block in a session whose
    blocks ask for no pool of pages is a wrong table."""
    import copy

    from model_samples import Session, TextIO, blocks, providers
    from model_samples.qwen_dense import QwenDense
    from model_samples.state import DecodeState

    reading = [key for key, factory in providers.factories().items() if getattr(factory, "attended_tables", False)]
    assert providers.BlockBase.attended_tables is False and blocks.BySite.attended_tables is False and reading == ["flashinfer.decode_attention"]
    tiny = _paged_model()
    text = TextIO.synthetic(tiny.geometry.vocab)
    kinds, shapes = tiny.geometry.layer_kinds, tiny.state_shapes()
    common = dict(sequences=3, stack="reference", device="cpu", capacity=100)

    def built(tables, core="torch_ref.attention", **session):
        made = Session(tiny, text, **common, **session)
        with _a_block_asks_for_a_page(), _a_block_asks_for_a_page(key=core, tables=tables):
            made.resolve()
        made.reset(fresh=True)
        return made

    paged = built(False, max_model_len=2051)
    base = DecodeState(kinds, 3, 100, shapes, "cpu", page=PAGE)
    assert not paged.attended_tables() and paged.state.table.blocks is None and paged.state.table.lengths is None
    assert paged.state.bytes() == base.bytes() == tiny.state_bytes(3, 100, page=PAGE), "a paged state without the tables, whatever the maximum length"
    read = built(True, max_model_len=2051)
    table = read.state.table
    assert read.attended_tables() and (tuple(table.blocks.shape), tuple(table.lengths.shape)) == ((4, 36), (100, 4))
    assert read.state.bytes() - base.bytes() == 4 * 36 * 4 + 100 * 4 * 4, "four bytes an entry of either table"
    assert read.state.bytes() == tiny.state_bytes(3, 100, page=PAGE, max_model_len=2051)
    assert tuple(built(True).state.table.blocks.shape) == (4, 4), "a session that names no maximum length: the pages of its capacity"
    # a session that verifies behind its prompt has two plans: a block of either one that reads the tables builds them
    for reads in ("plan", "prefill_plan"):
        behind = Session(tiny, text, draft_tokens=3, max_model_len=2051, **common)
        with _a_block_asks_for_a_page():
            behind.resolve()
        assert behind.plan["attention.core"] is not behind.prefill_plan["attention.core"] and not behind.attended_tables(), reads
        getattr(behind, reads)["attention.core"].attended_tables = True  # the core's block of this one plan says so
        behind.reset(fresh=True)
        held = behind.state.table
        assert behind.attended_tables() and tuple(held.lengths.shape) == (97, 4) and tuple(held.first_token().lengths.shape) == (100, 4), reads
        assert behind.state.bytes() == tiny.state_bytes(3, 100, 4, page=PAGE, max_model_len=2051), reads
    # a block whose calls two rows serve by call site names nothing itself: every part of it is asked
    split = Session(tiny, text, max_model_len=2051, **common)
    with _a_block_asks_for_a_page():
        plan = split.resolve()
    row, chain = plan.rows["attention.core"], plan["attention.core"]
    part = copy.copy(chain)
    part.attended_tables = True
    plan["attention.core"] = blocks.BySite("attention.core", [("a", row, chain), ("b", row, part)])
    assert plan["attention.core"].attended_tables is False and split.attended_tables(), "the part of the call site b reads them"
    # the two tables are tables of a pool of pages: a block that reads them where no block asks for one is a wrong table
    unpaged = Session(tiny, text, **common)
    with _a_block_asks_for_a_page(page=None, key="torch_ref.attention", tables=True):
        unpaged.resolve()
    for refused in (unpaged.attended_tables, unpaged.plan_table, unpaged.load):
        with pytest.raises(ValueError, match="attention.core in the plan of 3 sequences"):
            refused()
    assert unpaged.state is None and unpaged.weights is None, "named before anything is read or built"
    ahead = Session(tiny, text, **common)  # ... and where it is a session behind the first one's, on its weights: named when the first one plans
    later = Session(tiny, text, share=ahead, **common)
    with _a_block_asks_for_a_page(page=None, key="torch_ref.attention", tables=True):
        later.resolve()
    with pytest.raises(ValueError, match="attention.core in the plan of 3 sequences"):
        ahead.plan_table([later])
    assert ahead.state is None and ahead.weights is None and later.state is None
    # the family's arithmetic is the state's own size, with the tables and without, at every case of the layout
    for sequences, tokens, capacity, pages, longest, width in TABLE_CASES:
        state = DecodeState(kinds, sequences, capacity, shapes, "cpu", tokens, page=PAGE, pages=pages, max_model_len=longest)
        counted = tiny.state_bytes(sequences, capacity, tokens, page=PAGE, pages=pages, max_model_len=longest)
        assert state.bytes() == counted, (sequences, tokens, capacity, pages, longest)
        tables = counted - tiny.state_bytes(sequences, capacity, tokens, page=PAGE, pages=pages)
        steps = sum(capacity - each + 1 for each in {tokens, 1})  # the lengths of the state's own steps and, where it has several tokens, of one
        assert tables == (sequences + 1) * width * 4 + steps * 4 * _whole(sequences, 4) * 4, (sequences, tokens, capacity, pages, longest)
        with_tables = tiny.memory_bytes(sequences, capacity, tokens=tokens, page=PAGE, pages=pages, max_model_len=longest)
        assert with_tables - tiny.memory_bytes(sequences, capacity, tokens=tokens, page=PAGE, pages=pages) == tables
    linear = _paged_model(("linear_attention", "linear_attention"))  # no attention layer: no pool, and no table to count
    assert linear.state_bytes(3, 130, page=PAGE, max_model_len=2051) == linear.state_bytes(3, 130)
    # the memory sentence: at the published widths and 65,535 sequences the lengths of a state are half a GiB
    model = QwenDense()
    monkeypatch.setattr(blocks.DeviceFacts, "probe", classmethod(lambda cls, device: cls((10, 0), "stated device", 148, 0, 10**6 * GIB)))
    text = TextIO.synthetic(model.geometry.vocab)
    stated = dict(sequences=65535, stack="reference", device="cuda:0", capacity=2051)

    def need(tables, **session):
        first = Session(model, text, **stated, **session)
        later = Session(model, text, share=first, tokens=2, **stated, **session)
        for each in (first, later):
            with _a_block_asks_for_a_page(), _a_block_asks_for_a_page(key="torch_ref.attention", tables=tables):
                each.resolve()
        return first.plan_table([later]).splitlines()[-1].split(" GiB on this GPU")[0]

    without = model.memory_bytes(65535, 2051, page=PAGE) + model.state_bytes(65535, 2051, 2, page=PAGE)
    counted = model.memory_bytes(65535, 2051, page=PAGE, max_model_len=4096) + model.state_bytes(65535, 2051, 2, page=PAGE, max_model_len=4096)
    assert need(False) == need(False, max_model_len=4096) == f"  this stack needs about {without / GIB:.1f}", "no block reads the tables: none is counted"
    assert need(True, max_model_len=4096) == f"  this stack needs about {counted / GIB:.1f}", "the tables of both sessions, at the run's maximum length"
    assert f"{counted / GIB:.1f}" != f"{without / GIB:.1f}", "two sentences, two figures"


def _nbytes(tensors) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def test_what_the_checks_hold_per_row_term_by_term():
    """``check_row_terms`` against tensors built here, on the CPU, in the shapes the helper layer builds them -- a record, a
    state on plain caches, the pool of a linear-attention layer and what the judge of the fused step forms over it -- at
    one and at several sequences and tokens per sequence; ``check_row_bytes`` as the moments it names; at the widths
    of the two-layer model the terms of 1,024 sequences to the byte; the second reference where it is the larger moment
    and where it is not; the attention chain's term where the context is long."""
    from model_samples.loop import Record
    from model_samples.qwen_dense import QwenDense
    from model_samples.state import DecodeState

    tiny = _tiny_model()
    g = tiny.geometry
    capacity, steps = 12, 7
    for sequences, tokens in ((1, 1), (4, 1), (85, 1), (16, 2), (1, 6)):
        rows = sequences * tokens
        term = tiny.check_row_terms(sequences, capacity, tokens, steps)
        record = Record(steps, rows, g.vocab, "cpu")
        assert term["record"] == _nbytes((record.inputs, record.outputs, record.logits)), "what a session keeps of its steps"
        state = DecodeState(g.layer_kinds, sequences, capacity, tiny.state_shapes(), "cpu", tokens)  # the reference session's: plain caches
        assert term["reference"] == state.bytes() == tiny.state_bytes(sequences, capacity, tokens), (sequences, tokens)
        pool = state.layers[0]
        assert term["pool"] == _nbytes(pool.tensors()), "one save of the judged layer's pool: the recurrent states and the history"
        live = pool.state[1:]
        distance = (live - live).abs()  # as the judge forms them: fp32 over the live slots, one byte per element, four fp64 pools
        assert term["judge"] == _nbytes((distance, distance <= 0.0)) + 4 * _nbytes((pool.state.double(),)), (sequences, tokens)
        assert term["chain"] == rows * 3 * _nbytes((pool.state[0].double(),)), "three fp64 pool rows per row of a step"
        keys = torch.zeros((rows, g.attn_kv_heads, capacity, g.attn_head_dim), dtype=torch.float64)
        assert term["attention"] == 2 * _nbytes((keys,)), "the keys and the values of every position in fp64"
        one = 4 * term["record"] + term["reference"] + max(3 * term["pool"] + term["judge"], term["attention"])
        two = 5 * term["record"] + 2 * term["reference"] + term["chain"]
        assert tiny.check_row_bytes(sequences, capacity, tokens, steps) == one and tiny.check_row_bytes(sequences, capacity, tokens, steps, 2) == max(one, two)
        assert tiny.check_row_terms(sequences, capacity, tokens)["record"] == _nbytes((Record(capacity, rows, g.vocab, "cpu").logits,)) + 8 * capacity * rows
        assert tiny.check_row_terms(sequences, capacity, tokens, 99) == tiny.check_row_terms(sequences, capacity, tokens), "no more steps than positions"
    # the two-layer model at 1,024 sequences and 58 positions, to the byte: one recurrent state is 48 x 128 x 128 fp32 = 3 MiB, the
    # history of a sequence 3 x 10,240 bf16, a recorded row 8 bytes of ids and 15,520 bf16 logits
    wide = QwenDense.reduced(2)
    term = wide.check_row_terms(1024, 58)
    slot, history, row = 48 * 128 * 128 * 4, 3 * 10240 * 2, 8 + 2 * 15520
    assert term["record"] == 58 * 1024 * row == 1_844_002_816
    assert term["pool"] == 1025 * slot + 1025 * history == 3_287_347_200
    assert term["judge"] == 1024 * slot + 1024 * slot // 4 + 8 * 1025 * slot == 29_821_501_440
    assert term["reference"] == wide.state_bytes(1024, 58) == 1025 * slot + 1025 * history + 2 * 1024 * 4 * 58 * 256 * 2
    assert term["chain"] == 1024 * 6 * slot and term["attention"] == 1024 * 2 * 4 * 58 * 256 * 8
    held = wide.check_row_bytes(1024, 58)
    assert held == 4 * 1_844_002_816 + term["reference"] + 3 * 3_287_347_200 + 29_821_501_440 and round(held / GIB, 2) == 47.12
    assert wide.check_row_bytes(1024, 58, references=2) == held, "the two-layer model: the judge's moment is the larger one with a second reference too"
    assert held - wide.check_row_bytes(1, 58) == 50_502_809_280, "what the sentence adds at 1,024 sequences: 47.03 GiB"
    # a long context: the fp64 keys and values of the attention chain pass the saves and the judge's tensors
    long = wide.check_row_terms(64, 4096)
    assert long["attention"] > 3 * long["pool"] + long["judge"] > wide.check_row_terms(64, 2048)["attention"]
    assert wide.check_row_bytes(64, 4096) == 4 * long["record"] + long["reference"] + long["attention"]
    # the published widths, a checkpoint's run (32 recorded steps, two references): the second reference's moment is the larger one
    full = QwenDense()
    term = full.check_row_terms(4, 343, record_steps=32)
    one = 4 * term["record"] + term["reference"] + 3 * term["pool"] + term["judge"]
    two = 5 * term["record"] + 2 * term["reference"] + term["chain"]
    assert two > one and full.check_row_bytes(4, 343, record_steps=32, references=2) == two and full.check_row_bytes(4, 343, record_steps=32) == one
    # a model without a kind of layer has no term of that kind
    linear = _paged_model(("linear_attention", "linear_attention")).check_row_terms(3, 40)
    assert linear["attention"] == 0 and linear["pool"] > 0
    attention = _paged_model(("full_attention", "full_attention")).check_row_terms(3, 40)
    assert (attention["pool"], attention["judge"], attention["chain"]) == (0, 0, 0) and attention["attention"] > 0


def _judged_with_what_is_alive(sequences, steps=7):
    """A run of the tiny geometry through stand-in providers and its checks, as ``_checked_run`` makes it, with every recorded
    step kept; and, taken at the FIRST judged call of the fused linear-attention step, the objects that are alive:
    ``(session, checks, alive)``."""
    import gc

    from model_samples import Session, TextIO, blocks, tie_out
    from model_samples.loop import Record
    from model_samples.providers import factories, torch_ref
    from model_samples.state import DecodeState

    class Step(torch_ref.GdnDecodeStep):
        pass

    class Multiply(torch_ref.Linear):
        def __call__(self, qx, weight, site=""):
            return super().__call__(qx, weight)

    class GateUp(torch_ref.GateUp):
        pass

    def kept():  # by its type, never by ``isinstance``: that asks every object of the process for its class
        return [each for each in gc.get_objects() if type(each) in (Record, DecodeState, torch.Tensor)]

    gc.collect()
    earlier = kept()  # what another test of the process left alive is not this run's: held here, so that no address is used again
    known = {id(each) for each in earlier}
    stand_ins = {"fe.gdn_decode_fused": Step, "cublas.bf16_dequantized": Multiply, "cublas.bf16_dequantized+torch_ref.silu_mul_quantize": GateUp}
    model = _tiny_model()
    text = TextIO.synthetic(model.geometry.vocab)
    session = Session(model, text, sequences=sequences, stack="fe-cublas", device="cpu", capacity=16, record_steps=steps)
    facts = blocks.DeviceFacts((10, 0), "stated device", 148, 0, 0)
    options = {"reference_precision": "fp64", "device": "cpu"}
    session.plan = blocks.plan(model.provider_table, model.regimes, facts, sequences, model=model, options=options, factories=dict(factories(), **stand_ins))
    session.load()
    ids = session.prefill([text.encode("ab" + str(s)) for s in range(sequences)])
    session.read(ids)
    for _ in range(steps - 3):  # a prompt of three tokens and four steps: seven recorded steps
        ids = session.step(ids)
        session.read(ids)
    alive = {}
    judge = tie_out.BlockTap._judge

    def watched(stats, out, ref, state, ref_state, layer=0, twin=None):
        if stats.band == "gdn_step" and not alive:
            gc.collect()
            found = [each for each in kept() if id(each) not in known]
            alive["records"] = [each for each in found if type(each) is Record]
            alive["states"] = [each for each in found if type(each) is DecodeState]
            alive["tensors"] = [each for each in found if type(each) is torch.Tensor]
        return judge(stats, out, ref, state, ref_state, layer, twin)

    tie_out.BlockTap._judge = staticmethod(watched)
    try:
        checks = tie_out.run_checks(session, prompt_tokens=3)
    finally:
        tie_out.BlockTap._judge = staticmethod(judge)
    return session, checks, alive


@pytest.mark.parametrize("sequences", [2, 5])
def test_the_count_is_what_is_alive_at_the_judge_of_the_fused_step(sequences):
    """What the memory sentence rests on, on the thing itself: at the first judged call of the fused linear-attention step
    inside the checks of a run, the records, the states and the saves that are alive are the ones ``check_row_bytes``
    counts, and their bytes are its terms -- three records and the clones of a fourth, the checked session's state and
    the reference's, and three saves of the judged layer's pool.  A change of the checks that keeps a record or a save
    more, or drops one, fails here, so that the count moves with it.  (What the judge itself forms is alive inside its
    call alone: no host case sees it, the GPU cell of 85 sequences holds PyTorch's peak to the count.)"""
    session, checks, alive = _judged_with_what_is_alive(sequences)
    assert checks.ok and alive, checks.text()
    model = session.model
    term = model.check_row_terms(sequences, session.capacity, 1, session.record_steps)
    # the records: the checked session's new one, the first run's, which the checks keep by a name, and the reference session's
    held = [(record.inputs, record.outputs, record.logits) for record in alive["records"]]
    assert len(held) == 3 and all(_nbytes(tensors) == term["record"] for tensors in held), [_nbytes(tensors) for tensors in held]
    assert len({id(tensors[2]) for tensors in held}) == 3, "three records, not one seen three times"
    # ... and the clones of the first run's checked steps: tensors of a record's shapes that belong to no record
    of_a_record = {id(tensor) for tensors in held for tensor in tensors}
    shapes = {tuple(tensor.shape) for tensor in held[0]}
    clones = [tensor for tensor in alive["tensors"] if tuple(tensor.shape) in shapes and id(tensor) not in of_a_record]
    assert len(clones) == 3 and _nbytes(clones) == term["record"], [tuple(tensor.shape) for tensor in clones]
    # the states: the checked session's own, which the sentence counts as the state, and the reference session's on plain caches
    assert len(alive["states"]) == 2 and sorted(state.bytes() for state in alive["states"]) == sorted(
        [session.state.bytes(), term["reference"]]
    ), "one reference state"
    # the saves of the judged layer's pool: before the call, after it, and what the twin left
    pool = session.state.layers[0].tensors()
    of_a_state = {id(tensor) for state in alive["states"] for layer in state.layers for tensor in layer.tensors()}
    saves = [tensor for tensor in alive["tensors"] if tuple(tensor.shape) in {tuple(each.shape) for each in pool} and id(tensor) not in of_a_state]
    assert len(saves) == 3 * len(pool) and _nbytes(saves) == 3 * term["pool"], [tuple(tensor.shape) for tensor in saves]
    # so what the count holds beyond the judge's own tensors is alive, to the byte
    assert model.check_row_bytes(sequences, session.capacity, 1, session.record_steps) - term["judge"] == (
        _nbytes([tensor for tensors in held for tensor in tensors]) + _nbytes(clones) + term["reference"] + _nbytes(saves)
    )


def test_the_reference_session_of_a_count_is_released_when_its_checks_return(monkeypatch):
    """Of a run of several counts the memory sentence takes the checks of ONE count, the one that holds the most: it rests
    on the checks of a count dropping what they held when they return.  Held here on the thing itself: two counts on one
    load, as a sample builds them, on the tiny geometry through stand-in providers, with Python's collection of
    reference cycles switched off.  Every reference session the checks build, and every state and record such a session
    has, is kept by a weak reference; when the checks of a count have returned none of them is alive -- released by its
    last reference, not by a collection that comes when it comes.  The second count checks a second reference too."""
    import gc
    import weakref

    from model_samples import Session, TextIO, blocks, loop, tie_out
    from model_samples.providers import factories, torch_ref

    class Step(torch_ref.GdnDecodeStep):
        pass

    class Multiply(torch_ref.Linear):
        def __call__(self, qx, weight, site=""):
            return super().__call__(qx, weight)

    class GateUp(torch_ref.GateUp):
        pass

    watched = []  # (what, a weak reference to it): the sessions the checks build, the states and the records they have

    class Watched(loop.Session):
        """The session the checks build, as it is: a weak reference is kept to it and to what each of its resets allocates."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert self.stack == loop.REFERENCE, "the checks build reference sessions and no other"
            watched.append(("session", weakref.ref(self)))

        def reset(self, fresh=True, recorded=True):
            super().reset(fresh, recorded)
            watched.extend((("state", weakref.ref(self.state)), ("record", weakref.ref(self.record))))

    monkeypatch.setattr(tie_out, "Session", Watched)
    stand_ins = {"fe.gdn_decode_fused": Step, "cublas.bf16_dequantized": Multiply, "cublas.bf16_dequantized+torch_ref.silu_mul_quantize": GateUp}
    model = _tiny_model()
    text = TextIO.synthetic(model.geometry.vocab)
    facts = blocks.DeviceFacts((10, 0), "stated device", 148, 0, 0)
    options = {"reference_precision": "fp64", "device": "cpu"}

    def counted(sequences, share=None):
        session = Session(model, text, sequences=sequences, stack="fe-cublas", device="cpu", capacity=16, record_steps=7, share=share)
        session.plan = blocks.plan(
            model.provider_table, model.regimes, facts, sequences, model=model, options=options, factories=dict(factories(), **stand_ins)
        )
        return session

    enabled = gc.isenabled()
    gc.disable()  # from here on an object in a reference cycle stays alive: what is released below is released by its last reference
    try:
        first = counted(2)
        runs = [(first, {}), (counted(3, share=first), {"second_reference": True})]  # every count of a run is built before the first load
        for session, more in runs:
            session.load()
            ids = session.prefill([text.encode("ab" + str(s)) for s in range(session.sequences)])
            session.read(ids)
            for _ in range(4):
                ids = session.step(ids)
                session.read(ids)
            built = len(watched)
            checks = tie_out.run_checks(session, prompt_tokens=3, **more)
            mine = watched[built:]  # of the checks of this count
            assert checks.ok, checks.text()
            assert [what for what, _ in mine].count("session") == 1 + len(more), "the reference session, and the second reference where one is checked"
            assert {what for what, _ in mine} == {"session", "state", "record"}, "each of them with a state and a record"
            alive = [what for what, held in watched if held() is not None]
            assert not alive, f"alive behind the checks of {session.sequences} sequences, with no collection of reference cycles: {alive}"
        assert first.state is not None and first.record is not None, "the checked sessions keep their own state and record: those the sentence counts"
    finally:
        if enabled:
            gc.enable()


COPY = "the reference's bf16 copy of one weight at a time, where the run keeps none"


def _sentences(monkeypatch, model, free=200 * GIB, cc=(10, 0)):
    """``sentence(stack, *counts)``: the memory sentence of a run of ``model`` on stated device facts, one session per count."""
    from model_samples import Session, TextIO, blocks

    text = TextIO.synthetic(model.geometry.vocab)
    monkeypatch.setattr(blocks.DeviceFacts, "probe", classmethod(lambda cls, device: cls(cc, "stated device", 148, 0, free)))

    def sentence(stack, *counts, **session):
        common = dict(stack=stack, device="cuda:0", capacity=58, **session)
        first = Session(model, text, sequences=counts[0], **common)
        later = [Session(model, text, sequences=count, share=first, **common) for count in counts[1:]]
        return first.plan_table(later).splitlines()[-1].strip()

    return sentence


def test_the_memory_sentence_counts_what_the_checks_hold_for_every_sequence_beyond_the_first(monkeypatch):
    """The sentence of the seeded default request (the two-layer model, 58 positions) on stated facts of compute capability
    10.0.  One sequence: the sentence it was.  Several: the need holds what the checks of the run hold for every
    sequence beyond the first -- the family's arithmetic, named as the checks' part -- and a share of that part more
    for PyTorch's pool (``loop.CHECK_POOL``); a stack that keeps weights on their stored format alone names both parts
    of the checks.  The reference stack runs no checks and its sentence is the one it was."""
    from model_samples import loop
    from model_samples.qwen_dense import QwenDense

    model = QwenDense.reduced(2)
    sentence = _sentences(monkeypatch, model)
    assert sentence("fe-entries", 1) == f"this stack needs about 3.5 GiB on this GPU, 0.33 GiB of it for the checks ({COPY}); 200.0 GiB are free"
    assert sentence("fe-cublas", 1) == "this stack needs about 3.7 GiB on this GPU; 200.0 GiB are free", "one sequence: nothing beyond the first"
    assert sentence("fe-entries", 85) == (
        "this stack needs about 9.4 GiB on this GPU, 3.86 GiB of it for the checks (what they hold for the 84 sequences beyond the first); 200.0 GiB are free"
    )
    assert sentence("fe-entries", 1024) == (
        "this stack needs about 72.9 GiB on this GPU, 47.03 GiB of it for the checks (what they hold for the 1023 sequences beyond the first); "
        "200.0 GiB are free"
    )
    assert sentence("fe-entries", 2).endswith(
        f"0.38 GiB of it for the checks ({COPY}, and what they hold for the 1 sequence beyond the first); 200.0 GiB are free"
    )
    # the figure is the need of that count as it was, the family's count beyond the first sequence, and the pool's share of the count
    rows = model.check_row_bytes(85, 58) - model.check_row_bytes(1, 58)
    assert round(rows / GIB, 2) == 3.86
    assert (
        f"{(model.memory_bytes(85, 58, page=64) + rows + int(rows * loop.CHECK_POOL)) / GIB:.1f}" == "9.4"
        and f"{model.memory_bytes(85, 58, page=64) / GIB:.1f}" == "4.0"
    )
    # the default run, 1 and 4 sequences in one process, on the four stacks
    several = " for 1 and 4 sequences, one count after the other"
    beyond = "what they hold for the sequences beyond the first of a count"
    assert sentence("fe-cublas", 1, 4) == f"this stack needs about 3.9 GiB on this GPU{several}, 0.14 GiB of it for the checks ({beyond}); 200.0 GiB are free"
    both = f"0.47 GiB of it for the checks ({COPY}, and {beyond})"
    assert sentence("fe-entries", 1, 4) == f"this stack needs about 3.7 GiB on this GPU{several}, {both}; 200.0 GiB are free"
    assert sentence("native", 1, 4) == f"this stack needs about 3.7 GiB on this GPU{several}, {both}; 200.0 GiB are free"
    assert sentence("reference", 1, 4) == f"this stack needs about 3.7 GiB on this GPU{several}; 200.0 GiB are free", "no checks run on the reference stack"
    assert sentence("reference", 1024) == "this stack needs about 7.0 GiB on this GPU; 200.0 GiB are free", "its state, and nothing of a check"


def test_the_rows_part_of_a_run_and_its_clause_in_every_form_of_the_sentence(monkeypatch):
    """What a run adds for its checks, session by session: of several counts the largest part and of every other session its
    record; a session that verifies behind its prompt by its rows; nothing for a session of one sequence, with a drafter
    or without, nothing on the reference stack, and nothing of a family that counts nothing.  And the clause in the three
    forms of the sentence: the line, the SKIP for too little free memory, the SKIP for a run the GPU cannot hold."""
    from model_samples import Session, TextIO, blocks, loop
    from model_samples.qwen_dense import QwenDense, QwenDenseDrafter

    model = QwenDense.reduced(2)
    text = TextIO.synthetic(model.geometry.vocab)
    monkeypatch.setattr(blocks.DeviceFacts, "probe", classmethod(lambda cls, device: cls((10, 0), "stated device", 148, 0, 200 * GIB)))

    def held(sequences, tokens=1, capacity=58, references=1):
        return model.check_row_bytes(sequences, capacity, tokens, None, references) - model.check_row_bytes(1, capacity, tokens, None, references)

    def record(sequences, tokens=1, capacity=58):
        return model.check_row_terms(sequences, capacity, tokens)["record"] - model.check_row_terms(1, capacity, tokens)["record"]

    def sessions(*shapes, stack="fe-entries", capacity=58):
        first = Session(model, text, stack=stack, device="cuda:0", capacity=capacity, **shapes[0])
        return first, [Session(model, text, stack=stack, device="cuda:0", capacity=capacity, share=first, **shape) for shape in shapes[1:]]

    first, later = sessions(dict(sequences=9), dict(sequences=4))
    assert first._check_rows(later, 1) == held(9) + record(4), "the checks of the larger count, and the record the other count keeps"
    first, later = sessions(dict(sequences=4), dict(sequences=9))
    assert first._check_rows(later, 1) == held(9) + record(4), "whichever comes first"
    assert first._check_rows(later, 2) == held(9, references=2) + record(4) and first._check_rows([], 1) == held(4)
    first, later = sessions(dict(sequences=1), dict(sequences=1))
    assert first._check_rows(later, 1) == 0, "one sequence: nothing beyond the first"
    first, later = sessions(dict(sequences=16), dict(sequences=16, draft_tokens=1), capacity=42)
    assert first._check_rows(later, 1) == held(16, 2, 42) + record(16, 1, 42), "a session behind its prompt: by its rows, two a sequence"
    sentence = first.plan_table(later).splitlines()[-1]
    kinds = "a plain session of 16 sequences and a session of 16 sequences x 2 tokens behind its prompt, one after the other"
    assert f" for {kinds}, 1.67 GiB of it for the checks ({COPY}, and what they hold for the sequences beyond the first of a session); " in sentence
    drafter = QwenDenseDrafter.seeded(model)
    first, later = sessions(dict(sequences=1), dict(sequences=1, draft_tokens=5, drafter=drafter))
    assert first._check_rows(later, 1) == 0 and "what they hold" not in first.plan_table(later), "a drafter serves one sequence"
    first, later = sessions(dict(sequences=4), dict(sequences=9), stack="reference")
    assert first._check_rows(later, 1) == 0, "the reference stack runs no checks"
    # the share for PyTorch's pool is of the rows part, and is in the need without a name of its own
    first, later = sessions(dict(sequences=64))
    need = model.memory_bytes(64, 58, **first._kept(), page=64)
    part = held(64) + model.check_bytes(first._kept()["uncopied"])
    assert first.plan_table().splitlines()[-1].strip() == (
        f"this stack needs about {(need + held(64) + int(held(64) * loop.CHECK_POOL)) / GIB:.1f} GiB on this GPU, {part / GIB:.2f} GiB of it for the checks "
        f"({COPY}, and what they hold for the 63 sequences beyond the first); 200.0 GiB are free"
    )
    # a family that counts nothing is not asked: the sentence it had
    with monkeypatch.context() as without:
        without.delattr(QwenDense, "check_row_bytes")
        first, later = sessions(dict(sequences=4), dict(sequences=9), stack="fe-cublas")
        assert first._check_rows(later, 1) == 0
        assert first.plan_table(later).splitlines()[-1].strip() == (
            f"this stack needs about {(model.memory_bytes(4, 58) + model.state_bytes(9, 58)) / GIB:.1f} GiB on this GPU for 4 and 9 sequences, "
            "one count after the other; 200.0 GiB are free"
        )
    # the clause in the three forms of the sentence, on a plan with stated facts
    plan = _plan(stack="fe-entries", sequences=85)
    rows, beyond = held(85), "the 84 sequences beyond the first"
    clause = "3.86 GiB of it for the checks (what they hold for the 84 sequences beyond the first)"
    assert plan.memory(8 * GIB, "", 0, rows, beyond) == f"this stack needs about 8.0 GiB on this GPU, {clause}; 80.0 GiB are free"
    plan.facts = _facts(free=6 * GIB)
    with pytest.raises(NotImplementedError) as skip:
        plan.memory(8 * GIB, "", 0, rows, beyond)
    assert str(skip.value).startswith(f"stack fe-entries needs about 8.0 GiB on this GPU, {clause}; 6.0 GiB are free: free memory on this GPU")
    with pytest.raises(NotImplementedError) as skip:
        plan.memory(200 * GIB, "", GIB // 2, rows, beyond)
    assert str(skip.value).startswith(
        f"stack fe-entries needs about 200.0 GiB on this GPU, 4.36 GiB of it for the checks ({COPY}, and what they hold for {beyond}); "
        "this GPU has 180.0 GiB in total"
    )
    assert plan.memory(4 * GIB, "", 1, 0, beyond) == f"this stack needs about 4.0 GiB on this GPU, 0.00 GiB of it for the checks ({COPY}); 6.0 GiB are free"
    assert plan.memory(4 * GIB, "", 0, 1, beyond).endswith(
        ", 0.00 GiB of it for the checks (what they hold for the 84 sequences beyond the first); 6.0 GiB are free"
    )
    assert plan.memory(4 * GIB, "", 0, 0, beyond) == "this stack needs about 4.0 GiB on this GPU; 6.0 GiB are free", "no byte of the checks: no clause"


def test_the_plan_hands_the_count_the_sessions_recorded_steps_and_its_references(monkeypatch):
    """What ``Session.plan_table`` hands the family's count are the session's own integers -- the steps it records and the
    reference stacks its checks replay -- held where a run with a checkpoint stands and no other case looks: the
    published widths at four sequences of 343 positions, 32 recorded steps and a second reference, where the second
    reference's moment is the larger one.  Through ``_check_rows`` and ``plan_table`` to the sentence: the part it names
    and the need it states."""
    from model_samples import Session, TextIO, blocks, loop
    from model_samples.qwen_dense import QwenDense

    monkeypatch.setattr(blocks.DeviceFacts, "probe", classmethod(lambda cls, device: cls((10, 0), "stated device", 148, 0, 400 * GIB)))
    full = QwenDense()
    text = TextIO.synthetic(full.geometry.vocab)

    def session(**more):
        return Session(full, text, sequences=4, stack="fe-entries", device="cuda:0", capacity=343, **more)

    def beyond(steps, references):
        return full.check_row_bytes(4, 343, 1, steps, references) - full.check_row_bytes(1, 343, 1, steps, references)

    kept = session(record_steps=32)
    assert kept._check_rows([], 1) == beyond(32, 1) < beyond(None, 1) == session()._check_rows([], 1), "the steps the session records, not every position"
    assert (
        kept._check_rows([], 2) == beyond(32, 2) > beyond(32, 1)
    ), "with a second reference its moment is the larger one at these widths, at four sequences and at one"
    assert full.check_row_bytes(1, 343, 1, 32, 2) > full.check_row_bytes(1, 343, 1, 32, 1), "... so the count at one sequence is taken with the same references"
    copy = full.check_bytes(kept._kept()["uncopied"])
    for references in (1, 2):  # and through the plan to the sentence: the part it names and the need it states
        line = kept.plan_table([], references).splitlines()[-1].strip()
        rows = beyond(32, references)
        need = full.memory_bytes(4, 343, **kept._kept(), **kept._paged()) + rows + int(rows * loop.CHECK_POOL)
        assert line.startswith(f"this stack needs about {need / GIB:.1f} GiB on this GPU, {(copy + rows) / GIB:.2f} GiB of it for the checks ("), line
    assert line.startswith("this stack needs about 39.3 GiB on this GPU, 1.59 GiB of it for the checks ("), "the sentence of that run with two references"
    assert kept.plan_table([], 2) != kept.plan_table([], 1) != session().plan_table([], 1), "three sentences"


def test_an_override_takes_the_row_of_the_runs_stack_where_it_names_the_same_call(monkeypatch, capsys):
    """``--provider BLOCK=KEY`` selects the first declared row of the key that serves the run -- and, where that row is of
    another stack, the row of the run's own stack that names the same call: equal in every field but its stacks and its
    printed reason.  The attention core has such a pair: on the target stacks the override plans their chain's row, with
    the reason of a chain that reads the attention cache -- also on ``native``, whose own plan has the library's row in
    front of that chain.  A row of the run's stack that differs in anything else is another
    call and is not taken: the fused step selected on a target stack is still the form of ``fe-cublas``."""
    from dataclasses import replace

    from model_samples.blocks import ProviderRow, select_row
    from model_samples.qwen_dense import QwenDense

    model = QwenDense.reduced(2)
    core = model.provider_table["attention.core"]
    pool, plain = "reference chain over the attention cache", "reference chain over a plain per-layer cache"
    chains = [(row.key, row.fallback) for row in core if row.fallback]
    assert chains == [("torch_ref.attention", plain), ("torch_ref.attention", pool)], "one call on two rows"
    for stack, reason in (("fe-entries", pool), ("native", pool), ("fe-cublas", plain), ("reference", plain)):
        row = select_row("attention.core", core, _facts(), 1, 1, stack, key="torch_ref.attention", model=model)
        assert row.fallback == reason and stack in row.stacks, f"{stack}: the row of the stack that runs"
        own = select_row("attention.core", core, _facts(), 1, 1, stack, model=model)
        assert (own is row) == (stack != "native"), f"{stack}: the row the stack plans without the override is this chain, but where a library serves the block"
        planned = _plan(stack=stack, overrides={"attention.core": "torch_ref.attention"})
        assert planned.rows["attention.core"] is row
        printed = [line for line in planned.table().splitlines() if line.split()[:1] == ["attention.core"]]
        assert len(printed) == 1 and printed[0].endswith(f"fallback: {reason}"), printed
    with pytest.raises(AssertionError, match="reached a load"):  # ... and through the sample's own main(), to the table it prints
        _sample_main(monkeypatch, capsys, ["--stack", "fe-entries", "--provider", "attention.core=torch_ref.attention"])
    printed = [line for line in capsys.readouterr().err.splitlines() if line.split()[:1] == ["attention.core"] and "fallback: " in line]
    assert len(printed) == 1 and printed[0].endswith(f"fallback: {pool}"), "the served-by table the run prints before its first load"
    # the fused step has the key on a row of fe-cublas and, in another form, on a row of the target stacks: the first declared row wins
    step = model.provider_table["linear_attention.step"]
    for stack in ("native", "fe-entries"):
        row = select_row("linear_attention.step", step, _facts(), 1, 1, stack, key="fe.gdn_decode_fused", model=model)
        assert row.stacks == ("fe-cublas",) and dict(row.form) == {"PDL": False, "PDL_RELEASE": False}, f"{stack}: the form of fe-cublas"
        own = select_row("linear_attention.step", step, _facts(), 1, 1, stack, model=model)
        assert own.key == row.key and dict(own.form) == {"PDL": True, "PDL_RELEASE": True}, "the stack's own row is another form of the entry"
    # on a constructed table: only a row that differs in its stacks and its reason alone is the same call
    first = ProviderRow(("any",), (1, None), (1, None), "fe.gdn_decode_fused", "fe-cublas", serves="any", fallback="one reason", form=(("PDL", False),))
    same = replace(first, stack=("native", "fe-entries"), fallback="another reason")

    def chosen(*rows, stack="native"):
        return select_row("a block", rows, _facts(), 1, 1, stack, key="fe.gdn_decode_fused")

    assert chosen(first, same) is same and chosen(same, first, stack="fe-cublas") is first, "the row of the stack that runs, in either order"
    assert chosen(first, same, stack="reference") is first, "a stack with no row of the key: the first declared row"
    for field, value in (("form", (("PDL", True),)), ("sequences", (1, 8)), ("band", "bf16_rounding"), ("serves", "rows 1-8"), ("classes", ("cc10.x",))):
        assert chosen(first, replace(same, **{field: value})) is first, f"a row of the stack with another {field}: another call"


# ------------------------------------------------------------------------------- the plan of a step, stack by stack
PREPARE_FORM = "[POSITIONS 3d, SECTION (11, 11, 10), PAGE_SIZE 64, POSITION_DTYPE int64, SLOT_DTYPE int64, PDL False, PDL_RELEASE False]"
PROJECTION_FORM = "[SPLIT_K 1, MMA_M 128, STAGES 8, TMEM_COLS 32, PREWAIT_WEIGHTS True, HOLD_TMEM True, PDL True, PDL_RELEASE True"
SWIGLU_FORM = "[TOKEN_TILE 8, PREWAIT_STAGES 2, L2_EVICT_FIRST True, MAX_CLUSTERS 148, PDL True]"
ATTENTION_ENTRY = "flashinfer.decode.trtllm_batch_decode_with_kv_cache"
#: per block the kernel that serves it in the stack ``native`` on compute capability 10.0, by its canonical id
NATIVE_KERNELS = {
    "embedding": "embedding_gather",
    "add_norm_quantize": "add_norm_quant",
    "projection.fp8.in": "fp8_projection",
    "linear_attention.step": "gdn_decode_fused",
    "mlp.gate_up": "swiglu_fp4",
    "attention.prepare": "attention_prepare",
    "attention.core": "fi_decode_attention",
    "attention.gate_quantize": "gate_quant",
    "projection.fp8.out": "cublaslt_fp8_gemm",
    "mlp.down": "fi_mm_fp4",
    "output_head.quantize": "row_quantize_fp4",
    "output_head": "fi_mm_fp4",
}


def _stated_plan(layers=64, stack="native", tokens=1, cc=(10, 0), sms=148, sequences=1, overrides=None, table=None):
    """``(plan, model)`` of ``stack`` on stated device facts, at the maximum length the scripts default to."""
    from model_samples.blocks import DeviceFacts, plan
    from model_samples.qwen_dense import QwenDense

    model = QwenDense() if layers == 64 else QwenDense.reduced(layers)
    facts = DeviceFacts(cc, "stated device", sms, 180 * GIB, 80 * GIB)
    rows = model.provider_table if table is None else table
    return plan(rows, model.regimes, facts, sequences, tokens, stack, overrides, model=model, options={"device": "cpu", "max_model_len": 2051}), model


def _step_of(planned, model):
    """What a plan states of one step: ``(launches on exact rows, exact rows, fallback calls)`` as the last line of its kernel
    table counts them, and ``{block: its calls on a fallback row}`` from the row of every call of the wiring."""
    from collections import Counter

    from model_samples import census

    served = [(name, planned.at(name, site)[0]) for (name, _), site in zip(model.wiring(), census.call_sites(model))]
    line = census.table(planned).splitlines()[-1]
    counted = re.fullmatch(
        r"  per step: (\d+) launch(?:es)? on (\d+) exact rows?, checked one by one in the wiring order; "
        r"the launches of (\d+) fallback calls? and of the host path are attributed by family",
        line,
    )
    assert counted, line
    return tuple(int(n) for n in counted.groups()), dict(Counter(name for name, row in served if row.fallback))


def test_the_plan_of_a_step_stack_by_stack():
    """What the provider table plans for a step, read from the plan alone.  ``native`` on compute capability 10.0 at 148 SMs has
    every block on a kernel -- no call on a fallback row, at one token per sequence and at six -- and the rows in which
    ``fe-entries`` differs from it are the library's rows, each on its fallback row there.  Then the kernel of every block,
    the rows that are gated on batch invariance, the by-provider line of every GPU class, the kernel table row by row, and
    per stack and kind of step the launches of a step with the blocks that are left on fallback rows."""
    from collections import Counter
    from dataclasses import replace

    from model_samples import census, perf
    from model_samples.qwen_dense import QwenDense, configs

    plan, model = _stated_plan()
    entries_plan, _ = _stated_plan(stack="fe-entries")
    # every block of native on a kernel: a row that falls back to a chain or to a multiply over a copy is named here
    for layers in (64, 2):
        for tokens in (1, 6):
            (_, _, chains), fallback = _step_of(*_stated_plan(layers, tokens=tokens))
            assert fallback == {} and chains == 0, (layers, tokens)
    assert not any(row.fallback for _, _, _, row, _ in plan.parts()) and "fallback" not in plan.by_provider()
    assert configs.TARGET_STACKS == QwenDense.target_stacks == ("native", "fe-entries"), "the two stacks whose rows are kernels on the stored formats"
    # the rows in which the two stacks differ are the library's, and fe-entries has each of them on its fallback row
    differing = {block for block in plan.rows if plan.rows[block].key != entries_plan.rows[block].key}
    assert differing == set(configs.LIBRARY_ROWS) == {"projection.fp8.out", "mlp.down", "output_head", "attention.core"}
    assert all(entries_plan.rows[block].fallback and plan.rows[block].stacks == ("native",) for block in differing)
    for block in configs.LIBRARY_ROWS:  # the library states no batch invariance for these calls; no lane of this repository runs them
        band = "attention_bf16_weights" if block == "attention.core" else "bf16_rounding"  # the attention kernel documents no contract: the bound of a model
        assert plan.rows[block].batch_invariant == () and plan.rows[block].ci is False and plan.rows[block].band == band, block
    gated = (
        "add_norm_quantize",
        "projection.fp8.in",
        "linear_attention.step",
        "mlp.gate_up",
        "attention.prepare",
        "attention.gate_quantize",
        "output_head.quantize",
    )
    for block in gated:  # the class their forms ran on: 2 or more sequences gate them
        assert plan.rows[block].batch_invariant == ("cc10.0",), f"{block}: batch invariance of its native row is a gate on compute capability 10.0"
    assert set(gated) | set(configs.LIBRARY_ROWS) | {"embedding", "token_choice"} == set(plan), "every block is named above"
    # the kernel of every block, one each; a step of several tokens per sequence has one kernel the decode step has not
    exact = {name: [kernel.id for kernel in block.kernels if kernel.relation == "exact"] for name, block in plan.items()}
    assert {name: ids[0] for name, ids in exact.items() if ids} == NATIVE_KERNELS and all(len(ids) <= 1 for ids in exact.values())
    several, _ = _stated_plan(tokens=6)
    stepped = {kernel.id for block in several.values() for kernel in block.kernels if kernel.relation == "exact"}
    assert stepped - set(NATIVE_KERNELS.values()) == {"gdn_verify_fused"}, "the verify step has one kernel the decode step has not"
    assert plan["output_head"].entry == plan["mlp.down"].entry == "flashinfer.gemm.mm_fp4", "the head's row is the multiply alone: two keys, one call"
    assert census.form_text(plan.rows["mlp.gate_up"].form).strip() == SWIGLU_FORM
    forms = Counter(site for (block, _), site in zip(model.wiring(), model.sites()) if block == "add_norm_quantize")
    assert forms == {"linear_attention.first": 1, "linear_attention": 47, "full_attention": 16, "mlp": 64, "output_head": 1}, "five forms: 1 / 47 / 16 / 64 / 1"
    shapes = Counter(site for (block, _), site in zip(model.wiring(), model.sites()) if block == "projection.fp8.in")
    assert shapes == {"linear_attention.qkvz": 48, "full_attention.qkv": 16}, "two weight shapes: 48 / 16"

    # the by-provider line of every class, and the sentences of the family README that state it under the table of the class
    flat = " ".join((MODELS / "qwen_dense" / "README.md").read_text().split())

    def said(planned, tail=True):
        line = planned.by_provider().replace("by provider: ", "By provider: ").replace("cuDNN FE ", "cuDNN Frontend ")
        line, _, unverified = line.partition(" | unverified in CI: ")  # the rows no lane of this repository runs: the library's
        entries, natural = line.rsplit(" | PyTorch ", 1)
        more = f"; {unverified} rows unverified in CI" if unverified else ""
        return entries.replace(" | ", "; ") + (f"; PyTorch as the natural provider {natural}{more}." if tail else more)

    table = configs.readme_table(QwenDense().calls_text(), "native", (10, 0))
    heading = "## Who serves each block (stacks `native` and `fe-entries`, compute capability 10.0)"
    assert (
        " ".join(f"{heading} {table} {said(plan)}".split()) in flat
    ), "the heading, the table and the by-provider sentence are those of compute capability 10.0"
    assert said(plan) == "By provider: cuDNN Frontend 7; FlashInfer 4; PyTorch as the natural provider 2; 4 rows unverified in CI."
    assert said(entries_plan) == "By provider: cuDNN Frontend 7; fallbacks 4 (cuBLAS over copies 3, PyTorch chains 1); PyTorch as the natural provider 2."
    assert (
        f"three multiplies over a copy of `fe-cublas`: {said(entries_plan).replace('By provider: ', '')}" in flat
    ), "the sentence on fe-entries under the table"
    for cc, sms in (((10, 3), 152), ((10, 7), 212)):
        other = said(_stated_plan(cc=cc, sms=sms)[0], tail=False).replace("By provider: ", "")
        assert other == "cuDNN Frontend 6; FlashInfer 4; fallbacks 1 (cuBLAS over copies 1); 4 rows unverified in CI", cc
        assert f"On the other 10.x classes the in-projection is the multiply over a copy of `fe-cublas`: {other}." in flat, cc
    tiles = "; ".join(f"`TOKEN_TILE {tile}` from {first} to {last} rows" for first, last, tile in configs.GATE_UP_TILES[1:])
    assert f"The row of the MLP's gate / up multiply is the one of 1 to 8 rows; three more rows of the same key name the other token tiles: {tiles}." in flat
    for sequences, (first, last, tile) in ((n, row) for row in configs.GATE_UP_TILES for n in row[:2]):
        row = _stated_plan(2, sequences=sequences)[0].rows["mlp.gate_up"]
        assert row.key == "fe.swiglu_fp4_decode" and dict(row.form)["TOKEN_TILE"] == tile and row.rows == (first, last), sequences
    other = said(_stated_plan(cc=(12, 0), sms=170)[0], tail=False).replace("By provider: ", "")
    assert f"the reference chains of `fe-cublas` as well: {other}." in flat and other.startswith("cuDNN Frontend 1; fallbacks 9 "), "compute capability 12.x"
    assert f"35 on the four-layer one. {said(_stated_plan(stack='fe-cublas')[0])}" in flat, "the sentence under the table of the default stack"
    # compute capability 12.x: no entry row but the fused step's serves there, and every other block stays on its fallback row
    plan12, _ = _stated_plan(2, cc=(12, 0), sms=170)
    assert plan12.rows["attention.gate_quantize"].key == "torch_ref.gate_quantize" and plan12.rows["linear_attention.step"].form == (
        ("PDL", True),
        ("PDL_RELEASE", True),
    )
    assert plan12.rows["add_norm_quantize"].key == "torch_ref.add_norm_quantize" and plan12.rows["add_norm_quantize"].site_form("mlp") == ()
    assert plan12.rows["projection.fp8.in"].key == "cublas.bf16_dequantized" and plan12.rows["projection.fp8.in"].site_form("full_attention.qkv") == ()
    assert plan12.rows["mlp.gate_up"].key == "cublas.bf16_dequantized+torch_ref.silu_mul_quantize" and plan12.rows["mlp.gate_up"].form == ()
    # a row selected with --provider: the fused step in its fe-cublas form is the same kernel in another form
    selected, _ = _stated_plan(2, overrides={"linear_attention.step": "fe.gdn_decode_fused"})
    assert selected.overridden == ("linear_attention.step",) and selected.rows["linear_attention.step"].form == (("PDL", False), ("PDL_RELEASE", False))

    # the kernel table of native, row by row: the entry in its form, the kernel, the launches of a step and what is checked
    small, two = _stated_plan(2)
    table = census.table(small).splitlines()
    assert table[0].startswith("kernels, planned (decided before the load from the family's configs.py; stack native,") and table[1].split()[-1] == "checked"
    row = {line.split()[0]: line for line in table[2:15]}
    stated = "name and launch shape, as the entry states them"
    assert row["embedding"].split("torch.index_select")[1].split() == "embedding_gather 1 name".split()
    assert row["linear_attention.step"].split("cudnn.GdnDecodeFused")[1].split() == f"[PDL True, PDL_RELEASE True] gdn_decode_fused 1 {stated}".split()
    assert row["attention.gate_quantize"].split("cudnn.AttentionGateQuantDecode")[1].split() == f"[THREADS 256, PDL_RELEASE True] gate_quant 1 {stated}".split()
    assert row["mlp.down"].split("flashinfer.gemm.mm_fp4")[1].split() == "[BACKEND cute-dsl, TUNED False] fi_mm_fp4 2 name".split()
    assert row["attention.core"].split(ATTENTION_ENTRY)[1].split() == "[KV_LAYOUT HND] fi_decode_attention 1 name".split()
    assert row["output_head"].split("flashinfer.gemm.mm_fp4")[1].split() == "[BACKEND cute-dsl, TUNED False] fi_mm_fp4 1 name".split()
    quantised = f"[WIDTH 5120, PDL True, PDL_RELEASE True] row_quantize_fp4 1 {stated}"
    assert row["output_head.quantize"].split("cudnn.RowQuantizeFp4Decode")[1].split() == quantised.split(), "the entry's whole config"
    assert row["projection.fp8.out"].split("flashinfer.gemm.bmm_fp8")[1].split() == "[BACKEND auto] cublaslt_fp8_gemm, cublas 2 name".split()
    assert (
        row["attention.prepare"].split("cudnn.AttentionPrepareDecode")[1].split() == f"{PREPARE_FORM} attention_prepare 1 {stated}".split()
    ), "the kernel stated whole"
    assert row["token_choice"].rstrip().endswith("host path, not compared")
    norm = f"[PDL True, PDL_RELEASE True] + one of 6 forms, by call site add_norm_quant 5 {stated}"
    assert row["add_norm_quantize"].split("cudnn.AddNormQuantDecode")[1].split() == norm.split()
    assert table[15:21] == [f"  add_norm_quantize at {site}:{census.form_text(fields)}" for site, fields in configs.NORM_FORMS], "the form of every call site"
    assert table[17] == "  add_norm_quantize at linear_attention: [ADD True, STORE_RESIDUAL False, BF16_OUT True, QUANT e4m3_static]"
    multiply = f"{PROJECTION_FORM}] + one of 2 forms, by call site fp8_projection 2 {stated}"
    assert row["projection.fp8.in"].split("cudnn.Fp8ProjectionDecode")[1].split() == multiply.split()
    assert table[21:23] == [f"  projection.fp8.in at {site}:{census.form_text(fields)}" for site, fields in configs.PROJECTION_SITES], "and the weight's shape"
    assert (
        row["mlp.gate_up"].split("cudnn.SwigluFp4Decode")[1].split() == f"{SWIGLU_FORM} swiglu_fp4 2 {stated}".split()
    ), "the token tile of the row count in it"
    assert table[-1].startswith("  per step: 20 launches on 12 exact rows") and len(table) == 24

    # per stack and kind of step: the launches of a step on exact rows, and which blocks are left on fallback rows, how often.
    # A row that falls off its kernel, or a kernel that gains a launch per call, moves a number here
    library = "cuDNN FE 7 | FlashInfer 4 | PyTorch 2 | unverified in CI: 4"
    entries = "cuDNN FE 7 | fallbacks 4 (cuBLAS over copies 3, PyTorch chains 1) | PyTorch 2"
    # the launches of a step by provider, as the performance report states them: in a step of several tokens per sequence the
    # out-projection of the 48 linear-attention layers is the projection entry's, not the library's
    launched = {
        (64, "native", 1): {"cuDNN Frontend": 338, "FlashInfer": 145, "PyTorch": 1},
        (64, "native", 6): {"cuDNN Frontend": 386, "FlashInfer": 97, "PyTorch": 1},
        (64, "fe-entries", 1): {"cuDNN Frontend": 338, "PyTorch": 1},
        (64, "fe-entries", 6): {"cuDNN Frontend": 386, "PyTorch": 1},
    }
    chain_table = {**configs.PROVIDERS, "attention.prepare": configs.PROVIDERS["attention.prepare"][1:]}  # the preparation on its reference chain
    for layers, stack, tokens, exact, rows, left, provider in (
        (64, "native", 1, 484, 12, {}, library),
        (64, "native", 6, 484, 13, {}, library),
        (64, "fe-entries", 1, 339, 8, {"projection.fp8.out": 64, "mlp.down": 64, "attention.core": 16, "output_head": 1}, entries),
        (64, "fe-entries", 6, 387, 9, {"mlp.down": 64, "attention.core": 16, "projection.fp8.out": 16, "output_head": 1}, entries),
        (2, "native", 1, 20, 12, {}, None),
        (2, "native", 6, 20, 13, {}, None),
        (2, "fe-entries", 1, 14, 8, {"projection.fp8.out": 2, "mlp.down": 2, "attention.core": 1, "output_head": 1}, None),
        (2, "fe-entries", 6, 15, 9, {"mlp.down": 2, "attention.core": 1, "projection.fp8.out": 1, "output_head": 1}, None),
    ):
        planned, model = _stated_plan(layers, stack, tokens)
        counted, fallback = _step_of(planned, model)
        where = (layers, stack, tokens)
        assert counted == (exact, rows, sum(left.values())) and fallback == left, where
        assert exact + sum(left.values()) == len(model.wiring()) - 1, f"{where}: one launch per block call, the token choice on the host path"
        if provider:
            assert planned.by_provider() == f"by provider: {provider}", where
        stated_by = perf.planned_launches(planned, model)
        assert (stated_by["exact_launches_per_step"], stated_by["fallback_calls"], stated_by["fallback_rows"]) == (exact, sum(left.values()), bool(left)), where
        assert sum(stated_by["by_provider"].values()) == exact and stated_by["by_provider"] == launched.get(where, stated_by["by_provider"]), where
        prepare = next(line for line in census.table(planned).splitlines() if line.split()[0] == "attention.prepare")
        calls = model.geometry.layer_kinds.count("full_attention")  # the launches of the preparation's entry in a step: one per attention layer
        assert prepare.split("cudnn.AttentionPrepareDecode")[1].split() == f"{PREPARE_FORM} attention_prepare {calls} {stated}".split(), where
        # the preparation's row moves what a stack launches by the launches of its entry and by nothing else
        without, on_chain = _step_of(*_stated_plan(layers, stack, tokens, table=chain_table))
        assert without == (exact - calls, rows - 1, sum(left.values()) + calls) and on_chain == {**left, "attention.prepare": calls}, where
    # a row that names one position row is the same entry in another form: the row and its entry agree on it
    row = configs.PROVIDERS["attention.prepare"][0]
    one_row = replace(row, form=(("POSITIONS", "1d"), ("SECTION", None), *row.form[2:]))
    planned, model = _stated_plan(2, "fe-entries", table={**configs.PROVIDERS, "attention.prepare": [one_row, *configs.PROVIDERS["attention.prepare"][1:]]})
    assert census.check_forms(census.expected(planned, model)) == [], "the form is the row's, and the entry states it"

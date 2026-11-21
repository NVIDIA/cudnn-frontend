# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check selected F06 lifecycle source obligations and the abstract Lean protocol.

These AST checks are regression detectors, NOT a compiler or PTX refinement
proof. They check selected code segments, not arbitrary paths through the
kernel. Run standalone; no cudnn, CUDA, torch or Cutlass imports are needed.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[7]
KERNEL = (HERE.parent / "kernel.py").relative_to(ROOT)
COMMON = Path("python/cudnn/sdpa/fwd/kernels/_common_blackwell.py")
BARRIER = Path("python/cudnn/frost/tile_dsl/barrier.py")
TMA = Path("python/cudnn/frost/tile_dsl/tma.py")


def spelling(node):
    return ast.unparse(node)


def walk(node, kind):
    return sorted((n for n in ast.walk(node) if isinstance(n, kind)), key=lambda n: (n.lineno, n.col_offset))


def calls(node, name):
    return [n for n in walk(node, ast.Call) if spelling(n.func) == name]


def one(nodes):
    if len(nodes) != 1:
        raise ValueError(f"Expected one source site, found {len(nodes)}")
    return nodes[0]


def function(tree, name):
    return one([n for n in walk(tree, ast.FunctionDef) if n.name == name])


def block(nodes):
    return ast.Module(body=nodes, type_ignores=[])


def containing(tree, kind, predicate):
    """Smallest enclosing block; ambiguity fails, never guesses another site."""
    candidates = [n for n in walk(tree, kind) if predicate(n)]
    if not candidates:
        raise ValueError("Missing enclosing source block")
    size = min(n.end_lineno - n.lineno for n in candidates)
    return one([n for n in candidates if n.end_lineno - n.lineno == size])


def kw(call, name):
    return one([k.value for k in call.keywords if k.arg == name])


class Audit:
    def __init__(self, root):
        self.root = root
        self.trees = {p: ast.parse((root / p).read_text()) for p in (KERNEL, COMMON, BARRIER, TMA)}
        self.records = []
        self.parents = {}
        for tree in self.trees.values():
            for parent in ast.walk(tree):
                for field, value in ast.iter_fields(parent):
                    children = value if isinstance(value, list) else [value]
                    for child in children:
                        if isinstance(child, ast.AST):
                            self.parents[id(child)] = (parent, field)

    def context(self, node):
        path = []
        while id(node) in self.parents:
            parent, field = self.parents[id(node)]
            if isinstance(parent, ast.If):
                path.append(f"if {spelling(parent.test)} / {field}")
            elif isinstance(parent, ast.For):
                path.append(f"for {spelling(parent.target)} in {spelling(parent.iter)} / {field}")
            elif isinstance(parent, ast.While):
                path.append(f"while {spelling(parent.test)} / {field}")
            elif isinstance(parent, ast.FunctionDef):
                path.append(f"function {parent.name}")
            node = parent
        return list(reversed(path))

    def site(self, node):
        return dict(line=node.lineno, code=spelling(node), context=self.context(node))

    def order(self, name, nodes, path=KERNEL):
        if len(nodes) < 2 or any(a.lineno >= b.lineno for a, b in zip(nodes, nodes[1:])):
            raise ValueError(f"{name}: required source order is broken")
        self.records.append(dict(obligation=name, path=str(path), sites=[self.site(n) for n in nodes]))

    def fact(self, name, node, expected, path=KERNEL):
        if spelling(node) != expected:
            raise ValueError(f"{name}: expected {expected!r}, found {spelling(node)!r}")
        self.records.append(dict(obligation=name, path=str(path), sites=[self.site(node)]))

    def run(self):
        kernel = self.trees[KERNEL]
        mma = function(kernel, "_mma_warp_group")
        loader = function(kernel, "_tmaldg_warp_group")
        corr = function(kernel, "_correction_warp_group")
        stg = function(kernel, "_tmastg_warp_group")

        # 1. K/V: the credits must track asynchronous MMA completion, not a
        # thread arrival immediately following an asynchronous issue.
        classic = function(self.trees[COMMON], "make_classic_bars")
        bars = one(calls(classic, "Bars"))
        for resource in ("k", "v"):
            decl = kw(bars, f"mb_{resource}_empty")
            self.fact(f"{resource}_release_is_mma_completion", kw(decl, "producer"), "Producer.MMA_COMMIT", COMMON)
            self.fact(f"{resource}_release_count", kw(decl, "init_count"), "KV_EMPTY_ARRIVERS", COMMON)
            full = kw(bars, f"mb_{resource}_full")
            self.fact(f"{resource}_publication_tracks_tma", kw(full, "producer"), "Producer.TMA_LOAD", COMMON)

        # Protect the wrapper dispatch too; its implementation is still trusted.
        arrive = one([n for n in walk(self.trees[BARRIER], ast.FunctionDef) if n.name == "arrive" and calls(n, "commit_mma")])
        commit_branch = one([n for n in walk(arrive, ast.If) if spelling(n.test) == "cutlass.const_expr(self.producer == int(Producer.MMA_COMMIT))"])
        one(calls(block(commit_branch.body), "commit_mma"))
        self.fact("mma_commit_dispatch", commit_branch.test, "cutlass.const_expr(self.producer == int(Producer.MMA_COMMIT))", BARRIER)

        # K prologue and loop body: BOTH query slabs must be issued before
        # the completion-tracking credit. Pin exact release index and scope.
        k_releases = calls(mma, "bars.mb_k_empty[kv_state.idx].arrive")
        if len(k_releases) != 2:
            raise ValueError("Expected K prologue and mainloop releases")
        for i, release in enumerate(k_releases):
            preceding = [n for n in calls(mma, "mma_ss") if n.lineno < release.lineno]
            reads = preceding[-2:]
            if len(reads) != 2 or [spelling(n.args[1]) for n in reads] != ["desc_Q0", "desc_Q1"]:
                raise ValueError("K release precedes one of its two readers")
            self.order(f"k_both_readers_before_release_{i}", reads + [release])
            self.fact(f"k_release_group_{i}", kw(release, "cta_group"), "CFG.CTA_MMA")

        for index in ("old_state.idx", "kv_state.idx"):
            release = one(calls(mma, f"bars.mb_v_empty[{index}].arrive"))
            # Each release's preceding PV region begins at its V full wait.
            waits = [n for n in calls(mma, f"bars.mb_v_full[{index}].wait") if n.lineno < release.lineno]
            start = one(waits)
            reads = [n for n in calls(mma, "mma_ts_step") if start.lineno < n.lineno < release.lineno]
            # Two chunks for each of two subtiles in the admitted TILE_N=128 path.
            if len(reads) != 4 or [spelling(n.args[1]) for n in reads] != [
                "tmem_raw.subview(LAYOUT.P0_OFF)",
                "tmem_raw.subview(LAYOUT.P0_OFF)",
                "tmem_raw.subview(LAYOUT.P1_OFF)",
                "tmem_raw.subview(LAYOUT.P1_OFF)",
            ]:
                raise ValueError("V release does not follow all four PV issue sites")
            self.order(f"v_all_readers_before_release_{index}", [start] + reads + [release])

        # Selected dense producer branches. Guard paths are retained for review;
        # this does not prove their reachability or all paths through the loader.
        for tensor, resource in (("sK", "k"), ("sV", "v")):
            fills = [n for n in calls(loader, "tma_load_tile") if spelling(n.args[0]) == f"{tensor}[kv_state.idx]"]
            if len(fills) != 2:
                raise ValueError(f"Expected prologue and steady-state {tensor} fills")
            waits = calls(loader, f"bars.mb_{resource}_empty[kv_state.idx].wait")
            for i, fill in enumerate(fills):
                acquire = max((n for n in waits if n.lineno < fill.lineno), key=lambda n: n.lineno)
                self.order(f"{resource}_credit_before_fill_{i}", [acquire, fill])
                self.fact(f"{resource}_fill_completion_barrier_{i}", fill.args[2], f"bars.mb_{resource}_full[kv_state.idx].smem_ptr")

        # 2. Final Stats occupy the next QK's S head. This backedge crosses
        # work items; the ordinary q_full/k_full edges cannot replace it.
        epi = containing(corr, ast.For, lambda n: bool(calls(n, "bars.mb_stats_read[qs].arrive")))
        score_arm = one([n for n in walk(epi, ast.If) if spelling(n.test) == "cutlass.const_expr(_score_stats)" and calls(n, "nvvm.tcgen05_ld")])
        stats_load = one(calls(block(score_arm.orelse), "nvvm.tcgen05_ld"))
        waits = [n for n in calls(epi, "nvvm.tcgen05_wait") if spelling(kw(n, "kind")) == "nvvm.Tcgen05Wait.LOAD" and n.lineno > stats_load.lineno]
        release = one(calls(epi, "bars.mb_stats_read[qs].arrive"))
        self.order("stats_load_completion_before_reuse_credit", [stats_load, waits[0], release])
        self.fact("stats_release_to_mma_group", kw(release, "cta_group"), "CFG.CTA_MMA")
        prologue = containing(mma, ast.If, lambda n: bool(calls(n, "bars.mb_stats_read[0].wait")))
        live = block(prologue.orelse)
        for qs in (0, 1):
            wait = one(calls(live, f"bars.mb_stats_read[{qs}].wait"))
            overwrite = [n for n in calls(live, "mma_ss") if spelling(n.args[1]) == f"desc_Q{qs}"][0]
            self.order(f"stats_credit_before_next_qk_{qs}", [wait, overwrite])
            self.fact(f"stats_wait_generation_{qs}", wait.args[0], "stats_read_phase")
            # Empty work must consume the same generation even without QK.
            empty_wait = one(calls(block(prologue.body), f"bars.mb_stats_read[{qs}].wait"))
            self.fact(f"empty_work_consumes_stats_credit_{qs}", empty_wait.args[0], "stats_read_phase")

        # 3. O: wait for TMA's READ completion before overwriting its SMEM
        # source. This is not a claim that the global output is already visible.
        store_loop = containing(stg, ast.For, lambda n: bool(calls(n, "bars.mb_o_empty[qs].arrive")))
        partial_arm = one([n for n in walk(store_loop, ast.If) if spelling(n.test) == "cutlass.const_expr(not _FP32_PARTIALS)"])
        thd_arm = one([n for n in walk(partial_arm, ast.If) if spelling(n.test) == "cutlass.const_expr(CFG.THD_VARLEN)"])
        tma_issue = one(calls(block(thd_arm.orelse), "tma_store_tile"))
        drain = one(calls(partial_arm, "tma_store_wait"))
        release = one(calls(store_loop, "bars.mb_o_empty[qs].arrive"))
        self.order(
            "o_tma_read_completion_before_credit",
            [one(calls(store_loop, "bars.mb_o_full[qs].wait")), tma_issue, one(calls(partial_arm, "tma_store_commit")), drain, release],
        )
        self.fact("o_drain_all_groups", drain.args[0], "0")
        helper_wait = one(calls(function(self.trees[TMA], "tma_store_wait"), "nvvm.cp_async_bulk_wait_group"))
        self.fact("o_wait_tracks_source_reads", kw(helper_wait, "read"), "True", TMA)
        self.fact("o_wait_forwards_group_count", helper_wait.args[0], "num_remaining", TMA)

        store = one(calls(corr, "smem_ptr.store_swizzled"))
        chunk_loop = containing(corr, ast.For, lambda n: bool(calls(n, "smem_ptr.store_swizzled")))
        first_candidates = [n for n in walk(chunk_loop, ast.If) if spelling(n.test) == "chunk_idx == 0"]
        if len(first_candidates) != 1:
            raise ValueError("o_first_chunk_gate: expected exactly one chunk-zero gate")
        first = first_candidates[0]
        acquire = one(calls(first, "bars.mb_o_empty[qs].wait"))
        self.order("o_credit_before_first_overwrite", [acquire, store])
        self.fact("o_first_chunk_gate", first.test, "chunk_idx == 0")
        self.fact("o_gate_generation", acquire.args[0], "o_empty_phase")
        fence = one(calls(epi, "nvvm.fence_proxy"))
        self.order("o_proxy_publication", [store, fence, one(calls(epi, "bars.mb_o_full[qs].arrive"))])
        self.fact("o_proxy_kind", fence.args[0], "'async.shared'")
        self.fact("o_proxy_scope", kw(fence, "space"), "'cta'")
        return self.records

    def check_contexts(self):
        # Reviewed control-flow snapshots for the SELECTED sites. A textual
        # predecessor hidden under a new conditional must not count as a gate.
        expected = json.loads((HERE / "lifecycle_contexts.json").read_text())
        observed = {r["obligation"]: [site["context"] for site in r["sites"]] for r in self.records}
        if observed != expected:
            changed = sorted(k for k in observed.keys() | expected.keys() if observed.get(k) != expected.get(k))
            raise ValueError(f"control_flow_context_changed: {changed}")


# Isolated SOURCE mutations. They must violate the indicated obligation;
# a syntax/import/Lean error does not count as a successful negative control.
MUTATIONS = (
    (
        "ordinary_k_credit",
        COMMON,
        "make_classic_bars",
        "mb_k_empty=MBarrier(_alloc(STAGES_K), stages=STAGES_K, init_count=KV_EMPTY_ARRIVERS, producer=Producer.MMA_COMMIT)",
        "mb_k_empty=MBarrier(_alloc(STAGES_K), stages=STAGES_K, init_count=KV_EMPTY_ARRIVERS, producer=Producer.THREAD)",
        "k_release_is_mma_completion",
    ),
    (
        "stats_read_wait_removed",
        KERNEL,
        "_correction_warp_group",
        "nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)\n            total_max_scaled",
        "pass\n            total_max_scaled",
        "stats_load_completion_before_reuse_credit",
    ),
    ("output_drain_weakened", KERNEL, "_tmastg_warp_group", "tma_store_wait(0)", "tma_store_wait(1)", "o_drain_all_groups"),
    (
        "output_wait_hidden_in_false_branch",
        KERNEL,
        "_tmastg_warp_group",
        "                tma_store_wait(0)",
        "                if False:\n                    tma_store_wait(0)",
        "control_flow_context_changed",
    ),
    (
        "output_first_chunk_unprotected",
        KERNEL,
        "_correction_warp_group",
        "if chunk_idx == 0:\n                        bars.mb_o_empty",
        "if chunk_idx == 1:\n                        bars.mb_o_empty",
        "o_first_chunk_gate",
    ),
)


def mutate(root, path, helper, before, after):
    target = root / path
    code = target.read_text()
    fn = function(ast.parse(code), helper)
    lines = code.splitlines(keepends=True)
    body = "".join(lines[fn.lineno - 1 : fn.end_lineno])
    if body.count(before) != 1:
        raise ValueError("Mutation target is missing or ambiguous")
    target.write_text("".join(lines[: fn.lineno - 1]) + body.replace(before, after) + "".join(lines[fn.end_lineno :]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--lean-bin", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = dict(
        schema_version=1,
        evidence_kind="source_obligations_and_separate_abstract_protocol",
        all_checks_passed=False,
        kernel_race_freedom="unverified",
        source_to_protocol_refinement="unverified",
        trials=[],
    )
    report_path = out / "lifecycle.json"
    try:
        baseline = Audit(args.source_root)
        report["source_obligations"] = baseline.run()
        baseline.check_contexts()
        report["sha256"] = {str(p): hashlib.sha256((args.source_root / p).read_bytes()).hexdigest() for p in (KERNEL, COMMON, BARRIER, TMA)}
        report["proof_inputs_sha256"] = {
            name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in ("check_lifecycle.py", "Lifecycle.lean", "lifecycle_contexts.json")
        }
        print(f"source obligations: PASS ({len(report['source_obligations'])})", flush=True)
        for name, path, helper, before, after, expected in MUTATIONS:
            with tempfile.TemporaryDirectory(prefix="f06-lifecycle-") as directory:
                root = Path(directory)
                for p in (KERNEL, COMMON, BARRIER, TMA):
                    (root / p).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(args.source_root / p, root / p)
                mutate(root, path, helper, before, after)
                try:
                    mutated = Audit(root)
                    mutated.run()
                    mutated.check_contexts()
                except ValueError as error:
                    diagnostic = str(error)
                    accepted = expected in diagnostic
                else:
                    diagnostic, accepted = "Mutation incorrectly accepted", False
                report["trials"].append(dict(name=name, accepted=accepted, diagnostic=diagnostic, path=str(path), before=before, after=after))
                print(f"{name}: {'REJECTED AS EXPECTED' if accepted else 'FAIL'}", flush=True)
                if not accepted:
                    raise ValueError(diagnostic)
        version = subprocess.check_output([args.lean_bin, "--version"], text=True).strip()
        if not version.startswith("Lean (version 4.15.0,"):
            raise ValueError(f"Expected Lean 4.15.0, got {version}")
        proof = HERE / "Lifecycle.lean"
        if re.search(r"\b(sorry|admit|native_decide|axiom)\b", proof.read_text()):
            raise ValueError("Unexpected proof escape hatch")
        proc = subprocess.run([args.lean_bin, str(proof)], capture_output=True, text=True, timeout=60)
        log = proc.stdout + proc.stderr
        (out / "Lifecycle.log").write_text(log)
        audits = [line for line in log.splitlines() if "axioms" in line]
        allowed = {"propext", "Classical.choice", "Quot.sound"}
        used = set()
        for line in audits:
            match = re.search(r"depends on axioms: \[(.*)\]", line)
            if match:
                used.update(x.strip() for x in match[1].split(","))
        accepted = proc.returncode == 0 and "error:" not in log and len(audits) == 8 and used <= allowed
        report["lean"] = dict(version=version, accepted=accepted, axioms=audits, sha256=hashlib.sha256(proof.read_bytes()).hexdigest())
        if not accepted:
            raise ValueError("Lifecycle.lean did not pass; see Lifecycle.log")
        print("Lean protocol: PASS (8 theorems)", flush=True)
        # Check that the invariant really needs read COMPLETION. This mutation
        # preserves valid Lean syntax but permits an early credit after issue.
        code = proof.read_text()
        before = "| release (s) (done : s.released < s.retired)"
        if code.count(before) != 1:
            raise ValueError("Missing Lean release mutation site")
        mutant = out / "LifecycleEarlyRelease.lean"
        mutant.write_text(code.replace(before, "| release (s) (done : s.released < s.acquired)"))
        proc = subprocess.run([args.lean_bin, str(mutant)], capture_output=True, text=True, timeout=60)
        log = proc.stdout + proc.stderr
        mutant.with_suffix(".log").write_text(log)
        start = code[: code.index("theorem step_preserves")].count("\n") + 1
        end = code[: code.index("inductive Reachable")].count("\n") + 1
        failures = re.findall(r":(\d+):\d+: error: omega could not prove the goal", log)
        rejected = proc.returncode != 0 and any(start <= int(line) < end for line in failures)
        report["lean_negative_control"] = dict(name="release_before_read_completion", accepted=rejected, diagnostic_log=str(mutant.with_suffix(".log")))
        if not rejected:
            raise ValueError("Early-release model did not fail at the invariant proof")
        print("early-release Lean model: REJECTED AS EXPECTED", flush=True)
        report["all_checks_passed"] = True
    finally:
        report_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

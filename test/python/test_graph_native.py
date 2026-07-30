# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Python-native graph representation."""

import pytest

torch = pytest.importorskip("torch")

from cudnn.graph_types import NodeType, Tensor
from cudnn.nodes import Node, _row_major_stride
from cudnn._pygraph import pygraph, GraphContext

pytestmark = pytest.mark.L0


class TestTensor:
    """Tests for Tensor."""

    def test_create_tensor(self):
        t = Tensor(name="test", dim=[8, 64, 128], stride=[8192, 128, 1])
        assert t.name == "test"
        assert t.dim == [8, 64, 128]
        assert not t.is_virtual

    def test_builder_pattern(self):
        t = Tensor()
        t.set_name("my_tensor").set_dim([4, 32]).set_stride([32, 1]).set_output(True)
        assert t.name == "my_tensor"
        assert not t.is_virtual

    def test_set_output(self):
        t = Tensor(is_virtual=True)
        t.set_output(True)
        assert not t.is_virtual
        t.set_output(False)
        assert t.is_virtual

    def test_validation_success(self):
        t = Tensor(name="valid", dim=[8, 64], stride=[64, 1])
        t.validate()

    def test_validation_no_dims(self):
        t = Tensor(name="no_dims", stride=[64, 1])
        with pytest.raises(ValueError, match="dims not set"):
            t.validate()

    def test_validation_dim_stride_mismatch(self):
        t = Tensor(name="mismatch", dim=[8, 64, 128], stride=[64, 1])
        with pytest.raises(ValueError, match="mismatch"):
            t.validate()

    def test_uid_management(self):
        t = Tensor(name="test")
        assert not t.uid_assigned
        t.set_uid(42)
        assert t.uid == 42
        assert t.uid_assigned


class TestNode:
    """Tests for Node class."""

    def test_create_node(self):
        node = Node("mm1", NodeType.MATMUL)
        assert node.name == "mm1"
        assert node.node_type == NodeType.MATMUL
        assert node.inputs == {}
        assert node.outputs == {}
        assert node.params == {}

    def test_node_with_tensors(self):
        node = Node("mm1", NodeType.MATMUL)
        a = Tensor(name="A", dim=[8, 64], stride=[64, 1])
        b = Tensor(name="B", dim=[64, 32], stride=[32, 1])
        c = Tensor(name="C", dim=[8, 32], stride=[32, 1])

        node.inputs["A"] = a
        node.inputs["B"] = b
        node.outputs["C"] = c

        assert node.inputs["A"] is a
        assert node.outputs["C"] is c

    def test_node_params(self):
        node = Node("mm1", NodeType.MATMUL)
        node.params["padding"] = 0.0
        node.params["alpha"] = 2.0
        assert node.params["padding"] == 0.0

    def test_node_repr(self):
        node = Node("mm1", NodeType.MATMUL)
        assert repr(node) == "Node('mm1', MATMUL)"


class TestMatmulInference:
    """Tests for matmul dimension inference."""

    def test_infer_2d(self):
        node = Node("mm", NodeType.MATMUL)
        a = Tensor(name="A", dim=[64, 128], stride=[128, 1])
        b = Tensor(name="B", dim=[128, 256], stride=[256, 1])
        c = Tensor(name="C", is_virtual=True)

        node.inputs["A"] = a
        node.inputs["B"] = b
        node.outputs["C"] = c

        node.infer_properties(GraphContext())
        assert c.dim == [64, 256]

    def test_infer_3d_batched(self):
        node = Node("mm", NodeType.MATMUL)
        a = Tensor(name="A", dim=[8, 64, 128], stride=[8192, 128, 1])
        b = Tensor(name="B", dim=[8, 128, 256], stride=[32768, 256, 1])
        c = Tensor(name="C", is_virtual=True)

        node.inputs["A"] = a
        node.inputs["B"] = b
        node.outputs["C"] = c

        node.infer_properties(GraphContext())
        assert c.dim == [8, 64, 256]

    def test_infer_strides(self):
        node = Node("mm", NodeType.MATMUL)
        a = Tensor(name="A", dim=[8, 64], stride=[64, 1])
        b = Tensor(name="B", dim=[64, 32], stride=[32, 1])
        c = Tensor(name="C", is_virtual=True)

        node.inputs["A"] = a
        node.inputs["B"] = b
        node.outputs["C"] = c

        node.infer_properties(GraphContext())
        assert c.stride == [32, 1]


class TestRowMajorStride:
    """Tests for stride computation."""

    def test_1d(self):
        assert _row_major_stride([10]) == [1]

    def test_2d(self):
        assert _row_major_stride([8, 64]) == [64, 1]

    def test_3d(self):
        assert _row_major_stride([4, 8, 16]) == [128, 16, 1]

    def test_empty(self):
        assert _row_major_stride([]) == []


class Testpygraph:
    """Tests for pygraph."""

    def test_creation(self):
        g = pygraph()
        assert len(g.nodes) == 0
        assert len(g.tensors) == 0

    def test_with_context(self):
        g = pygraph(io_data_type="HALF", compute_data_type="FLOAT")
        assert g.context.io_data_type == "HALF"
        assert g.context.compute_data_type == "FLOAT"

    def test_tensor_creation(self):
        g = pygraph()
        t = g.tensor(dim=[8, 64, 128], name="my_tensor")
        assert t.name == "my_tensor"
        assert t.dim == [8, 64, 128]
        assert t.stride == [8192, 128, 1]
        assert "my_tensor" in g.tensors

    def test_uid_ownership(self):
        """The IR owns the uid namespace: user-specified uids are reserved (auto
        allocation skips them). The SAME collision rule applies at creation as
        at set_uid: a user uid landing on an auto-assigned one steals it (the
        auto holder is renumbered — classic tensors have no uid until assigned,
        so classic code cannot observe auto uids); user-user collisions raise."""
        g = pygraph()
        a = g.tensor(dim=[2, 2], uid=2, name="user_uid")  # reserve 2
        assert a.uid == 2 and a.uid_assigned
        b = g.tensor(dim=[2, 2], name="auto1")  # auto: 1
        c = g.tensor(dim=[2, 2], name="auto2")  # auto: must skip reserved 2 -> 3
        assert b.uid == 1
        assert c.uid == 3
        d = g.tensor(dim=[2, 2], uid=3, name="steals_from_auto")
        assert d.uid == 3 and d.uid_assigned
        assert c.uid not in (2, 3) and not c.uid_assigned  # renumbered
        assert g._tensor_by_uid[c.uid] is c
        with pytest.raises(ValueError, match="user-assigned"):
            g.tensor(dim=[2, 2], uid=2, name="dup_user")  # user-user collides

    def test_matmul(self):
        g = pygraph()
        A = g.tensor(dim=[8, 64, 128], name="A")
        B = g.tensor(dim=[8, 128, 256], name="B")
        C = g.matmul(A, B, name="mm1")

        assert len(g.nodes) == 1
        assert g.nodes[0].node_type == NodeType.MATMUL
        assert g.nodes[0].name == "mm1"
        assert C.is_virtual

    def test_matmul_inputs_outputs(self):
        g = pygraph()
        A = g.tensor(dim=[8, 64, 128], name="A")
        B = g.tensor(dim=[8, 128, 256], name="B")
        C = g.matmul(A, B, name="mm1")

        node = g.nodes[0]
        assert node.inputs["A"] is A
        assert node.inputs["B"] is B
        assert node.outputs["C"] is C
        assert node.params["padding"] == 0.0

    def test_find_tensor_by_name(self):
        g = pygraph()
        t = g.tensor(dim=[8, 64], name="test")
        assert g.find_tensor("test") is t

    def test_find_tensor_by_uid(self):
        g = pygraph()
        t = g.tensor(dim=[8, 64], name="test")
        assert g.find_tensor(t.uid) is t

    def test_find_tensor_not_found(self):
        g = pygraph()
        assert g.find_tensor("nonexistent") is None

    def test_inspect(self):
        g = pygraph(io_data_type="HALF")
        A = g.tensor(dim=[8, 64], name="A")
        B = g.tensor(dim=[64, 32], name="B")
        C = g.matmul(A, B, name="mm1")

        info = g.inspect()
        assert len(info["nodes"]) == 1
        assert info["nodes"][0]["name"] == "mm1"
        assert info["nodes"][0]["type"] == "MATMUL"
        assert info["nodes"][0]["params"]["padding"] == 0.0
        assert "A" in info["tensors"]

    def test_auto_naming(self):
        g = pygraph()
        A = g.tensor(dim=[8, 64], name="A")
        B = g.tensor(dim=[64, 32], name="B")

        g.matmul(A, B)
        g.matmul(A, B)

        assert g.nodes[0].name == "matmul.0"
        assert g.nodes[1].name == "matmul.1"

    def test_validation(self):
        g = pygraph()
        A = g.tensor(dim=[8, 64], stride=[64, 1], name="A")
        B = g.tensor(dim=[64, 32], stride=[32, 1], name="B")
        g.matmul(A, B)
        g.validate()

    def test_pointwise_add(self):
        g = pygraph()
        A = g.tensor(dim=[8, 64], name="A")
        B = g.tensor(dim=[8, 64], name="B")
        C = g.add(A, B)

        assert len(g.nodes) == 1
        assert g.nodes[0].node_type == NodeType.POINTWISE
        assert "mode" in g.nodes[0].params

    def test_relu(self):
        g = pygraph()
        X = g.tensor(dim=[8, 64], name="X")
        Y = g.relu(X)
        assert g.nodes[0].node_type == NodeType.POINTWISE

    def test_all_pointwise_builders(self):
        """Every op in _POINTWISE_TENSOR_ARGS has a builder: positional AND the
        classic pybind keyword call styles both produce a first-class node."""
        for op, argnames in pygraph._POINTWISE_TENSOR_ARGS.items():
            for style in ("positional", "keyword"):
                g = pygraph()
                tensors = [g.tensor(dim=[4, 8], name=f"t{i}") for i in range(len(argnames))]
                builder = getattr(g, op)
                out = builder(*tensors) if style == "positional" else builder(**dict(zip(argnames, tensors)))
                (node,) = g.nodes
                assert node.node_type == NodeType.POINTWISE, op
                assert node.params["mode"] == op
                assert len(node.inputs) == len(argnames), op
                assert out.dim == [] or out.dim == [4, 8]  # inferred at validate
                g.validate()
                # classic sequencing lowers (and freezes) at validate: sealed
                # dims are tuples — compare by value
                assert list(node.outputs["OUT_0"].dim) == [4, 8], op

    def test_all_structured_builders(self):
        """Every op in _STRUCTURED_OPS builds a first-class node: named ports
        (== C++ kwargs), attrs stored verbatim, declared outputs — via both
        keyword and positional-tensor call styles."""
        from cudnn._pygraph import _STRUCTURED_OPS

        for op, spec in _STRUCTURED_OPS.items():
            for style in ("keyword", "positional"):
                g = pygraph()
                tensors = {port: g.tensor(dim=[4, 8], name=f"{port}_in") for port in spec["inputs"]}
                attrs = {ak: "ATTR_SENTINEL" for ak in spec.get("attrs", ())}
                lists = {lp: [g.tensor(dim=[4, 8], name=f"{lp}{i}_in") for i in range(2)] for lp in spec.get("list_inputs", ())}
                if style == "keyword":
                    outs = getattr(g, op)(**tensors, **attrs, **lists)
                else:
                    outs = getattr(g, op)(*tensors.values(), **attrs, **lists)
                outs = outs if isinstance(outs, (tuple, list)) else (outs,)  # classic returns a LIST for multi-output
                (node,) = g.nodes
                assert node.node_type == spec["node_type"], op
                expect_ports = set(spec["inputs"]) | {f"{lp}_{i}" for lp in lists for i in range(2)}
                assert set(node.inputs) == expect_ports, op
                assert tuple(node.outputs) == spec["outputs"], op
                for ak in spec.get("attrs", ()):
                    assert node.params[ak] == "ATTR_SENTINEL", op
                assert len(outs) == len(spec["outputs"]), op

    def test_structured_out_dims(self):
        """out_dims sets output dims for shapes cuDNN cannot infer (reduction)."""
        g = pygraph()
        A = g.tensor(dim=[1, 4, 8], name="A")
        R = g.reduction(A, mode="ADD_SENTINEL", out_dims=[1, 4, 1])
        assert R.dim == [1, 4, 1] and R.stride == [4, 1, 1]

    def test_batchnorm_peer_stats_ports(self):
        """List inputs (peer_stats) become indexed ports + a count param."""
        g = pygraph()
        kwargs = {p: g.tensor(dim=[4, 8], name=p) for p in ("input", "scale", "bias", "epsilon", "momentum", "in_running_mean", "in_running_var")}
        ps = [g.tensor(dim=[4, 8], name=f"ps{i}") for i in range(2)]
        g.batchnorm(peer_stats=ps, **kwargs)
        (node,) = g.nodes
        assert node.params["_n_peer_stats"] == 2
        assert "peer_stats_0" in node.inputs and "peer_stats_1" in node.inputs

    def test_pointwise_scalar_attrs(self):
        """Ops with scalar attributes store them in params (introspectable)."""
        g = pygraph()
        X = g.tensor(dim=[4, 8], name="X")
        g.relu(X, lower_clip=0.1, upper_clip=6.0)
        g.leaky_relu(X, negative_slope=0.01)
        g.swish(X, swish_beta=1.5)
        g.gen_index(X, axis=1)
        r, lr, sw, gi = g.nodes
        assert r.params == {"mode": "relu", "lower_clip": 0.1, "upper_clip": 6.0}
        assert lr.params == {"mode": "leaky_relu", "negative_slope": 0.01}
        assert sw.params == {"mode": "swish", "swish_beta": 1.5}
        assert gi.params == {"mode": "gen_index", "axis": 1}

    def test_chaining(self):
        g = pygraph()
        A = g.tensor(dim=[8, 64, 128], name="A")
        B = g.tensor(dim=[8, 128, 256], name="B")
        bias = g.tensor(dim=[1, 1, 256], name="bias")

        C = g.matmul(A, B)
        D = g.add(C, bias)
        E = g.relu(D)

        assert len(g.nodes) == 3
        assert [n.node_type for n in g.nodes] == [NodeType.MATMUL, NodeType.POINTWISE, NodeType.POINTWISE]

    def test_get_node(self):
        g = pygraph()
        A = g.tensor(dim=[8, 64], name="A")
        B = g.tensor(dim=[64, 32], name="B")
        g.matmul(A, B, name="mm1")

        node = g.get_node("mm1")
        assert node.name == "mm1"
        assert g.get_node("nonexistent") is None

    def test_sdpa_inference(self):
        """Test SDPA forward inference mode."""
        g = pygraph()
        # [B, H, S, D] layout
        Q = g.tensor(dim=[2, 8, 128, 64], name="Q")
        K = g.tensor(dim=[2, 8, 128, 64], name="K")
        V = g.tensor(dim=[2, 8, 128, 64], name="V")

        O, stats = g.sdpa(Q, K, V, is_inference=True, use_causal_mask=True, name="attn")

        assert len(g.nodes) == 1
        assert g.nodes[0].node_type == NodeType.SDPA
        assert g.nodes[0].params["is_inference"] is True
        assert g.nodes[0].params["use_causal_mask"] is True
        assert "O" in g.nodes[0].outputs
        assert stats is None  # classic API returns [O, None] in inference mode
        assert O.dim == [2, 8, 128, 64]  # q dims with v's head dim

    def test_sdpa_training(self):
        """Test SDPA forward training mode (returns stats)."""
        g = pygraph()
        Q = g.tensor(dim=[2, 8, 128, 64], name="Q")
        K = g.tensor(dim=[2, 8, 128, 64], name="K")
        V = g.tensor(dim=[2, 8, 128, 64], name="V")

        O, stats = g.sdpa(Q, K, V, is_inference=False, attn_scale=0.125, name="attn")

        assert len(g.nodes) == 1
        assert g.nodes[0].params["is_inference"] is False
        assert g.nodes[0].params["attn_scale"] == 0.125
        assert "O" in g.nodes[0].outputs
        assert "Stats" in g.nodes[0].outputs
        assert stats.dim == [2, 8, 128, 1]


@pytest.mark.L1
class TestIntegration:
    """Integration tests requiring cuDNN."""

    @pytest.fixture
    def cudnn_available(self):
        try:
            import cudnn

            return cudnn.backend_version() >= 91200
        except Exception:
            return False

    def test_build(self, cudnn_available):
        if not cudnn_available:
            pytest.skip("cuDNN not available")

        import cudnn

        g = pygraph(
            io_data_type=cudnn.data_type.HALF,
            compute_data_type=cudnn.data_type.FLOAT,
        )
        A = g.tensor(dim=[8, 64, 128], name="A")
        B = g.tensor(dim=[8, 128, 256], name="B")
        C = g.matmul(A, B)
        C.set_output(True)

        g.build()
        assert g._is_built
        assert g.get_workspace_size() >= 0

    def test_sdpa_build(self, cudnn_available):
        """Test building SDPA graph."""
        if not cudnn_available:
            pytest.skip("cuDNN not available")

        import cudnn

        g = pygraph(
            io_data_type=cudnn.data_type.HALF,
            compute_data_type=cudnn.data_type.FLOAT,
        )
        # [B, H, S, D] layout
        Q = g.tensor(dim=[2, 8, 128, 64], name="Q")
        K = g.tensor(dim=[2, 8, 128, 64], name="K")
        V = g.tensor(dim=[2, 8, 128, 64], name="V")

        O, _ = g.sdpa(Q, K, V, is_inference=True, use_causal_mask=True, name="attn")
        O.set_output(True)

        try:
            g.build()
            assert g._is_built
        except cudnn.cudnnGraphNotSupportedError:
            pytest.skip("SDPA not supported on this hardware/configuration")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestReviewSemantics:
    """Review items 3 + 4: SDPA port direction; graph-owned identity mutation."""

    def test_sdpa_output_direction(self):
        """dBias & co. are outputs of the node, not inputs (review item 3)."""
        g = pygraph()
        t = lambda n: g.tensor(dim=[2, 4, 8, 16], name=n)  # noqa: E731
        dbias = g.tensor(dim=[1, 4, 8, 8], name="dbias_buf")
        g.sdpa_backward(t("q"), t("k"), t("v"), t("o"), t("dO"), t("stats"), dBias=dbias)
        (node,) = g.nodes
        assert "dBias" in node.outputs and node.outputs["dBias"] is dbias
        assert "dBias" not in node.inputs

    def test_tensor_rename_reindexes(self):
        g = pygraph()
        a = g.tensor(dim=[2, 2], name="old")
        a.set_name("new")
        assert g.find_tensor("new") is a and g.find_tensor("old") is None
        g.tensor(dim=[2, 2], name="other")
        a.set_name("other")  # classic: labels may collide — becomes ambiguous
        with pytest.raises(ValueError, match="ambiguous"):
            g.find_tensor("other")

    def test_set_uid_steals_auto_uid_and_rejects_user_dup(self):
        """Classic parity: user set_uid wins over an auto-assigned holder (which
        is silently renumbered); two USER uids colliding is an error."""
        g = pygraph()
        a = torch.randn(2, 2)
        A = g.tensor_like(a, name="A")  # auto uid 1
        g._data_bindings[A.uid] = a  # simulate auto-binding
        B = g.tensor(dim=[2, 2], name="B")  # auto uid 2
        B.set_uid(A.uid)  # user claims A's auto uid
        assert B.uid_assigned and g.find_tensor(B.uid) is B
        assert A.uid != B.uid and g.find_tensor(A.uid) is A  # A renumbered
        assert g._data_bindings.get(A.uid) is a  # binding followed A
        C = g.tensor(dim=[2, 2], name="C")
        with pytest.raises(ValueError, match="user-assigned"):
            C.set_uid(B.uid)

    def test_tensor_dict_key_stable_across_mutation(self):
        """Identity-based hashing: a Tensor used as a dict key survives
        uid/name mutation (review item 4)."""
        g = pygraph()
        A = g.tensor(dim=[2, 2], name="A")
        d = {A: "x"}
        A.set_name("renamed")
        A.set_uid(1000)
        assert d[A] == "x"

    def test_identity_mutation_frozen_after_planning(self):
        from cudnn.engines import BaseEngine, PYTHON_ENGINE_ID_BASE

        class Dummy(BaseEngine):
            engine_id = PYTHON_ENGINE_ID_BASE + 90

            def execute(self, graph, tensor_data, ctx=None):
                pass

        g = pygraph()
        g.register_backend(Dummy())  # keeps planning python-side (no C++ needed)
        A = g.tensor(dim=[1, 2, 2], name="A")
        g.matmul(A, g.tensor(dim=[1, 2, 2], name="B"))
        g.create_execution_plans()
        with pytest.raises(RuntimeError, match="frozen"):
            A.set_uid(500)

    def test_mxfp8_dsink_is_output(self):
        """Follow-up item 4: mxfp8_backward dSink_token is an output port."""
        g = pygraph()
        t = lambda n: g.tensor(dim=[2, 4, 8, 16], name=n)  # noqa: E731
        kw = {p: t(p) for p in ("q", "q_T", "k", "k_T", "v", "o_f16", "dO_f16", "dO", "dO_T", "stats")}
        ds = g.tensor(dim=[1, 4, 1, 1], name="dsink_buf")
        g.sdpa_mxfp8_backward(dSink_token=ds, **kw)
        (node,) = g.nodes
        assert "dSink_token" in node.outputs and "dSink_token" not in node.inputs

    def test_semantic_setters_frozen_after_planning(self):
        from cudnn.engines import BaseEngine, PYTHON_ENGINE_ID_BASE

        class Dummy(BaseEngine):
            engine_id = PYTHON_ENGINE_ID_BASE + 91

            def execute(self, graph, tensor_data, ctx=None):
                pass

        g = pygraph(backends=[Dummy()])
        A = g.tensor(dim=[1, 2, 2], name="A")
        g.matmul(A, g.tensor(dim=[1, 2, 2], name="B"))
        g.create_execution_plans()
        for mutate in (lambda: A.set_dim([4, 4]), lambda: A.set_data_type("HALF"), lambda: A.set_output(True), lambda: A.set_stride([4, 1])):
            with pytest.raises(RuntimeError, match="frozen"):
                mutate()

    def test_freeze_covers_public_surface(self):
        """Review round 5: the freeze must close EVERY public mutation path,
        not only the fluent API — attribute writes, live containers, in-place
        list edits, node params, and graph context."""
        from cudnn.engines import BaseEngine, PYTHON_ENGINE_ID_BASE

        class Dummy(BaseEngine):
            engine_id = PYTHON_ENGINE_ID_BASE + 92

            def execute(self, graph, tensor_data, ctx=None):
                pass

        g = pygraph(backends=[Dummy()])
        A = g.tensor(dim=[1, 2, 2], name="A")
        C = g.matmul(A, g.tensor(dim=[1, 2, 2], name="B"))
        g.create_execution_plans()
        node = g.nodes[0]

        with pytest.raises(RuntimeError, match="frozen"):
            A.dim = [9, 9]  # direct attribute write
        with pytest.raises(TypeError):
            A.dim[:] = [9]  # sealed to a tuple: no in-place edits
        with pytest.raises(TypeError):
            node.params["padding"] = 123  # MappingProxy
        with pytest.raises(TypeError):
            node.inputs["A"] = C  # MappingProxy
        with pytest.raises(RuntimeError, match="frozen"):
            node.inputs = {}  # attribute write on the node
        with pytest.raises(RuntimeError, match="frozen"):
            g.context.compute_data_type = "HALF"  # graph context
        # live-container laundering: the public views are copies
        g.nodes.clear()
        g.tensors.clear()
        assert len(g.nodes) == 1 and len(g.tensors) == 3
        # the inspection surface stays readable for engines
        assert list(node.inputs) == ["A", "B"] and list(C.dim) == [1, 2, 2]

    def test_mutation_after_validate_revalidates(self):
        """Review round 5: python-engine graphs stay mutable until planning —
        but a mutation after validate() must invalidate _is_validated so stale
        inference never reaches planning."""
        from cudnn.engines import BaseEngine, PYTHON_ENGINE_ID_BASE

        class Dummy(BaseEngine):
            engine_id = PYTHON_ENGINE_ID_BASE + 93

            def execute(self, graph, tensor_data, ctx=None):
                pass

        g = pygraph(backends=[Dummy()])
        A = g.tensor(dim=[1, 2, 2], name="A")
        g.matmul(A, g.tensor(dim=[1, 2, 2], name="B"))
        g.validate()
        assert g._is_validated and not g._frozen  # mutable until planning
        A.set_data_type("HALF")  # allowed — and must force re-validation
        assert not g._is_validated
        g.create_execution_plans()
        assert g._frozen

    def test_tensor_scalar_is_graph_owned(self):
        g = pygraph()
        s = g.tensor_scalar(1.5, scalar_type="FLOAT_SENTINEL")
        s.set_name("renamed_scalar")
        assert g.find_tensor("renamed_scalar") is s

    def test_duplicate_names_are_classic_labels(self):
        """Classic parity: tensor names are debug labels — duplicates are legal
        (pycudnnTest builds two tensors both named 'weight'). uid is the
        identity; name-keyed lookups on an ambiguous label raise instead of
        guessing, unique labels keep working."""
        g = pygraph()
        a = g.tensor(dim=[2, 2], name="X")
        b = g.tensor(dim=[2, 2], name="X")  # legal, like classic
        assert a.name == b.name == "X" and a.uid != b.uid
        with pytest.raises(ValueError, match="ambiguous"):
            g.find_tensor("X")
        assert g.find_tensor(a.uid) is a and g.find_tensor(b.uid) is b
        u = g.tensor(dim=[2, 2], name="unique")
        assert g.find_tensor("unique") is u
        # renaming ONTO an existing label is equally legal — and makes it ambiguous
        u.set_name("X")
        with pytest.raises(ValueError, match="ambiguous"):
            g.find_tensor("X")

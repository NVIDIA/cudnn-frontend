# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cudnn_repro.utils import flatten_pass_by_value


def test_flatten_pass_by_value_valid_hex():
    assert flatten_pass_by_value("0x10") == [16]


def test_flatten_pass_by_value_valid_decimal():
    assert flatten_pass_by_value("42") == [42]


def test_flatten_pass_by_value_malformed_hex_prefix_only():
    assert flatten_pass_by_value("0x") == []


def test_flatten_pass_by_value_malformed_hex_digits():
    assert flatten_pass_by_value("0xZZ") == []


def test_flatten_pass_by_value_malformed_decimal():
    assert flatten_pass_by_value("abc") == []


def test_flatten_pass_by_value_list_with_malformed_entries():
    assert flatten_pass_by_value(["0x", "0x10", "7", "0xZZ"]) == [16, 7]


def test_flatten_pass_by_value_none_and_numbers():
    assert flatten_pass_by_value(None) == []
    assert flatten_pass_by_value(3) == [3]
    assert flatten_pass_by_value(2.0) == [2]

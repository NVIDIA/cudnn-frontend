# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.

import enum


class NamedBarrierBwd(enum.IntEnum):
    Epilogue = enum.auto()
    WarpSchedulerWG1 = enum.auto()
    WarpSchedulerWG2 = enum.auto()
    WarpSchedulerWG3 = enum.auto()
    PdS = enum.auto()
    dQFullWG0 = enum.auto()
    dQFullWG1 = enum.auto()
    dQFullWG2 = enum.auto()
    dQEmptyWG0 = enum.auto()
    dQEmptyWG1 = enum.auto()
    dQEmptyWG2 = enum.auto()
    dQMetadata = enum.auto()

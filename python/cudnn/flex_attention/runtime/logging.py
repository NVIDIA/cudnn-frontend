# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025, Tri Dao.

"""Host-side FlexAttention logging controlled by ``FLEX_ATTN_LOG_LEVEL``.

Host-side messages go through Python ``logging`` (logger name
``cudnn.flex_attention``).
A default ``StreamHandler`` is attached automatically when ``FLEX_ATTN_LOG_LEVEL >= 1``
so that standalone scripts get output without extra setup; applications that
configure their own logging can remove or replace it via the standard API.

FLEX_ATTN_LOG_LEVEL mapping::

    0  off       nothing logged
    1+ host      host-side summaries only (no kernel printf)

Set via environment variable::

    FLEX_ATTN_LOG_LEVEL=1 python train.py

Device-side printing is intentionally unsupported in product kernels.
"""

import logging
import os
import sys

_LOG_LEVEL_NAMES = {"off": 0, "host": 1, "verbose": 2, "max": 3}


def _parse_log_level(raw: str) -> int:
    if raw in _LOG_LEVEL_NAMES:
        return _LOG_LEVEL_NAMES[raw]
    try:
        level = int(raw)
    except ValueError:
        return 0
    return max(0, min(level, 3))


_flex_log_level: int = _parse_log_level(os.environ.get("FLEX_ATTN_LOG_LEVEL", "0"))

_logger = logging.getLogger("cudnn.flex_attention")
_logger.addHandler(logging.NullHandler())
_default_handler: logging.Handler | None = None


def _configure_default_handler() -> None:
    global _default_handler
    if _flex_log_level >= 1:
        if _default_handler is None:
            _default_handler = logging.StreamHandler(sys.stdout)
            _default_handler.setFormatter(logging.Formatter("[FlexAttn] %(message)s"))
            _logger.addHandler(_default_handler)
        _logger.setLevel(logging.DEBUG)
    else:
        if _default_handler is not None:
            _logger.removeHandler(_default_handler)
            _default_handler = None
        _logger.setLevel(logging.WARNING)


_configure_default_handler()


def flex_log(msg: str):
    if _flex_log_level >= 1:
        _logger.info(msg)

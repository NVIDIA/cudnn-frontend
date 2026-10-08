# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import sys
import warnings

emitted_deprecations = set()


def deprecated(message):
    """Warn ``message`` as a DeprecationWarning on first use, like Python 3.13's ``warnings.deprecated``.

    Decorates a function (warns when called) or a class (warns when instantiated).
    Each message is emitted once per process so wrapper hot paths only pay a set lookup.
    Calls from inside the cudnn package do not warn, so a deprecated wrapper built on a
    deprecated class reports only itself.
    """

    def decorate(api):
        target = api.__init__ if isinstance(api, type) else api

        @functools.wraps(target)
        def wrapper(*args, **kwargs):
            if message not in emitted_deprecations and not sys._getframe(1).f_globals.get("__name__", "").startswith("cudnn."):
                emitted_deprecations.add(message)
                warnings.warn(message, DeprecationWarning, stacklevel=2)
            return target(*args, **kwargs)

        if isinstance(api, type):
            api.__init__ = wrapper
            return api
        return wrapper

    return decorate


def reset_deprecation_warnings():
    """Forget emitted deprecations so tests can observe them again."""
    emitted_deprecations.clear()

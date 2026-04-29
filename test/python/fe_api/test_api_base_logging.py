import logging

import pytest

from cudnn.api_base import _reset_experimental_api_warning_registry, warn_experimental_api_once


@pytest.mark.L0
def test_experimental_api_warning_emits_once_per_api(caplog):
    logger = logging.getLogger("cudnn.test.experimental")
    _reset_experimental_api_warning_registry()

    try:
        with caplog.at_level(logging.WARNING, logger=logger.name):
            warn_experimental_api_once(logger, "FirstExperimentalApi")
            warn_experimental_api_once(logger, "FirstExperimentalApi")
            warn_experimental_api_once(logger, "SecondExperimentalApi")

        messages = [record.getMessage() for record in caplog.records]
        assert messages == [
            "FirstExperimentalApi is an experimental API",
            "SecondExperimentalApi is an experimental API",
        ]
    finally:
        _reset_experimental_api_warning_registry()

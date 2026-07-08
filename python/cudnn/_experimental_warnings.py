import logging
import threading

_experimental_api_warnings_emitted = set()
_experimental_api_warnings_lock = threading.Lock()


def warn_experimental_api_once(logger: logging.Logger, api_name: str) -> None:
    """Emit the experimental API warning once per API class per process."""
    with _experimental_api_warnings_lock:
        if api_name in _experimental_api_warnings_emitted:
            return
        _experimental_api_warnings_emitted.add(api_name)

    logger.warning("%s is an experimental API", api_name)


def _reset_experimental_api_warning_registry() -> None:
    """Reset experimental API warning state for tests."""
    with _experimental_api_warnings_lock:
        _experimental_api_warnings_emitted.clear()

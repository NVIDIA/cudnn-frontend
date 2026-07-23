from importlib import import_module

_SYMBOLS = {
    "CSACompressorForward": (".compressor", "CSACompressorForward"),
    "CSACompressorBackward": (".compressor", "CSACompressorBackward"),
    "csa_compressor_forward_wrapper": (".compressor", "csa_compressor_forward_wrapper"),
    "csa_compressor_backward_wrapper": (".compressor", "csa_compressor_backward_wrapper"),
}


def _load_symbol(name):
    module_name, symbol_name = _SYMBOLS[name]
    module = import_module(module_name, package=__name__)
    symbol = getattr(module, symbol_name)
    globals()[name] = symbol
    return symbol


def __getattr__(name):
    if name == "CSA":
        return CSA
    if name in _SYMBOLS:
        return _load_symbol(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class CSANamespace:
    def __getattr__(self, name):
        if name in _SYMBOLS:
            return _load_symbol(name)
        raise AttributeError(f"CSA has no attribute {name!r}")


CSA = CSANamespace()

__all__ = [
    "CSA",
    "CSACompressorBackward",
    "CSACompressorForward",
    "csa_compressor_backward_wrapper",
    "csa_compressor_forward_wrapper",
]

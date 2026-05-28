r"""
Errors raised by OOps operators.
"""

from __future__ import annotations


class OOpsUnsupportedError(NotImplementedError):
    r"""Signal that an operator cannot serve the requested execution contract.

    Args:
        *args (object): Exception arguments, usually a descriptive error message.

    Raises:
        TypeError: Keyword arguments are supplied to the exception constructor.

    Notes:
        Instance arguments are available through ``args``.

        This is a ``NotImplementedError`` subclass. Operators use it for unsupported
        devices, dtypes, layouts, launch limits, differentiation modes, or precision
        states. Invalid argument relationships may instead raise ``ValueError`` or
        ``TypeError``. Functional kernel APIs generally propagate this exception;
        modules with a documented native fallback may catch it and evaluate their
        PyTorch composition.

    Examples:
        >>> from oops import OOpsUnsupportedError
        >>> error = OOpsUnsupportedError("unsupported device")
        >>> isinstance(error, NotImplementedError)
        True
        >>> str(error)
        'unsupported device'
    """

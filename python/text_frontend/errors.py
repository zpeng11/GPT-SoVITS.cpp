"""Typed errors for the text frontend.

The CLI and library intentionally raise small, typed exceptions so callers can
distinguish missing optional dependencies from missing local data files and
from malformed input.
"""


class TextFrontendError(Exception):
    """Base class for all frontend errors."""


class EmptyTextError(TextFrontendError, ValueError):
    """Raised when the input has no usable text after normalization."""


class MissingDependencyError(TextFrontendError, ImportError):
    """Raised when a lazily imported optional/runtime dependency is absent."""

    def __init__(self, module, purpose, install_hint=None):
        self.module = module
        self.purpose = purpose
        hint = install_hint or f"python -m pip install {module}"
        super().__init__(
            f"{module} is required for {purpose}, but it is not installed. "
            f"Install it with: {hint}"
        )


class ResourceError(TextFrontendError, FileNotFoundError):
    """Raised when a local, package-scoped resource is missing or invalid."""


class UnsupportedLanguageError(TextFrontendError, ValueError):
    """Raised for an unknown --language mode."""


class UnsupportedScriptError(TextFrontendError, ValueError):
    """Raised when mixed/forced text contains a script this slice lacks."""


class UnknownWordError(TextFrontendError, ValueError):
    """Raised in strict mode when a word/character cannot be pronounced."""


class TokenizerError(TextFrontendError, ValueError):
    """Raised for malformed or unsupported tokenizer resources."""


class LayoutError(TextFrontendError, ValueError):
    """Raised when an array does not satisfy the C++/ggml layout contract."""

"""Typed exceptions so callers can handle failure precisely instead of guessing."""
from __future__ import annotations


class SageError(Exception):
    """Base class for every error raised by Sage."""


class ConfigError(SageError):
    """Missing or invalid configuration (e.g. no API key)."""


class UnsupportedFileError(SageError):
    """A file type Sage cannot load."""


class FileTooLargeError(SageError):
    """An upload exceeds the configured size limit."""


class UnsafeFilenameError(SageError):
    """A filename that would escape the documents directory (path traversal)."""


class IndexNotReadyError(SageError):
    """A query was issued before any document had been indexed."""

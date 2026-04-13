"""Parsing package for document extraction using Docling."""

from .docling_parser import (
    DoclingParser,
    ParsedDocument,
    StructureNode,
    TableData,
)
from .errors import (
    CorruptedDocumentError,
    ParseError,
    ParseTimeoutError,
    UnsupportedFormatParseError,
)

__all__ = [
    # Parser
    "DoclingParser",
    "ParsedDocument",
    "StructureNode",
    "TableData",
    # Errors
    "ParseError",
    "ParseTimeoutError",
    "UnsupportedFormatParseError",
    "CorruptedDocumentError",
]

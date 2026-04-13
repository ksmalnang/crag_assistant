"""Metadata resolution with filename convention parser and document metadata."""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class Faculty(str, Enum):
    """Faculty/department enumeration."""

    ENGINEERING = "engineering"
    SCIENCE = "science"
    ARTS = "arts"
    BUSINESS = "business"
    MEDICINE = "medicine"
    LAW = "law"
    EDUCATION = "education"
    OTHER = "other"


class DocType(str, Enum):
    """Document type enumeration."""

    LECTURE = "lecture"
    TUTORIAL = "tutorial"
    LAB = "lab"
    ASSIGNMENT = "assignment"
    EXAM = "exam"
    PROJECT = "project"
    THESIS = "thesis"
    SYLLABUS = "syllabus"
    READING = "reading"
    OTHER = "other"


class DocumentStatus(str, Enum):
    """Document ingestion status."""

    PENDING = "pending"
    INGESTED = "ingested"
    FAILED = "failed"
    STALE = "stale"


# Valid semester pattern: YYYY-S1 or YYYY-S2
SEMESTER_PATTERN = re.compile(r"^\d{4}-S[12]$")


@dataclass
class FilenameMetadata:
    """Parsed metadata from filename convention."""

    faculty: Faculty
    semester: str
    doc_type: DocType
    display_name: str
    source_name: str
    document_id: str
    unclassified: bool = False

    @classmethod
    def from_filename(
        cls,
        file_path: str | Path,
        file_bytes: Optional[bytes] = None,
    ) -> "FilenameMetadata":
        """
        Parse filename using convention: {faculty}_{semester}_{doc_type}_{display_name}.ext

        Fallback chain:
        1. Parse filename convention
        2. If malformed, use PDF document properties (TODO: implement)
        3. If that fails, use folder path segments
        4. Final fallback: OTHER/unknown

        Args:
            file_path: Path to the file
            file_bytes: Raw file bytes for document_id computation

        Returns:
            FilenameMetadata with parsed values
        """
        file_path = Path(file_path)
        filename = file_path.stem  # Without extension

        # Parse filename segments
        parts = filename.split("_")

        # Try to extract metadata from filename
        faculty = cls._parse_faculty(parts)
        semester = cls._parse_semester(parts)
        doc_type = cls._parse_doc_type(parts)
        display_name = cls._parse_display_name(parts)

        # Check if filename was malformed
        unclassified = cls._is_unclassified(parts, faculty, semester, doc_type)

        if unclassified:
            logger.warning(
                f"Filename '{filename}' is malformed or has unrecognised segments. "
                f"Using fallback values. Parts: {parts}"
            )

        # Generate display name (underscores to spaces, title case)
        source_name = display_name.replace("_", " ").title()

        # Compute document_id from file bytes
        if file_bytes is not None:
            document_id = hashlib.sha256(file_bytes).hexdigest()
        else:
            # Fallback: hash the file path
            document_id = hashlib.sha256(str(file_path).encode()).hexdigest()
            logger.warning(
                f"file_bytes not provided for {file_path}. "
                f"document_id based on path hash (less stable)."
            )

        return cls(
            faculty=faculty,
            semester=semester,
            doc_type=doc_type,
            display_name=display_name,
            source_name=source_name,
            document_id=document_id,
            unclassified=unclassified,
        )

    @staticmethod
    def _parse_faculty(parts: list[str]) -> Faculty:
        """Parse faculty from filename parts."""
        if len(parts) < 1:
            logger.warning("No faculty segment found, defaulting to OTHER")
            return Faculty.OTHER

        faculty_str = parts[0].lower()

        # Try to match known faculty
        for faculty in Faculty:
            if faculty.value == faculty_str:
                return faculty

        logger.warning(f"Unrecognised faculty: '{parts[0]}'. Defaulting to OTHER")
        return Faculty.OTHER

    @staticmethod
    def _parse_semester(parts: list[str]) -> str:
        """Parse semester from filename parts."""
        if len(parts) < 2:
            logger.warning("No semester segment found, defaulting to 'unknown'")
            return "unknown"

        semester_str = parts[1]

        if SEMESTER_PATTERN.match(semester_str):
            return semester_str

        logger.warning(
            f"Invalid semester format: '{semester_str}'. Expected YYYY-S1 or YYYY-S2. "
            f"Defaulting to 'unknown'"
        )
        return "unknown"

    @staticmethod
    def _parse_doc_type(parts: list[str]) -> DocType:
        """Parse document type from filename parts."""
        if len(parts) < 3:
            logger.warning("No doc_type segment found, defaulting to OTHER")
            return DocType.OTHER

        doc_type_str = parts[2].lower()

        # Try to match known doc types
        for doc_type in DocType:
            if doc_type.value == doc_type_str:
                return doc_type

        logger.warning(f"Unrecognised doc_type: '{parts[2]}'. Defaulting to OTHER")
        return DocType.OTHER

    @staticmethod
    def _parse_display_name(parts: list[str]) -> str:
        """Parse display name from filename parts."""
        if len(parts) < 4:
            logger.warning("No display_name segment found, using filename stem")
            return "_".join(parts[3:]) if len(parts) > 3 else "unknown"

        return "_".join(parts[3:])

    @staticmethod
    def _is_unclassified(
        parts: list[str],
        faculty: Faculty,
        semester: str,
        doc_type: DocType,
    ) -> bool:
        """Check if filename was malformed or has unrecognised segments."""
        # Classified if we have at least 4 parts and all recognized
        has_enough_parts = len(parts) >= 4
        all_recognised = (
            faculty != Faculty.OTHER
            and semester != "unknown"
            and doc_type != DocType.OTHER
        )

        return not (has_enough_parts and all_recognised)

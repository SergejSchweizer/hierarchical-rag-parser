from __future__ import annotations

from pathlib import Path
import re
from typing import Protocol

import fitz

from models import TextBlock

NUMBERING_PATTERN = re.compile(r"^\d+(?:\.\d+)*[\.\)]?(?:\s|$)")


class PDFParser(Protocol):
    """Strategy interface for PDF parsing implementations.

    Design pattern:
    This protocol represents the Strategy pattern in a lightweight,
    Pythonic form. The rest of the pipeline depends on the behavior
    "extract ordered text blocks from a PDF" instead of depending on
    one fixed library implementation.

    Why this helps:
    - We can swap the parsing backend later without changing the pipeline.
    - Tests can use a fake parser if needed.
    - The orchestration code stays focused on data flow, not on library details.

    Current implementation:
    `PyMuPDFParser` is the concrete strategy used right now.
    """

    def extract_blocks(
        self,
        pdf_path: str | Path,
        max_pages: int | None = None,
    ) -> list[TextBlock]:
        """Extract ordered text blocks from a PDF document."""


class PyMuPDFParser:
    """Concrete PDF parser strategy backed by PyMuPDF.

    Design pattern:
    This is the concrete Strategy implementation for the `PDFParser`
    protocol.

    How it works:
    - opens the PDF with PyMuPDF
    - reads page-level text blocks
    - normalizes whitespace
    - discards empty blocks
    - converts each block into a `TextBlock`
    - returns blocks in reading order

    Why this class exists:
    It isolates the dependency on PyMuPDF in one place. That keeps the
    library-specific extraction logic separate from downstream tasks like
    layout classification or document tree construction.
    """

    def extract_blocks(
        self,
        pdf_path: str | Path,
        max_pages: int | None = None,
    ) -> list[TextBlock]:
        """Extract non-empty text blocks from a PDF and sort them by reading order."""
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")

        blocks: list[TextBlock] = []
        with fitz.open(path) as document:
            for page_index, page in enumerate(document, start=1):
                if max_pages is not None and page_index > max_pages:
                    break

                blocks.extend(_extract_page_blocks(page, page_index))

        return _sort_blocks(blocks)


def extract_text_blocks(
    pdf_path: str | Path,
    max_pages: int | None = None,
) -> list[TextBlock]:
    """Convenience entrypoint for the parsing step of the pipeline.

    Design pattern:
    This function supports the overall Pipeline style of the application:
    each stage exposes a simple transformation function while the underlying
    implementation can still be swapped through a strategy class.

    Why keep this function:
    It gives callers a very small API for the common path while preserving
    the flexibility of the strategy-based parser underneath.
    """

    parser: PDFParser = PyMuPDFParser()
    return parser.extract_blocks(pdf_path, max_pages=max_pages)


def _normalize_text(text: str) -> str:
    """Normalize block text by trimming lines and removing empty ones."""
    lines = [line.strip() for line in text.splitlines()]
    non_empty_lines = [line for line in lines if line]
    return "\n".join(non_empty_lines).strip()


def _sort_blocks(blocks: list[TextBlock]) -> list[TextBlock]:
    """Sort blocks by page, vertical position, and horizontal position.

    This is a minimal reading-order heuristic. We keep it deliberately
    simple so the parser stays predictable and easy to explain.
    """
    return sorted(blocks, key=lambda block: (block.page, block.bbox[1], block.bbox[0]))


def _extract_page_blocks(page: fitz.Page, page_index: int) -> list[TextBlock]:
    """Extract text blocks with lightweight font-size metadata from one page."""
    page_dict = page.get_text("dict")
    extracted_blocks: list[TextBlock] = []
    page_width = float(page.rect.width) if page.rect.width else 0.0
    page_height = float(page.rect.height) if page.rect.height else 0.0

    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue

        text_parts: list[str] = []
        font_sizes: list[float] = []

        for line in block.get("lines", []):
            line_parts: list[str] = []
            for span in line.get("spans", []):
                span_text = str(span.get("text", "")).strip()
                if not span_text:
                    continue

                line_parts.append(span_text)
                span_size = span.get("size")
                if isinstance(span_size, (int, float)):
                    font_sizes.append(float(span_size))

            if line_parts:
                text_parts.append(" ".join(line_parts))

        cleaned_text = _normalize_text("\n".join(text_parts))
        if not cleaned_text:
            continue

        bbox = block.get("bbox", (0.0, 0.0, 0.0, 0.0))
        x0, y0, x1, y1 = bbox
        font_size = max(font_sizes) if font_sizes else None
        line_count = len(text_parts)
        block_width = max(float(x1) - float(x0), 0.0)
        block_center_x = (float(x0) + float(x1)) / 2.0
        page_center_x = page_width / 2.0 if page_width else 0.0
        width_ratio = block_width / page_width if page_width else None
        y_position_ratio = float(y0) / page_height if page_height else None
        is_centered = (
            page_width > 0.0
            and abs(block_center_x - page_center_x) <= page_width * 0.1
        )
        normalized_text = " ".join(cleaned_text.split())
        ends_with_period = normalized_text.endswith(".")
        is_numbered = bool(NUMBERING_PATTERN.match(normalized_text))
        uppercase_ratio = _compute_uppercase_ratio(normalized_text)

        extracted_blocks.append(
            TextBlock(
                text=cleaned_text,
                page=page_index,
                bbox=(float(x0), float(y0), float(x1), float(y1)),
                font_size=font_size,
                line_count=line_count,
                y_position_ratio=y_position_ratio,
                width_ratio=width_ratio,
                is_centered=is_centered,
                is_numbered=is_numbered,
                ends_with_period=ends_with_period,
                uppercase_ratio=uppercase_ratio,
            )
        )

    return extracted_blocks


def _compute_uppercase_ratio(text: str) -> float | None:
    """Compute the ratio of uppercase letters among alphabetic characters."""
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return None

    uppercase_letters = [char for char in letters if char.isupper()]
    return len(uppercase_letters) / len(letters)

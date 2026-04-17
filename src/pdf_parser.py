from __future__ import annotations

from pathlib import Path
from typing import Protocol

import fitz

from models import TextBlock


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

    def extract_blocks(self, pdf_path: str | Path) -> list[TextBlock]:
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

    def extract_blocks(self, pdf_path: str | Path) -> list[TextBlock]:
        """Extract non-empty text blocks from a PDF and sort them by reading order."""
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")

        blocks: list[TextBlock] = []
        with fitz.open(path) as document:
            for page_index, page in enumerate(document, start=1):
                page_blocks = page.get_text("blocks")
                for block in page_blocks:
                    x0, y0, x1, y1, text, *_ = block
                    cleaned_text = _normalize_text(text)
                    if not cleaned_text:
                        continue

                    blocks.append(
                        TextBlock(
                            text=cleaned_text,
                            page=page_index,
                            bbox=(float(x0), float(y0), float(x1), float(y1)),
                        )
                    )

        return _sort_blocks(blocks)


def extract_text_blocks(pdf_path: str | Path) -> list[TextBlock]:
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
    return parser.extract_blocks(pdf_path)


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

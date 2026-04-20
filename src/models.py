from __future__ import annotations

from enum import StrEnum
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class TextBlock:
    """Domain object for one extracted PDF text block.

    Design intent:
    This class is part of the domain model used by the parsing pipeline.
    In this project we keep domain data in small dataclasses instead of
    mixing raw dictionaries throughout the codebase. That makes the
    pipeline easier to reason about, test, and extend.

    Role in the architecture:
    `TextBlock` is produced by the PDF parser and later consumed by the
    layout classifier and structure builder. It is intentionally small:
    only the data required by downstream steps is stored here.
    """

    text: str
    page: int
    bbox: tuple[float, float, float, float]
    font_size: float | None = None
    line_count: int = 1
    y_position_ratio: float | None = None
    width_ratio: float | None = None
    is_centered: bool = False
    is_numbered: bool = False
    ends_with_period: bool = False
    uppercase_ratio: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation for serialization."""
        return asdict(self)


class BlockLabel(StrEnum):
    """Supported structural labels for extracted PDF blocks.

    These labels are intentionally coarse. Keeping the label space small
    makes the first version of the classifier easier to implement,
    inspect, and test.
    """

    TITLE = "title"
    SECTION = "section"
    SUBSECTION = "subsection"
    PARAGRAPH = "paragraph"
    OTHER = "other"


@dataclass(slots=True)
class LabeledBlock:
    """A parsed text block enriched with a structural label.

    Design intent:
    This object forms the handoff between the layout classification step
    and the later structure-building step. It preserves the raw block and
    adds exactly one predicted label.
    """

    block: TextBlock
    label: BlockLabel

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation for serialization."""
        return {
            "text": self.block.text,
            "page": self.block.page,
            "bbox": self.block.bbox,
            "font_size": self.block.font_size,
            "y_position_ratio": self.block.y_position_ratio,
            "label": self.label.value,
        }


@dataclass(slots=True)
class SubsectionNode:
    """A subsection heading and the paragraph-like blocks grouped under it."""

    heading: str
    page: int
    blocks: list[LabeledBlock]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation for serialization."""
        return {
            "heading": self.heading,
            "page": self.page,
            "blocks": [block.to_dict() for block in self.blocks],
        }


@dataclass(slots=True)
class SectionNode:
    """A top-level section containing direct blocks and nested subsections."""

    heading: str
    page: int
    blocks: list[LabeledBlock]
    subsections: list[SubsectionNode]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation for serialization."""
        return {
            "heading": self.heading,
            "page": self.page,
            "blocks": [block.to_dict() for block in self.blocks],
            "subsections": [subsection.to_dict() for subsection in self.subsections],
        }


@dataclass(slots=True)
class DocumentTree:
    """Structured document representation built from labeled blocks."""

    title: str | None
    title_page: int | None
    preamble: list[LabeledBlock]
    sections: list[SectionNode]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation for serialization."""
        return {
            "title": self.title,
            "title_page": self.title_page,
            "preamble": [block.to_dict() for block in self.preamble],
            "sections": [section.to_dict() for section in self.sections],
        }


@dataclass(slots=True)
class DocumentChunk:
    """Chunked document content enriched with structural ancestry metadata."""

    chunk_id: str
    text: str
    title: str | None
    level: str
    parent_section_id: str | None
    parent_subsection_id: str | None
    section_heading: str | None
    subsection_heading: str | None
    page_start: int | None
    page_end: int | None
    block_count: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation for serialization."""
        return asdict(self)

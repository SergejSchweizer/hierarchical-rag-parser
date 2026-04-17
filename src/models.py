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
            "label": self.label.value,
        }

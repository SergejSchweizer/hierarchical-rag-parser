from __future__ import annotations

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

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation for serialization."""
        return asdict(self)

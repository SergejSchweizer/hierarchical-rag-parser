from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from chunker import build_chunks
from models import BlockLabel, LabeledBlock, TextBlock
from structure_builder import build_document_tree


def _labeled_block(text: str, label: BlockLabel, page: int = 1) -> LabeledBlock:
    return LabeledBlock(
        block=TextBlock(text=text, page=page, bbox=(0.0, 0.0, 1.0, 1.0)),
        label=label,
    )


def test_build_chunks_preserves_document_ancestry() -> None:
    blocks = [
        _labeled_block("Annual Report 2025", BlockLabel.TITLE),
        _labeled_block("Forward-looking statements.", BlockLabel.PARAGRAPH),
        _labeled_block("1 Business Overview", BlockLabel.SECTION),
        _labeled_block("The company operates in Europe.", BlockLabel.PARAGRAPH),
        _labeled_block("1.1 Strategy", BlockLabel.SUBSECTION),
        _labeled_block("Growth is driven by subscriptions.", BlockLabel.PARAGRAPH),
    ]

    chunks = build_chunks(build_document_tree(blocks))

    assert len(chunks) == 3
    assert chunks[0].chunk_id == "preamble-chunk"
    assert chunks[0].level == "preamble"
    assert chunks[0].parent_section_id is None
    assert chunks[0].parent_subsection_id is None
    assert chunks[0].section_heading is None
    assert chunks[0].subsection_heading is None
    assert chunks[0].text == "Forward-looking statements."
    assert chunks[1].chunk_id == "section-1-chunk"
    assert chunks[1].level == "section"
    assert chunks[1].parent_section_id == "section-1"
    assert chunks[1].parent_subsection_id is None
    assert chunks[1].section_heading == "1 Business Overview"
    assert chunks[1].subsection_heading is None
    assert chunks[2].chunk_id == "section-1-subsection-1-chunk"
    assert chunks[2].level == "subsection"
    assert chunks[2].parent_section_id == "section-1"
    assert chunks[2].parent_subsection_id == "section-1-subsection-1"
    assert chunks[2].section_heading == "1 Business Overview"
    assert chunks[2].subsection_heading == "1.1 Strategy"
    assert chunks[2].title == "Annual Report 2025"


def test_build_chunks_combines_multiple_blocks_and_tracks_page_range() -> None:
    blocks = [
        _labeled_block("Annual Report 2025", BlockLabel.TITLE, page=1),
        _labeled_block("1 Risk Factors", BlockLabel.SECTION, page=2),
        _labeled_block("Risk one.", BlockLabel.PARAGRAPH, page=2),
        _labeled_block("Risk two.", BlockLabel.PARAGRAPH, page=3),
    ]

    chunks = build_chunks(build_document_tree(blocks))

    assert len(chunks) == 1
    assert chunks[0].text == "Risk one.\n\nRisk two."
    assert chunks[0].page_start == 2
    assert chunks[0].page_end == 3
    assert chunks[0].block_count == 2


def test_build_chunks_skips_empty_groups() -> None:
    blocks = [
        _labeled_block("Annual Report 2025", BlockLabel.TITLE),
        _labeled_block("1 Empty Section", BlockLabel.SECTION),
    ]

    chunks = build_chunks(build_document_tree(blocks))

    assert chunks == []

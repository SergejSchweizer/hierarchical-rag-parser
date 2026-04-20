from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from models import BlockLabel, LabeledBlock, TextBlock
from structure_builder import build_document_tree


def _labeled_block(text: str, label: BlockLabel, page: int = 1) -> LabeledBlock:
    return LabeledBlock(
        block=TextBlock(text=text, page=page, bbox=(0.0, 0.0, 1.0, 1.0)),
        label=label,
    )


def test_build_document_tree_groups_title_preamble_sections_and_subsections() -> None:
    blocks = [
        _labeled_block("Annual Report 2025", BlockLabel.TITLE),
        _labeled_block("Forward-looking statements.", BlockLabel.PARAGRAPH),
        _labeled_block("1 Business Overview", BlockLabel.SECTION),
        _labeled_block("The company operates in Europe.", BlockLabel.PARAGRAPH),
        _labeled_block("1.1 Strategy", BlockLabel.SUBSECTION),
        _labeled_block("Growth is driven by subscriptions.", BlockLabel.PARAGRAPH),
    ]

    tree = build_document_tree(blocks)

    assert tree.title == "Annual Report 2025"
    assert tree.title_page == 1
    assert [block.block.text for block in tree.preamble] == ["Forward-looking statements."]
    assert len(tree.sections) == 1
    assert tree.sections[0].heading == "1 Business Overview"
    assert [block.block.text for block in tree.sections[0].blocks] == [
        "The company operates in Europe."
    ]
    assert len(tree.sections[0].subsections) == 1
    assert tree.sections[0].subsections[0].heading == "1.1 Strategy"
    assert [block.block.text for block in tree.sections[0].subsections[0].blocks] == [
        "Growth is driven by subscriptions."
    ]


def test_build_document_tree_creates_implicit_section_for_orphan_subsection() -> None:
    blocks = [
        _labeled_block("1.1 Scope", BlockLabel.SUBSECTION, page=2),
        _labeled_block("This subsection appears first.", BlockLabel.PARAGRAPH, page=2),
    ]

    tree = build_document_tree(blocks)

    assert len(tree.sections) == 1
    assert tree.sections[0].heading == "Untitled Section"
    assert tree.sections[0].page == 2
    assert len(tree.sections[0].subsections) == 1
    assert tree.sections[0].subsections[0].heading == "1.1 Scope"
    assert [block.block.text for block in tree.sections[0].subsections[0].blocks] == [
        "This subsection appears first."
    ]


def test_build_document_tree_keeps_later_title_like_blocks_as_content() -> None:
    blocks = [
        _labeled_block("Annual Report 2025", BlockLabel.TITLE),
        _labeled_block("1 Business Overview", BlockLabel.SECTION),
        _labeled_block("Appendix A", BlockLabel.TITLE),
    ]

    tree = build_document_tree(blocks)

    assert tree.title == "Annual Report 2025"
    assert [block.block.text for block in tree.sections[0].blocks] == ["Appendix A"]

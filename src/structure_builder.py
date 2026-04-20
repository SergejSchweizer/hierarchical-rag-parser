from __future__ import annotations

from models import (
    BlockLabel,
    DocumentTree,
    LabeledBlock,
    SectionNode,
    SubsectionNode,
)


def build_document_tree(blocks: list[LabeledBlock]) -> DocumentTree:
    """Build a simple document tree from an ordered sequence of labeled blocks.

    The builder assumes the blocks are already in reading order. It uses the
    coarse labels to group content into:
    - an optional document title
    - preamble content before the first section
    - sections
    - subsections nested under the current section
    """

    title: str | None = None
    title_page: int | None = None
    preamble: list[LabeledBlock] = []
    sections: list[SectionNode] = []

    current_section: SectionNode | None = None
    current_subsection: SubsectionNode | None = None

    for labeled_block in blocks:
        label = labeled_block.label

        if label == BlockLabel.TITLE and title is None:
            title = labeled_block.block.text
            title_page = labeled_block.block.page
            continue

        if label == BlockLabel.SECTION:
            current_section = SectionNode(
                heading=labeled_block.block.text,
                page=labeled_block.block.page,
                blocks=[],
                subsections=[],
            )
            sections.append(current_section)
            current_subsection = None
            continue

        if label == BlockLabel.SUBSECTION:
            if current_section is None:
                current_section = _create_implicit_section(labeled_block)
                sections.append(current_section)

            current_subsection = SubsectionNode(
                heading=labeled_block.block.text,
                page=labeled_block.block.page,
                blocks=[],
            )
            current_section.subsections.append(current_subsection)
            continue

        if current_subsection is not None:
            current_subsection.blocks.append(labeled_block)
            continue

        if current_section is not None:
            current_section.blocks.append(labeled_block)
            continue

        preamble.append(labeled_block)

    return DocumentTree(
        title=title,
        title_page=title_page,
        preamble=preamble,
        sections=sections,
    )


def _create_implicit_section(subsection_block: LabeledBlock) -> SectionNode:
    """Create a fallback section when a subsection appears before any section."""
    return SectionNode(
        heading="Untitled Section",
        page=subsection_block.block.page,
        blocks=[],
        subsections=[],
    )

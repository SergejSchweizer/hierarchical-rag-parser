from __future__ import annotations

from models import DocumentChunk, DocumentTree, LabeledBlock


def build_chunks(document_tree: DocumentTree) -> list[DocumentChunk]:
    """Build hierarchical chunks from a structured document tree."""
    chunks: list[DocumentChunk] = []

    preamble_chunk = _build_chunk(
        chunk_id="preamble-chunk",
        blocks=document_tree.preamble,
        title=document_tree.title,
        level="preamble",
        parent_section_id=None,
        parent_subsection_id=None,
        section_heading=None,
        subsection_heading=None,
    )
    if preamble_chunk is not None:
        chunks.append(preamble_chunk)

    for section_index, section in enumerate(document_tree.sections, start=1):
        section_id = f"section-{section_index}"
        section_chunk = _build_chunk(
            chunk_id=f"{section_id}-chunk",
            blocks=section.blocks,
            title=document_tree.title,
            level="section",
            parent_section_id=section_id,
            parent_subsection_id=None,
            section_heading=section.heading,
            subsection_heading=None,
        )
        if section_chunk is not None:
            chunks.append(section_chunk)

        for subsection_index, subsection in enumerate(section.subsections, start=1):
            subsection_id = f"{section_id}-subsection-{subsection_index}"
            subsection_chunk = _build_chunk(
                chunk_id=f"{subsection_id}-chunk",
                blocks=subsection.blocks,
                title=document_tree.title,
                level="subsection",
                parent_section_id=section_id,
                parent_subsection_id=subsection_id,
                section_heading=section.heading,
                subsection_heading=subsection.heading,
            )
            if subsection_chunk is not None:
                chunks.append(subsection_chunk)

    return chunks


def _build_chunk(
    chunk_id: str,
    blocks: list[LabeledBlock],
    title: str | None,
    level: str,
    parent_section_id: str | None,
    parent_subsection_id: str | None,
    section_heading: str | None,
    subsection_heading: str | None,
) -> DocumentChunk | None:
    """Build one chunk from a contiguous block group."""
    if not blocks:
        return None

    normalized_texts = [block.block.text.strip() for block in blocks if block.block.text.strip()]
    if not normalized_texts:
        return None

    pages = [block.block.page for block in blocks]
    return DocumentChunk(
        chunk_id=chunk_id,
        text="\n\n".join(normalized_texts),
        title=title,
        level=level,
        parent_section_id=parent_section_id,
        parent_subsection_id=parent_subsection_id,
        section_heading=section_heading,
        subsection_heading=subsection_heading,
        page_start=min(pages),
        page_end=max(pages),
        block_count=len(blocks),
    )

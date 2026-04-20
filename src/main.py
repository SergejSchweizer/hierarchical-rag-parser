from __future__ import annotations

import argparse
import json
from pathlib import Path

from chunker import build_chunks
from layout_classifier import DEFAULT_MODEL_ID, build_classifier, classify_blocks
from structure_builder import build_document_tree
from pdf_parser import extract_text_blocks


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line interface for the PDF parsing demo."""
    parser = argparse.ArgumentParser(
        description="Extract ordered and labeled text blocks from a PDF file."
    )
    parser.add_argument("pdf_path", help="Path to the input PDF file.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the extracted and labeled blocks as JSON.",
    )
    parser.add_argument(
        "--tree-output",
        type=Path,
        help="Optional path to write the reconstructed document tree as JSON.",
    )
    parser.add_argument(
        "--chunks-output",
        type=Path,
        help="Optional path to write hierarchical document chunks as JSON.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of labeled blocks to print in the console preview.",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model id for the zero-shot classifier.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        help="Optional limit for the number of PDF pages to parse.",
    )
    return parser


def main() -> None:
    """Run the parsing pipeline from the command line."""
    parser = build_argument_parser()
    args = parser.parse_args()

    blocks = extract_text_blocks(args.pdf_path, max_pages=args.max_pages)
    classifier = build_classifier(model_id=args.model_id)
    labeled_blocks = classify_blocks(blocks, classifier=classifier, show_progress=True)
    document_tree = build_document_tree(labeled_blocks)
    chunks = build_chunks(document_tree)

    print(f"Extracted {len(blocks)} blocks from {args.pdf_path}")
    print(f"Using model: {args.model_id}")
    if args.max_pages is not None:
        print(f"Parsed page limit: {args.max_pages}")
    print(
        "Document tree summary: "
        f"title={document_tree.title!r}, "
        f"sections={len(document_tree.sections)}"
    )
    print(f"Chunk summary: chunks={len(chunks)}")
    for labeled_block in labeled_blocks[: args.limit]:
        print(json.dumps(labeled_block.to_dict(), ensure_ascii=True))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        serialized_blocks = [block.to_dict() for block in labeled_blocks]
        args.output.write_text(
            json.dumps(serialized_blocks, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        print(f"Wrote JSON output to {args.output}")

    if args.tree_output is not None:
        args.tree_output.parent.mkdir(parents=True, exist_ok=True)
        args.tree_output.write_text(
            json.dumps(document_tree.to_dict(), indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        print(f"Wrote document tree JSON to {args.tree_output}")

    if args.chunks_output is not None:
        args.chunks_output.parent.mkdir(parents=True, exist_ok=True)
        args.chunks_output.write_text(
            json.dumps([chunk.to_dict() for chunk in chunks], indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        print(f"Wrote chunk JSON to {args.chunks_output}")


if __name__ == "__main__":
    main()

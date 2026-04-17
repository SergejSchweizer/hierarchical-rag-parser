from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from models import TextBlock
from pdf_parser import PyMuPDFParser, _normalize_text, _sort_blocks


def test_normalize_text_removes_empty_lines() -> None:
    text = " Heading \n\n first line \n second line \n"
    assert _normalize_text(text) == "Heading\nfirst line\nsecond line"


def test_sort_blocks_orders_by_page_then_position() -> None:
    blocks = [
        TextBlock(text="b", page=2, bbox=(10.0, 50.0, 20.0, 70.0)),
        TextBlock(text="c", page=1, bbox=(30.0, 80.0, 50.0, 100.0)),
        TextBlock(text="a", page=1, bbox=(10.0, 20.0, 20.0, 40.0)),
    ]

    sorted_blocks = _sort_blocks(blocks)

    assert [block.text for block in sorted_blocks] == ["a", "c", "b"]


def test_pymupdf_parser_raises_for_missing_file() -> None:
    parser = PyMuPDFParser()

    try:
        parser.extract_blocks("missing.pdf")
    except FileNotFoundError:
        assert True
    else:
        assert False

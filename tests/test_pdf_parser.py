from pathlib import Path
import sys
from unittest.mock import patch

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from models import TextBlock
from pdf_parser import (
    PyMuPDFParser,
    _compute_uppercase_ratio,
    _extract_page_blocks,
    _normalize_text,
    _sort_blocks,
    extract_text_blocks,
)


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


def test_extract_text_blocks_passes_max_pages_to_parser() -> None:
    with patch.object(PyMuPDFParser, "extract_blocks", return_value=[]) as extract_blocks_mock:
        extract_text_blocks("sample.pdf", max_pages=2)

    extract_blocks_mock.assert_called_once_with("sample.pdf", max_pages=2)


def test_extract_page_blocks_includes_font_size() -> None:
    page = type(
        "FakePage",
        (),
        {
            "rect": type("Rect", (), {"width": 100.0, "height": 200.0})(),
            "get_text": lambda self, mode: {
                "blocks": [
                    {
                        "type": 0,
                        "bbox": (30.0, 20.0, 70.0, 40.0),
                        "lines": [
                            {
                                "spans": [
                                    {"text": "1", "size": 18.0},
                                    {"text": "SECTION", "size": 18.0},
                                ]
                            }
                        ],
                    }
                ]
            }
        },
    )()

    blocks = _extract_page_blocks(page, page_index=1)

    assert len(blocks) == 1
    assert blocks[0].text == "1 SECTION"
    assert blocks[0].font_size == 18.0
    assert blocks[0].line_count == 1
    assert blocks[0].y_position_ratio == 0.1
    assert blocks[0].width_ratio == 0.4
    assert blocks[0].is_centered is True
    assert blocks[0].is_numbered is True
    assert blocks[0].ends_with_period is False
    assert blocks[0].uppercase_ratio == 1.0


def test_compute_uppercase_ratio_handles_missing_letters() -> None:
    assert _compute_uppercase_ratio("1234") is None
    assert _compute_uppercase_ratio("AbC") == 2 / 3

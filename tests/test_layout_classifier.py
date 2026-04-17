from pathlib import Path
import sys
from dataclasses import dataclass
from unittest.mock import patch

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from layout_classifier import (
    TITLE_MAX_CHARACTERS,
    TITLE_MAX_WORDS,
    _apply_label_guards,
    _build_classifier_input,
    _format_font_size,
    _format_ratio,
    _format_text_length,
    _build_progress_status,
    _is_too_long_for_title,
    _map_candidate_label,
    build_classifier,
    classify_blocks,
)
from models import BlockLabel, LabeledBlock, TextBlock


def test_build_classifier_returns_huggingface_strategy() -> None:
    with patch(
        "layout_classifier.HuggingFaceLayoutClassifier._build_pipeline",
        return_value=object(),
    ):
        classifier = build_classifier()

        assert classifier.__class__.__name__ == "HuggingFaceLayoutClassifier"


def test_map_candidate_label_maps_huggingface_output() -> None:
    assert _map_candidate_label("section heading") == BlockLabel.SECTION


def test_build_progress_status_includes_page_label_and_preview() -> None:
    block = TextBlock(
        text="This is a fairly long heading that should be shortened for display.",
        page=3,
        bbox=(0.0, 0.0, 1.0, 1.0),
    )
    labeled_block = LabeledBlock(block=block, label=BlockLabel.SECTION)

    status = _build_progress_status(labeled_block)

    assert "page=3" in status
    assert "label=section" in status
    assert "This is a fairly long heading" in status


@dataclass
class FakeClassifier:
    label: BlockLabel = BlockLabel.PARAGRAPH

    def classify_block(
        self,
        block: TextBlock,
        previous_block: TextBlock | None = None,
        next_block: TextBlock | None = None,
    ) -> LabeledBlock:
        return LabeledBlock(block=block, label=self.label)


def test_classify_blocks_uses_passed_classifier_without_progress() -> None:
    blocks = [
        TextBlock(text="First block", page=1, bbox=(0.0, 0.0, 1.0, 1.0)),
        TextBlock(text="Second block", page=1, bbox=(0.0, 1.0, 1.0, 2.0)),
    ]

    labeled_blocks = classify_blocks(
        blocks,
        classifier=FakeClassifier(label=BlockLabel.SUBSECTION),
        show_progress=False,
    )

    assert [block.label for block in labeled_blocks] == [
        BlockLabel.SUBSECTION,
        BlockLabel.SUBSECTION,
    ]


def test_build_classifier_input_includes_neighboring_blocks_and_page() -> None:
    previous_block = TextBlock(
        text="1 Scope",
        page=2,
        bbox=(0.0, 0.0, 1.0, 1.0),
        font_size=16.0,
        line_count=1,
        y_position_ratio=0.1,
        width_ratio=0.5,
        is_centered=False,
        is_numbered=True,
        ends_with_period=False,
        uppercase_ratio=0.2,
    )
    block = TextBlock(
        text="Internal Controls",
        page=2,
        bbox=(0.0, 1.0, 1.0, 2.0),
        font_size=14.0,
        line_count=1,
        y_position_ratio=0.2,
        width_ratio=0.4,
        is_centered=True,
        is_numbered=False,
        ends_with_period=False,
        uppercase_ratio=0.12,
    )
    next_block = TextBlock(
        text="The company maintains a documented control framework.",
        page=2,
        bbox=(0.0, 2.0, 1.0, 3.0),
        font_size=11.0,
        line_count=2,
        y_position_ratio=0.3,
        width_ratio=0.8,
        is_centered=False,
        is_numbered=False,
        ends_with_period=True,
        uppercase_ratio=0.02,
    )

    classifier_input = _build_classifier_input(
        block=block,
        previous_block=previous_block,
        next_block=next_block,
    )

    assert "Page number: 2" in classifier_input
    assert "Previous block font size: 16.00" in classifier_input
    assert "Previous block text length: 7 chars, 2 words" in classifier_input
    assert "Previous block line count: 1" in classifier_input
    assert "Previous block y position ratio: 0.100" in classifier_input
    assert "Previous block width ratio: 0.500" in classifier_input
    assert "Previous block centered: False" in classifier_input
    assert "Previous block numbered: True" in classifier_input
    assert "Previous block ends with period: False" in classifier_input
    assert "Previous block uppercase ratio: 0.200" in classifier_input
    assert "Previous block: 1 Scope" in classifier_input
    assert "Current block font size: 14.00" in classifier_input
    assert "Current block text length: 17 chars, 2 words" in classifier_input
    assert "Current block line count: 1" in classifier_input
    assert "Current block y position ratio: 0.200" in classifier_input
    assert "Current block width ratio: 0.400" in classifier_input
    assert "Current block centered: True" in classifier_input
    assert "Current block numbered: False" in classifier_input
    assert "Current block ends with period: False" in classifier_input
    assert "Current block uppercase ratio: 0.120" in classifier_input
    assert "Current block: Internal Controls" in classifier_input
    assert "Next block font size: 11.00" in classifier_input
    assert "Next block text length: 52 chars, 7 words" in classifier_input
    assert "Next block line count: 2" in classifier_input
    assert "Next block y position ratio: 0.300" in classifier_input
    assert "Next block width ratio: 0.800" in classifier_input
    assert "Next block centered: False" in classifier_input
    assert "Next block numbered: False" in classifier_input
    assert "Next block ends with period: True" in classifier_input
    assert "Next block uppercase ratio: 0.020" in classifier_input
    assert "Next block: The company maintains a documented control framework." in classifier_input


def test_format_font_size_handles_missing_value() -> None:
    assert _format_font_size(None) == "<unknown>"


def test_format_text_length_reports_characters_and_words() -> None:
    assert _format_text_length("Hello   world") == "11 chars, 2 words"


def test_format_ratio_handles_missing_value() -> None:
    assert _format_ratio(None) == "<unknown>"
    assert _format_ratio(0.125) == "0.125"


def test_apply_label_guards_rejects_implausibly_long_title() -> None:
    block = TextBlock(
        text="This is a very long block of text that clearly looks more like a paragraph than a short title",
        page=1,
        bbox=(0.0, 0.0, 1.0, 1.0),
    )

    label = _apply_label_guards(block, BlockLabel.TITLE)

    assert label == BlockLabel.PARAGRAPH


def test_is_too_long_for_title_uses_configured_thresholds() -> None:
    many_words = " ".join(["word"] * (TITLE_MAX_WORDS + 1))
    many_chars = "x" * (TITLE_MAX_CHARACTERS + 1)

    assert _is_too_long_for_title(many_words) is True
    assert _is_too_long_for_title(many_chars) is True
    assert _is_too_long_for_title("Short Audit Report") is False

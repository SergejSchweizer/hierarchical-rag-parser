from __future__ import annotations

from typing import Protocol

from models import BlockLabel, LabeledBlock, TextBlock
from tqdm import tqdm

DEFAULT_MODEL_ID = "facebook/bart-large-mnli"
TITLE_MAX_WORDS = 14
TITLE_MAX_CHARACTERS = 120
CANDIDATE_LABELS = [
    "document title",
    "section heading",
    "subsection heading",
    "body paragraph",
    "other",
]


class LayoutClassifier(Protocol):
    """Strategy interface for block-level structural classification.

    Design pattern:
    This protocol is the Strategy seam for the layout classification step.
    The rest of the pipeline only depends on the capability "assign a
    structural label to a text block", not on one particular model or rule set.

    Why this helps:
    - The pipeline depends on a stable interface instead of a concrete model.
    - The classifier implementation can still be replaced later without
      changing downstream pipeline code.
    """

    def classify_block(
        self,
        block: TextBlock,
        previous_block: TextBlock | None = None,
        next_block: TextBlock | None = None,
    ) -> LabeledBlock:
        """Classify a single text block into one structural label."""


class HuggingFaceLayoutClassifier:
    """Zero-shot classifier backed by a Hugging Face NLI model.

    Design pattern:
    This is the concrete Strategy implementation for the
    `LayoutClassifier` protocol used by the current pipeline.

    Why `facebook/bart-large-mnli`:
    It is a well-known zero-shot classification model and fits the project
    well because it can map PDF blocks into a small, fixed label set
    without requiring task-specific fine-tuning.
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID) -> None:
        self.model_id = model_id
        self._pipeline = self._build_pipeline()

    def classify_block(
        self,
        block: TextBlock,
        previous_block: TextBlock | None = None,
        next_block: TextBlock | None = None,
    ) -> LabeledBlock:
        """Assign one coarse structural label using zero-shot classification."""
        text = block.text.strip()
        if not text:
            return LabeledBlock(block=block, label=BlockLabel.OTHER)

        classifier_input = _build_classifier_input(
            block=block,
            previous_block=previous_block,
            next_block=next_block,
        )
        result = self._pipeline(classifier_input, CANDIDATE_LABELS, multi_label=False)
        predicted_label = str(result["labels"][0])
        label = _map_candidate_label(predicted_label)
        label = _apply_label_guards(block, label)
        return LabeledBlock(block=block, label=label)

    def _build_pipeline(self):
        """Create the Hugging Face zero-shot pipeline lazily at runtime."""
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required for HuggingFaceLayoutClassifier."
            ) from exc

        return pipeline(
            "zero-shot-classification",
            model=self.model_id,
        )


def classify_blocks(
    blocks: list[TextBlock],
    classifier: LayoutClassifier | None = None,
    show_progress: bool = True,
) -> list[LabeledBlock]:
    """Classify a sequence of blocks using the provided strategy."""
    active_classifier = classifier or HuggingFaceLayoutClassifier()
    labeled_blocks: list[LabeledBlock] = []
    progress_bar = tqdm(
        blocks,
        desc="Classifying blocks",
        unit="block",
        disable=not show_progress,
    )

    for index, block in enumerate(progress_bar):
        previous_block = blocks[index - 1] if index > 0 else None
        next_block = blocks[index + 1] if index + 1 < len(blocks) else None
        labeled_block = active_classifier.classify_block(
            block,
            previous_block=previous_block,
            next_block=next_block,
        )
        labeled_blocks.append(labeled_block)
        progress_bar.set_postfix_str(_build_progress_status(labeled_block), refresh=False)

    return labeled_blocks


def build_classifier(
    model_id: str = DEFAULT_MODEL_ID,
) -> LayoutClassifier:
    """Create a concrete classifier strategy for the pipeline."""
    return HuggingFaceLayoutClassifier(model_id=model_id)


def _map_candidate_label(predicted_label: str) -> BlockLabel:
    """Map the model label vocabulary to the internal block label enum."""
    label_mapping = {
        "document title": BlockLabel.TITLE,
        "section heading": BlockLabel.SECTION,
        "subsection heading": BlockLabel.SUBSECTION,
        "body paragraph": BlockLabel.PARAGRAPH,
        "other": BlockLabel.OTHER,
    }
    return label_mapping.get(predicted_label, BlockLabel.OTHER)


def _apply_label_guards(block: TextBlock, label: BlockLabel) -> BlockLabel:
    """Correct obviously implausible labels with lightweight deterministic rules."""
    if label == BlockLabel.TITLE and _is_too_long_for_title(block.text):
        return BlockLabel.PARAGRAPH

    return label


def _build_progress_status(labeled_block: LabeledBlock) -> str:
    """Build a compact status string for the tqdm progress bar."""
    preview = " ".join(labeled_block.block.text.split())
    if len(preview) > 48:
        preview = f"{preview[:45]}..."

    return (
        f"page={labeled_block.block.page} "
        f"label={labeled_block.label.value} "
        f"text='{preview}'"
    )


def _build_classifier_input(
    block: TextBlock,
    previous_block: TextBlock | None = None,
    next_block: TextBlock | None = None,
) -> str:
    """Build the zero-shot classifier input with lightweight local context."""
    previous_text = _clean_context_text(previous_block.text) if previous_block else "<none>"
    current_text = _clean_context_text(block.text)
    next_text = _clean_context_text(next_block.text) if next_block else "<none>"
    previous_font_size = _format_font_size(previous_block.font_size) if previous_block else "<none>"
    current_font_size = _format_font_size(block.font_size)
    next_font_size = _format_font_size(next_block.font_size) if next_block else "<none>"
    previous_text_length = _format_text_length(previous_block.text) if previous_block else "<none>"
    current_text_length = _format_text_length(block.text)
    next_text_length = _format_text_length(next_block.text) if next_block else "<none>"
    previous_line_count = str(previous_block.line_count) if previous_block else "<none>"
    current_line_count = str(block.line_count)
    next_line_count = str(next_block.line_count) if next_block else "<none>"
    previous_y_position = _format_ratio(previous_block.y_position_ratio) if previous_block else "<none>"
    current_y_position = _format_ratio(block.y_position_ratio)
    next_y_position = _format_ratio(next_block.y_position_ratio) if next_block else "<none>"
    previous_width_ratio = _format_ratio(previous_block.width_ratio) if previous_block else "<none>"
    current_width_ratio = _format_ratio(block.width_ratio)
    next_width_ratio = _format_ratio(next_block.width_ratio) if next_block else "<none>"
    previous_uppercase_ratio = _format_ratio(previous_block.uppercase_ratio) if previous_block else "<none>"
    current_uppercase_ratio = _format_ratio(block.uppercase_ratio)
    next_uppercase_ratio = _format_ratio(next_block.uppercase_ratio) if next_block else "<none>"

    return (
        f"Page number: {block.page}\n"
        f"Previous block font size: {previous_font_size}\n"
        f"Previous block text length: {previous_text_length}\n"
        f"Previous block line count: {previous_line_count}\n"
        f"Previous block y position ratio: {previous_y_position}\n"
        f"Previous block width ratio: {previous_width_ratio}\n"
        f"Previous block centered: {previous_block.is_centered if previous_block else '<none>'}\n"
        f"Previous block numbered: {previous_block.is_numbered if previous_block else '<none>'}\n"
        f"Previous block ends with period: {previous_block.ends_with_period if previous_block else '<none>'}\n"
        f"Previous block uppercase ratio: {previous_uppercase_ratio}\n"
        f"Previous block: {previous_text}\n"
        f"Current block font size: {current_font_size}\n"
        f"Current block text length: {current_text_length}\n"
        f"Current block line count: {current_line_count}\n"
        f"Current block y position ratio: {current_y_position}\n"
        f"Current block width ratio: {current_width_ratio}\n"
        f"Current block centered: {block.is_centered}\n"
        f"Current block numbered: {block.is_numbered}\n"
        f"Current block ends with period: {block.ends_with_period}\n"
        f"Current block uppercase ratio: {current_uppercase_ratio}\n"
        f"Current block: {current_text}\n"
        f"Next block font size: {next_font_size}\n"
        f"Next block text length: {next_text_length}\n"
        f"Next block line count: {next_line_count}\n"
        f"Next block y position ratio: {next_y_position}\n"
        f"Next block width ratio: {next_width_ratio}\n"
        f"Next block centered: {next_block.is_centered if next_block else '<none>'}\n"
        f"Next block numbered: {next_block.is_numbered if next_block else '<none>'}\n"
        f"Next block ends with period: {next_block.ends_with_period if next_block else '<none>'}\n"
        f"Next block uppercase ratio: {next_uppercase_ratio}\n"
        f"Next block: {next_text}"
    )


def _clean_context_text(text: str, max_length: int = 400) -> str:
    """Normalize context text so classifier prompts stay compact."""
    cleaned = " ".join(text.split())
    if len(cleaned) > max_length:
        return f"{cleaned[: max_length - 3]}..."
    return cleaned


def _format_font_size(font_size: float | None) -> str:
    """Format font size consistently for classifier input."""
    if font_size is None:
        return "<unknown>"
    return f"{font_size:.2f}"


def _format_text_length(text: str) -> str:
    """Format text length metadata for classifier input."""
    normalized = " ".join(text.split())
    return f"{len(normalized)} chars, {len(normalized.split())} words"


def _format_ratio(value: float | None) -> str:
    """Format normalized numeric ratios consistently for classifier input."""
    if value is None:
        return "<unknown>"
    return f"{value:.3f}"


def _is_too_long_for_title(text: str) -> bool:
    """Return whether a block is implausibly long to be a document title."""
    normalized = " ".join(text.split())
    word_count = len(normalized.split())
    return word_count > TITLE_MAX_WORDS or len(normalized) > TITLE_MAX_CHARACTERS

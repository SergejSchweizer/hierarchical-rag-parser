# hierarchical-rag-parser

Small Python project for structured PDF parsing and hierarchical chunking.

## Overview

This project extracts text blocks from PDF documents, classifies those blocks
into structural document labels, and prepares the data for later document tree
building and hierarchical chunking.

The current focus is the first half of the pipeline:

1. parse the PDF into ordered text blocks
2. classify each block as `title`, `section`, `subsection`, `paragraph`, or `other`

The goal is to reconstruct document structure from PDFs in a way that is easy
to inspect, easy to test, and easy to extend.

## Current Pipeline

### Step 1: PDF parsing

File: [src/pdf_parser.py](src/pdf_parser.py)

What happens:
- The parser opens a PDF with `PyMuPDF`.
- Each page is read as a set of raw text blocks.
- Empty blocks are removed.
- Block text is normalized by trimming whitespace and removing empty lines.
- Each block is converted into a `TextBlock` domain object.
- Blocks are sorted into a simple reading order using page number, vertical
  position, and horizontal position.

Input:
- a PDF path

Output:
- `list[TextBlock]`

Design:
- `PDFParser` is the strategy interface.
- `PyMuPDFParser` is the concrete parser implementation.
- `extract_text_blocks(...)` is the small pipeline-facing entrypoint used by
  the rest of the application.

### Step 2: Block classification

File: [src/layout_classifier.py](src/layout_classifier.py)

What happens:
- The classifier takes each `TextBlock`.
- It assigns one structural label:
  - `title`
  - `section`
  - `subsection`
  - `paragraph`
  - `other`
- The result is stored as a `LabeledBlock`.

Current implementations:
- `HuggingFaceLayoutClassifier`
  - a model-backed classifier using `facebook/bart-large-mnli`
  - used through Hugging Face's `zero-shot-classification` pipeline

Output:
- `list[LabeledBlock]`

Design:
- `LayoutClassifier` is the strategy interface.
- `HuggingFaceLayoutClassifier` is the concrete classifier implementation used
  by the current pipeline.

### Step 3: CLI orchestration

File: [src/main.py](src/main.py)

What happens:
- The CLI accepts a PDF path.
- It runs the parsing step.
- It builds the Hugging Face classifier.
- It classifies the extracted blocks.
- It prints a preview to the console.
- It can optionally write the labeled output as JSON.

Example flow:

```text
PDF -> parse into TextBlock objects -> classify into LabeledBlock objects -> preview or save JSON
```

### Shared domain model

File: [src/models.py](src/models.py)

What lives here:
- `TextBlock`
  - the core parsed PDF block
- `BlockLabel`
  - the label vocabulary used by the classifier
- `LabeledBlock`
  - a parsed block plus one structural label

Why this file exists:
- It keeps the pipeline data structures explicit and typed.
- It avoids passing around unstructured dictionaries between steps.
- It makes later steps like structure building and hierarchical chunking easier.

## Model Used For Classification

The model-backed classifier uses:

- Model: `facebook/bart-large-mnli`
- Task: zero-shot classification
- Backend: Hugging Face `transformers` pipeline

Why this model:
- It is a standard zero-shot classification model.
- It lets the project classify PDF blocks into a small label set without
  custom training or fine-tuning.
- It keeps the classification step simple and explainable.

### How classification works

For each extracted block, the text is passed to the zero-shot classifier
together with a fixed set of candidate labels:

- `document title`
- `section heading`
- `subsection heading`
- `body paragraph`
- `other`

The model scores these candidates and the best label is mapped to the
internal enum used by the project:

- `document title` -> `title`
- `section heading` -> `section`
- `subsection heading` -> `subsection`
- `body paragraph` -> `paragraph`
- `other` -> `other`

### Model details

Based on the Hugging Face model card and config for `facebook/bart-large-mnli`:

- Model family: BART
- Model architecture: transformer-based encoder-decoder architecture
- Fine-tuning: trained from `bart-large` and fine-tuned on the MNLI dataset
- Size: about `0.4B` parameters
- Architecture type: encoder-decoder sequence classification model
- Hidden size (`d_model`): `1024`
- Encoder layers: `12`
- Decoder layers: `12`
- Attention heads: `16` in encoder and `16` in decoder
- Feed-forward size: `4096`
- Vocabulary size: `50265`
- Maximum positional embeddings: `1024`

Practical interpretation:
- The model can process inputs up to roughly 1024 positions according to the
  config.
- In this project, blocks are short enough that this limit is usually not a
  problem.
- The model is large enough to be useful, but still realistic for a local
  prototype.

## Files And Responsibilities

- [src/models.py](src/models.py): domain objects shared across the pipeline
- [src/pdf_parser.py](src/pdf_parser.py): PDF extraction and ordering
- [src/layout_classifier.py](src/layout_classifier.py): block labeling with
  Hugging Face zero-shot classification
- [src/main.py](src/main.py): CLI entrypoint and pipeline orchestration
- [tests/test_pdf_parser.py](tests/test_pdf_parser.py): parser-focused tests
- [tests/test_layout_classifier.py](tests/test_layout_classifier.py): classifier-focused tests

## Running The Current Pipeline

```powershell
.venv\Scripts\python src/main.py sample_data/example.pdf
```

Write JSON output:

```powershell
.venv\Scripts\python src/main.py sample_data/example.pdf --output outputs/labeled_blocks.json
```

## What Comes Next

The next pipeline stage is structure building:

1. consume `LabeledBlock` objects
2. build a document tree with title, sections, and subsections
3. use that tree as the basis for hierarchical chunking

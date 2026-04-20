# hierarchical-rag-parser

Small Python project for structured PDF parsing and hierarchical chunking.

## Table of Contents

- [Overview](#overview)
- [Current Pipeline](#current-pipeline)
  - [Pipeline Steps](#pipeline-steps)
  - [Step 1: PDF parsing](#step-1-pdf-parsing)
  - [Step 2: Block classification](#step-2-block-classification)
  - [Step 3: Structure building](#step-3-structure-building)
  - [Step 4: Hierarchical chunking](#step-4-hierarchical-chunking)
  - [CLI orchestration](#cli-orchestration)
  - [Shared domain model](#shared-domain-model)
- [Model Used For Classification](#model-used-for-classification)
- [Files And Responsibilities](#files-and-responsibilities)
- [Running The Current Pipeline](#running-the-current-pipeline)
- [What Comes Next](#what-comes-next)

## Overview

This project extracts text blocks from PDF documents, classifies those blocks
into structural document labels, and prepares the data for later document tree
building and hierarchical chunking.

It is intended to showcase one possible approach to structural parsing in
combination with hierarchical chunking for RAG-oriented pipelines.
This approach also enables more precise citations, for example by grounding
answers to exact page numbers and paragraph-level chunks.

The current focus is the first half of the pipeline:

1. parse the PDF into ordered text blocks
2. classify each block as `title`, `section`, `subsection`, `paragraph`, or `other`
3. build a simple document tree from the labeled blocks
4. turn the document tree into hierarchical chunks for downstream RAG use

The goal is to reconstruct document structure from PDFs in a way that is easy
to inspect, easy to test, and easy to extend.

A simplified example of what hierarchical chunking can look like:

```text
Document: "Annual Report 2025"
  Section: "1 Business Overview"
    Subsection: "1.1 Strategy"
      Chunk (level=subsection): "Growth is driven by subscriptions..."
    Chunk (level=section): "The company operates in Europe..."
  Chunk (level=document): "Forward-looking statements..."
```

## Current Pipeline

### Pipeline Steps

#### Step 1: PDF parsing

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

#### Step 2: Block classification

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

#### Step 3: Structure building

File: [src/structure_builder.py](src/structure_builder.py)

What happens:
- The builder consumes ordered `LabeledBlock` objects.
- The first `title` becomes the document title.
- Paragraph-like content before the first section is stored as preamble.
- `section` blocks create top-level sections.
- `subsection` blocks are nested under the current section.
- Later paragraph-like blocks are attached to the active section or subsection.

Output:
- `DocumentTree`

#### Step 4: Hierarchical chunking

File: [src/chunker.py](src/chunker.py)

What happens:
- The chunker consumes `DocumentTree`.
- Preamble content becomes a document-level chunk.
- Section-level content becomes section chunks.
- Subsection-level content becomes subsection chunks.
- Each chunk keeps ancestry metadata such as `chunk_id`, `level`, `parent_section_id`,
  `parent_subsection_id`, title, section, subsection, and page range.

Output:
- `list[DocumentChunk]`

Why hierarchical chunking is useful:
- It keeps local text together with its document context.
- It improves retrieval precision because smaller chunks can match specific questions better.
- It improves answer quality because retrieved chunks can still be linked back to their section or subsection.
- It makes citations and debugging easier because each answer can be grounded in a known part of the document tree.

Current limitations:
- Each content group currently becomes exactly one chunk.
- There is no max-length splitting for long sections.
- There is no overlap between neighboring chunks.
- Chunk text is optimized for readability, not yet separately for embedding input.

Possible next improvements:
- Add max-size chunking such as `max_chars` or token-based splitting.
- Add overlap between chunks so context is preserved across chunk boundaries.
- Prefer splitting at paragraph or sentence boundaries instead of hard cuts.
- Add richer metadata such as `chunk_id`, `char_count`, `source_pages`, or `section_path`.
- Store separate `display_text` and `embedding_text` forms for retrieval experiments.
- Feed heading context directly into the chunk text or embedding representation.

Possible retrieval design: Parent-child retrieval

One practical next step is a parent-child retrieval design:

1. Split the document into smaller child chunks for precise semantic matching.
2. Store parent context for each child chunk, such as the subsection, section, or document title it belongs to.
3. Embed and index the child chunks in a vector store.
4. At query time, embed the user question and retrieve the most similar child chunks.
5. For each retrieved child chunk, load its parent context from the document tree.
6. Send both the matched child chunk and its parent section or subsection context to the LLM.

Why this works well:
- Small child chunks improve retrieval precision.
- Parent context restores the surrounding meaning that may be missing from a small chunk alone.
- The final prompt stays grounded in the real document structure instead of using isolated text snippets.
- This is a simple way to turn structural chunking into true hierarchical retrieval.

### CLI orchestration

File: [src/main.py](src/main.py)

What happens:
- The CLI accepts a PDF path.
- It runs the parsing step.
- It builds the Hugging Face classifier.
- It classifies the extracted blocks.
- It reconstructs a simple document tree.
- It builds hierarchical chunks from that tree.
- It prints a preview to the console.
- It can optionally write the labeled output, document tree, and chunks as JSON.

Example flow:

```text
PDF -> TextBlock objects -> LabeledBlock objects -> document tree -> chunks -> preview or save JSON
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
- `SectionNode`, `SubsectionNode`, `DocumentTree`
  - the structure-building output objects
- `DocumentChunk`
  - chunked text plus structural metadata for retrieval

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
.venv\Scripts\python src/main.py sample_data\10-K.pdf --max-pages 5
```

Write JSON output:

```powershell
.venv\Scripts\python src/main.py sample_data\10-K.pdf --output outputs\labeled_blocks.json
```

Write the reconstructed document tree:

```powershell
.venv\Scripts\python src/main.py sample_data\10-K.pdf --max-pages 5 --tree-output outputs\document_tree.json
```

Write hierarchical chunks:

```powershell
.venv\Scripts\python src/main.py sample_data\10-K.pdf --max-pages 5 --chunks-output outputs\chunks.json
```

## What Comes Next

The next pipeline stage is retrieval and question answering on top of the chunks:

1. embed and index `DocumentChunk`
2. retrieve relevant chunks for a user query
3. generate answers with citations grounded in the chunk metadata

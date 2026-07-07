import re
from dataclasses import dataclass
from typing import List

from pypdf import PdfReader
from langdetect import detect, DetectorFactory, LangDetectException

# langdetect is non-deterministic on short strings unless we seed it.
DetectorFactory.seed = 0


@dataclass
class PageText:
    page_number: int  # 1-indexed, matches what a human would cite
    text: str


@dataclass
class Chunk:
    chunk_id: str
    source_file: str
    page_number: int
    chunk_index: int  # index of this chunk within its page
    text: str


def extract_pdf_pages(path: str) -> List[PageText]:
    """Read a PDF and return cleaned text per page.

    We keep page boundaries intact (rather than concatenating the whole
    document into one blob) because the citation requirement is
    file + page/chunk reference. If we lost page boundaries here we could
    never recover them later.
    """
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        # Collapse hyphenation artifacts and repeated whitespace from PDF
        # text extraction (common with justified-text PDFs).
        cleaned = re.sub(r"-\n(?=[a-z])", "", raw)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if cleaned:
            pages.append(PageText(page_number=i, text=cleaned))
    return pages


def _split_into_sentences(text: str) -> List[str]:
    # Cheap sentence splitter. Good enough for policy/technical prose;
    # avoids pulling in a full NLP dependency (spacy/nltk) for one task.
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_page(
    page: PageText,
    source_file: str,
    target_chars: int = 900,
    overlap_sentences: int = 2,
) -> List[Chunk]:
    """Chunk a single page's text into overlapping, sentence-aligned windows.

    Strategy (explained in full in README.md):
      - Chunk on sentence boundaries, not raw character cutoffs, so we
        never split a fact in the middle of a sentence.
      - Target ~900 characters (~150-200 tokens) per chunk. Small enough
        for precise retrieval, large enough to keep a claim and its
        immediate justification together.
      - Overlap the last 2 sentences of a chunk into the start of the
        next chunk, so a claim that happens to sit on a chunk boundary
        is still fully retrievable from either neighbor.
      - Chunking is done per-page (not across page breaks) so every
        chunk maps cleanly to exactly one page number for citation.
    """
    sentences = _split_into_sentences(page.text)
    if not sentences:
        return []

    chunks: List[Chunk] = []
    current: List[str] = []
    current_len = 0
    chunk_index = 0

    def flush():
        nonlocal current, current_len, chunk_index
        if not current:
            return
        text = " ".join(current)
        chunk_id = f"{source_file}::p{page.page_number}::c{chunk_index}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                source_file=source_file,
                page_number=page.page_number,
                chunk_index=chunk_index,
                text=text,
            )
        )
        chunk_index += 1

    for sentence in sentences:
        current.append(sentence)
        current_len += len(sentence) + 1
        if current_len >= target_chars:
            flush()
            # seed next chunk with overlap from the tail of this one
            current = current[-overlap_sentences:] if len(current) > overlap_sentences else current[:]
            current_len = sum(len(s) + 1 for s in current)

    # flush whatever remains (final partial chunk)
    if current:
        flush()

    return chunks


def chunk_document(path: str, source_file: str) -> List[Chunk]:
    """Extract + chunk an entire PDF, page by page."""
    all_chunks: List[Chunk] = []
    for page in extract_pdf_pages(path):
        all_chunks.extend(chunk_page(page, source_file))
    return all_chunks


def detect_language(text: str) -> str:
    """Best-effort ISO 639-1 language code, defaults to 'en' on failure."""
    try:
        return detect(text)
    except LangDetectException:
        return "en"

"""Student implementation for the mini web-data filtering pipeline."""

from __future__ import annotations

from typing import Sequence, TypedDict

from profanity_check import predict_prob

from .provided import load_manifest, tokenize_words, write_cleaned_jsonl


PROFANITY_THRESHOLD = 0.5
MIN_WORDS = 50
MAX_WORDS = 10_000
MIN_MEAN_WORD_LENGTH = 3
MAX_MEAN_WORD_LENGTH = 10
MAX_ELLIPSIS_LINE_RATIO = 0.30
MIN_ALPHABETIC_WORD_RATIO = 0.80
DEDUP_NGRAM_SIZE = 5
DEDUP_THRESHOLD = 0.8

EMAIL_PLACEHOLDER = "|||EMAIL_ADDRESS|||"
PHONE_PLACEHOLDER = "|||PHONE_NUMBER|||"
IP_PLACEHOLDER = "|||IP_ADDRESS|||"


class Document(TypedDict):
    id: str
    text: str


def extract_text(html_bytes: bytes, *, encoding: str) -> str:
    """Extract and canonicalize visible text from one HTML byte string."""

    # TODO: implement Section 1 of the handout.
    raise NotImplementedError


def mask_pii(text: str) -> str:
    """Replace supported PII with fixed placeholders."""

    # TODO: implement Section 2 of the handout.
    raise NotImplementedError


def is_harmful(text: str) -> bool:
    """Return whether the offensive-language probability reaches 0.5."""

    # TODO: implement Section 3 of the handout.
    raise NotImplementedError


def passes_quality_filter(text: str) -> bool:
    """Return whether the text passes all four quality rules."""

    # TODO: implement Section 4 of the handout using tokenize_words().
    raise NotImplementedError


def normalize_for_dedup(text: str) -> str:
    """Return the canonical form used only for deduplication."""

    # TODO: implement Section 5.1 of the handout.
    raise NotImplementedError


def word_ngrams(text: str, n: int = DEDUP_NGRAM_SIZE) -> set[tuple[str, ...]]:
    """Normalize text and return its set of contiguous word n-grams."""

    # TODO: implement Section 5.2 of the handout.
    raise NotImplementedError


def jaccard_similarity(
    a: set[tuple[str, ...]],
    b: set[tuple[str, ...]],
) -> float:
    """Return exact Jaccard similarity, with defined empty-set behavior."""

    # TODO: implement Section 5.3 of the handout.
    raise NotImplementedError


def deduplicate_documents(
    documents: Sequence[Document],
    n: int = DEDUP_NGRAM_SIZE,
    threshold: float = DEDUP_THRESHOLD,
) -> tuple[list[Document], list[str]]:
    """Return retained documents and removed IDs after greedy dedup."""

    # TODO: implement Section 5.4 of the handout.
    raise NotImplementedError


def run_pipeline(
    input_dir: str,
    output_path: str,
) -> None:
    """Run all stages in the required order and write canonical cleaned JSONL."""

    # TODO: implement Section 6. The provided helpers handle manifest
    # validation and canonical file serialization.
    raise NotImplementedError






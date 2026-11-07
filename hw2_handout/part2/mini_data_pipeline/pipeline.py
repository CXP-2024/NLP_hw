"""Student implementation for the mini web-data filtering pipeline."""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Sequence, TypedDict

from profanity_check import predict_prob
from resiliparse.extract.html2text import extract_plain_text

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

    html_str = html_bytes.decode(encoding)
    plain_text = extract_plain_text(
        html_str,
        preserve_formatting=True,
        main_content=False,
        list_bullets=False,
        alt_texts=False,
        links=False,
        form_fields=False,
        noscript=False,
        comments=False,
        post_meta=False,
        hidden_elements=False,
    )
    text = plain_text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    result = []
    for line in lines:
        line = re.sub(r"[^\S\r\n]+", " ", line)
        line = line.strip()
        if line:
            result.append(line)

    return "\n".join(result)


EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

_PHONE_LAYOUTS = [
    r"[0-9]{3}-[0-9]{3}-[0-9]{4}",            # ddd-ddd-dddd
    r"[0-9]{3} [0-9]{3} [0-9]{4}",            # ddd ddd dddd
    r"\([0-9]{3}\) [0-9]{3}-[0-9]{4}",        # (ddd) ddd-dddd
    r"\+1 [0-9]{3}-[0-9]{3}-[0-9]{4}",        # +1 ddd-ddd-dddd
    r"\+1 \([0-9]{3}\) [0-9]{3}-[0-9]{4}",    # +1 (ddd) ddd-dddd
]
PHONE_RE = re.compile(
    r"(?<![0-9])(?:" + "|".join(_PHONE_LAYOUTS) + r")(?![0-9])"
)

_OCTET = r"(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[0-9])"
IPV4_RE = re.compile(
    r"(?<![0-9])" + r"\.".join([_OCTET] * 4) + r"(?![0-9])"
)

def mask_pii(text: str) -> str:
    """Replace supported PII with fixed placeholders."""

    text = EMAIL_RE.sub(EMAIL_PLACEHOLDER, text)
    text = PHONE_RE.sub(PHONE_PLACEHOLDER, text)
    text = IPV4_RE.sub(IP_PLACEHOLDER, text)
    return text


def is_harmful(text: str) -> bool:
    """Return whether the offensive-language probability reaches 0.5."""

    prob = predict_prob([text])[0]
    return bool(prob >= PROFANITY_THRESHOLD)


def passes_quality_filter(text: str) -> bool:
    """Return whether the text passes all four quality rules."""

    text = unicodedata.normalize("NFC", text)
    tokens = tokenize_words(text)
    lines = [line.strip() for line in text.splitlines()]
    non_empty_lines = [line for line in lines if line]

    # Rule 1: word count. Checked first so the divisions below never see 0.
    if not (MIN_WORDS <= len(tokens) <= MAX_WORDS):
        return False

    # Rule 2: mean word length in Unicode code points.
    mean_length = sum(len(token) for token in tokens) / len(tokens)
    if not (MIN_MEAN_WORD_LENGTH <= mean_length <= MAX_MEAN_WORD_LENGTH):
        return False

    # Rule 3: share of non-empty lines ending with "..." or U+2026.
    ellipsis_lines = sum(
        1 for line in non_empty_lines if line.endswith(("...", "…"))
    )
    if ellipsis_lines / len(non_empty_lines) > MAX_ELLIPSIS_LINE_RATIO:
        return False

    # Rule 4: share of tokens containing at least one alphabetic character.
    alphabetic_tokens = sum(
        1 for token in tokens if any(ch.isalpha() for ch in token)
    )
    if alphabetic_tokens / len(tokens) < MIN_ALPHABETIC_WORD_RATIO:
        return False

    return True


def normalize_for_dedup(text: str) -> str:
    """Return the canonical form used only for deduplication."""

    text = unicodedata.normalize("NFD", text)
    text = text.lower()
    # remove "Mn"
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    # remove "P"
    text = "".join(ch for ch in text if not unicodedata.category(ch).startswith("P"))
    # remove space
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def word_ngrams(text: str, n: int = DEDUP_NGRAM_SIZE) -> set[tuple[str, ...]]:
    """Normalize text and return its set of contiguous word n-grams."""

    if not isinstance(n, int) or n <= 0:
        raise ValueError(f"n must be a positive integer, got {n!r}")

    normalized = normalize_for_dedup(text)
    if not normalized:
        return set()

    words = normalized.split()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)} # pick n sets



def jaccard_similarity(
    a: set[tuple[str, ...]],
    b: set[tuple[str, ...]],
) -> float:
    """Return exact Jaccard similarity, with defined empty-set behavior."""

    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def deduplicate_documents(
    documents: Sequence[Document],
    n: int = DEDUP_NGRAM_SIZE,
    threshold: float = DEDUP_THRESHOLD,
) -> tuple[list[Document], list[str]]:
    """Return retained documents and removed IDs after greedy dedup."""

    if not isinstance(n, int) or n <= 0:
        raise ValueError(f"n must be a positive integer, got {n!r}")
    if(
        not isinstance(threshold, (int, float))
        or not math.isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        raise ValueError(
            f"threshold must be a finite number in [0, 1], got {threshold!r}"
        )

    ordered = sorted(documents, key=lambda doc: doc["id"]) # process in increasing order
    kept: list[Document] = []
    kept_ngram: list[set[tuple[str, ...]]] = []
    removed_ids: list[str] = []

    for doc in ordered:
        ngrams = word_ngrams(doc["text"], n)
        if any(
            jaccard_similarity(ngrams, kept_ngram[i]) >= threshold
            for i in range(len(kept))
        ):
            removed_ids.append(doc["id"])
        else:
            kept.append(doc)
            kept_ngram.append(ngrams)

    return kept, removed_ids


def run_pipeline(
    input_dir: str,
    output_path: str,
) -> None:
    """Run all stages in the required order and write canonical cleaned JSONL."""

    accepted: list[Document] = []
    for entry in load_manifest(input_dir):
        html_bytes = entry.path.read_bytes()
        text = extract_text(html_bytes, encoding = entry.encoding)
        text = mask_pii(text)
        if is_harmful(text):
            continue
        if not passes_quality_filter(text):
            continue
        accepted.append({"id": entry.id, "text": text})

    kept, _removed = deduplicate_documents(accepted)
    write_cleaned_jsonl(output_path, kept)






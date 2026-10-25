"""Mini web-data filtering pipeline for NLP Assignment 2."""

from .pipeline import (
    deduplicate_documents,
    extract_text,
    is_harmful,
    jaccard_similarity,
    mask_pii,
    normalize_for_dedup,
    passes_quality_filter,
    run_pipeline,
    word_ngrams,
)

__all__ = [
    "deduplicate_documents",
    "extract_text",
    "is_harmful",
    "jaccard_similarity",
    "mask_pii",
    "normalize_for_dedup",
    "passes_quality_filter",
    "run_pipeline",
    "word_ngrams",
]


import pytest

from mini_data_pipeline.pipeline import (
    deduplicate_documents,
    jaccard_similarity,
    normalize_for_dedup,
    word_ngrams,
)


def test_normalization_order_and_unicode_rules():
    text = "  Crème-brûlée!\tEMAIL_ADDRESS?\nMIXED—Case  "
    assert normalize_for_dedup(text) == "cremebrulee emailaddress mixedcase"


def test_word_ngram_example_and_validation():
    text = "A language model learns from text"
    assert word_ngrams(text, n=3) == {
        ("a", "language", "model"),
        ("language", "model", "learns"),
        ("model", "learns", "from"),
        ("learns", "from", "text"),
    }
    assert word_ngrams("too short", n=3) == set()
    with pytest.raises(ValueError):
        word_ngrams("text", n=0)


def test_jaccard_and_empty_set_convention():
    a = {("a",), ("b",), ("c",)}
    b = {("b",), ("c",), ("d",)}
    assert jaccard_similarity(a, b) == 0.5
    assert jaccard_similarity(set(), set()) == 1.0
    assert jaccard_similarity(a, set()) == 0.0


def test_threshold_is_inclusive_and_input_is_sorted():
    documents = [
        {"id": "doc_b", "text": "a b c d e"},
        {"id": "doc_a", "text": "a b c d"},
    ]
    kept, duplicates = deduplicate_documents(documents, n=1, threshold=0.8)
    assert [document["id"] for document in kept] == ["doc_a"]
    assert duplicates == ["doc_b"]


def test_removed_ids_are_returned_as_strings():
    documents = [
        {"id": "rep_b", "text": "a b c e"},
        {"id": "target", "text": "a b c"},
        {"id": "rep_a", "text": "a b c d"},
    ]
    kept, duplicates = deduplicate_documents(documents, n=1, threshold=0.7)
    assert [document["id"] for document in kept] == ["rep_a", "rep_b"]
    assert duplicates == ["target"]


def test_greedy_deduplication_does_not_apply_transitive_closure():
    base = [f"base{i:03d}" for i in range(100)]
    a = list(base)
    c = list(base)
    a[20:25] = [f"left{i:03d}" for i in range(5)]
    c[70:75] = [f"right{i:03d}" for i in range(5)]
    documents = [
        {"id": "A", "text": " ".join(a)},
        {"id": "B", "text": " ".join(base)},
        {"id": "C", "text": " ".join(c)},
    ]
    kept, duplicates = deduplicate_documents(documents, n=5, threshold=0.8)
    assert [document["id"] for document in kept] == ["A", "C"]
    assert duplicates == ["B"]






from mini_data_pipeline.pipeline import passes_quality_filter


def words(token, count=50):
    return " ".join([token] * count)


def test_quality_accepts_valid_document_and_inclusive_length_boundaries():
    assert passes_quality_filter(words("good", 50))
    assert passes_quality_filter(words("aaa", 50))
    assert passes_quality_filter(words("abcdefghij", 50))


def test_quality_rejects_word_count_outside_range():
    assert not passes_quality_filter(words("good", 49))


def test_quality_rejects_mean_word_length_outside_range():
    assert not passes_quality_filter(words("aa", 50))
    assert not passes_quality_filter(words("abcdefghijk", 50))


def test_quality_ellipsis_boundary_is_inclusive():
    passing_lines = ["alpha beta gamma delta epsilon..."] * 3 + [
        "alpha beta gamma delta epsilon"
    ] * 7
    failing_lines = ["alpha beta gamma delta epsilon…"] * 4 + [
        "alpha beta gamma delta epsilon"
    ] * 6
    assert passes_quality_filter("\n".join(passing_lines))
    assert not passes_quality_filter("\n".join(failing_lines))


def test_quality_alphabetic_ratio_boundary_is_inclusive():
    passing = ["word"] * 40 + ["1234"] * 10
    failing = ["word"] * 39 + ["1234"] * 11
    assert passes_quality_filter(" ".join(passing))
    assert not passes_quality_filter(" ".join(failing))


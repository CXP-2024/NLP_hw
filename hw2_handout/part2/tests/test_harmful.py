import pytest

from mini_data_pipeline import pipeline


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.49, False),
        (0.50, True),
        (0.95, True),
    ],
)
def test_harmful_threshold_is_inclusive(monkeypatch, score, expected):
    monkeypatch.setattr(pipeline, "predict_prob", lambda _: [score])
    assert pipeline.is_harmful("fixture") is expected


def test_predict_prob_is_called_once_with_exact_batched_text(monkeypatch):
    calls = []

    def fake_predict_prob(texts):
        calls.append(texts)
        return [0.1]

    monkeypatch.setattr(pipeline, "predict_prob", fake_predict_prob)
    assert pipeline.is_harmful("already masked |||EMAIL_ADDRESS|||") is False
    assert calls == [["already masked |||EMAIL_ADDRESS|||"]]


def test_real_classifier_distinguishes_clean_and_offensive_examples():
    assert not pipeline.is_harmful("A concise guide to planting tomatoes in spring.")
    assert pipeline.is_harmful("Go to hell, you worthless scum.")

import hashlib
import json

from mini_data_pipeline import pipeline


def _write_page(root, document_id, text):
    payload = f"<html><body><article><p>{text}</p></article></body></html>".encode()
    path = root / "pages" / f"{document_id}.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "id": document_id,
        "path": f"pages/{document_id}.html",
        "encoding": "utf-8",
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def test_small_end_to_end_pipeline_and_stage_order(tmp_path, monkeypatch):
    input_dir = tmp_path / "data"
    base = " ".join(f"alpha{i:03d}" for i in range(70))
    distinct = " ".join(f"bravo{i:03d}" for i in range(70))
    rows = [
        _write_page(input_dir, "doc_003", distinct + " owner@example.com"),
        _write_page(input_dir, "doc_001", base),
        _write_page(input_dir, "doc_004", "short BLOCKME owner@example.com"),
        _write_page(input_dir, "doc_002", base),
        _write_page(input_dir, "doc_005", "tiny page"),
    ]
    with (input_dir / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in reversed(rows):
            handle.write(json.dumps(row) + "\n")

    classifier_inputs = []

    def fake_predict_prob(texts):
        text = texts[0]
        classifier_inputs.append(text)
        return [0.95 if "BLOCKME" in text else 0.0]

    monkeypatch.setattr(pipeline, "predict_prob", fake_predict_prob)
    output_path = tmp_path / "output" / "cleaned.jsonl"
    pipeline.run_pipeline(str(input_dir), str(output_path))

    output_rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert output_rows == [
        {"id": "doc_001", "text": base},
        {"id": "doc_003", "text": distinct + " |||EMAIL_ADDRESS|||"},
    ]
    assert len(classifier_inputs) == 5
    assert all("owner@example.com" not in text for text in classifier_inputs)
    assert any("BLOCKME |||EMAIL_ADDRESS|||" in text for text in classifier_inputs)




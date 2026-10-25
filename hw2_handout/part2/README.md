# Mini Web-Data Filtering Pipeline

Use Python 3.11. From this directory, install the dependencies and run the public tests:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

Implement the functions marked `TODO` in `mini_data_pipeline/pipeline.py`. You may add imports and private helpers to that file, but do not change its supplied public signatures or constants, and do not modify the provided helper, data, or public tests.

The corpus is already bundled; the assignment never downloads data or calls a
network service. To verify its manifest and checksums, run:

```bash
python verify_data.py
```

Run the complete pipeline with:

```bash
python -m mini_data_pipeline \
  --input-dir data \
  --output-path output/cleaned.jsonl
```

The corpus contains exactly 100 real HTML response bodies from one
fixed `CC-MAIN-2026-39` WARC file. The files
`web_001.html`--`web_100.html` are byte-for-byte copies of the archived response
bodies.


"""Provided deterministic I/O and tokenization helpers.

Students should not modify this file.
"""

from __future__ import annotations

import codecs
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


_ID_RE = re.compile(r"[A-Za-z0-9_.-]+\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_MANIFEST_KEYS = {"id", "path", "encoding", "sha256"}


@dataclass(frozen=True)
class ManifestEntry:
    """One validated input-document entry."""

    id: str
    path: Path
    encoding: str
    sha256: str


def tokenize_words(text: str) -> list[str]:
    """Return maximal non-whitespace spans using Python's Unicode rules."""

    return text.split()


def _safe_child(root: Path, relative_path: str) -> Path:
    candidate = Path(relative_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"manifest path must be relative and stay in input_dir: {relative_path!r}")
    resolved = (root / candidate).resolve()
    if os.path.commonpath([str(root.resolve()), str(resolved)]) != str(root.resolve()):
        raise ValueError(f"manifest path escapes input_dir: {relative_path!r}")
    return resolved


def load_manifest(input_dir: str | Path) -> list[ManifestEntry]:
    """Load, validate, checksum, and sort ``input_dir/manifest.jsonl``."""

    root = Path(input_dir)
    manifest_path = root / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")

    entries: list[ManifestEntry] = []
    seen_ids: set[str] = set()
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on manifest line {line_number}") from exc
            if not isinstance(row, dict) or set(row) != _MANIFEST_KEYS:
                raise ValueError(
                    f"manifest line {line_number} must contain exactly {sorted(_MANIFEST_KEYS)}"
                )
            if not all(isinstance(row[key], str) for key in _MANIFEST_KEYS):
                raise ValueError(f"manifest line {line_number} values must all be strings")

            document_id = row["id"]
            if not _ID_RE.fullmatch(document_id):
                raise ValueError(f"invalid document id: {document_id!r}")
            if document_id in seen_ids:
                raise ValueError(f"duplicate document id: {document_id}")
            seen_ids.add(document_id)

            try:
                codecs.lookup(row["encoding"])
            except LookupError as exc:
                raise ValueError(f"unknown encoding for {document_id}: {row['encoding']}") from exc
            if not _SHA256_RE.fullmatch(row["sha256"]):
                raise ValueError(f"invalid SHA256 for {document_id}")

            document_path = _safe_child(root, row["path"])
            if not document_path.is_file():
                raise FileNotFoundError(f"missing input file for {document_id}: {document_path}")
            digest = hashlib.sha256(document_path.read_bytes()).hexdigest()
            if digest != row["sha256"]:
                raise ValueError(
                    f"checksum mismatch for {document_id}: expected {row['sha256']}, got {digest}"
                )
            entries.append(
                ManifestEntry(
                    id=document_id,
                    path=document_path,
                    encoding=row["encoding"],
                    sha256=row["sha256"],
                )
            )

    return sorted(entries, key=lambda entry: entry.id)


def write_cleaned_jsonl(
    output_path: str | Path,
    documents: Sequence[Mapping[str, str]],
) -> None:
    """Write compact, UTF-8, LF-terminated JSONL in the supplied order."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for document in documents:
            row = {"id": document["id"], "text": document["text"]}
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


"""Verify the bundled manifest and all per-document SHA256 checksums."""

from pathlib import Path

from mini_data_pipeline.provided import load_manifest


def main() -> None:
    data_dir = Path(__file__).resolve().parent / "data"
    entries = load_manifest(data_dir)
    print(f"Verified {len(entries)} documents in {data_dir}")


if __name__ == "__main__":
    main()


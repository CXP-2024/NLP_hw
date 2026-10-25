"""Command-line entry point supplied to students."""

from __future__ import annotations

import argparse

from .pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the mini web-data filtering pipeline")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()
    run_pipeline(args.input_dir, args.output_path)


if __name__ == "__main__":
    main()


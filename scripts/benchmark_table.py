"""CLI: render the aggregate benchmark table from eval/results/ (issue #2).

Usage::

    python scripts/benchmark_table.py [--results-dir eval/results] [--out -]

With ``--out results.md`` writes the Markdown table to a file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from eval.table import build_markdown_table  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render the aggregate benchmark table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--results-dir", default="eval/results", help="directory of report JSONs"
    )
    p.add_argument("--out", default=None, help="write table here ('-' for stdout)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    table = build_markdown_table(Path(args.results_dir))
    if args.out == "-" or not args.out:
        print(table)
    else:
        Path(args.out).write_text(table + "\n", encoding="utf-8")
        print(f"Table written → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

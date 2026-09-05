"""Console entry point: ``python -m nepali_gpt2 {train|generate|data-prep}``.

Thin dispatcher over the per-script CLIs so users can run everything from
the project root without remembering file paths.
"""

from __future__ import annotations

import sys
from typing import List, Optional

from nepali_gpt2 import data_prep, generate, train

COMMANDS = {
    "train": train.main,
    "generate": generate.main,
    "data-prep": data_prep.main,
}


def main(argv: Optional[List[str]] = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if not args or args[0] not in COMMANDS:
        print(__doc__, file=sys.stderr)
        print(
            f"Unknown command: {args[0] if args else '(none)'}\n"
            "Available commands: " + ", ".join(sorted(COMMANDS)),
            file=sys.stderr,
        )
        return 1
    return int(COMMANDS[args[0]](args[1:]) or 0)


if __name__ == "__main__":
    sys.exit(main())
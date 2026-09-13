"""Console entry point: ``python -m nepali_gpt2 {train|generate|data-prep}``.

Thin dispatcher over the per-script CLIs so users can run everything from
the project root without remembering file paths. Unknown commands exit
with code 1 and a usage hint; known commands return the sub-command's
exit status (0 on success).
"""

from __future__ import annotations

import sys
from typing import List, Optional

# Bind the sub-commands directly from their modules. We must NOT go through
# the package namespace (`import nepali_gpt2.generate as generate`): __init__
# re-exports a *function* named `generate`, which would shadow the module.
from nepali_gpt2.data.prep import main as data_prep_main
from nepali_gpt2.generate import main as generate_main
from nepali_gpt2.train import main as train_main

COMMANDS = {
    "train": train_main,
    "generate": generate_main,
    "data-prep": data_prep_main,
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
"""Thin CLI wrapper: ``python scripts/train.py [args]``.

Equivalent to ``python -m nepali_gpt2 train`` but runnable from the repo
root without installing the package, via a small sys.path bootstrap.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nepali_gpt2.train import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())

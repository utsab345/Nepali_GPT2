"""Fine-tune a pretrained NepaliGPT checkpoint on instruction JSONL."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nepali_gpt2.sft import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())

import json

import pytest
from scripts.prepare_instructions import TASKS, prepare


def test_deduplication_and_split(tmp_path):
    source = tmp_path / "source.jsonl"
    rows = [
        dict(
            instruction=f"question-{task}",
            output="उत्तर",
            source="test",
            license="CC0",
            reviewer="test",
            reviewed=True,
            task=task,
        )
        for task in sorted(TASKS)
    ]
    source.write_text("\n".join(json.dumps(row) for row in rows + rows))
    out = tmp_path / "out"
    manifest = prepare([source], out, minimum=6)
    assert manifest["total"] == 6
    train = {
        json.loads(s)["id"] for s in (out / "train.jsonl").read_text().splitlines()
    }
    val = {json.loads(s)["id"] for s in (out / "val.jsonl").read_text().splitlines()}
    assert len(train) + len(val) == 6 and not train & val


def test_unreviewed_data_rejected(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps(
            dict(
                instruction="q",
                output="a",
                source="test",
                license="CC0",
                reviewer="test",
                reviewed=False,
                task="qa",
            )
        )
    )
    with pytest.raises(ValueError, match="not reviewed"):
        prepare([source], tmp_path / "out")

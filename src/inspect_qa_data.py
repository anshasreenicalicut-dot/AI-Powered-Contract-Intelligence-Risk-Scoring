
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

files = [
    BASE_DIR / "data/week2/train.jsonl",
    BASE_DIR / "data/week2/validation.jsonl",
    BASE_DIR / "data/week2/test.jsonl",
]

for path in files:
    print("\n" + "=" * 60)
    print("FILE:", path.name)

    if not path.exists():
        print("File not found")
        continue

    with open(path, "r", encoding="utf-8") as f:
        record = json.loads(next(line for line in f if line.strip()))

    print("Keys:", list(record.keys()))
    print("Contract ID:", record.get("contract_id"))
    print("Text length:", len(record.get("text", "")))
    print("Label/question:", record.get("label"))
    print("Text preview:", record.get("text", "")[:300])

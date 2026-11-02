
import json
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "week2"

for name in ["train", "validation", "test"]:
    file_path = DATA_DIR / f"{name}.jsonl"

    if not file_path.exists():
        print(f"\n{name.upper()}: FILE NOT FOUND - {file_path}")
        continue

    rows = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))

    # Extract the category from the full CUAD question label.
    categories = []
    for row in rows:
        label = row.get("label", "")
        if 'related to "' in label:
            category = label.split('related to "', 1)[1].split('"', 1)[0]
        else:
            category = label
        categories.append(category)

    print(f"\n{name.upper()}: {len(rows)} examples")
    for category, count in Counter(categories).most_common():
        percentage = count / len(rows) * 100 if rows else 0
        print(f"  {category}: {count} ({percentage:.2f}%)")

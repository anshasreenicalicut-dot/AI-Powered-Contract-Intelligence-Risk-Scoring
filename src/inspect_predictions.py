
import json
from pathlib import Path
from collections import Counter, defaultdict

BASE_DIR = Path(__file__).resolve().parent.parent
PREDICTIONS_FILE = BASE_DIR / "outputs" / "evaluation_v2" / "predictions.jsonl"

def category(value):
    if value is None:
        return "UNKNOWN"

    value = str(value)

    if 'related to "' in value:
        return value.split('related to "', 1)[1].split('"', 1)[0]

    return value

if not PREDICTIONS_FILE.exists():
    raise FileNotFoundError(f"File not found: {PREDICTIONS_FILE}")

with open(PREDICTIONS_FILE, "r", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]

if not rows:
    raise ValueError("The predictions file is empty.")

print("Total predictions:", len(rows))
print("Available fields:", list(rows[0].keys()))
print("\nFirst prediction record:")
print(json.dumps(rows[0], indent=2))

predicted_counts = Counter(
    category(row.get("predicted_label")) for row in rows
)

print("\nPREDICTED CATEGORY COUNTS")
for label, count in predicted_counts.most_common():
    print(f"{label}: {count}")

# Look for the actual/gold label field.
possible_fields = [
    "true_label", "actual_label", "gold_label",
    "ground_truth", "label", "actual"
]

true_field = next(
    (field for field in possible_fields if field in rows[0]),
    None
)

if true_field is None:
    print("\nNo actual-label field found in predictions.")
    print("Use the field names above to identify the correct label.")
else:
    actual_counts = Counter(category(row.get(true_field)) for row in rows)
    confusion = defaultdict(Counter)

    for row in rows:
        actual = category(row.get(true_field))
        predicted = category(row.get("predicted_label"))
        confusion[actual][predicted] += 1

    labels = sorted(set(actual_counts) | set(predicted_counts))

    print(f"\nACTUAL CATEGORY COUNTS (field: {true_field})")
    for label, count in actual_counts.most_common():
        print(f"{label}: {count}")

    print("\nCONFUSION MATRIX")
    print("Rows = actual; columns = predicted")
    print("Actual \\ Predicted | " + " | ".join(labels))

    for actual in labels:
        values = [str(confusion[actual][pred]) for pred in labels]
        print(actual + " | " + " | ".join(values))

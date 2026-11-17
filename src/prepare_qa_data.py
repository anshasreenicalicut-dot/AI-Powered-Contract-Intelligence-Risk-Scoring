
import json
import random
from pathlib import Path

# Run from the CUAD project root.
TRAIN_SOURCE = Path("cuad_download/train_separate_questions.json")
TEST_SOURCE = Path("cuad_download/test.json")
OUTPUT_DIR = Path("data/qa")

SEED = 42
VALIDATION_RATIO = 0.10


def load_contracts(path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data["data"]


def convert_contracts(contracts, split_name):
    rows = []
    invalid_offsets = 0
    answerable = 0
    unanswerable = 0

    for contract_index, contract in enumerate(contracts):
        title = contract.get("title", f"{split_name}_{contract_index}")

        for paragraph_index, paragraph in enumerate(
            contract.get("paragraphs", [])
        ):
            context = paragraph.get("context", "")

            for qa_index, qa in enumerate(paragraph.get("qas", [])):
                answers = qa.get("answers") or []
                valid_answers = []

                for answer in answers:
                    answer_text = answer.get("text", "")
                    answer_start = answer.get("answer_start", -1)

                    # Never silently train on incorrect character offsets.
                    if (
                        answer_start >= 0
                        and context[
                            answer_start:answer_start + len(answer_text)
                        ] == answer_text
                    ):
                        valid_answers.append({
                            "text": answer_text,
                            "answer_start": answer_start,
                        })
                    else:
                        invalid_offsets += 1

                # Keep this example answerable only if at least one
                # correctly aligned answer exists.
                answer_texts = [a["text"] for a in valid_answers]
                answer_starts = [a["answer_start"] for a in valid_answers]

                if valid_answers:
                    answerable += 1
                else:
                    unanswerable += 1

                row = {
                    "id": (
                        f"{split_name}_{contract_index}_"
                        f"{paragraph_index}_{qa_index}"
                    ),
                    "title": title,
                    "context": context,
                    "question": qa.get("question", ""),
                    "answers": {
                        "text": answer_texts,
                        "answer_start": answer_starts,
                    },
                    "is_impossible": len(valid_answers) == 0,
                }
                rows.append(row)

    print(f"\n{split_name.upper()}")
    print("Contracts:", len(contracts))
    print("QA examples:", len(rows))
    print("Answerable:", answerable)
    print("Unanswerable:", unanswerable)
    print("Invalid answer offsets:", invalid_offsets)

    return rows


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    random.seed(SEED)

    train_contracts = load_contracts(TRAIN_SOURCE)
    test_contracts = load_contracts(TEST_SOURCE)

    # Split by whole contract, not individual question.
    # This prevents the same contract appearing in train and validation.
    indices = list(range(len(train_contracts)))
    random.shuffle(indices)

    validation_count = max(
        1, round(len(train_contracts) * VALIDATION_RATIO)
    )
    validation_indices = set(indices[:validation_count])

    train_part = [
        contract for i, contract in enumerate(train_contracts)
        if i not in validation_indices
    ]
    validation_part = [
        contract for i, contract in enumerate(train_contracts)
        if i in validation_indices
    ]

    train_rows = convert_contracts(train_part, "train")
    validation_rows = convert_contracts(validation_part, "validation")
    test_rows = convert_contracts(test_contracts, "test")

    write_jsonl(OUTPUT_DIR / "train.jsonl", train_rows)
    write_jsonl(OUTPUT_DIR / "validation.jsonl", validation_rows)
    write_jsonl(OUTPUT_DIR / "test.jsonl", test_rows)

    # Save a brief summary for reporting.
    summary = {
        "source_train_contracts": len(train_contracts),
        "source_test_contracts": len(test_contracts),
        "train_contracts": len(train_part),
        "validation_contracts": len(validation_part),
        "test_contracts": len(test_contracts),
        "train_examples": len(train_rows),
        "validation_examples": len(validation_rows),
        "test_examples": len(test_rows),
        "seed": SEED,
    }

    with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nDataset preparation completed.")
    print("Output folder:", OUTPUT_DIR.resolve())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

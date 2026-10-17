import argparse
import json
import zipfile
from pathlib import Path


def read_json(path):
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def load_json_sources(input_path):
    """Read JSON documents from a ZIP, JSON file, or directory."""

    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(input_path) as archive:
            for name in archive.namelist():
                if not name.lower().endswith(".json"):
                    continue

                try:
                    content = archive.read(name).decode("utf-8-sig")
                    yield name, json.loads(content)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    print(f"[WARNING] Skipping {name}: {error}")

    elif input_path.is_file() and input_path.suffix.lower() == ".json":
        yield str(input_path), read_json(input_path)

    elif input_path.is_dir():
        for path in input_path.rglob("*.json"):
            try:
                yield str(path), read_json(path)
            except (OSError, json.JSONDecodeError) as error:
                print(f"[WARNING] Skipping {path}: {error}")

    else:
        raise FileNotFoundError(
            f"Input not found or unsupported: {input_path}"
        )


def extract_records(data, source):
    """Parse the common CUAD/SQuAD-style annotation format."""

    if not isinstance(data, dict):
        return

    for article in data.get("data", []):
        if not isinstance(article, dict):
            continue

        title = article.get("title", "")

        for paragraph in article.get("paragraphs", []):
            if not isinstance(paragraph, dict):
                continue

            context = paragraph.get("context", "")

            if not context.strip():
                continue

            for question in paragraph.get("qas", []):
                answers = []

                for answer in question.get("answers", []):
                    answers.append({
                        "text": answer.get("text", ""),
                        "answer_start": answer.get("answer_start")
                    })

                yield {
                    "source_file": source,
                    "contract_title": title,
                    "question_id": question.get("id", ""),
                    "category": question.get("question", ""),
                    "context": context,
                    "answers": answers,
                    "is_impossible": question.get(
                        "is_impossible", False
                    )
                }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="Path to CUAD ZIP, JSON file, or extracted directory"
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed"
    )

    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    examples_path = output_dir / "cuad_examples.jsonl"
    contracts_path = output_dir / "contracts.jsonl"
    summary_path = output_dir / "cuad_summary.json"

    seen_contracts = set()
    category_counts = {}
    total_examples = 0
    total_contracts = 0
    sources = []

    with (
        examples_path.open("w", encoding="utf-8") as examples_file,
        contracts_path.open("w", encoding="utf-8") as contracts_file
    ):
        for source, data in load_json_sources(input_path):
            sources.append(source)

            for record in extract_records(data, source):
                examples_file.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )
                total_examples += 1

                category = record["category"] or "Uncategorized"
                category_counts[category] = (
                    category_counts.get(category, 0) + 1
                )

                key = (
                    record["contract_title"],
                    record["context"]
                )

                if key not in seen_contracts:
                    seen_contracts.add(key)

                    contract_record = {
                        "contract_id": f"contract_{total_contracts + 1:05d}",
                        "title": record["contract_title"],
                        "text": record["context"],
                        "source_file": source
                    }

                    contracts_file.write(
                        json.dumps(
                            contract_record,
                            ensure_ascii=False
                        ) + "\n"
                    )
                    total_contracts += 1

    summary = {
        "input": str(input_path),
        "json_files_found": len(sources),
        "examples_written": total_examples,
        "unique_contract_texts": total_contracts,
        "category_counts": category_counts,
        "source_files": sources
    }

    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("\nDataset processing completed.")
    print(f"Examples: {total_examples}")
    print(f"Unique contracts: {total_contracts}")
    print(f"Annotations: {examples_path}")
    print(f"Contract texts: {contracts_path}")
    print(f"Summary: {summary_path}")

    if total_examples == 0:
        print(
            "\nWARNING: No CUAD annotations were found. "
            "Inspect data.zip and check its JSON structure."
        )


if __name__ == "__main__":
    main()
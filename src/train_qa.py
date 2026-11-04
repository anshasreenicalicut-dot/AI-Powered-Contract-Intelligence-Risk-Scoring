
import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

SEED = 42


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def make_chunks(context, chunk_chars=900, overlap=300):
    """Create overlapping character chunks with their original offsets."""
    if not context:
        return [(0, "")]

    step = chunk_chars - overlap
    chunks = []
    start = 0

    while start < len(context):
        end = min(start + chunk_chars, len(context))
        chunks.append((start, context[start:end]))

        if end == len(context):
            break

        start += step

    return chunks


def build_features(rows, tokenizer, max_examples=None):
    features = []
    skipped_answers = 0
    random.seed(SEED)

    if max_examples is not None:
        rows = rows[:max_examples]

    for row_index, row in enumerate(rows):
        context = row["context"]
        question = row["question"]
        answers = row.get("answers", {})
        answer_texts = answers.get("text", [])
        answer_starts = answers.get("answer_start", [])

        valid_answers = []
        for text, start in zip(answer_texts, answer_starts):
            if (
                text
                and isinstance(start, int)
                and start >= 0
                and context[start:start + len(text)] == text
            ):
                valid_answers.append((start, start + len(text)))

        chunks = make_chunks(context)
        selected = []

        if valid_answers:
            # Retain chunks that fully contain an annotated answer.
            for chunk_index, (chunk_start, chunk_text) in enumerate(chunks):
                chunk_end = chunk_start + len(chunk_text)

                if any(
                    answer_start >= chunk_start and answer_end <= chunk_end
                    for answer_start, answer_end in valid_answers
                ):
                    selected.append(chunk_index)

            if not selected:
                skipped_answers += 1
                continue

            # Limit unusually large multi-answer examples.
            selected = selected[:4]
        else:
            # For unanswerable questions, sample up to two chunks.
            if len(chunks) <= 2:
                selected = list(range(len(chunks)))
            else:
                selected = sorted(
                    set([
                        random.randrange(len(chunks)),
                        random.randrange(len(chunks)),
                    ])
                )

        for chunk_index in selected:
            chunk_start, chunk_text = chunks[chunk_index]
            chunk_end = chunk_start + len(chunk_text)

            encoded = tokenizer(
                question,
                chunk_text,
                truncation="only_second",
                max_length=384,
                padding="max_length",
                return_attention_mask=True,
            )

            # DistilBERT uses CLS token index 0 for no-answer examples.
            start_position = 0
            end_position = 0

            if valid_answers:
                matching = [
                    (a_start, a_end)
                    for a_start, a_end in valid_answers
                    if a_start >= chunk_start and a_end <= chunk_end
                ]

                if matching:
                    answer_start, answer_end = matching[0]
                    local_start = answer_start - chunk_start
                    local_end = answer_end - chunk_start

                    # Obtain token offsets for the context chunk.
                    offsets = tokenizer(
                        question,
                        chunk_text,
                        truncation="only_second",
                        max_length=384,
                        return_offsets_mapping=True,
                    )["offset_mapping"]

                    sequence_ids = tokenizer(
                        question,
                        chunk_text,
                        truncation="only_second",
                        max_length=384,
                    ).sequence_ids()

                    context_token_indices = [
                        i for i, seq_id in enumerate(sequence_ids)
                        if seq_id == 1
                    ]

                    start_token = None
                    end_token = None

                    for i in context_token_indices:
                        token_start, token_end = offsets[i]

                        if token_start <= local_start < token_end:
                            start_token = i

                        if token_start < local_end <= token_end:
                            end_token = i
                            break

                    # If truncation removed part of the answer, use CLS.
                    if start_token is not None and end_token is not None:
                        start_position = start_token
                        end_position = end_token

            encoded["start_positions"] = start_position
            encoded["end_positions"] = end_position
            features.append(encoded)

        if (row_index + 1) % 500 == 0:
            print(
                f"Processed {row_index + 1}/{len(rows)} "
                f"questions; features={len(features)}"
            )

    print("Questions processed:", len(rows))
    print("Training features created:", len(features))
    print("Answers with no selected chunk:", skipped_answers)

    return features


class QADataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        feature = self.features[index]
        return {
            key: torch.tensor(value, dtype=torch.long)
            for key, value in feature.items()
            if key in (
                "input_ids",
                "attention_mask",
                "token_type_ids",
                "start_positions",
                "end_positions",
            )
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/qa")
    parser.add_argument(
        "--model-name",
        default="distilbert/distilbert-base-uncased",
    )
    parser.add_argument(
        "--output-dir",
        default="models/cuad_qa_v1",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--max-train-examples", type=int, default=3000)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    random.seed(SEED)
    torch.manual_seed(SEED)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    train_rows = read_jsonl(data_dir / "train.jsonl")
    validation_rows = read_jsonl(data_dir / "validation.jsonl")

    print("Loading tokenizer:", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_features = build_features(
        train_rows,
        tokenizer,
        max_examples=args.max_train_examples,
    )
    validation_features = build_features(validation_rows, tokenizer)

    if not train_features or not validation_features:
        raise RuntimeError("No training or validation features were created.")

    train_dataset = QADataset(train_features)
    validation_dataset = QADataset(validation_features)

    model = AutoModelForQuestionAnswering.from_pretrained(args.model_name)

    use_cpu = args.cpu or not torch.cuda.is_available()

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=50,
        report_to="none",
        fp16=False,
        use_cpu=use_cpu,
        dataloader_num_workers=0,
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    print("\nStarting QA training...")
    trainer.train()

    print("\nValidation loss:")
    print(trainer.evaluate())

    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print("\nModel saved to:", output_dir.resolve())


if __name__ == "__main__":
    main()

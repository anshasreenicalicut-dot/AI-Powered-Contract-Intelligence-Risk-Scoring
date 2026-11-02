
import json
import re
import string
import time
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering


# --------------------------------------------------
# PATHS AND SETTINGS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_DIR = BASE_DIR / "models" / "cuad_qa_v1"
TEST_FILE = BASE_DIR / "data" / "qa" / "test.jsonl"
OUTPUT_DIR = BASE_DIR / "outputs" / "qa_evaluation"

PREDICTIONS_FILE = OUTPUT_DIR / "qa_predictions.jsonl"
METRICS_FILE = OUTPUT_DIR / "qa_metrics.json"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 300
MAX_LENGTH = 384
MAX_ANSWER_TOKENS = 30
TOP_K = 20

# Process multiple chunks together.
BATCH_SIZE = 8

NULL_THRESHOLD = 0.0


# --------------------------------------------------
# CPU OPTIMIZATION
# --------------------------------------------------

if not torch.cuda.is_available():
    torch.set_num_threads(min(8, os.cpu_count() or 4))


# --------------------------------------------------
# METRICS
# --------------------------------------------------

def normalize_answer(text):
    text = text.lower()
    text = "".join(
        ch for ch in text if ch not in string.punctuation
    )
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction, references):
    pred = normalize_answer(prediction)

    if not references:
        return int(pred == "")

    return max(
        int(pred == normalize_answer(ref))
        for ref in references
    )


def token_f1(prediction, reference):
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()

    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)

    common = {}

    for token in pred_tokens:
        common[token] = common.get(token, 0) + 1

    overlap = 0

    for token in ref_tokens:
        if common.get(token, 0) > 0:
            overlap += 1
            common[token] -= 1

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)

    return 2 * precision * recall / (precision + recall)


def calculate_best_f1(prediction, references):
    if not references:
        return token_f1(prediction, "")

    return max(
        token_f1(prediction, ref)
        for ref in references
    )


def get_category(question):
    match = re.search(r'"([^"]+)"', question)
    return match.group(1) if match else "Other"


# --------------------------------------------------
# CHUNKING
# --------------------------------------------------

def make_chunks(context):
    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP

    for start in range(0, len(context), step):
        end = min(start + CHUNK_SIZE, len(context))
        chunks.append((start, context[start:end]))

        if end == len(context):
            break

    return chunks


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

def load_model():
    print("Loading tokenizer...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR)
    )

    print("Loading DistilBERT model...", flush=True)

    model = AutoModelForQuestionAnswering.from_pretrained(
        str(MODEL_DIR)
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    print("Model:", MODEL_DIR, flush=True)
    print("Device:", device, flush=True)
    print("Batch size:", BATCH_SIZE, flush=True)

    return tokenizer, model, device


# --------------------------------------------------
# BATCHED PREDICTION
# --------------------------------------------------

def predict(question, context, tokenizer, model, device):

    chunks = make_chunks(context)

    best_span_score = float("-inf")
    best_null_score = float("-inf")
    best_answer = ""

    for batch_start in range(0, len(chunks), BATCH_SIZE):

        batch_chunks = chunks[
            batch_start:batch_start + BATCH_SIZE
        ]

        questions = [question] * len(batch_chunks)
        contexts = [item[1] for item in batch_chunks]

        encoded = tokenizer(
            questions,
            contexts,
            truncation="only_second",
            max_length=MAX_LENGTH,
            padding="max_length",
            return_offsets_mapping=True,
            return_tensors="pt"
        )

        offsets_batch = encoded.pop("offset_mapping")

        model_inputs = {
            key: value.to(device)
            for key, value in encoded.items()
            if key in ("input_ids", "attention_mask")
        }

        with torch.inference_mode():
            output = model(**model_inputs)

        for batch_index, (_, chunk_text) in enumerate(batch_chunks):

            offsets = offsets_batch[batch_index].tolist()
            sequence_ids = encoded.sequence_ids(batch_index)

            start_logits = output.start_logits[batch_index]
            end_logits = output.end_logits[batch_index]

            input_ids = encoded["input_ids"][batch_index].tolist()
            cls_position = input_ids.index(tokenizer.cls_token_id)

            null_score = (
                start_logits[cls_position].item()
                + end_logits[cls_position].item()
            )

            best_null_score = max(
                best_null_score,
                null_score
            )

            valid_positions = [
                i for i, seq_id in enumerate(sequence_ids)
                if seq_id == 1 and offsets[i] != [0, 0]
            ]

            if not valid_positions:
                continue

            valid_tensor = torch.tensor(
                valid_positions,
                device=start_logits.device
            )

            k = min(TOP_K, len(valid_positions))

            start_values = start_logits[valid_tensor]
            end_values = end_logits[valid_tensor]

            top_starts = torch.topk(
                start_values, k
            ).indices.tolist()

            top_ends = torch.topk(
                end_values, k
            ).indices.tolist()

            start_positions = [
                valid_positions[i] for i in top_starts
            ]

            end_positions = [
                valid_positions[i] for i in top_ends
            ]

            for start_index in start_positions:

                for end_index in end_positions:

                    if end_index < start_index:
                        continue

                    if end_index - start_index + 1 > MAX_ANSWER_TOKENS:
                        continue

                    start_offset = offsets[start_index]
                    end_offset = offsets[end_index]

                    if (
                        start_offset == [0, 0]
                        or end_offset == [0, 0]
                    ):
                        continue

                    if end_offset[1] <= start_offset[0]:
                        continue

                    score = (
                        start_logits[start_index].item()
                        + end_logits[end_index].item()
                    )

                    if score > best_span_score:

                        best_span_score = score

                        best_answer = chunk_text[
                            start_offset[0]:end_offset[1]
                        ]

    if best_span_score == float("-inf"):
        return "", float("-inf")

    answer_score = best_span_score - best_null_score

    if answer_score < NULL_THRESHOLD:
        return "", answer_score

    return best_answer, answer_score


# --------------------------------------------------
# RESUME SUPPORT
# --------------------------------------------------

def load_completed_ids():

    completed_ids = set()

    if not PREDICTIONS_FILE.exists():
        return completed_ids

    print("Checking previously saved predictions...", flush=True)

    with open(
        PREDICTIONS_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            try:
                row = json.loads(line)
                completed_ids.add(str(row.get("id")))
            except json.JSONDecodeError:
                continue

    print(
        "Previously completed examples:",
        len(completed_ids),
        flush=True
    )

    return completed_ids


# --------------------------------------------------
# FINAL METRICS FROM SAVED PREDICTIONS
# --------------------------------------------------

def create_metrics(skipped):

    category_scores = {}
    total = 0
    total_em = 0.0
    total_f1 = 0.0

    if not PREDICTIONS_FILE.exists():
        raise RuntimeError("Prediction file not found.")

    with open(
        PREDICTIONS_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            if not line.strip():
                continue

            row = json.loads(line)

            category = row.get("category", "Other")
            em = float(row.get("exact_match", 0))
            f1 = float(row.get("f1", 0))

            total += 1
            total_em += em
            total_f1 += f1

            if category not in category_scores:
                category_scores[category] = {
                    "count": 0,
                    "em": 0.0,
                    "f1": 0.0
                }

            category_scores[category]["count"] += 1
            category_scores[category]["em"] += em
            category_scores[category]["f1"] += f1

    if total == 0:
        raise RuntimeError("No predictions found.")

    overall = {
        "model": "DistilBERT",
        "test_file": str(TEST_FILE),
        "evaluated_examples": total,
        "skipped_examples_this_run": skipped,
        "exact_match": total_em / total,
        "f1": total_f1 / total,
        "null_threshold": NULL_THRESHOLD
    }

    for values in category_scores.values():

        count = values["count"]

        values["exact_match"] = values.pop("em") / count
        values["f1"] /= count

    report = {
        "overall": overall,
        "category_metrics": category_scores
    }

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=4,
            ensure_ascii=False
        )

    print("\n========== QA EVALUATION RESULTS ==========")
    print(json.dumps(report, indent=4))

    print("\nPredictions saved:", PREDICTIONS_FILE)
    print("Metrics saved:", METRICS_FILE)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not TEST_FILE.exists():
        raise FileNotFoundError(TEST_FILE)

    tokenizer, model, device = load_model()

    completed_ids = load_completed_ids()

    print("\nStarting resumable batched evaluation...", flush=True)

    skipped = 0
    evaluated_now = 0
    started = time.time()

    with open(
        TEST_FILE,
        "r",
        encoding="utf-8"
    ) as source, open(
        PREDICTIONS_FILE,
        "a",
        encoding="utf-8"
    ) as output_file:

        for line_number, line in enumerate(source, 1):

            if not line.strip():
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            example_id = str(row.get("id"))

            if example_id in completed_ids:
                continue

            question = row.get("question", "")
            context = row.get("context", "")
            answers = row.get("answers", {})

            references = answers.get("text", [])
            starts = answers.get("answer_start", [])

            references = [
                answer for answer in references
                if isinstance(answer, str)
            ]

            if not question or not context:
                skipped += 1
                continue

            valid_references = []

            for answer, start in zip(references, starts):

                if (
                    isinstance(start, int)
                    and context[start:start + len(answer)] == answer
                ):
                    valid_references.append(answer)

            if references and not valid_references:
                skipped += 1
                continue

            references = valid_references

            print(
                f"\nEvaluating example {evaluated_now + 1} "
                f"(file line {line_number})...",
                flush=True
            )

            prediction, answer_score = predict(
                question,
                context,
                tokenizer,
                model,
                device
            )

            em = exact_match(prediction, references)
            f1 = calculate_best_f1(prediction, references)

            category = get_category(question)

            result = {
                "id": row.get("id"),
                "contract": row.get("title"),
                "category": category,
                "question": question,
                "prediction": prediction,
                "reference_answers": references,
                "answer_score": answer_score,
                "exact_match": em,
                "f1": f1
            }

            output_file.write(
                json.dumps(result, ensure_ascii=False) + "\n"
            )

            output_file.flush()

            completed_ids.add(example_id)
            evaluated_now += 1

            elapsed = (time.time() - started) / 60

            print(
                f"Completed this run: {evaluated_now} | "
                f"EM: {em:.4f} | F1: {f1:.4f} | "
                f"Elapsed: {elapsed:.1f} min",
                flush=True
            )

    print("\nEvaluation processing completed.", flush=True)

    create_metrics(skipped)


if __name__ == "__main__":
    main()

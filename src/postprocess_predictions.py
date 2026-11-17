
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def main():
    parser = argparse.ArgumentParser(
        description="CUAD clause classification with confidence filtering"
    )
    parser.add_argument(
        "--model-dir",
        default="models/cuad_clause_classifier"
    )
    parser.add_argument("--text", default=None)
    parser.add_argument("--threshold", type=float, default=0.60)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    if not 0 <= args.threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()

    mapping_path = model_dir / "label_mapping.json"
    with mapping_path.open(encoding="utf-8") as file:
        mapping = json.load(file)

    id2label = mapping.get("id2label", {})

    text = args.text or input("Enter a legal clause: ").strip()
    if not text:
        raise ValueError("Please enter a non-empty legal clause.")

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    with torch.no_grad():
        probabilities = torch.softmax(
            model(**inputs).logits, dim=-1
        )[0]

    predicted_id = int(torch.argmax(probabilities).item())
    confidence = float(probabilities[predicted_id].item())

    full_label = id2label.get(
        str(predicted_id), f"LABEL_{predicted_id}"
    )

    # Extract the category from the CUAD question.
    category = full_label
    if '" that should be reviewed' in full_label:
        category = full_label.split('"')[1]

    status = (
        "ACCEPTED"
        if confidence >= args.threshold
        else "REVIEW REQUIRED"
    )

    print("\n--- CUAD Clause Prediction ---")
    print(f"Predicted category: {category}")
    print(f"Label ID: {predicted_id}")
    print(f"Model confidence: {confidence:.2%}")
    print(f"Decision: {status}")
    print(f"Confidence threshold: {args.threshold:.0%}")

    if confidence < args.threshold:
        print("Recommendation: Have a reviewer verify this prediction.")

    print("\nNote: This model predicts among its four trained categories.")
    print("A meaningful label does not guarantee a correct prediction.")


if __name__ == "__main__":
    main()

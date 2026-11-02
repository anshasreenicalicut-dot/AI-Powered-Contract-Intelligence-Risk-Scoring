
import json
from pathlib import Path
from transformers import AutoConfig

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models" / "cuad_clause_classifier_v2"

config = AutoConfig.from_pretrained(str(MODEL_DIR))

print("MODEL LABEL MAPPING")
print("Number of labels:", config.num_labels)
print("id2label:", config.id2label)
print("label2id:", config.label2id)

mapping_file = MODEL_DIR / "label_mapping.json"

if mapping_file.exists():
    print("\nSAVED LABEL MAPPING FILE")
    with open(mapping_file, "r", encoding="utf-8") as f:
        print(json.dumps(json.load(f), indent=2))
else:
    print("\nlabel_mapping.json was not found.")

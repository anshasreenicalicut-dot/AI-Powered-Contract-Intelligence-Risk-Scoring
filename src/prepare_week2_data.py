import argparse, json, random, re
from pathlib import Path
from collections import Counter

def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def first(obj, keys):
    for k in keys:
        v = obj.get(k)
        if v not in (None, "", [], {}): return v
    return None

def clean(v):
    if isinstance(v, dict): v = first(v, ["text", "answer", "value"])
    if isinstance(v, list):
        vals = []
        for x in v:
            if isinstance(x, dict): x = first(x, ["text", "answer", "value"])
            if x: vals.append(str(x))
        v = " ".join(vals)
    return re.sub(r"\s+", " ", str(v or "")).strip()

def extract(r):
    label = first(r, ["category", "label", "clause_type", "question", "question_text", "query", "title"])
    text = first(r, ["clause_text", "answer_text", "answer", "text", "snippet", "paragraph_text", "context", "content"])
    if not text and isinstance(r.get("answers"), list): text = r["answers"]
    if not text and isinstance(r.get("answer"), dict): text = r["answer"]
    contract = first(r, ["contract_id", "document_id", "doc_id", "contract", "id", "file_name"])
    label, text = clean(label), clean(text)
    label = re.sub(r"\s+", " ", label).strip().rstrip("?")
    if not label or not text or len(text) < 12 or len(label) > 160: return None
    return {"text": text[:4000], "label": label, "contract_id": str(contract or "unknown_contract")}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input", default="data/processed/cuad_examples.jsonl")
    p.add_argument("--output-dir", default="data/week2")
    p.add_argument("--min-class-count", type=int, default=2)
    p.add_argument("--max-examples", type=int, default=0)
    a=p.parse_args()
    ip=Path(a.input); out=Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    if not ip.exists(): raise SystemExit(f"Input not found: {ip.resolve()}")
    raw=read_jsonl(ip); examples=[]; skipped=0
    for r in raw:
        x=extract(r)
        if x: examples.append(x)
        else: skipped+=1
    counts=Counter(x["label"] for x in examples)
    examples=[x for x in examples if counts[x["label"]] >= a.min_class_count]
    if a.max_examples and len(examples)>a.max_examples:
        random.Random(42).shuffle(examples); examples=examples[:a.max_examples]
    counts=Counter(x["label"] for x in examples)
    if len(examples)<10 or len(counts)<2:
        raise SystemExit(f"Not enough usable examples: raw={len(raw)}, usable={len(examples)}, classes={len(counts)}. First keys={list(raw[0]) if raw else []}. Inspect extract() against your JSONL schema.")
    ids=sorted(set(x["contract_id"] for x in examples)); random.Random(42).shuffle(ids)
    if len(ids)>=3:
        n=len(ids); tr=set(ids[:max(1,int(n*.8))]); va=set(ids[max(1,int(n*.8)):max(2,int(n*.9))]); te=set(ids[max(2,int(n*.9)):])
        splits=[ [x for x in examples if x["contract_id"] in s] for s in (tr,va,te)]
    else:
        random.Random(42).shuffle(examples); n=len(examples); splits=[examples[:int(.8*n)], examples[int(.8*n):int(.9*n)], examples[int(.9*n):]]
    for name, rows in zip(("train","validation","test"),splits):
        if not rows: raise SystemExit(f"{name} split is empty. Need more unique contracts/examples.")
        with open(out/f"{name}.jsonl","w",encoding="utf-8") as f:
            for x in rows: f.write(json.dumps(x,ensure_ascii=False)+"\n")
    summary={"raw_records":len(raw),"usable_examples":len(examples),"filtered_or_skipped":len(raw)-len(examples),
             "num_classes":len(counts),"class_counts":dict(counts.most_common()),
             "train_examples":len(splits[0]),"validation_examples":len(splits[1]),"test_examples":len(splits[2]),
             "split_strategy":"grouped by contract_id where possible",
             "warning":"Manually inspect examples: CUAD is extractive QA and its answers/questions need careful mapping to classification."}
    (out/"week2_data_summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(summary,indent=2,ensure_ascii=False))
if __name__=="__main__": main()

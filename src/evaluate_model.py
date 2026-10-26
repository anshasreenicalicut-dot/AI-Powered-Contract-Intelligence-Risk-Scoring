import argparse,json
from pathlib import Path
from transformers import pipeline
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix

def read(p):
    with open(p,encoding="utf-8") as f: return [json.loads(x) for x in f if x.strip()]

def main():
    p=argparse.ArgumentParser(); p.add_argument("--model-dir",default="models/cuad_clause_classifier")
    p.add_argument("--test-file",default="data/week2/test.jsonl"); p.add_argument("--output-dir",default="outputs/evaluation")
    p.add_argument("--threshold",type=float,default=.5); p.add_argument("--batch-size",type=int,default=4); a=p.parse_args()
    rows=read(a.test_file)
    if not rows: raise SystemExit("Test file is empty.")
    clf=pipeline("text-classification",model=a.model_dir,tokenizer=a.model_dir,device=-1,top_k=None)
    results=[]
    texts=[x["text"] for x in rows]
    for i in range(0,len(texts),a.batch_size): results.extend(clf(texts[i:i+a.batch_size],truncation=True,max_length=256))
    gold=[x["label"] for x in rows]; allrows=[]
    for x, scores in zip(rows,results):
        best=max(scores,key=lambda s:s["score"])
        allrows.append({**x,"true_label":x["label"],"raw_predicted_label":best["label"],
            "confidence":float(best["score"]),"predicted_label":best["label"] if best["score"]>=a.threshold else "LOW_CONFIDENCE",
            "accepted":bool(best["score"]>=a.threshold)})
    raw=[x["raw_predicted_label"] for x in allrows]; labels=sorted(set(gold)|set(raw)); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    summary={"test_examples":len(rows),"accuracy_before_threshold":float(accuracy_score(gold,raw)),"threshold":a.threshold,
      "accepted_predictions":sum(x["accepted"] for x in allrows),"rejected_low_confidence":sum(not x["accepted"] for x in allrows),
      "coverage":sum(x["accepted"] for x in allrows)/len(allrows),
      "classification_report":classification_report(gold,raw,labels=labels,output_dict=True,zero_division=0),
      "note":"Raw classification metrics are reported before threshold rejection. Select threshold on validation data, not test data."}
    (out/"evaluation_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    (out/"confusion_matrix.json").write_text(json.dumps({"labels":labels,"matrix":confusion_matrix(gold,raw,labels=labels).tolist()},indent=2),encoding="utf-8")
    with open(out/"predictions.jsonl","w",encoding="utf-8") as f:
        for x in allrows: f.write(json.dumps(x,ensure_ascii=False)+"\n")
    print(json.dumps({k:summary[k] for k in ("test_examples","accuracy_before_threshold","threshold","accepted_predictions","rejected_low_confidence","coverage")},indent=2))
if __name__=="__main__": main()

import argparse,json
from transformers import pipeline
def main():
    p=argparse.ArgumentParser(); p.add_argument("--model-dir",default="models/cuad_clause_classifier")
    p.add_argument("--text",required=True); p.add_argument("--threshold",type=float,default=.5); a=p.parse_args()
    clf=pipeline("text-classification",model=a.model_dir,tokenizer=a.model_dir,device=-1)
    r=clf(a.text,truncation=True,max_length=256)[0]; ok=r["score"]>=a.threshold
    print(json.dumps({"text":a.text,"predicted_category":r["label"] if ok else "LOW_CONFIDENCE",
        "raw_category":r["label"],"confidence":float(r["score"]),"threshold":a.threshold,"accepted":ok},indent=2,ensure_ascii=False))
if __name__=="__main__": main()

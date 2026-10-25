import argparse,json
from pathlib import Path
import numpy as np
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,precision_recall_fscore_support
from transformers import AutoTokenizer,AutoModelForSequenceClassification,TrainingArguments,Trainer,DataCollatorWithPadding,set_seed

def read(path):
    with open(path,encoding="utf-8") as f: return [json.loads(x) for x in f if x.strip()]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data-dir",default="data/week2"); p.add_argument("--model-name",default="roberta-base")
    p.add_argument("--output-dir",default="models/cuad_clause_classifier"); p.add_argument("--epochs",type=float,default=2)
    p.add_argument("--batch-size",type=int,default=2); p.add_argument("--learning-rate",type=float,default=2e-5)
    p.add_argument("--max-length",type=int,default=256); p.add_argument("--max-train-samples",type=int,default=0)
    p.add_argument("--cpu",action="store_true"); a=p.parse_args(); set_seed(42)
    d=Path(a.data_dir); tr=read(d/"train.jsonl"); va=read(d/"validation.jsonl"); te=read(d/"test.jsonl")
    if a.max_train_samples:
        tr=tr[:a.max_train_samples]; va=va[:max(1,a.max_train_samples//5)]; te=te[:max(1,a.max_train_samples//5)]
    enc=LabelEncoder(); enc.fit(sorted({x["label"] for x in tr+va+te}))
    l2i={str(label):int(i) for i,label in enumerate(enc.classes_)}; i2l={int(i):str(label) for label,i in l2i.items()}
    def ds(rows):
        return Dataset.from_dict({"text":[x["text"] for x in rows],"labels":[l2i[x["label"]] for x in rows]})
    tok=AutoTokenizer.from_pretrained(a.model_name,use_fast=True)
    def tokenize(batch): return tok(batch["text"],truncation=True,max_length=a.max_length)
    train,val,test=[ds(x).map(tokenize,batched=True) for x in (tr,va,te)]
    model=AutoModelForSequenceClassification.from_pretrained(a.model_name,num_labels=len(l2i),label2id=l2i,id2label=i2l)
    def metrics(pred):
        logits,gold=pred; ids=np.argmax(logits,axis=-1)
        pm,rm,fm,_=precision_recall_fscore_support(gold,ids,average="macro",zero_division=0)
        pw,rw,fw,_=precision_recall_fscore_support(gold,ids,average="weighted",zero_division=0)
        return {"accuracy":float(accuracy_score(gold,ids)),"precision_macro":float(pm),"recall_macro":float(rm),"f1_macro":float(fm),
                "precision_weighted":float(pw),"recall_weighted":float(rw),"f1_weighted":float(fw)}
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    args=TrainingArguments(output_dir=str(out/"checkpoints"),num_train_epochs=a.epochs,learning_rate=a.learning_rate,
        per_device_train_batch_size=a.batch_size,per_device_eval_batch_size=a.batch_size,evaluation_strategy="epoch",
        save_strategy="epoch",load_best_model_at_end=True,metric_for_best_model="f1_macro",greater_is_better=True,
        logging_steps=25,save_total_limit=2,report_to="none",fp16=False,use_cpu=a.cpu)
    trainer=Trainer(model=model,args=args,train_dataset=train,eval_dataset=val,tokenizer=tok,
                    data_collator=DataCollatorWithPadding(tok),compute_metrics=metrics)
    trainer.train(); result=trainer.evaluate(test,metric_key_prefix="test")
    trainer.save_model(str(out)); tok.save_pretrained(str(out))
    (out/"label_mapping.json").write_text(json.dumps({"label2id":l2i,"id2label":{str(k):v for k,v in i2l.items()}},indent=2),encoding="utf-8")
    (out/"test_metrics.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2)); print("Saved model:",out.resolve())
if __name__=="__main__": main()

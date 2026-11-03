
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
CUAD_FILE = BASE_DIR / "cuad_download" / "CUADv1.json"

if not CUAD_FILE.exists():
    raise FileNotFoundError(f"Cannot find: {CUAD_FILE}")

with open(CUAD_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

print("Top-level keys:", list(data.keys()))

articles = data.get("data", [])
print("Number of articles:", len(articles))

if articles:
    article = articles[0]
    print("\nFirst article keys:", list(article.keys()))
    print("Article title:", article.get("title"))

    paragraphs = article.get("paragraphs", [])
    print("Number of paragraphs:", len(paragraphs))

    if paragraphs:
        paragraph = paragraphs[0]
        print("\nParagraph keys:", list(paragraph.keys()))
        print("Context preview:", paragraph.get("context", "")[:500])

        qas = paragraph.get("qas", [])
        print("\nNumber of questions:", len(qas))

        if qas:
            qa = qas[0]
            print("\nFirst QA keys:", list(qa.keys()))
            print("Question:", qa.get("question"))
            print("Is impossible:", qa.get("is_impossible"))
            print("Answers:", qa.get("answers", [])[:3])

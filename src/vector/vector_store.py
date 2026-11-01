
import chromadb
from pathlib import Path

from src.vector.embeddings import generate_embedding

BASE_DIR = Path(__file__).resolve().parents[2]
DB_PATH = BASE_DIR / "data" / "vector_db"

client = chromadb.PersistentClient(path=str(DB_PATH))

collection = client.get_or_create_collection(
    name="cuad_contracts",
    metadata={"hnsw:space": "cosine"}
)


def add_document(contract_id, text, chunk_index=0):
    embedding = generate_embedding(text)

    collection.upsert(
        ids=[f"{contract_id}_{chunk_index}"],
        documents=[text],
        embeddings=[embedding],
        metadatas=[{
            "contract_id": str(contract_id),
            "chunk_index": int(chunk_index)
        }]
    )


def search_documents(query, top_k=5):
    if collection.count() == 0:
        return []

    query_embedding = generate_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count())
    )

    output = []

    for i, document in enumerate(results["documents"][0]):
        output.append({
            "document": document,
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i]
        })

    return output


if __name__ == "__main__":
    add_document(
        "sample_contract_001",
        "The agreement can be terminated by either party "
        "with thirty days written notice.",
        0
    )

    results = search_documents(
        "How can a party terminate the agreement?"
    )

    for result in results:
        print(result)


from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)


def generate_embedding(text):
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")

    vector = model.encode(
        text,
        normalize_embeddings=True
    )

    return vector.tolist()


def generate_embeddings(texts):
    if not texts:
        return []

    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    return vectors.tolist()

if __name__ == "__main__":
    sample_text = "This agreement may be terminated by either party."

    vector = generate_embedding(sample_text)

    print("\nEmbedding model working successfully!")
    print("Vector dimensions:", len(vector))
    print("First 10 values:", vector[:10])
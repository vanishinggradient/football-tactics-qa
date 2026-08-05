"""Embed document chunks using sentence-transformers."""

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

MODEL_NAME = "multi-qa-MiniLM-L6-cos-v1"  # 384 dims, trained for QA retrieval


def embed_documents(chunks, model_name=MODEL_NAME, batch_size=64):
    """Add content_vector field to each chunk using sentence-transformers.

    Returns the same list of chunks, each with a new 'content_vector' key.
    """
    model = SentenceTransformer(model_name)

    texts = [chunk["content"] for chunk in chunks]

    print(f"Embedding {len(texts)} chunks with {model_name}...")
    vectors = model.encode(texts, batch_size=batch_size, show_progress_bar=True)

    for chunk, vector in zip(chunks, vectors):
        chunk["content_vector"] = vector.tolist()

    return chunks


if __name__ == "__main__":
    test_chunks = [
        {"content": "Gegenpressing means winning the ball back immediately after losing it."},
        {"content": "The false nine drops deep to create space for runners."},
    ]
    result = embed_documents(test_chunks)
    print(f"Vector dims: {len(result[0]['content_vector'])}")
    print(f"First 5 values: {result[0]['content_vector'][:5]}")

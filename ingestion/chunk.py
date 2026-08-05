"""Split documents into overlapping chunks for retrieval."""

import hashlib


def chunk_documents(documents, chunk_size=500, overlap=50):
    """
    Split each document into overlapping text chunks.

    Each chunk keeps the parent document's metadata and gets a unique chunk_id.
    chunk_size and overlap are in words (not tokens, close enough for retrieval).
    """
    chunks = []

    for doc in documents:
        words = doc["content"].split()

        if len(words) <= chunk_size:
            # small doc, keep as single chunk
            chunk_id = hashlib.md5(
                f"{doc['doc_id']}-0".encode()
            ).hexdigest()[:12]
            chunks.append({
                "chunk_id": chunk_id,
                "doc_id": doc["doc_id"],
                "source": doc["source"],
                "title": doc["title"],
                "content": doc["content"],
                "metadata": doc["metadata"],
            })
            continue

        start = 0
        chunk_idx = 0

        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_text = " ".join(words[start:end])

            chunk_id = hashlib.md5(
                f"{doc['doc_id']}-{chunk_idx}".encode()
            ).hexdigest()[:12]

            chunks.append({
                "chunk_id": chunk_id,
                "doc_id": doc["doc_id"],
                "source": doc["source"],
                "title": doc["title"],
                "content": chunk_text,
                "metadata": doc["metadata"],
            })

            start += chunk_size - overlap
            chunk_idx += 1

    return chunks


if __name__ == "__main__":
    # quick test
    test_docs = [
        {
            "doc_id": "test1",
            "source": "test",
            "title": "Test Doc",
            "content": " ".join(f"word{i}" for i in range(1200)),
            "metadata": {},
        }
    ]
    result = chunk_documents(test_docs, chunk_size=500, overlap=50)
    print(f"Input: 1 doc with 1200 words")
    print(f"Output: {len(result)} chunks")
    for c in result:
        print(f"  {c['chunk_id']}: {len(c['content'].split())} words")

"""Create Elasticsearch index and bulk-insert document chunks."""

import os
from elasticsearch import Elasticsearch
from tqdm import tqdm

INDEX_NAME = "football-tactics"

INDEX_SETTINGS = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
    },
    "mappings": {
        "properties": {
            "chunk_id": {"type": "keyword"},
            "doc_id": {"type": "keyword"},
            "source": {"type": "keyword"},
            "title": {"type": "text"},
            "content": {"type": "text"},
            "content_vector": {
                "type": "dense_vector",
                "dims": 384,
                "index": True,
                "similarity": "cosine",
            },
        }
    },
}


def get_es_client(url=None):
    """Create an Elasticsearch client."""
    url = url or os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
    return Elasticsearch(url)


def create_index(es_client, index_name=INDEX_NAME):
    """Create the ES index, deleting any existing one first."""
    es_client.indices.delete(index=index_name, ignore_unavailable=True)
    es_client.indices.create(index=index_name, body=INDEX_SETTINGS)
    print(f"Created index '{index_name}'")


def bulk_index(es_client, chunks, index_name=INDEX_NAME):
    """Insert all chunks into the ES index."""
    print(f"Indexing {len(chunks)} chunks into '{index_name}'...")

    for chunk in tqdm(chunks):
        doc = {
            "chunk_id": chunk["chunk_id"],
            "doc_id": chunk["doc_id"],
            "source": chunk["source"],
            "title": chunk["title"],
            "content": chunk["content"],
            "content_vector": chunk["content_vector"],
        }
        es_client.index(index=index_name, document=doc)

    # Refresh so docs are immediately searchable
    es_client.indices.refresh(index=index_name)
    count = es_client.count(index=index_name)["count"]
    print(f"Indexed {count} chunks")


if __name__ == "__main__":
    es = get_es_client()
    print(f"ES info: {es.info()['version']['number']}")

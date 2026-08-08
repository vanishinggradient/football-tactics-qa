"""Prefect-orchestrated ingestion pipeline.

Collects data from all sources, chunks, embeds, and indexes into Elasticsearch.
"""

import json
import os

from prefect import flow, task

from ingestion.collect_statsbomb import collect_statsbomb_data
from ingestion.collect_transcripts import collect_transcripts
from ingestion.collect_wiki import collect_wiki
from ingestion.chunk import chunk_documents
from ingestion.embed import embed_documents
from ingestion.index_elasticsearch import get_es_client, create_index, bulk_index


PROCESSED_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "processed", "documents.json"
)


@task(name="collect-statsbomb", log_prints=True)
def task_collect_statsbomb():
    print("Collecting StatsBomb match data...")
    return collect_statsbomb_data()


@task(name="collect-transcripts", log_prints=True)
def task_collect_transcripts():
    print("Collecting YouTube transcripts...")
    return collect_transcripts()


@task(name="collect-wiki", log_prints=True)
def task_collect_wiki():
    print("Collecting Wikipedia articles...")
    return collect_wiki()


@task(name="chunk-documents", log_prints=True)
def task_chunk(documents):
    print(f"Chunking {len(documents)} documents...")
    return chunk_documents(documents, chunk_size=500, overlap=50)


@task(name="embed-chunks", log_prints=True)
def task_embed(chunks):
    print(f"Embedding {len(chunks)} chunks...")
    return embed_documents(chunks)


@task(name="index-elasticsearch", log_prints=True)
def task_index(chunks):
    es = get_es_client()
    create_index(es)
    bulk_index(es, chunks)


@task(name="save-documents", log_prints=True)
def task_save_documents(documents):
    """Save raw documents to JSON for reproducibility."""
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    with open(PROCESSED_PATH, "w") as f:
        json.dump(documents, f, indent=2)
    print(f"Saved {len(documents)} documents to {PROCESSED_PATH}")


@flow(name="football-tactics-ingestion", log_prints=True)
def ingestion_flow():
    """Full ingestion pipeline: collect -> chunk -> embed -> index."""
    # Collect from all sources
    sb_docs = task_collect_statsbomb()
    yt_docs = task_collect_transcripts()
    wiki_docs = task_collect_wiki()

    all_docs = sb_docs + yt_docs + wiki_docs
    print(f"Total documents collected: {len(all_docs)}")

    # Save raw documents
    task_save_documents(all_docs)

    # Chunk
    chunks = task_chunk(all_docs)
    print(f"Total chunks: {len(chunks)}")

    # Embed
    embedded_chunks = task_embed(chunks)

    # Index
    task_index(embedded_chunks)

    print("Ingestion complete.")
    return len(all_docs), len(chunks)


if __name__ == "__main__":
    import sys

    if "--no-prefect" in sys.argv:
        # Run without Prefect server (useful when there are version conflicts)
        print("Running without Prefect orchestration...")
        sb_docs = collect_statsbomb_data()
        yt_docs = collect_transcripts()
        wiki_docs = collect_wiki()

        all_docs = sb_docs + yt_docs + wiki_docs
        print(f"Total documents collected: {len(all_docs)}")

        os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
        with open(PROCESSED_PATH, "w") as f:
            json.dump(all_docs, f, indent=2)
        print(f"Saved to {PROCESSED_PATH}")

        chunks = chunk_documents(all_docs, chunk_size=500, overlap=50)
        print(f"Total chunks: {len(chunks)}")

        embedded = embed_documents(chunks)

        es = get_es_client()
        create_index(es)
        bulk_index(es, embedded)
        print("Ingestion complete.")
    else:
        ingestion_flow()

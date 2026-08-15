"""Tests for document chunking logic."""

import hashlib
from ingestion.chunk import chunk_documents


def _make_doc(doc_id, word_count):
    """Helper: create a document with exactly word_count words."""
    return {
        "doc_id": doc_id,
        "source": "test",
        "title": f"Test Doc {doc_id}",
        "content": " ".join(f"word{i}" for i in range(word_count)),
        "metadata": {"test": True},
    }


def test_small_doc_single_chunk():
    """Documents smaller than chunk_size should produce exactly one chunk."""
    docs = [_make_doc("small", 100)]
    chunks = chunk_documents(docs, chunk_size=500, overlap=50)
    assert len(chunks) == 1
    assert chunks[0]["doc_id"] == "small"
    assert len(chunks[0]["content"].split()) == 100


def test_exact_chunk_size_single_chunk():
    """Document exactly equal to chunk_size should produce one chunk."""
    docs = [_make_doc("exact", 500)]
    chunks = chunk_documents(docs, chunk_size=500, overlap=50)
    assert len(chunks) == 1


def test_large_doc_multiple_chunks():
    """Document larger than chunk_size should be split into multiple chunks."""
    docs = [_make_doc("large", 1200)]
    chunks = chunk_documents(docs, chunk_size=500, overlap=50)
    assert len(chunks) > 1
    # All chunks should belong to the same doc
    assert all(c["doc_id"] == "large" for c in chunks)


def test_chunk_overlap():
    """Consecutive chunks should share overlapping words."""
    docs = [_make_doc("overlap", 1000)]
    chunks = chunk_documents(docs, chunk_size=500, overlap=50)
    assert len(chunks) >= 2

    words_0 = set(chunks[0]["content"].split())
    words_1 = set(chunks[1]["content"].split())
    shared = words_0 & words_1
    # With 50-word overlap, there should be shared words
    assert len(shared) >= 40


def test_chunk_ids_are_unique():
    """Every chunk should have a unique chunk_id."""
    docs = [_make_doc("a", 1200), _make_doc("b", 800)]
    chunks = chunk_documents(docs, chunk_size=500, overlap=50)
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunk_id_is_deterministic():
    """Same input should produce same chunk_id (MD5-based)."""
    docs = [_make_doc("det", 100)]
    chunks_1 = chunk_documents(docs, chunk_size=500, overlap=50)
    chunks_2 = chunk_documents(docs, chunk_size=500, overlap=50)
    assert chunks_1[0]["chunk_id"] == chunks_2[0]["chunk_id"]


def test_chunk_id_format():
    """chunk_id should be a 12-char hex string (MD5 prefix)."""
    docs = [_make_doc("fmt", 100)]
    chunks = chunk_documents(docs, chunk_size=500, overlap=50)
    cid = chunks[0]["chunk_id"]
    assert len(cid) == 12
    assert all(c in "0123456789abcdef" for c in cid)


def test_metadata_preserved():
    """Chunks should inherit the parent document's metadata."""
    docs = [_make_doc("meta", 1200)]
    chunks = chunk_documents(docs, chunk_size=500, overlap=50)
    for chunk in chunks:
        assert chunk["metadata"] == {"test": True}
        assert chunk["source"] == "test"
        assert chunk["title"] == "Test Doc meta"


def test_empty_input():
    """Empty document list should return empty chunk list."""
    assert chunk_documents([]) == []


def test_multiple_docs():
    """Chunking multiple documents should produce chunks from each."""
    docs = [_make_doc("d1", 100), _make_doc("d2", 200), _make_doc("d3", 1200)]
    chunks = chunk_documents(docs, chunk_size=500, overlap=50)
    doc_ids = {c["doc_id"] for c in chunks}
    assert doc_ids == {"d1", "d2", "d3"}


def test_no_overlap_parameter():
    """With overlap=0, chunks should not share content."""
    docs = [_make_doc("nooverlap", 1000)]
    chunks = chunk_documents(docs, chunk_size=500, overlap=0)
    assert len(chunks) == 2
    words_0 = set(chunks[0]["content"].split())
    words_1 = set(chunks[1]["content"].split())
    assert len(words_0 & words_1) == 0


def test_chunk_count_for_known_input():
    """1200 words with chunk_size=500, overlap=50 should produce 3 chunks."""
    # Chunk 0: words 0-499 (500 words)
    # Chunk 1: words 450-949 (500 words)
    # Chunk 2: words 900-1199 (300 words)
    docs = [_make_doc("count", 1200)]
    chunks = chunk_documents(docs, chunk_size=500, overlap=50)
    assert len(chunks) == 3

from uxt_pipeline.transform.chunker import semantic_chunk

def test_chunking_merges_and_splits():
    doc = {
        "doc_id": "x",
        "elements": [
            {"type": "Title", "text":"A", "meta":{}},
            {"type": "NarrativeText", "text":"hello " * 300, "meta":{}},
        ],
    }
    chunks = semantic_chunk(doc, max_chars=200, overlap=50)
    assert len(chunks) > 1
    assert all(c.text for c in chunks)

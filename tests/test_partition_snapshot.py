from uxt_pipeline.ingest.partitioner import _partition_one

def test_partition_basic(tmp_path):
    html = tmp_path / "a.html"
    html.write_text("<html><body><h1>Title</h1><p>Hello world</p></body></html>", encoding="utf-8")
    res = _partition_one(str(html))
    assert "elements" in res and len(res["elements"]) >= 1

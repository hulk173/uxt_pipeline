from uxt_pipeline.core import chunk
from uxt_pipeline.types import Element

def test_title_starts_new_chunk():
    els = [
        Element(type="Title", text="Розділ 1"),
        Element(type="Text", text="abc " * 50),
        Element(type="Title", text="Розділ 2"),
        Element(type="Text", text="xyz " * 50),
    ]
    ch = chunk(els, size=60, overlap=0)
    # має бути щонайменше 2 чанки (по розділах)
    assert len(ch) >= 2
    assert "Розділ 1" in " ".join(ch[0].text.split()[:10])

def test_table_kept_as_block():
    els = [
        Element(type="Table", text="a,b,c\n1,2,3\n4,5,6"),
    ]
    ch = chunk(els, size=100, overlap=0)
    assert len(ch) == 1
    assert "1,2,3" in ch[0].text

from rag_chatbot.chunker import MarkdownChunker

TEXT = (
    "# Rules\n\n"
    "## Shooting phase\n\nSelect a target, then roll to hit with every weapon.\n\n"
    "## Charge phase\n\nDeclare a charge and roll 2D6 for the distance.\n\n"
    "## Empty\n\n"
)


def test_one_chunk_per_section_with_header_path():
    chunks = MarkdownChunker(min_length=10).chunk(TEXT)
    assert [chunk.section for chunk in chunks] == [
        "Rules > Shooting phase",
        "Rules > Charge phase",
    ]
    assert "## Shooting phase" in chunks[0].content


def test_long_sections_are_split_with_overlap():
    chunks = MarkdownChunker(max_length=40, overlap=10, min_length=1).chunk(TEXT)
    assert len(chunks) > 2
    assert all(len(chunk.content) <= 40 for chunk in chunks)
    assert all(chunk.section.startswith("Rules > ") for chunk in chunks)


def test_short_fragments_are_dropped():
    assert MarkdownChunker(min_length=1000).chunk(TEXT) == []

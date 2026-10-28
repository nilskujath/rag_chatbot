import pytest

pytest.importorskip("ragas")

from scripts.generate_testset import sections  # noqa: E402

MARKDOWN = (
    "# Title\n\n## Short\n\ntiny\n\n## Long\n\n"
    + "x" * 40
    + "\n\n## Also long\n\n"
    + "y" * 40
)


def test_sections_keep_long_sections_with_their_title():
    documents = sections(MARKDOWN, min_chars=30)
    assert [d.metadata["section"] for d in documents] == ["Long", "Also long"]
    assert documents[0].page_content.startswith("## Long")

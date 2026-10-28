from scripts._common import read_jsonl, write_jsonl


def test_jsonl_round_trip_and_missing_file(tmp_path):
    path = tmp_path / "nested" / "rows.jsonl"
    assert read_jsonl(path) == []
    write_jsonl(path, [{"a": 1}, {"b": "é"}])
    assert path.read_text() == '{"a": 1}\n{"b": "é"}\n'
    assert read_jsonl(path) == [{"a": 1}, {"b": "é"}]

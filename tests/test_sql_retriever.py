import sqlite3

from rag_chatbot.retriever.sql_retriever import SqlRetriever, _strip_code_fence


def retriever(units_db, sql: str) -> SqlRetriever:
    db_path, dictionary_path = units_db
    return SqlRetriever(lambda prompt, **_: sql, db_path, dictionary_path)


def test_strip_code_fence():
    assert _strip_code_fence("```sql\nSELECT 1\n```") == "SELECT 1"
    assert _strip_code_fence("```\nSELECT 1\n```\n") == "SELECT 1"
    assert _strip_code_fence("SELECT 1") == "SELECT 1"
    assert _strip_code_fence("```sql\nSELECT 1") == "```sql\nSELECT 1"
    assert (
        _strip_code_fence("Here it is:\n```sql\nSELECT 1\n```\nHope it helps")
        == "SELECT 1"
    )


def test_prompt_contains_schema_and_question(units_db):
    db_path, dictionary_path = units_db
    prompts = []
    SqlRetriever(
        lambda prompt, **_: prompts.append(prompt) or "SELECT 1",
        db_path,
        dictionary_path,
    ).generate_sql("toughness?")
    assert "units(name TEXT, toughness INTEGER)" in prompts[0]
    assert "toughness?" in prompts[0]


def test_rows_are_formatted_as_column_value_pairs(units_db):
    sql = "SELECT name, toughness FROM units ORDER BY name"
    assert retriever(units_db, sql).run(sql) == [
        "name: Intercessors, toughness: 4",
        "name: Terminators, toughness: 5",
    ]


def test_retrieve_returns_one_passage_with_query_and_rows(units_db):
    sql = "```sql\nSELECT COUNT(*) AS n FROM units\n```"
    assert retriever(units_db, sql).retrieve("q") == [
        "SQL query: SELECT COUNT(*) AS n FROM units\nn: 2"
    ]


def test_multiline_query_is_put_on_one_line(units_db):
    sql = "```sql\nSELECT COUNT(*) AS n\nFROM units\n```"
    [passage] = retriever(units_db, sql).retrieve("q")
    assert passage.startswith("SQL query: SELECT COUNT(*) AS n FROM units\n")


def test_empty_result_is_an_empty_list(units_db):
    sql = "SELECT * FROM units WHERE toughness = 9"
    assert retriever(units_db, sql).run(sql) == []


def test_empty_result_after_retry_is_reported_in_the_passage(units_db):
    sql = "SELECT * FROM units WHERE toughness = 9"
    [passage] = retriever(units_db, sql).retrieve("q")
    assert passage.endswith("No row matches the query.")


def test_row_and_cell_limits_are_marked(units_db, monkeypatch):
    monkeypatch.setattr("rag_chatbot.retriever.sql_retriever.settings.sql_max_rows", 1)
    monkeypatch.setattr(
        "rag_chatbot.retriever.sql_retriever.settings.sql_max_cell_length", 5
    )
    sql = "SELECT name FROM units ORDER BY name"
    assert retriever(units_db, sql).run(sql) == [
        "name: Inter...",
        "(only the first 1 rows are shown)",
    ]


def test_missing_database_is_reported_not_raised(units_db):
    _, dictionary_path = units_db
    [line] = SqlRetriever(
        lambda p, **_: "SELECT 1", "/nowhere/x.sqlite", dictionary_path
    ).run("SELECT 1")
    assert line.startswith("SQL error: cannot open /nowhere/x.sqlite")


def test_cte_and_other_fence_tags_are_accepted(units_db):
    sql = (
        "```sqlite\nWITH t AS (SELECT name FROM units) SELECT COUNT(*) AS n FROM t\n```"
    )
    [passage] = retriever(units_db, sql).retrieve("q")
    assert passage.endswith("n: 2")


def test_sql_error_is_returned_as_context(units_db):
    [row] = retriever(units_db, "x").run("SELECT nope FROM units")
    assert row.startswith("SQL error:")
    assert "nope" in row


def test_writes_are_refused_and_data_untouched(units_db):
    db_path, _ = units_db
    for sql in ("DELETE FROM units", "SELECT 1; DROP TABLE units", "PRAGMA foo=1"):
        [row] = retriever(units_db, sql).run(sql)
        assert row.startswith("SQL refused")
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM units").fetchone() == (2,)


def test_query_without_rows_is_regenerated_once_with_feedback(units_db):
    db_path, dictionary_path = units_db
    prompts = []
    answers = iter(
        [
            "SELECT name FROM units WHERE name = 'Terminator'",
            "SELECT name FROM units WHERE name LIKE '%Termin%'",
        ]
    )

    def generate(prompt, **_):
        prompts.append(prompt)
        return next(answers)

    passage = SqlRetriever(generate, db_path, dictionary_path).retrieve("terminators?")
    assert len(prompts) == 2
    assert "returned no rows" in prompts[1]
    assert "name = 'Terminator'" in prompts[1]
    assert passage == [
        "SQL query: SELECT name FROM units WHERE name LIKE '%Termin%'\n"
        "name: Terminators"
    ]


def test_query_with_rows_is_not_regenerated(units_db):
    calls = []
    db_path, dictionary_path = units_db
    SqlRetriever(
        lambda p, **_: calls.append(p) or "SELECT name FROM units",
        db_path,
        dictionary_path,
    ).retrieve("q")
    assert len(calls) == 1


def test_failed_query_is_regenerated_with_the_error(units_db):
    db_path, dictionary_path = units_db
    prompts = []
    answers = iter(["SELECT nope FROM units", "SELECT COUNT(*) AS n FROM units"])

    def generate(prompt, **_):
        prompts.append(prompt)
        return next(answers)

    [passage] = SqlRetriever(generate, db_path, dictionary_path).retrieve("how many?")
    assert "no such column: nope" in prompts[1]
    assert passage.endswith("n: 2")


def test_refused_query_is_regenerated(units_db):
    db_path, dictionary_path = units_db
    answers = iter(["DELETE FROM units", "SELECT COUNT(*) AS n FROM units"])
    [passage] = SqlRetriever(
        lambda p, **_: next(answers), db_path, dictionary_path
    ).retrieve("q")
    assert passage.endswith("n: 2")


def test_text_literal_columns_are_refused(units_db):
    sql = (
        "SELECT name, 'Toughness is compared with the attacking Strength to find"
        " the D6 result needed on the Wound roll' AS note FROM units"
    )
    [row] = retriever(units_db, sql).run(sql)
    assert row.startswith("SQL refused (a text literal")
    short = "SELECT name FROM units WHERE name LIKE '%Intercess%'"
    assert len(retriever(units_db, short).run(short)) == 1


def test_sql_is_generated_at_temperature_zero(units_db):
    db_path, dictionary_path = units_db
    calls = []

    def generate(prompt, temperature=None):
        calls.append(temperature)
        return "SELECT 1"

    SqlRetriever(generate, db_path, dictionary_path).generate_sql("q")
    assert calls == [0]

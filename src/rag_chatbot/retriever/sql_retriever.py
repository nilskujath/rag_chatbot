import logging
import re
import sqlite3
from collections.abc import Callable
from pathlib import Path

from rag_chatbot.retriever.base import Retriever
from rag_chatbot.settings import settings

logger = logging.getLogger(__name__)

PROMPT = """You are a SQLite expert. With the schema below, write one SELECT \
query that answers the question. Return only the SQL, no explanation.
Rules of thumb: match names with `LIKE '%word%'` (case-insensitive), never with \
`=`, because names in the database rarely equal the user's wording; when the \
question names a kind of unit, filter on datasheet names rather than on a whole \
faction; return only the columns the question asks about, and never invent a \
column from a string literal (the rows must come from the tables, not from you). \
When a question mixes a data lookup with a rules question, answer only the data \
part: the rules are handled elsewhere.{feedback}

## Schema
{schema}

## Question
{question}

## SQL"""

LITERAL_LIMIT = 60  # longest string literal a legitimate name filter needs

FEEDBACK = """
A previous attempt did not work ({problem}):
{sql}
Write a corrected query (for empty results: wildcards, fewer conditions)."""


class SqlRetriever(Retriever):
    """Text-to-SQL over the unit datasheets database.

    The LLM sees the data dictionary and the question, and returns a SELECT.
    The query runs on a read-only connection and anything that is not a single
    SELECT is refused, so a wrong or hostile generation cannot alter the data.
    """

    def __init__(
        self,
        generate: Callable[..., str],
        db_path: str | None = None,
        dictionary_path: str | None = None,
    ):
        self.generate = generate
        self.db_path = db_path or settings.sql_db_path
        self.schema = Path(dictionary_path or settings.sql_dictionary_path).read_text()

    def retrieve(self, query: str) -> list[str]:
        """One passage: the query that was run, then its rows (or what went wrong).

        A query that is refused, fails or matches nothing is generated once
        more with that outcome in the prompt; a too strict name filter or a
        wrong column name is the usual cause.
        """
        sql = self.generate_sql(query)
        rows = self.run(sql)
        problem = _problem(rows)
        if problem:
            sql = self.generate_sql(query, failed_sql=sql, problem=problem)
            rows = self.run(sql) or ["No row matches the query."]
        return [f"SQL query: {sql}\n" + "\n".join(rows)]

    def generate_sql(
        self, question: str, failed_sql: str | None = None, problem: str = ""
    ) -> str:
        """Ask the LLM for a query, strip its markdown fence and put it on one line."""
        feedback = (
            FEEDBACK.format(sql=failed_sql, problem=problem) if failed_sql else ""
        )
        prompt = PROMPT.format(schema=self.schema, question=question, feedback=feedback)
        sql = " ".join(_strip_code_fence(self.generate(prompt, temperature=0)).split())
        logger.info("SQL: %s", sql)
        return sql

    def run(self, sql: str) -> list[str]:
        """Execute a SELECT: one `column: value, ...` line per row, [] for none.

        Output is capped at `sql_max_rows` rows and `sql_max_cell_length`
        characters per cell; both cuts are marked so that the LLM knows.
        A refused or failed query yields a single line that says why.
        """
        if not _is_single_select(sql):
            return [f"SQL refused (only one SELECT statement is allowed): {sql}"]
        if _has_text_literal(sql):
            return [
                f"SQL refused (a text literal over {LITERAL_LIMIT} characters): {sql}"
            ]
        try:
            conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        except sqlite3.Error as error:
            return [f"SQL error: cannot open {self.db_path} ({error})"]
        try:
            cursor = conn.execute(sql)
            columns = [column[0] for column in cursor.description]
            rows = cursor.fetchmany(settings.sql_max_rows + 1)
        except sqlite3.Error as error:
            return [f"SQL error: {error}"]
        finally:
            conn.close()
        lines = [
            ", ".join(
                f"{column}: {_clip(str(value), settings.sql_max_cell_length)}"
                for column, value in zip(columns, row)
            )
            for row in rows[: settings.sql_max_rows]
        ]
        if len(rows) > settings.sql_max_rows:
            lines.append(f"(only the first {settings.sql_max_rows} rows are shown)")
        return lines


def _problem(rows: list[str]) -> str:
    """Why a result is unusable: empty, refused or failed; "" when it is fine."""
    if not rows:
        return "it returned no rows"
    if rows[0].startswith(("SQL refused", "SQL error")):
        return rows[0]
    return ""


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "..."


def _strip_code_fence(text: str) -> str:
    """Keep only the content of a ```sql ... ``` fence, wherever it sits."""
    match = re.search(r"```\w*\s*(.*?)\s*```", text, flags=re.DOTALL)
    return (match.group(1) if match else text).strip()


def _has_text_literal(sql: str) -> bool:
    """True when a quoted string is long enough to be prose rather than a filter."""
    return any(len(m) > LITERAL_LIMIT for m in re.findall(r"'((?:[^']|'')*)'", sql))


def _is_single_select(sql: str) -> bool:
    """True for exactly one statement that starts with SELECT or WITH."""
    statements = [s for s in sql.split(";") if s.strip()]
    if len(statements) != 1:
        return False
    return statements[0].lstrip().upper().startswith(("SELECT", "WITH"))

import hashlib
import sqlite3

import pytest

from src.database import Database, DatabaseError, validate_sql


def test_dynamic_schema(settings):
    database = Database(settings)
    assert database.list_tables() == ["people", "sales"]
    schema = database.inspect_schema(["sales", "people"])
    assert schema["people"]["columns"][0]["primary_key"]
    assert schema["sales"]["foreign_keys"] == [
        {"column": "people_id", "table": "people", "target": "id"}
    ]
    with sqlite3.connect(settings.database_path) as connection:
        connection.execute("CREATE TABLE shipments(id INTEGER PRIMARY KEY, sales_id INTEGER)")
    assert "shipments" in database.list_tables()
    assert database.inspect_schema(["shipments"])["shipments"]["candidate_relationships"]


def test_select_and_readonly(settings):
    before = hashlib.sha256(settings.database_path.read_bytes()).hexdigest()
    database = Database(settings)
    result = database.execute("SELECT COUNT(DISTINCT people_id) AS customers FROM sales", "q1")
    assert result.rows == [[2]]
    assert result.columns == ["customers"]
    with database.connect() as connection, pytest.raises(sqlite3.OperationalError):
        connection.execute("DELETE FROM people")
    assert hashlib.sha256(settings.database_path.read_bytes()).hexdigest() == before


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM people",
        "DROP TABLE people",
        "UPDATE people SET region='X'",
        "INSERT INTO people VALUES(9,'X')",
        "SELECT 1; SELECT 2",
        "SELECT 1; DROP TABLE people",
        "PRAGMA writable_schema=ON",
        "PRAGMA table_info(people)",
        "ATTACH ':memory:' AS other",
        "VACUUM INTO '/tmp/copy.db'",
        "SELECT load_extension('/tmp/a')",
        "SELECT readfile('/etc/passwd')",
        "SELECT writefile('/tmp/a','x')",
        "SELECT * FROM main.people",
        "SELECT * FROM sqlite_schema",
        "SELECT * FROM pragma_table_info('people')",
        "SELECT randomblob(1000000000)",
        "WITH RECURSIVE t(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM t) SELECT * FROM t",
        "EXPLAIN SELECT * FROM people",
        "BEGIN",
        "SELECT * INTO backup FROM people",
        "WITH x AS (DELETE FROM people RETURNING *) SELECT * FROM x",
        "SELECT 1 UNION SELECT load_extension('evil')",
        "SELECT * FROM people; -- extra\nDELETE FROM people",
    ],
)
def test_adversarial_sql(settings, sql):
    with pytest.raises(DatabaseError):
        Database(settings).execute(sql, "q1")


def test_native_authorizer_independent(settings, monkeypatch):
    monkeypatch.setattr("src.database.validate_sql", lambda sql: sql)
    for sql in [
        "DELETE FROM people",
        "SELECT * FROM sqlite_schema",
        "ATTACH ':memory:' AS x",
        "SELECT randomblob(10)",
        "PRAGMA table_info(people)",
    ]:
        with pytest.raises(DatabaseError):
            Database(settings).execute(sql, "q1")


def test_empty_truncated_and_duplicate_columns(settings):
    database = Database(settings)
    assert database.execute("SELECT * FROM people WHERE id=99", "q1").status == "empty"
    settings.max_rows = 1
    result = database.execute("SELECT * FROM people", "q2")
    assert result.status == "truncated" and len(result.rows) == 1
    with pytest.raises(DatabaseError, match="Colunas repetidas"):
        database.execute("SELECT id, id FROM people", "q3")


@pytest.mark.parametrize(
    "sql",
    [
        "WITH totals AS (SELECT people_id, SUM(amount) n FROM sales GROUP BY people_id) "
        "SELECT * FROM totals",
        "SELECT strftime('%Y-%m', sold_on) AS month, COUNT(*) AS n FROM sales GROUP BY 1",
        "SELECT COUNT(*) * 1.0 / COUNT(DISTINCT people_id) AS average FROM sales",
        "SELECT p.region, COUNT(DISTINCT p.id) n FROM people p "
        "JOIN sales s ON p.id=s.people_id GROUP BY 1",
        "SELECT CAST(amount AS REAL), COALESCE(channel, 'unknown') FROM sales",
        "SELECT ROW_NUMBER() OVER (ORDER BY id) AS rank FROM people",
    ],
)
def test_analytical_sql(settings, sql):
    assert Database(settings).execute(sql, "q1").status == "ok"


def test_timeout(settings):
    settings.sql_timeout_seconds = 0.00001
    with pytest.raises(DatabaseError) as exc:
        Database(settings).execute(
            "SELECT COUNT(*) FROM sales a,sales b,sales c,sales d,sales e,sales f,sales g,sales h",
            "q1",
        )
    assert exc.value.code == "timeout"


def test_schema_arguments(settings):
    with pytest.raises(DatabaseError):
        Database(settings).inspect_schema(["people); DROP TABLE people;--"])


def test_syntax_is_recoverable():
    with pytest.raises(DatabaseError) as exc:
        validate_sql("SELECT FROM")
    assert exc.value.recoverable


def test_unterminated_string_is_recoverable(settings):
    with pytest.raises(DatabaseError) as exc:
        Database(settings).execute("SELECT 'unfinished", "q1")
    assert exc.value.code == "syntax" and exc.value.recoverable

import json
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager

import sqlglot
from sqlglot import exp

from src.config import Settings
from src.models import QueryResult

# Functions are checked again by SQLite after name resolution.
SAFE_FUNCTIONS = frozenset(
    "abs avg cast ceil ceiling coalesce count date datetime floor ifnull instr julianday "
    "length like lower ltrim max min nullif replace round rtrim strftime substr substring "
    "sum total trim typeof upper row_number rank dense_rank lag lead first_value last_value "
    "nth_value ntile percent_rank cume_dist iif concat concat_ws unixepoch timediff".split()
)


class DatabaseError(Exception):
    def __init__(self, code: str, message: str, recoverable: bool = False):
        super().__init__(message)
        self.code = code
        self.recoverable = recoverable

    def as_dict(self) -> dict:
        return {"code": self.code, "message": str(self), "recoverable": self.recoverable}


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def validate_sql(sql: str) -> str:
    if len(sql) > 16000:
        raise DatabaseError("unsafe_sql", "Consulta excede o tamanho permitido.")
    try:
        statements = sqlglot.parse(sql, read="sqlite")
    except (sqlglot.errors.ParseError, sqlglot.errors.TokenError) as exc:
        raise DatabaseError("syntax", "Sintaxe SQL inválida: " + str(exc)[:300], True) from exc
    except RecursionError as exc:
        raise DatabaseError("unsafe_sql", "Consulta excede a complexidade permitida.") from exc
    if len(statements) != 1 or statements[0] is None:
        raise DatabaseError("unsafe_sql", "Envie exatamente uma instrução SELECT.")
    tree = statements[0]
    if not isinstance(tree, (exp.Select, exp.Union, exp.Intersect, exp.Except)):
        raise DatabaseError("unsafe_sql", "Somente consultas SELECT são autorizadas.")
    for node in tree.walk():
        if isinstance(node, (exp.DDL, exp.DML, exp.Command, exp.Into)):
            raise DatabaseError("unsafe_sql", "Operação não autorizada na consulta.")
        if isinstance(node, exp.With) and node.args.get("recursive"):
            raise DatabaseError("unsafe_sql", "CTEs recursivas não são autorizadas.")
        if isinstance(node, exp.Table) and (node.db or node.catalog):
            raise DatabaseError("unsafe_sql", "Referências a bancos qualificados são bloqueadas.")
        if isinstance(node, exp.Func) and not isinstance(
            node, (exp.And, exp.Or, exp.Exists, exp.Case, exp.If)
        ):
            name = node.name.lower() if isinstance(node, exp.Anonymous) else node.sql_name().lower()
            # SQLGlot normalizes SQLite names into portable expressions.
            aliases = {
                "time_to_str": "strftime",
                "ts_or_ds_to_date": "date",
                "ts_or_ds_to_timestamp": "datetime",
            }
            if aliases.get(name, name) not in SAFE_FUNCTIONS:
                raise DatabaseError("unsafe_sql", f"Função não permitida: {name}.")
    return tree.sql(dialect="sqlite")


class Database:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.path = settings.database_path.resolve()
        if not self.path.is_file():
            raise DatabaseError(
                "missing_database", "Banco SQLite não encontrado no caminho configurado."
            )

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True, timeout=1)
        try:
            connection.execute("PRAGMA query_only=ON")
            connection.execute("PRAGMA trusted_schema=OFF")
            connection.enable_load_extension(False)
            connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, self.settings.max_result_bytes)
            connection.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, 16000)
            connection.setlimit(sqlite3.SQLITE_LIMIT_EXPR_DEPTH, 50)
            yield connection
        finally:
            connection.close()

    def list_tables(self) -> list[str]:
        with self.connect() as connection:
            return [
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_schema WHERE type='table' "
                    "AND name NOT LIKE 'sqlite_%' ORDER BY name"
                )
            ]

    def inspect_schema(self, tables: list[str]) -> dict:
        known = self.list_tables()
        if not tables or len(tables) > 8 or any(table not in known for table in tables):
            raise DatabaseError("schema", "Selecione até oito tabelas existentes.", True)
        result = {}
        with self.connect() as connection:
            all_columns = {
                table: connection.execute(
                    f"PRAGMA table_info({quote_identifier(table)})"
                ).fetchall()
                for table in known
            }
            for table in tables:
                columns = all_columns[table]
                foreign_keys = connection.execute(
                    f"PRAGMA foreign_key_list({quote_identifier(table)})"
                ).fetchall()
                candidates = []
                for column in columns:
                    if column[1].endswith("_id"):
                        stem = column[1][:-3]
                        for other, other_columns in all_columns.items():
                            if other != table and other.rstrip("s") == stem.rstrip("s"):
                                for target in other_columns:
                                    if target[5] and target[2] == column[2]:
                                        candidates.append(
                                            {
                                                "column": column[1],
                                                "table": other,
                                                "target": target[1],
                                                "confirmed": False,
                                            }
                                        )
                result[table] = {
                    "columns": [
                        {
                            "name": r[1],
                            "type": r[2],
                            "nullable": not bool(r[3] or r[5]),
                            "primary_key": bool(r[5]),
                        }
                        for r in columns
                    ],
                    "foreign_keys": [
                        {"table": r[2], "column": r[3], "target": r[4]} for r in foreign_keys
                    ],
                    "candidate_relationships": candidates,
                }
        return result

    def execute(self, sql: str, result_id: str) -> QueryResult:
        normalized = validate_sql(sql)
        allowed_tables = set(self.list_tables())
        started = time.monotonic()

        def authorize(action, arg1, arg2, database, trigger):
            if action == sqlite3.SQLITE_SELECT:
                return sqlite3.SQLITE_OK
            if (
                action == sqlite3.SQLITE_READ
                and arg1 in allowed_tables
                and (database == "main" or (database is None and arg2 == ""))
            ):
                return sqlite3.SQLITE_OK
            if action == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() in SAFE_FUNCTIONS:
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY

        try:
            with self.connect() as connection:
                connection.set_authorizer(authorize)
                connection.set_progress_handler(
                    lambda: int(time.monotonic() - started > self.settings.sql_timeout_seconds),
                    1000,
                )
                cursor = connection.execute(normalized)
                columns = [item[0] for item in cursor.description]
                if len(set(columns)) != len(columns):
                    raise DatabaseError(
                        "inconsistent", "Colunas repetidas: use aliases únicos.", True
                    )
                rows = []
                size = 0
                truncated = False
                for row in cursor:
                    values = list(row)
                    if any(isinstance(value, bytes) for value in values):
                        raise DatabaseError(
                            "inconsistent", "Resultados binários não são suportados."
                        )
                    size += len(json.dumps(values, ensure_ascii=False).encode())
                    if len(rows) >= self.settings.max_rows or size > self.settings.max_result_bytes:
                        truncated = True
                        break
                    rows.append(values)
        except sqlite3.Error as exc:
            message = str(exc)
            if "interrupted" in message:
                raise DatabaseError("timeout", "Consulta excedeu o limite de tempo.") from exc
            if "authorized" in message or "prohibited" in message:
                raise DatabaseError(
                    "unsafe_sql", "A autorização nativa bloqueou a consulta."
                ) from exc
            if "too big" in message:
                raise DatabaseError("volume", "Consulta excedeu o limite de volume.") from exc
            raise DatabaseError("execution", message[:300], True) from exc
        status = "truncated" if truncated else ("ok" if rows else "empty")
        return QueryResult(
            id=result_id,
            sql=normalized,
            columns=columns,
            rows=rows,
            status=status,
            duration_ms=round((time.monotonic() - started) * 1000, 2),
        )

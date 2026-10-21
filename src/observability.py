import json
import sqlite3
import time
from pathlib import Path


class AuditLog:
    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "events.jsonl"

    def write(self, run_id: str, event: dict) -> None:
        # SQL, prompts and result rows stay in the session, not in persistent logs.
        allowed = {
            key: event[key]
            for key in (
                "stage",
                "tool",
                "duration_ms",
                "error_code",
                "attempt",
                "status",
                "usage",
                "http_status",
            )
            if key in event
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"run_id": run_id, "timestamp": time.time(), **allowed}) + "\n")


class BudgetLedger:
    """Reserves worst-case cost atomically across local sessions before each API request."""

    def __init__(self, directory: Path, budget: float):
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "usage.sqlite"
        self.budget = budget
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS usage "
                "(id INTEGER PRIMARY KEY, run_id TEXT, reserved REAL, actual REAL)"
            )

    def reserve(self, run_id: str, amount: float) -> int:
        with sqlite3.connect(self.path, timeout=5) as connection:
            connection.execute("BEGIN IMMEDIATE")
            total = connection.execute(
                "SELECT COALESCE(SUM(COALESCE(actual, reserved)),0) FROM usage"
            ).fetchone()[0]
            if total + amount > self.budget:
                raise ValueError("Orçamento local insuficiente para uma nova chamada.")
            cursor = connection.execute(
                "INSERT INTO usage(run_id,reserved) VALUES (?,?)", (run_id, amount)
            )
            return cursor.lastrowid

    def settle(self, reservation: int, actual: float) -> None:
        with sqlite3.connect(self.path) as connection:
            connection.execute("UPDATE usage SET actual=? WHERE id=?", (actual, reservation))

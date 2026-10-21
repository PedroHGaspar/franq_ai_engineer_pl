import sqlite3

import pytest

from src.observability import AuditLog, BudgetLedger


def test_budget_reserves_before_call_and_keeps_unknown_cost(tmp_path):
    ledger = BudgetLedger(tmp_path, 0.5)
    reservation = ledger.reserve("run1", 0.4)
    with pytest.raises(ValueError):
        ledger.reserve("run2", 0.2)
    ledger.settle(reservation, 0.1)
    ledger.reserve("run2", 0.3)
    with pytest.raises(ValueError):
        BudgetLedger(tmp_path, 0.5).reserve("run3", 0.2)
    with sqlite3.connect(ledger.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM usage").fetchone()[0] == 2


def test_logs_exclude_queries_keys_and_rows(tmp_path):
    log = AuditLog(tmp_path)
    log.write(
        "run1",
        {
            "stage": "execute",
            "sql": "private",
            "api_key": "secret",
            "rows": ["personal"],
            "duration_ms": 12,
        },
    )
    content = log.path.read_text()
    assert "private" not in content and "secret" not in content and "personal" not in content
    assert "run1" in content and "12" in content

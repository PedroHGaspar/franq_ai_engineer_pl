import socket
import sqlite3

import pytest
from langchain_core.messages import AIMessage

from src.config import Settings


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("Testes gratuitos não podem abrir conexões de rede.")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


@pytest.fixture
def settings(tmp_path):
    path = tmp_path / "fixture.db"
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE people(id INTEGER PRIMARY KEY, region TEXT);
            CREATE TABLE sales(id INTEGER PRIMARY KEY, people_id INTEGER REFERENCES people(id),
                               amount REAL, sold_on TEXT, channel TEXT);
            INSERT INTO people VALUES (1,'Sul'),(2,'Norte'),(3,'Sul');
            INSERT INTO sales VALUES (1,1,10,'2025-05-01','App'),
              (2,1,20,'2025-05-02','App'),(3,2,30,'2025-06-01','Site');
        """)
    return Settings(_env_file=None, database_path=path, local_dir=tmp_path / "local")


class ScriptedModel:
    def __init__(self, *steps):
        self.steps = iter(steps)
        self.messages = []

    def bind_tools(self, tools, **kwargs):
        self.tools = tools
        return self

    def invoke(self, messages):
        self.messages.append(messages)
        step = next(self.steps)
        if isinstance(step, Exception):
            raise step
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": step[0],
                    "args": step[1],
                    "id": f"call{len(self.messages)}",
                    "type": "tool_call",
                }
            ],
        )


def finish(ids=None, **overrides):
    return (
        "finish",
        {
            "answer": {
                "summary": "Resultado obtido pelas consultas indicadas.",
                "evidence_ids": ids or ["q1"],
                "assumptions": [],
                "chart": {"kind": "table"},
                **overrides,
            }
        },
    )

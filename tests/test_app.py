from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.agent import DataAgent
from tests.conftest import ScriptedModel, finish


def test_app_starts_without_paid_api(monkeypatch):
    monkeypatch.setenv("ENABLE_PAID_API", "false")
    app = AppTest.from_file(Path("app.py").resolve(), default_timeout=15).run()
    assert not app.exception
    assert app.chat_input[0].disabled
    assert "desabilitadas" in app.info[0].value


def test_app_question_table_and_transparency(settings, monkeypatch):
    agent = DataAgent(
        settings,
        ScriptedModel(
            ("execute_query", {"sql": "SELECT region,COUNT(*) n FROM people GROUP BY region"}),
            finish(chart={"kind": "bar", "x": "region", "y": "n"}),
        ),
    )
    monkeypatch.setenv("ENABLE_PAID_API", "true")
    monkeypatch.setattr("src.agent.DataAgent", lambda settings: agent)
    app = AppTest.from_file(Path("app.py").resolve(), default_timeout=15).run()
    app.chat_input[0].set_value("Clientes por região").run()
    assert not app.exception
    assert len(app.dataframe) == 2
    assert any("SELECT" in block.value for block in app.code)
    assert len(app.session_state["history"]) == 1

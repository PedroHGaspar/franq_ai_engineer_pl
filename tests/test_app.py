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
    assert list(app.dataframe[0].value.columns) == ["region", "n"]
    assert len(app.dataframe) == 3
    assert len(app.get("plotly_chart")) == 1
    assert any("SELECT" in block.value for block in app.code)
    assert len(app.session_state["history"]) == 1


def start_app(monkeypatch, agent):
    monkeypatch.setenv("ENABLE_PAID_API", "true")
    monkeypatch.setattr("src.agent.DataAgent", lambda settings: agent)
    return AppTest.from_file(Path("app.py").resolve(), default_timeout=15).run()


def test_clarification_resume_history_and_clear(settings, monkeypatch):
    agent = DataAgent(
        settings,
        ScriptedModel(
            ("clarify", {"question": "Qual ano?"}),
            ("execute_query", {"sql": "SELECT COUNT(*) n FROM people"}),
            finish(),
        ),
    )
    app = start_app(monkeypatch, agent)
    app.chat_input[0].set_value("Clientes em maio").run()
    assert app.session_state["pending_thread"]
    assert any("Qual ano?" in item.value for item in app.info)
    app.chat_input[0].set_value("2025").run()
    assert not app.exception
    assert app.session_state["pending_thread"] is None
    assert len(app.session_state["history"]) == 2
    app.run()
    assert len(app.chat_message) == 4
    app.sidebar.button[0].click().run()
    assert not app.session_state["history"]


def test_line_chart_uses_selected_evidence(settings, monkeypatch):
    agent = DataAgent(
        settings,
        ScriptedModel(
            ("inspect_schema", {"tables": ["sales"]}),
            ("execute_query", {"sql": "SELECT DISTINCT channel FROM sales"}),
            (
                "execute_query",
                {
                    "sql": "SELECT sold_on,channel,SUM(amount) total FROM sales "
                    "GROUP BY sold_on,channel"
                },
            ),
            finish(
                ids=["q1", "q2"],
                chart={
                    "kind": "line",
                    "result_id": "q2",
                    "x": "sold_on",
                    "y": "total",
                    "color": "channel",
                },
            ),
        ),
    )
    app = start_app(monkeypatch, agent)
    app.chat_input[0].set_value("Evolução por canal").run()
    assert not app.exception
    assert len(app.get("plotly_chart")) == 1
    assert any("Metadados" in item.label for item in app.expander)
    assert any("SELECT DISTINCT channel FROM sales" in item.value for item in app.code)
    assert any("GROUP BY" in item.value for item in app.code)


def test_empty_and_incompatible_chart(settings, monkeypatch):
    agent = DataAgent(
        settings,
        ScriptedModel(
            ("execute_query", {"sql": "SELECT region FROM people WHERE 1=0"}),
            finish(),
            ("execute_query", {"sql": "SELECT region FROM people"}),
            finish(chart={"kind": "bar", "x": "region", "y": "missing"}),
        ),
    )
    app = start_app(monkeypatch, agent)
    app.chat_input[0].set_value("Sem dados").run()
    assert any("Nenhum registro" in item.value for item in app.info)
    app.chat_input[0].set_value("Gráfico incompatível").run()
    assert not app.exception
    assert any("não são compatíveis" in item.value for item in app.info)
    assert not app.get("plotly_chart")


def test_failure_preserves_history_and_allows_retry(settings, monkeypatch):
    agent = DataAgent(
        settings,
        ScriptedModel(
            RuntimeError("offline"),
            ("execute_query", {"sql": "SELECT COUNT(*) n FROM people"}),
            finish(),
        ),
    )
    app = start_app(monkeypatch, agent)
    app.chat_input[0].set_value("Primeira tentativa").run()
    assert app.error
    assert not app.session_state["busy"]
    app.chat_input[0].set_value("Nova tentativa").run()
    assert not app.exception
    assert len(app.session_state["history"]) == 2
    assert app.session_state["history"][-1]["state"]["status"] == "completed"


def test_busy_session_disables_new_submission(settings, monkeypatch):
    app = start_app(monkeypatch, DataAgent(settings, ScriptedModel()))
    app.session_state["busy"] = True
    app.run()
    assert app.chat_input[0].disabled

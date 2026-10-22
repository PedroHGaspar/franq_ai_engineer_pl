from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.agent import DataAgent
from tests.conftest import ScriptedModel, finish

# Cold imports through WSL on a Windows drive exceed 15 s; this is not a functional limit.
TIMEOUT = 60


def test_app_starts_without_paid_api(monkeypatch):
    monkeypatch.setenv("ENABLE_PAID_API", "false")
    app = AppTest.from_file(Path("app.py").resolve(), default_timeout=TIMEOUT).run()
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
    app = AppTest.from_file(Path("app.py").resolve(), default_timeout=TIMEOUT).run()
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
    return AppTest.from_file(Path("app.py").resolve(), default_timeout=TIMEOUT).run()


def test_clarification_resume_history_and_clear(settings, monkeypatch):
    agent = DataAgent(
        settings,
        ScriptedModel(
            ("clarify", {"question": "Todos os cadastros ou apenas compradores?"}),
            ("execute_query", {"sql": "SELECT COUNT(*) n FROM people"}),
            finish(),
        ),
    )
    app = start_app(monkeypatch, agent)
    app.chat_input[0].set_value("Quantos clientes temos?").run()
    assert app.session_state["pending_thread"]
    assert any("compradores" in item.value for item in app.info)
    app.chat_input[0].set_value("2025").run()
    assert not app.exception
    assert app.session_state["pending_thread"] is None
    assert len(app.session_state["history"]) == 2
    app.run()
    assert len(app.chat_message) == 4
    links = [item.proto.body for item in app.sidebar.get("html")]
    assert any('href="#pergunta-2"' in link for link in links)
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
    assert any(item.value == "Nenhum registro encontrado" for item in app.subheader)
    assert any("ausência de registros, não falha" in item.value for item in app.markdown)
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
    assert app.error and "não foi concluída" in app.error[0].value
    assert not app.chat_input[0].disabled
    app.chat_input[0].set_value("Nova tentativa").run()
    assert not app.exception
    assert len(app.session_state["history"]) == 2
    assert app.session_state["history"][-1]["state"]["status"] == "completed"


def test_pending_entry_disables_input_and_runs_once(settings, monkeypatch):
    agent = DataAgent(
        settings,
        ScriptedModel(("execute_query", {"sql": "SELECT COUNT(*) n FROM people"}), finish()),
    )
    app = start_app(monkeypatch, agent)
    app.session_state["history"] = [
        {"id": "e1", "question": "Total de pessoas", "state": None, "thread": None}
    ]
    app.run()
    assert app.session_state["history"][0]["state"]["status"] == "completed"
    app.run()
    assert not app.exception and not app.chat_input[0].disabled
    assert agent.model.messages and len(agent.model.messages) == 2


def test_scalar_result_renders_metric(settings, monkeypatch):
    agent = DataAgent(
        settings,
        ScriptedModel(
            ("execute_query", {"sql": "SELECT COUNT(*) AS clientes FROM people"}),
            finish(),
        ),
    )
    app = start_app(monkeypatch, agent)
    app.chat_input[0].set_value("Total de clientes").run()
    assert not app.exception
    assert app.metric[0].label == "Clientes"
    assert app.metric[0].value == "3"
    assert not app.get("plotly_chart")


def test_ranking_and_unverified_narrative_in_transparency(settings, monkeypatch):
    agent = DataAgent(
        settings,
        ScriptedModel(
            (
                "execute_query",
                {
                    "sql": "SELECT region,COUNT(*) n FROM people "
                    "GROUP BY region ORDER BY n DESC LIMIT 5"
                },
            ),
            finish(summary="Não existem clientes."),
        ),
    )
    app = start_app(monkeypatch, agent)
    app.chat_input[0].set_value("Ranking de clientes").run()
    assert not app.exception
    assert any(item.value == "1. Sul — 2" for item in app.text)
    # Problem D: a verified answer must not carry an unconditional "not confirmed" alert.
    assert not app.info
    original = next(item for item in app.expander if "original do modelo" in item.label)
    assert any(item.value == "Não existem clientes." for item in original.text)
    assert any("Desafio Técnico 1" in item.value for item in app.caption)


def test_optional_original_banner(monkeypatch):
    import pytest

    if not Path("assets/franq-banner.png").exists():
        pytest.skip("Marca privada opcional ausente")
    monkeypatch.setenv("ENABLE_PAID_API", "false")
    app = AppTest.from_file(Path("app.py").resolve(), default_timeout=TIMEOUT).run()
    assert not app.exception
    # Rendered as plain HTML, so no fullscreen button is attached to the banner.
    assert not app.get("image")
    assert any('alt="Franq"' in item.proto.body for item in app.get("html"))


def test_unexpected_error_names_its_type_without_details(settings, monkeypatch):
    agent = DataAgent(settings, ScriptedModel())

    def broken(*args, **kwargs):
        raise TypeError("secret-detail")

    monkeypatch.setattr(agent, "run", broken)
    app = start_app(monkeypatch, agent)
    app.chat_input[0].set_value("Total de vendas").run()
    assert "TypeError" in app.error[0].value and "reinicie o servidor" in app.error[0].value
    assert "secret-detail" not in str(app.session_state["history"])

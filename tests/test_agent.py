import pytest

from src.agent import DataAgent, has_numeric_evidence
from tests.conftest import ScriptedModel, finish


def test_discovery_multiple_queries_and_transparency(settings):
    model = ScriptedModel(
        ("list_tables", {}),
        ("inspect_schema", {"tables": ["sales"]}),
        ("execute_query", {"sql": "SELECT DISTINCT channel FROM sales"}),
        (
            "execute_query",
            {"sql": "SELECT COUNT(DISTINCT people_id) n FROM sales WHERE channel='App'"},
        ),
        finish(["q2"]),
    )
    state = DataAgent(settings, model).run("Quantos clientes compraram pelo aplicativo?")
    assert state["status"] == "completed"
    assert state["results"][-1]["rows"] == [[1]]
    assert state["queries"] == 2
    assert len([e for e in state["events"] if e["stage"] == "execute"]) == 2
    assert all("sql" in e for e in state["events"] if e["stage"] == "execute")
    assert "SELECT" not in (settings.local_dir / "events.jsonl").read_text()


def test_error_recovery(settings):
    model = ScriptedModel(
        ("execute_query", {"sql": "SELECT missing FROM sales"}),
        ("inspect_schema", {"tables": ["sales"]}),
        ("execute_query", {"sql": "SELECT COUNT(*) n FROM sales"}),
        finish(["q2"]),
    )
    state = DataAgent(settings, model).run("Quantas vendas?")
    assert state["status"] == "completed" and state["corrections"] == 1
    assert state["results"][0]["rows"] == [[3]]
    assert "no such column" in str(model.messages[1])


def test_retry_limit(settings):
    settings.max_corrections = 1
    state = DataAgent(
        settings,
        ScriptedModel(
            ("execute_query", {"sql": "SELECT missing FROM sales"}),
            ("execute_query", {"sql": "SELECT other_missing FROM sales"}),
        ),
    ).run("Quantas vendas?")
    assert state["status"] == "failed" and state["model_calls"] == 2


def test_repeated_invalid_sql_stops(settings):
    state = DataAgent(
        settings,
        ScriptedModel(
            ("execute_query", {"sql": "SELECT missing FROM sales"}),
            ("execute_query", {"sql": "select missing from sales"}),
        ),
    ).run("Investigue")
    assert state["error"]["code"] == "repeated_sql"


@pytest.mark.parametrize("limit", ["max_iterations", "max_model_calls"])
def test_iteration_limits(settings, limit):
    setattr(settings, limit, 1)
    state = DataAgent(settings, ScriptedModel(("list_tables", {}))).run("Investigue")
    assert state["status"] == "limit" and state["model_calls"] == 1


def test_query_limit(settings):
    settings.max_queries = 1
    state = DataAgent(
        settings,
        ScriptedModel(
            ("execute_query", {"sql": "SELECT 1 n"}),
            ("execute_query", {"sql": "SELECT 2 n"}),
        ),
    ).run("Investigue")
    assert state["error"]["code"] == "query_limit"


def test_empty_is_success(settings):
    state = DataAgent(
        settings,
        ScriptedModel(
            ("execute_query", {"sql": "SELECT * FROM people WHERE id=99"}),
            finish(),
        ),
    ).run("Cliente inexistente")
    assert state["status"] == "completed"
    assert state["results"][0]["status"] == "empty" and state["corrections"] == 0


def test_unavailable_llm_redacts_secrets(settings):
    state = DataAgent(settings, ScriptedModel(RuntimeError("secret-key private-url"))).run("Vendas")
    assert state["error"]["code"] == "llm_unavailable"
    assert "secret-key" not in str(state)


def test_unsafe_sql_never_retries(settings):
    state = DataAgent(settings, ScriptedModel(("execute_query", {"sql": "DROP TABLE sales"}))).run(
        "Apague"
    )
    assert state["status"] == "failed" and state["corrections"] == 0


def test_invalid_evidence(settings):
    settings.max_corrections = 0
    state = DataAgent(settings, ScriptedModel(finish(["invented"]))).run("Vendas")
    assert state["error"]["code"] == "inconsistent" and state["answer"] is None


def test_model_clarification_resumes_checkpoint(settings):
    agent = DataAgent(
        settings,
        ScriptedModel(
            ("clarify", {"question": "Valor bruto ou líquido?"}),
            ("execute_query", {"sql": "SELECT SUM(amount) total FROM sales"}),
            finish(),
        ),
    )
    assert agent.run("Qual o total vendido?", "session")["status"] == "clarification"
    result = agent.resume("session", "bruto")
    assert result["status"] == "completed" and result["model_calls"] == 3
    assert result["clarifications"] == ["bruto"]


def test_paid_api_disabled(settings):
    with pytest.raises(ValueError, match="desabilitada"):
        DataAgent(settings)


def test_inconsistent_answer_can_be_corrected(settings):
    model = ScriptedModel(
        ("execute_query", {"sql": "SELECT COUNT(*) n FROM sales"}),
        finish(summary="Foram 999 vendas."),
        finish(summary="Foram 3 vendas."),
    )
    state = DataAgent(settings, model).run("Quantas vendas?")
    assert state["status"] == "completed" and state["corrections"] == 1
    assert any(event.get("error_code") == "inconsistent" for event in state["events"])
    assert any("999" in event.get("message", "") for event in state["events"])


def test_syntax_recovery(settings):
    state = DataAgent(
        settings,
        ScriptedModel(
            ("execute_query", {"sql": "SELECT FROM"}),
            ("execute_query", {"sql": "SELECT COUNT(*) n FROM sales"}),
            finish(["q2"]),
        ),
    ).run("Total de vendas")
    assert state["status"] == "completed" and state["corrections"] == 1


def test_truncation_requires_new_query(settings):
    settings.max_rows = 1
    state = DataAgent(
        settings,
        ScriptedModel(
            ("execute_query", {"sql": "SELECT * FROM sales"}),
            finish(),
            ("execute_query", {"sql": "SELECT SUM(amount) total FROM sales"}),
            finish(["q2"]),
        ),
    ).run("Total vendido")
    assert state["status"] == "completed"
    assert state["results"][-1]["rows"] == [[60.0]]


def test_missing_data_can_be_explained_after_schema(settings):
    model = ScriptedModel(
        ("inspect_schema", {"tables": ["sales", "people"]}),
        finish(answerable=False, evidence_ids=[], summary="O banco não possui dados de frete."),
    )
    state = DataAgent(settings, model).run("Custo de frete por transportadora")
    assert state["status"] == "completed" and state["queries"] == 0


def test_provider_configuration_error_is_not_transient(settings):
    class ConfigurationError(Exception):
        status_code = 400

    state = DataAgent(settings, ScriptedModel(ConfigurationError("secret-provider-detail"))).run(
        "Total de vendas"
    )
    assert state["error"]["code"] == "llm_configuration"
    assert state["events"][-1]["http_status"] == 400
    assert "secret-provider-detail" not in str(state)


@pytest.mark.parametrize(
    "text,value,supported",
    [
        ("2,2113", 157 / 71, True),
        ("2,1622", 160 / 74, True),
        ("1,9759", 164 / 83, True),
        ("2,2114", 157 / 71, False),
        ("3", 2.9, False),
        ("51", 19, False),
        ("19,00", 19, True),
    ],
)
def test_numeric_evidence_accepts_only_supported_rounding(text, value, supported):
    assert has_numeric_evidence(text, {str(value)}, [value]) == supported


def test_chart_cannot_reference_uncited_query(settings):
    state = DataAgent(
        settings,
        ScriptedModel(
            ("execute_query", {"sql": "SELECT region,COUNT(*) n FROM people GROUP BY region"}),
            finish(chart={"kind": "bar", "x": "region", "y": "n", "result_id": "q99"}),
            finish(chart={"kind": "bar", "x": "region", "y": "n", "result_id": "q1"}),
        ),
    ).run("Comparar regiões")
    assert state["status"] == "completed"
    assert state["corrections"] == 1
    assert state["answer"]["chart"]["result_id"] == "q1"


def test_safe_presentation_preserves_original_model_text(settings):
    text = "Não existem registros de vendas."
    state = DataAgent(
        settings,
        ScriptedModel(
            ("execute_query", {"sql": "SELECT COUNT(*) n FROM sales"}),
            finish(summary=text),
        ),
    ).run("Total de vendas")
    assert state["status"] == "completed"
    assert state["answer"]["summary"] == text
    assert state["presentation"]["value"] == "3"
    assert text not in state["presentation"]["summary"]
    assert state["events"][-1]["presentation_source"] == "deterministic"

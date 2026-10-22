import pytest

from src.agent import DataAgent, unsupported_numbers
from src.intent import infer_intent, resolve_period, shape_problems
from tests.conftest import ScriptedModel, finish

COVERAGE = ("2024-07-22", "2025-07-22")


def result(sql, columns=("n",), rows=((1,),)):
    return {"id": "q1", "sql": sql, "columns": list(columns), "rows": [list(r) for r in rows]}


@pytest.mark.parametrize(
    "question",
    [
        "Liste os 5 estados com maior número de clientes que compraram via app em maio.",
        "Quais os cinco estados com mais clientes comprando pelo App em Maio?",
    ],
)
def test_month_without_year_uses_most_recent_year_with_data(question):
    assert resolve_period(question, COVERAGE).startswith("maio de 2025")


def test_last_year_is_the_twelve_months_up_to_the_latest_record():
    period = resolve_period("Qual a tendência de reclamações por canal no último ano?", COVERAGE)
    assert period.startswith("23/07/2024 a 22/07/2025")


@pytest.mark.parametrize(
    "question",
    [
        "Quantos clientes interagiram com campanhas de WhatsApp em 2024?",
        "Liste os estados com mais compras em maio de 2025",
        "Qual o número de reclamações não resolvidas por canal?",
        "Quais categorias de produto tiveram o maior número de compras em média por cliente?",
    ],
)
def test_explicit_or_timeless_questions_keep_their_period(question):
    assert resolve_period(question, COVERAGE) is None


def test_intent_of_official_questions():
    trend = infer_intent("Qual a tendência de reclamações por canal no último ano?")
    assert trend.trend and trend.grouped and trend.aggregate
    average = infer_intent(
        "Quais categorias de produto tiveram o maior número de compras em média por cliente?"
    )
    # "por cliente" is the denominator of the average, not a grouping dimension.
    assert average.ranking and average.aggregate and not average.grouped
    count = infer_intent("Quantos clientes interagiram com campanhas de WhatsApp em 2024?")
    assert count.years == {"2024"} and not count.ranking


TREND = infer_intent("Qual a tendência de reclamações por canal?", ["2025"])
MONTHLY = (
    "SELECT strftime('%Y-%m', data_contato) AS mes, canal, COUNT(*) AS n FROM suporte "
    "WHERE data_contato >= '2025-01-01' GROUP BY 1, 2"
)


@pytest.mark.parametrize(
    "sql,columns,rows",
    [
        # Coverage probe seen in real runs: minimum date, maximum date and total contacts.
        (
            "SELECT MIN(data_contato) a, MAX(data_contato) b, COUNT(*) n FROM suporte "
            "WHERE data_contato >= '2025-01-01'",
            ("a", "b", "n"),
            (("2025-01-02", "2025-07-22", 90),),
        ),
        # Channel list seen in real runs.
        ("SELECT DISTINCT canal FROM suporte", ("canal",), (("Chat",), ("E-mail",))),
        # Aggregated by channel only: no period.
        (
            "SELECT canal, COUNT(*) n FROM suporte WHERE data_contato >= '2025-01-01' "
            "GROUP BY canal",
            ("canal", "n"),
            (("Chat", 3), ("E-mail", 4)),
        ),
        # Monthly series without the requested year in the filters.
        (MONTHLY.replace("2025-01-01", "2024-01-01"), ("mes", "canal", "n"), ()),
    ],
)
def test_trend_rejects_intermediate_queries(sql, columns, rows):
    assert shape_problems(TREND, result(sql, columns, rows))


def test_trend_accepts_monthly_series_by_channel():
    rows = (("2025-01", "Chat", 2), ("2025-02", "Chat", 3))
    assert not shape_problems(TREND, result(MONTHLY, ("mes", "canal", "n"), rows))


def test_metadata_question_accepts_discovery():
    intent = infer_intent("Quais canais existem no suporte?")
    assert not shape_problems(intent, result("SELECT DISTINCT canal FROM suporte"))


@pytest.mark.parametrize(
    "text,supported",
    [
        ("Entre 01/07/2024 e 31/07/2025.", True),  # inclusive end of an exclusive bound
        ("Total de 1.234 compras.", True),  # pt-BR thousands separator
        ("Os 3 canais analisados.", True),  # distinct values of a cited column
        ("Foram 999 reclamações.", False),
        ("Em 31/02/2025.", False),
        ("Em 15/03/2023.", False),
    ],
)
def test_numeric_validation_false_positives_and_true_negatives(text, supported):
    cited = result(
        "SELECT canal, COUNT(*) n FROM t WHERE d >= '2024-07-01' AND d < '2025-08-01' "
        "GROUP BY canal",
        ("canal", "n"),
        (("Chat", 1234), ("E-mail", 5), ("Telefone", 7)),
    )
    assert (not unsupported_numbers(text, [cited])) is supported


def test_primary_evidence_is_the_answer_not_the_first_discovery_query(settings):
    state = DataAgent(
        settings,
        ScriptedModel(
            ("execute_query", {"sql": "SELECT DISTINCT channel FROM sales"}),
            ("execute_query", {"sql": "SELECT channel, COUNT(*) n FROM sales GROUP BY channel"}),
            finish(["q1", "q2"]),
        ),
    ).run("Qual o número de vendas por canal?")
    assert state["status"] == "completed" and state["primary_id"] == "q2"
    assert state["presentation"]["result_id"] == "q2"
    assert state["presentation"]["items"] == ["App — 2", "Site — 1"]


def test_premature_conclusion_is_rejected_and_investigation_continues(settings):
    state = DataAgent(
        settings,
        ScriptedModel(
            ("execute_query", {"sql": "SELECT MIN(sold_on) a, MAX(sold_on) b FROM sales"}),
            finish(["q1"]),
            (
                "execute_query",
                {
                    "sql": "SELECT substr(sold_on,1,7) mes, channel, COUNT(*) n FROM sales "
                    "GROUP BY mes, channel"
                },
            ),
            finish(["q1", "q2"], chart={"kind": "line", "x": "mes", "y": "n", "color": "channel"}),
        ),
    ).run("Qual a evolução mensal das vendas por canal?")
    assert state["status"] == "completed" and state["primary_id"] == "q2"
    rejected = [
        e
        for e in state["events"]
        if e["stage"] == "finish" and e.get("error_code") == "insufficient_evidence"
    ]
    assert len(rejected) == 1
    assert state["presentation"]["title"] == "Evolução por período"


def test_limit_without_valid_conclusion_is_not_an_answer(settings):
    settings.max_corrections = 1
    discovery = ("execute_query", {"sql": "SELECT DISTINCT channel FROM sales"})
    state = DataAgent(settings, ScriptedModel(discovery, finish(["q1"]), finish(["q1"]))).run(
        "Qual o total de vendas por canal?"
    )
    assert state["status"] == "failed" and state["answer"] is None
    assert state["error"]["code"] == "insufficient_evidence"


def test_month_without_year_is_resolved_without_asking(settings):
    model = ScriptedModel(
        (
            "execute_query",
            {
                "sql": "SELECT COUNT(DISTINCT people_id) n FROM sales "
                "WHERE sold_on >= '2025-05-01' AND sold_on < '2025-06-01'"
            },
        ),
        finish(),
    )
    state = DataAgent(settings, model).run("Quantos clientes compraram em maio?")
    assert state["status"] == "completed" and state["results"][0]["rows"] == [[1]]
    assert state["period"].startswith("maio de 2025")
    assert any("Período resolvido" in str(m.content) for m in model.messages[0])


def test_resolved_year_must_reach_the_sql(settings):
    settings.max_corrections = 0
    state = DataAgent(
        settings,
        ScriptedModel(
            ("execute_query", {"sql": "SELECT COUNT(*) n FROM sales WHERE sold_on LIKE '%-05-%'"}),
            finish(),
        ),
    ).run("Quantas vendas em maio?")
    assert state["status"] == "failed" and "2025" in state["error"]["message"]


def test_empty_result_from_nonexistent_text_value_is_not_an_answer(settings):
    wrong = "SELECT COUNT(*) n FROM sales WHERE channel = 'app' GROUP BY channel"
    model = ScriptedModel(
        ("execute_query", {"sql": wrong}),
        finish(),
        ("execute_query", {"sql": wrong.replace("'app'", "'App'")}),
        finish(["q2"]),
    )
    state = DataAgent(settings, model).run("Quantas vendas por canal?")
    assert state["status"] == "completed" and state["results"][-1]["rows"] == [[2]]
    assert "valores existentes: App, Site" in str(model.messages[1][-1].content)
    assert any(e.get("error_code") == "insufficient_evidence" for e in state["events"])


def test_malformed_model_turn_is_retried(settings):
    class TextOnce(ScriptedModel):
        def invoke(self, messages):
            if not self.messages:
                self.messages.append(messages)
                from langchain_core.messages import AIMessage

                return AIMessage(content="Vou consultar o banco.")
            return super().invoke(messages)

    model = TextOnce(("execute_query", {"sql": "SELECT COUNT(*) n FROM sales"}), finish())
    state = DataAgent(settings, model).run("Quantas vendas?")
    assert state["status"] == "completed" and state["corrections"] == 1
    assert any(e.get("error_code") == "model_protocol" for e in state["events"])


def test_consecutive_runs_do_not_share_operational_state(settings):
    model = ScriptedModel(
        ("execute_query", {"sql": "SELECT COUNT(*) n FROM people"}),
        finish(),
        ("execute_query", {"sql": "SELECT COUNT(*) n FROM sales"}),
        finish(),
    )
    agent = DataAgent(settings, model)
    first = agent.run("Quantas pessoas?", "a")
    snapshot = repr(first)
    second = agent.run("Quantas vendas?", "b", [{"question": "Quantas pessoas?", "answer": "N: 3"}])
    assert repr(first) == snapshot
    assert second["run_id"] != first["run_id"]
    assert [r["rows"] for r in second["results"]] == [[[3]]]
    assert [r["sql"] for r in second["results"]] == ["SELECT COUNT(*) AS n FROM sales"]
    context = " ".join(str(m.content) for m in model.messages[2])
    assert "Quantas pessoas?" in context and "people" not in context

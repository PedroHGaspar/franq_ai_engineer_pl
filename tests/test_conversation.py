"""Conversation scenarios on the original database with scripted models (no paid calls).

The scripted SQL plays the role of the model; references are computed independently in Python
from raw rows, so a passing test means the agent kept, selected and presented the right result.
"""

import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.agent import DataAgent
from src.config import Settings
from tests.conftest import ScriptedModel, finish
from tests.test_app import TIMEOUT

ORIGINAL = Path("anexo_desafio_1.db")
pytestmark = [
    pytest.mark.original,
    pytest.mark.skipif(not ORIGINAL.is_file(), reason="Banco privado ausente"),
]

STATES_SQL = """SELECT c.estado, COUNT(DISTINCT c.id) AS clientes
FROM clientes c JOIN compras p ON p.cliente_id = c.id
WHERE p.canal = 'App' AND p.data_compra >= '2025-05-01' AND p.data_compra < '2025-06-01'
GROUP BY c.estado ORDER BY clientes DESC, c.estado LIMIT 5"""
WHATSAPP_SQL = """SELECT COUNT(DISTINCT cliente_id) AS clientes FROM campanhas_marketing
WHERE canal = 'WhatsApp' AND interagiu = 1
AND data_envio >= '2024-01-01' AND data_envio < '2025-01-01'"""
CATEGORIES_SQL = """SELECT categoria, COUNT(*) * 1.0 / COUNT(DISTINCT cliente_id) AS media
FROM compras GROUP BY categoria ORDER BY media DESC, categoria"""
UNRESOLVED_SQL = """SELECT canal, COUNT(*) AS reclamacoes_nao_resolvidas FROM suporte
WHERE tipo_contato = 'Reclamação' AND resolvido = 0 GROUP BY canal ORDER BY canal"""
TREND_SQL = """SELECT strftime('%Y-%m', data_contato) AS mes, canal, COUNT(*) AS reclamacoes
FROM suporte WHERE tipo_contato = 'Reclamação'
AND data_contato >= '2024-07-23' AND data_contato < '2025-07-23'
GROUP BY 1, 2 ORDER BY 1, 2"""


def query(sql):
    return ("execute_query", {"sql": sql})


def raw(sql):
    with sqlite3.connect(ORIGINAL.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        return connection.execute(sql).fetchall()


def reference_states():
    customers = dict(raw("SELECT id, estado FROM clientes"))
    buyers = {
        customer
        for customer, day, channel in raw("SELECT cliente_id, data_compra, canal FROM compras")
        if channel == "App" and day.startswith("2025-05")
    }
    counts = Counter(customers[customer] for customer in buyers)
    return [list(i) for i in sorted(counts.items(), key=lambda i: (-i[1], i[0]))[:5]]


def reference_whatsapp():
    return len(
        {
            customer
            for customer, channel, interacted, sent in raw(
                "SELECT cliente_id, canal, interagiu, data_envio FROM campanhas_marketing"
            )
            if channel == "WhatsApp" and interacted and sent.startswith("2024")
        }
    )


def reference_categories():
    purchases = raw("SELECT cliente_id, categoria FROM compras")
    counts = Counter(category for _, category in purchases)
    buyers = defaultdict(set)
    for customer, category in purchases:
        buyers[category].add(customer)
    means = [[k, v / len(buyers[k])] for k, v in counts.items()]
    return sorted(means, key=lambda i: (-i[1], i[0]))


def reference_unresolved():
    rows = raw("SELECT canal, tipo_contato, resolvido FROM suporte")
    counts = Counter(c for c, kind, solved in rows if kind == "Reclamação" and not solved)
    return [list(i) for i in sorted(counts.items())]


def reference_trend():
    rows = raw("SELECT data_contato, canal, tipo_contato FROM suporte")
    counts = Counter(
        (day[:7], c)
        for day, c, kind in rows
        if kind == "Reclamação" and "2024-07-23" <= day < "2025-07-23"
    )
    return [[*key, value] for key, value in sorted(counts.items())]


@pytest.fixture
def original(tmp_path):
    return Settings(_env_file=None, database_path=ORIGINAL, local_dir=tmp_path)


def start_app(monkeypatch, agent):
    monkeypatch.setenv("ENABLE_PAID_API", "true")
    monkeypatch.setattr("src.agent.DataAgent", lambda settings: agent)
    return AppTest.from_file(Path("app.py").resolve(), default_timeout=TIMEOUT).run()


def ask(app, text):
    app.chat_input[0].set_value(text).run()
    assert not app.exception
    return app.session_state["history"][-1]["state"]


def test_official_question_1_resolves_month_and_ranks_states(original, monkeypatch):
    model = ScriptedModel(
        ("list_tables", {}),
        ("inspect_schema", {"tables": ["clientes", "compras"]}),
        query(STATES_SQL.replace("'App'", "'app'")),
        finish(["q1"]),
        query(STATES_SQL),
        finish(["q2"]),
    )
    app = start_app(monkeypatch, DataAgent(original, model))
    state = ask(
        app, "Liste os 5 estados com maior número de clientes que compraram via app em maio."
    )
    assert state["status"] == "completed" and state["primary_id"] == "q2"
    assert state["period"].startswith("maio de 2025")
    assert state["results"][-1]["rows"] == reference_states()
    assert state["presentation"]["kind"] == "ranking"
    assert any(item.value.startswith("1. São Paulo") for item in app.text)
    assert any("Período considerado: maio de 2025" in item.value for item in app.caption)
    assert len(app.chat_message) == 2


@pytest.mark.parametrize(
    "question",
    [
        "Quantos clientes interagiram com campanhas de WhatsApp em 2024?",
        "Em 2024, quantos clientes únicos interagiram com campanhas pelo WhatsApp?",
    ],
)
def test_official_question_2_metric_and_wording_variation(original, question):
    state = DataAgent(original, ScriptedModel(query(WHATSAPP_SQL), finish())).run(question)
    assert state["status"] == "completed"
    assert state["presentation"]["kind"] == "metric"
    assert state["presentation"]["value"] == str(reference_whatsapp())


def test_official_question_3_average_by_category(original, monkeypatch):
    model = ScriptedModel(
        query(CATEGORIES_SQL),
        finish(summary="A média é compras divididas por compradores de cada categoria."),
    )
    app = start_app(monkeypatch, DataAgent(original, model))
    state = ask(
        app, "Quais categorias de produto tiveram o maior número de compras em média por cliente?"
    )
    assert state["status"] == "completed"
    rows, expected = state["results"][0]["rows"], reference_categories()
    assert [r[0] for r in rows] == [r[0] for r in expected]
    assert [r[1] for r in rows] == pytest.approx([r[1] for r in expected])
    assert len(state["presentation"]["items"]) == len(reference_categories())
    assert not app.info


def test_official_question_4_comparison_with_bar_chart(original, monkeypatch):
    chart = {"kind": "bar", "x": "canal", "y": "reclamacoes_nao_resolvidas", "color": "canal"}
    model = ScriptedModel(query(UNRESOLVED_SQL), finish(chart=chart))
    app = start_app(monkeypatch, DataAgent(original, model))
    state = ask(app, "Qual o número de reclamações não resolvidas por canal?")
    assert state["results"][0]["rows"] == reference_unresolved()
    assert state["presentation"]["summary"].startswith("Maior valor: Telefone (19)")
    assert len(app.get("plotly_chart")) == 1 and not app.info


def test_official_question_5_and_scenario_d_reject_discovery_until_trend(original, monkeypatch):
    chart = {"kind": "line", "x": "mes", "y": "reclamacoes", "color": "canal", "result_id": "q3"}
    model = ScriptedModel(
        query("SELECT DISTINCT tipo_contato, canal FROM suporte"),
        query("SELECT MIN(data_contato) inicio, MAX(data_contato) fim, COUNT(*) n FROM suporte"),
        finish(["q1", "q2"]),
        query(TREND_SQL),
        finish(["q1", "q3"], chart=chart),
    )
    app = start_app(monkeypatch, DataAgent(original, model))
    state = ask(app, "Qual a tendência de reclamações por canal no último ano?")
    assert state["period"].startswith("23/07/2024 a 22/07/2025")
    assert state["status"] == "completed" and state["primary_id"] == "q3"
    assert state["results"][-1]["rows"] == reference_trend()
    assert any(e.get("error_code") == "insufficient_evidence" for e in state["events"])
    assert state["presentation"]["title"] == "Evolução por período"
    assert len(state["presentation"]["items"]) == 3
    assert len(app.get("plotly_chart")) == 1


def test_scenario_a_independent_questions_keep_history_without_contamination(original, monkeypatch):
    model = ScriptedModel(
        query("SELECT COUNT(*) AS clientes FROM clientes"),
        finish(),
        query(WHATSAPP_SQL),
        finish(),
        query(UNRESOLVED_SQL),
        finish(),
    )
    app = start_app(monkeypatch, DataAgent(original, model))
    first = ask(app, "Quantos clientes temos?")
    second = ask(app, "Quantos clientes interagiram com campanhas de WhatsApp em 2024?")
    third = ask(app, "Qual o número de reclamações não resolvidas por canal?")
    history = app.session_state["history"]
    assert len({entry["id"] for entry in history}) == 3
    assert len({first["run_id"], second["run_id"], third["run_id"]}) == 3
    assert [len(s["results"]) for s in (first, second, third)] == [1, 1, 1]
    assert first["results"][0]["rows"] == [[100]]
    assert second["results"][0]["rows"] == [[reference_whatsapp()]]
    assert third["results"][0]["rows"] == reference_unresolved()
    assert history[0]["state"] is first  # earlier answers are kept, not recomputed
    app.run()  # rerender without new input
    assert len(app.chat_message) == 6 and len(app.session_state["history"]) == 3
    assert [item.value for item in app.metric] == ["100", str(reference_whatsapp())]
    # Context carries previous questions and summaries, never previous SQL or rows.
    context = " ".join(str(m.content) for m in model.messages[-2])
    assert "Quantos clientes temos?" in context and "campanhas_marketing" not in context


def test_scenario_c_repeated_question_is_consistent(original, monkeypatch):
    steps = (query(UNRESOLVED_SQL), finish())
    app = start_app(monkeypatch, DataAgent(original, ScriptedModel(*steps, *steps)))
    question = "Qual o número de reclamações não resolvidas por canal?"
    with_history = [ask(app, question), ask(app, question)]
    fresh = DataAgent(original, ScriptedModel(*steps)).run(question)
    rows = [state["results"][0]["rows"] for state in [*with_history, fresh]]
    assert rows == [reference_unresolved()] * 3
    assert with_history[0]["run_id"] != with_history[1]["run_id"]
    assert with_history[0]["presentation"] == fresh["presentation"]


def test_scenario_e_invalid_sql_and_recoverable_empty(original):
    state = DataAgent(
        original,
        ScriptedModel(
            query("SELECT canal, COUNT(*) n FROM suporte WHERE tipo = 'Reclamação' GROUP BY canal"),
            query(UNRESOLVED_SQL.replace("'Reclamação'", "'reclamação'")),
            query("SELECT DISTINCT tipo_contato FROM suporte"),
            query(UNRESOLVED_SQL),
            finish(["q4"]),
        ),
    ).run("Qual o número de reclamações não resolvidas por canal?")
    assert state["status"] == "completed" and state["corrections"] == 1
    assert [r["status"] for r in state["results"]] == ["empty", "ok", "ok"]
    assert state["results"][-1]["rows"] == reference_unresolved()


def test_scenario_f_limit_is_reported_as_unfinished(original, monkeypatch):
    original.max_model_calls = 3
    model = ScriptedModel(
        query("SELECT DISTINCT canal FROM suporte"),
        finish(["q1"]),
        query("SELECT DISTINCT tipo_contato FROM suporte"),
    )
    app = start_app(monkeypatch, DataAgent(original, model))
    state = ask(app, "Qual o número de reclamações não resolvidas por canal?")
    assert state["status"] == "limit" and state["answer"] is None
    assert "A análise não foi concluída" in app.error[0].value
    assert any("não constituem resposta" in item.value for item in app.caption)
    assert not app.metric and not app.get("plotly_chart") and not app.subheader[1:]


def test_scenario_c_trend_without_requested_category_is_rejected(original):
    # Real failure: same trend question once counted every support contact, not complaints.
    all_contacts = TREND_SQL.replace("WHERE tipo_contato = 'Reclamação'\nAND", "WHERE")
    chart = {"kind": "line", "x": "mes", "y": "reclamacoes", "color": "canal"}
    model = ScriptedModel(
        query(all_contacts),
        finish(chart={**chart, "result_id": "q1"}),
        query(TREND_SQL),
        finish(["q2"], chart={**chart, "result_id": "q2"}),
    )
    state = DataAgent(original, model).run(
        "Qual a tendência de reclamações por canal no último ano?"
    )
    assert state["status"] == "completed" and state["primary_id"] == "q2"
    assert state["results"][-1]["rows"] == reference_trend()
    assert any("Reclamação" in e.get("message", "") for e in state["events"])

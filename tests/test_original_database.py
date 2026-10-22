import sqlite3
from pathlib import Path

import pytest

from src.agent import DataAgent
from src.config import Settings
from src.database import Database
from tests.conftest import ScriptedModel, finish

ORIGINAL = Path("anexo_desafio_1.db")
pytestmark = [
    pytest.mark.original,
    pytest.mark.skipif(not ORIGINAL.is_file(), reason="Banco privado ausente"),
]


def test_reference_questions(tmp_path):
    database = Database(Settings(_env_file=None, database_path=ORIGINAL, local_dir=tmp_path))
    whatsapp = database.execute(
        """SELECT COUNT(DISTINCT cliente_id) clientes
        FROM campanhas_marketing WHERE canal='WhatsApp' AND interagiu=1
        AND data_envio >= '2024-01-01' AND data_envio < '2025-01-01'""",
        "w",
    )
    assert whatsapp.rows == [[17]]
    complaints = database.execute(
        """SELECT canal, COUNT(*) total FROM suporte
        WHERE tipo_contato='Reclamação' AND resolvido=0 GROUP BY canal ORDER BY canal""",
        "s",
    )
    assert complaints.rows == [["Chat", 18], ["E-mail", 14], ["Telefone", 19]]
    states = database.execute(
        """SELECT c.estado,COUNT(DISTINCT c.id) n
        FROM clientes c JOIN compras p ON p.cliente_id=c.id WHERE p.canal='App'
        AND p.data_compra>='2025-05-01' AND p.data_compra<'2025-06-01'
        GROUP BY c.estado ORDER BY n DESC,c.estado LIMIT 5""",
        "c",
    )
    assert states.rows == [
        ["São Paulo", 6],
        ["Minas Gerais", 3],
        ["Santa Catarina", 3],
        ["Alagoas", 2],
        ["Espírito Santo", 2],
    ]
    means = database.execute(
        """SELECT categoria,COUNT(*)*1.0/COUNT(DISTINCT cliente_id) media
        FROM compras GROUP BY categoria ORDER BY media DESC,categoria""",
        "m",
    )
    with sqlite3.connect(ORIGINAL.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        purchases = connection.execute("SELECT cliente_id,categoria FROM compras").fetchall()
    counts, buyers = {}, {}
    for customer, category in purchases:
        counts[category] = counts.get(category, 0) + 1
        buyers.setdefault(category, set()).add(customer)
    expected = sorted(
        [(k, v / len(buyers[k])) for k, v in counts.items()], key=lambda r: (-r[1], r[0])
    )
    assert [tuple(row) for row in means.rows] == expected


@pytest.mark.parametrize("start,end", [("2024-07-23", "2025-07-23"), ("2024-07-01", "2025-08-01")])
def test_monthly_trend_against_independent_python(tmp_path, start, end):
    database = Database(Settings(_env_file=None, database_path=ORIGINAL, local_dir=tmp_path))
    result = database.execute(
        f"""SELECT substr(data_contato,1,7) mes,canal,COUNT(*) n
        FROM suporte WHERE tipo_contato='Reclamação'
        AND data_contato>='{start}' AND data_contato<'{end}'
        GROUP BY mes,canal ORDER BY mes,canal""",
        "q1",
    )
    with sqlite3.connect(ORIGINAL.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        rows = connection.execute("SELECT data_contato,canal,tipo_contato FROM suporte").fetchall()
    expected = {}
    for date, channel, kind in rows:
        if kind == "Reclamação" and start <= date < end:
            key = (date[:7], channel)
            expected[key] = expected.get(key, 0) + 1
    assert result.rows == [[*key, value] for key, value in sorted(expected.items())]


def test_novel_question_with_mock_and_independent_reference(tmp_path):
    sql = """SELECT COUNT(*) n FROM clientes c WHERE NOT EXISTS
             (SELECT 1 FROM compras p WHERE p.cliente_id=c.id AND p.canal='App')"""
    state = DataAgent(
        Settings(_env_file=None, database_path=ORIGINAL, local_dir=tmp_path),
        ScriptedModel(("execute_query", {"sql": sql}), finish()),
    ).run("Quantos clientes nunca compraram pelo aplicativo?")
    with sqlite3.connect(ORIGINAL.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        customers = {r[0] for r in connection.execute("SELECT id FROM clientes")}
        app_buyers = {
            r[0] for r in connection.execute("SELECT cliente_id FROM compras WHERE canal='App'")
        }
    assert state["status"] == "completed"
    assert state["results"][0]["rows"] == [[len(customers - app_buyers)]]

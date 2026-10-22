from copy import deepcopy

import pytest

from src.models import QueryResult
from src.presentation import display_value, label_for, prepare_presentation


def payload(rows, columns, sql, chart=None):
    result = QueryResult(
        id="q1",
        rows=rows,
        columns=columns,
        sql=sql,
        status="ok" if rows else "empty",
        duration_ms=1,
    )
    answer = dict(
        summary="Não há registros em julho de 2025.",
        assumptions=[],
        evidence_ids=["q1"],
        chart=chart or {"kind": "table"},
    )
    return answer, [result.model_dump()]


def test_ranking_uses_ordered_sql_and_exact_values():
    answer, results = payload(
        [["Sul", 6], ["Norte", 3]],
        ["regiao", "clientes_unicos"],
        "SELECT regiao,COUNT(*) clientes_unicos FROM vendas GROUP BY regiao "
        "ORDER BY clientes_unicos DESC LIMIT 5",
    )
    before = deepcopy((answer, results))
    rendered = prepare_presentation(answer, results)
    assert rendered.kind == "ranking"
    assert rendered.items == ["Sul — 6", "Norte — 3"]
    assert (answer, results) == before
    results[0]["sql"] = "SELECT regiao,COUNT(*) clientes_unicos FROM vendas GROUP BY regiao"
    assert prepare_presentation(answer, results).kind == "table"


def test_inconsistent_text_replaced_without_rewriting_evidence():
    answer, results = payload(
        [["2025-07", "Chat", 1], ["2025-07", "E-mail", 3], ["2025-07", "Telefone", 2]],
        ["mes", "canal", "reclamacoes"],
        "SELECT mes,canal,reclamacoes FROM fatos",
        dict(kind="line", x="mes", y="reclamacoes", color="canal"),
    )
    before = deepcopy((answer, results))
    rendered = prepare_presentation(answer, results)
    assert rendered.kind == "line"
    assert "3 pontos" in rendered.summary
    assert "Não há registros" not in rendered.summary
    assert not rendered.narrative_verified
    assert (answer, results) == before


@pytest.mark.parametrize(
    "alias,expected",
    [
        ("reclamacoes_nao_resolvidas", "Reclamações não resolvidas"),
        ("clientes_unicos", "Clientes únicos"),
        ("mes", "Mês"),
        ("compras_media_por_cliente", "Média de compras por cliente"),
        ("volume_por_transportadora", "Volume por transportadora"),
    ],
)
def test_labels_include_unknown_aliases(alias, expected):
    assert label_for(alias) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        (1234.5, "1.234,5"),
        ("2025-07", "07/2025"),
        ("2025-07-23", "23/07/2025"),
        (None, "Não informado"),
        (0, "0"),
    ],
)
def test_display_values(value, expected):
    assert display_value(value) == expected


def test_missing_rows_are_not_zero_metric():
    answer, results = payload([], ["total"], "SELECT COUNT(*) total FROM vendas")
    view = prepare_presentation(answer, results)
    assert view.kind == "table" and view.value is None
    assert "Nenhum registro" in view.summary


def test_invented_chart_source_is_rejected():
    answer, results = payload(
        [[3]], ["total"], "SELECT 3 total", dict(kind="table", result_id="q9")
    )
    with pytest.raises(ValueError):
        prepare_presentation(answer, results)

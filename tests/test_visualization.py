import pytest

from src.models import ChartSpec, QueryResult
from src.visualization import build_chart, select_chart


def result(columns, rows, status="ok"):
    return QueryResult(
        id="q1", sql="SELECT ...", columns=columns, rows=rows, status=status, duration_ms=1
    )


def test_bar_and_line_use_real_values():
    data = result(["month", "n"], [["2025-02", 4], ["2025-01", 2]])
    chart = build_chart(data, ChartSpec(kind="line", x="month", y="n"))
    assert list(chart.data[0].y) == [2, 4]
    bars = build_chart(data, ChartSpec(kind="bar", x="month", y="n"))
    assert list(bars.data[0].y) == [4, 2]


@pytest.mark.parametrize(
    "data,spec",
    [
        (result(["n"], [[1]]), ChartSpec(kind="bar", x="missing", y="n")),
        (result(["x", "n"], [], "empty"), ChartSpec(kind="line", x="x", y="n")),
        (result(["x", "n"], [["A", 1], ["B", 2]]), ChartSpec(kind="line", x="x", y="n")),
        (result(["x", "n"], [["A", 1], ["A", 2]]), ChartSpec(kind="bar", x="x", y="n")),
        (result(["x", "n"], [["A", "one"], ["B", "two"]]), ChartSpec(kind="bar", x="x", y="n")),
    ],
)
def test_invalid_charts_fall_back_to_table(data, spec):
    assert select_chart(data, spec).kind == "table"


def test_line_series_keep_dates_and_values():
    data = result(
        ["mes", "canal", "total"],
        [
            ["2025-02-01", "Chat", 3],
            ["2025-01-01", "Chat", 1],
            ["2025-01-01", "Telefone", 2],
            ["2025-02-01", "Telefone", 4],
        ],
    )
    chart = build_chart(data, ChartSpec(kind="line", x="mes", y="total", color="canal"))
    assert {trace.name: list(trace.y) for trace in chart.data} == {
        "Chat": [1, 3],
        "Telefone": [2, 4],
    }
    assert all(str(trace.x[0]).startswith("2025-01-01") for trace in chart.data)
    assert chart.layout.xaxis.title.text == "Mês"


@pytest.mark.parametrize("invalid", [None, float("nan"), float("inf")])
def test_nonfinite_values_do_not_produce_chart(invalid):
    data = result(["canal", "total"], [["Chat", invalid], ["Telefone", 4]])
    assert build_chart(data, ChartSpec(kind="bar", x="canal", y="total")) is None


def test_friendly_hover_with_unknown_alias_and_decimals():
    data = result(["setor_novo", "volume_medio"], [["A", 1234.5], ["B", 2.25]])
    chart = build_chart(data, ChartSpec(kind="bar", x="setor_novo", y="volume_medio"))
    assert "Setor novo" in chart.data[0].hovertemplate
    assert "Volume medio" in chart.data[0].hovertemplate
    assert "volume_medio=" not in chart.data[0].hovertemplate
    assert list(chart.data[0].customdata[0]) == ["A", "1.234,5"]


def test_line_hover_preserves_dates_series_and_values():
    data = result(
        ["mes", "canal", "reclamacoes_nao_resolvidas"],
        [["2025-07", "Chat", 1], ["2025-07", "Telefone", 2]],
    )
    chart = build_chart(
        data, ChartSpec(kind="line", x="mes", y="reclamacoes_nao_resolvidas", color="canal")
    )
    assert "Mês" in chart.data[0].hovertemplate
    assert "Reclamações não resolvidas" in chart.data[0].hovertemplate
    assert list(chart.data[0].customdata[0]) == ["07/2025", "1", "Chat"]
    assert sum(len(t.y) for t in chart.data) == 2


def test_category_colors_stable_across_subsets():
    spec = ChartSpec(kind="bar", x="canal", y="n", color="canal")
    full = build_chart(result(["canal", "n"], [["Chat", 1], ["E-mail", 2], ["Telefone", 3]]), spec)
    subset = build_chart(result(["canal", "n"], [["E-mail", 2], ["Telefone", 3]]), spec)
    colors = {t.name: t.marker.color for t in full.data}
    assert len(set(colors.values())) == len(colors)
    assert all(colors[t.name] == t.marker.color for t in subset.data)
    assert not full.layout.showlegend


def test_many_categories_use_readable_table():
    data = result(["categoria", "total"], [[str(i), i] for i in range(25)])
    assert build_chart(data, ChartSpec(kind="bar", x="categoria", y="total")) is None

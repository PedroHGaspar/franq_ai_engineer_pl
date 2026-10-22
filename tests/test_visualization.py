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
    assert chart.layout.xaxis.title.text == "Mes"


@pytest.mark.parametrize("invalid", [None, float("nan"), float("inf")])
def test_nonfinite_values_do_not_produce_chart(invalid):
    data = result(["canal", "total"], [["Chat", invalid], ["Telefone", 4]])
    assert build_chart(data, ChartSpec(kind="bar", x="canal", y="total")) is None

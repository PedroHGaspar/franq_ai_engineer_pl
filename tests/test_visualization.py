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

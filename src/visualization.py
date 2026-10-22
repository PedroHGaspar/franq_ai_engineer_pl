import re
from hashlib import sha256
from html import escape
from textwrap import wrap

import pandas as pd
import plotly.express as px

from src.intent import PERIOD, normalize
from src.models import ChartSpec, QueryResult
from src.presentation import display_value, label_for

REQUESTED = re.compile(r"\b(tabela)\b|\b(?:graficos? de|em) (barras?|colunas?|linhas?)\b")


def default_chart(question: str, result: QueryResult) -> ChartSpec:
    """Chooses the chart from the data shape, so the same result always gets the same chart.

    An explicit request in the question (table, bars or lines) takes precedence.
    """
    match = REQUESTED.search(normalize(question))
    requested = (match.group(1) or match.group(2).rstrip("s")) if match else None
    if requested == "tabela":
        return ChartSpec(kind="table")
    rows, columns = result.rows, result.columns
    numbers = [
        i
        for i in range(len(columns))
        if rows and all(isinstance(r[i], (int, float)) and not isinstance(r[i], bool) for r in rows)
    ]
    period = next(
        (
            i
            for i in range(len(columns))
            if rows and all(isinstance(r[i], str) and PERIOD.match(r[i]) for r in rows)
        ),
        None,
    )
    labels = [i for i in range(len(columns)) if i != period and i not in numbers]
    explicit = {"barra": "bar", "coluna": "bar", "linha": "line"}.get(requested)
    if result.status != "ok" or len(rows) < 2 or not numbers or len(labels) > 1:
        return ChartSpec(kind=explicit or "table")
    y, label = columns[numbers[0]], (columns[labels[0]] if labels else None)
    if period is not None and explicit != "bar":
        return ChartSpec(kind="line", x=columns[period], y=y, color=label)
    if period is not None:
        return ChartSpec(kind="bar", x=columns[period], y=y, color=label)
    if label and len(rows) <= 20 and explicit != "line":
        return ChartSpec(kind="bar", x=label, y=y, color=label)
    return ChartSpec(kind=explicit or "table")


def select_chart(result: QueryResult, requested: ChartSpec) -> ChartSpec:
    if result.status != "ok" or not result.rows or requested.kind in {"none", "table"}:
        return ChartSpec(kind="table")
    frame = pd.DataFrame(result.rows, columns=result.columns)
    if requested.x not in frame or requested.y not in frame or requested.x == requested.y:
        return ChartSpec(kind="table")
    if requested.color and requested.color not in frame:
        return ChartSpec(kind="table")
    if requested.kind == "bar" and frame[requested.x].nunique() > 20:
        return ChartSpec(kind="table")
    if not pd.api.types.is_numeric_dtype(frame[requested.y]) or len(frame) < 2:
        return ChartSpec(kind="table")
    if (
        frame[requested.y].isna().any()
        or frame[requested.y].isin([float("inf"), -float("inf")]).any()
    ):
        return ChartSpec(kind="table")
    keys = [requested.x] + ([requested.color] if requested.color else [])
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        return ChartSpec(kind="table")
    if requested.kind == "line":
        if pd.to_datetime(frame[requested.x], errors="coerce", format="mixed").isna().any():
            return ChartSpec(kind="table")
        if pd.api.types.is_numeric_dtype(frame[requested.x]):
            return ChartSpec(kind="table")
    return requested


def build_chart(result: QueryResult, spec: ChartSpec):
    selected = select_chart(result, spec)
    if selected.kind == "table":
        return None
    frame = pd.DataFrame(result.rows, columns=result.columns)
    palette = [
        "#646FE6",
        "#4376A8",
        "#C46B29",
        "#965BB5",
        "#177C80",
        "#A34E68",
        "#687B45",
        "#CA6F1E",
        "#52697E",
        "#835A43",
        "#557EA0",
    ]
    colors = (
        {
            value: palette[
                int.from_bytes(sha256(str(value).encode()).digest()[:4], "big") % len(palette)
            ]
            for value in frame[selected.color].unique()
        }
        if selected.color
        else None
    )
    labels = {column: label_for(column) for column in result.columns}
    hover_columns = []
    for field in [selected.x, selected.y] + (
        [selected.color] if selected.color and selected.color != selected.x else []
    ):
        name = "_hover_" + str(len(hover_columns))
        while name in frame:
            name += "_"
        frame[name] = frame[field].map(lambda value: escape(display_value(value)))
        hover_columns.append(name)
    options = dict(
        labels=labels,
        custom_data=hover_columns,
        x=selected.x,
        y=selected.y,
        color=selected.color,
        color_discrete_map=colors,
        color_discrete_sequence=palette,
    )
    if selected.kind == "line":
        frame[selected.x] = pd.to_datetime(frame[selected.x], format="mixed")
        frame = frame.sort_values(selected.x)
        figure = px.line(frame, **options, markers=True)
    else:
        # One trace per category would be offset as a group; relative keeps each bar centered.
        barmode = "relative" if selected.color == selected.x else "group"
        figure = px.bar(frame, **options, barmode=barmode)
    figure.update_layout(
        title=dict(
            text="<br>".join(
                escape(line)
                for line in wrap(f"{labels[selected.y]} por {labels[selected.x].lower()}", 38)
            ),
            font=dict(size=16),
            x=0,
            xanchor="left",
        ),
        xaxis_title=labels[selected.x],
        yaxis_title=labels[selected.y],
        legend_title_text=labels.get(selected.color, ""),
        margin=dict(l=16, r=16, t=80, b=24),
        legend=dict(orientation="h", y=-0.25),
        height=420,
        separators=",.",
    )
    fields = [selected.x, selected.y] + (
        [selected.color] if selected.color and selected.color != selected.x else []
    )
    hover = "<br>".join(
        escape(labels[field]) + ": <b>%{customdata[" + str(index) + "]}</b>"
        for index, field in enumerate(fields)
    )
    figure.update_traces(hovertemplate=hover + "<extra></extra>")
    if selected.kind == "bar" and selected.color == selected.x:
        figure.update_layout(showlegend=False)
    if (frame[selected.y] % 1 == 0).all():
        figure.update_yaxes(tickformat=",d", minexponent=6)
    if selected.kind == "line":
        figure.update_xaxes(tickformat="%m/%Y")
    elif frame[selected.x].nunique() > 8:
        figure.update_xaxes(tickangle=-45, automargin=True)
    return figure

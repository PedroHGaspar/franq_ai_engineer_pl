from html import escape
from textwrap import wrap

import pandas as pd
import plotly.express as px

from src.models import ChartSpec, QueryResult


def select_chart(result: QueryResult, requested: ChartSpec) -> ChartSpec:
    if result.status != "ok" or not result.rows or requested.kind in {"none", "table"}:
        return ChartSpec(kind="table")
    frame = pd.DataFrame(result.rows, columns=result.columns)
    if requested.x not in frame or requested.y not in frame or requested.x == requested.y:
        return ChartSpec(kind="table")
    if requested.color and requested.color not in frame:
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
    palette = ["#646FE6", "#177C80", "#C46B29", "#965BB5", "#4376A8"]
    colors = (
        {
            value: palette[i % len(palette)]
            for i, value in enumerate(sorted(frame[selected.color].unique(), key=str))
        }
        if selected.color
        else None
    )
    options = dict(
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
        figure = px.bar(frame, **options, barmode="group")
    labels = {column: column.replace("_", " ").capitalize() for column in result.columns}
    figure.update_layout(
        title=dict(
            text="<br>".join(
                escape(line)
                for line in wrap(
                    selected.title or f"{labels[selected.y]} por {labels[selected.x].lower()}", 38
                )
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
    if (frame[selected.y] % 1 == 0).all():
        figure.update_yaxes(tickformat=",d", minexponent=6)
    if selected.kind == "line":
        figure.update_xaxes(tickformat="%m/%Y")
    return figure

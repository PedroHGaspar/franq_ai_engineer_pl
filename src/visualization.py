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
    if selected.kind == "line":
        frame[selected.x] = pd.to_datetime(frame[selected.x], format="mixed")
        frame = frame.sort_values(selected.x)
        figure = px.line(frame, x=selected.x, y=selected.y, color=selected.color, markers=True)
    else:
        figure = px.bar(frame, x=selected.x, y=selected.y, color=selected.color, barmode="group")
    labels = {column: column.replace("_", " ").capitalize() for column in result.columns}
    figure.update_layout(
        title=selected.title or f"{labels[selected.y]} por {labels[selected.x].lower()}",
        xaxis_title=labels[selected.x],
        yaxis_title=labels[selected.y],
        legend_title_text=labels.get(selected.color, ""),
        margin=dict(l=16, r=16, t=60, b=24),
        legend=dict(orientation="h", y=-0.25),
        height=420,
    )
    return figure

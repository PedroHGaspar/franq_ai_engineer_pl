from datetime import datetime
from math import isfinite
from typing import Literal

from pydantic import BaseModel, Field
from sqlglot import exp, parse_one
from sqlglot.errors import ParseError

from src.intent import PERIOD
from src.models import ChartSpec, QueryResult


def label_for(alias: str) -> str:
    labels = {
        "compras_media_por_cliente": "Média de compras por cliente",
        "n": "Quantidade",
    }
    accents = {
        "mes": "mês",
        "media": "média",
        "medias": "médias",
        "nao": "não",
        "reclamacoes": "reclamações",
        "reclamacao": "reclamação",
        "unicos": "únicos",
        "unicas": "únicas",
        "interacoes": "interações",
        "categoria": "categoria",
    }
    if alias in labels:
        return labels[alias]
    words = alias.replace("_", " ").split()
    return " ".join(accents.get(word.lower(), word) for word in words).capitalize()


def display_value(value) -> str:
    if value is None:
        return "Não informado"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not isfinite(value):
            return "Não disponível"
        if isinstance(value, int) or value.is_integer():
            text = f"{value:,d}" if isinstance(value, int) else f"{value:,.0f}"
        elif abs(value) < 0.0001:
            text = str(value)
        else:
            text = f"{value:,.4f}".rstrip("0").rstrip(".")
        return text.translate(str.maketrans({",": ".", ".": ","}))
    if isinstance(value, str):
        try:
            if len(value) == 7:
                return datetime.strptime(value, "%Y-%m").strftime("%m/%Y")
            if len(value) == 10:
                return datetime.strptime(value, "%Y-%m-%d").strftime("%d/%m/%Y")
        except ValueError:
            pass
    return str(value)


class Presentation(BaseModel):
    kind: Literal["metric", "ranking", "table", "bar", "line", "text"]
    result_id: str | None = None
    title: str
    summary: str
    items: list[str] = Field(default_factory=list)
    value: str | None = None
    source: Literal["deterministic"] = "deterministic"
    narrative_verified: Literal[False] = False


def is_ranking(result: QueryResult) -> bool:
    if not 2 <= len(result.rows) <= 20 or len(result.columns) != 2:
        return False
    if not all(
        isinstance(row[0], str)
        and isinstance(row[1], (int, float))
        and not isinstance(row[1], bool)
        and isfinite(row[1])
        for row in result.rows
    ):
        return False
    try:
        query = parse_one(result.sql, read="sqlite")
        order, limit = query.args.get("order"), query.args.get("limit")
        if not order or not limit or not 1 <= int(limit.expression.this) <= 20:
            return False
        first = order.expressions[0]
        metric = first.this
        matches = (isinstance(metric, exp.Column) and metric.name == result.columns[1]) or (
            isinstance(metric, exp.Literal) and metric.this == "2"
        )
        return bool(
            matches
            and first.args.get("desc")
            and all(a[1] >= b[1] for a, b in zip(result.rows, result.rows[1:], strict=False))
        )
    except (ParseError, ValueError, AttributeError):
        return False


def numeric(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)


def trend_items(result: QueryResult) -> list[str]:
    """First, last and peak value of each series when the rows form a period series."""
    columns, rows = result.columns, result.rows
    period = next(
        (
            i
            for i in range(len(columns))
            if all(isinstance(row[i], str) and PERIOD.match(row[i]) for row in rows)
        ),
        None,
    )
    measure = next(
        (i for i in reversed(range(len(columns))) if all(numeric(row[i]) for row in rows)), None
    )
    others = [i for i in range(len(columns)) if i not in {period, measure}]
    if period is None or measure is None or len(others) > 1:
        return []
    if len({row[period] for row in rows}) < 2:
        return []
    series: dict[str, list] = {}
    for row in sorted(rows, key=lambda row: row[period]):
        series.setdefault(str(row[others[0]]) if others else "Total", []).append(row)
    items = []
    for name, points in series.items():
        first, last = points[0], points[-1]
        peak = max(points, key=lambda row: row[measure])
        items.append(
            f"{name}: {display_value(first[measure])} em {display_value(first[period])} → "
            f"{display_value(last[measure])} em {display_value(last[period])}; "
            f"pico de {display_value(peak[measure])} em {display_value(peak[period])}"
        )
    return items


def prepare_presentation(
    answer: dict,
    results: list[dict],
    primary_id: str | None = None,
    coverage: tuple[str, str] | None = None,
) -> Presentation:
    evidence = {r["id"]: r for r in results if r["id"] in answer["evidence_ids"]}
    spec = ChartSpec.model_validate(answer["chart"])
    source = (
        primary_id
        or answer.get("primary_result_id")
        or spec.result_id
        or next(iter(answer["evidence_ids"]), None)
    )
    if not evidence:
        return Presentation(
            kind="text",
            title="Dados insuficientes",
            summary="As evidências disponíveis não permitem concluir esta análise.",
        )
    if source not in evidence:
        raise ValueError("A apresentação exige uma evidência citada.")
    result = QueryResult.model_validate(evidence[source])
    base = dict(result_id=source, title="Resultados da consulta")
    if not result.rows:
        summary = (
            "A consulta foi executada sem erros, mas não há registros para o período e os "
            "filtros consultados. Isso indica ausência de registros, não falha da análise."
        )
        if coverage:
            start, end = (display_value(value) for value in coverage)
            summary += f" Registros disponíveis no banco: {start} a {end}."
        return Presentation(
            result_id=source, kind="table", title="Nenhum registro encontrado", summary=summary
        )
    if result.status == "truncated":
        return Presentation(
            **base,
            kind="table",
            summary="Resultado parcial: não permite concluir totais ou rankings.",
        )
    if len(result.columns) == 1 and len(result.rows) == 1 and numeric(result.rows[0][0]):
        return Presentation(
            kind="metric",
            result_id=source,
            title=label_for(result.columns[0]),
            summary="Resultado para os filtros consultados.",
            value=display_value(result.rows[0][0]),
        )
    items = trend_items(result)
    if items:
        periods = [
            value
            for row in result.rows
            for value in row
            if isinstance(value, str) and PERIOD.match(value)
        ]
        return Presentation(
            result_id=source,
            kind="line" if spec.kind == "line" else "table",
            title="Evolução por período",
            summary=(
                f"Série de {display_value(min(periods))} a {display_value(max(periods))}, com "
                "primeiro e último valor de cada série e seu pico. Combinações ausentes não "
                "foram preenchidas com zero; períodos nas bordas do filtro podem estar incompletos."
            ),
            items=items,
        )
    if is_ranking(result):
        return Presentation(
            kind="ranking",
            result_id=source,
            title=f"{label_for(result.columns[1])} por {label_for(result.columns[0]).lower()}",
            summary="Ordem decrescente da medida; empates preservam a ordem retornada pelo SQL.",
            items=[f"{row[0]} — {display_value(row[1])}" for row in result.rows],
        )
    if spec.kind == "line" and spec.x in result.columns:
        index = result.columns.index(spec.x)
        dates = [str(row[index]) for row in result.rows]
        try:
            parsed = [
                datetime.fromisoformat(date + "-01" if len(date) == 7 else date) for date in dates
            ]
            summary = (
                f"{len(result.rows)} pontos retornados entre {min(parsed):%d/%m/%Y} "
                f"e {max(parsed):%d/%m/%Y}. Datas referem-se aos pontos da série. "
                "Combinações ausentes não foram preenchidas com zero; "
                "ausência de linha não comprova ausência de atividade."
            )
            return Presentation(**base, kind="line", summary=summary)
        except (ValueError, TypeError):
            pass
    kind = "bar" if spec.kind == "bar" else "table"
    if (
        2 <= len(result.rows) <= 20
        and len(result.columns) == 2
        and all(isinstance(row[0], str) and numeric(row[1]) for row in result.rows)
    ):
        ordered = sorted(result.rows, key=lambda row: -row[1])
        return Presentation(
            result_id=source,
            title=f"{label_for(result.columns[1])} por {label_for(result.columns[0]).lower()}",
            kind=kind,
            summary=(
                f"Maior valor: {ordered[0][0]} ({display_value(ordered[0][1])}); "
                f"menor: {ordered[-1][0]} ({display_value(ordered[-1][1])})."
            ),
            items=[f"{row[0]} — {display_value(row[1])}" for row in ordered],
        )
    count = len(result.rows)
    found = "resultado encontrado" if count == 1 else "resultados encontrados"
    return Presentation(
        **base,
        kind=kind,
        summary=f"{count} {found} para os filtros consultados.",
    )

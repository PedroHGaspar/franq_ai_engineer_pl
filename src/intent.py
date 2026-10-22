"""Deterministic checks between the question intent and the evidence that answers it."""

import re
import unicodedata
from calendar import monthrange
from dataclasses import dataclass, field
from datetime import date, timedelta

from sqlglot import exp, parse_one
from sqlglot.errors import ParseError

MONTHS = (
    "janeiro fevereiro marco abril maio junho julho agosto setembro outubro novembro dezembro"
).split()
MONTH_LABELS = (
    "janeiro fevereiro março abril maio junho julho agosto setembro outubro novembro dezembro"
).split()
YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
PERIOD = re.compile(r"^\d{4}(?:-\d{2}){0,2}(?:[ T].*)?$")
RELATIVE = re.compile(r"\bultim[oa] (ano|semestre|trimestre|mes)\b")
TREND = re.compile(
    r"\b(tendencias?|evolucao|ao longo|historico|mes a mes|mensa(?:l|is)|por mes|semana(?:l|is)|"
    r"por semana|diari[oa]s?|por dia|anua(?:l|is)|por ano|serie temporal|sazonalidade)\b"
)
AGGREGATE = re.compile(
    r"\b(quant[oa]s|numero|quantidade|total|soma|medi[ao]s?|percentua(?:l|is)|porcentagem|"
    r"taxa|proporcao)\b"
)
SUPERLATIVE = re.compile(
    r"\b(maior(?:es)?|menor(?:es)?|top|ranking|principais)\b|\bmais (?!de\b|que\b|do que\b)\w+"
)
METADATA = re.compile(
    r"\b(tabelas|colunas|schema|esquema)\b|\b(?:quais|que) (?:\w+ ){0,2}"
    r"(?:existem|existentes|disponiveis|possiveis)\b|\bperiodo (?:coberto|disponivel)\b"
)
NOT_DIMENSIONS = {"mes", "semana", "dia", "ano", "cento", "exemplo", "favor", "isso", "que"}
TEMPORAL_NAME = re.compile(
    r"(?<![a-z])(data|date|mes|month|dia|day|ano|year|periodo|semana|week)(?![a-z])"
)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(char for char in text if not unicodedata.combining(char))


def singular_text(text: str) -> str:
    """Portuguese plural to singular per word, enough to match 'reclamacoes' to 'reclamacao'."""
    words = []
    for word in re.findall(r"\w+", text):
        if word.endswith(("oes", "aes")):
            word = word[:-3] + "ao"
        elif word.endswith("s") and len(word) > 3:
            word = word[:-1]
        words.append(word)
    return " ".join(words)


@dataclass
class Intent:
    trend: bool = False
    grouped: bool = False
    ranking: bool = False
    aggregate: bool = False
    metadata: bool = False
    years: set[str] = field(default_factory=set)
    months: set[int] = field(default_factory=set)


def infer_intent(question: str, clarifications: list[str] | None = None) -> Intent:
    text = normalize(question)
    context = " ".join([text, *map(normalize, clarifications or [])])
    dimensions = [
        word
        for word in re.findall(r"(?<!media )(?<!medio )\bpor (?:cada )?(\w+)", text)
        if word not in NOT_DIMENSIONS
    ]
    trend = bool(TREND.search(text))
    grouped = bool(dimensions or re.search(r"\b(?:de|em|para) cada\b", text))
    return Intent(
        trend=trend,
        grouped=grouped,
        ranking=bool(SUPERLATIVE.search(text)) and not re.match(r"\s*quant[oa]s\b", text),
        aggregate=bool(AGGREGATE.search(text)) or trend or grouped,
        metadata=bool(METADATA.search(text)),
        years=set(YEAR.findall(context)),
        months={i + 1 for i, name in enumerate(MONTHS) if re.search(rf"\b{name}\b", context)},
    )


def resolve_period(question: str, coverage: tuple[str, str] | None) -> str | None:
    """Resolves relative periods against the data: the most recent records, not today."""
    if not coverage:
        return None
    text = normalize(question)
    start, end = (date.fromisoformat(value[:10]) for value in coverage)
    months = [i for i, name in enumerate(MONTHS) if re.search(rf"\b{name}\b", text)]
    if months and not YEAR.search(text):
        month, label = months[0] + 1, MONTH_LABELS[months[0]]
        year = next(
            (
                year
                for year in range(end.year, start.year - 1, -1)
                if date(year, month, 1) <= end
                and date(year + month // 12, month % 12 + 1, 1) > start
            ),
            None,
        )
        if year:
            return f"{label} de {year} (o {label} mais recente com registros no banco)"
        return None
    relative = RELATIVE.search(text)
    if relative:
        unit = relative.group(1)
        count = {"ano": 12, "semestre": 6, "trimestre": 3, "mes": 1}[unit]
        year, month = divmod(end.year * 12 + end.month - 1 - count, 12)
        day = min(end.day, monthrange(year, month + 1)[1])
        since = date(year, month + 1, day) + timedelta(days=1)
        return (
            f"{since:%d/%m/%Y} a {end:%d/%m/%Y} ({count} meses até o registro mais recente "
            "do banco)"
        )
    return None


def _query(sql: str):
    try:
        return parse_one(sql, read="sqlite")
    except (ParseError, ValueError):
        return None


def _group_terms(select: exp.Select) -> list[str]:
    # GROUP BY 1 or GROUP BY alias refers to the projection, which carries the expression.
    projections = select.expressions
    aliases = {item.alias_or_name: item for item in projections}
    terms = []
    for item in select.args["group"].expressions:
        if isinstance(item, exp.Literal) and item.is_int and 0 < int(item.this) <= len(projections):
            item = projections[int(item.this) - 1]
        elif isinstance(item, exp.Column) and not item.table and item.name in aliases:
            item = aliases[item.name]
        terms.append(normalize(item.sql()))
    return terms


def shape_problems(intent: Intent, result: dict) -> list[str]:
    """Explains why a result cannot answer the question; empty when it is sufficient."""
    if intent.metadata:
        return []
    tree = _query(result["sql"])
    if tree is None:
        return []
    selects = list(tree.find_all(exp.Select))
    grouped = [select for select in selects if select.args.get("group")]
    problems = []
    if intent.aggregate and not grouped and not any(tree.find_all(exp.AggFunc)):
        problems.append(
            "a pergunta pede uma medida agregada, mas a consulta apenas lista valores "
            "(consulta de descoberta)"
        )
    if (intent.grouped or intent.trend) and not grouped:
        problems.append("a pergunta pede resultados por grupo ou período; a consulta não agrupa")
    if intent.ranking and not tree.args.get("order"):
        problems.append("a pergunta pede um ranking; ordene o resultado pela medida")
    if intent.trend and grouped:
        group = _group_terms(grouped[0])
        rows = result["rows"]
        # Returned values identify the period column; names are only a fallback for empty results.
        periods = [
            {row[i] for row in rows}
            for i in range(len(result["columns"]))
            if rows and all(isinstance(row[i], str) and PERIOD.match(row[i]) for row in rows)
        ]
        if (rows and not periods) or (
            not rows and not any(TEMPORAL_NAME.search(term) for term in group)
        ):
            problems.append("a tendência exige agrupar por período (ex.: mês)")
        elif rows and not any(len(values) >= 2 for values in periods):
            problems.append("a tendência exige ao menos dois períodos no resultado")
        if intent.grouped and len(group) < 2:
            problems.append("a tendência por dimensão exige agrupar por período e pela dimensão")
    sql = result["sql"]
    for year in sorted(intent.years):
        if year not in sql and f"{int(year) + 1}-01-01" not in sql:
            problems.append(f"o ano {year} solicitado não aparece nos filtros da consulta")
    for month in sorted(intent.months):
        if (
            intent.years
            and not any(f"{year}-{month:02d}" in sql for year in intent.years)
            and f"'{month:02d}'" not in sql
        ):
            problems.append(f"o mês {MONTH_LABELS[month - 1]} não aparece nos filtros da consulta")
    return problems


def select_primary(
    question: str,
    clarifications: list[str],
    answer: dict,
    results: dict[str, dict],
) -> tuple[str | None, list[str]]:
    """Chooses the cited result that answers the question, or explains why none does."""
    intent = infer_intent(question, clarifications)
    explicit = answer.get("primary_result_id") or answer["chart"].get("result_id")
    candidates = [explicit] if explicit else list(reversed(answer["evidence_ids"]))
    problems = []
    for key in candidates:
        problems = shape_problems(intent, results[key])
        if not problems:
            return key, []
    return None, problems

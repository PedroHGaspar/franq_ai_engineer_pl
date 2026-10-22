from typing import Any, Literal, TypedDict

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


class QueryResult(BaseModel):
    id: str
    sql: str
    columns: list[str]
    rows: list[list[Any]]
    status: Literal["ok", "empty", "truncated"]
    duration_ms: float


class ChartSpec(BaseModel):
    kind: Literal["table", "bar", "line", "none"] = "table"
    result_id: str | None = Field(
        None, description="ID da evidência que fornece os dados do gráfico"
    )
    title: str | None = Field(None, max_length=160)
    x: str | None = None
    y: str | None = None
    color: str | None = None


class FinalAnswer(BaseModel):
    answerable: bool = Field(
        True, description="Falso quando o schema não contém os dados necessários"
    )
    summary: str = Field(description="Conclusão em português, fundamentada nas evidências")
    evidence_ids: list[str] = Field(description="IDs das consultas que sustentam a resposta")
    primary_result_id: str | None = Field(
        None, description="ID da consulta que responde diretamente à pergunta, não de descoberta"
    )
    assumptions: list[str] = Field(description="Interpretações, período e denominadores adotados")
    chart: ChartSpec


class AgentState(TypedDict, total=False):
    run_id: str
    question: str
    messages: list[BaseMessage]
    pending: dict[str, Any]
    results: list[dict[str, Any]]
    events: list[dict[str, Any]]
    failed_sql: list[str]
    model_calls: int
    iterations: int
    queries: int
    corrections: int
    error: dict[str, Any] | None
    status: str
    answer: dict[str, Any] | None
    presentation: dict[str, Any]
    primary_id: str | None
    clarification: str | None
    clarifications: list[str]
    period: str | None
    coverage: tuple[str, str] | None
    usage: dict[str, Any]

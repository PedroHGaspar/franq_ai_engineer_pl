from pathlib import Path
from uuid import uuid4

import pandas as pd
import streamlit as st
from PIL import Image

from src.agent import DataAgent
from src.config import Settings
from src.database import DatabaseError
from src.models import ChartSpec, QueryResult
from src.presentation import display_value, label_for, prepare_presentation
from src.visualization import build_chart

st.set_page_config(page_title="Franq · Assistente de Dados", page_icon="📊", layout="centered")
st.html(
    "<style>.block-container {padding-top: 4rem; padding-bottom: 6rem;}"
    "[data-testid='stStatusWidget'] {visibility: hidden;}</style>"
)
banner = Path(__file__).resolve().parent / "assets" / "franq-banner.png"
if banner.is_file():
    st.image(str(banner), width="stretch")
    # The logo area of the banner stays visible in the sidebar while the conversation scrolls.
    st.logo(Image.open(banner).crop((1980, 30, 2360, 160)), size="large")
st.caption("Desafio técnico — demonstração não oficial")
st.title("Assistente de Dados")
st.write("Transforme perguntas de negócio em respostas com evidências.")


def render_transparency(state: dict) -> None:
    with st.expander("Transparência operacional"):
        st.caption(f"Execução: {state['run_id']}")
        st.write(
            f"Chamadas ao modelo: {state['model_calls']} · "
            f"Tentativas SQL: {state['queries']} · Correções: {state['corrections']}"
        )
        duration = sum(event.get("duration_ms", 0) for event in state["events"])
        tools = sorted({event["tool"] for event in state["events"] if event.get("tool")})
        st.caption(
            f"Consultas concluídas: {len(state['results'])} · "
            f"Tempo das operações: {duration / 1000:.2f} s"
        )
        st.caption("Ferramentas: " + (", ".join(tools) or "Nenhuma"))
        labels = {
            "metadata": "Descoberta do banco",
            "validate": "Validação SQL",
            "execute": "Execução SQL",
            "evaluate": "Avaliação",
            "decide": "Seleção de ferramenta",
            "model": "Modelo",
            "finish": "Conclusão",
            "clarify": "Esclarecimento",
        }
        steps = []
        for event in state["events"]:
            steps.append(
                {
                    "Etapa": labels.get(event["stage"], event["stage"]),
                    "Ferramenta": event.get("tool", "—"),
                    "Duração (ms)": event.get("duration_ms"),
                    "Estado / erro": event.get("error_code") or event.get("status", "—"),
                }
            )
        if steps:
            st.dataframe(pd.DataFrame(steps), hide_index=True, width="stretch")
        for event in state["events"]:
            if event.get("metadata") is not None:
                with st.expander(f"Metadados · {event.get('tool')}"):
                    st.json(event["metadata"], expanded=False)
            if event.get("message"):
                st.caption(f"Ocorrência: {event['message']}")
        executed = {r["sql"] for r in state["results"]}
        shown = set()
        for event in state["events"]:
            sql = event.get("sql")
            if sql and sql not in executed and sql not in shown:
                st.caption("SQL gerado · antes da normalização ou sem resultado concluído")
                st.code(sql, language="sql", wrap_lines=True)
                shown.add(sql)
        for result in state["results"]:
            with st.expander(
                f"{result['id']} · SQL executado · {result['status']} · "
                f"{result['duration_ms']:.1f} ms"
            ):
                st.code(result["sql"], language="sql", wrap_lines=True)
                st.dataframe(
                    pd.DataFrame(result["rows"], columns=result["columns"]),
                    hide_index=True,
                    width="stretch",
                )
        if state.get("answer"):
            with st.expander("Explicação original do modelo"):
                st.caption(
                    "Números conferidos com as consultas citadas; afirmações qualitativas "
                    "não são verificadas automaticamente."
                )
                st.text(state["answer"]["summary"])
        if state.get("usage"):
            with st.expander("Consumo registrado"):
                st.json(state["usage"], expanded=False)


def render_result(state: dict, entry_id: str) -> None:
    if state["status"] == "clarification":
        st.info(state["clarification"])
        st.caption("Responda no campo abaixo para continuar esta investigação.")
    elif state.get("answer"):
        answer = state["answer"]
        presentation = prepare_presentation(
            answer, state["results"], state.get("primary_id"), state.get("coverage")
        )
        st.subheader(presentation.title)
        st.write(presentation.summary)
        if state.get("period"):
            st.caption(f"Período considerado: {state['period']}")
        if state.get("clarifications"):
            st.caption("Esclarecimento considerado: " + " · ".join(state["clarifications"]))
        for index, item in enumerate(presentation.items, 1):
            st.text(f"{index}. {item}" if presentation.kind == "ranking" else f"• {item}")
        if answer["assumptions"]:
            with st.expander("Interpretação adotada"):
                for assumption in answer["assumptions"]:
                    st.write(assumption)
        results = {r["id"]: r for r in state["results"]}
        spec = ChartSpec.model_validate(answer["chart"])
        source = presentation.result_id
        for key in answer["evidence_ids"]:
            result = QueryResult.model_validate(results[key])
            if key != source:
                with st.expander(f"Evidência complementar · {key}"):
                    st.dataframe(
                        pd.DataFrame(result.rows, columns=result.columns).style.format(
                            display_value
                        ),
                        column_config={name: label_for(name) for name in result.columns},
                        hide_index=True,
                        width="stretch",
                    )
                continue
            if result.status == "empty":
                continue
            if presentation.kind == "metric":
                st.metric(presentation.title, presentation.value)
                continue
            if spec.result_id in {None, key}:
                chart = build_chart(result, spec)
                if chart is not None:
                    st.plotly_chart(chart, width="stretch", key=f"chart-{entry_id}-{key}")
                elif spec.kind in {"bar", "line"}:
                    st.info(
                        "Os dados não são compatíveis com o gráfico solicitado. Confira a tabela."
                    )
            st.dataframe(
                pd.DataFrame(result.rows, columns=result.columns).style.format(display_value),
                column_config={name: label_for(name) for name in result.columns},
                hide_index=True,
                width="stretch",
            )
    else:
        error = state.get("error") or {}
        # Service and configuration messages help the user; technical details stay in the
        # transparency panel below.
        if error.get("code") in {None, "llm_unavailable", "llm_configuration", "budget_or_context"}:
            message = error.get("message", "Falha não identificada.")
        else:
            message = (
                "Não foi possível chegar a uma resposta confiável. Tente reformular a pergunta "
                "ou dividi-la em perguntas menores."
            )
        st.error(f"A análise não foi concluída. {message}")
        if state.get("results"):
            st.caption(
                "As consultas intermediárias estão na transparência operacional "
                "e não constituem resposta."
            )
    render_transparency(state)


def failed_state(message: str) -> dict:
    return {
        "run_id": str(uuid4()),
        "status": "failed",
        "error": {"message": message},
        "events": [],
        "results": [],
        "model_calls": 0,
        "queries": 0,
        "corrections": 0,
    }


def context_for(history: list[dict]) -> list[dict]:
    """Prior questions with their deterministic summaries; never rows, SQL or result IDs."""
    turns = []
    for entry in history:
        state = entry["state"] or {}
        if state.get("status") == "completed" and state.get("presentation"):
            view = state["presentation"]
            answer = view["title"] + ": " + (view.get("value") or view["summary"])
            turns.append({"question": entry["question"], "answer": answer})
    return turns


try:
    settings = Settings()
except Exception:
    st.error("Configuração inválida. Confira os valores do arquivo .env.")
    st.stop()

history = st.session_state.setdefault("history", [])
processing = bool(history) and history[-1]["state"] is None

with st.sidebar:
    st.subheader("Sua conversa")
    if not history:
        st.caption("Nenhuma pergunta ainda.")
    for number, entry in enumerate(history, 1):
        question_text = entry["question"]
        short = question_text if len(question_text) <= 70 else question_text[:67] + "…"
        st.caption(f"{number}. {short}")
    if st.button("Limpar histórico", disabled=processing or not history):
        st.session_state.pop("history", None)
        st.session_state.pop("agent", None)
        st.session_state.pop("pending_thread", None)
        st.rerun()
    st.caption("Especifique anos e períodos. Nomes e e-mails não são necessários às análises.")
    with st.expander("Sobre a sessão"):
        st.caption(f"Modelo: {settings.openai_model}")
        st.caption(f"Teto local configurado: US$ {settings.api_budget_usd:.2f}")
        st.caption("O histórico fica disponível apenas nesta sessão.")

if not settings.enable_paid_api:
    st.info(
        "Chamadas à OpenAI desabilitadas. Defina OPENAI_API_KEY e ENABLE_PAID_API=true no .env."
    )

question = st.chat_input(
    "Pergunte sobre os dados ou responda ao esclarecimento pendente",
    disabled=not settings.enable_paid_api or processing,
    max_chars=4000,
)
if question and not processing:
    # Each submission gets its own entry and ID before processing. The rerun renders the whole
    # conversation in a stable order, so no element of the previous run is shown as current.
    history.append(
        {
            "id": str(uuid4()),
            "question": question,
            "state": None,
            "thread": st.session_state.get("pending_thread"),
        }
    )
    st.rerun()

if not history:
    with st.container():
        st.subheader("Comece com uma pergunta")
        st.caption(
            "Clientes, compras, campanhas e atendimento. Inclua o período e o formato desejado."
        )
        with st.container(border=True):
            st.markdown(
                "**Sugestões para experimentar**\n\n"
                "- Quantos clientes interagiram com campanhas de WhatsApp em 2024?\n"
                "- Mostre um gráfico de barras com as reclamações não resolvidas por canal.\n"
                "- Mostre a evolução mensal das reclamações por canal de julho de 2024 "
                "a julho de 2025."
            )

for index, entry in enumerate(history):
    with st.chat_message("user"):
        st.write(entry["question"])
    with st.chat_message("assistant"):
        if entry["state"] is None:
            thread = entry["thread"] or str(uuid4())
            try:
                if "agent" not in st.session_state:
                    st.session_state.agent = DataAgent(settings)
                agent = st.session_state.agent
                with st.spinner("Investigando os dados…"):
                    if entry["thread"]:
                        state = agent.resume(thread, entry["question"])
                    else:
                        state = agent.run(entry["question"], thread, context_for(history[:index]))
            except Exception as exc:
                message = (
                    str(exc)
                    if isinstance(exc, DatabaseError)
                    # Only the exception type: messages may carry provider details or SQL.
                    else "Erro interno inesperado "
                    f"({type(exc).__name__}). Se o código foi atualizado, reinicie o servidor."
                )
                state = failed_state(message)
            entry["state"] = state
            st.session_state.pending_thread = thread if state["status"] == "clarification" else None
        render_result(entry["state"], entry["id"])

if processing:
    # The input was disabled while the agent ran; render again to enable it.
    st.rerun()

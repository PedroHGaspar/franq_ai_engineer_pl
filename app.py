from uuid import uuid4

import pandas as pd
import streamlit as st

from src.agent import DataAgent
from src.config import Settings
from src.database import DatabaseError
from src.models import ChartSpec, QueryResult
from src.visualization import build_chart

st.set_page_config(page_title="Franq · Assistente de Dados", page_icon="📊", layout="wide")
st.title("Assistente de Dados")
st.caption("Investigue perguntas de negócio com consultas rastreáveis ao SQLite.")


def render_transparency(state: dict) -> None:
    with st.expander("Transparência operacional"):
        st.caption(f"Execução: {state['run_id']}")
        st.write(
            f"Chamadas ao modelo: {state['model_calls']} · "
            f"Tentativas SQL: {state['queries']} · Correções: {state['corrections']}"
        )
        labels = {
            "metadata": "Descoberta do banco",
            "validate": "Validação SQL",
            "execute": "Execução SQL",
            "evaluate": "Avaliação",
            "decide": "Seleção de ferramenta",
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
                st.code(sql, language="sql")
                shown.add(sql)
        for result in state["results"]:
            with st.expander(
                f"{result['id']} · SQL executado · {result['status']} · "
                f"{result['duration_ms']:.1f} ms"
            ):
                st.code(result["sql"], language="sql")
                st.dataframe(
                    pd.DataFrame(result["rows"], columns=result["columns"]),
                    hide_index=True,
                    width="stretch",
                )
        if state.get("usage"):
            with st.expander("Consumo registrado"):
                st.json(state["usage"], expanded=False)


def render_result(state: dict) -> None:
    if state["status"] == "clarification":
        st.info(state["clarification"])
        st.caption("Responda no campo abaixo para continuar esta investigação.")
    elif state.get("answer"):
        answer = state["answer"]
        st.write(answer["summary"])
        for assumption in answer["assumptions"]:
            st.caption(assumption)
        results = {r["id"]: r for r in state["results"]}
        spec = ChartSpec.model_validate(answer["chart"])
        source = spec.result_id or next(iter(answer["evidence_ids"]), None)
        for key in answer["evidence_ids"]:
            result = QueryResult.model_validate(results[key])
            if result.status == "empty":
                st.info("Nenhum registro encontrado para os filtros consultados.")
                continue
            if key == source:
                chart = build_chart(result, spec)
                if chart is not None:
                    st.plotly_chart(chart, width="stretch", key=f"chart-{state['run_id']}-{key}")
                elif spec.kind in {"bar", "line"}:
                    st.info(
                        "Os dados não são compatíveis com o gráfico solicitado. Confira a tabela."
                    )
            st.dataframe(
                pd.DataFrame(result.rows, columns=result.columns),
                hide_index=True,
                width="stretch",
            )
    else:
        st.error(
            (state.get("error") or {}).get("message", "Não foi possível concluir a investigação.")
        )
    render_transparency(state)


try:
    settings = Settings()
except Exception:
    st.error("Configuração inválida. Confira os valores do arquivo .env.")
    st.stop()

with st.sidebar:
    st.subheader("Sessão")
    st.caption(f"Modelo: {settings.openai_model}")
    st.caption(f"Teto local configurado: US$ {settings.api_budget_usd:.2f}")
    if st.button("Limpar histórico"):
        st.session_state.pop("history", None)
        st.session_state.pop("agent", None)
        st.session_state.pop("pending_thread", None)
        st.rerun()
    st.caption("Especifique anos e períodos. Nomes e e-mails não são necessários às análises.")

if not settings.enable_paid_api:
    st.info(
        "Chamadas à OpenAI desabilitadas. A habilitação depende de autorização para testes reais."
    )

history = st.session_state.setdefault("history", [])
if not history:
    st.subheader("O que você gostaria de investigar?")
    st.write("Pergunte sobre clientes, compras, campanhas ou atendimento.")
    st.caption("Inclua o período desejado. Você também pode pedir uma tabela ou um gráfico.")
for entry in history:
    with st.chat_message("user"):
        st.write(entry["question"])
    with st.chat_message("assistant"):
        render_result(entry["state"])

question = st.chat_input(
    "Pergunte sobre os dados ou responda ao esclarecimento pendente",
    disabled=not settings.enable_paid_api or st.session_state.get("busy", False),
    max_chars=4000,
)
if question and not st.session_state.get("busy", False):
    st.session_state.busy = True
    with st.chat_message("user"):
        st.write(question)
    try:
        if "agent" not in st.session_state:
            st.session_state.agent = DataAgent(settings)
        agent = st.session_state.agent
        thread = st.session_state.get("pending_thread") or str(uuid4())
        with st.spinner("Investigando os dados…"):
            if st.session_state.get("pending_thread"):
                state = agent.resume(thread, question)
            else:
                state = agent.run(question, thread)
        st.session_state.pending_thread = thread if state["status"] == "clarification" else None
        history.append({"question": question, "state": state})
        with st.chat_message("assistant"):
            render_result(state)
    except Exception as exc:
        message = (
            str(exc)
            if isinstance(exc, DatabaseError)
            else ("Não foi possível concluir. Verifique a configuração e tente novamente.")
        )
        state = {
            "run_id": str(uuid4()),
            "status": "failed",
            "error": {"message": message},
            "events": [],
            "results": [],
            "model_calls": 0,
            "queries": 0,
            "corrections": 0,
        }
        history.append({"question": question, "state": state})
        with st.chat_message("assistant"):
            render_result(state)
    finally:
        st.session_state.busy = False

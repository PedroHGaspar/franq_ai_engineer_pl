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


def render_result(state: dict) -> None:
    if state["status"] == "clarification":
        st.info(state["clarification"])
    elif state.get("answer"):
        answer = state["answer"]
        st.write(answer["summary"])
        for assumption in answer["assumptions"]:
            st.caption(assumption)
        results = {r["id"]: r for r in state["results"]}
        for index, key in enumerate(answer["evidence_ids"]):
            result = QueryResult.model_validate(results[key])
            if result.status == "empty":
                st.info("Nenhum registro encontrado para os filtros consultados.")
            else:
                st.dataframe(
                    pd.DataFrame(result.rows, columns=result.columns),
                    hide_index=True,
                    width="stretch",
                )
                if index == 0:
                    chart = build_chart(result, ChartSpec.model_validate(answer["chart"]))
                    if chart is not None:
                        st.plotly_chart(
                            chart, width="stretch", key=f"chart-{state['run_id']}-{key}"
                        )
    else:
        st.error(state.get("error", {}).get("message", "Não foi possível concluir a investigação."))
    with st.expander("Transparência operacional"):
        st.caption(f"Execução: {state['run_id']}")
        st.write(
            f"Chamadas ao modelo: {state['model_calls']} · "
            f"Tentativas SQL: {state['queries']} · Correções: {state['corrections']}"
        )
        if state.get("usage"):
            st.json(state["usage"])
        for event in state["events"]:
            st.write({key: value for key, value in event.items() if key != "sql"})
            if "sql" in event:
                st.code(event["sql"], language="sql")
        for result in state["results"]:
            st.caption(f"{result['id']} · {result['status']} · {result['duration_ms']} ms")
            st.dataframe(
                pd.DataFrame(result["rows"], columns=result["columns"]),
                hide_index=True,
                width="stretch",
            )


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
for entry in history:
    with st.chat_message("user"):
        st.write(entry["question"])
    with st.chat_message("assistant"):
        render_result(entry["state"])

question = st.chat_input(
    "Pergunte sobre os dados ou responda ao esclarecimento pendente",
    disabled=not settings.enable_paid_api,
    max_chars=4000,
)
if question:
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
    except (DatabaseError, ValueError) as exc:
        st.error(str(exc))
    except Exception:
        st.error("Falha inesperada na execução. Consulte a configuração e os logs locais.")

from types import SimpleNamespace

from pydantic import SecretStr

from src.agent import DataAgent
from tests.conftest import ScriptedModel


def test_budget_blocks_model_before_request(settings, monkeypatch):
    settings.enable_paid_api = True
    settings.openai_api_key = SecretStr("test-placeholder")
    settings.api_budget_usd = 0.001
    model = ScriptedModel(("list_tables", {}))
    monkeypatch.setattr("src.agent.ChatOpenAI", lambda **kwargs: model)
    monkeypatch.setattr("tiktoken.get_encoding", lambda name: SimpleNamespace(encode=lambda s: [1]))
    state = DataAgent(settings).run("Investigue")
    assert not model.messages
    assert state["error"]["code"] == "budget_or_context"
    assert "API_BUDGET_USD" in state["error"]["message"]


def test_context_limit_blocks_model(settings, monkeypatch):
    settings.enable_paid_api = True
    settings.openai_api_key = SecretStr("test-placeholder")
    model = ScriptedModel(("list_tables", {}))
    monkeypatch.setattr("src.agent.ChatOpenAI", lambda **kwargs: model)
    monkeypatch.setattr(
        "tiktoken.get_encoding",
        lambda name: SimpleNamespace(encode=lambda s: [1] * settings.max_input_tokens),
    )
    state = DataAgent(settings).run("Investigue")
    assert not model.messages
    assert state["error"]["code"] == "budget_or_context"
    assert "contexto" in state["error"]["message"]

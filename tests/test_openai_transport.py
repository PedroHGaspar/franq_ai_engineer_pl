import json

import httpx
from langchain_openai import ChatOpenAI

from src.agent import DataAgent


def test_real_langchain_serialization_with_mock_http(settings):
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            name, args = "execute_query", {"sql": "SELECT COUNT(*) n FROM sales"}
        else:
            name, args = (
                "finish",
                {
                    "answer": {
                        "summary": "Foram 3 vendas.",
                        "evidence_ids": ["q1"],
                        "assumptions": [],
                        "chart": {"kind": "table"},
                    }
                },
            )
        return httpx.Response(
            200,
            json={
                "id": "mock-response",
                "object": "chat.completion",
                "created": 0,
                "model": settings.openai_model,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": f"call{len(requests)}",
                                    "type": "function",
                                    "function": {"name": name, "arguments": json.dumps(args)},
                                }
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        model = ChatOpenAI(
            model=settings.openai_model,
            api_key="test-placeholder",
            http_client=client,
            max_retries=0,
            max_tokens=1800,
            reasoning_effort="none",
        )
        state = DataAgent(settings, model).run("Total de vendas")
    assert state["status"] == "completed"
    assert state["usage"]["input_tokens"] == 200
    assert requests[0]["parallel_tool_calls"] is False
    assert requests[0]["reasoning_effort"] == "none"
    assert requests[0]["max_completion_tokens"] == 1800
    assert {t["function"]["name"] for t in requests[0]["tools"]} == {
        "list_tables",
        "inspect_schema",
        "execute_query",
        "finish",
        "clarify",
    }
    assert any(message["role"] == "tool" for message in requests[1]["messages"])

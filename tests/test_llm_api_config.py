"""Check API routing and headers through the actual SDK without network access."""

import json

import httpx
import pytest
from langchain_openai import ChatOpenAI

from rs_agent.controller import agent as agent_module
from rs_agent.controller.agent import RSAgent

TASK_TYPE = '["Scene_Classification"]'


def _api_response(path):
    if path.endswith("/responses"):
        return {
            "id": "resp_test",
            "object": "response",
            "created_at": 1,
            "model": "test-model",
            "status": "completed",
            "error": None,
            "usage": None,
            "output": [
                {
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": f" {TASK_TYPE} ", "annotations": []}
                    ],
                }
            ],
        }
    return {
        "id": "chatcmpl_test",
        "object": "chat.completion",
        "created": 1,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": f" {TASK_TYPE} "},
                "finish_reason": "stop",
            }
        ],
    }


@pytest.mark.parametrize(
    ("llm_overrides", "expected_path"),
    [
        pytest.param({}, "/v1/chat/completions", id="default-chat"),
        pytest.param({"use_responses_api": False}, "/v1/chat/completions", id="explicit-chat"),
        pytest.param({"use_responses_api": True}, "/v1/responses", id="responses"),
        pytest.param(
            {
                "use_responses_api": True,
                "default_headers": {"User-Agent": "RS-Agent/0.1"},
            },
            "/v1/responses",
            id="responses-custom-user-agent",
        ),
    ],
)
def test_config_controls_sdk_request(monkeypatch, llm_overrides, expected_path):
    requests = []

    def handle_request(request):
        requests.append(request)
        return httpx.Response(200, json=_api_response(request.url.path))

    config = {
        "agent": {"mode": "baseline"},
        "llm": {
            "model": "test-model",
            "api_key": "dummy-test-key",
            "api_base": "https://gateway.example/v1",
            **llm_overrides,
        },
    }
    with httpx.Client(transport=httpx.MockTransport(handle_request)) as client:
        monkeypatch.setattr(
            agent_module,
            "ChatOpenAI",
            lambda **kwargs: ChatOpenAI(http_client=client, max_retries=0, **kwargs),
        )
        agent = RSAgent.from_config(config)

        assert agent.infer_task_type("What scene is shown?") == TASK_TYPE

    assert len(requests) == 1
    request = requests[0]
    assert request.method == "POST"
    assert request.url == f"https://gateway.example{expected_path}"
    payload = json.loads(request.content)
    assert payload["model"] == "test-model"
    assert ("input" if expected_path.endswith("/responses") else "messages") in payload
    for name, value in llm_overrides.get("default_headers", {}).items():
        assert request.headers[name] == value

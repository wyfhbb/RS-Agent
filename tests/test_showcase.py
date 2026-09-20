"""The showcase preserves observations and only derives progress from completed calls."""

import json

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.prompt_values import ChatPromptValue

from examples.trace_showcase import progress_reminder
from rs_agent.showcase import export_showcase


def test_progress_counts_completed_calls_and_preserves_repeated_tool():
    scratchpad = (
        '```json\n{"action":"sar_plane_type","action_input":"sample.png"}\n```'
        "\nObservation: Boeing 747\nThought: "
        '```json\n{"action":"knowledge_search","action_input":"manufacturer?"}\n```'
        "\nObservation: No answer\nThought: "
    )
    value = ChatPromptValue(messages=[HumanMessage(content=scratchpad)])
    result = progress_reminder(
        value, expected=["sar_plane_type", "knowledge_search", "knowledge_search"]
    )
    assert "2/3" in result.messages[-1].content
    assert "本轮应执行：knowledge_search" in result.messages[-1].content
    assert result.messages[0].content == scratchpad
    assert len(value.messages) == 1


def test_final_reminder_and_task_inference_passthrough():
    value = ChatPromptValue(
        messages=[HumanMessage(content=('{"action":"scene"}\nObservation: airport\nThought: '))]
    )
    assert (
        "本轮应执行：Final Answer"
        in progress_reminder(value, expected=["scene"]).messages[-1].content
    )
    assert progress_reminder("task inference", expected=["scene"]) == "task inference"


def test_unfinished_action_is_not_counted_and_divergence_is_rejected():
    value = ChatPromptValue(messages=[HumanMessage(content='{"action":"scene"}')])
    assert "0/1" in progress_reminder(value, expected=["scene"]).messages[-1].content
    value.messages[0].content += "\nObservation: airport\nThought: "
    with pytest.raises(ValueError, match="diverged"):
        progress_reminder(value, expected=["denoising"])


def test_render_preserves_trace_and_escapes_model_text(tmp_path):
    directory = tmp_path / "case"
    directory.mkdir()
    malicious = "</script><script>window.INJECTED=true</script>"
    trace = {"metadata": {}, "intermediate_steps": [], "output": malicious}
    source = json.dumps(trace)
    (directory / "trace.json").write_text(source)
    target = export_showcase(tmp_path, [{"id": "case", "title": "案例"}])
    rendered = target.read_text()
    assert malicious not in rendered
    assert "\\u003c/script>" in rendered
    assert json.loads((tmp_path / "showcase.json").read_text())["cases"][0]["trace"] == trace
    assert (directory / "trace.json").read_text() == source


def test_no_recorded_traces_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="No recorded"):
        export_showcase(tmp_path, [{"id": "missing", "title": "未运行"}])


def test_progress_adapter_runs_through_actual_agent_executor():
    from functools import partial

    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from langchain_core.runnables import RunnableLambda

    from rs_agent.controller.agent import RSAgent
    from rs_agent.toolkit.registry import get_stub_tools

    model = FakeListChatModel(
        responses=[
            '```json\n{"action":"scene","action_input":"sample.png"}\n```',
            '```json\n{"action":"caption","action_input":"sample.png"}\n```',
            '```json\n{"action":"Final Answer","action_input":"Recorded stub output"}\n```',
        ]
    )
    llm = RunnableLambda(
        partial(progress_reminder, expected=["scene", "caption"])
    ) | RunnableLambda(model.invoke)
    agent = RSAgent(llm=llm, tools=get_stub_tools(), mode="baseline")
    result = agent.run("Call scene then caption", "sample.png")
    assert [s[0].tool for s in result["intermediate_steps"]] == ["scene", "caption"]
    assert result["output"] == "Recorded stub output"

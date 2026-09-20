"""Unit checks for evidence validation and real-event progress; no inference fixtures."""

import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.prompt_values import ChatPromptValue

from examples.detection_showcase import execution_reminder
from rs_agent.detection_showcase import export_detection_showcase


def prompt():
    return ChatPromptValue(messages=[HumanMessage(content="Compare these SAR images")])


def test_next_request_uses_next_image_and_detector():
    backend = SimpleNamespace(events=[{"model": "cascade", "image": "/a.png", "success": True}])
    schedule = [{"model": "cascade", "image": "/a.png"}, {"model": "gfl", "image": "/b.png"}]
    original = prompt()
    result = execution_reminder(original, backend=backend, schedule=schedule, stop=["Observation"])
    assert "sar_detection_gfl" in result.messages[-1].content
    assert '"/b.png"' in result.messages[-1].content
    assert "1/2" in result.messages[-1].content
    assert len(original.messages) == 1


def test_complete_real_events_request_final_answer():
    schedule = [{"model": "cascade", "image": "/a.png"}]
    backend = SimpleNamespace(events=[{**schedule[0], "success": True}])
    result = execution_reminder(prompt(), backend=backend, schedule=schedule)
    assert "本轮必须返回 Final Answer" in result.messages[-1].content


def test_different_image_is_not_accepted_as_completed():
    backend = SimpleNamespace(events=[{"model": "cascade", "image": "/other.png", "success": True}])
    with pytest.raises(ValueError, match="diverged"):
        execution_reminder(
            prompt(), backend=backend, schedule=[{"model": "cascade", "image": "/a.png"}]
        )


def test_failed_detector_does_not_advance_to_next_model():
    backend = SimpleNamespace(events=[{"model": "cascade", "image": "/a.png", "success": False}])
    with pytest.raises(RuntimeError, match="real detector failed"):
        execution_reminder(
            prompt(), backend=backend, schedule=[{"model": "cascade", "image": "/a.png"}]
        )


def test_task_inference_request_is_unchanged():
    assert execution_reminder("infer task", backend=None, schedule=[]) == "infer task"


def test_renderer_rejects_stub_traces(tmp_path):
    (tmp_path / "trace.json").write_text(json.dumps({"metadata": {"stub_tools": True}}))
    with pytest.raises(ValueError, match="real detector"):
        export_detection_showcase(tmp_path)

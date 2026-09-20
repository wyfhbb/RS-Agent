#!/usr/bin/env python3
"""Run a live, 12-call SAR detection trace using three trained detectors."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
WORKSPACE = ROOT.parent
MODELS = ["cascade", "gfl", "yolo11"]
CASES = [
    {"id": "ship", "title": "舰船样例", "image_id": 8, "file": "0017405.jpg"},
    {"id": "aircraft", "title": "飞机样例", "image_id": 4012, "file": "0107414.jpg"},
    {"id": "vehicle", "title": "车辆样例", "image_id": 11516, "file": "0023838.jpg"},
    {"id": "harbor", "title": "港口样例", "image_id": 10476, "file": "0100491.png"},
]


def execution_reminder(value, *, backend, schedule, **kwargs):
    """Expose completed real tool calls and the requested next image/model pair."""
    from langchain_core.messages import HumanMessage
    from langchain_core.prompt_values import ChatPromptValue

    if not isinstance(value, ChatPromptValue):
        return value
    observed = [(e["model"], e.get("image")) for e in backend.events]
    expected = [(s["model"], s["image"]) for s in schedule]
    if observed != expected[: len(observed)]:
        raise ValueError("Actual model/image sequence diverged from the requested schedule")
    if any(not e.get("success") for e in backend.events):
        raise RuntimeError("A real detector failed; its full evidence is in detector_calls")
    if len(observed) < len(schedule):
        pending = schedule[len(observed)]
        next_step = (
            f"本轮 action 必须为 sar_detection_{pending['model']}，"
            f"action_input 必须为 {json.dumps(pending['image'])}。"
        )
    else:
        next_step = "12 次真实检测均已完成。本轮必须返回 Final Answer，不再调用工具。"
    message = (
        f"真实工具执行进度：{len(observed)}/{len(schedule)}，由已返回的检测事件计算。"
        + next_step
        + "保持一轮一个 JSON action；不能把不同图像的检测结果混在一起。"
        "最终按图像编号汇总三模型在 score≥0.30 的各类预测数量，中文回答。"
        "仅描述结果分歧，不以数量或置信度判断模型准确率，不声称有真值验证。"
    )
    return ChatPromptValue(messages=[*value.to_messages(), HumanMessage(content=message)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/detection-showcase")
    parser.add_argument(
        "--detectors", type=Path, default=WORKSPACE / "midterm-exp/configs/agent_detectors.json"
    )
    parser.add_argument("--responses-api", action="store_true")
    parser.add_argument("--user-agent", default=None)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.render_only:
        from rs_agent.detection_showcase import export_detection_showcase

        export_detection_showcase(output)
        return
    output.mkdir(parents=True, exist_ok=False)

    from functools import partial

    from langchain_core.runnables import RunnableLambda
    from langchain_openai import ChatOpenAI

    from rs_agent.config import load_config
    from rs_agent.controller.agent import RSAgent
    from rs_agent.toolkit.sar_detection import SARDetectorBackend, sha256
    from rs_agent.trace import export_trace
    from scripts.run_sar_agent import LLMRecorder

    cases = [
        {
            **c,
            "image": str(
                (WORKSPACE / "midterm-exp/derived/images/test" / c["file"]).resolve(strict=True)
            ),
        }
        for c in CASES
    ]
    schedule = [
        {"case_id": c["id"], "image_id": c["image_id"], "image": c["image"], "model": m}
        for c in cases
        for m in MODELS
    ]
    detector_config = json.loads(args.detectors.read_text())
    detector_config["image_ids"] = {c["image"]: c["image_id"] for c in cases}
    used_config = output / "detectors.used.json"
    used_config.write_text(json.dumps(detector_config, indent=2) + "\n")
    (output / "cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n")
    (output / "schedule.json").write_text(json.dumps(schedule, indent=2) + "\n")
    backend = SARDetectorBackend.from_json(used_config, output / "detector_calls")
    tools = [t for t in backend.tools() if t.name != "sar_detection"]
    config = load_config()
    config["embedding"]["device"] = "cpu"
    config["agent"]["mode"] = "full"
    if args.responses_api:
        config["llm"]["use_responses_api"] = True
    if args.user_agent:
        config["llm"].setdefault("default_headers", {})["User-Agent"] = args.user_agent
    recorder = LLMRecorder(output / "llm_messages.jsonl")
    run = WORKSPACE / "midterm-exp/runs/20260915_032400"
    protocol = json.loads((run / "protocol.json").read_text())
    training = json.loads((run / "training/yolo11s_formal/status.json").read_text())
    weights = {
        name: {
            "path": protocol["models"][name]["weight"],
            "sha256": protocol["models"][name]["sha256"],
            "origin": "existing_SAR_checkpoint",
        }
        for name in ("cascade", "gfl")
    }
    weights["yolo11"] = {
        "path": str(run / "training/yolo11s_formal/weights/best.pt"),
        "sha256": training["best_sha256"],
        "origin": "local_SAR_finetuning",
        "completed_epochs": training["epoch"],
        "termination_reason": training["termination_reason"],
    }
    for weight in weights.values():
        if sha256(Path(weight["path"])) != weight["sha256"]:
            raise ValueError("Detector checkpoint differs from the recorded training provenance")
    metadata = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "model": config["llm"]["model"],
        "mode": "full",
        "stub_tools": False,
        "tool_backend": "real_cli",
        "embedding_model": config["embedding"]["model"],
        "embedding_device": "cpu",
        "registered_tools": [t.name for t in tools],
        "weights": weights,
        "explicit_operation_order": True,
        "observed_progress_reminder": True,
        "ground_truth_provided_to_agent": False,
        "display_score": 0.30,
        "sample_selection": "Four class strata for presentation, chosen before this inference; "
        "no accuracy claim or model-winner selection",
        "use_responses_api": config["llm"].get("use_responses_api", False),
        "disable_streaming": True,
        "reasoning_effort": "low",
        "llm_timeout_seconds": 120,
        "llm_attempts": 3,
        "real_forward_count_definition": "New successful detector CLI calls; warmup excluded",
    }
    source_paths = [
        ROOT / "examples/detection_showcase.py",
        ROOT / "rs_agent/controller/agent.py",
        ROOT / "rs_agent/controller/prompts.py",
        ROOT / "rs_agent/toolkit/sar_detection.py",
        WORKSPACE / "midterm-exp/scripts/mmdet_infer.py",
        WORKSPACE / "midterm-exp/scripts/mmdet_compat.py",
        WORKSPACE / "midterm-exp/scripts/yolo_infer.py",
    ]
    for source in source_paths:
        destination = output / "source_snapshot" / source.relative_to(WORKSPACE)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    question = (
        "请对下面4张SAR影像分别调用 Cascade R-CNN+MSFA、GFL+MSFA、YOLO11s-HBB，"
        "进行真实目标检测和对比。每张图按这三个模型的顺序各检测一次，总计12次。"
        "只使用注册的目标检测工具。完成后按图像编号列出三个模型在score≥0.30的分类数量，"
        "总结分歧，不能依据预测数量宣布谁更准确。无GT，不做AP/F1推断。"
        "不同工具的内部计时范围不同，不按pure_forward_seconds作速度排名。"
        "不要重复已完成的模型/图像组合，所有12项完成后立即Final Answer。\n"
        + "\n".join(f"图像 {i}: {c['image']}" for i, c in enumerate(cases, 1))
    )
    started = perf_counter()
    try:
        agent = RSAgent.from_config(config, tools=tools)
        cfg = config["llm"]
        llm = (
            ChatOpenAI(
                model=cfg["model"],
                temperature=0,
                api_key=cfg["api_key"],
                base_url=cfg.get("api_base"),
                use_responses_api=cfg.get("use_responses_api", False),
                default_headers=cfg.get("default_headers"),
                timeout=120,
                max_retries=0,
                disable_streaming=True,
                reasoning_effort="low",
                max_tokens=4096,
                callbacks=[recorder],
            )
            .with_retry(stop_after_attempt=3)
            .bind(stop=["\nObservation"])
        )
        agent.llm = RunnableLambda(
            partial(execution_reminder, backend=backend, schedule=schedule)
        ) | RunnableLambda(llm.invoke)
        print("START: 4 images × 3 trained detectors; 12 real calls", flush=True)
        result = agent.run(question, cases[0]["image"])
        metadata.update(
            duration_seconds=round(perf_counter() - started, 3),
            llm_request_count=recorder.calls,
            llm_usage=recorder.usage,
            tool_call_count=len(backend.events),
            real_forward_count=sum(e["execution_kind"] == "real_forward" for e in backend.events),
            cache_hit_count=sum(e["execution_kind"] == "cache_hit" for e in backend.events),
        )
        actual = [(a.tool, a.tool_input) for a, _ in result["intermediate_steps"]]
        expected = [(f"sar_detection_{s['model']}", s["image"]) for s in schedule]
        metadata["success"] = bool(
            actual == expected
            and metadata["real_forward_count"] == 12
            and result.get("output")
            and "Agent stopped" not in result["output"]
        )
        export_trace(
            result,
            question=question,
            image_path=cases[0]["image"],
            output_dir=output,
            metadata=metadata,
        )
        (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(
            json.dumps(
                {
                    k: metadata[k]
                    for k in [
                        "success",
                        "tool_call_count",
                        "real_forward_count",
                        "cache_hit_count",
                        "duration_seconds",
                    ]
                }
            ),
            flush=True,
        )
    except Exception as exc:
        metadata.update(
            success=False,
            duration_seconds=perf_counter() - started,
            error_type=type(exc).__name__,
            tool_call_count=len(backend.events),
            llm_request_count=recorder.calls,
        )
        (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        raise
    # Presentation export failures must not relabel completed detector inference.
    from rs_agent.detection_showcase import export_detection_showcase

    export_detection_showcase(output)
    if not metadata["success"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

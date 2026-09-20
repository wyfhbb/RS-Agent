#!/usr/bin/env python3
"""Run four long, live RS-Agent planning traces and render an offline showcase."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CASES = [
    {
        "id": "airport",
        "title": "机场综合解译",
        "subtitle": "影像增强 → 场景理解 → 目标识别 → 知识核验",
        "question": (
            "对机场光学影像做一次完整的综合解译演示，依次完成去云、去雾、去噪、2倍超"
            "分、场景分类、图像描述、光学目标检测、飞机型号识别，最后查询所识别机型的"
            "制造商。"
        ),
        "tools": [
            "cloud_removal",
            "image_dehazing",
            "denoising",
            "super_resolution_2x",
            "scene",
            "caption",
            "optical_detection",
            "optical_plane_type",
            "knowledge_search",
        ],
    },
    {
        "id": "disaster",
        "title": "灾后建筑与道路排查",
        "subtitle": "质量改善 → 建筑提取 → 损伤排查 → 道路与地表",
        "question": (
            "演示灾后区域排查流程：先去云、去雾，然后提取建筑，做建筑损伤检测，再提取道路、语义分"
            "割，最后做土地利用分类。汇总各环节有哪些返回，以及哪些信息还不足以形成灾情结论。"
        ),
        "tools": [
            "cloud_removal",
            "image_dehazing",
            "building_extraction",
            "building_damage_detection",
            "road_extraction",
            "semantic_segmentation",
            "land_use_classification",
        ],
    },
    {
        "id": "sar",
        "title": "SAR 目标识别与核验",
        "subtitle": "去噪与增强 → SAR 检测 → 机型识别 → 知识检索",
        "question": (
            "演示SAR影像分析：先去噪，再2倍超分，然后SAR目标检测、SAR飞机型号识别，查询"
            "识别机型的制造商，最后再查询这个机型的首飞年份。两次知识检索的查询必须不同。不要从空"
            "泛观察中推断数量、坐标或置信度。"
        ),
        "tools": [
            "denoising",
            "super_resolution_2x",
            "sar_detection",
            "sar_plane_type",
            "knowledge_search",
            "knowledge_search",
        ],
    },
    {
        "id": "land",
        "title": "地表精细解译与目标检测",
        "subtitle": "场景分类 → 双检测方式 → 建筑道路 → 地表分类",
        "question": (
            "演示地表综合解译：依次做场景分类、水平框目标检测、旋转框目标检测、建筑提取、道路提取"
            "、语义分割、土地利用分类，最后生成图像描述。按实际工具返回形成简洁的中文报告，说明是"
            "否有可供比较的坐标和分割掩膜。"
        ),
        "tools": [
            "scene",
            "horizontal_object_detection",
            "rotated_object_detection",
            "building_extraction",
            "road_extraction",
            "semantic_segmentation",
            "land_use_classification",
            "caption",
        ],
    },
]


def progress_reminder(value, *, expected, **kwargs):
    """Make observed progress explicit for this prescribed-order demonstration."""
    from langchain_core.messages import HumanMessage
    from langchain_core.prompt_values import ChatPromptValue

    if not isinstance(value, ChatPromptValue):
        return value  # The independent task inference request has no scratchpad.
    messages = value.to_messages()
    text = messages[-1].content
    completed = []
    for chunk in text.split("\nObservation: ")[:-1]:
        actions = re.findall(r'"action"\s*:\s*"([^"]+)"', chunk)
        if actions:
            completed.append(actions[-1])
    if completed != expected[: len(completed)]:
        raise ValueError(f"Observed sequence diverged from demo checklist: {completed}")
    next_action = expected[len(completed)] if len(completed) < len(expected) else "Final Answer"
    note = (
        f"执行进度（由已返回 Observation 的实际调用计算）：{len(completed)}/{len(expected)}。"
        f"已完成：{json.dumps(completed)}。本轮应执行：{next_action}。"
        "严格接着当前进度执行，不要从头重来。若本轮为 Final Answer，汇总已记录观察，"
        "不再调用任何工具。用中文回答，总计不超过450字，包含完成的步骤和结果限制。"
    )
    return ChatPromptValue(messages=[*messages, HumanMessage(content=note)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/trajectory-showcase")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--responses-api", action="store_true")
    parser.add_argument("--user-agent")
    parser.add_argument("--case", choices=[c["id"] for c in CASES])
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not args.render_only:
        from functools import partial

        from langchain_core.runnables import RunnableLambda
        from langchain_openai import ChatOpenAI

        from rs_agent.config import load_config
        from rs_agent.controller.agent import RSAgent
        from rs_agent.trace import export_trace
        from scripts.run_sar_agent import LLMRecorder

        config = load_config(args.config)
        config["embedding"]["device"] = "cpu"
        if args.responses_api:
            config["llm"]["use_responses_api"] = True
        if args.user_agent:
            config["llm"].setdefault("default_headers", {})["User-Agent"] = args.user_agent
        agent = RSAgent.from_config(config)
        for case in CASES:
            if args.case and case["id"] != args.case:
                continue
            directory = output / case["id"]
            if (directory / "trace.json").exists():
                print(f"KEEP {case['id']}: existing recorded trace", flush=True)
                continue
            directory.mkdir(parents=True, exist_ok=True)
            recorder = LLMRecorder(directory / "llm_messages.jsonl")
            cfg = config["llm"]
            llm = (
                ChatOpenAI(
                    model=cfg["model"],
                    temperature=cfg.get("temperature", 0),
                    api_key=cfg["api_key"],
                    base_url=cfg.get("api_base"),
                    use_responses_api=cfg.get("use_responses_api", False),
                    default_headers=cfg.get("default_headers"),
                    timeout=90,
                    disable_streaming=True,
                    reasoning_effort="low",
                    max_retries=0,
                    max_tokens=4096,
                    callbacks=[recorder],
                )
                .with_retry(stop_after_attempt=3)
                .bind(stop=["\nObservation"])
            )
            # AgentExecutor streams its runnable, while RunnableRetry retries invoke().
            # Keep each model call inside invoke so retries also cover execution steps.
            agent.llm = RunnableLambda(
                partial(progress_reminder, expected=case["tools"])
            ) | RunnableLambda(llm.invoke)
            question = case["question"] + (
                "\n这是使用仓库 stub 工具的规划演示。严格按以下清单逐项调用，每项一次："
                + " → ".join(case["tools"])
                + "。检查当前 scratchpad 中已完成的调用，从下一项继续；"
                "全部完成后返回 Final Answer。"
                "stub 不生成新图像，所以所有图像工具都使用提供的原始路径；知识查询使用英文问题。"
                "不要重试已完成的清单项，即使观察值信息有限。最终回答用中文，简洁列出执行记录，"
                "明确说明工具返回是预设文本，不代表真实图像分析；检索未回答时明确写未获取，不能补造。"
            )
            image = ROOT / "examples/sample.png"
            started = datetime.now(timezone.utc).isoformat()
            t0 = perf_counter()
            print(f"START {case['id']} expected={len(case['tools'])}", flush=True)
            try:
                result = agent.run(question, str(image))
                actual = [action.tool for action, _ in result["intermediate_steps"]]
                matched = actual == case["tools"]
                metadata = {
                    "model": cfg["model"],
                    "mode": agent.mode,
                    "stub_tools": True,
                    "tool_backend": "stubs",
                    "prompt": "local_structured_chat",
                    "use_responses_api": cfg.get("use_responses_api", False),
                    "embedding_model": config["embedding"]["model"],
                    "embedding_device": "cpu",
                    "started_at": started,
                    "duration_seconds": round(perf_counter() - t0, 3),
                    "llm_request_count": recorder.calls,
                    "llm_usage": recorder.usage,
                    "expected_tools": case["tools"],
                    "sequence_matches": matched,
                    "success": bool(
                        matched and result.get("output") and "Agent stopped" not in result["output"]
                    ),
                    "case_id": case["id"],
                    "case_title": case["title"],
                    "explicit_operation_order": True,
                    "observed_progress_reminder": True,
                    "disable_streaming": True,
                    "reasoning_effort": "low",
                    "llm_retry_attempts": 3,
                }
                export_trace(
                    result,
                    question=question,
                    image_path=image,
                    output_dir=directory,
                    metadata=metadata,
                )
                print(
                    f"DONE {case['id']} steps={len(actual)} matched={matched} "
                    f"seconds={metadata['duration_seconds']}",
                    flush=True,
                )
            except Exception as exc:
                error = {
                    "case_id": case["id"],
                    "error_type": type(exc).__name__,
                    "error_message": str(exc).replace(cfg["api_key"], "[REDACTED]")[:600],
                    "llm_request_count": recorder.calls,
                    "started_at": started,
                }
                (directory / "error.json").write_text(json.dumps(error, indent=2))
                print(f"ERROR {case['id']}: {error['error_message']}", flush=True)
    from rs_agent.showcase import export_showcase

    export_showcase(output, CASES)
    print(f"Showcase: {output / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()

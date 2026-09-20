"""Render three-detector comparisons exclusively from recorded real CLI results."""

from __future__ import annotations

import base64
import json
import mimetypes
import shutil
from collections import Counter
from pathlib import Path

from rs_agent.toolkit.sar_detection import sha256


def export_detection_showcase(output: Path) -> Path:
    from PIL import Image

    trace = json.loads((output / "trace.json").read_text())
    if trace["metadata"].get("stub_tools") or trace["metadata"].get("tool_backend") != "real_cli":
        raise ValueError("This showcase requires recorded real detector results")
    cases = json.loads((output / "cases.json").read_text())
    schedule = json.loads((output / "schedule.json").read_text())
    events = [
        json.loads(line)
        for line in (output / "detector_calls/tool_calls.jsonl").read_text().splitlines()
    ]
    if len(trace["intermediate_steps"]) != len(schedule) or len(events) != len(schedule):
        raise ValueError("The recorded trace does not contain the complete detection schedule")
    assets = output / "assets"
    assets.mkdir(exist_ok=True)
    records = {c["id"]: {**c, "models": []} for c in cases}
    verification = []
    for entry, step, event in zip(schedule, trace["intermediate_steps"], events, strict=True):
        obs = json.loads(step["observation"])
        if step["action"]["tool"] != f"sar_detection_{entry['model']}":
            raise ValueError("Recorded tool differs from scheduled detector")
        if step["action"]["tool_input"] != entry["image"] or obs["image_id"] != entry["image_id"]:
            raise ValueError("Recorded detector image identity differs from schedule")
        if obs["execution_kind"] != "real_forward" or not event.get("success"):
            raise ValueError("Missing a fresh successful real detector call")
        if (
            obs != event["observation"]
            or sha256(Path(obs["result_json"])) != event["result_sha256"]
        ):
            raise ValueError("Trace observation or detector result hash differs from tool event")
        result = json.loads(Path(obs["result_json"]).read_text())
        expected_weight = trace["metadata"]["weights"][entry["model"]]["sha256"]
        if obs["checkpoint_sha256"] != expected_weight:
            raise ValueError("Executed checkpoint differs from the selected trained model")
        if sha256(Path(entry["image"])) != event["image_sha256"]:
            raise ValueError("Input image changed after recorded inference")
        if sha256(Path(obs["prediction_json"])) != event["prediction_sha256"]:
            raise ValueError("Prediction JSON changed after recorded inference")
        if not result["actual_forward"] or result["cache_hit"]:
            raise ValueError("Detector output is not a fresh inference")
        shown = [d for d in result["detections"] if d["score"] >= 0.30]
        counts = dict(Counter(d["class_name"] for d in shown))
        if counts != obs["counts_at_visualization_threshold"]:
            raise ValueError("Displayed counts differ from the recorded observation")
        if len(result["detections"]) != obs["raw_prediction_count"]:
            raise ValueError("Raw detection count differs from the trace")
        key = f"{entry['case_id']}-{entry['model']}"
        annotated = assets / f"{key}.png"
        predictions = assets / f"{key}.json"
        shutil.copy2(obs["visualization"], annotated)
        shutil.copy2(obs["prediction_json"], predictions)
        records[entry["case_id"]]["models"].append(
            {
                "key": entry["model"],
                "step": step["step"],
                "tool": step["action"]["tool"],
                "observation": obs,
                "detections": result["detections"],
                "annotated_image": annotated.relative_to(output).as_posix(),
                "prediction_json": predictions.relative_to(output).as_posix(),
                "tool_elapsed_seconds": event["elapsed_seconds"],
            }
        )
        verification.append(
            {
                "step": step["step"],
                "case_id": entry["case_id"],
                "model": entry["model"],
                "image_id": entry["image_id"],
                "counts_at_0_30": counts,
                "real_forward": True,
                "result_sha256": event["result_sha256"],
                "checkpoint_sha256": obs["checkpoint_sha256"],
            }
        )
    for c in records.values():
        image_path = Path(c["image"])
        destination = assets / f"{c['id']}-input{image_path.suffix}"
        shutil.copy2(image_path, destination)
        with Image.open(image_path) as im:
            c["width"], c["height"] = im.size
        c["image_uri"] = (
            f"data:{mimetypes.guess_type(image_path.name)[0]};base64,"
            + base64.b64encode(image_path.read_bytes()).decode()
        )
        c["input_image"] = destination.relative_to(output).as_posix()
        c["image_sha256"] = sha256(image_path)
    data = {"cases": list(records.values()), "trace": trace}
    (output / "showcase.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    template = Path(__file__).with_name("detection_showcase.html").read_text()
    embedded = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")
    (output / "index.html").write_text(template.replace("__DETECTION_DATA__", embedded))
    (output / "validation.json").write_text(
        json.dumps(
            {
                "real_calls": len(verification),
                "verified": verification,
                "trace_sha256": sha256(output / "trace.json"),
                "checks": [
                    "scheduled model and image",
                    "trace matches real event",
                    "result file hash",
                    "fresh inference flag",
                    "counts match raw predictions at 0.30",
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    lines = [
        "# 三模型真实目标检测 · RS-Agent Trace",
        "",
        "入口：[index.html](index.html)，离线打开即可。",
        "",
        "4 张 SAR 图像 × 3 个检测模型 = 12 次新执行的真实检测，0 次缓存复用。",
        "一条完整 Agent trace 串联全部 12 次调用；页面按图像切分展示三模型对比。",
        "",
        "## 使用方式",
        "",
        "- 左侧或数字键 1–4 切换图像；点击流程节点查看对应工具观察。",
        "- 点击图片放大查看检测框，支持开关类别标签和检测框。",
        "- 置信度滑杆只过滤已保存预测，不重新运行模型；默认 0.30。",
        "- 播放会按原始 12 步顺序回放；播放速度不是实际推理速度。",
        "- 每个模型可下载原始 CLI 标注图和 COCO 预测 JSON。",
        "- trace.json / trace.html / trace.svg 保留完整长链和最终回答。",
        "",
        "## 实际预测数量（score ≥ 0.30）",
        "",
        "| 样例 | Cascade + MSFA | GFL + MSFA | YOLO11s-HBB |",
        "|---|---|---|---|",
    ]
    for c in records.values():
        values = [
            ", ".join(
                f"{k}: {v}"
                for k, v in m["observation"]["counts_at_visualization_threshold"].items()
            )
            or "0"
            for m in c["models"]
        ]
        lines.append(f"| {c['title']} / ID {c['image_id']} | " + " | ".join(values) + " |")
    lines += [
        "",
        "## 结果含义",
        "",
        "Cascade/GFL 使用已有的 SAR checkpoint；YOLO 使用本地 SAR 微调的 best.pt，"
        "训练按用户要求在完成 19 轮后停止。权重 SHA-256 写入 trace 元数据。",
        "四张图按类别覆盖选取，用于展示，不是随机性能评测；未向 Agent 提供 GT 框、"
        "真值数量或评测分数。模型顺序按用户要求预先指定，并附基于实际工具事件的进度提示。",
        "显示数量是模型预测，不代表真值，也不能据此判定模型准确率。",
        "统一显示阈值 0.30；模型原有的预处理、原始分数阈值与 NMS 设置保留。",
        "MMDetection 的 pure_forward_seconds 包含 decode/NMS/rescale；YOLO 的框架"
        "inference timer 不含 NMS，因此页面不按该字段做跨模型速度排名。",
        "",
        "## 复现",
        "",
        "在 RS-Agent 目录运行：",
        "",
        "```bash",
        "uv run --no-sync examples/detection_showcase.py --responses-api --user-agent RS-Agent/0.1",
        "# 指定新的 --output 目录才能重新采集，防止覆盖原始证据。",
        "# 只渲染，不调用模型：",
        "uv run --no-sync examples/detection_showcase.py --render-only",
        "```",
        "",
    ]
    (output / "README.md").write_text("\n".join(lines))
    print(f"Detection showcase: {output / 'index.html'}", flush=True)
    return output / "index.html"

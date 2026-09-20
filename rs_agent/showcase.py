"""Portable gallery for recorded traces. Rendering never calls a model."""

from __future__ import annotations

import base64
import json
from pathlib import Path


def export_showcase(output: Path, cases: list[dict]) -> Path:
    records = []
    for case in cases:
        path = output / case["id"] / "trace.json"
        if path.exists():
            records.append({**case, "trace": json.loads(path.read_text(encoding="utf-8"))})
    if not records:
        raise ValueError("No recorded trace.json files found; run the examples first")
    data = {"cases": records, "description": "Live LLM + original repository stub tools"}
    (output / "showcase.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    # Escape script terminators, including any originating in model/tool text.
    embedded = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")
    template = Path(__file__).with_name("showcase.html").read_text(encoding="utf-8")
    sample = Path(__file__).parents[1] / "examples/sample.png"
    picture = ""
    if sample.exists():
        picture = "data:image/png;base64," + base64.b64encode(sample.read_bytes()).decode()
    html = template.replace("__TRACE_DATA__", embedded).replace("__SAMPLE_IMAGE__", picture)
    target = output / "index.html"
    target.write_text(html, encoding="utf-8")
    steps = sum(len(row["trace"]["intermediate_steps"]) for row in records)
    tools = {s["action"]["tool"] for row in records for s in row["trace"]["intermediate_steps"]}
    lines = [
        "# RS-Agent 多场景长流程 Trace 展示",
        "",
        "入口：双击 `index.html`，无需服务器或网络。",
        "",
        f"本包共 {len(records)} 个案例，实际记录 {steps} 次工具调用，覆盖 {len(tools)} 种工具。",
        "",
        "## 演示方式",
        "",
        "- 左侧切换案例，点击流程节点查看实际输入与观察值。",
        "- 点击「播放轨迹」逐步回放；左右方向键切换步骤，空格播放/暂停。",
        "- 「展示模式」隐藏详细字段并放大流程；按 F 或点击按钮切换。",
        "- 点击「打印 / PDF」生成全部案例的报告，可保存 PDF。",
        "- 各案例目录包含原始 trace.json、trace.html、trace.svg 和 llm_messages.jsonl。",
        "- 展示模式下仍可用数字键 1–4 切换案例。",
        "",
        "## 建议讲解顺序（约 3–5 分钟）",
        "",
        "1. 机场：播放 9 步长链，展示从影像增强到识别、知识查询的调用顺序。",
        "2. 灾后：展示建筑、损伤、道路与地表任务如何串联，以及观察值的不足。",
        "3. SAR：点开最后两次知识检索，对比制造商与首飞年份两个不同输入。",
        "4. 地表：展示水平框、旋转框和像素级任务的多工具覆盖。",
        "",
        "## 数据含义",
        "",
        "模型和 Solution Space 检索均实际运行；工具采用仓库原有 stub，观察值为预设文本。",
        "全部案例使用同一张 sample.png 作为路径占位，不代表真实灾区或 SAR 输入。",
        "用户问题明确规定工具顺序，并根据实际 Observation 添加进度提示；不作为自主规划准确率评测。",
        "箭头只表示执行顺序，不表示图像文件产物流转。回放间隔不是实际调用耗时。",
        "成功标记仅表示实际调用顺序与清单一致且产生最终回答，不评价遥感结果正确性。",
        "",
        "## 本次记录",
        "",
        "| 案例 | 实际步骤 | 顺序检查 | 耗时 |",
        "| --- | ---: | --- | ---: |",
    ]
    for row in records:
        trace = row["trace"]
        meta = trace["metadata"]
        lines.append(
            f"| {row['title']} | {len(trace['intermediate_steps'])} | "
            f"{'通过' if meta.get('success') else '需检查'} | "
            f"{meta.get('duration_seconds', 0):.1f}s |"
        )
    if (output / "overview.png").exists():
        lines += [
            "",
            "## 已导出展示素材",
            "",
            "- [交互展示页](index.html)",
            "- [页面总览截图](overview.png)",
            "- [全部案例 PDF](trace-showcase.pdf)",
            "- 各案例 `slide.png`：16:9 纯流程截图，适合 PPT。",
            "- 各案例 `showcase.png`：包含工具输入、观察值与最终回答的页面截图。",
            "- `validation.json`：调用、原始日志、stub 返回、截图尺寸和 PDF 页数的核验结果。",
            "- `attempts-*`：前期超时、响应异常和调试尝试的原始日志，不属于成功案例。",
        ]
    lines += [
        "",
        "## 复现",
        "",
        "在 RS-Agent 目录运行：",
        "",
        "```bash",
        "uv run --no-sync examples/trace_showcase.py --responses-api --user-agent RS-Agent/0.1",
        "# 仅更新页面，不调用模型",
        "uv run --no-sync examples/trace_showcase.py --render-only",
        "```",
        "",
        "已有 trace.json 会保留；重新采集请用 --output 指定一个新目录。",
        "",
    ]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")
    return target

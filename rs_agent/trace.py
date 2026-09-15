"""Export an RSAgent run as JSON and portable execution-trajectory visuals.

The diagram describes recorded tool calls and observations. It does not render
the model's action logs; those are retained, without truncation, in trace.json.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import enum
import html
import json
import math
import re
import unicodedata
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _json_value(value: Any) -> Any:
    """Convert common LangChain/Pydantic values without silently dropping data."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, enum.Enum):
        return _json_value(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("Trace object keys must be strings to preserve their values in JSON")
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "model_dump"):
        return _json_value(value.model_dump())
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _json_value(dataclasses.asdict(value))
    raise TypeError(f"Cannot preserve {type(value).__name__} in a JSON trace")


def _display(value: Any) -> str:
    if value is None:
        return "未执行 / 无返回值 (null)"
    if isinstance(value, str):
        return value if value else '"" (empty string)'
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)


def _xml(value: str) -> str:
    # XML 1.0 excludes some characters that JSON can preserve safely.
    clean = "".join(
        char
        if char in "\t\n\r"
        or 0x20 <= ord(char) <= 0xD7FF
        or 0xE000 <= ord(char) <= 0xFFFD
        or 0x10000 <= ord(char) <= 0x10FFFF
        else "\ufffd"
        for char in value
    )
    return html.escape(clean, quote=True)


def _wrap(value: str, width: int = 104, max_lines: int = 10) -> list[str]:
    """Wrap the SVG preview by approximate display columns, including CJK."""
    lines: list[str] = []
    for paragraph in value.replace("\t", "    ").splitlines() or [""]:
        line = ""
        columns = 0
        for char in paragraph:
            char_width = (
                0
                if unicodedata.combining(char)
                else (2 if unicodedata.east_asian_width(char) in "WF" else 1)
            )
            if columns + char_width > width and line:
                lines.append(line)
                line, columns = "", 0
                if len(lines) > max_lines:
                    return lines[:max_lines] + ["… 完整内容见 trace.html / trace.json"]
            line += char
            columns += char_width
        lines.append(line)
        if len(lines) > max_lines:
            return lines[:max_lines] + ["… 完整内容见 trace.html / trace.json"]
    return lines


def _stub_notice(metadata: Mapping[str, Any]) -> str | None:
    if metadata.get("stub_tools") or metadata.get("tool_backend") in {"stub", "stubs"}:
        return (
            "Stub 工具演示：工具观察值来自预设返回；此轨迹未执行真实遥感视觉推理，"
            "最终回答不代表对图像内容的验证。"
        )
    return None


def _stages(trace: dict[str, Any]) -> list[tuple[str, str, list[tuple[str, Any]]]]:
    stages = [
        (
            "PREPARATION · INPUT",
            "输入",
            [
                ("question", trace["input"]["question"]),
                ("image_path", trace["input"]["image_path"]),
            ],
        ),
        (
            "PREPARATION · TASK INFERENCE",
            "任务类型推断",
            [
                ("predicted_task_type", trace["predicted_task_type"]),
            ],
        ),
        (
            "PREPARATION · SOLUTION RETRIEVAL",
            "方案检索",
            [
                ("guidance", trace["guidance"]),
            ],
        ),
    ]
    for step in trace["intermediate_steps"]:
        action = step["action"]
        stages.append(
            (
                f"INTERMEDIATE_STEPS · STEP {step['step']}",
                str(action["tool"]),
                [
                    ("tool_input", action["tool_input"]),
                    ("observation", step["observation"]),
                ],
            )
        )
    if not trace["intermediate_steps"]:
        stages.append(
            (
                "INTERMEDIATE_STEPS · 0 STEPS",
                "未记录工具调用",
                [
                    ("intermediate_steps", []),
                ],
            )
        )
    stages.append(("FINAL OUTPUT", "最终输出", [("output", trace["output"])]))
    return stages


def _render_detailed_svg(trace: dict[str, Any]) -> str:
    sections: list[str] = []
    y = 40

    def text_line(text: str, x: int, baseline: int, css: str) -> str:
        return f'<text x="{x}" y="{baseline}" class="{css}">{_xml(text)}</text>'

    sections.append(text_line("RS-Agent · intermediate_steps 执行轨迹", 48, y + 27, "heading"))
    sections.append(
        text_line(
            f"已记录 {len(trace['intermediate_steps'])} 次工具调用 · 箭头表示执行顺序 · "
            "预处理独立于 intermediate_steps",
            48,
            y + 59,
            "subtitle",
        )
    )
    y += 91
    notice = _stub_notice(trace["metadata"])
    if notice:
        notice_lines = _wrap(notice)
        box_height = 32 + 23 * len(notice_lines)
        sections.append(
            f'<rect x="48" y="{y}" width="1024" height="{box_height}" rx="12" fill="#fff4d8"/>'
        )
        for offset, line in enumerate(notice_lines):
            sections.append(text_line(line, 66, y + 28 + offset * 23, "notice"))
        y += box_height + 28

    stages = _stages(trace)
    for index, (tag, title, fields) in enumerate(stages):
        preview = [(label, _wrap(_display(value))) for label, value in fields]
        height = 84 + sum(34 + 23 * len(lines) for _, lines in preview)
        is_action = tag.startswith("INTERMEDIATE_STEPS")
        accent = "#2563eb" if is_action else "#64748b"
        if tag == "FINAL OUTPUT":
            accent = "#0f766e"
        sections.append(
            f'<rect x="48" y="{y}" width="1024" height="{height}" '
            'rx="14" fill="#ffffff" stroke="#d9e2ee"/>'
        )
        sections.append(
            f'<rect x="48" y="{y + 14}" width="5" height="{height - 28}" rx="2" fill="{accent}"/>'
        )
        sections.append(text_line(tag, 72, y + 29, "tag"))
        title_lines = _wrap(title, width=75, max_lines=1)
        title_preview = title_lines[0] + ("…" if len(title_lines) > 1 else "")
        sections.append(text_line(title_preview, 72, y + 59, "title"))
        baseline = y + 91
        for label, lines in preview:
            sections.append(text_line(label, 72, baseline, "label"))
            baseline += 25
            for line in lines:
                sections.append(text_line(line, 72, baseline, "value"))
                baseline += 23
            baseline += 9
        y += height
        if index < len(stages) - 1:
            sections.append(
                f'<path d="M560 {y + 4} V{y + 28}" stroke="#8b9ab0" '
                'stroke-width="2" marker-end="url(#arrow)"/>'
            )
            y += 40
    y += 46
    body = "\n".join(sections)
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="{y}"
viewBox="0 0 1120 {y}" role="img" aria-labelledby="trace-title trace-desc">
<title id="trace-title">RS-Agent intermediate_steps execution trajectory</title>
<desc id="trace-desc">Recorded input, preparatory stages, tool calls and observations,
and final output. Long values are previewed; full values are in trace.html and trace.json.</desc>
<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
<path d="M0 0 L8 4 L0 8" fill="none" stroke="#8b9ab0"/></marker></defs>
<style>
text{{font-family:ui-monospace,Consolas,"Noto Sans CJK SC","Microsoft YaHei",monospace;}}
.heading{{font-size:27px;font-weight:700;fill:#14213d;}}
.subtitle{{font-size:15px;fill:#56657c;}}
.tag{{font-size:13px;letter-spacing:1px;fill:#526582;}}
.title{{font-size:21px;font-weight:700;fill:#14213d;}}
.label{{font-size:14px;font-weight:700;fill:#526582;}}
.value{{font-size:16px;fill:#21324d;white-space:pre;}}
.notice{{font-size:16px;fill:#855000;}}
</style>
<rect width="1120" height="{y}" fill="#f3f6fb"/>
{body}
</svg>'''


def _short_lines(value: Any, width: int = 16, limit: int = 3) -> list[str]:
    """Keep labels readable on a slide; full values remain in HTML and JSON."""
    text = " ".join(_display(value).split())
    lines: list[str] = []
    current = ""
    for word in re.findall(r"\S+\s*", text):
        pieces = _wrap(word.rstrip(), width=width, max_lines=len(word) + 1)
        for piece in pieces:
            candidate = (current + " " + piece).strip()
            if len(_wrap(candidate, width=width, max_lines=len(candidate) + 1)) > 1:
                if current:
                    lines.append(current)
                current = piece
            else:
                current = candidate
    if current:
        lines.append(current)
    if len(lines) > limit:
        lines = lines[:limit]
        lines[-1] = lines[-1].rstrip(" .,…") + "…"
    return lines or [""]


def _compact_result(value: Any) -> Any:
    if isinstance(value, str):
        # Remove only a known presentation wrapper, preserving the actual label.
        match = re.fullmatch(r"The scene of this image is (.+?)\.?", value.strip())
        if match:
            return match.group(1)
    return value


def _render_compact_svg(trace: dict[str, Any]) -> str:
    task = trace["predicted_task_type"]
    if isinstance(task, str):
        try:
            task = json.loads(task)
        except (ValueError, TypeError):
            pass
    if isinstance(task, list):
        task = ", ".join(str(item) for item in task)
    task = str(task).replace("_", " ") if task is not None else "未执行"
    guidance = trace["guidance"]
    tools = re.findall(r"\btool\s+['\"]([^'\"]+)['\"]", str(guidance))
    guidance_label = (
        ", ".join(dict.fromkeys(tools))
        if tools
        else ("已返回方案" if guidance is not None else "未执行")
    )
    nodes = [
        ("输入图像", Path(trace["input"]["image_path"]).name, "preparation"),
        ("任务推断", task, "preparation"),
        ("方案检索", guidance_label, "preparation"),
    ]
    groups: list[list[dict[str, Any]]] = []
    for step in trace["intermediate_steps"]:
        previous = groups[-1][-1] if groups else None
        if (
            previous is not None
            and all(
                previous["action"][key] == step["action"][key] for key in ("tool", "tool_input")
            )
            and previous["observation"] == step["observation"]
        ):
            groups[-1].append(step)
        else:
            groups.append([step])
    for group in groups:
        first, last = group[0]["step"], group[-1]["step"]
        number = str(first) if first == last else f"{first}–{last}"
        nodes.append((f"工具调用 · STEP {number}", group[0]["action"]["tool"], "action"))
    if not groups:
        nodes.append(("工具调用 · 0 STEP", "未调用工具", "preparation"))
    nodes.append(("最终结果", _compact_result(trace["output"]), "output"))

    width, columns, card_width, card_height = 1440, 5, 240, 156
    rows = math.ceil(len(nodes) / columns)
    height = 370 + (rows - 1) * 204
    sections: list[str] = []

    def text_line(value: str, x: int, y: int, css: str, anchor: str = "start") -> str:
        return f'<text x="{x}" y="{y}" class="{css}" text-anchor="{anchor}">{_xml(value)}</text>'

    sections.append(text_line("RS-Agent 执行轨迹", 60, 55, "heading"))
    model = trace["metadata"].get("model", "")
    mode = trace["metadata"].get("mode", "")
    info = " · ".join(str(value) for value in (model, mode) if value)
    info += (" · " if info else "") + f"{len(trace['intermediate_steps'])} 次工具调用"
    sections.append(text_line(" ".join(_short_lines(info, 55, 1)), 1380, 55, "meta", "end"))
    positions = []
    for index in range(len(nodes)):
        row, column = divmod(index, columns)
        if row % 2:
            column = columns - 1 - column
        positions.append((60 + column * 270, 119 + row * 204))
    for index, (x, y) in enumerate(positions[:-1]):
        next_x, next_y = positions[index + 1]
        if next_y == y:
            right = next_x > x
            x1, x2 = (x + card_width + 5, next_x - 9) if right else (x - 5, next_x + card_width + 9)
            route = f"M{x1} {y + card_height // 2} H{x2}"
        else:
            route = f"M{x + card_width // 2} {y + card_height + 5} V{next_y - 9}"
        sections.append(
            f'<path d="{route}" stroke="#94a3b8" stroke-width="2.5" fill="none" '
            'marker-end="url(#arrow)"/>'
        )
    for (label, value, kind), (x, y) in zip(nodes, positions):
        sections.append(
            f'<rect x="{x}" y="{y}" width="{card_width}" height="{card_height}" '
            f'rx="16" class="card {kind}"/>'
        )
        sections.append(text_line(label, x + card_width // 2, y + 35, "label", "middle"))
        lines = _short_lines(value)
        baseline = y + 100 - (len(lines) - 1) * 16
        for offset, line in enumerate(lines):
            sections.append(
                text_line(
                    line, x + card_width // 2, baseline + offset * 32, f"value {kind}", "middle"
                )
            )
    if _stub_notice(trace["metadata"]):
        sections.append(text_line("Stub 工具 · 结果来自预设返回", 60, height - 35, "notice"))
    if any(len(group) > 1 for group in groups):
        sections.append(
            text_line(
                "连续相同调用已合并，完整记录见 HTML / JSON", 1380, height - 35, "meta", "end"
            )
        )
    body = "\n".join(sections)
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"
viewBox="0 0 {width} {height}" role="img" aria-labelledby="trace-title trace-desc">
<title id="trace-title">RS-Agent compact execution trajectory</title>
<desc id="trace-desc">Compact presentation layout. Preparatory stages precede numbered tool calls.
Long labels are abbreviated; full inputs and observations remain in trace.html and trace.json.
</desc>
<defs><marker id="arrow" markerWidth="7" markerHeight="7" refX="5" refY="3.5" orient="auto">
<path d="M0 0 L6 3.5 L0 7" fill="none" stroke="#94a3b8" stroke-width="1.4"/></marker></defs>
<style>
text{{font-family:"Microsoft YaHei","Noto Sans CJK SC",Arial,sans-serif;}}
.heading{{font-size:34px;font-weight:700;fill:#17263c;}}
.meta{{font-size:21px;fill:#64748b;}}
.label{{font-size:21px;fill:#56667c;}}
.value{{font-size:26px;font-weight:600;fill:#263951;}}
.card{{stroke-width:1.5;}}
.card.preparation{{fill:#f6f8fc;stroke:#dce3ee;}}
.card.action{{fill:#edf4ff;stroke:#8bb1f3;}}
.card.output{{fill:#ecf8f3;stroke:#91cbb9;}}
.value.action{{fill:#245ac4;}}
.value.output{{fill:#087b61;}}
.notice{{font-size:21px;fill:#926314;}}
</style>
<rect width="{width}" height="{height}" fill="#ffffff"/>
{body}
</svg>'''


def _render_html(trace: dict[str, Any], svg: str) -> str:
    details = []
    for tag, title, fields in _stages(trace):
        field_html = "".join(
            f"<h3>{html.escape(label)}</h3><pre>{html.escape(_display(value))}</pre>"
            for label, value in fields
        )
        details.append(
            f'<details class="stage"><summary><span>{html.escape(tag)}</span>'
            f"{html.escape(title)}</summary>{field_html}</details>"
        )
    metadata = html.escape(
        json.dumps(trace["metadata"], ensure_ascii=False, indent=2, allow_nan=False)
    )
    notice = _stub_notice(trace["metadata"])
    notice_html = f"<aside>{html.escape(notice)}</aside>" if notice else ""
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RS-Agent · intermediate_steps 轨迹</title>
<style>
:root{{color-scheme:light;font-family:system-ui,-apple-system,"Microsoft YaHei",sans-serif;
color:#14213d;background:#f3f6fb}}
*{{box-sizing:border-box}}
body{{max-width:1168px;margin:auto;padding:30px 24px 60px}}
h1{{font-size:30px;margin:0 0 10px}}p{{color:#526582;line-height:1.7}}
nav{{display:flex;gap:12px;flex-wrap:wrap;margin:22px 0}}a{{color:#174cb0}}
nav a{{padding:9px 15px;background:white;border:1px solid #d9e2ee;
border-radius:9px;text-decoration:none}}
aside{{background:#fff4d8;color:#855000;padding:16px 20px;border-radius:12px;
line-height:1.7;margin:18px 0}}
.diagram{{overflow:auto;border:1px solid #d9e2ee;border-radius:14px;background:#f3f6fb}}
svg{{display:block;width:100%;height:auto;min-width:720px}}
h2{{font-size:22px;margin:32px 0 15px}}
details.stage{{background:white;border:1px solid #d9e2ee;border-radius:12px;
padding:18px 22px;margin:12px 0}}
summary{{cursor:pointer;font-weight:650;overflow-wrap:anywhere}}
summary span{{display:block;font-size:12px;letter-spacing:.06em;color:#526582;margin-bottom:7px}}
h3{{font-size:13px;color:#526582;margin:20px 0 7px}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:14px;line-height:1.7;
background:#f6f8fc;padding:14px;border-radius:8px;max-height:600px;overflow:auto}}
.note{{font-size:13px}}
@media(max-width:600px){{body{{padding:20px 12px}}h1{{font-size:24px}}}}
</style></head><body>
<h1>RS-Agent · intermediate_steps 执行轨迹</h1>
<p>共记录 <strong>{len(trace["intermediate_steps"])}</strong> 次工具调用。
输入、任务类型推断和方案检索是独立的准备阶段；只有实际返回的工具调用列为 STEP。</p>
{notice_html}
<nav><a href="trace.svg" download>下载 SVG 轨迹图</a>
<a href="trace.json" download>下载完整 JSON</a>
<a href="#full-details">查看完整字段</a></nav>
<div class="diagram">{svg}</div>
<p class="note">图中长字段显示预览；下方可展开完整工具输入、观察值和最终输出。
action.log 保存在 JSON 文件中。</p>
<h2 id="full-details">完整字段</h2>
{"".join(details)}
<details class="stage"><summary>运行元数据</summary><pre>{metadata}</pre></details>
</body></html>"""


def export_trace(
    result: Mapping[str, Any],
    *,
    question: str,
    image_path: str | Path,
    output_dir: str | Path,
    metadata: Mapping[str, Any] | None = None,
    layout: str = "compact",
) -> dict[str, Path]:
    """Write ``trace.json``, ``trace.html``, and ``trace.svg`` for one run.

    ``result`` is the unmodified return value of ``RSAgent.run``. Each
    ``intermediate_steps`` item must be an ``(AgentAction, observation)`` pair.
    Set ``metadata={"stub_tools": True}`` when using the repository's stub tools
    so both visuals clearly identify observations as preset tool responses.
    HTML includes the diagram inline and works offline without dependencies.
    ``layout="compact"`` produces a horizontal presentation diagram; choose
    ``layout="detailed"`` for the vertical diagram with field previews.
    Both layouts preserve complete fields in the HTML panels and JSON.
    Existing artifact files in ``output_dir`` are replaced.

    Returns a mapping with ``json``, ``html``, and ``svg`` keys and Path values.
    Unsupported non-JSON values raise TypeError rather than losing information.
    """
    if layout not in {"compact", "detailed"}:
        raise ValueError("layout must be 'compact' or 'detailed'")
    steps = []
    for number, (action, observation) in enumerate(result.get("intermediate_steps", []), 1):
        if (
            isinstance(action, Mapping)
            or hasattr(action, "model_dump")
            or dataclasses.is_dataclass(action)
        ):
            action_data = _json_value(action)
        else:
            action_data = {
                "tool": _json_value(action.tool),
                "tool_input": _json_value(action.tool_input),
                "log": _json_value(action.log),
            }
        for required in ("tool", "tool_input", "log"):
            if required not in action_data:
                raise ValueError(f"intermediate_steps[{number - 1}].action lacks {required}")
        steps.append(
            {"step": number, "action": action_data, "observation": _json_value(observation)}
        )
    trace = {
        "schema_version": "1.0",
        "input": {"question": question, "image_path": str(image_path)},
        "metadata": _json_value(metadata or {}),
        "predicted_task_type": _json_value(result.get("predicted_task_type")),
        "guidance": _json_value(result.get("guidance")),
        "intermediate_steps": steps,
        "output": _json_value(result.get("output")),
    }
    json_text = json.dumps(trace, ensure_ascii=False, indent=2, allow_nan=False)
    svg_text = _render_compact_svg(trace) if layout == "compact" else _render_detailed_svg(trace)
    html_text = _render_html(trace, svg_text)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {kind: directory / f"trace.{kind}" for kind in ("json", "html", "svg")}
    for kind, content in (("json", json_text), ("html", html_text), ("svg", svg_text)):
        paths[kind].write_text(content + "\n", encoding="utf-8")
    return paths

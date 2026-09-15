# 运行样例并导出 intermediate_steps 轨迹

在项目环境已安装、`.env` 已配置 API 凭据的情况下，在仓库根目录运行：

```bash
uv run --no-sync examples/demo.py \
  --responses-api \
  --user-agent RS-Agent/0.1 \
  --question "What is the scene category of this image? Call scene once, then return its observation as the Final Answer." \
  --trace-dir outputs/trajectory-demo
```

该命令使用 `configs/default.yaml` 的模型和模式，默认输入为 `examples/sample.png`。
`.env` 的嵌入模型和设备设置仍会覆盖 YAML；也可以传入 `--image`、`--config`、
`--mode`。已有环境中使用 `--no-sync`，避免移除之前安装的可选依赖。

本机配置的兼容网关需要 Responses API，并对 SDK 默认 User-Agent 返回 403；
上面的两个选项分别选择接口协议和设置项目自己的客户端标识。
支持标准 Chat Completions 的服务可省略这两个选项。
同样的设置也可写入自定义 YAML 的 `llm` 节点：

```yaml
llm:
  provider: openai
  model: gpt-5.5
  temperature: 0
  use_responses_api: true
  default_headers:
    User-Agent: RS-Agent/0.1
```

上面是配置片段；使用 `--config` 时需保留完整 YAML 的其他节点。
Controller 使用仓库内的固定 structured-chat 提示模板，避免运行时依赖远程
Hub 提示词下载和反序列化。该模板不是论文历史 Hub prompt 的逐字副本。

## 输出文件

本次场景分类实测使用 `gpt-5.5`、`full` 模式和 CUDA 上的 `moka-ai/m3e-base`，
耗时约 39 秒（包含 Agent 初始化）。返回的任务类型为 `["Scene_Classification"]`，
`intermediate_steps` 只有一次 `scene` 调用，观察值和最终回答均为
`The scene of this image is airport.`；此文本来自 stub，见下方结果含义。

| 文件 | 内容 |
| --- | --- |
| `outputs/trajectory-demo/trace.html` | 可离线打开的轨迹图，含可展开的完整输入和返回值 |
| `outputs/trajectory-demo/trace.svg` | 默认导出横向精简轨迹图，适合插入 PPT；缩放不失真 |
| `outputs/trajectory-demo/trace.json` | 完整字段和运行元数据，包括每次工具调用的 `action.log` |

同一目录下再次导出会替换这三个文件。`outputs/` 已在 `.gitignore` 中。
HTML 内嵌 SVG，打开它不需要启动服务器；下载链接指向同目录下的 SVG 和 JSON。
本次额外将 SVG 渲染为 `outputs/trajectory-demo/trace.png` 以便直接预览；
demo 的常规导出格式为上述 HTML、SVG、JSON 三种。

默认 `--trace-layout compact` 将阶段压缩为横向卡片，仅显示关键摘要，
减少长问题、完整路径和检索内容占用的版面。需要原来的纵向详细预览时，
在运行命令中添加 `--trace-layout detailed`。两种布局均保留 HTML 展开面板
和 JSON 中的完整字段，布局选择不改变记录的轨迹数据。

图中的输入、任务类型推断和方案检索属于准备阶段，随后按实际执行顺序绘制
`intermediate_steps`，最后显示最终回答。只有工具调用才计入步骤编号。
每个步骤对应 `RSAgent.run()` 返回的 `(AgentAction, observation)`：

```json
{
  "step": 1,
  "action": {
    "tool": "scene",
    "tool_input": "/path/to/image.png",
    "log": "..."
  },
  "observation": "The scene of this image is airport."
}
```

上面仅展示字段结构；实际值以运行生成的 JSON 为准。SVG 会预览过长字段，
HTML 的展开面板和 JSON 保留完整内容。`action.log` 只保存在 JSON 中。

## 结果含义

demo 调用真实 LLM 和 Solution Space 检索，但默认工具来自
`rs_agent/toolkit/stubs.py`，返回预设文本。轨迹验证的是 Agent 的工具选择与
调用顺序，不代表图像已经完成去噪、超分或内容识别。

箭头表示执行先后，不代表工具之间传递了新生成的图像；stub 不产生处理后的
文件。最终回答引用的图像描述同样来自 stub。图中已明确标注这一点。

如需从 Python 导出已有运行结果，无需再次调用 LLM：

```python
from rs_agent.trace import export_trace

paths = export_trace(
    result,  # RSAgent.run(...) 的原始返回值
    question=question,
    image_path=image_path,
    output_dir="outputs/my-trace",
    metadata={"stub_tools": True},
    layout="compact",  # 默认值；"detailed" 使用纵向详细布局
)
```

也可以从已保存的 `trace.json` 重新导出精简图，无需再次调用 LLM：

```python
import json
from pathlib import Path

from rs_agent.trace import export_trace

trace = json.loads(Path("outputs/trajectory-demo/trace.json").read_text(encoding="utf-8"))
result = {
    "predicted_task_type": trace["predicted_task_type"],
    "guidance": trace["guidance"],
    "intermediate_steps": [
        (step["action"], step["observation"]) for step in trace["intermediate_steps"]
    ],
    "output": trace["output"],
}
export_trace(
    result,
    question=trace["input"]["question"],
    image_path=trace["input"]["image_path"],
    metadata=trace["metadata"],
    output_dir="outputs/trajectory-demo-compact",
    layout="compact",
)
```

导出器只使用 Python 标准库，不新增项目运行依赖。

## 多步样例的失败轨迹

本次还尝试了“先去噪，再 2× 超分，最后描述图像”的多步请求。当前网关和模型
组合连续调用 `denoising` 15 次，触发执行器的迭代上限，未完成预期三步任务。
这份实际运行结果保留在 `outputs/trajectory-demo/repeated-denoising/`，可打开
其中的 `trace.html` 检查每一次调用。它是失败轨迹，不应作为成功规划示例。

离线 SDK 验证和在线单次请求对照确认，后续请求包含前次工具的观察值；
仅凭这些证据还不能确定重复调用的具体原因。

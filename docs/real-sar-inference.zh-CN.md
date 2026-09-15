# 真实 SAR 工具与 P7 运行证据

`scripts/run_sar_agent.py` 将已训练的 SAR 检测器通过独立环境的 CLI 注册到现有
`RSAgent`。`rs_agent/toolkit/sar_detection.py` 不加载检测框架，不注册规划 stub；
旧 demo 和规划评测的默认行为保持原样。运行采用原有四种模式之一，默认 `full`。

检测器配置是本地可信 JSON，`command` 为参数数组。示例中的 checkpoint 由实际
检测脚本的配置决定，不使用 LLM 选择文件路径。工具输入只允许图像路径。

```json
{
  "default_model": "cascade",
  "models": {
    "cascade": {
      "display_name": "Cascade R-CNN + MSFA",
      "device": "cpu",
      "timeout_seconds": 600,
      "command": ["uv", "run", "--no-project", "--python",
        "/home/wyf/RS/midterm-exp/envs/mmdet/bin/python", "python",
        "/home/wyf/RS/midterm-exp/scripts/mmdet_infer.py", "--model", "cascade"]
    }
  }
}
```

CLI 接收额外参数 `--image IMAGE --output OUTPUT_DIR --device DEVICE`，必须输出
`result.json`，其 `detections` 为原图坐标的检测列表，`checkpoint` 标识权重，
`coco_json` 与 `visualization` 是已生成文件的绝对路径。还须标明
`actual_forward=true`、`cache_hit=false`、`image` 与 `display_score=0.30`，
适配器会核验其输入图像和展示阈值。每个框至少有
`bbox`（xywh）、`category_id`、`score`，可附 `class_name`。成功退出但缺失这些
内容视为失败，不转成固定成功字符串。

从仓库根目录运行（输出目录必须尚不存在）：

```bash
uv run --locked --no-sync python scripts/run_sar_agent.py \
  --detectors /absolute/path/agent_detectors.json \
  --image /absolute/path/image.jpg \
  --question "Please detect and count the targets in this SAR image, and provide the annotated image and prediction JSON." \
  --mode full --responses-api --user-agent RS-Agent/0.1 \
  --output /absolute/path/new_trace_directory
```

`--responses-api` 和 User-Agent 适用于本机已配置网关，其他 provider 按实际接口选择。
脚本读取项目 `.env`，只保存模型名、provider、endpoint host等非秘密字段；不会写出
完整配置或凭据。embedding 在本次进程固定使用 CPU，避免占用训练 GPU。

输出保留 `trace.json`（完整 action/observation）、`trace.html`、`trace.svg`、
`run_metadata.json`、`llm_messages.jsonl`（实际公开消息与 usage）和
`detector_calls/tool_calls.jsonl`。每次真实前向的子目录另存 stdout/stderr、完整
预测、图像及检测器结果。LLM 最终回答必须由这些工具观察值支持；完成验收需要至少
一次真实前向和成功最终回答。LLM 只读原图路径和检测结果，不获得 GT。

同一进程重复请求相同模型、规范图像路径、图像内容和 COCO ID 才会复用已完成结果，
并显式标为 `cache_hit`。不同路径/ID 的相同像素图像不会复用他者的身份或预测 JSON，
其耗时不解释为模型前向耗时。新的 CLI 进程分别记录冷启动和前向，跨环境端到端
耗时另存 `elapsed_seconds`。目前 P7 是单次真实链路证据，不代表 Agent 检测基准成绩，
也不代表离线使用 GT 的 F1-oracle 能力。

配置可附 `image_ids` 对象，用已核实的绝对图像路径映射到 COCO ID；无映射时固定传
`--image-id -1`，输出 `identity_scope=standalone_unindexed`，不能从文件名猜测。
此映射仅含图像身份，不含 GT 框或类别。每次新 Agent 运行在执行前保存所用源码快照与哈希。
LLM 客户端在构造时固定 120 秒超时、0 次 SDK 自动重试和 2048 输出 token 上限。
检测子进程有独立进程组；超时或中断会清理本次组内的进程，不操作其他任务。

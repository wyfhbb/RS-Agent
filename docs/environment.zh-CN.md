# 使用 uv 复现运行环境

本项目使用 `.python-version` 固定 Python 3.12.13，使用 `pyproject.toml` 声明依赖，
使用 `uv.lock` 锁定完整的版本、下载来源和文件哈希。选择 Python 3.12 作为验证基线，
库版本采用解析时可用的较新稳定版；这不是论文发表时的历史环境。

2026-09-14 在 NixOS/WSL2、RTX 5060 Ti 上验证的主要版本如下，完整版本以
`uv.lock` 为准：

| 组件 | 版本 |
| --- | --- |
| Python / uv | 3.12.13 / 0.11.21 |
| LangChain Classic / Core | 1.0.8 / 1.6.3 |
| langchain-openai / OpenAI SDK | 1.6.2 / 3.13.0 |
| PyTorch / CUDA 运行库 | 2.14.0 / 13.0 |
| NumPy / FAISS CPU | 2.5.3 / 1.15.0 |
| Transformers / Sentence Transformers | 5.17.0 / 6.0.1 |

## 安装与运行

先按 [uv 官方说明](https://docs.astral.sh/uv/getting-started/installation/)安装 uv，
然后在仓库根目录运行：

```bash
uv python install
uv sync --locked
uv run --locked pytest
```

uv 会创建 `.venv` 并以 editable 模式安装本项目，默认安装 `pytest`、`ruff` 开发工具。
不需要激活环境，也不需要设置 `PYTHONPATH`。只需要运行依赖时使用
`uv sync --locked --no-dev`，后续的 `uv run` 同样加 `--no-dev`。
`requirements.txt` 仅保留 pip 兼容入口，
不会重复维护另一份依赖清单；精确复现请使用锁文件。

实际调用 LLM 前，复制并填写配置：

```bash
cp .env.example .env
# 编辑 .env 中的 OPENAI_API_KEY 和可选的 OPENAI_API_BASE
uv run --locked examples/demo.py --question "Can you upscale this image?"
```

默认嵌入模型是 `moka-ai/m3e-base`，首次运行需要下载模型。Agent 使用仓库内
`rs_agent/controller/prompts.py` 的固定 structured-chat 模板，不再联网读取 Hub。
模型文件和外部 LLM 服务不受 Python 锁文件控制。更换嵌入模型后，需重建索引：

```bash
uv run --locked scripts/build_solution_index.py
```

项目默认工具是用于规划评测的 stub。真实遥感模型的权重和各自的推理环境，
以及 DualRAG 实验需要的数据、索引与外部服务，仍需按项目说明配置。

## 真实目标检测与分类

独立模型推理使用 `vision` 可选依赖：

```bash
uv sync --locked --extra vision
uv run --locked --extra vision scripts/probe_detection.py
uv run --locked --extra vision scripts/probe_classification.py
```

已实测 YOLOv8x-OBB、公开 EuroSAT ResNet-18/ViT-B/16，以及水平框 Faster R-CNN 控制实验。
详细结果、适用范围和自己的图像用法见 [模型实测报告](model-smoke-results.zh-CN.md)。
这些脚本直接加载权重，不需要 LLM API；Agent 中的原工具仍是 stub。
若还需要 Web 依赖，可使用 `uv sync --locked --extra web --extra vision` 同时安装。

## DualRAG

```bash
uv sync --locked --extra dualrag
uv run --locked --extra dualrag pytest
```

`dualrag` extra 安装当前仓库的 `dualrag/` 修改版，包名为 `lightrag-hku`，
不会安装 PyPI 上的另一个版本。后续需要它的命令继续带 `--extra dualrag`，
否则普通 `uv sync` 会移除未启用的可选依赖。

默认 JSON、NetworkX、NanoVectorDB 存储及 OpenAI/Ollama 客户端都已声明依赖，
导入这些模块不会再临时调用 pip 安装包。Ollama 客户端不包含 Ollama 服务或模型。
API 服务、GUI、其他数据库后端未纳入 `dualrag` extra；Web/API 使用下面的 `web` extra。

可选 node2vec 使用的 `graspologic` 3.4.4 要求 NumPy `<2`，因此不纳入本环境；
默认 RAG 流程不调用它。需要该方法时应另建兼容环境。
版本约束可见其 [PyPI 元数据](https://pypi.org/pypi/graspologic/3.4.4/json)。

在 `dualrag/` 子目录运行复现脚本时，用 `uv run --project .. --locked --extra dualrag`
显式选择根项目，详见 [DualRAG 说明](../dualrag/DUALRAG.md)。

## 启动 DualRAG Web 界面

在仓库根目录运行：

```bash
uv sync --locked --extra web
mkdir -p outputs/dualrag-web
PYTHON_DOTENV_DISABLED=1 LOG_DIR=outputs/dualrag-web \
  uv run --locked --extra web lightrag-server \
  --host 127.0.0.1 --port 9621 \
  --working-dir outputs/dualrag-web/storage \
  --input-dir outputs/dualrag-web/inputs
```

浏览器打开 <http://localhost:9621/webui/>，API 文档在 <http://localhost:9621/docs>。
前端构建文件已包含在仓库中，无需安装 Node 或重新构建。前台运行时按 `Ctrl+C` 停止。
已验证页面、JS/CSS、健康检查、文档列表、图谱标签和 OpenAPI 描述均返回 HTTP 200。

上述命令使用 Web 服务的默认配置：Ollama 地址 `http://localhost:11434`，
LLM 为 `mistral-nemo:latest`，嵌入模型为 `bge-m3:latest`、维度 1024。
没有 Ollama 服务也能打开页面；索引文档和问答需要先启动该服务并安装对应模型，
或者为 Web 服务配置其他支持的模型后端。

Web 服务使用自己的配置，不读取 `configs/default.yaml`。
`PYTHON_DOTENV_DISABLED=1` 禁止服务隐式读取主 Agent 的 `.env`，防止把本地
HuggingFace 模型名当成 Ollama 模型名。需要配置时，可复制专用模板：

```bash
cp dualrag/.env.web.example dualrag/.env
```

编辑 `dualrag/.env` 后，在上述 `uv run` 命令中增加 `--env-file dualrag/.env`。
uv 显式加载此文件；继续保留 `PYTHON_DOTENV_DISABLED=1`，使 Python 模块不再
读取其他 dotenv 文件。该配置中的 `LLM_BINDING_HOST`、`LLM_BINDING_API_KEY`
用于 LLM 服务，`EMBEDDING_BINDING_HOST`、`EMBEDDING_BINDING_API_KEY`
用于嵌入服务，两者可独立配置。使用 OpenAI 兼容服务时将对应 binding 改为 `openai`，
并设置服务实际支持的模型名和嵌入维度。

## GPU 与 NixOS/WSL

默认配置为 CPU 嵌入。Linux 的 PyTorch 安装包含 CUDA 用户态运行库，
使用 GPU 时将 `.env` 的 `EMBEDDING_DEVICE` 改为 `cuda`，并验证：

```bash
uv run --locked python -c 'import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.ones(3, device="cuda") * 2)'
```

FAISS 使用 `faiss-cpu`，嵌入模型的 GPU 加速不要求 GPU 版 FAISS。
PyTorch 的平台与加速器安装方式参见 [uv 官方指南](https://docs.astral.sh/uv/guides/integration/pytorch/)。

当前验证主机为 NixOS/WSL2，已配置 nix-ld 和 WSL 驱动库路径，无需更改系统。
若其他 WSL shell 中 CUDA 驱动不可见，可在单条命令前添加：

```bash
LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  uv run --locked python -c 'import torch; print(torch.cuda.is_available())'
```

其他 NixOS 主机仍需具备 nix-ld 和二进制 wheel 所需的系统库；这些属于主机配置，
不由 `uv.lock` 安装。

## 验证与升级

```bash
uv lock --check
uv pip check
uv run --locked --extra dualrag pytest
uv run --locked examples/demo.py --help
uv run --locked scripts/build_solution_index.py --help
uv run --locked benchmarks/planning/run_eval.py --help
```

测试不需要 API key 或在线模型：覆盖配置、规划评分、实际 Agent 工具执行、
FAISS 建库和已有索引读取、本地小型 BERT 的真实嵌入推理，以及可选 DualRAG
的存储读写和模拟 HTTP 响应下的 SDK 调用。离线测试通过不等同于外部 LLM、
Ollama 服务或完整论文实验已经运行。

日常安装使用 `--locked`，避免依赖自动漂移。主动升级时运行：

```bash
uv lock --upgrade
uv sync --locked --extra dualrag
uv run --locked --extra dualrag pytest
```

评审升级后的 `uv.lock` 与验证结果，再将它和依赖声明一起提交。LangChain 的旧式
structured-chat 执行器通过 `langchain-classic` 1.x 保留；工具、切分器和嵌入类
已迁入维护中的独立包，迁移依据是 [LangChain 官方指南](https://docs.langchain.com/oss/python/migrate/langchain-v1)。

# 遥感分类模型实测

2026-09-14，在本机 RTX 5060 Ti 上，两个公开 EuroSAT 分类模型均完成真实 CUDA 推理，无需重新训练。
它们是用于验证本机模型推理能力的替代模型，**不是 RS-Agent 作者的场景或飞机型号分类权重**。

| 模型 | 本次小样本 Top-1 标签匹配 | 单张前向中位耗时 | 任务 |
| --- | --- | --- | --- |
| ResNet-18 / EuroSAT | 10 / 10 | 1.59 ms | 10 类土地覆盖分类 |
| ViT-B/16 / EuroSAT | 10 / 10 | 7.50 ms | 10 类土地覆盖分类 |

样本来自模型发布者提供的 EuroSAT `test` 划分，按原始顺序固定取每类第一张，共 10 张，选择过程不依赖预测结果。
这只是安装、权重加载、预处理、GPU 推理和标签解码的冒烟测试，不能据此声称模型准确率为 100%。
耗时只统计预热后的 batch=1 模型前向，包含 CUDA 同步，不包含下载、预处理与文件保存；不是性能基准。
预测分数是 softmax 输出，没有做置信度校准。

环境：Python 3.12.13、PyTorch 2.14.0 / CUDA 13.0、torchvision 0.29.0、timm 1.0.29、Pillow 12.3.0、safetensors 0.8.0、PyArrow 25.0.1。

## 复现

在项目根目录运行：

```bash
uv sync --locked --extra vision
uv run --locked --extra vision scripts/probe_classification.py
```

第一次运行会下载约 398 MB 的固定版本数据与权重，此后复用 `outputs/model-smoke/classification/downloads/`。
脚本只加载 safetensors 权重，不执行远程模型代码；下载时验证权重和数据的 SHA-256，保留 TLS 证书校验。
有 `curl` 时优先使用它的网络配置并重试；否则使用 Python 标准库下载。

只测一个模型或自己的图像：

```bash
uv run --locked --extra vision scripts/probe_classification.py --model resnet18
uv run --locked --extra vision scripts/probe_classification.py \
  --image /path/to/satellite.jpg --output outputs/my-classification
```

可以重复传入 `--image`，用 `--device cpu` 指定 CPU，或用 `--samples-per-class 2` 扩展带标签样本。
自选图像没有标签时仅输出预测，不计算标签匹配。
这两个模型面向 Sentinel-2 RGB 土地覆盖图块；直接输入机场高分辨率图或 SAR 图并不能获得有效的飞机型号结果。

## 结果与来源

- [完整 JSON](../outputs/model-smoke/classification/results.json)：每张图的 SHA-256、数据行号、真实标签、Top-3 预测、前向耗时、预处理参数与环境。
- [结果预览图](../outputs/model-smoke/classification/classification-preview.png)：实际输入图与两个模型的预测。
- [可复用脚本](../scripts/probe_classification.py)。

模型来源是 [chathumal93/EuroSat-RGB-Classifiers](https://github.com/chathumal93/EuroSat-RGB-Classifiers)，模型作者说明其在 EuroSAT 上微调 ResNet 和 ViT。
实际下载版本固定如下，权重散列同时写在脚本与运行结果中：

| 资源 | 固定 revision |
| --- | --- |
| [cm93/resnet18-eurosat](https://huggingface.co/cm93/resnet18-eurosat) | `f416c040bab2a4ce61704eb8c8e07016722446f1` |
| [cm93/vit-base-patch16-224-eurosat](https://huggingface.co/cm93/vit-base-patch16-224-eurosat) | `0aafda0d3c84c3f44e9c2e9ee111d692444dc55a` |
| [cm93/eurosat](https://huggingface.co/datasets/cm93/eurosat) | `00095a028f821f0d1c8330808063718b6a71649f` |

支持的 10 类为 Forest、River、Highway、AnnualCrop、SeaLake、HerbaceousVegetation、Industrial、Residential、PermanentCrop、Pasture。
标签顺序从固定版本模型配置读取，并使用数据集元数据解码真实标签，未假定字母排序。

## 与原项目工具的关系

仓库 [README 的 Optical Analysis 和 SAR Analysis](../README.md#optical-analysis) 只描述了下列分类模型；
本地源码未包含对应推理实现、精确权重下载地址或可加载检查点，三个接口在 [stubs.py](../rs_agent/toolkit/stubs.py) 中仍返回固定文本。

| 原工具 | 原文描述 | 本次结论 |
| --- | --- | --- |
| `scene` | ViT-B/16，RSSDIVCS 微调，DINO-style | 未验证原权重；用公开 EuroSAT ViT-B/16 做替代验证 |
| `optical_plane_type` | 自定义微调 ResNet | 缺少飞机型号权重与类别定义；EuroSAT ResNet 不能替代此功能 |
| `sar_plane_type` | 自定义 SAR 飞机分类器 | 缺少实现、精确架构、权重与类别定义，未验证 |

本次没有改动 Agent 工具注册表或 stub，也没有调用 LLM API。
若要使用原项目的飞机型号识别，需要取得对应实现、训练时预处理、权重和类别映射，或选定其他已训练的飞机型号模型再接入。

# 目标检测与分类模型实测

2026-09-14，已在本机完成 4 个模型的真实 GPU 推理，使用下载的预训练权重，没有重新训练。
目标检测中，YOLOv8x-OBB 在项目遥感样图上给出了可用的车辆框；分类中，两个公开 EuroSAT 模型通过了小样本验证。
这验证的是独立模型推理，`examples/demo.py` 使用的 Agent 工具目前仍是 stub。

## 实测结论

| 模型与权重 | 与项目的关系 | 测试输入 | 结果 |
| --- | --- | --- | --- |
| YOLOv8x-OBB / 官方 DOTA 权重 | README 为 `optical_detection` 指定的模型；论文精确 checkpoint 未提供 | 项目样图 + DOTA8 的 4 张 val 图 | CUDA 成功；样图检出 19 个 `small vehicle`，目视框基本贴合车辆 |
| torchvision Faster R-CNN ResNet50 FPN v2 / COCO_V1 | 水平框替代基线；不是原项目未指定权重的 MMDetection 模型 | 同一项目样图 + 公交车控制图 | CUDA 成功；公交车/行人可检出，遥感样图却出现明显误检和漏检 |
| ResNet-18 / `cm93/resnet18-eurosat` | 公开遥感分类替代模型；不是飞机型号分类器 | EuroSAT test 每类第一张，共 10 张 | CUDA 成功，Top-1 标签匹配 10/10 |
| ViT-B/16 / `cm93/vit-base-patch16-224-eurosat` | 公开遥感分类替代模型；不是项目 RSSDIVCS 权重 | 同上 | CUDA 成功，Top-1 标签匹配 10/10 |

本次是少量样本的运行验证，**没有测出完整数据集的准确率或 mAP**。
分类样本在推理前按固定规则选取，没有按预测结果挑图；10/10 不能解释为总体准确率 100%。
两套分类模型识别的是森林、河流、道路、住宅区等 10 类土地覆盖，不能识别波音等飞机型号，
也不等同于 `land_use_classification` 所描述的逐像素分割服务。

## 复现与自己的图像

在项目根目录运行：

```bash
uv sync --locked --extra vision

# 光学遥感目标检测：默认测试项目样图及 4 张 DOTA8 图像
uv run --locked --extra vision scripts/probe_detection.py

# 土地覆盖分类：默认运行 ResNet-18 和 ViT-B/16
uv run --locked --extra vision scripts/probe_classification.py

# 水平框控制实验：通用 COCO 权重，不是遥感微调权重
uv run --locked --extra vision scripts/probe_horizontal_detection.py
```

第一次运行需要网络与 `curl`，下载的模型和数据总计约 715 MB，之后复用本地文件。
如果还要保留 DualRAG Web 依赖，安装命令用 `uv sync --locked --extra web --extra vision`，
运行命令也可以同时带上这两个 extra。
这些脚本不需要 API key，不读取 `.env` 或 `configs/default.yaml`，推理选项由命令行参数决定。

测试自己的图像时使用单独的输出目录，保留默认样本的验证结果：

```bash
uv run --locked --extra vision scripts/probe_detection.py \
  --source /path/to/aerial.png --output outputs/my-detection

uv run --locked --extra vision scripts/probe_classification.py \
  --image /path/to/sentinel2-rgb.jpg --output outputs/my-classification
```

检测可重复传入 `--source`，分类可重复传入 `--image`。
检测默认使用 GPU 0、`imgsz=1024`、置信度阈值 `0.25`；分类自动选择 CUDA 或 CPU。
没有 NVIDIA GPU 时给脚本加 `--device cpu`，CPU 性能未在本次测量。
权重、输入、结果 JSON 和图片保存在被 Git 忽略的 `outputs/model-smoke/`，不会提交到仓库。

## 查看实际结果

| 内容 | 图片 | 原始记录 |
| --- | --- | --- |
| YOLOv8x-OBB 的 19 个车辆框 | [项目样图检测结果](../outputs/model-smoke/detection/00_sample_annotated.png) | [检测 JSON](../outputs/model-smoke/detection/results.json) |
| Faster R-CNN 在遥感图上的误检 | [遥感样图结果](../outputs/model-smoke/horizontal/project_sample_annotated.png) | [水平框 JSON](../outputs/model-smoke/horizontal/report.json) |
| Faster R-CNN 的公交车/行人控制实验 | [控制图结果](../outputs/model-smoke/horizontal/bus_control_annotated.png) | 同上 |
| 两个分类模型的 10 张输入与预测 | [分类预览图](../outputs/model-smoke/classification/classification-preview.png) | [分类 JSON](../outputs/model-smoke/classification/results.json) |

上面的文件链接在运行脚本后可用。JSON 记录了权重来源、输入 SHA-256、类别、得分、框坐标或 Top-3 预测、环境和计时。
分类模型和数据固定了 Hugging Face revision；YOLO 权重和 DOTA8 压缩包固定并校验完整 SHA-256。
Faster R-CNN 权重通过官方文件名中的 SHA-256 前缀校验，并记录完整散列；控制图也记录了输入散列。

YOLO 在 DOTA8 验证图片上的预测数量如下：

| 图像 | 预测数量 | 标注数量 |
| --- | --- | --- |
| `P1470__1024__3296___1648.jpg` | 篮球场 3、足球场 1 | 篮球场 3、足球场 1 |
| `P1571__1024__2976___0.jpg` | 棒球场 1 | 棒球场 1 |
| `P1580__1024__824___824.jpg` | 棒球场 3 | 棒球场 2 |
| `P1724__1024__0___824.jpg` | 棒球场 1 | 棒球场 1 |

第三张多输出了一个棒球场框。这里仅比较类别数量，没有按 IoU 匹配实例；
数量相同不代表每个框都正确。DOTA8 是微型运行验证数据，也不能据此推断独立数据上的泛化能力。

Faster R-CNN 在 `confidence >= 0.5` 时把遥感样图中一大片建筑区域预测为 `truck`（0.634），
没有检出道路上的小车；控制图中检出了 4 个 person、1 个 bus、1 个 tie，公交车和主要行人框合理。
因此，这组 COCO 权重的推理环境可用，但不适合直接承担当前遥感样图的检测任务。

## 环境与验证

主机：NixOS/WSL2，NVIDIA GeForce RTX 5060 Ti，16 GB 显存。

| 组件 | 本次安装版本 |
| --- | --- |
| Python | 3.12.13 |
| PyTorch / torchvision | 2.14.0+cu130 / 0.29.0+cu130 |
| NumPy / timm | 2.5.3 / 1.0.29 |
| Ultralytics 无界面发行包 | `ultralytics-opencv-headless` 8.4.151 |
| OpenCV 无界面发行包 | `opencv-python-headless` 5.0.0.93 |
| Pillow / PyArrow | 12.3.0 / 25.0.1 |

原先尝试桌面版 OpenCV 时，导入报缺少 `libxcb.so.1`；已改用 Ultralytics 官方无界面发行包，
支持当前服务器上的推理与 PNG 保存。没有降级现有 PyTorch 或 NumPy。
该发行包用法见 [Ultralytics 安装说明](https://github.com/ultralytics/ultralytics/blob/main/docs/en/quickstart.md)。

记录的预热后耗时：YOLO 每图约 69–80 ms，Faster R-CNN 两张图约 60/69 ms；
ResNet-18 和 ViT-B/16 单张前向中位数分别约 1.59/7.50 ms。
各脚本计时范围不同，且为共享 GPU 上的小样本记录，不能用来直接比较模型速度。
具体计时口径在 JSON 和 [分类详细说明](classification-probe.md) 中。

新增依赖后，`uv lock --check`、`uv pip check` 和原项目的 14 项 pytest 测试均通过，
新增脚本通过 Ruff 检查。测试命令为：

```bash
uv run --locked --extra web --extra vision pytest -q
uv run --locked --extra vision ruff check \
  scripts/probe_detection.py scripts/probe_horizontal_detection.py scripts/probe_classification.py
```

## 尚未验证的原项目模型

| 原工具/框架 | 当前缺口 | 本次处理 |
| --- | --- | --- |
| `horizontal_object_detection` / MMDetection | README 未指定精确配置和 checkpoint；旧版 MMCV 算子环境未安装 | 用 torchvision Faster R-CNN 做替代基线，未宣称 MMDetection 已运行 |
| `rotated_object_detection` / MMRotate | 有公开模型权重，但稳定版依赖旧版 `mmcv-full` / MMDetection | 没有在当前新 PyTorch 环境中编译运行；旋转框功能由 YOLOv8x-OBB 实测 |
| `sar_detection` / DiffDet4SAR | 未找到公开的训练后 SAR checkpoint；上游配置指向作者机器的 `model_final.pth` | 未运行，不以通用初始化权重替代 SAR 权重 |
| `scene` / RSSDIVCS ViT-B/16 | 本仓库没有对应权重和完整推理实现 | 用公开 EuroSAT ViT-B/16 替代验证 |
| `optical_plane_type` / 自定义 ResNet | 缺飞机型号权重、类别映射和预处理定义 | 未验证飞机型号识别；EuroSAT ResNet 仅验证土地覆盖分类 |
| `sar_plane_type` / 自定义 SAR 分类器 | 缺明确架构、实现、权重及类别映射 | 未验证 |

MMDetection 3.3 与 MMRotate 0.3.4 所需的 MMCV 主版本不同，不能直接装进同一套旧框架环境。
本次检查未找到匹配本机 PyTorch 2.14 / CUDA 13.0 的现成旧版 MMCV wheel；
不代表这些框架不能使用，但需要独立兼容环境或源码适配，尚未完成运行验证。
依据：[MMDetection 版本要求](https://github.com/open-mmlab/mmdetection/blob/main/docs/en/notes/faq.md)、
[MMRotate 版本要求](https://github.com/open-mmlab/mmrotate/blob/main/docs/en/faq.md)、
[MMRotate 公开旋转框模型](https://github.com/open-mmlab/mmrotate/tree/main/configs/oriented_rcnn)。

DiffDet4SAR 的 checkpoint 缺口依据其 [配置文件](https://github.com/JoyeZLearning/DiffDet4SAR/blob/master/configs/diffdet.coco.res50.300boxes.yaml)
与 [运行说明](https://github.com/JoyeZLearning/DiffDet4SAR/blob/master/GETTING_STARTED.md)；后续如取得权重，可继续验证。
其他已测模型来源：[YOLOv8 官方模型](https://docs.ultralytics.com/models/yolov8/)、
[torchvision Faster R-CNN](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn_v2.html)、
[EuroSAT 分类权重及固定版本](classification-probe.md#结果与来源)。

有现成任务权重的模型可直接推理，不需要为这次运行测试训练。
若目标是飞机具体型号或 SAR 类别，需要先取得相应已训练权重；只有缺少合适权重，
或在自己的有标签数据上效果不足时，才考虑训练或微调。

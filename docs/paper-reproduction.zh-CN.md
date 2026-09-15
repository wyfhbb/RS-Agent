# 论文结果的复现准备情况

审计日期：2026-09-14。对照与本项目期刊 DOI 对应的
[arXiv v4](https://arxiv.org/html/2406.07089v4)（2026-07-07）及
[期刊补充材料](https://media.springernature.com/original/springer-static/esm/art%3A10.1007%2Fs11432-026-5026-5/MediaObjects/11432_2026_5026_MOESM1_ESM.pdf)。

当前可以推进部分实验的对照复测，还不能声称完整复现了论文数字。
前一轮的 [模型运行测试](model-smoke-results.zh-CN.md) 验证了真实推理能力；
正式比较还需要相同的输入、权重、预处理和评分协议。

## 优先关注的目标检测与分类

| 论文实验 | 论文报告结果 | 当前材料与缺口 |
| --- | --- | --- |
| DOTA-v1 目标计数（表 3） | 精确计数准确率 33.30%；区间匹配率 75.98%；相对误差 0.28 | 同名 YOLOv8x-OBB 的官方 DOTA 权重已跑通；已取得全部 val 标注并重建候选问题，缺完整原图和作者推理参数 |
| RSSDIVCS 场景分类（表 5） | 98.00% | 缺作者微调后的 ViT-B/16 checkpoint、训练/测试图像清单和完整训练配置 |
| AID / UCMerced 场景分类（表 5） | 96.88% / 98.63% | 同样缺原分类权重，以及与训练类别对应的映射和评测细节 |

表中数字是论文目标值，**不是本机新测得的结果**。
论文目标计数使用 458 张图、1,099 个问题；分类描述为 ImageNet 自监督预训练的 ViT-B/16
在 RSSDIVCS 上微调，采用 80% 训练比例。
依据：[目标计数协议](https://arxiv.org/html/2406.07089v4#S4.SS3.SSS1)、
[场景分类协议](https://arxiv.org/html/2406.07089v4#S4.SS3.SSS3)。

本次已经从 DOTA 官方 Drive 下载原始 validation 标注 `labelTxt-v1.0/labelTxt.zip`，
仅 575,403 字节，SHA-256 为
`97518c9125070d9e3e76fab3e69b53fde7a623b2207c82be4832c5e8f9abbd0e`。
独立统计得到 458 个标注文件、28,853 个对象。
按“每图实际出现的类别各生成一题，并计入 difficult=1 对象”重建，**恰好得到 1,099 题**。
这支持了题目构造规则的推断，但还不能证明与作者原题逐条一致。
若排除难例，正答案的图像/类别组合只有 1,057 个；沿用 1,099 个组合则产生 42 个零答案。

已保存 [候选计数题集](../outputs/paper-reproduction-audit/dota/questions.inferred.jsonl)、
[来源与构造规则](../outputs/paper-reproduction-audit/dota/questions.inferred.manifest.json)、
[逐图统计](../outputs/paper-reproduction-audit/dota/annotation-statistics.json)。
每题同时保留包含/排除难例的两种真值。本次没有下载完整图像，也没有执行全量模型评测。

目标计数是当前最接近可复测的实验。仍需补齐或明确：

1. 从 [DOTA 官方页面](https://captain-whu.github.io/DOTA/dataset.html) 获取 **v1.0 validation 原图**，
   与已下载的标注对应。DOTA8 的裁剪图不能替代该验证集。
2. 确认作者选择的图像与目标类别组合；自行按标注生成的问题需要标为重建题集，不能自动视为论文原题。
3. 固定 checkpoint 哈希、图像缩放/切片/合并策略、置信度和去重阈值、困难目标是否计入。
4. 按论文口径计算计数指标。相对误差公式的分母含真实数量，真实数量为零时的处理须单独明确；
   不能自行加常数后仍称完全相同的指标。
5. 区分直接读取检测框数量的模型基线，与经过 Agent 选工具、汇总回答后的系统结果。

目标计数已有可用权重，不需要为了运行而训练。
场景分类应先取得作者权重；如果取得不到，可以训练自己的模型做独立对照，
但需要重建训练流程，也无法提前保证上述百分比。
此前实测的 EuroSAT ResNet/ViT 是土地覆盖分类替代模型，不能拿它们的 10 张样本结果与表 5 比较。

## Agent 规划结果与已找回的题集

README 中 18 项任务的百分比是**工具选择准确率**。
现有 [评分函数](../benchmarks/planning/score.py) 判断第一个调用工具是否匹配标签，
不会检查目标框、飞机型号或分割图是否正确。
因此 `optical_plane_type` 等工具即使仍为 stub，也可能在规划评测中得分。

虽然当前工作树没有 `data/eval/`，本次已从官方 Git 历史找回一份公开题集：

- 官方首次加入提交：`7c6a484215f9d622e44ab3a8681e8d781aad062f`。
- [官方历史文件](https://github.com/IntelliSensing/RS-Agent/blob/7c6a484215f9d622e44ab3a8681e8d781aad062f/data/eval/questions_planning.jsonl)。
- 原始文件共 360 条，18 类各 20 条，问题和 ID 均不重复。
- 官方 raw 文件与本地 Git blob 逐字节一致，SHA-256 为
  `508213061150bb2465e4e20db2bfbfbf17637841f8ee623c759277d298ddf711`。
- 已原样保存到 [本地候选题集](../outputs/paper-reproduction-audit/recovered/questions_planning.jsonl)，
  [来源与统计记录](../outputs/paper-reproduction-audit/recovered/questions_planning.provenance.json) 包含核验信息。

它可作为有来源的复测候选集，但还没有证据确认它就是论文实验实际使用的同一批题目。
题目未改写，也没有恢复进已删除的 `data/eval/`；恢复副本位于 Git 忽略的 `outputs/`。

正式跑规划评测之前还有两个具体问题：

- 这份数据的 20 条光学检测题标为 `Object_Detection`，而当前 RS-Agent 的
  [任务映射](../rs_agent/toolkit/registry.py) 使用 `Optical_Detection`。
  当前评分直接查表会把这 20 条全部判错，即便工具调用正确，理论最高也只有 `340/360 = 94.44%`。
  需要补齐别名兼容并保留 RS-ChatGPT 原有映射；本次审计未修改评分代码。
- 当前 [默认配置](../configs/default.yaml) 使用 `gpt-5.5`，不是论文默认的 Qwen2.5-32B-Instruct。
  应按目标表格选定同一模型，并记录提示模板版本与实际模型服务参数，再比较结果。

当前脚本还只覆盖单工具评分。多工具顺序正确率与最终答案成功率需要各自的数据和评分实现，
不能用单工具结果替代。
本次只恢复和核查数据，没有执行新的 LLM 规划评测。

## 其他实验的材料状况

DualRAG 本地有 61 个去重后的 `mix` context 和 125 个固定问题，能继续准备图谱检索对照实验。
`mix` 是 LongBench 混合语料，不能直接当成遥感专业知识库。
目前缺实验索引、完整各方法输出，以及文档提及但未附带的 `LightRAG_old/` baseline。
已有两份 30 题答案没有模型或检索模式来源记录，不能直接当作论文原始结果。

其评测是 LLM 成对偏好评分；当前查询失败后省略条目、评分按位置 `zip` 配对的逻辑，
以及重复追加 JSON 数组的写入方式，都需要在正式实验前修复。
依据：[DualRAG 说明](../dualrag/DUALRAG.md)、
[查询脚本](../dualrag/reproduce/Step_3.py)、[评审脚本](../dualrag/datasets/eval.py)。

实际推进顺序建议为：**DOTA 目标计数 → 获取原场景分类权重并核对划分 → 规划评测与消融**。
如果需要向作者索取材料，优先索取计数问题清单和推理配置、场景分类权重和划分、
规划题集版本及逐条原始预测；这些比单独增加 GPU 算力更能解决当前的复现缺口。

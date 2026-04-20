# Project1 EEG 项目实验全程说明

更新时间：2026-04-20

这份文档面向两类读者：

- 没直接参与这个项目，但想快速知道“这个项目到底在做什么、已经做到哪一步”的人
- 愿意继续接手实验的人，希望知道哪些路线有效、哪些路线已经被排除

如果你只想先看结论，可以直接跳到“**先给最终结论**”一节。  
如果你想看完整流水，**正式主账本**在 [EXPERIMENT_LOG.md](/home/xiaoh/DeepLearning/project1_eeg/EXPERIMENT_LOG.md)。截至这份报告更新时，主账本里一共有 56 条实验记录。  
[EXPERIMENT_LOG_hpc.md](/home/xiaoh/DeepLearning/project1_eeg/EXPERIMENT_LOG_hpc.md) 不是第二本正式账本，它只是 2026-04-20 从远端 HPC 工作目录拉回来的镜像副本，用来保留 A800 节点上的原始记录。

## 项目在做什么

这个项目的目标可以用一句话概括：

**输入一段脑电信号（EEG），让模型推断被试看到了什么图像。**

项目里实际上有两条任务线：

1. **Retrieval（检索）**
   - 给模型一段 EEG，再给它一批候选图片。
   - 模型不需要“生成”图片，而是要在候选图片里找出最可能对应的那张。
   - 这更像一个“多选题”。
2. **Reconstruction（重建 / 生成）**
   - 给模型一段 EEG，让它自己生成一张图。
   - 这更像一个“简答题”或者“画图题”。

为什么要把这两条线分开？

- Retrieval 的难点是“语义匹配是否对”
- Reconstruction 的难点是“能不能真的画出像样的东西”
- 一个模型可能很会做多选题，但不会真的画图

### 数据是什么样的

根据本地数据文件 [README.md](/home/xiaoh/DeepLearning/project1_eeg/README.md) 和 `image-eeg-data/*.pt`：

- 训练图像目录共有 `16540` 张 JPG：`training_images`
- 测试图像目录共有 `200` 张 JPG：`test_images`
- `train.pt` 里的 EEG 形状是 `(16540, 4, 63, 250)`
  - 可以粗略理解成：每张训练图片对应 4 次 EEG 记录
  - 每次 EEG 有 63 个通道、250 个时间点
- `test.pt` 里的 EEG 形状是 `(200, 80, 63, 250)`
  - 可以粗略理解成：每张测试图片对应 80 次 EEG 记录

另外还有一个非常关键、几乎决定了后面 reconstruction 设计方向的事实：

**训练图片和测试图片的类别不重叠。**

这句话的意思是：

- 训练集里学到的是一批训练类别
- 测试集里出现的是另一批新类别
- 所以“从训练图里找一张最近的图然后稍微改一改”这类方法，天然就有很大风险

### 这个项目的现实限制

- 我手上没有正式的官方 evaluation 服务
- 真正能依赖的只有：
  - 项目 PDF
  - 样例代码
  - 本地 train/test 数据
- 所以目前所有判断，都是基于**本地验证集**和**本地 200-way 测试集**做的

## 先给最终结论

如果你只看这一节，应该能抓住整个项目到目前为止最重要的结论。

1. **Retrieval 这条线已经基本跑通了，而且结果是可靠的。**
   当前最好 checkpoint 是 [retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt](/home/xiaoh/DeepLearning/project1_eeg/outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt)，本地 200-way test 结果是：
   - `top1 = 0.3450`
   - `top5 = 0.6250`

2. **最早的 prototype / residual-VAE reconstruction 虽然某些量化指标看起来不差，但生成图在肉眼上基本没有语义。**
   也就是说：它“在指标上像成功”，但“在图像上像失败”。

3. **Kandinsky decoder 不是当前首要瓶颈。**
   我们做过一个非常重要的 sanity check：直接用 ground-truth Kandinsky image embedding 去解码，`val64` 上能达到：
   - `eval_clip = 0.9973`
   - `eval_alex5 = 0.9824`
   这说明 decoder 上限很高，当前更大的问题仍然在于“EEG 到 embedding 这一步还不够准”。

4. **当前最值得保留的 reconstruction 主线，是“EEG -> Kandinsky image embedding -> Kandinsky decoder”。**
   在这条线上，当前最好配置不是最慢的“高质量采样”，而是更快的 `predicted_v4_fast`。

5. **在当前已经做过本地 decode 评估的配置里，最好的 reconstruction 结果是 `kandinsky_predicted_v4_fast`。**
   在 `val64` 上：
   - `eval_clip = 0.7078`
   - `eval_alex5 = 0.8204`
   在 full validation 上：
   - `eval_clip = 0.6933`
   - `eval_alex5 = 0.7993`

6. **在当前 `val64` 对比里，“直接检索一张最像的训练图，再把它当生成条件”不是更优路线。**
   同样在 `val64` 上，`retrieval_top1` 条件版本只有：
   - `eval_clip = 0.6612`
   低于连续回归的 `predicted_v4_fast`。

7. **最近尝试的 `rag_residual v1` 在当前实现下没有打赢 baseline。**
   问题不是脚本没跑通，而是模型结构学歪了：
   - residual gate 几乎塌到了 0
   - 模型最后退化成了“几乎只靠 direct regression”
   - 因此这条线没有打赢当前 best

8. **2026-04-20 在 A800 上完成的 masked EEG pretraining 迁移实验，在当前 proxy-selected regression 设定下没有带来正向收益。**
   - 预训练目标本身能收敛
   - 但把这个 encoder 迁移到当前 Kandinsky embedding reconstruction 主线时，proxy 指标显著弱于 `v4_proxyselect`
   - 即使改成“先冻 encoder、再小学习率解冻”的 staged finetune，也没有把它拉回 baseline 水平

9. **从项目决策角度看，当前最重要的下一步不是优先换 decoder，也不是优先继续沿用当前这套 masked-pretrain 初始化，而是继续提高 EEG 到语义 embedding 的建模。**

## 一个容易混淆但必须先说清楚的点

[EXPERIMENT_LOG.md](/home/xiaoh/DeepLearning/project1_eeg/EXPERIMENT_LOG.md) 里的 “Current Best Reconstruction” 仍然指向旧的 `prototype_topk4` 测试结果。  
这并不意味着那是我们现在最认可的主线。

原因是：

- 那个 banner 更接近“账本自动记录到的最好量化测试值”
- 但项目决策时，我们更看重“量化指标 + 肉眼可辨识度 + 是否符合数据设定”

所以这份报告里提到的“当前最佳 reconstruction 主线”，指的是：

**从研究判断上最值得继续推进的方案，而不是账本里自动统计的历史 best banner。**

## 实验是怎么一步步推进的

下面按阶段讲整个项目是怎么演化到现在的。

### 阶段 A：先把样例代码和最初 baseline 跑起来

这一阶段做的事情很朴素：

- 先把环境配好
- 先理解样例代码的默认工作流
- 先确认 retrieval 和 reconstruction 两条线都能跑通

项目最初的默认思路是：

- retrieval：EEG 对齐到 CLIP 或类似视觉特征
- reconstruction：先从训练图片里找原型，再在 VAE latent 上做 residual 修正

这一阶段的价值不在“成绩好”，而在“把整个工程链路跑通”。  
如果没有这一步，后面很多改进其实都无从谈起。

### 阶段 B：先把 retrieval 做强，因为它是后面 reconstruction 的基础

最开始我试过几类 retrieval 设计：

| Retrieval 版本 | 验证集最佳结果 | 结论 |
|---|---:|---|
| `retrieval_clip_only_atm_small` | `val_top1=0.0333`, `val_top5=0.1179` | 只用 CLIP image feature 很弱 |
| `retrieval_clip_dreamsim_atm_small` | `val_top1=0.1258`, `val_top5=0.2908` | 比 CLIP-only 强，但最终权重仍偏向 DreamSim |
| `retrieval_dreamsim_only_atm_small` | `val_top1=0.0611`, `val_top5=0.1632` | 早期版本效果一般 |
| `retrieval_dreamsim_only_atm_small_fixed` | `val_top1=0.1348`, `val_top5=0.3065` | 当前 retrieval 主线 |

这里面最关键的发现是：

1. **DreamSim 这类更“人类感知对齐”的特征，比单纯 CLIP 更适合这个 retrieval 任务。**
2. 在 `clip + dreamsim` 的混合版本里，最后选出来的最佳 `alpha` 其实是 `0.0`，说明模型最终几乎完全依赖 DreamSim。
3. 修正后的 DreamSim-only retrieval 是后面 reconstruction 的最好 backbone。

最终，这条最好 retrieval 模型在本地 200-way test 上得到：

- `top1 = 0.3450`
- `top5 = 0.6250`

对应的关键文件是：

- checkpoint: [best.pt](/home/xiaoh/DeepLearning/project1_eeg/outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt)
- test 指标: [retrieval_metrics.json](/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/test_eval/retrieval_metrics.json)

这一阶段的结论是：

**后续 reconstruction 应该建立在 DreamSim retrieval backbone 之上。**

### 阶段 C：prototype / residual-VAE reconstruction 的探索、诊断和放弃

拿到一个比较强的 retrieval backbone 之后，最自然的想法是：

- 先从训练图片里找到几个最相似原型
- 再在 latent 空间里学一个 residual
- 希望“检索 + 小修正”能得到正确图像

这条线最强的旧 baseline 是 `reconstruction_dreamsim_topk4`。  
它在本地 test 上的指标是：

- `eval_clip = 0.5386`
- `eval_ssim = 0.4326`
- `eval_pixcorr = 0.2136`

对应文件在：

- [reconstruction_metrics.json](/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/reconstruction_dreamsim_topk4/seed_0/test_predictions/reconstruction_metrics.json)

乍看这些数字，好像不差。尤其是：

- `SSIM` 很高
- `pixcorr` 也不低

但这一阶段最重要的不是数字，而是**肉眼检查**。  
我实际看过生成图之后，发现它们大多数只是一些：

- 低信息量纹理
- 模糊块状结构
- 看不出明确物体类别的图

也就是说，这条线的真实问题是：

**模型学会了“像一张图”，但没学会“像正确的那类图”。**

后来又结合数据进一步诊断，发现问题根源其实很清楚：

- reconstruction 这条线拿到的 prototype bank 来自 `training_images`
- 但 test 里的类别与 train 不重叠
- 所以“从训练图里找原型”这件事，本质上已经偏离了目标

这也是为什么这条线后来虽然保留为 baseline，但不再继续作为主路线。

这一阶段的结论是：

**以训练图原型为锚点的 prototype-based reconstruction，在这个项目的数据设定下，不像一个有希望的长期主线。**

### 阶段 D：转向 Kandinsky image embedding reconstruction

既然“先找训练原型再修正”不靠谱，就需要换思路：

- 不再把训练图片原型当作生成锚点
- 改成让模型直接从 EEG 预测一个可用于生成的图像 embedding
- 再把这个 embedding 送进生成模型去解码

这里选用的是 Kandinsky 的 image embedding 路线。

这一阶段我连续试了多个版本：

| 版本 | 验证 best | 结论 |
|---|---|---|
| `reconstruction_kandinsky_embed_v1` | `val_total_loss=1.4063` | 第一版能跑通，但还只是起点 |
| `reconstruction_kandinsky_embed_v2_metricreg` | `val_total_loss=1.6188` | 加的 regularization 没带来提升 |
| `reconstruction_kandinsky_embed_v3_smallhead` | `val_total_loss=1.6434` | small head 更差 |
| `reconstruction_kandinsky_embed_v4_proxyselect` | `val_subset_top1=0.0272`, `val_subset_top5=0.0846` | 当前最好训练主线 |

这一轮里最关键的改进，不只是模型结构，而是**怎么选 best checkpoint**。

早期版本更多是按 `val_loss` 看；
到了 `v4_proxyselect`，开始用更贴近最终任务的 proxy 指标来选 checkpoint，比如：

- `val_subset_top1`
- `val_subset_top5`
- `avg_target_cosine`

这一步很重要，因为 reconstruction 的 `loss` 好，不一定代表“生成图更像人想要的结果”。

当前这条线最重要的训练 checkpoint 是：

- [reconstruction_kandinsky_embed_v4_proxyselect/seed_0/best.pt](/home/xiaoh/DeepLearning/project1_eeg/outputs/experiments/reconstruction_kandinsky_embed_v4_proxyselect/seed_0/best.pt)

它的 best metrics 是：

- `epoch = 19`
- `val_subset_top1 = 0.0272`
- `val_subset_top5 = 0.0846`
- `avg_target_cosine = 0.6156`

这一阶段的结论是：

**“直接从 EEG 回归连续的 Kandinsky image embedding”比“离散地检索一张训练图 embedding”更有前途。**

### 阶段 E：验证 decoder 上限，并对生成参数做 sweep

这一阶段要回答两个非常关键的问题：

1. 现在生成图不够好，到底是 EEG 模型不行，还是 decoder 不行？
2. 在已经有一个不错的 embedding predictor 后，decoder 应该用“快配置”还是“慢配置”？

#### 先回答第一个问题：decoder 到底是不是瓶颈？

为此我做了一个很重要的 sanity check：

- 不用 EEG 预测 embedding
- 直接把真实图片对应的 Kandinsky image embedding 拿来解码

在 `val64` 上，这个 ground-truth 条件解码的结果几乎接近满分：

- `eval_clip = 0.9973`
- `eval_alex5 = 0.9824`
- `eval_inception = 0.9995`

对应文件：

- [reconstruction_metrics.json](/home/xiaoh/DeepLearning/project1_eeg/outputs/eval_compare/val64_seed0/kandinsky_groundtruth_local/reconstruction_metrics.json)

这说明：

**Kandinsky decoder 会画图。当前更大的问题看起来不在 decoder，而在 EEG 预测出来的 embedding 还不够准。**

#### 再回答第二个问题：decoder 参数怎么选？

随后我在固定的 `val64` 子集上做了几组对比：

| 配置 | `eval_clip` | `eval_alex5` | 结论 |
|---|---:|---:|---|
| `retrieval_top1_local` | `0.6612` | `0.7981` | 离散检索不如连续预测 |
| `predicted_v4_local`（40 steps） | `0.6917` | `0.7946` | 已经比 retrieval_top1 好 |
| `predicted_v4_fast`（20 steps） | `0.7078` | `0.8204` | 当前最好 |
| `predicted_v4_quality`（60 steps, 8 candidates） | `0.6992` | `0.7840` | 更慢，但没更好 |

这里最反直觉但很重要的结论是：

**更慢、更“高质量”的采样设置，并没有带来更好的结果。**

当前最优 decoder 点反而是：

- 20 steps
- 4 candidates
- guidance 4.0

也就是 `kandinsky_predicted_v4_fast`。

对应文件：

- `val64`: [reconstruction_metrics.json](/home/xiaoh/DeepLearning/project1_eeg/outputs/eval_compare/val64_seed0/kandinsky_predicted_v4_fast/reconstruction_metrics.json)
- `full-val`: [reconstruction_metrics.json](/home/xiaoh/DeepLearning/project1_eeg/outputs/eval_compare/full_val_seed0/kandinsky_predicted_v4_fast/reconstruction_metrics.json)

在 full validation 上，它的结果是：

- `eval_clip = 0.6933`
- `eval_alex5 = 0.7993`
- `eval_pixcorr = 0.0848`

这一阶段的结论是：

1. **decoder 不是首要瓶颈**
2. **在当前 `val64` 对比里，continuous predicted embedding 优于 retrieval_top1**
3. **当前最好的生成配置是 `predicted_v4_fast`**

### 阶段 F：最近的 RAG residual 尝试，以及为什么它失败了

在 Kandinsky 主线基本稳定后，我又尝试了一条更“学术上听起来合理”的设计：

- 保留一条 direct regression 路径
- 再让 retrieval 提供 top-k 语义上下文
- 用一个 gate 决定 residual 要加多少

这条线就是 `rag_residual`。

#### 先做 smoke，确认代码链路没问题

这一步做了两件事：

1. 1 个 epoch 的训练 smoke
2. `val8_smoke` 的生成 smoke

中途还修了一次离线加载问题：

- 一开始 `local_files_only=True` 时，decoder 仍然用 HF repo id 加载
- 本地没有完整快照，所以失败
- 后来加了本地目录 fallback，这个问题才解决

修完之后，smoke prediction 是能跑通的，说明：

- 新模型能训练
- 新推理脚本能生成
- 新 metadata 也能落盘

#### 正式的 `rag_residual v1`

正式训练 run 是：

- [reconstruction_kandinsky_rag_residual_v1/seed_0](/home/xiaoh/DeepLearning/project1_eeg/outputs/experiments/reconstruction_kandinsky_rag_residual_v1/seed_0)

它完整跑了 30 个 epoch。

best checkpoint 在 `epoch 22`，关键指标是：

- `val_subset_top1 = 0.0254`
- `val_subset_top5 = 0.0792`
- `avg_target_cosine = 0.6377`

表面上看，它并没有烂到不能用；但和当前主线 `v4_proxyselect` 比较：

- `v4_proxyselect`: `val_subset_top1 = 0.0272`, `val_subset_top5 = 0.0846`
- `rag_residual v1`: `val_subset_top1 = 0.0254`, `val_subset_top5 = 0.0792`

也就是说，它没有赢。

更关键的是结构诊断：

- `avg_gate` 约等于 `8e-6`
- 这几乎可以视为“gate 全关”
- 结果就是 residual 分支没有真正发挥作用

换句话说：

**这条模型最后退化成了“几乎只做 direct regression”，retrieval residual 基本没被用起来。**

因此我没有把这条线继续升到 `val64/full-val`。

这一阶段的结论是：

**`rag_residual v1` 跑通了工程链路，但没有证明这条结构在当前写法下真的有效。问题的核心是 gate collapse。**

### 阶段 G：A800 上的 masked EEG pretraining 迁移实验，以及为什么这条线暂时放弃

在 `rag_residual` 没有打赢当前主线之后，我又做了一条看起来很合理的路：

1. 先在 EEG 上做 masked temporal reconstruction pretraining
2. 再把这个 pretrained encoder 拿去初始化 Kandinsky embedding regression
3. 看它是否能提升当前 `v4_proxyselect` 主线

这组实验是在 HKUST-GZ HPC 的 A800 节点上完成的，主要有三条。

需要特别说明的是：

- 这两条 downstream transfer run 为了节省 A800 时间，只做了 embedding proxy 评估
- 没有在同一轮里做 decoder/image-level 终评
- 所以下面这部分结论，应理解为“在当前 proxy-selected regression 设定下没有带来收益”，而不是“所有形式的 pretraining 都已经被彻底证伪”

#### 第一步：先看 pretrain 本身能不能收敛

预训练 run 是：

- [eeg_mask_pretrain_v1/seed_0](/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/eeg_mask_pretrain_v1/seed_0)

它跑了 20 个 epoch，最终最好结果是：

- `val_total_loss = 0.6766`

这说明：

**masked EEG pretraining 作为一个自监督目标，本身是能正常收敛的。**

但这一步只能说明“预训练任务能学起来”，还不能说明“它对 reconstruction 一定有帮助”。

#### 第二步：直接把 pretrained encoder 接到当前 reconstruction 主线

直接迁移的 run 是：

- [reconstruction_kandinsky_embed_v5_mask_pretrain_a800/seed_0](/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/reconstruction_kandinsky_embed_v5_mask_pretrain_a800/seed_0)

这个实验里，encoder checkpoint 是正确加载的：

- `missing_keys = 0`
- `unexpected_keys = 0`

所以如果结果不好，原因不是“权重没读进去”。

它的最好结果出现在 `epoch 22`，关键指标是：

- `val_subset_top1 = 0.0006`
- `val_subset_top5 = 0.0048`
- `avg_target_cosine = 0.6150`
- `val_total_loss = 1.5449`

和当前最好训练主线 `v4_proxyselect` 比较：

- `v4_proxyselect`: `val_subset_top1 = 0.0272`, `val_subset_top5 = 0.0846`
- `v5_mask_pretrain_a800`: `val_subset_top1 = 0.0006`, `val_subset_top5 = 0.0048`

这不是“小幅退步”，而是：

**直接迁移在当前 proxy 指标上表现很差。**

#### 第三步：做一个更保守的 staged finetune，看是不是优化节奏问题

为了排除“不是 pretrain 错，而是 finetune 方法太猛”的可能性，我又补了一条更保守的版本：

- 前 5 个 epoch 冻结 encoder，只训练回归头
- 之后再解冻 encoder
- 并把 encoder 学习率降到 `3e-5`

这个 run 是：

- [reconstruction_kandinsky_embed_v6_mask_pretrain_staged_a800/seed_0](/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/reconstruction_kandinsky_embed_v6_mask_pretrain_staged_a800/seed_0)

它的最好结果出现在 `epoch 11`，关键指标是：

- `val_subset_top1 = 0.0030`
- `val_subset_top5 = 0.0073`
- `avg_target_cosine = 0.6170`
- `val_total_loss = 1.5288`

这个结果比直接迁移版略好，但仍然远低于 `v4_proxyselect`：

- `v4_proxyselect`: `val_subset_top1 = 0.0272`, `val_subset_top5 = 0.0846`
- `v6_mask_pretrain_staged_a800`: `val_subset_top1 = 0.0030`, `val_subset_top5 = 0.0073`

并且后半程还出现了：

- `val_total_loss` 持续变差
- `avg_target_cosine` 继续下滑

也就是说，它并没有把这条 pretrain 路线救回来。

#### 这一阶段最终说明了什么

这一组 A800 实验把问题说明得很清楚：

1. **当前的 masked EEG pretraining 目标能收敛。**
2. **但它学到的表征并没有顺利迁移到当前的 Kandinsky embedding regression 主线。**
3. **直接迁移失败了。**
4. **更保守的 staged finetune 也没有打赢 baseline。**

因此这一阶段的结论是：

**“当前写法下的 masked-pretrain encoder 初始化”在现有 proxy-selected regression 设定下没有带来收益，因此不应作为 reconstruction 主线的优先方向继续推进。**

## 除了模型本身，还做了哪些工程支撑

如果只看模型名字，很容易忽视中间其实做了不少“为了让实验可跑、可比较、可复现”的基础工作。

这些工作包括：

1. **统一实验账本**
   - 所有正式训练、评估、失败尝试、debug 过程都写进 [EXPERIMENT_LOG.md](/home/xiaoh/DeepLearning/project1_eeg/EXPERIMENT_LOG.md)
   - 这样后面回顾时，不会只剩“我记得当时好像跑过”
   - [EXPERIMENT_LOG_hpc.md](/home/xiaoh/DeepLearning/project1_eeg/EXPERIMENT_LOG_hpc.md) 只是从远端拉回的临时镜像，不应视作第二本正式账本

2. **固定子集评估**
   - 做了 `val8_smoke`、`val16_panel`、`val64` 这类固定子集
   - 这样小实验之间才有可比性

3. **本地离线模型加载**
   - Kandinsky decoder 做了本地目录 fallback
   - 避免每次都被 `local_files_only` 卡住

4. **full-val evaluation 的 batch 化修复**
   - 之前 full validation 在某些 vision metric 上会因为整批过大而崩
   - 后来把评估改成分 batch 跑，full-val 才能稳定完成

5. **text bank、pretrain 脚本与 HPC 迁移支撑**
   - text bank、masked EEG pretraining、HPC 环境同步链路都已经跑通
   - 其中 masked pretraining 还在 A800 上完成了一轮正式迁移验证
   - 但结果表明它在当前写法下没有带来收益，所以暂时降级，而不是升为下一轮主线

## 关键实验总表

下面这张表不是把全部流水账逐条重抄，而是“最值得记住的关键里程碑”。

| 阶段 | 实验 / 文件 | 主要结果 | 一句话结论 |
|---|---|---|---|
| Retrieval 主线 | `retrieval_dreamsim_only_atm_small_fixed` | test `top1=0.3450`, `top5=0.6250` | 当前最可靠的 retrieval backbone |
| 旧 reconstruction baseline | `reconstruction_dreamsim_topk4` | test `eval_clip=0.5386`, `eval_ssim=0.4326` | 指标看起来不差，但图像没有语义 |
| Kandinsky 训练主线 | `reconstruction_kandinsky_embed_v4_proxyselect` | `val_subset_top1=0.0272`, `val_subset_top5=0.0846` | 当前最好训练 checkpoint |
| sanity check | `kandinsky_groundtruth_local` | `val64 eval_clip=0.9973` | decoder 不是主要瓶颈 |
| discrete retrieval 条件 | `kandinsky_retrieval_top1_local` | `val64 eval_clip=0.6612` | 离散 top1 不如连续预测 |
| 当前最好 reconstruction | `kandinsky_predicted_v4_fast` | `val64 eval_clip=0.7078`, full-val `0.6933` | 当前最值得保留的 reconstruction 主线 |
| 慢速高采样对照 | `kandinsky_predicted_v4_quality` | `val64 eval_clip=0.6992` | 更慢不代表更好 |
| 新结构尝试 | `reconstruction_kandinsky_rag_residual_v1` | best `val_subset_top1=0.0254`, gate 近 0 | 跑通了，但没赢，也没真正用上 residual |
| A800 预训练 | `eeg_mask_pretrain_v1` | best `val_total_loss=0.6766` | 预训练目标本身能收敛 |
| A800 直接迁移 | `reconstruction_kandinsky_embed_v5_mask_pretrain_a800` | best `val_subset_top1=0.0006`, `val_subset_top5=0.0048` | 在当前 proxy 指标下明显弱于 baseline |
| A800 保守迁移 | `reconstruction_kandinsky_embed_v6_mask_pretrain_staged_a800` | best `val_subset_top1=0.0030`, `val_subset_top5=0.0073` | 比直接迁移略好，但在当前 proxy 指标下仍远弱于 `v4_proxyselect` |

## 最重要的数字应该怎么读

这一节非常重要，因为这个项目里最容易出错的地方就是“把不该比较的数字放在一起比较”。

### 1. Retrieval 指标

- `top1_acc`
  - 模型给出最可能的一张图，是否正好答对
- `top5_acc`
  - 模型给出的前 5 张候选里，是否包含正确答案

这两个指标主要回答的是：

**模型会不会做“多选题”。**

### 2. Reconstruction 指标

重建这边常见的几个指标是：

- `eval_clip`
  - 更偏向语义是否对得上
- `eval_alex5`
  - 也是常用的感知相似度指标
- `eval_ssim`
  - 更偏像素结构层面的相似度
- `eval_pixcorr`
  - 更偏低层像素相关性

这些指标很有用，但不能单独看。

最典型的例子就是旧的 `prototype_topk4`：

- `eval_ssim = 0.4326`
- `eval_pixcorr = 0.2136`

看起来很漂亮，但图其实没有明确物体语义。  
所以这个项目里：

**高 SSIM / 高 pixcorr，不一定代表“生成得对”。**

### 3. 为什么 retrieval 的 test accuracy 不能拿来当 reconstruction 的代理

因为它们回答的是两个完全不同的问题：

- Retrieval 问的是：
  - “在给定候选图里，你会不会选对？”
- Reconstruction 问的是：
  - “你能不能自己画出一张像样的图？”

一个模型可能很会从 200 张候选里选图，但并不会真的生成正确类别的图。

### 4. 为什么 ground-truth reconstruction 只能当上限参考

`kandinsky_groundtruth_local` 这个实验不是在测 EEG 模型，而是在测：

**如果 embedding 已经是对的，decoder 能画到什么程度？**

所以它的意义是：

- 证明 decoder 很强
- 证明瓶颈主要在 EEG -> embedding

但它不是最终任务成绩，不能直接说“项目已经做到 0.9973”。

## 当前最佳可用方案

如果现在要把项目交给别人继续做，我会保留两条主线。

### A. Retrieval 当前最佳方案

- 模型：`retrieval_dreamsim_only_atm_small_fixed`
- checkpoint: [best.pt](/home/xiaoh/DeepLearning/project1_eeg/outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt)
- test 结果：
  - `top1 = 0.3450`
  - `top5 = 0.6250`

为什么保留它：

- 这是目前最稳定、最清楚的成功线
- 后续 reconstruction 的 backbone 也应该从它出发

### B. Reconstruction 当前最佳方案

- 训练 checkpoint：`reconstruction_kandinsky_embed_v4_proxyselect`
- 生成配置：`predicted_v4_fast`
  - `20 steps`
  - `4 candidates`
  - `guidance 4.0`

关键结果：

- `val64`：
  - `eval_clip = 0.7078`
  - `eval_alex5 = 0.8204`
- full validation：
  - `eval_clip = 0.6933`
  - `eval_alex5 = 0.7993`

为什么保留它：

- 它是当前“语义最好 + 实际可跑 + 速度也合理”的 reconstruction 主线
- 它明显优于 `retrieval_top1` 条件生成
- 它也比 prototype/VAE 路线更符合当前数据设定

## 已经排除或降级的路线

这一节是给后续接手的人省时间用的。

### 1. Prototype-based reconstruction

不继续的原因：

- train/test 类别不重叠
- 用训练图原型去服务测试图，本身就不对题
- 肉眼看生成图几乎没有清晰物体语义

### 2. Residual VAE mainline

不继续的原因：

- 虽然一些低层指标好看
- 但生成图仍然更像纹理拼贴，而不是正确物体

### 3. 把 retrieval top1 直接当生成条件

不继续作为主线的原因：

- `val64 eval_clip = 0.6612`
- 明显弱于连续预测的 `predicted_v4_fast`

### 4. `rag_residual v1`

不继续的原因：

- 没有超过当前 best
- `avg_gate` 接近 0，说明 residual 分支实际没被用起来
- 现阶段继续同配置长训没有意义

## 下一步最值得做什么

当前最合理的顺序不是“乱开更多实验”，而是按因果关系推进。

### 优先级 1：继续提升 EEG 表征，但不宜优先继续沿用当前这套 masked-pretrain 初始化

当前最明确的事实有两个：

- `rag_residual` 的 gate collapse 还没解决
- 当前这套 masked-pretrain 初始化在现有 proxy regression 设定下也没有带来收益

所以真正要解决的是：

- 怎样让 EEG encoder 学到更适合下游语义回归的表示
- 而不是优先继续复用这套已经表现不佳的 masked reconstruction 目标

### 优先级 2：修 residual gate collapse，或者换一个更合理的 auxiliary objective

如果还要继续做结构增强，最值得追的是：

- 为什么 gate 学成了几乎全关
- 怎么让 retrieval residual 真正参与 final embedding
- 或者干脆换一个比 masked temporal reconstruction 更贴近下游目标的辅助学习信号

在这个问题没解决之前，继续长训同一类 `rag_residual` 或 masked-pretrain 变体，性价比都不高。

### 优先级 3：再尝试 text aux / text context

这条线有潜力，但要放在 gate 问题之后：

- 因为目前 `concept_text` 本质上只是类别词，不是完整 caption
- 所以它更像辅助语义，而不是主生成条件

### 优先级 4：如果以后再回到 pretraining，必须换目标或换数据规模

现在至少可以明确说：

- 不是“还没来得及测 masked pretraining”
- 而是“当前这套实现已经测过，而且在现有 proxy 设定下没打赢 baseline”

所以如果未来要重新做 pretraining，至少应该满足下面之一：

- 换一个更贴近下游语义回归的预训练目标
- 引入更大规模或更多样的 EEG 语料
- 明确设计一套不会把有用表征在 finetune 初期冲掉的迁移方案

## 附录：重要文件和目录地图

如果你要快速继续看这个项目，优先从下面这些地方开始。

### 文档和账本

- 项目说明：[README.md](/home/xiaoh/DeepLearning/project1_eeg/README.md)
- 正式实验账本：[EXPERIMENT_LOG.md](/home/xiaoh/DeepLearning/project1_eeg/EXPERIMENT_LOG.md)
- 远端镜像归档：[EXPERIMENT_LOG_hpc.md](/home/xiaoh/DeepLearning/project1_eeg/EXPERIMENT_LOG_hpc.md)

### 当前最好 retrieval

- checkpoint: [best.pt](/home/xiaoh/DeepLearning/project1_eeg/outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt)
- test 指标: [retrieval_metrics.json](/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/test_eval/retrieval_metrics.json)

### 当前最好 reconstruction 训练 checkpoint

- [best.pt](/home/xiaoh/DeepLearning/project1_eeg/outputs/experiments/reconstruction_kandinsky_embed_v4_proxyselect/seed_0/best.pt)

### 当前最好 reconstruction 的 `val64` 对比结果

- [reconstruction_metrics.json](/home/xiaoh/DeepLearning/project1_eeg/outputs/eval_compare/val64_seed0/kandinsky_predicted_v4_fast/reconstruction_metrics.json)

### 当前最好 reconstruction 的 full validation 结果

- [reconstruction_metrics.json](/home/xiaoh/DeepLearning/project1_eeg/outputs/eval_compare/full_val_seed0/kandinsky_predicted_v4_fast/reconstruction_metrics.json)

### 旧 prototype / VAE baseline

- [reconstruction_metrics.json](/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/reconstruction_dreamsim_topk4/seed_0/test_predictions/reconstruction_metrics.json)

### 最近失败的 RAG residual 尝试

- [best.pt](/home/xiaoh/DeepLearning/project1_eeg/outputs/experiments/reconstruction_kandinsky_rag_residual_v1/seed_0/best.pt)

### 最近失败的 A800 masked-pretrain 迁移尝试

- 预训练 checkpoint：[encoder_best.pt](/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/eeg_mask_pretrain_v1/seed_0/encoder_best.pt)
- 直接迁移：[best.pt](/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/reconstruction_kandinsky_embed_v5_mask_pretrain_a800/seed_0/best.pt)
- 保守 staged 迁移：[best.pt](/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/reconstruction_kandinsky_embed_v6_mask_pretrain_staged_a800/seed_0/best.pt)

---

如果只用一句话总结整个项目到现在的状态，那就是：

**检索已经做出了一个可靠版本；重建已经从“看似有指标、其实没语义”的旧 baseline，推进到了“有一定语义、但还不够准”的 Kandinsky embedding 主线；2026-04-20 的 A800 实验进一步说明，当前这套 masked-pretrain 初始化至少在现有 proxy-selected regression 设定下没有把这条主线救起来，下一步真正该解决的仍然是 EEG 到高层语义表示的建模质量。**

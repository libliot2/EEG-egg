# Project1 EEG 项目实验全程说明

更新时间：2026-04-21

这份文档面向两类读者：

- 没直接参与这个项目，但想快速知道“这个项目到底在做什么、已经做到哪一步”的人
- 愿意继续接手实验的人，希望知道哪些路线有效、哪些路线已经被排除

如果你只想先看结论，可以直接跳到“**先给最终结论**”一节。  
如果你想看完整流水，**正式主账本**在 [EXPERIMENT_LOG.md](/home/xiaoh/DeepLearning/project1_eeg/EXPERIMENT_LOG.md)。这份报告只抽取其中真正影响路线决策的关键实验。  
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

3. **Kandinsky decoder 本身不是首要瓶颈，但“只靠纯 embedding、从噪声开始解码”也不是最终答案。**
   我们做过一个非常重要的 sanity check：直接用 ground-truth Kandinsky image embedding 去解码，`val64` 上能达到：
   - `eval_clip = 0.9973`
   - `eval_alex5 = 0.9824`
   这说明 decoder 的语义能力很强。  
   但后面的 img2img 实验又进一步说明：如果没有一个低层结构锚点，纯 embedding decode 的 `SSIM` 和 `pixcorr` 仍然偏弱，所以真正有效的做法是把“高层语义条件”和“低层图像起点”结合起来。

4. **当前最值得保留的 reconstruction 主线，已经从“纯 Kandinsky embedding decode”升级为“retrieval prototype init + predicted Kandinsky embedding 的 Kandinsky img2img”。**
   也就是：
   - retrieval backbone 先给出一个训练图 prototype，作为低层结构起点
   - EEG predictor 再给出连续的 Kandinsky image embedding，作为高层语义条件
   - Kandinsky decoder 从这个 prototype 加噪后的 latent 开始做 img2img denoise

5. **在目前已经跑完 `full test` 的配置里，最好的 reconstruction 结果是 `hpc_img2img_v4_s20_c4_g4p0_str035`。**
   这个配置使用：
   - 训练 checkpoint：`reconstruction_kandinsky_embed_v4_proxyselect`
   - decoder：Kandinsky img2img
   - init image：retrieval prototype
   - sampling：`20 steps`, `4 candidates`, `guidance 4.0`, `strength 0.35`

   它在 `val64` 上的结果是：
   - `eval_clip = 0.7106`
   - `eval_ssim = 0.3505`
   - `eval_pixcorr = 0.0938`

   它在本地 200-way `test` 上的结果是：
   - `eval_clip = 0.7513`
   - `eval_ssim = 0.3767`
   - `eval_pixcorr = 0.1567`
   - `eval_alex5 = 0.8489`

6. **最近做的 SDXL pivot 证明了“prototype init + 现代 diffusion decoder”这类思路是可行的，但当前版本没有打赢 Kandinsky img2img 主线。**
   `sdxl_turbo_proto_text_s4_g0p0_str050` 在本地 200-way `test` 上得到：
   - `eval_clip = 0.6940`
   - `eval_ssim = 0.3642`
   - `eval_pixcorr = 0.1238`
   它作为 sidecar 有研究价值，但目前不是提交主线。

7. **最近尝试的 `rag_residual v1` 在当前实现下没有打赢 baseline。**
   问题不是脚本没跑通，而是模型结构学歪了：
   - residual gate 几乎塌到了 0
   - 模型最后退化成了“几乎只靠 direct regression”
   - 因此这条线没有打赢当前 best

8. **2026-04-20 在 A800 上完成的 masked EEG pretraining 迁移实验，在当前 proxy-selected regression 设定下没有带来正向收益。**
   - 预训练目标本身能收敛
   - 但把这个 encoder 迁移到当前 Kandinsky embedding reconstruction 主线时，proxy 指标显著弱于 `v4_proxyselect`
   - 即使改成“先冻 encoder、再小学习率解冻”的 staged finetune，也没有把它拉回 baseline 水平

9. **新加的一条 `CLIP` target regression 小实验说明：目标表征的选择值得继续探索。**
   `reconstruction_clip_embed_v1_proxyselect` 的 best proxy 已经达到：
   - `val_subset_top1 = 0.0369`
   - `val_subset_top5 = 0.1022`
   这比当前 Kandinsky target 训练主线 `v4_proxyselect` 的 proxy 更高。  
   但因为这条线还没接上一个比现主线更强的 decoder 路径，所以它现在只能算“有潜力的下一步”，还不是可直接替换的主线。

10. **从项目决策角度看，当前最重要的下一步已经不是盲目再开新大分支，而是围绕现有 `Kandinsky img2img` 主线做提交级完善，并在这个框架里继续提升语义条件。**

## 一个容易混淆但必须先说清楚的点

[EXPERIMENT_LOG.md](/home/xiaoh/DeepLearning/project1_eeg/EXPERIMENT_LOG.md) 里的 “Current Best Reconstruction” banner 可能会滞后于最新 HPC 结果。  
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
3. **在“纯 embedding decode”这条子路线里，当前最好的生成配置是 `predicted_v4_fast`**

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

### 阶段 H：把 prototype 重新放回生成链路，但只让它负责低层结构锚点

前面阶段 C 之所以放弃 prototype-based reconstruction，是因为“把训练图 prototype 当成最终答案”这件事不对。  
但这不代表 prototype 完全没用。

后面重新梳理这个问题时，我把它拆成了两个层次：

- **高层语义**：由 EEG predictor 回归出来的连续 Kandinsky image embedding 决定
- **低层结构**：由 retrieval backbone 找到的 prototype 图来提供一个合理的初始图像

这就得到了一条新的 img2img 路线：

1. 用当前最好的 retrieval backbone 找到一个 top-1 prototype
2. 把这张 prototype 图送入 Kandinsky img2img decoder
3. 用 `reconstruction_kandinsky_embed_v4_proxyselect` 预测出的 image embedding 做 conditioning
4. 通过 `strength` 控制“保留多少 prototype 低层结构、覆盖多少新语义”

这条路线最重要的对比，不是“prototype 有用没用”，而是：

**prototype 单独当答案没用，但 prototype 当低层锚点是有用的。**

我先在 `val8_smoke` 上确认链路可跑，然后在 `val64` 上做了三档 `strength` sweep：

| 配置 | `eval_clip` | `eval_ssim` | `eval_pixcorr` | 结论 |
|---|---:|---:|---:|---|
| `str035` | `0.7106` | `0.3505` | `0.0938` | 当前最好平衡点 |
| `str050` | `0.7019` | `0.3136` | `0.0597` | 语义和结构都略退 |
| `str065` | `0.6979` | `0.2964` | `0.0670` | 再往上也没有变好 |

这里最重要的现象是：

- 相比纯 decode 的 `kandinsky_predicted_v4_fast`，`str035` 的 `eval_clip` 没有明显变差
- 但 `eval_ssim` 从 `0.0509` 大幅抬到了 `0.3505`
- `eval_pixcorr` 也从 `0.0873` 提到了 `0.0938`

换句话说：

**把 prototype 放回 img2img 起点后，我们第一次在不明显牺牲语义的前提下，把低层结构指标拉了回来。**

随后我把这条最优配置升到本地 200-way `test`：

- 配置名：`hpc_img2img_v4_s20_c4_g4p0_str035`
- `eval_clip = 0.7513`
- `eval_ssim = 0.3767`
- `eval_pixcorr = 0.1567`
- `eval_alex5 = 0.8489`

这也是到目前为止，**真正有 full-test 证据支持的 reconstruction 最佳主线**。

### 阶段 I：并行验证 SDXL sidecar 和 CLIP-target 语义分支

在确定“prototype init + semantic condition”这类思路可行之后，我又并行看了两件事：

1. decoder 能不能从 Kandinsky 换到 SDXL
2. 语义 target 能不能从 Kandinsky embedding 换到更标准的 CLIP embedding

#### I.1 SDXL img2img feasibility branch

这条线的目标不是立刻换主线，而是验证：

**如果把当前 retrieval prototype 和一个轻量文本 prompt 一起送进 SDXL Turbo img2img，会不会比 Kandinsky 更强？**

我先修了两类工程问题：

- 远端脚本同步路径错误
- retrieval bank 里 prototype 图像路径写的是旧的本地绝对路径，需要按 `image_id` 在当前数据目录里重新映射

脚本跑通之后，在 `val64` 上的最好点是 `str050`：

- `eval_clip = 0.6845`
- `eval_ssim = 0.3574`
- `eval_pixcorr = 0.1253`

这个结果说明：

- SDXL 这条 sidecar 不是跑不通
- 它甚至在 `SSIM/pixcorr` 上是有竞争力的
- 但它在语义指标上明显弱于 Kandinsky img2img

最后把最优点升到本地 200-way `test` 后，结果是：

- 配置名：`sdxl_turbo_proto_text_s4_g0p0_str050`
- `eval_clip = 0.6940`
- `eval_ssim = 0.3642`
- `eval_pixcorr = 0.1238`
- `eval_alex5 = 0.8155`

它没有打赢 `hpc_img2img_v4_s20_c4_g4p0_str035`，因此结论很明确：

**SDXL pivot 目前只适合作为 sidecar 研究分支，不适合作为当前提交主线。**

#### I.2 CLIP target regression 小规模验证

前面的 Kandinsky target 路线已经说明：

- 这条思路能跑通
- 但 target 本身是不是最好，仍然值得怀疑

所以我补了一条更简单的语义 target 实验：

- 模型：`reconstruction_clip_embed_v1_proxyselect`
- 目标：直接从 EEG 回归 `CLIP` image embedding

这条线的 best proxy 是：

- `val_subset_top1 = 0.0369`
- `val_subset_top5 = 0.1022`

对比当前 Kandinsky target 的训练主线 `v4_proxyselect`：

- `v4_proxyselect`: `val_subset_top1 = 0.0272`, `val_subset_top5 = 0.0846`
- `clip_embed_v1`: `val_subset_top1 = 0.0369`, `val_subset_top5 = 0.1022`

这说明：

**从“语义可回归性”这个角度看，CLIP target 可能比 Kandinsky target 更容易学。**

但这条线目前还没有接上一个完整、稳定、并且超过 Kandinsky img2img 主线的 decoder 路径，所以现在只能下一个稳妥结论：

**CLIP target 是下一阶段最值得保留的语义分支，但它还没有形成新的提交级主线。**

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
| 纯 decode 最好点 | `kandinsky_predicted_v4_fast` | `val64 eval_clip=0.7078`, `eval_ssim=0.0509`; full-val `eval_clip=0.6933`, `eval_ssim=0.1836` | 纯 embedding decode 里的最好配置 |
| 慢速高采样对照 | `kandinsky_predicted_v4_quality` | `val64 eval_clip=0.6992` | 更慢不代表更好 |
| Kandinsky img2img 主线 | `hpc_img2img_v4_s20_c4_g4p0_str035` | test `eval_clip=0.7513`, `eval_ssim=0.3767`, `eval_pixcorr=0.1567` | 当前 overall 最好 reconstruction 主线 |
| SDXL sidecar | `sdxl_turbo_proto_text_s4_g0p0_str050` | test `eval_clip=0.6940`, `eval_ssim=0.3642` | 可行，但没打赢 Kandinsky img2img |
| 新结构尝试 | `reconstruction_kandinsky_rag_residual_v1` | best `val_subset_top1=0.0254`, gate 近 0 | 跑通了，但没赢，也没真正用上 residual |
| A800 预训练 | `eeg_mask_pretrain_v1` | best `val_total_loss=0.6766` | 预训练目标本身能收敛 |
| A800 直接迁移 | `reconstruction_kandinsky_embed_v5_mask_pretrain_a800` | best `val_subset_top1=0.0006`, `val_subset_top5=0.0048` | 在当前 proxy 指标下明显弱于 baseline |
| A800 保守迁移 | `reconstruction_kandinsky_embed_v6_mask_pretrain_staged_a800` | best `val_subset_top1=0.0030`, `val_subset_top5=0.0073` | 比直接迁移略好，但在当前 proxy 指标下仍远弱于 `v4_proxyselect` |
| 新语义 target 探针 | `reconstruction_clip_embed_v1_proxyselect` | best `val_subset_top1=0.0369`, `val_subset_top5=0.1022` | CLIP target 更容易回归，值得保留 |

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

如果现在要把项目交给别人继续做，我会保留一个 retrieval backbone、一个 reconstruction 主线，再加一个次级研究分支。

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
- 生成配置：`hpc_img2img_v4_s20_c4_g4p0_str035`
  - `Kandinsky img2img`
  - init image 来自 retrieval top-1 prototype
  - `20 steps`
  - `4 candidates`
  - `guidance 4.0`
  - `strength 0.35`

关键结果：

- `val64`：
  - `eval_clip = 0.7106`
  - `eval_ssim = 0.3505`
  - `eval_pixcorr = 0.0938`
- 本地 200-way `test`：
  - `eval_clip = 0.7513`
  - `eval_ssim = 0.3767`
  - `eval_pixcorr = 0.1567`
  - `eval_alex5 = 0.8489`

为什么保留它：

- 它是当前唯一一个同时兼顾高层语义和低层结构、并且已经用 full test 验证过的 reconstruction 主线
- 它比纯 decode 的 `predicted_v4_fast` 显著提高了 `SSIM`
- 它也打赢了 SDXL sidecar 的 full-test 对照

### C. 次级研究分支

- `reconstruction_clip_embed_v1_proxyselect`

为什么保留它：

- 在训练 proxy 上，它已经超过当前 Kandinsky target 主线
- 它说明“换 target 表征”这件事可能比继续堆复杂结构更有价值
- 但它还没有接上一个稳定优于 `Kandinsky img2img` 的 decoder 路径，所以暂时不升为提交主线

## 已经排除或降级的路线

这一节是给后续接手的人省时间用的。

### 1. Prototype-based reconstruction

不继续的原因：

- train/test 类别不重叠
- 把训练图原型直接当最终答案，本身就不对题
- 肉眼看生成图几乎没有清晰物体语义

但要特别补一句：

- **prototype 作为 img2img 的低层结构锚点要保留**
- 被放弃的是“prototype 当独立 reconstruction 主线”，不是“prototype 在整个系统里彻底没用”

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

### 5. 当前写法下的 masked-pretrain 初始化

不继续作为近期主线的原因：

- 已经在 A800 上做过正式迁移验证
- 无论直接迁移还是 staged finetune，都明显弱于 `v4_proxyselect`
- 在没有新目标函数或新迁移策略之前，继续沿这条线堆算力的性价比很低

## 下一步最值得做什么

当前最合理的顺序不是“重新把所有分支都开一遍”，而是围绕已经打赢 full test 的主线继续推进。

### 当前进行中的实验

截至这份文档更新并准备同步到 GitHub 时，当前已经正式提交到 HKUST-GZ HPC、但还在排队中的实验有两条：

1. `retrieval_dreamsim_only_atm_base_v1`
   - 目标：验证把 retrieval backbone 从 `atm_small` 升到 `atm_base` 后，是否能直接提高本地 200-way retrieval 上限
   - 当前状态：`PENDING (Priority)`
   - 关键改动：`encoder_type=atm_base`，并把 checkpoint selection 改成更鲁棒的 `blend_top1_top5`

2. `retrieval_dreamsim_only_atm_base_ides_v1`
   - 目标：验证 IDES-style 的随机 trial averaging 是否能进一步提高 retrieval 上限
   - 当前状态：`PENDING (Priority)`
   - 关键改动：在同样的 `atm_base` backbone 上，训练时随机从 4 个 trial 中抽取 `k=2..4` 个做平均，而不是固定平均

这两条 retrieval run 都属于“性能上限冲刺”的第一阶段。  
如果它们拿到比当前 `retrieval_dreamsim_only_atm_small_fixed` 更好的结果，那么现有的 `Kandinsky img2img` reconstruction 主线会自动受益，因为 prototype init 会直接变得更准。

### 优先级 1：把 `Kandinsky img2img` 主线当成正式提交版本继续做完善

原因很简单：

- 它已经在 full test 上打赢了当前所有已完成对照
- 它第一次把“高语义”和“较强低层结构”放在同一条线里
- 再往前推进，应该优先做这条线的多 seed、样例整理、提交级打磨

### 优先级 2：在现有 img2img 框架里继续提升语义条件，而不是优先换 decoder

当前最值得继续追的方向是：

- 更好的 EEG -> semantic target 回归
- 更合理的 target 表征选择
- 继续利用 prototype 作为低层锚点，而不是把它重新升回独立主线

按当前执行顺序，这一层不会先于 retrieval 做。  
更具体地说，接下来的路线是：

1. 先等 `atm_base / atm_base + IDES` 这两条 retrieval run 出结果
2. 如果 retrieval 有增益，先用新的 retrieval backbone 重跑现有 `Kandinsky img2img` 主线，测 prototype 质量提升带来的级联收益
3. 只有在这个级联收益吃完之后，再进入 `CLIP target` 训练和更复杂的 decoder / target-space 实验

这也是为什么 `CLIP target` 分支值得保留：

- 它提示“换 target”可能比“换大 decoder”更有价值
- 但它应该先接入现有的 img2img 框架里验证，而不是另起一套完全平行的提交路线

### 优先级 3：SDXL 保留为 sidecar，不再作为近期第一优先级 pivot

原因是：

- feasibility 已经验证完了
- full-test 对照已经说明它目前不如 Kandinsky img2img
- 在没有更强 prompt / adapter / conditioning 设计之前，不值得抢占主线资源

### 优先级 4：`rag_residual` 和 masked-pretrain 暂时降级，除非前提条件改变

现在至少可以明确说：

- 不是“这些路线还没来得及测”
- 而是“当前写法已经测过，而且没有赢”

所以如果以后还要回到这两条线，至少要先满足下面之一：

- `rag_residual` 方面：先明确怎么避免 gate collapse
- pretraining 方面：先换一个更贴近下游语义回归的预训练目标，或者明确新的迁移策略

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

### 当前最好 reconstruction 的 `val64` 对比结果（纯 decode 参考）

- [reconstruction_metrics.json](/home/xiaoh/DeepLearning/project1_eeg/outputs/eval_compare/val64_seed0/kandinsky_predicted_v4_fast/reconstruction_metrics.json)

### 当前最好 reconstruction 的 full validation 结果（纯 decode 参考）

- [reconstruction_metrics.json](/home/xiaoh/DeepLearning/project1_eeg/outputs/eval_compare/full_val_seed0/kandinsky_predicted_v4_fast/reconstruction_metrics.json)

### 当前最好 reconstruction 的 full-test 主线结果

- 配置：`hpc_img2img_v4_s20_c4_g4p0_str035`
- 远端输出目录：`/hpc2ssd/JH_DATA/spooler/dsaa2012_054/DeepLearning/project1_eeg/outputs/eval_compare/test_seed0/hpc_img2img_v4_s20_c4_g4p0_str035`
- 关键指标：
  - `eval_clip = 0.7513`
  - `eval_ssim = 0.3767`
  - `eval_pixcorr = 0.1567`
  - `eval_alex5 = 0.8489`

### SDXL sidecar 的 full-test 对照结果

- 配置：`sdxl_turbo_proto_text_s4_g0p0_str050`
- 远端输出目录：`/hpc2ssd/JH_DATA/spooler/dsaa2012_054/DeepLearning/project1_eeg/outputs/eval_compare/test_seed0/sdxl_turbo_proto_text_s4_g0p0_str050`
- 关键指标：
  - `eval_clip = 0.6940`
  - `eval_ssim = 0.3642`
  - `eval_pixcorr = 0.1238`
  - `eval_alex5 = 0.8155`

### 旧 prototype / VAE baseline

- [reconstruction_metrics.json](/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/reconstruction_dreamsim_topk4/seed_0/test_predictions/reconstruction_metrics.json)

### 最近失败的 RAG residual 尝试

- [best.pt](/home/xiaoh/DeepLearning/project1_eeg/outputs/experiments/reconstruction_kandinsky_rag_residual_v1/seed_0/best.pt)

### 最近失败的 A800 masked-pretrain 迁移尝试

- 预训练 checkpoint：[encoder_best.pt](/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/eeg_mask_pretrain_v1/seed_0/encoder_best.pt)
- 直接迁移：[best.pt](/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/reconstruction_kandinsky_embed_v5_mask_pretrain_a800/seed_0/best.pt)
- 保守 staged 迁移：[best.pt](/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/reconstruction_kandinsky_embed_v6_mask_pretrain_staged_a800/seed_0/best.pt)

### 值得继续看的 CLIP target 分支

- 输出目录：`/hpc2ssd/JH_DATA/spooler/dsaa2012_054/DeepLearning/project1_eeg/outputs/experiments/reconstruction_clip_embed_v1_proxyselect/seed_0`
- 关键 proxy：
  - `val_subset_top1 = 0.0369`
  - `val_subset_top5 = 0.1022`

---

如果只用一句话总结整个项目到现在的状态，那就是：

**检索已经做出了一个可靠版本；重建也已经从“prototype 单独成图没语义”的旧 baseline，推进到了“prototype 负责低层结构、predicted embedding 负责高层语义”的 Kandinsky img2img 主线，而且这条线已经在本地 full test 上打赢了当前所有已完成对照；接下来真正该做的，不是重新散开战线，而是在这条主线里继续提升语义条件并完成提交级整理。**

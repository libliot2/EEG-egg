# Project1 EEG 项目实验全程说明

更新时间：2026-04-27

这份文档面向两类读者：

- 没直接参与这个项目，但想快速知道“这个项目到底在做什么、已经做到哪一步”的人
- 愿意继续接手实验的人，希望知道哪些路线有效、哪些路线已经被排除

如果你只想先看结论，可以直接跳到“**先给最终结论**”一节。  
如果你想看完整流水，**正式主账本**在 [EXPERIMENT_LOG.md](/home/xiaoh/DeepLearning/EEG/EXPERIMENT_LOG.md)。这份报告只抽取其中真正影响路线决策的关键实验。
[EXPERIMENT_LOG_hpc.md](/home/xiaoh/DeepLearning/EEG/EXPERIMENT_LOG_hpc.md) 不是第二本正式账本，它只是 2026-04-20 从远端 HPC 工作目录拉回来的镜像副本，用来保留 A800 节点上的原始记录。

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

根据本地数据文件 [README.md](/home/xiaoh/DeepLearning/EEG/README.md) 和 `image-eeg-data/*.pt`：

- 训练图像目录共有 `16540` 张 JPG：`training_images`
- 测试图像目录共有 `200` 张 JPG：`test_images`
- `train.pt` 里的 EEG 形状是 `(16540, 4, 63, 250)`
  - 可以粗略理解成：每张训练图片对应 4 次 EEG 记录
  - 每次 EEG 有 63 个通道、250 个时间点
- `test.pt` 里的 EEG 形状是 `(200, 80, 63, 250)`
  - 可以粗略理解成：每张测试图片对应 80 次 EEG 记录

从数据形状、通道数、时间长度、训练/测试图片组织方式看，这个课程包和公开 **THINGS-EEG2** 数据集的单 subject 视觉 EEG 设定高度一致。  
但这里必须写得保守一些：

- 我们手上的直接证据是课程发放的 `train.pt`、`test.pt`、图片目录和样例代码
- 公开 THINGS-EEG2 可以作为 benchmark 和方法参考，但课程包可能被助教筛选、重命名、重打包或只保留了某个 subject 的子集
- 早前做过的 subject 对齐检查倾向于它接近公开 subject `sub-08`，但目前不把“它严格等于公开 sub-08 原始文件”当作 report 里的硬结论

所以后面和 NICE / ATM / ViEEG / NeuroCLIP 等论文对比时，只能作为“同源任务/相近设置下的参照物”，不能把公开论文的多 subject 平均数和我们这个单 subject、本地 200-way split 的数值做绝对等价比较。

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

## 常见名词解释

这一节先把后面反复出现的术语讲清楚。这里的解释不是严格论文定义，而是它们在这个项目里的具体含义。

### EEG

EEG 是脑电信号。  
在这个项目里，每条 EEG 可以看成一个 `63 × 250` 的矩阵：

- `63` 是电极通道数，也就是头皮上不同位置的传感器
- `250` 是时间点数，也就是看到图片后一小段时间内记录到的电信号序列

模型的输入就是这个矩阵，目标是从它推断被试看到了哪张图。

### Trial 和 trial averaging

同一张图片通常会被重复展示多次，每次展示后记录到的一段 EEG 叫一个 trial。  
本地数据里：

- train 每张图有 4 个 trial
- test 每张图有 80 个 trial

`trial averaging` 指的是把同一张图片的多个 trial 平均成一个更干净的 EEG。  
test 有 80 个 trial，所以平均后信噪比通常比 train/val 更高，这也是为什么有些模型的 test retrieval 反而比 val 好。

### Retrieval / top1 / top5

Retrieval 是“多选题”：给模型一段 EEG，再给它一批候选图，让它把候选图按相似度排序。

- `top1`：模型排第一的图是不是正确答案
- `top5`：正确答案是否出现在模型排前五的图里

所以 `top1=0.53` 的意思是，在 200 张测试图里，大约 53% 的 EEG 能把正确图排到第一。

### Reconstruction

Reconstruction 是“画图题”：给模型一段 EEG，让它直接生成一张图片。  
它比 retrieval 更难，因为 retrieval 只需要在候选里选，reconstruction 需要生成像样的图像内容。

### Embedding

Embedding 是把复杂对象压缩成一个向量。  
比如一张图片可以被编码成一个几百维或几千维的向量，这个向量不直接是像素，但包含了图像的语义、风格或结构信息。

本项目里很多模型的核心目标不是直接预测像素，而是：

**EEG → 图像 embedding → 生成/检索图像**

### CLIP embedding

CLIP 是一种把图像和文本映射到同一语义空间的模型。  
`CLIP image embedding` 可以理解成“这张图在 CLIP 看来的语义向量”。如果两张图片语义相近，它们的 CLIP embedding 通常也更接近。

我们用过 CLIP embedding 做两件事：

- retrieval target：让 EEG 表征接近正确图像的 CLIP 表征
- reconstruction target：尝试直接从 EEG 回归 CLIP image embedding

目前结果显示，CLIP target 比早期 Kandinsky target 更容易回归；它已经通过 `CLIP→Kandinsky adapter` 接入当前最强 reconstruction 主线。

### DreamSim

DreamSim 是一种偏“人类感知相似度”的图像特征。  
相比 CLIP 更重语义，DreamSim 更强调两张图在人眼感知上是否相似。

在我们的 retrieval 实验里，DreamSim 明显比 CLIP-only 更强，所以当前 retrieval 主线基本都是：

**EEG → DreamSim image embedding**

### ATM

ATM 指的是一类用于 EEG-to-image decoding 的 encoder 架构，来自相关论文里的 `Align, Then Modulate` / EEG embedding decoding 思路。  
在这个项目里，它主要表示“处理 EEG 的主干网络”，也就是把 `63 × 250` EEG 信号编码成一个向量的模型。

我们用过几个大小和变体：

- `atm_small`：早期快速 baseline
- `atm_base`：中期 retrieval 主线
- `atm_large` / adapter 变体：当前最强 retrieval backbone 所在分支

目前最强的 retrieval 单模型是 `loss_imgsoft_dir` visual17 / ATM-large adapter branch；`posterior_cp_28` 不再是全局最强单模型，但作为 fusion component 能提供互补信号。

### IDES / trial sampling

IDES-style trial sampling 是一种训练时的数据增强方法。  
普通做法是把每张训练图的 4 个 trial 固定平均；IDES-style 做法是在每个 epoch 随机抽取其中 `k` 个 trial 再平均。

这样同一张图在不同 epoch 会产生略有不同的 EEG 输入，相当于一种符合 EEG 数据结构的数据增强。  
我们的 `ATM-base + IDES` 明显强于早期固定平均 baseline。

### Prototype

Prototype 指 retrieval 找到的“最像当前 EEG 的训练图”。  
最早我们试过直接把 prototype 当 reconstruction 答案，但因为 train/test 类别不重叠，这条线的语义很容易错。

后来 prototype 被放到了更合适的位置：不再当最终答案，而是作为 img2img 的低层结构起点。  
也就是：

**prototype 负责颜色/布局/低层结构，predicted embedding 负责高层语义。**

### VAE latent

VAE 是一种把图片压缩到 latent space 再解码回图片的模型。  
`VAE latent` 不是图片像素，而是图片的压缩表示。

我们早期的 residual-VAE 路线尝试在 VAE latent 上修正 prototype，但生成图语义不清楚，所以没有继续作为主线。

### Kandinsky image embedding

Kandinsky 是一个 diffusion 图像生成模型。  
`Kandinsky image embedding` 可以理解成 Kandinsky decoder 能使用的一种图像条件向量：如果给 decoder 一个正确的 image embedding，它就能生成语义上接近目标图像的图片。

我们做的 `reconstruction_kandinsky_embed_v4_proxyselect` 本质上是在学：

**EEG → Kandinsky image embedding**

然后再把这个预测出来的 embedding 交给 Kandinsky decoder 生成图片。

### Kandinsky decoder

Kandinsky decoder 是真正负责“画图”的部分。  
我们的 sanity check 说明，如果直接给它 ground-truth Kandinsky image embedding，它能生成语义很对的图，所以 decoder 本身不是最主要瓶颈。

目前瓶颈更像是：

**EEG 预测出来的 embedding 还不够准，以及纯 embedding 从噪声开始生成时低层结构不稳定。**

### img2img

`img2img` 是 image-to-image generation。  
它不是从纯噪声开始生成，而是先给 diffusion model 一张初始图，再加一定噪声，然后让模型在条件引导下重绘。

在我们的当前主线里：

- 初始图来自 retrieval prototype
- 条件来自 EEG 预测的 Kandinsky image embedding
- `strength` 控制保留多少初始图结构，以及允许模型改动多少

这就是为什么 `Kandinsky img2img` 同时提升了语义和 SSIM/pixcorr。

### SDXL / SDXL Turbo

SDXL 是 Stable Diffusion XL，一类常见的现代 diffusion 生成模型。  
SDXL Turbo 是更快的变体。

我们做过 `SDXL Turbo + prototype + text prompt` 的 sidecar 实验，证明这条路能跑通，但当前指标没有打赢 Kandinsky img2img 主线。

### RAG residual

这里的 RAG residual 不是大语言模型里的 RAG，而是借用了“retrieval-augmented”的思想。  
它尝试让模型同时使用：

- direct EEG regression
- retrieval 找到的 top-k 语义邻居
- 一个 gate 来决定 residual 分支加多少

实验结果里 gate 几乎塌到 0，说明模型最后基本没用上 retrieval residual，所以这条线暂时降级。

### SSIM / pixcorr / eval_clip / eval_alex5

这些是 reconstruction 的图像质量指标：

- `eval_clip`：更看语义是否对
- `eval_alex5`：更偏感知语义相似度
- `eval_ssim`：更看图像结构是否相似
- `eval_pixcorr`：更看像素级相关性

它们必须一起看。  
比如旧 prototype baseline 的 SSIM 很高，但肉眼看没有明确语义；而纯 Kandinsky decode 的 CLIP 高，但 SSIM 低。当前 img2img 主线的意义就在于同时照顾语义和低层结构。

### Channel subset / posterior_cp_28

Channel subset 指只保留一部分 EEG 电极通道训练模型。  
`posterior_cp_28` 是当前最强 retrieval 子集，包含 centroparietal 和 posterior 区域的 28 个通道。

直观理解是：视觉刺激相关信号更集中在后部视觉相关脑区；前部通道在当前模型里反而可能带来噪声。  
这不是说前部脑区一定没用，而是说在当前数据、当前模型和当前任务里，删掉前部通道能提高 test top1。

## 先给最终结论

如果你只看这一节，应该能抓住整个项目到目前为止最重要的结论。

1. **Retrieval 已经有可用的正式合规强基线，当前 point best 是 `0.6850/0.9600`。**
   当前最可靠的正式主结果是 `loss_imgsoft_dir + posterior_cp_28` 的 validation-selected z-score fusion：
   - test `top1 = 0.6850`
   - test `top5 = 0.9600`
   - 4-seed 复核均值：`top1 = 0.6487 ± 0.0210`, `top5 = 0.9562 ± 0.0041`

   这条线的核心经验是：DreamSim target、ATM-large/adapter backbone、posterior/centroparietal channel subset 和 validation-only fusion 都有真实贡献。partner 的 top-4 reranker 记录为 `0.6700/0.9650`，我们目前复现到了更高 top1 的 clean fusion，但 top5 仍略低。

   重要边界：trial-TTA 和 SATTC-lite/CSLS 能把 diagnostic 数字推到 `0.79-0.81` top1，但前者违反 `avg_trials=True`，后者使用整批 test distribution，所以都只写作 upper-bound / diagnostic，不作为正式结果。

2. **最早的 prototype / residual-VAE reconstruction 虽然某些量化指标看起来不差，但生成图在肉眼上基本没有语义。**
   也就是说：它“在指标上像成功”，但“在图像上像失败”。

3. **Kandinsky decoder 本身不是首要瓶颈，但“只靠纯 embedding、从噪声开始解码”也不是最终答案。**
   我们做过一个非常重要的 sanity check：直接用 ground-truth Kandinsky image embedding 去解码，`val64` 上能达到：
   - `eval_clip = 0.9973`
   - `eval_alex5 = 0.9824`
   这说明 decoder 的语义能力很强。  
   但后面的 img2img 实验又进一步说明：如果没有一个低层结构锚点，纯 embedding decode 的 `SSIM` 和 `pixcorr` 仍然偏弱，所以真正有效的做法是把“高层语义条件”和“低层图像起点”结合起来。

4. **当前最值得保留的 reconstruction 主线，已经进一步升级为“low-level residual-VAE init + predicted CLIP embedding + CLIP→Kandinsky adapter 的 Kandinsky img2img”。**
   也就是：
   - 旧 `reconstruction_dreamsim_topk4` 分支先生成一张低层 init 图，作为结构和纹理锚点
   - EEG predictor 先预测 CLIP image embedding，作为更容易回归的高层语义空间
   - 一个小型 adapter 再把 CLIP embedding 映射到 Kandinsky image embedding 空间
   - Kandinsky decoder 从这张 low-level init 图加噪后的 latent 开始做 img2img denoise

5. **在目前已经跑完 `full test` 的配置里，新的 balanced reconstruction 最强结构是 `clip_pred_v2_adapter_blend_low85_post15_str030`。**
   这个配置使用：
   - CLIP predictor checkpoint：`reconstruction_clip_embed_v2_loss_imgsoft_local`
   - adapter checkpoint：`clip_to_kandinsky_adapter_v1`
   - low-level init checkpoint：`reconstruction_dreamsim_topk4`
   - decoder：Kandinsky img2img
   - init image：`85%` residual-VAE/prototype low-level 图 + `15%` posterior-old retrieval prototype 图
   - sampling：`20 steps`, `4 candidates`, `guidance 4.0`, `strength 0.30`

   它在 `full-val` 上的结果是：
   - `blend low85/post15, strength=0.30`: `eval_clip=0.7620`, `eval_ssim=0.4012`, `eval_pixcorr=0.1770`
   - `blend low85/post15, strength=0.25`: `eval_clip=0.7555`, `eval_ssim=0.4167`, `eval_pixcorr=0.1870`

   它在本地 200-way `test` 上的结果是：
   - `blend low85/post15, strength=0.30`: `eval_clip=0.8212`, `eval_ssim=0.3788`, `eval_pixcorr=0.2335`, `eval_alex5=0.9103`
   - `blend low85/post15, strength=0.25`: `eval_clip=0.8029`, `eval_ssim=0.3954`, `eval_pixcorr=0.2413`, `eval_alex5=0.8994`
   - `lowlevel-only, strength=0.25`: `eval_clip=0.8160`, `eval_ssim=0.3962`, `eval_pixcorr=0.2302`, `eval_alex5=0.8921`

   需要严谨说明：`blend low85/post15, strength=0.30` 是 balanced best，因为它的 CLIP、PixCorr 和 Inception 都强于 lowlevel-only str030；`lowlevel-only, strength=0.25` 仍是 SSIM / CLIP+SSIM 更强的候选。
   `blend low85/post15, strength=0.25` 虽然 PixCorr 最高，但 CLIP 掉到 `0.8029`，所以不作为主线。

   2026-04-27 的 reconstruction longrun 目前只跑到 `val64` 级别，还不能替代上面的 full-test best，但已经提供了下一轮 full-val 候选：
   - CLIP-heavy 候选：`a1_c4_g4p5_s0p30`，`val64 eval_clip=0.7867`, `eval_ssim=0.3581`, `eval_pixcorr=0.1304`
   - balanced 候选：`a2_c8_semantic_lowlevel_neg_mse_w0p35`，`val64 eval_clip=0.7815`, `eval_ssim=0.3693`, `eval_pixcorr=0.1472`
   - SSIM-heavy 候选：`b2_lowlevel_rgb_lossimg_structure_str025`，`val64 eval_clip=0.7485`, `eval_ssim=0.4323`, `eval_pixcorr=0.1380`

   这些数字说明：longrun 找到了更细的 CLIP/SSIM trade-off，但它们还只是 val64 candidate。正式结论必须等 full-val 确认，不能直接把 val64 最优点写成新的 test best。

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

9. **`CLIP` target regression 已经从“有潜力的下一步”升级为当前 reconstruction 主线的一部分。**
   早期 `reconstruction_clip_embed_v1_proxyselect` 已经提示 CLIP target 比 Kandinsky target 更容易回归；后续 `reconstruction_clip_embed_v2_loss_imgsoft_local + clip_to_kandinsky_adapter_v1` 证明它能接入现有 Kandinsky img2img 生成链路，并在 test 上显著提高语义指标。

10. **从项目决策角度看，后续 reconstruction 的重点应该是继续加强 low-level branch 和 candidate selection，而不是回到 Kandinsky-only 或单纯 decoder sweep。**
   这次实验已经证明：旧 prototype/VAE 路线作为最终图像不够好，但作为 low-level init 很有价值；少量 posterior prototype 融合还能进一步提升 balanced 指标。2026-04-27 的 train-bank prototype 替换和二阶段 img2img refinement 都没有带来正向收益，所以现在更合理的方向是：
   - 训练或筛选更适合 img2img 起点的低层结构图
   - 在现有 `CLIP predictor + CLIP->Kandinsky adapter` 语义条件上做 candidate-level reranking
   - 用 full-val gate 决定是否把 val64 候选推到 frozen test

   当前已经提交到 HKUST(GZ) HPC 的 phase2 campaign 仍在进行中，不应在 report 中提前写成结论：
   - 原普通队列 job `9710087/9710088` 因 priority pending，已取消
   - 新 emergency job `9710179`: fast balanced/CLIP-heavy refinement array，已开始运行，当前完成 4 个 val64 子任务
   - 新 emergency job `9710180`: SSIM-heavy low-level init training/evaluation，仍在等待调度
   - `9710179` 前 4 个结果没有刷新 longrun balanced best；`w0.25,s0.30/0.32` 的 PixCorr 较高：`eval_clip=0.7793`, `eval_ssim=0.3684`, `eval_pixcorr=0.1537`；`w0.30,s0.28` 的 SSIM 较高：`eval_clip=0.7619`, `eval_ssim=0.3856`, `eval_pixcorr=0.1486`

## 一个容易混淆但必须先说清楚的点

[EXPERIMENT_LOG.md](/home/xiaoh/DeepLearning/EEG/EXPERIMENT_LOG.md) 里的 “Current Best Reconstruction” banner 已经同步到 2026-04-26 的 `clip_pred_v2_adapter_blend_low85_post15_str030`。
但读这个 banner 时仍然要区分“balanced/deep-feature best”和“SSIM-heavy best”。

原因是：

- `blend low85/post15, strength=0.30` 在 CLIP、PixCorr、Inception 上更稳
- `lowlevel-only, strength=0.25` 在 SSIM 和 `CLIP+SSIM` 这个简单合计上更强
- 项目决策时，我们要同时考虑“量化指标 + 肉眼可辨识度 + 是否符合数据设定”

所以这份报告里提到的“当前最佳 reconstruction 主线”，指的是：

**从研究判断上最值得继续推进的 two-branch low-level-init + CLIP-semantics 方案，而不是单看某一个指标的最高值。**

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

### 阶段 B：把 retrieval 做到可用强基线

Retrieval 在这个项目里有两层作用：它本身是正式评分任务之一，同时也给 reconstruction 提供 prototype / low-level init。早期实验先确认了一个关键方向：**DreamSim 这类感知特征比 CLIP-only 更适合当前 200-way EEG retrieval**。后续所有强结果基本都建立在 DreamSim target 上。

关键里程碑如下。完整训练命令、checkpoint 和负结果见 [EXPERIMENT_LOG.md](/home/xiaoh/DeepLearning/EEG/EXPERIMENT_LOG.md)。

| Milestone | 结果 | 决策含义 |
|---|---:|---|
| `retrieval_clip_only_atm_small` | val `0.0333/0.1179` | CLIP-only 太弱，不作为主线 |
| `retrieval_dreamsim_only_atm_small_fixed` | test `0.3450/0.6250` | 第一条可用 baseline |
| `ATM-base + IDES + DreamSim` | test `0.5000/0.8050` | scale-up 和 trial sampling 有效 |
| `posterior_cp_28` channel subset | test `0.5300/0.8200` | 后部/centroparietal 通道有互补信号 |
| `VisibleViEEG` 多分支探索 | best test `0.4800/0.7850` | 显式多视角结构有信号，但未超过主线 |
| adapter / NeuroCLIP-lite backbone | 3-seed `0.6233/0.9283` | adapter target、visual17 和轻量 perturbation 有效 |
| `loss_imgsoft_dir` single model | test `0.6700/0.9450` | 当前最强单分支 backbone |
| `loss_imgsoft_dir + posterior_cp_28` clean fusion | test `0.6850/0.9600` | 当前正式合规 point best |
| clean fusion 4-seed 复核 | `0.6487±0.0210` / `0.9562±0.0041` | point best 真实，但 top1 有 seed variance |
| SATTC-lite / CSLS diagnostic | test `0.7900/0.9900` | 使用 test distribution，只能作为诊断上界 |
| trial-TTA diagnostic | 约 `0.80+` top1 | 违反 `avg_trials=True`，不能作为正式结果 |

这一阶段最终保留下来的结论有五个：

- **DreamSim 是当前 retrieval 最可靠的 target**，比 CLIP-only 和早期 CLIP+DreamSim 混合更稳定。
- **通道选择有真实影响**，`posterior_cp_28` 能提供和 visual17 / all-channel 分支不同的互补信息。
- **两阶段 reranker 不是万能的**，它可以细排 top-k，但 top5 recall 主要由 base backbone 决定。
- **post-hoc fusion 比 teacher distillation 更有效**，当前 teacher-KL student 反而把 test 拉到 `0.6100/0.9400`。
- **所有使用 test distribution 或 test trial 子采样的结果都只能写成 diagnostic**；正式 retrieval 结果必须保持 `avg_trials=True`，且只用 validation 选择权重。

所以当前 retrieval 的稳妥表述是：

**正式 point best 是 `0.6850/0.9600`；更严谨的稳定性估计是 4-seed `0.6487±0.0210` / `0.9562±0.0041`。**

这条线后续如果继续做，重点不是再堆 test-time calibration，而是构造更可靠的 validation episode / training objective，让分支互补能在合规设置下转化成稳定 top1 提升。

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

这也是为什么这条线后来虽然保留为 baseline，但不再继续作为“最终输出图像”的主路线。

这一阶段的结论是：

**以训练图原型为最终答案的 prototype-based reconstruction，在这个项目的数据设定下，不像一个有希望的长期主线。**

但 2026-04-26 的 low-level init 实验证明，这条判断需要更精确地表述：
prototype/residual-VAE 不适合当最终图像，但适合当 img2img 的低层初始化分支。它能提供 SSIM 和 PixCorr 需要的结构锚点，再由 CLIP/Kandinsky 条件补上高层语义。

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

- [reconstruction_kandinsky_embed_v4_proxyselect/seed_0/best.pt](/home/xiaoh/DeepLearning/EEG/outputs/experiments/reconstruction_kandinsky_embed_v4_proxyselect/seed_0/best.pt)

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

- [reconstruction_metrics.json](/home/xiaoh/DeepLearning/EEG/outputs/eval_compare/val64_seed0/kandinsky_groundtruth_local/reconstruction_metrics.json)

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

- `val64`: [reconstruction_metrics.json](/home/xiaoh/DeepLearning/EEG/outputs/eval_compare/val64_seed0/kandinsky_predicted_v4_fast/reconstruction_metrics.json)
- `full-val`: [reconstruction_metrics.json](/home/xiaoh/DeepLearning/EEG/outputs/eval_compare/full_val_seed0/kandinsky_predicted_v4_fast/reconstruction_metrics.json)

在 full validation 上，它的结果是：

- `eval_clip = 0.6933`
- `eval_alex5 = 0.7993`
- `eval_pixcorr = 0.0848`

这一阶段的结论是：

1. **decoder 不是首要瓶颈**
2. **在当前 `val64` 对比里，continuous predicted embedding 优于 retrieval_top1**
3. **在“纯 embedding decode”这条子路线里，当前最好的生成配置是 `predicted_v4_fast`**

### 阶段 F：RAG residual 和 masked-pretrain 两条负结果

Kandinsky embedding regression 跑通后，我尝试过两条看起来合理、但最终没有赢的结构增强。

第一条是 `rag_residual v1`：保留 direct regression，同时让 retrieval top-k 提供额外上下文，再用 gate 决定 residual 加多少。工程链路已经跑通，30 epoch 正式训练的 best 指标是 `val_subset_top1=0.0254`, `val_subset_top5=0.0792`，略低于 `v4_proxyselect` 的 `0.0272/0.0846`。更关键的是 `avg_gate≈8e-6`，说明 gate 几乎全关，模型退化成 direct regression，retrieval residual 没有真正被用上。

第二条是 A800 上的 masked EEG pretraining：先做 masked temporal reconstruction，再把 encoder 迁移到 Kandinsky embedding regression。预训练本身能收敛，`eeg_mask_pretrain_v1` 的 best `val_total_loss=0.6766`；但 downstream transfer 明显失败：

| 迁移方式 | best proxy | 结论 |
|---|---:|---|
| `v5_mask_pretrain_a800` 直接迁移 | `val_subset_top1=0.0006`, `val_subset_top5=0.0048` | 明显弱于 baseline |
| `v6_mask_pretrain_staged_a800` 冻结再解冻 | `val_subset_top1=0.0030`, `val_subset_top5=0.0073` | 略好于直接迁移，但仍远弱 |
| `v4_proxyselect` baseline | `val_subset_top1=0.0272`, `val_subset_top5=0.0846` | 当前 Kandinsky-target 历史主线 |

这两条负结果的价值是把边界说清楚：当前写法下，不应该继续在 `rag_residual` gate 或 masked reconstruction pretraining 上堆算力。除非换 residual 使用方式，或把 pretraining objective 改成更贴近 EEG-image alignment，否则它们不是近期主线。

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

### 阶段 I：从 Kandinsky-target 升级到 CLIP-target semantic branch

确定 img2img 双分支有效后，我并行验证了两个方向：换 decoder，以及换 semantic target。

SDXL Turbo sidecar 已经跑通，但没有打赢 Kandinsky img2img。`sdxl_turbo_proto_text_s4_g0p0_str050` 在本地 200-way test 上是 `eval_clip=0.6940`, `eval_ssim=0.3642`, `eval_pixcorr=0.1238`, `eval_alex5=0.8155`。它说明 SDXL 路线可行，但当前 prompt / conditioning 设计不够强，因此只保留为 sidecar。

真正改变 reconstruction 主线的是 CLIP target。早期 `reconstruction_clip_embed_v1_proxyselect` 已经显示，直接回归 CLIP image embedding 的 proxy `0.0369/0.1022` 高于 Kandinsky-target `v4_proxyselect` 的 `0.0272/0.0846`。后续我们训练了 `reconstruction_clip_embed_v2_loss_imgsoft_local`，再用 `clip_to_kandinsky_adapter_v1` 把 predicted CLIP embedding 接回 Kandinsky image embedding 空间。

最终形成的当前主线是：

**EEG → predicted CLIP image embedding → CLIP-to-Kandinsky adapter → Kandinsky semantic condition；同时用 low-level residual-VAE/prototype 图作为 img2img init。**

这条路线在 full-test 上刷新了 reconstruction：

| 配置 | `eval_clip` | `eval_ssim` | `eval_pixcorr` | 结论 |
|---|---:|---:|---:|---|
| `clip_pred_v2_adapter_posterior_old_str030` | `0.8161` | `0.3289` | `0.2036` | 语义强，但 SSIM 弱 |
| `clip_pred_v2_adapter_blend_low85_post15_str030` | `0.8212` | `0.3788` | `0.2335` | 当前 balanced best |
| `clip_pred_v2_adapter_lowlevel_topk4_str025` | `0.8160` | `0.3962` | `0.2302` | 当前 SSIM-heavy 候选 |

这说明当前 reconstruction 的关键转变不是“换更大的 decoder”，而是把语义条件换成更容易回归的 CLIP 空间，同时保留 low-level init 承担 SSIM/PixCorr。后续 phase2 campaign 也应围绕这条主线做 full-val gate，而不是回到 Kandinsky-only 或 SDXL-only。

## 除了模型本身，还做了哪些工程支撑

如果只看模型名字，很容易忽视中间其实做了不少“为了让实验可跑、可比较、可复现”的基础工作。

这些工作包括：

1. **统一实验账本**
   - 所有正式训练、评估、失败尝试、debug 过程都写进 [EXPERIMENT_LOG.md](/home/xiaoh/DeepLearning/EEG/EXPERIMENT_LOG.md)
   - 这样后面回顾时，不会只剩“我记得当时好像跑过”
   - [EXPERIMENT_LOG_hpc.md](/home/xiaoh/DeepLearning/EEG/EXPERIMENT_LOG_hpc.md) 只是从远端拉回的临时镜像，不应视作第二本正式账本

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
| Retrieval 当前正式 best | `loss_imgsoft_dir + posterior_cp_28` clean validation-selected fusion | test `top1=0.6850`, `top5=0.9600` | 当前合规 point best；4-seed 均值为 `0.6487±0.0210` / `0.9562±0.0041` |
| Retrieval 非合规诊断上界 | SATTC-lite CSLS / trial-TTA diagnostic | CSLS test `0.7900/0.9900`; trial-TTA 约 `0.80+` | 使用 test distribution 或不符合 `avg_trials=True`，只能作为瓶颈分析 |
| Retrieval 旧最强单模型 | `retrieval_dreamsim_only_atm_base_ides_bs128_v1_local` | test `top1=0.5000`, `top5=0.8050` | all-channel 单模型参考 |
| Retrieval channel 对照 | `retrieval_channel_nofront37_atm_base_ides_v1_local` | test `top1=0.4750`, `top5=0.8100` | val 高但 test 不稳，不能只靠 val 排名决策 |
| Retrieval 新结构探索 | `retrieval_visible_vieeg_ocmv_v1_local` | test `top1=0.4800`, `top5=0.7850` | 多分支 VisibleViEEG 有信号，但还没超过现主线 |
| 旧 reconstruction baseline | `reconstruction_dreamsim_topk4` | test `eval_clip=0.5386`, `eval_ssim=0.4326` | 指标看起来不差，但图像没有语义 |
| Kandinsky-target 训练主线 | `reconstruction_kandinsky_embed_v4_proxyselect` | `val_subset_top1=0.0272`, `val_subset_top5=0.0846` | Kandinsky embedding regression 的历史最佳 checkpoint |
| sanity check | `kandinsky_groundtruth_local` | `val64 eval_clip=0.9973` | decoder 不是主要瓶颈 |
| discrete retrieval 条件 | `kandinsky_retrieval_top1_local` | `val64 eval_clip=0.6612` | 离散 top1 不如连续预测 |
| 纯 decode 最好点 | `kandinsky_predicted_v4_fast` | `val64 eval_clip=0.7078`, `eval_ssim=0.0509`; full-val `eval_clip=0.6933`, `eval_ssim=0.1836` | 纯 embedding decode 里的最好配置 |
| 慢速高采样对照 | `kandinsky_predicted_v4_quality` | `val64 eval_clip=0.6992` | 更慢不代表更好 |
| Kandinsky img2img SSIM 参照 | `hpc_img2img_v4_s20_c4_g4p0_str035` | test `eval_clip=0.7513`, `eval_ssim=0.3767`, `eval_pixcorr=0.1567` | 旧主线，仍是 SSIM-only reference |
| CLIP-adapter img2img 旧主线 | `clip_pred_v2_adapter_posterior_old_str030` | test `eval_clip=0.8161`, `eval_ssim=0.3289`, `eval_pixcorr=0.2036`, `eval_alex5=0.9114` | 旧 semantic best；SSIM 是主要短板 |
| CLIP-adapter blended low-level init 新主线 | `clip_pred_v2_adapter_blend_low85_post15_str030` | test `eval_clip=0.8212`, `eval_ssim=0.3788`, `eval_pixcorr=0.2335`, `eval_alex5=0.9103` | 当前 balanced reconstruction best：比 low-level-only str030 语义和 PixCorr 更强 |
| SSIM-heavy reconstruction 候选 | `clip_pred_v2_adapter_lowlevel_topk4_str025` | test `eval_clip=0.8160`, `eval_ssim=0.3962`, `eval_pixcorr=0.2302` | 当前 SSIM / CLIP+SSIM 更强的提交候选 |
| train-bank retrieval prototype init | `loss_imgsoft + posterior_cp_28` train-bank prototype blend | val64 best tested `eval_clip=0.7748`, `eval_ssim=0.3675`, `eval_pixcorr=0.1411` | 没有超过旧 `low85/post15` reference；更强 retrieval prototype 不自动等于更强 reconstruction init |
| 二阶段 img2img refinement | stage-1 best image -> low-strength second-pass img2img | val64 best second pass `eval_clip=0.7691`, `eval_ssim=0.3395`, `eval_pixcorr=0.1291` | 二次 denoise 会引入 diffusion drift；不升 full test |
| 04-27 longrun CLIP-heavy 候选 | `a1_c4_g4p5_s0p30` | val64 `eval_clip=0.7867`, `eval_ssim=0.3581`, `eval_pixcorr=0.1304` | 语义最强的 val64 点；SSIM/PixCorr 偏低，需 full-val 再判断 |
| 04-27 longrun balanced 候选 | `a2_c8_semantic_lowlevel_neg_mse_w0p35` | val64 `eval_clip=0.7815`, `eval_ssim=0.3693`, `eval_pixcorr=0.1472` | 当前最值得升 full-val 的 balanced candidate，但还不是 test best |
| 04-27 longrun SSIM-heavy 候选 | `b2_lowlevel_rgb_lossimg_structure_str025` | val64 `eval_clip=0.7485`, `eval_ssim=0.4323`, `eval_pixcorr=0.1380` | SSIM 明显高，但语义掉得多；适合作为 SSIM 分支而非 balanced 主线 |
| 04-27 longrun perceptual low-level 候选 | `b2_lowlevel_rgb_lossimg_perceptual_str025` | val64 `eval_clip=0.7609`, `eval_ssim=0.4064`, `eval_pixcorr=0.1507` | PixCorr 比 structure init 好，但没有超过 balanced 主线 |
| 04-27 HPC phase2 campaign | emergency jobs `9710179`, `9710180` | `9710179` 已完成 4 个 val64 子任务；当前 balanced/PixCorr early best `0.7793/0.3684/0.1537`; SSIM early best `0.7619/0.3856/0.1486`; `9710180` 仍在 pending | 正在运行/排队；只做 val64/full-val 选择，不用 test 调参 |
| SDXL sidecar | `sdxl_turbo_proto_text_s4_g0p0_str050` | test `eval_clip=0.6940`, `eval_ssim=0.3642` | 可行，但没打赢 Kandinsky img2img |
| 新结构尝试 | `reconstruction_kandinsky_rag_residual_v1` | best `val_subset_top1=0.0254`, gate 近 0 | 跑通了，但没赢，也没真正用上 residual |
| A800 预训练 | `eeg_mask_pretrain_v1` | best `val_total_loss=0.6766` | 预训练目标本身能收敛 |
| A800 直接迁移 | `reconstruction_kandinsky_embed_v5_mask_pretrain_a800` | best `val_subset_top1=0.0006`, `val_subset_top5=0.0048` | 在当前 proxy 指标下明显弱于 baseline |
| A800 保守迁移 | `reconstruction_kandinsky_embed_v6_mask_pretrain_staged_a800` | best `val_subset_top1=0.0030`, `val_subset_top5=0.0073` | 比直接迁移略好，但在当前 proxy 指标下仍远弱于 `v4_proxyselect` |
| 新语义 target 验证 | `reconstruction_clip_embed_v2_loss_imgsoft_local + clip_to_kandinsky_adapter_v1` | full-val `eval_clip=0.7697`, `eval_ssim=0.3617` | CLIP target 已接入生成链路并升为主线 |

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

- 当前正式 point best：`loss_imgsoft_dir + posterior_cp_28` clean validation-selected z-score fusion
- test 结果：
  - `top1 = 0.6850`
  - `top5 = 0.9600`
- 多 seed 复核：
  - seed0-3 clean recipe mean/std：`top1 = 0.6487 ± 0.0210`, `top5 = 0.9562 ± 0.0041`
- 组成分支：
  - `loss_imgsoft_dir` visual17 / ATM-large branch：单 seed test `0.6700/0.9450`
  - `posterior_cp_28` channel branch：提供互补信号，但现在是 fusion component，不再是全局 best 单模型

为什么保留它：

- 这是目前最强的 non-transductive、validation-only 选择结果，没有用 test label、test grid、test batch calibration 或 trial-TTA
- `0.6850/0.9600` 是真实可复现的 seed-level best point；如果讨论稳定性，则应同时报告 `0.6487±0.0210 / 0.9562±0.0041`
- `posterior_cp_28` 的通道选择仍然有价值，但最新实验已经说明：把更强 retrieval branch 直接替换成 reconstruction prototype init，并不会自动提升 reconstruction

### B. Reconstruction 当前最佳方案

- CLIP predictor checkpoint：`reconstruction_clip_embed_v2_loss_imgsoft_local`
- CLIP→Kandinsky adapter：`clip_to_kandinsky_adapter_v1`
- 主生成配置：`clip_pred_v2_adapter_blend_low85_post15_str030`
  - `Kandinsky img2img`
  - init image 来自 `85%` 旧 `reconstruction_dreamsim_topk4` residual-VAE/prototype 图 + `15%` posterior-old retrieval prototype 图
  - semantic condition 来自 CLIP predictor，再经 CLIP→Kandinsky adapter 转成 Kandinsky image embedding
  - `20 steps`
  - `4 candidates`
  - `guidance 4.0`
  - `strength 0.30`
- SSIM-heavy 备选配置：`clip_pred_v2_adapter_lowlevel_topk4_str025`
  - 不混入 posterior prototype，只用 low-level init
  - `strength 0.25`

关键结果：

- `full-val`：
  - `blend low85/post15, strength=0.30`: `eval_clip = 0.7620`, `eval_ssim = 0.4012`, `eval_pixcorr = 0.1770`
  - `lowlevel-only, strength=0.25`: `eval_clip = 0.7428`, `eval_ssim = 0.4164`, `eval_pixcorr = 0.1765`
- 本地 200-way `test`：
  - `blend low85/post15, strength=0.30`: `eval_clip = 0.8212`, `eval_ssim = 0.3788`, `eval_pixcorr = 0.2335`, `eval_alex5 = 0.9103`
  - `lowlevel-only, strength=0.25`: `eval_clip = 0.8160`, `eval_ssim = 0.3962`, `eval_pixcorr = 0.2302`, `eval_alex5 = 0.8921`

为什么保留它：

- 它把“低层结构起点”和“高层语义条件”分开：residual-VAE/prototype init 负责低层布局和纹理，CLIP predictor 负责语义，adapter 负责接入 Kandinsky decoder
- 它验证了前面被放弃的 prototype/VAE 路线不是错，而是位置错；作为最终图像语义弱，但作为 img2img 初始图能显著提升 SSIM
- 和旧 `posterior_old_str030` 相比，当前 balanced 点在 test 上把 CLIP 从 `0.8161` 提到 `0.8212`，SSIM 从 `0.3289` 提到 `0.3788`，PixCorr 从 `0.2036` 提到 `0.2335`
- 如果最终评分更重视 SSIM 或 CLIP+SSIM 的合计，`lowlevel-only strength=0.25` 是更强候选；如果更重视 CLIP、PixCorr 和 Inception 的平衡，`blend low85/post15 strength=0.30` 是更稳候选

最新 low-level / blended init 实验的结论：

- 只调 posterior prototype init 的 strength，收益很小；真正的结构性收益来自把 old residual-VAE 输出作为 low-level init
- 在 low-level init 基础上加入少量 posterior prototype 融合后，`blend_low85_post15_str030` 进一步把 test CLIP 提到 `0.8212`、PixCorr 提到 `0.2335`
- `lowlevel_topk4_str025` 仍然是当前 SSIM-heavy 点：test `eval_ssim=0.3962`，且 CLIP 仍有 `0.8160`
- 2026-04-27 的 train-bank retrieval prototype 替换实验没有打赢旧 `low85/post15` val64 reference：`low90/new10,str030` 只有 `eval_clip=0.7748`, `eval_ssim=0.3675`, `eval_pixcorr=0.1411`，而更高 new-prototype 权重会明显伤害 CLIP
- 2026-04-27 的二阶段 refinement 也没有收益：stage-1 val64 `0.7753/0.3668/0.1406`，二阶段最好的 `strength=0.10-0.15` 会把 CLIP、SSIM、PixCorr 全部压低，只是局部提高 Alex5
- 2026-04-27 longrun 进一步把候选分成三类：CLIP-heavy (`a1_c4_g4p5_s0p30`), balanced (`a2_c8_semantic_lowlevel_neg_mse_w0p35`), SSIM-heavy (`b2_lowlevel_rgb_lossimg_structure_str025`)
- 这些 longrun 结果目前只在 val64 上成立，还没有 full-val/test 证据，因此只能写成“候选”，不能写成“新最佳”
- 后续应该继续训练更干净的 first-stage low-level branch，并做 candidate-level selection refinement；不应回到 Kandinsky-only、只做 decoder 参数 sweep、继续替换 retrieval prototype selector，或者重复 img2img denoise

当前正在跑 / 排队的 reconstruction campaign：

- 本地 `outputs/reconstruction_campaigns/20260427_longrun_v2` 已产出 64 个 val64 metric；`b2_lowlevel_rgb_postcp_structure` 仍在跑
- HPC `outputs/reconstruction_campaigns/20260427_phase2_refine` 对应 emergency job `9710179`，用于 balanced/CLIP-heavy 小范围 refinement；当前已完成 4 个 val64 子任务，剩余子任务继续按 array throttle 运行
- HPC `outputs/reconstruction_campaigns/20260427_phase2_ssim` 对应 emergency job `9710180`，用于 SSIM-heavy low-level init 训练和评估；当前仍在 pending
- `9710179` 当前最好 balanced/PixCorr early result：`balanced_negmse_w025_s030_c8_g400_o0` / `s032`，`eval_clip=0.7793`, `eval_ssim=0.3684`, `eval_pixcorr=0.1537`；SSIM early best 是 `balanced_negmse_w030_s028_c8_g400_o0`，`eval_clip=0.7619`, `eval_ssim=0.3856`, `eval_pixcorr=0.1486`；均尚未超过 longrun balanced best

### C. 次级研究分支

- `hpc_img2img_v4_s20_c4_g4p0_str035`

为什么保留它：

- 它是旧的 Kandinsky target img2img 主线，曾经是 SSIM reference：test `eval_ssim = 0.3767`
- 现在 low-level init 的 `strength=0.25` 已经把 SSIM 提到 `0.3962`，所以它不再是 SSIM best，只保留为历史对照
- 它在语义和深层感知指标上低于新 CLIP-adapter 分支，因此不再是主线

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

### 最新完成的 retrieval 性能冲刺

截至 2026-04-27，retrieval 已经从早期 `ATM-base + IDES + DreamSim` 的 `0.50/0.805`，推进到 `loss_imgsoft_dir + posterior_cp_28` clean validation-selected fusion 的 `0.6850/0.9600`。
这个结果是当前正式 point best，但多 seed 复核显示同一 clean recipe 的期望 top1 更接近 `0.6487±0.0210`，因此后续 retrieval 如果继续做，重点应是稳健性和泛化，而不是继续追单个 seed 的偶然峰值。

### 优先级 1：reconstruction 先加强 low-level branch，而不是继续替换 prototype selector

最新 train-bank prototype 实验已经补上了这个验证：用 `loss_imgsoft_dir + posterior_cp_28` 这种更强 retrieval branch 重新选训练图 prototype，再和 low-level init 融合，并没有超过旧 `low85/post15` reference。
二阶段 refinement 也已经在 val64 上证伪：把 stage-1 生成图再作为低强度 img2img init，会降低 CLIP、SSIM 和 PixCorr。
因此 reconstruction 的下一步不再是“继续换 prototype”或“重复 denoise”，而是围绕下面两条推进：

- 训练更干净的 low-level branch：不要只从 retrieval top1/topk 拿训练图原型，而是让低层分支直接学习“适合 img2img 起步”的结构图
- 在现有 Kandinsky img2img 输出的多候选里加入轻量 low-level-aware selection，让语义候选不要牺牲过多 SSIM/PixCorr

### 优先级 2：把 `Kandinsky img2img` 主线当成正式提交版本继续做完善

原因很简单：

- 它已经在 full test 上打赢了当前所有已完成 reconstruction 对照
- 它第一次把“高语义”和“较强低层结构”放在同一条线里
- 再往前推进，应该优先做这条线的多 seed、样例整理、提交级打磨
- 但最新证据表明 prototype selector 替换和二阶段 refinement 都不是主要瓶颈；真正值得做的是 low-level branch 质量和 candidate selection

### 优先级 3：在现有 img2img 框架里继续提升语义条件，而不是优先换 decoder

当前最值得继续追的方向是：

- 更好的 EEG -> semantic target 回归
- 更合理的 target 表征选择
- 继续利用 prototype 作为低层锚点，而不是把它重新升回独立主线

更具体地说，接下来的路线是：

1. 保留 `CLIP predictor + CLIP->Kandinsky adapter + low-level init` 作为主线，不再回退到 Kandinsky-only
2. 二阶段 refinement 已经在 val64 上证伪，不升 full test
3. 把 04-27 longrun 的三个 val64 候选升到 full-val：CLIP-heavy、balanced、SSIM-heavy 各保留一条
4. 等 HPC phase2 (`9710179`, `9710180`) 产出后，只按 val64/full-val gate 决定是否进入 frozen test
5. 下一轮如果继续追 reconstruction 上限，应转向训练新的 low-level init branch，而不是继续替换 retrieval prototype selector

这也是为什么 `CLIP target` 分支值得保留：

- 它提示“换 target”可能比“换大 decoder”更有价值
- 它已经接入现有 img2img 框架并成为当前主线；后续重点是让低层起点更稳定地服务这个语义条件

### 优先级 4：SDXL 保留为 sidecar，不再作为近期第一优先级 pivot

原因是：

- feasibility 已经验证完了
- full-test 对照已经说明它目前不如 Kandinsky img2img
- 在没有更强 prompt / adapter / conditioning 设计之前，不值得抢占主线资源

### 优先级 5：`rag_residual` 和 masked-pretrain 暂时降级，除非前提条件改变

现在至少可以明确说：

- 不是“这些路线还没来得及测”
- 而是“当前写法已经测过，而且没有赢”

所以如果以后还要回到这两条线，至少要先满足下面之一：

- `rag_residual` 方面：先明确怎么避免 gate collapse
- pretraining 方面：先换一个更贴近下游语义回归的预训练目标，或者明确新的迁移策略

## 附录：重要文件和目录地图

如果你要快速继续看这个项目，优先从下面这些地方开始。

### 文档和账本

- 项目说明：[README.md](/home/xiaoh/DeepLearning/EEG/README.md)
- 正式实验账本：[EXPERIMENT_LOG.md](/home/xiaoh/DeepLearning/EEG/EXPERIMENT_LOG.md)
- 远端镜像归档：[EXPERIMENT_LOG_hpc.md](/home/xiaoh/DeepLearning/EEG/EXPERIMENT_LOG_hpc.md)

### 当前最好 retrieval

- 当前正式 best：`loss_imgsoft_dir + posterior_cp_28` clean validation-selected z-score fusion
- 输出目录：[compliant_ensemble_loss_imgsoft_oldposterior_val_w7525](/home/xiaoh/DeepLearning/EEG/outputs_local/experiments/compliant_ensemble_loss_imgsoft_oldposterior_val_w7525)
- 指标：
  - `top1 = 0.6850`
  - `top5 = 0.9600`
- 重要约束：这是 non-transductive 结果，test 保持 `avg_trials=True`，不使用 test labels、test grid、test-batch distribution、trial TTA。
- 诊断上界：CSLS/SATTC-lite `0.7900/0.9900` 和 trial-TTA `0.80+` 都只能写作 non-compliant diagnostic，不能作为正式提交结果。

### Channel subset 配置

- 通道子集定义：[channel_subsets.json](/home/xiaoh/DeepLearning/EEG/configs/channel_subsets.json)
- 当前最有用的互补通道分支仍是 `posterior_cp_28`，但它现在是 fusion component，不再是全局 best 单模型。
- 对照组 `no_front_37` 在 seed2 同 split retrain 下 test `top1=0.5100`, `top5=0.8550`；能提供互补信号，但简单 validation-selected fusion 没有超过当前 best。

### 当前最好 reconstruction 训练与生成组件

- CLIP predictor：[best.pt](/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/reconstruction_clip_embed_v2_loss_imgsoft_local/seed_0/best.pt)
- CLIP→Kandinsky adapter：[best.pt](/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/clip_to_kandinsky_adapter_v1/seed_0/best.pt)
- Low-level init checkpoint：[best.pt](/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/reconstruction_dreamsim_topk4/seed_0/best.pt)
- Balanced 输出目录：`/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/reconstruction_compare/test/clip_pred_v2_adapter_blend_low85_post15_str030`
- SSIM-heavy 输出目录：`/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/reconstruction_compare/test/clip_pred_v2_adapter_lowlevel_topk4_str025`

### 旧 Kandinsky predictor checkpoint

- [best.pt](/home/xiaoh/DeepLearning/EEG/outputs/experiments/reconstruction_kandinsky_embed_v4_proxyselect/seed_0/best.pt)
- 这个 checkpoint 仍然是 Kandinsky-target 分支的历史主线，但不再是当前 reconstruction best。

### 纯 decode 参考结果

- `val64` 参考：[reconstruction_metrics.json](/home/xiaoh/DeepLearning/EEG/outputs/eval_compare/val64_seed0/kandinsky_predicted_v4_fast/reconstruction_metrics.json)
- full validation 参考：[reconstruction_metrics.json](/home/xiaoh/DeepLearning/EEG/outputs/eval_compare/full_val_seed0/kandinsky_predicted_v4_fast/reconstruction_metrics.json)

### 当前最好 reconstruction 的 full-test 主线结果

- Balanced 配置：`clip_pred_v2_adapter_blend_low85_post15_str030`
- 本地输出目录：`/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/reconstruction_compare/test/clip_pred_v2_adapter_blend_low85_post15_str030`
- 关键指标：
  - `eval_clip = 0.8212`
  - `eval_ssim = 0.3788`
  - `eval_pixcorr = 0.2335`
  - `eval_alex5 = 0.9103`
  - `eval_inception = 0.7652`

- SSIM-heavy 配置：`clip_pred_v2_adapter_lowlevel_topk4_str025`
- 本地输出目录：`/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/reconstruction_compare/test/clip_pred_v2_adapter_lowlevel_topk4_str025`
- 关键指标：
  - `eval_clip = 0.8160`
  - `eval_ssim = 0.3962`
  - `eval_pixcorr = 0.2302`
  - `eval_alex5 = 0.8921`
  - `eval_inception = 0.7387`

旧 semantic best 仍作为参照保留：

- 配置：`clip_pred_v2_adapter_posterior_old_str030`
- 本地输出目录：`/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/reconstruction_compare/test/clip_pred_v2_adapter_posterior_old_str030`
- 关键指标：`eval_clip = 0.8161`, `eval_ssim = 0.3289`, `eval_pixcorr = 0.2036`, `eval_alex5 = 0.9114`

### SSIM-only reference reconstruction 的 full-test 结果

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

- [best.pt](/home/xiaoh/DeepLearning/EEG/outputs/experiments/reconstruction_kandinsky_rag_residual_v1/seed_0/best.pt)

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

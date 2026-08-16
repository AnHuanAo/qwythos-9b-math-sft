# 背景
身边有位数学专业朋友，在做一个将大几百本GTM教材整理成电子知识库的项目，他已经将其中60多本push到github上，我会将本文及朋友的github链接放在文末，欢迎感兴趣的朋友交流~

而我在听到这个项目的第一念头，就是能不能基于这个知识库，去训练或者微调一个模型，让其数学能力及知识储备大幅提升，同时或许也能帮助朋友节省成本（api费用实在是太恐怖了）

而在反复思考后，便决定先尝试一次微调，便有了如下的实验：

- **模型**：Qwythos-9B（Qwen3.5 架构、94 亿参数、1M 上下文、Apache 2.0 开源）
- **数据**：从知识库里整理出的 40,443 条 SFT 样本（13,277 条"证明题" + 27,166 条"概念讲解"）
- **硬件**：AutoDL 租的 RTX 5090（32GB 显存），会员价/学生价2.78元/h
- **方法**：QLoRA 4-bit 量化 + LoRA（r=16, alpha=32），单卡微调约 5.5 小时
- **结果**：模型学会了"教材证明的皮"，但正确性仍需验证闭环

# 数据整理
朋友的知识库里，每个数学概念有5类文件：概念定义（yaml）、定理陈述（latex）、英文讲解（markdown）、证明、习题

但要微调一个模型，这样松散的数据难以直接使用，需要将其整合成jsonl等形式化的格式才有效，因此先花了些时间，将知识库整理成两类SFT任务：

```JSON
{"task": "proof", "instruction": "请证明以下定理。", "input": "<定理陈述>", "output": "<完整证明>", "meta": {...}}
{"task": "explain", "instruction": "请详细讲解以下数学概念。", "input": "<概念信息>", "output": "<概念讲解>", "meta": {...}}
```

最终得到 4w+ 条样本，覆盖代数、分析、拓扑、数论、概率、数学基础等 10 个数学领域。

# 环境部署
捣鼓环境基本上是每次不得不品的美味了。
首先就是自带镜像pytorch的版本可能会出问题，可以手动调整一下：
```shell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

然后就是 torchaudio 版本对不上（它需要 CUDA 13，而我用的是 CUDA 12.8），导入直接报错；解决办法是把 torchaudio 和 torchvision 都装成与 torch 2.8 配套的版本（torchaudio 2.8 / torchvision 0.23）

最令人抓狂的是，本来是使用 LLaMA-Factory作为训练框架，结果到模型加载的时候，tokenize全部正常，但一到训练就直接推出，退出码为0，查了很久都没发现是什么问题，只好绕开框架，直接用TRL手写训练脚本，终于跑通了。

但好景不长，在bf16精度下，9B模型需要18.8GB的显存（bf16，batch 2，长度4096），再加上激活值，直接OOM退出训练；随后将其换成4-bit，batch和长度分别保持在2和2048，依然在第50步OOM；改成batch 1后依旧OOM，最后判定为每次都是同一批长度2048的数据撞穿了显存墙；最后的最后，使用4-bit + batch 1 + 长度 1024 + gradient checkpointing，总算是跑起来了···

值得注意的是，Qwythos 用了混合注意力架构，不装专用算子的训练速度很慢，需要装 `causal-conv1d` + `flash-linear-attention` 才能保证 GPU 利用率保持高位

# 训练配置与结果
最终训练配置：

| 参数 | 值 |
|---|---|
| 量化 | 4-bit QLoRA（bitsandbytes） |
| LoRA | r=16, alpha=32, dropout=0，全线性层 |
| 最大长度 | 1024 token |
| batch | 1 × 梯度累积 8（等效 8） |
| 学习率 | 2e-4，cosine，warmup 3% |
| 轮数 | 1 epoch（5,005 步） |
| 耗时 | 约 5.5 小时 |

结果：

- train_loss：**0.7176**（初始约 0.84）
- eval_loss：**0.6905**（400 条验证样本）
- eval 平均 token 准确率：**0.812**

# 微调前后的风格变化

我拿同样的三道证明题分别问微调前/后的模型，差异非常直观：

**微调前（base）**——像是在"讲怎么证明"：

> 1. **Identify the theorem**... 2. **Determine the appropriate proof strategy**... 3. **Structure the proof**...

**微调后**——直接开证：

> The theorem is the Cauchy theorem... **Proof.** Proceed by induction on $|G|$...

用文本统计量化一下：

| 指标 | 微调前 | 微调后 |
|---|---|---|
| 元讨论词（Identify/Strategy/Plan） | 3 次 | 0 次 |
| "第几步"式规划列表 | 4 处 | 0 处 |

模型甚至学会了语料里的口癖——有一道题的回答里出现了"the proof section is truncated, I will reconstruct..."，这正是教材抽取流水线的常见措辞。**风格迁移是肉眼可见的，且可归因到训练语料。**

然鹅，就在我高兴的时候，让大模型检查Qwythos的证明过程，发现了致命的错误，Qwythos声称"每个非中心共轭类的大小都被 p 整除"，虽然我看不懂，但大模型看得懂啊（笑）。

这说明 SFT 只教会了模型"写证明的样子"，没教会它写出正确的证明。没有验证闭环，它会把错误步骤写得信心满满。

想要改进这一缺点，可以再整理一份习题-解答数据集，再进行微调，这也是下一步我打算做的（不知道要多久之后）。

# 总结
这次实验也是我第一次微调一个基模，中间也通过各种配环境、调参数，大致掌握了LoRA微调的流程


## 资源链接

- Qwythos 权重下载：[empero-ai/Qwythos-9B-Claude-Mythos-5-1M](https://huggingface.co/empero-ai/Qwythos-9B-Claude-Mythos-5-1M)（Apache 2.0）
- 朋友的 mathpkg 数学知识库仓库：[XuanzhengZhou/mathpkg](https://github.com/XuanzhengZhou/mathpkg)

# 零基础保姆级指南：AutoDL + RTX 5090 微调 Qwythos-9B

> 目标：把你已经生成的 40,443 条数学数据（证明题 + 知识讲解），微调到 Qwythos-9B 模型上。
> 模型：`empero-ai/Qwythos-9B-Claude-Mythos-5-1M`（开源 Apache 2.0，数学能力很强）。
> 你只需要：会复制粘贴命令 + 能看懂"成功/失败"的提示。

---

## 第 0 步：先搞懂几个词（很重要，别跳过）

| 词 | 大白话解释 |
|---|---|
| **服务器 / 租的卡** | 你租的不是"一张显卡"，是一台**远方的电脑**（带显卡）。它没有屏幕，你用键盘命令远程指挥它。 |
| **终端 / 命令行** | 一个"只能打字、不能点鼠标"的窗口。你输入一行命令按回车，电脑执行并显示结果。 |
| **SSH** | 一种"远程登录"协议。用 SSH 连上后，你的终端就变成了那台服务器的终端，敲的命令都在服务器上执行。 |
| **conda 环境** | 相当于给服务器开一个"独立小房间"，在这个房间里装软件不会弄乱房间外的系统。不同项目用不同房间，互不干扰。 |
| **pip** | Python 的"应用商店"。`pip install 某某` = 从网上下载并安装某某软件。 |
| **模型下载** | Qwythos-9B 是一个 18GB 左右的文件包，训练前先要把它下载到服务器硬盘上。 |
| **冒烟测试（smoke test）** | 新买电器先通电试试有没有冒烟。训练前只拿 **200 条**数据跑几步，确认"环境没坏、流程能走通"，再全量跑 4 万条。避免你等 3 小时才发现一开始就错了。 |
| **loss（损失值）** | 训练时的"错题率"指标，**越小越好**。如果 loss 在下降，说明模型在学习。 |
| **LoRA** | 一种省钱的微调方法：不重训整个模型，只在模型上"贴便利贴"（加一层小补丁），效果接近全量训练，但省时间省显存。 |
| **epoch（轮次）** | 把所有训练数据完整过一遍 = 1 个 epoch。 |
| **checkpoint（检查点）** | 训练中途自动保存的"存档"。训练断了可以从存档继续，不用从头来。 |

---

## 第 1 步：登录你的服务器（两种方式，选一种）

### 方式 A：网页版 JupyterLab（强烈推荐新手用这个）

1. 打开 AutoDL 控制台（autodl.com），找到你租的实例；
2. 点右边的 **「JupyterLab」** 按钮，浏览器会打开一个网页界面；
3. 网页左上角菜单 **File → New → Terminal**，会打开一个"终端"窗口；
4. 这个终端就是服务器的命令行，**后面的命令都贴在这里执行**（不需要 SSH）。

> 好处：不用配置任何东西，浏览器里直接操作，还能用网页的文件管理器拖拽上传数据。

### 方式 B：本地 PowerShell 用 SSH

1. 打开 PowerShell：按键盘 **Win 键**，输入 `powershell`，回车；
2. 回到 AutoDL 控制台，找到 **「SSH 指令」**，里面有一行类似：
   ```
   ssh -p 12345 root@connect.xxx.seetacloud.com
   ```
3. 把这行**整段复制**，粘贴到 PowerShell 里，回车；
4. 提示 `password:` 时，粘贴控制台显示的密码（**密码输入时屏幕不显示任何字符，这是正常的**），直接回车；
5. 看到类似 `root@autodl-container-...:~#` 的提示符，说明登录成功。

> 粘贴技巧：PowerShell 里复制是 `Ctrl+C`，**粘贴是右键或 `Ctrl+Shift+V`**。

---

## 第 2 步：检查服务器（确认显卡在）

在终端里输入下面命令，回车：

```bash
nvidia-smi
```

你会看到一张表格，里面有 **NVIDIA GeForce RTX 5090** 和显存 32GB。看到就说明显卡正常。

再看一下当前文件夹（刚登录默认在用户目录）：

```bash
pwd        # 显示当前在哪个目录（Print Working Directory）
ls         # 列出当前目录里的文件（LiSt）
```

---

## 第 3 步：创建独立环境 + 安装 torch（关键！）

`torch`（PyTorch）是 AI 训练的核心软件。你的 5090 是新一代显卡（Blackwell，代号 sm_120），
**必须装 cu128 版本的 torch（≥2.7）**，否则显卡不干活。旧版本/默认版本都不行。

逐条复制执行（每条按一次回车）：

```bash
# 1. 创建环境。解释：conda create=创建房间；-p 后面是房间的位置；
#    放在 /root/autodl-tmp 是因为这是"数据盘"（50GB），系统盘只有 30GB 装不下。
conda create -p /root/autodl-tmp/envs/mathpkg python=3.12 -y

# 2. 进入这个房间
conda activate /root/autodl-tmp/envs/mathpkg

# 3. 安装 torch。--index-url 意思是"从 cu128 这个官方仓库下载"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 4. 验证。期望看到：torch 版本号、True、(12, 0)
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability(0))"
```

**验证结果怎么看**：
- 如果输出类似 `2.9.0 True (12, 0)` → 成功！
- 如果输出 `False` 或报错 → 说明 torch 版本不对，把第 3 条命令重跑一遍（或截图报错给我）。

> 如果提示 `conda: command not found`：先执行 `source /root/miniconda3/etc/profile.d/conda.sh` 再重新试。

---

## 第 4 步：安装其他依赖

```bash
# 常用 AI 库：transformers=模型加载、datasets=数据集、trl=训练器、bitsandbytes=4bit量化、peft=LoRA
pip install -U transformers datasets accelerate trl bitsandbytes peft triton

# Qwythos 特有的注意力算子（装不上也能继续，只是慢。先试，失败就跳过）
pip install flash-linear-attention causal-conv1d

# 安装 LLaMA-Factory（我们的训练工具）
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .
```

> `pip install -e .` 里的 `.` 表示"当前文件夹"（也就是刚 clone 下来的 LLaMA-Factory 文件夹）。
> **不要**用 `pip install -e .[torch]`，那会重新装 torch 覆盖掉 cu128 版本。

---

## 第 5 步：下载模型（国内必须用镜像）

模型在 HuggingFace 上，国内直连很慢/连不上，所以先设置"镜像站"环境变量：

```bash
# 告诉下载工具：去 hf-mirror.com 这个国内镜像下载
export HF_ENDPOINT=https://hf-mirror.com

# 建目录
mkdir -p /root/autodl-tmp/models

# 下载模型（18GB 左右，根据网速可能要 10-40 分钟）
# 注意：新版 huggingface_hub 用 hf 命令，huggingface-cli 已停用
hf download empero-ai/Qwythos-9B-Claude-Mythos-5-1M --local-dir /root/autodl-tmp/models/qwythos-9b
```

下载完验证能加载：

```bash
python - <<'EOF'
from transformers import AutoModelForImageTextToText, AutoTokenizer
m = "/root/autodl-tmp/models/qwythos-9b"
tok = AutoTokenizer.from_pretrained(m)
model = AutoModelForImageTextToText.from_pretrained(m, dtype="bfloat16", device_map="auto")
print("模型加载成功!")
EOF
```

看到 `模型加载成功!` 就 OK。

---

## 第 6 步：上传训练数据（两种方式）

### 方式 A：JupyterLab 网页拖拽（最简单）

1. 在 JupyterLab 里先建目录，打开 Terminal 执行：
   ```bash
   mkdir -p /root/autodl-tmp/data
   ```
2. 在左侧文件栏导航到 `autodl-tmp/data`（点目录名进入）；
3. 把本地电脑上的这两个文件**直接拖进网页**：
   - `E:\My_Project\mathpkg\v1训练数据\autodl\train.jsonl`
   - `E:\My_Project\mathpkg\v1训练数据\autodl\dataset_info.json`

### 方式 B：本地 PowerShell 用 scp 传

在**本地 PowerShell**（不是服务器终端）执行，把 `-P 12345` 和 `connect.xxx` 换成你控制台里真实的：

```powershell
scp -P 12345 "E:\My_Project\mathpkg\v1训练数据\autodl\train.jsonl" root@connect.xxx.seetacloud.com:/root/autodl-tmp/data/
scp -P 12345 "E:\My_Project\mathpkg\v1训练数据\autodl\dataset_info.json" root@connect.xxx.seetacloud.com:/root/autodl-tmp/data/
```

输入密码后看到进度条走完就是成功。

上传完，在服务器终端确认文件在：

```bash
ls -lh /root/autodl-tmp/data/
# 应看到 train.jsonl（约 38MB）和 dataset_info.json
```

再把训练配置文件也传上去（或者直接复制下面内容到服务器上创建，见第 7 步）。

---

## 第 7 步：训练配置文件

把 `llamafactory_train.yaml` 也传到 `/root/autodl-tmp/data/`（同样用拖拽或 scp）。

文件内容（你也可以在服务器终端里用 `nano` 创建）：

```yaml
model_name_or_path: /root/autodl-tmp/models/qwythos-9b
template: qwen3
stage: sft
finetuning_type: lora

dataset: mathpkg_v1
dataset_dir: /root/autodl-tmp/data
cutoff_len: 2048

learning_rate: 2.0e-4
num_train_epochs: 1.0
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
max_samples: 200
lr_scheduler_type: cosine
warmup_ratio: 0.03
logging_steps: 10
save_steps: 500
eval_strategy: steps
eval_steps: 500
val_size: 0.01
optim: adamw_torch

lora_rank: 16
lora_alpha: 32
lora_dropout: 0.0
lora_target: all

bf16: true
quantization_bit: 4
gradient_checkpointing: true

output_dir: /root/autodl-tmp/output/mathpkg_v1
plot_loss: true
```

**每个参数是干嘛的（新手版）**：

| 参数 | 含义 |
|---|---|
| `template: qwen3` | 模型对话格式（Qwythos 继承 Qwen3.5）。报错就改成 `default` |
| `cutoff_len: 4096` | 每条数据最多截断成 4096 个 token（约几千字）。你的样本很短，够用 |
| `learning_rate: 2.0e-4` | 学习速度。太快学不稳，太慢学不动 |
| `num_train_epochs: 1.0` | 全部数据过 1 遍 |
| `per_device_train_batch_size: 2` × `gradient_accumulation_steps: 8` | 一次喂 2 条、攒 8 次再更新 = 等效一次 16 条（更稳） |
| `max_samples: 200` | **冒烟测试开关**：只取 200 条。全量训练时删掉这行 |
| `val_size: 0.01` | 切 1% 数据当"考试卷"，训练时顺便看模型在没见过的题上表现 |
| `save_steps: 500` | 每 500 步存一次 checkpoint（存档） |
| `lora_rank/alpha` | LoRA 补丁的大小和强度，16/32 是常用默认值 |
| `bf16: true` | 用半精度训练，省显存（5090 支持） |
| `output_dir` | 训练结果存哪 |

---

## 第 8 步：冒烟测试（先跑 200 条）

**冒烟测试是干嘛的**：就像新电脑先开机看看会不会冒烟。全量训练可能要跑 2-3 小时，
如果环境有问题（比如版本不匹配），跑 5 分钟就报错最好，别等 2 小时才发现。

执行：

```bash
cd /root/autodl-tmp/LLaMA-Factory   # 或你 clone 的位置
llamafactory-cli train /root/autodl-tmp/data/llamafactory_train.yaml
```

第一次运行会：加载模型（1-3 分钟）→ 处理数据（1-2 分钟）→ 开始训练。

**看什么**：
1. 屏幕上出现 `loss = 1.2 → 1.1 → 1.0 ...`，数字在变小 → 正常；
2. 出现 `loss = nan` 或红字报错 → 有问题，把报错截图发我；
3. 另开一个终端跑 `nvidia-smi`，看显存使用（32GB 内就是安全的）。

常见的冒烟测试报错和解法：

| 报错 | 解法 |
|---|---|
| `template 'qwen3' not found` | yaml 里 `template` 改成 `default` 再跑 |
| `CUDA out of memory` | 显存不够。yaml 里 `per_device_train_batch_size` 改成 `1`，或 `cutoff_len` 改成 `2048` |
| `bitsandbytes ... sm_120` | `pip install -U bitsandbytes`；还不行就把 yaml 里加 `quantization_bit: 4` 去掉，用 bf16（32GB 够） |
| 提示缺某某模块 | `pip install 某某` 补装 |

冒烟测试能顺利跑起来、loss 在下降，就可以进入全量训练。

---

## 第 9 步：全量训练

1. 把 yaml 里的 **`max_samples: 200` 这一行删掉**（可以用网页版的文件编辑器，或终端 `nano`）；
2. 重新执行第 8 步的命令；
3. 训练时间：40,443 条约 1500 万 token，5090 上预计 **1.5-3 小时**，远小于 25 小时预算；
4. 中途会看到 `loss` 和 `eval_loss`（验证集错题率）。**eval_loss 还在下降 → 学得不错**；训练结束后想更好，把 `num_train_epochs` 改成 `2` 再跑；
5. **训练中断了别慌**：模型保存在 `output_dir` 里，重新跑会自动从最新 checkpoint 继续；
6. 日志和图表会存到 `output_dir/training_loss.png`，训练完可以下载看 loss 曲线。

---

## 第 10 步：合并模型 + 部署 + 测试

训练完得到的是"补丁"（LoRA 适配器），先和原模型合并成完整模型：

```bash
llamafactory-cli export \
  --model_name_or_path /root/autodl-tmp/models/qwythos-9b \
  --adapter_name_or_path /root/autodl-tmp/output/mathpkg_v1 \
  --template qwen3 \
  --finetuning_type lora \
  --export_dir /root/autodl-tmp/output/mathpkg_v1_merged \
  --export_size 4 \
  --export_device cpu
```

然后部署成一个"聊天接口"（用 vLLM）：

```bash
pip install -U vllm
vllm serve /root/autodl-tmp/output/mathpkg_v1_merged --max-model-len 65536 --port 8000
```

看到 `Uvicorn running on http://0.0.0.0:8000` 就说明部署成功（这个终端保持开着）。

**另开一个终端**测试：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/root/autodl-tmp/output/mathpkg_v1_merged",
    "messages": [{"role": "user", "content": "请证明以下定理：设 G 是有限群，p 是整除 |G| 的素数，则 G 中存在 p 阶元。"}],
    "max_tokens": 4096,
    "temperature": 0.6,
    "top_p": 0.95
  }'
```

返回 JSON 里的 `message.content` 就是模型的回答。拿同样的题问**微调前**和**微调后**两个模型，对比质量。

---

## 第 11 步：关机省钱

- 训练完、测试完，回 AutoDL 控制台点 **「关机」**；
- 关机后只按"数据盘存储"收费（很便宜），**你的模型和数据都还在** `/root/autodl-tmp`；
- 下次要用：开机 → SSH/JupyterLab 登录 → `conda activate /root/autodl-tmp/envs/mathpkg` → 继续干活；
- 只想整理数据不想用显卡时，可以开 **「无卡模式」**，几乎不花钱。

---

## 常见问题速查

| 问题 | 解决 |
|---|---|
| `ssh: command not found` | 别用 SSH 了，改用方式 A（JupyterLab 网页终端） |
| 密码输进去没反应/屏幕不显示 | 正常，直接回车 |
| torch 报 `no kernel image available for execution` | 说明 torch 不是 cu128，重跑第 3 步的 pip install |
| `conda: command not found` | 先执行 `source /root/miniconda3/etc/profile.d/conda.sh` |
| 下载模型很慢/失败 | 确认执行过 `export HF_ENDPOINT=https://hf-mirror.com`，再重试 |
| 训练时 loss 是 `nan` | 停掉，把 `learning_rate` 改小（比如 `1.0e-4`）再试 |
| 显存不够（OOM） | batch 改 1、cutoff 改 2048，或换 bf16（去掉 4bit） |
| 不知道报错什么意思 | **把红字报错整段复制发给我**，我帮你看 |

---

## 本地文件对应

- `train.jsonl`：40,443 条训练数据（sharegpt 格式），上传用
- `dataset_info.json`：数据集的"说明书"，LLaMA-Factory 靠它认数据
- `llamafactory_train.yaml`：训练配置
- `prep_data.py`：想重新生成数据时用（比如 `--task proof` 只要证明题、`--max 200` 冒烟）
- `train_unsloth.py`：备选训练脚本（主方案 LLaMA-Factory 不行时再用）

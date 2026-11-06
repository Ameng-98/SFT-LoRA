---
name: qwen-lora-sft
description: 用 LoRA/QLoRA 对 Qwen、Llama、Mistral 等因果语言模型做监督微调（SFT）。当任务涉及「用 instruct/alpaca/对话数据微调大模型」「配置 peft + trl 训练参数」「排查显存不足、版本不兼容、中文乱码、pad_token 报错」「合并 LoRA 权重或验证微调效果」时使用本技能。
---

# Qwen LoRA 监督微调（SFT）

把 `train_qwen.py` 里跑通的一套流程（加载模型 → 挂 LoRA → 格式化数据 → SFTTrainer 训练）
抽象成可复用的技能：换模型、换数据、换超参，只需要改一份 JSON 配置。

## 何时使用

适用：
- 把某个领域的数据（医疗问答、客服话术、代码等）注入 Qwen2.5 / Llama / Mistral 等基座
- 显存有限，用 LoRA/QLoRA 而不是全参数微调
- 需要一条固定流水线：数据校验 → 训练 → 推理验证 → 合并部署

不适用：预训练（从零训练）、DPO/PPO/GRPO 等偏好对齐。这类任务可以复用本技能的
数据解析、模型加载和显存估算经验，但训练器要换成 `DPOTrainer` 等。

## 目录结构

```
skills/qwen-lora-sft/
├── SKILL.md                      # 本文件：流程与关键约束
├── scripts/
│   ├── _common.py                # 公共工具：UTF-8 修复、精度选择、配置合并、字段格式化
│   ├── env_report.py             # 1) 环境自检（版本/GPU/bf16/路径）
│   ├── prepare_dataset.py        # 2) 数据校验、清洗、去重、统计
│   ├── finetune.py               # 3) LoRA SFT 训练（支持 --dry-run）
│   ├── infer_lora.py             # 4) 加载 adapter 做单轮/多轮推理
│   └── merge_lora.py             # 5) 合并 adapter → 完整模型
├── assets/
│   ├── finetune_config.gpu.json  # GPU 训练配置模板
│   ├── finetune_config.cpu.json  # CPU 小样本冒烟配置模板
│   └── data.example.json         # alpaca 格式样例数据
└── reference/
    ├── config_reference.md       # 每个配置字段的含义与取值
    ├── hyperparameters.md        # LoRA / 训练超参调法、显存估算
    └── troubleshooting.md        # 常见报错 → 原因 → 处理
```

## 环境基线（本项目已验证）

```text
Python 3.11 / torch 2.14(+cpu) / transformers 5.18 / trl 1.17 / peft 0.21 / datasets 5.1
```

这套版本组合有几个**会直接报错**的坑，脚本里已经做了自动兼容，但排查时必须知道：

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `unexpected keyword argument 'max_seq_length'` | trl 1.x 改名为 `max_length` | `finetune.build_training_args()` 已按 `SFTConfig` 签名自动切换 |
| `'torch_dtype' is deprecated` 或 v5 下不生效 | transformers v5 改用 `dtype` | `_common.dtype_kwarg_name()` 自动判断版本 |
| `SFTTrainer.__init__() got an unexpected keyword 'tokenizer'` | trl 1.x 用 `processing_class` | 训练脚本按签名自动选择传参名 |
| batch padding 报 `pad_token` 相关错误 | Qwen 等模型没有 pad_token | 统一 `tokenizer.pad_token = tokenizer.eos_token` |
| Windows 读数据/打印中文乱码 | 默认 GBK | `setup_utf8()` 强制 UTF-8，并 patch `Path.read_text` |
| DataLoader 在 Windows 卡死 | 多进程 fork 问题 | `dataloader_num_workers=0` |

## 标准流程（6 步）

以下命令都在**技能目录**下执行（`skills/qwen-lora-sft`），路径可换成绝对路径。

### 0. 安装依赖

```bash
pip install torch transformers datasets peft trl accelerate
```

### 1. 环境自检

```bash
python scripts/env_report.py --config assets/finetune_config.gpu.json
```

看到 `✅ 依赖齐全` 再往下走。它会打印当前 GPU、bf16 支持情况、trl 用的是
`max_length` 还是 `max_seq_length`，以及配置里的模型/数据路径是否存在。

### 2. 数据准备与校验

```bash
python scripts/prepare_dataset.py \
  --input ./dataset/raw.json \
  --output ./dataset/train.json \
  --format alpaca \
  --report ./dataset/report.json
```

- 默认去重，并按 `--min-output-chars`（默认 10）过滤过短回答
- `--fixed-instruction` 可以把 alpaca 的 instruction 统一成一句话
- 报告里的 `drop_reasons` 能看出是字段名不对、还是数据太脏

### 3. 准备配置

复制模板再改 4 个关键字段：

```bash
cp assets/finetune_config.gpu.json my_config.json
```

```jsonc
{
  "model":   { "path": "D:/models/Qwen2.5-1.5B-Instruct" },  // 本地目录或 Qwen/Qwen2.5-1.5B-Instruct
  "data":    { "path": "./dataset/train.json" },             // 上一步产出的文件
  "output":  { "dir": "./output/my_lora" },                  // adapter 输出目录
  "training":{ "per_device_train_batch_size": 2 }            // 按显存调整
}
```

其余字段的含义见 `reference/config_reference.md`。

### 4. dry-run 校验（强烈建议）

```bash
python scripts/finetune.py --config my_config.json --dry-run
```

不加载模型、不训练，只做：
- 检查模型/数据路径
- 加载 tokenizer，用真实 chat template 渲染前 2 条样本（**这是最容易出错的地方**）
- 打印 LoRA 目标层、等效 batch、预计训练步数

如果预览文本里没有 `<|im_start|>user` / `<|im_start|>assistant` 之类的模板标记，
说明 tokenizer 缺 chat_template，需要换 Instruct 版模型或改用 `text` 格式。

### 5. 正式训练

```bash
python scripts/finetune.py --config my_config.json
```

训练产物：
```text
output/my_lora/
├── adapter_config.json / adapter_model.safetensors   # LoRA 权重（只有几十 MB）
├── tokenizer 相关文件
├── checkpoint-*/                                     # 按 save_strategy 保存
├── training_summary.json                             # 步数、耗时、损失、超参快照
└── run_config.json                                   # 完整配置，便于复现
```

断点续训：
```bash
python scripts/finetune.py --config my_config.json --resume-from-checkpoint
```

### 6. 验证与合并

```bash
# 单轮快速验证
python scripts/infer_lora.py --model "D:/models/Qwen2.5-1.5B-Instruct" \
  --adapter ./output/my_lora --prompt "介绍一下你自己"

# 多轮交互（不带 --prompt）
python scripts/infer_lora.py --model "D:/models/Qwen2.5-1.5B-Instruct" --adapter ./output/my_lora

# 合并成独立模型（部署给 vLLM/Ollama 用）
python scripts/merge_lora.py --base-model "D:/models/Qwen2.5-1.5B-Instruct" \
  --adapter ./output/my_lora --output ./output/my_lora_merged
```

验证要点：问一个训练集内的问题（应明显带上领域风格）+ 一个训练集外的问题
（看是否出现灾难性遗忘/胡言乱语）。

## 安装与复用

技能是纯文件，没有额外依赖打包，复制目录即可：

```powershell
# 1) 在本项目内使用：直接在 skills/qwen-lora-sft 下执行命令
cd skills\qwen-lora-sft

# 2) 复制到其它项目
Copy-Item -Recurse skills\qwen-lora-sft D:\new_project\skills\

# 3) 安装为全局技能（支持 Agent Skills 的客户端会扫描该目录）
Copy-Item -Recurse skills\qwen-lora-sft "$env:USERPROFILE\.claude\skills\qwen-lora-sft"
```

```bash
# Linux / macOS
cp -r skills/qwen-lora-sft ~/.claude/skills/qwen-lora-sft
```

两个使用约定：

- **配置里的相对路径以「执行命令时的当前目录」为准**，不是相对配置文件。
  因此建议在项目根目录执行（`python skills/qwen-lora-sft/scripts/finetune.py ...`），
  或直接在配置里写绝对路径。Windows 路径在 JSON 里要写双反斜杠 `E:\\models\\Qwen`。
- `scripts/_common.py` 是脚本的公共模块，必须与其它脚本保持在同一目录，
  单独把某个脚本拷走会导入失败。

## 新项目复用的最短路径

1. 复制整个 `qwen-lora-sft/` 目录到新项目（或安装到 `~/.claude/skills/` 全局复用）
2. `python scripts/prepare_dataset.py` 校验新数据
3. 新建 `my_config.json`，只改 `model.path / data.path / output.dir`
4. `--dry-run` → 正式训练 → `infer_lora.py` 验证

模型架构不是 Qwen/Llama 系时，只要 `lora.target_modules` 写 `"auto"`，
脚本会读 `config.json` 的 `model_type` 推断目标层；不认识就退回 `"all-linear"`（最稳但参数略多）。

## 检查清单（训练前逐条确认）

- [ ] `env_report.py` 全部 ✅，没有缺包
- [ ] `pad_token` 已兜底、`chat_template` 存在（dry-run 会打印）
- [ ] `data.format` 与实际字段一致；预览文本包含完整的 user/assistant 两段
- [ ] `per_device_train_batch_size × gradient_accumulation_steps` 的等效 batch ≥ 8
- [ ] `max_length` 与数据实际长度匹配（绝大多数中文问答 512 足够）
- [ ] 输出目录不在数据集目录里（避免把 checkpoint 混进数据）
- [ ] 训练后 `training_summary.json` 里的 loss 有下降趋势

## 参考文档

- 配置字段全解：`reference/config_reference.md`
- 超参与显存估算：`reference/hyperparameters.md`
- 报错排查表：`reference/troubleshooting.md`

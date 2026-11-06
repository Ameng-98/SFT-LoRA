# 电商智能客服 LoRA 微调（Qwen2.5-1.5B-Instruct）

对应《电商智能客服大模型微调系统需求规格说明书》V1.0 的微调阶段交付物：
用 LoRA/SFT 让通用模型具备六大业务场景的客服应答能力，并守住事实/权限边界。

## 1. 环境与硬件

| 项 | 值 |
| --- | --- |
| Python | 3.11.16（conda 环境 `langchain`） |
| 解释器 | `C:\Users\16096\.conda\envs\langchain\python.exe` |
| torch | 2.14.1+cpu（无 CUDA） |
| transformers | 5.18.0（`from_pretrained(..., dtype=...)`） |
| trl | 1.17.1（`max_length` + `processing_class`） |
| peft | 0.21.2 |
| datasets / accelerate | 5.1.0 / 1.15.0 |
| CPU / 内存 | AMD Ryzen 7 8C16T / 31.3 GB |
| GPU | 无独显（Radeon 780M 集显，训练不使用） |

> 因此本次是 **CPU 训练**：加载精度 float32、`use_cpu=true`、`fp16/bf16=false`，
> 由 skill 的 `resolve_precision()` 自动处理。

## 2. 基座模型（本地，已下载）

```
E:\Desktop\wolin\zuoye\week3_day4\test\model\models\Qwen--Qwen2.5-1.5B-Instruct\snapshots\master
```

`config.json`：`Qwen2ForCausalLM` / `model_type=qwen2` / 28 层 / hidden 1536 / bf16 权重 2.88 GB。

## 3. 目录结构

```
LoRA_train/
├── config/
│   ├── finetune_cpu.json        # 正式训练配置（2 轮，603 条全量）
│   └── finetune_smoke.json      # 冒烟配置（8 条、2 步，只验证流程）
├── data/
│   ├── train_messages.json      # 603 条三角色 messages 训练数据（脚本生成）
│   └── dataset_report.json      # 数据统计报告（类别/难度/长度分位）
├── scripts/
│   ├── build_sft_dataset.py     # alpaca → system/user/assistant 三角色
│   └── test_ecommerce_model.py  # 18 条验收用例评测（lora / base / both）
├── logs/                        # 后台训练日志 train.log / train.err.log
└── output/                      # 训练产物与评测报告
```

## 4. 数据

原始数据：`E:\Desktop\wolin\project_final\dataset\ecommerce_customer_service_sft_600.json`

| 项 | 值 |
| --- | --- |
| 样本数 | 603（product 100 / order 103 / logistics 100 / after_sales 100 / promotion 100 / complaint 100） |
| 难度 | easy 205 / medium 199 / hard 199 |
| 空 instruction | 3 条，已用出现 600 次的系统提示词回填（需求 14.1） |
| 去重 | 0 条重复 |
| 用户输入字符 | 平均 56 / 最长 79 |
| 客服回复字符 | 平均 74 / 最长 109 |
| chat template 后 token | 平均 165.6 / p95 183 / 最长 205 |

因为最长 205 token，`max_length` 取 **256** 即可零截断，同时把 CPU 计算量压到最低。

### 为什么不用 skill 的 alpaca 通道

需求 15.1 要求训练与推理统一为 `system + user + assistant` 三角色 chat template；
skill 的 `alpaca` 通道会把 `instruction` 拼进 user 内容、丢掉 system 角色。
所以先用 `build_sft_dataset.py` 转成 `messages` 列，再用 skill 的 `format: "messages"` 训练。
dry-run 输出已确认渲染结果为：

```
<|im_start|>system
你是综合数码电商平台的在线客服。……不得索取密码和验证码。<|im_end|>
<|im_start|>user
<业务上下文>
用户问题：<用户问题><|im_end|>
<|im_start|>assistant
<客服回复><|im_end|>
```


## 5. 训练配置（`config/finetune_cpu.json`）

| 分组 | 参数 | 值 | 依据 |
| --- | --- | --- | --- |
| LoRA | r / alpha / dropout | 16 / 32 / 0.1 | skill 默认；603 条数据量属小样本，r=16 足够 |
| LoRA | target_modules | auto → q,k,v,o,gate,up,down | 1.18% 参数可训练（18.46M / 1.56B） |
| 数据 | format | messages | 需求 15.1 三角色协议 |
| 训练 | per_device_batch × grad_accum | 1 × 8 = 等效 batch 8 | CPU 内存受限，等效 batch 落在 skill 建议的 8~32 |
| 训练 | epochs | 2 | 603 条小数据 + 防复读/过拟合（需求 11.2） |
| 训练 | lr / scheduler / warmup | 1e-4 / cosine / 5 步 | skill 推荐组合，平稳预热 + 后期退火 |
| 训练 | max_length | 256 | 数据最长 205 token，零截断 |
| 训练 | max_grad_norm | 0.3 | 防梯度爆炸 |
| 训练 | save_strategy | epoch（保留 2 个） | 便于逐轮评估/回退 |

总步数 = ceil(603 / 8) × 2 = **152 步**。

## 6. 运行步骤

```powershell
$py = "C:\Users\16096\.conda\envs\langchain\python.exe"
$skill = "E:\Desktop\wolin\project_final\LoRA_skills\qwen-lora-sft"
Set-Location "E:\Desktop\wolin\project_final\LoRA_train"

# 0) 生成三角色训练数据（已完成，可重复执行）
& $py scripts\build_sft_dataset.py `
  --input "E:\Desktop\wolin\project_final\dataset\ecommerce_customer_service_sft_600.json" `
  --output "data\train_messages.json" --report "data\dataset_report.json" `
  --tokenizer "E:\Desktop\wolin\zuoye\week3_day4\test\model\models\Qwen--Qwen2.5-1.5B-Instruct\snapshots\master"

# 1) 环境自检
& $py "$skill\scripts\env_report.py" --config config\finetune_cpu.json

# 2) dry-run（不加载模型，校验配置+chat template）
& $py "$skill\scripts\finetune.py" --config config\finetune_cpu.json --dry-run

# 3) 冒烟训练（8 条 2 步，约 1.5 分钟，验证全链路）
& $py "$skill\scripts\finetune.py" --config config\finetune_smoke.json

# 4) 正式训练（152 步，CPU 约 2.5 小时）
& $py "$skill\scripts\finetune.py" --config config\finetune_cpu.json

# 5) 18 条验收用例评测（基座 vs 微调）
& $py scripts\test_ecommerce_model.py --model both --adapter output\lora_adapter
```

断点续训：`--resume-from-checkpoint`（自动找最新 checkpoint）。

## 7. 产物

```
output/lora_adapter/
├── adapter_config.json          # LoRA 结构（r=16, alpha=32, 7 个目标层）
├── adapter_model.safetensors    # LoRA 权重（约 74 MB）
├── tokenizer*                   # 与训练一致的 tokenizer 文件
├── checkpoint-76 / -152/        # 每轮快照
├── training_summary.json        # 步数、耗时、loss、超参快照
└── run_config.json              # 完整配置，便于复现
output/test_results_*.md         # 18 条用例评测结果
```

## 8. 验收对应关系

| 验收项 | 落地方式 |
| --- | --- |
| ACC-001~006 六大场景 | 603 条按 6 类均衡训练，评测结果逐类对比 |
| ACC-007 信息不足追问 | 训练集含信息不足样本；评测用例 PRODUCT_002/003 等检查 |
| ACC-008 事实边界 | system 提示词 + 训练样本强化“只能依据上下文” |
| ACC-009 权限边界 | `test_ecommerce_model.py` 自动检测“已为您取消/一定送达”等越权承诺 |
| ACC-010 无复读 | 自动检测同一句连续重复 ≥3 次 |
| ACC-011 无幻觉续写 | 推理侧 `truncate_at_marker()` 截断编造的对话轮次，并做自动标记 |
| ACC-012 基座对比 | `--model both` 同一套解码参数下逐条对比，输出 Markdown |


## 9. 后台运行与监控

正式训练以独立进程在后台运行（CPU 单次约 2.5 小时），不占用交互终端：

```powershell
# 启动（已在本次执行）
$p = Start-Process "C:\Users\16096\.conda\envs\langchain\python.exe" `
  -ArgumentList @("-u", "<skill>\scripts\finetune.py", "--config", "config\finetune_cpu.json") `
  -WorkingDirectory "E:\Desktop\wolin\project_final\LoRA_train" `
  -RedirectStandardOutput logs\train.log -RedirectStandardError logs\train.err.log `
  -WindowStyle Hidden -PassThru

# 查看进度（步骤、loss、产物、评测状态）
powershell -ExecutionPolicy Bypass -File .\scripts\status.ps1
```

训练结束后 `scripts\auto_eval_after_train.ps1` 会自动执行 18 条用例评测，
报告写入 `output\test_results_both_<时间戳>.md`，过程记录在 `logs\eval.log` / `logs\eval_watch.log`。

一键重跑（数据 → 自检 → dry-run → 训练 → 评测）：`run_all.ps1`。

## 10. 本次已知限制

- **无 GPU**：1.56B 参数 float32 CPU 训练约 57 秒/步（8 条/步），152 步约 2.5 小时；
  如需更快，在 CUDA 机器上把 `training.bf16` 设为 `true`、`per_device_train_batch_size` 调到 2~4。
- **全序列 loss**：沿用 skill 的“text 字段”训练方式，loss 覆盖 system/user/assistant 全部 token；
  若追求更高质量，可改为 completion-only loss（只对 assistant 段计损）。
- **自动检查为启发式**：复读/幻觉/越权通过正则+句频判断，最终以人工复核 Markdown 结果为准。


## 11. 人工手动测试

启动入口：`E:\Desktop\wolin\project_final\LoRA_train\start_manual_test.ps1`

```powershell
cd E:\Desktop\wolin\project_final\LoRA_train

.\start_manual_test.ps1                 # 交互对话（LoRA 微调模型）
.\start_manual_test.ps1 -Mode base      # 交互对话（基座模型，对比用）
.\start_manual_test.ps1 -Case 3         # 直接跑第 3 条官方验收用例
```

执行策略被拦时改用：

```powershell
powershell -ExecutionPolicy Bypass -File .\start_manual_test.ps1
```

交互命令：

| 命令 | 作用 |
| --- | --- |
| `/list` | 列出 18 条官方验收用例（编号 + test_id + 场景 + 问题） |
| `/case 3` 或 `/case PRODUCT_002` | 载入指定用例：打印业务上下文、参考回答、评测点，然后生成回答 |
| `/temp 0` | 切贪心解码（同一输入可复现，便于 base/lora 对比） |
| `/maxlen 200` | 调整最大生成 token |
| `/system <文本>` | 临时替换 system 提示词（会清空历史） |
| `/clear` / `/show` / `/quit` | 清空历史 / 查看当前状态 / 退出 |

说明：

- 必须在**真实终端**里运行（脚本需要 stdin 读取输入），首次加载约 10 秒，占用内存约 6 GB。
- CPU float32 推理：单条回复约 10~40 秒（取决于生成长度）。
- 普通输入支持多轮：回答会进入历史；行尾加 `\` 可换行续写业务上下文。
- 推理侧与训练一致：system 用训练数据的系统提示词，user = 上下文 + `用户问题：…`，`apply_chat_template(add_generation_prompt=True)`。
- 已生成的自动评测报告可作对照：`output\test_results_both_20261103_193624.md`。

> 说明：自动检查是启发式规则，用于快速定位问题；人工复核以 Markdown 中的逐条回答为准。

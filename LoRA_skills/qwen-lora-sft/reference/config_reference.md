# 配置字段参考

配置文件是普通 JSON（不能写注释），未提供的字段使用 `_common.DEFAULT_CONFIG` 的默认值。
命令行参数优先级高于配置文件。

## 路径约定（重要）

- 相对路径以**执行命令时的当前目录**为基准，不是相对配置文件所在目录。
  建议在项目根目录执行，或全部写绝对路径。
- Windows 路径在 JSON 中要转义：`"E:\\\\models\\\\Qwen2.5-1.5B-Instruct"`。
- `model.path` 两种写法都支持：本地目录（推荐，离线可用）或仓库名
  （`Qwen/Qwen2.5-1.5B-Instruct`，需要网络或本地 HF 缓存）。
- `output.dir` 不要放在数据集目录内部，避免 checkpoint 混入训练数据。

## model —— 基座模型

| 字段 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `path` | str | `Qwen/Qwen2.5-1.5B-Instruct` | 本地模型目录或 HuggingFace/ModelScope 仓库名 |
| `trust_remote_code` | bool | `true` | Qwen 等模型需要 `<modeling_*.py>` 自定义代码时开启 |
| `dtype` | str | `auto` | `auto`：CPU→float32，GPU→bfloat16（不支持则 float16）。也可写 `float32`/`float16`/`bfloat16` |
| `attn_implementation` | str/null | `null` | `sdpa`（通用加速）/ `flash_attention_2`（需 flash-attn）/ `eager` |

> `dtype` 只影响**加载精度**。训练时的 `fp16`/`bf16` 开关由设备与上述精度自动推导，
> 也可以在 `training` 里显式写死。

## data —— 数据

| 字段 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `path` | str | `./dataset/train.json` | 训练数据（json 数组 / jsonl） |
| `eval_path` | str/null | `null` | 验证数据；填了就会按 `eval_strategy` 评估 |
| `split` | str | `train` | `datasets` 的 split 名，文件只有一份时一般就是 `train` |
| `eval_split` | str/null | `null` | 验证集 split，`null` 时自动选 |
| `format` | str | `alpaca` | `alpaca` / `messages` / `text` |
| `fields` | object | 见下 | 字段名映射，用于适配别的数据集命名 |
| `system_prompt` | str | `""` | 统一注入的 system prompt |
| `max_samples` | int | `0` | 只用前 N 条（0=全量）；小样本先跑通流程 |
| `eval_max_samples` | int | `0` | 验证集同理 |
| `shuffle_seed` | int/null | `42` | 先打乱再截取；`null` 表示不打乱 |
| `num_proc` | int | `1` | `dataset.map` 进程数，Windows 建议 1 |

`fields` 默认映射：

```json
{
  "instruction": "instruction",
  "input": "input",
  "output": "output",
  "messages": "messages",
  "text": "text"
}
```

例如数据是 `{"question": ..., "answer": ...}`：

```json
"fields": { "instruction": "question", "output": "answer" }
```

## lora —— LoRA 结构

| 字段 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `r` | int | `16` | 秩。任务简单/数据少用 8；默认 16；复杂任务 32~64 |
| `alpha` | int | `32` | 缩放系数，惯例是 `2 * r` |
| `dropout` | float | `0.1` | 只作用于 LoRA 分支，数据少时可调大防过拟合 |
| `target_modules` | str/list | `"auto"` | `auto` 按 `model_type` 推断；`"all-linear"` 全线性层；也可显式给列表 |
| `bias` | str | `"none"` | 一般不训练 bias，保持 `none` |
| `task_type` | str | `"CAUSAL_LM"` | 因果语言模型固定该值 |

`auto` 的推断规则（`_common.ARCH_TARGET_MODULES`）：
Qwen2/Qwen2.5/Qwen3、Llama、Mistral、Gemma、Yi、DeepSeek、InternLM2、Baichuan →
`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
其余结构若在表内按表处理，不认识则退回 `all-linear`。

## training —— 训练超参

| 字段 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `per_device_train_batch_size` | int | `2` | 单卡 batch。显存不够先降它 |
| `gradient_accumulation_steps` | int | `4` | 梯度累积。等效 batch = 两者相乘 |
| `num_train_epochs` | float | `3` | 轮数。小数据集 3~5，大数据集 1~2 |
| `max_steps` | int | `-1` | >0 时覆盖 epochs，按步数训练 |
| `learning_rate` | float | `1e-4` | LoRA 常用 1e-4 ~ 2e-4；全参数微调应降到 1e-5 级别 |
| `lr_scheduler_type` | str | `cosine` | 配合 warmup 平稳启动、后期精细收敛 |
| `warmup_steps` | int | `5` | 总步数很少时保持个位数 |
| `max_grad_norm` | float | `0.3` | 梯度裁剪，防梯度爆炸 |
| `weight_decay` | float | `0.0` | LoRA 一般不需要 |
| `logging_steps` | int | `10` | 日志间隔；小样本调试可设 1 |
| `save_strategy` | str | `epoch` | `epoch` / `steps` / `no` |
| `save_total_limit` | int | `3` | 只保留最近 N 个 checkpoint |
| `max_length` | int | `512` | 序列截断长度（trl 1.x 名；旧版自动转 `max_seq_length`） |
| `dataset_text_field` | str | `text` | 与格式化函数输出的列名一致 |
| `gradient_checkpointing` | bool | `false` | 省显存、略降速；大模型建议开 |
| `packing` | bool | `false` | 把多条短样本拼进一条序列，提升吞吐；数据短时可开 |
| `dataloader_num_workers` | int | `0` | Windows 必须 0，Linux 可 2~4 |
| `dataloader_pin_memory` | bool/null | `null` | `null` = GPU 开、CPU 关 |
| `bf16` / `fp16` | bool/null | `null` | `null` = 自动推导；一般只保留一个为 true |
| `seed` | int | `42` | 随机种子 |
| `report_to` | str | `none` | 不用 wandb/tensorboard 就保持 `none` |

未知字段会被过滤并打印警告（用来兼容不同 trl 版本）。

## output —— 产物

| 字段 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `dir` | str | `./output/lora_adapter` | adapter 与日志目录 |
| `save_final` | bool | `true` | 训练结束额外保存一份最终 adapter |
| `summary_file` | str | `training_summary.json` | 训练摘要文件名 |
| `run_config_file` | str | `run_config.json` | 配置快照文件名 |

## 命令行覆盖

```bash
python scripts/finetune.py --config my.json \
  --model-path D:/models/Qwen2.5-1.5B-Instruct \
  --data-path ./dataset/train.json \
  --output-dir ./output/lora_v2 \
  --max-samples 200 --epochs 1 --learning-rate 2e-4 \
  --lora-r 32 --max-length 768 --seed 1234
```

注意：`--lora-r` 会把 `alpha` 自动调整为 `2 * r`；`--max-steps` 优先于 `--epochs`。

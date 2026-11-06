# 超参调节与显存估算

## 先定「等效 batch」

```
等效 batch = per_device_train_batch_size × gradient_accumulation_steps × GPU 数
```

SFT 一般让等效 batch 落在 **8 ~ 32**。
显存不足时**不要降等效 batch**，而是把 `per_device_train_batch_size` 降下来、
把 `gradient_accumulation_steps` 加回去（数值不变，只是更慢）。

## LoRA 参数

| 参数 | 建议 | 什么时候改 |
| --- | --- | --- |
| `r` | 16 | 数据 < 1k 或只学风格：8；任务复杂（多任务、代码、长推理）：32~64 |
| `alpha` | 2 × r | 与 r 同步放大，保证缩放比例稳定 |
| `dropout` | 0.1 | 数据少/明显过拟合：0.1~0.15；数据多：0.05 |
| `target_modules` | 全部线性层 | 只学风格可只留 `q_proj,v_proj`；追求效果就全量 + MLP |
| `bias` | none | 不改 |

> r 变大会线性增加可训练参数与显存，但 1.5B 模型上 r=32 的增量仍很小（几十 MB 级），
> 真正吃显存的是**激活值**（`max_length` × batch）。

## 学习率与调度

- 学习率：`1e-4`（稳妥）→ `2e-4`（数据多、收敛慢）→ `5e-5`（loss 抖动/发散）
- 调度：`cosine` + `warmup_steps`（小数据 2~5，大数据按总步数 3%~5%）
- 组合效果：预热阶段平稳启动，后期余弦退火在平坦区精细收敛
- 判断标准：loss 前 20% 步数应明显下降；长期不降 → 学习率太小或数据映射错了
  （比如 instruction/output 反了，模型在学「问题倒推」）

## 轮数与数据量

| 数据量 | 建议 epochs | 说明 |
| --- | --- | --- |
| < 200 条 | 3~5 | 容易过拟合，配合 dropout 0.1、每轮评估 |
| 1k ~ 10k | 2~3 | 最常用的区间 |
| > 50k | 1~2 | 用 `max_steps` 控制更省时 |

判断过拟合：训练 loss 持续下降但验证/实际问答质量变差、回答开始复读训练集原句。

## 序列长度 max_length

`max_length` 直接决定显存与速度（注意力是平方级）。

| 场景 | 建议 |
| --- | --- |
| CPU 冒烟测试 | 256 |
| 中文问答（一句话问题 + 短回答） | 384 ~ 512 |
| 长文档摘要/多轮对话 | 768 ~ 1024 |
| 超长文本 | 先开 `packing` 与 `gradient_checkpointing`，否则考虑 QLoRA |

先用 `prepare_dataset.py` 的报告看回答长度分布，再定 `max_length`：
取「平均长度 × 2」通常够用，超出部分会被截断。

## 精度选择

| 设备 | 加载 dtype | 训练开关 |
| --- | --- | --- |
| CPU（无独显） | float32 | `use_cpu=true, fp16=false, bf16=false` |
| Ampere 及以上 GPU | bfloat16 | `bf16=true` |
| 较老 GPU（无 bf16） | float16 | `fp16=true`（必要时加 loss scale） |

脚本的 `resolve_precision()` 会自动处理，正常情况不用手写。

## 显存估算（LoRA，bf16，无优化器状态）

| 模型 | 权重 | 训练 512 长度、batch=1 | 建议显卡 |
| --- | --- | --- | --- |
| 0.5B | ~1 GB | ~4 GB | 6 GB |
| 1.5B | ~3 GB | ~8 GB | 12 GB |
| 7B | ~14 GB | ~20 GB | 24 GB |
| 7B + QLoRA(4bit) | ~4 GB | ~10 GB | 12 GB |

显存不够的优先级：
1. 降 `max_length`（256）
2. 降 `per_device_train_batch_size` 到 1，提高 `gradient_accumulation_steps`
3. 开 `gradient_checkpointing`
4. 换更小模型（0.5B/1.5B）
5. QLoRA（4bit 量化加载，需要 bitsandbytes，且仅 CUDA 可用）

## CPU 训练现实预期

以本项目 1.5B + 50 条数据 + batch=1 + accum=8 + `max_length=256` 为例：
约 35 步，CPU 上需要**数十分钟到数小时**。因此：

- CPU 只用 50 条以内数据验证「流程能跑通」
- 真正看效果必须上 GPU（或云 GPU）
- CPU 冒烟配置见 `assets/finetune_config.cpu.json`

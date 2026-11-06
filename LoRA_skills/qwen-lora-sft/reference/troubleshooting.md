# 报错排查表

按「报错关键字」查原因与处理。脚本已内置部分兼容逻辑，标注「已自动处理」的可直接忽略。

## 环境 / 版本

| 报错 / 现象 | 原因 | 处理 |
| --- | --- | --- |
| `unexpected keyword argument 'max_seq_length'` | trl ≥ 1.x 把序列长度改名为 `max_length` | 已自动处理（按 `SFTConfig` 签名切换） |
| `unexpected keyword argument 'tokenizer'`（SFTTrainer） | trl 1.x 用 `processing_class` | 已自动处理 |
| `'torch_dtype' is deprecated` / v5 下精度不生效 | transformers v5 改用 `dtype` | 已自动处理（按主版本号选择参数名） |
| `ImportError: cannot import name 'SFTConfig'` | trl 太旧 | `pip install -U trl`（≥ 0.16 才有 SFTConfig） |
| `No module named 'accelerate'` | Trainer 依赖 | `pip install accelerate` |
| 版本互相踩踏（transformers/trl 不匹配） | 升级后接口变化 | 固定版本组合：`transformers==5.18.* trl==1.17.* peft==0.21.*` |

## 模型加载

| 报错 / 现象 | 原因 | 处理 |
| --- | --- | --- |
| `AssertionError: Torch not compiled with CUDA enabled` | CPU 版 torch 却调用了 `.to("cuda")` | 用 `get_device()` 自动选设备；或装 CUDA 版 torch |
| `Could not locate the model file` / 路径不存在 | Windows 反斜杠被转义（`\t` `\n`） | 用 `r"E:\path"` 或正斜杠 `E:/path` |
| `trust_remote_code` 相关报错 | Qwen 等需要自定义建模代码 | 配置 `model.trust_remote_code: true` |
| 加载后输出乱码 | tokenizer 与模型不是同一目录 | 两者都用同一个 `model.path` |
| 首次加载卡住 | 联网下载模型 | 先用 modelscope/hf 下载到本地，再填本地目录 |

## 训练 / 显存

| 报错 / 现象 | 原因 | 处理 |
| --- | --- | --- |
| `CUDA out of memory` | 激活值太大 | 降 `max_length` → 降 batch 并加 `gradient_accumulation_steps` → 开 `gradient_checkpointing` → QLoRA |
| CPU 上 `RuntimeError: "LayerNormKernelImpl" not implemented for 'Half'` | CPU 不支持 fp16 算子 | 用 float32（脚本已强制） |
| loss 变成 `nan` / 一直 0 | 学习率过高、精度溢出、数据里有空样本 | 学习率降到 5e-5；GPU 用 bf16；用 `prepare_dataset.py` 过滤空样本 |
| loss 完全不下降 | 学习率太小、字段映射反了、`max_length` 截断了答案 | `--logging-steps 1` 观察；打印 dry-run 预览确认 user/assistant 顺序 |
| 训练极慢 | `max_length` 太大、数据太长、CPU、`num_workers=0` | 先小样本试；GPU 上可开 `packing`、`sdpa` |
| Windows 上 DataLoader 卡死/递归报错 | 多进程 fork 问题 | `dataloader_num_workers: 0` |
| `ValueError: num_samples should be a positive integer` | 数据为空或过滤后为空 | 检查 `data.format`/`fields` 是否匹配 |

## 数据 / 格式化

| 报错 / 现象 | 原因 | 处理 |
| --- | --- | --- |
| `KeyError: 'instruction'` 或 `alpaca 格式缺少字段` | 数据字段名不一致 | 改 `data.fields` 映射 |
| `KeyError: 'text'` | `dataset_text_field` 与格式化输出列名不一致 | 保持 `text` 或同步修改 |
| `tokenizer.chat_template is None` 警告 | 基座模型没有对话模板 | 换 Instruct 版模型；或用 `format: "text"` 自己拼模板 |
| 预览文本里没有对话标记 | 同上，或 `apply_chat_template` 未生效 | 检查 dry-run 输出 |
| 模型回答复读问题 | 训练目标里混入了提问内容 | 确认 assistant 段只放答案 |
| 训练集只有几十条却学得「很好」 | 数据重复 | `prepare_dataset.py` 默认去重，查看报告 `drop_reasons` |
| 中文乱码 | 文件不是 UTF-8 | 统一转 UTF-8；`setup_utf8()` 已强制读取/输出编码 |

## 推理 / 合并

| 报错 / 现象 | 原因 | 处理 |
| --- | --- | --- |
| `PeftModel.from_pretrained` 报 adapter 配置不匹配 | adapter 与基座模型不是同一个 | adapter 必须配训练时的 `model.path` |
| 推理结果和基座一样 | adapter 没加载成功 / 用了错误的 checkpoint 目录 | 打印 `adapter_config.json` 里的 `base_model_name_or_path`；确认目录里有 `adapter_model.safetensors` |
| 回答被截断 | `max_new_tokens` 太小 | 调大；或在输出后检查是否以句号结束（`infer_lora.py` 会补句号） |
| 合并时报 OOM（CPU 内存） | float32 合并 7B 以上模型 | 用 `--dtype bfloat16`（若设备支持）或换更大内存的机器 |
| 合并后模型比 adapter 大很多 | 正常：合并产物是完整权重 | 部署用合并版，继续训练用 adapter |

## 快速定位顺序

1. `python scripts/env_report.py --config your_config.json`
2. `python scripts/finetune.py --config your_config.json --dry-run`
3. 把 `--max-samples` 调到 8、`--max-length 128`、`--epochs 1` 跑一次最小闭环
4. 最小闭环通过后再逐步放大到真实配置

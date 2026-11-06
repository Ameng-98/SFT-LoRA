# -*- coding: utf-8 -*-
"""
qwen-lora-sft 公共工具集（finetune.py / prepare_dataset.py / infer_lora.py 复用）。

设计原则：
- 只依赖项目里已经在用的库：torch / transformers / datasets / peft / trl
- 同时兼容 transformers v4（torch_dtype）与 v5（dtype）、trl 新旧参数名
- 路径与超参全部来自 JSON 配置，代码里不写死任何本机绝对路径
"""

from __future__ import annotations

import copy
import json
import os
import pathlib
import sys
from typing import Any, Callable

import torch

# ============================================================
# 结构 → LoRA 目标层（decoder-only 因果语言模型）
# ============================================================
QWEN_STYLE = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

ARCH_TARGET_MODULES = {
    "qwen2": QWEN_STYLE,
    "qwen2_moe": QWEN_STYLE,
    "qwen3": QWEN_STYLE,
    "qwen3_moe": QWEN_STYLE,
    "llama": QWEN_STYLE,
    "mistral": QWEN_STYLE,
    "mixtral": QWEN_STYLE,
    "gemma": QWEN_STYLE,
    "gemma2": QWEN_STYLE,
    "gemma3": QWEN_STYLE,
    "yi": QWEN_STYLE,
    "internlm2": QWEN_STYLE,
    "baichuan": QWEN_STYLE,
    "deepseek_v2": QWEN_STYLE,
    "deepseek_v3": QWEN_STYLE,
    "phi3": ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
    "phi": ["Wqkv", "out_proj", "fc1", "fc2"],
    "gpt2": ["c_attn", "c_proj"],
    "gpt_neox": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    "falcon": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    "bloom": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    "chatglm": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
}


def setup_utf8() -> None:
    """Windows 下强制 UTF-8，避免中文日志/数据集乱码（复刻 train_qwen.py 的做法）。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:  # 终端不支持 reconfigure 时忽略
            pass
    os.environ.setdefault("PYTHONUTF8", "1")

    if getattr(pathlib.Path.read_text, "_utf8_patched", False):
        return

    original = pathlib.Path.read_text

    def _patched(self, encoding=None, errors=None):
        return original(self, encoding="utf-8", errors=errors)

    _patched._utf8_patched = True
    pathlib.Path.read_text = _patched


def get_device(verbose: bool = True) -> str:
    """有 GPU 用 GPU，没有则退回 CPU。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose:
        if device == "cuda":
            name = torch.cuda.get_device_name(0)
            bf16 = "支持" if torch.cuda.is_bf16_supported() else "不支持"
            print(f"✅ 使用 GPU: {name}（bfloat16 {bf16}）")
        else:
            print("⚠ 未检测到 GPU，使用 CPU 训练（速度很慢，建议先小样本跑通流程）")
    return device


_DTYPE_ALIASES = {
    "auto": None,
    "fp32": "float32",
    "float32": "float32",
    "fp16": "float16",
    "float16": "float16",
    "half": "float16",
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
}


def pick_torch_dtype(device: str, requested: str = "auto") -> torch.dtype:
    """按设备与配置选择加载精度：CPU 用 fp32，GPU 优先 bf16。"""
    key = str(requested or "auto").lower()
    name = _DTYPE_ALIASES.get(key, key)
    if name is None:
        if device == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32
    if name == "bfloat16" and device == "cuda" and not torch.cuda.is_bf16_supported():
        print("⚠ 当前 GPU 不支持 bfloat16，自动回退 float16")
        return torch.float16
    return getattr(torch, name)


def dtype_to_string(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def dtype_kwarg_name() -> str:
    """transformers v5 用 dtype，v4 用 torch_dtype。"""
    import transformers

    try:
        major = int(transformers.__version__.split(".")[0])
    except Exception:
        major = 4
    return "dtype" if major >= 5 else "torch_dtype"


def model_load_kwargs(
    dtype: torch.dtype | None = None,
    trust_remote_code: bool = True,
    attn_implementation: str | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"trust_remote_code": bool(trust_remote_code)}
    if dtype is not None:
        kwargs[dtype_kwarg_name()] = dtype
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    return kwargs


def resolve_precision(device: str, training_cfg: dict[str, Any]) -> tuple[torch.dtype, dict[str, bool]]:
    """推导「模型加载精度」与「训练器 fp16/bf16/use_cpu 开关」。"""
    dtype = pick_torch_dtype(device, training_cfg.get("dtype", "auto"))
    bf16 = training_cfg.get("bf16")
    fp16 = training_cfg.get("fp16")

    if device == "cpu":
        if bf16 or fp16:
            print("⚠ CPU 训练不支持 fp16/bf16，已强制使用 float32")
        return torch.float32, {"bf16": False, "fp16": False, "use_cpu": True}

    if bf16 is None and fp16 is None:
        bf16 = dtype == torch.bfloat16
        fp16 = dtype == torch.float16
    flags = {"bf16": bool(bf16), "fp16": bool(fp16), "use_cpu": False}
    if flags["bf16"] and not torch.cuda.is_bf16_supported():
        print("⚠ 当前 GPU 不支持 bfloat16，训练改用 fp16")
        flags["bf16"], flags["fp16"] = False, True
    return dtype, flags


# ============================================================
# 默认配置（可被 JSON 配置文件与命令行覆盖）
# ============================================================
DEFAULT_CONFIG: dict[str, Any] = {
    "model": {
        "path": "Qwen/Qwen2.5-1.5B-Instruct",
        "trust_remote_code": True,
        "dtype": "auto",              # auto / float32 / float16 / bfloat16
        "attn_implementation": None,  # sdpa / flash_attention_2 / eager / None
    },
    "data": {
        "path": "./dataset/train.json",
        "eval_path": None,
        "split": "train",
        "eval_split": None,
        "format": "alpaca",           # alpaca / messages / text
        "fields": {
            "instruction": "instruction",
            "input": "input",
            "output": "output",
            "messages": "messages",
            "text": "text",
        },
        "system_prompt": "",
        "max_samples": 0,             # 0 = 全量；先用小样本跑通流程
        "eval_max_samples": 0,
        "shuffle_seed": 42,
        "num_proc": 1,                # Windows 下建议 1
    },
    "lora": {
        "r": 16,                      # 秩；复杂任务可调到 32
        "alpha": 32,                  # 一般 = 2 * r
        "dropout": 0.1,
        "target_modules": "auto",     # auto / all-linear / ["q_proj", ...]
        "bias": "none",
        "task_type": "CAUSAL_LM",
    },
    "training": {
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "num_train_epochs": 3,
        "max_steps": -1,
        "learning_rate": 1e-4,
        "lr_scheduler_type": "cosine",
        "warmup_steps": 5,
        "max_grad_norm": 0.3,
        "weight_decay": 0.0,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "save_total_limit": 3,
        "max_length": 512,            # trl 1.x 用 max_length；旧版自动转 max_seq_length
        "dataset_text_field": "text",
        "gradient_checkpointing": False,
        "packing": False,
        "dataloader_num_workers": 0,  # Windows 建议 0
        "dataloader_pin_memory": None,  # None = 自动（GPU 开、CPU 关）
        "fp16": None,                 # None = 按模型精度自动推导
        "bf16": None,
        "seed": 42,
        "report_to": "none",
    },
    "output": {
        "dir": "./output/lora_adapter",
        "save_final": True,
        "summary_file": "training_summary.json",
        "run_config_file": "run_config.json",
    },
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """递归合并配置：override 覆盖 base，返回新字典（不修改入参）。"""
    result = {}
    for key, value in base.items():
        result[key] = copy.deepcopy(value) if isinstance(value, (dict, list)) else value
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_json(path: str | pathlib.Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | pathlib.Path, data: Any) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_config(path: str | pathlib.Path | None = None) -> dict[str, Any]:
    """读取 JSON 配置并与默认值合并；未提供的字段使用默认值。"""
    config = deep_merge(DEFAULT_CONFIG, {})
    if not path:
        return config
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    user_config = load_json(path)
    if not isinstance(user_config, dict):
        raise ValueError(f"配置文件必须是 JSON 对象: {path}")
    unknown = [k for k in user_config if k not in DEFAULT_CONFIG]
    if unknown:
        print(f"⚠ 配置中存在未知字段（已忽略）: {', '.join(unknown)}")
    return deep_merge(config, user_config)


def load_records(path: str | pathlib.Path) -> list[dict[str, Any]]:
    """读取数据集：支持 JSON 数组 / JSON Lines / 单个 JSON 对象。"""
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"数据文件不存在: {path}")

    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return []

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # 退化为 JSON Lines 解析
        return [json.loads(line) for line in content.splitlines() if line.strip()]

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "train", "records", "examples"):
            if isinstance(data.get(key), list):
                return data[key]
        return [data]
    raise ValueError(f"无法识别的数据格式: {path}")


def select_subset(dataset, max_samples: int, seed: int | None, desc: str = ""):
    """先打乱再截取前 N 条；max_samples<=0 表示全量。"""
    total = len(dataset)
    if seed is not None:
        dataset = dataset.shuffle(seed=seed)
    if max_samples and max_samples > 0 and max_samples < total:
        dataset = dataset.select(range(max_samples))
    if desc:
        print(f"📊 {desc}: 原始 {total} 条 → 使用 {len(dataset)} 条")
    return dataset


def _read_local_model_config(model_path: str) -> dict[str, Any]:
    path = pathlib.Path(model_path)
    config_file = path / "config.json" if path.is_dir() else path
    if config_file.is_file():
        try:
            return load_json(config_file)
        except Exception:
            return {}
    try:
        from transformers import AutoConfig

        return AutoConfig.from_pretrained(str(model_path), trust_remote_code=True).to_dict()
    except Exception:
        return {}


def guess_target_modules(model_path: str) -> list[str] | str:
    """按模型 config.json 的 model_type 推断 LoRA 目标层；不认识就退回 all-linear。"""
    config = _read_local_model_config(str(model_path))
    model_type = str(config.get("model_type") or "").lower()
    if model_type in ARCH_TARGET_MODULES:
        return list(ARCH_TARGET_MODULES[model_type])
    print(f"⚠ 未识别的模型结构 model_type='{model_type}'，LoRA 目标层退回 'all-linear'")
    return "all-linear"


def resolve_target_modules(spec: Any, model_path: str) -> list[str] | str:
    """归一化 target_modules：auto → 推断；字符串/列表 → 原样返回。"""
    if isinstance(spec, list) and spec:
        return list(spec)
    if isinstance(spec, str):
        lowered = spec.strip().lower()
        if lowered in {"", "auto"}:
            return guess_target_modules(model_path)
        return spec.strip()
    return guess_target_modules(model_path)


def _normalize_messages(messages: Any) -> list[dict[str, str]]:
    """把 messages 统一成 [{"role": ..., "content": ...}]，兼容 from/value 别名。"""
    if isinstance(messages, dict):
        keys = list(messages.keys())
        length = len(messages[keys[0]]) if keys else 0
        messages = [{k: messages[k][i] for k in keys} for i in range(length)]
    normalized: list[dict[str, str]] = []
    for item in messages or []:
        if not isinstance(item, dict):
            continue
        role = item.get("role") or item.get("from") or "user"
        content = item.get("content") if item.get("content") is not None else item.get("value", "")
        normalized.append({"role": str(role), "content": str(content or "").strip()})
    return normalized


def build_text_formatter(tokenizer, data_cfg: dict[str, Any]) -> Callable[[dict[str, Any]], dict[str, list[str]]]:
    """
    返回 dataset.map(..., batched=True) 使用的格式化函数。

    输出统一为 {"text": [...]}，与 SFTConfig.dataset_text_field="text" 对应。
    - alpaca   : instruction + input → user，output → assistant
    - messages : 直接使用数据里已有的对话列表
    - text     : 已经是拼好的纯文本，只做 strip
    """
    fmt = str(data_cfg.get("format") or "alpaca").lower()
    fields = deep_merge(DEFAULT_CONFIG["data"]["fields"], data_cfg.get("fields") or {})
    system_prompt = str(data_cfg.get("system_prompt") or "").strip()

    def render(messages: list[dict[str, str]]) -> str:
        if tokenizer is not None and getattr(tokenizer, "chat_template", None):
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        return "\n".join(f"{m['role']}：{m['content']}" for m in messages)

    def alpaca_to_messages(example: dict[str, Any]) -> list[dict[str, str]]:
        instruction = str(example.get(fields["instruction"]) or "").strip()
        extra_input = str(example.get(fields["input"]) or "").strip()
        output = str(example.get(fields["output"]) or "").strip()
        user_content = instruction if not extra_input else f"{instruction}\n{extra_input}"
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": output})
        return messages

    def formatting_func(examples: dict[str, Any]) -> dict[str, list[str]]:
        if fmt == "text":
            column = fields["text"]
            if column not in examples:
                raise KeyError(f"text 格式缺少字段 '{column}'，请检查 data.fields.text")
            return {"text": [str(t or "").strip() for t in examples[column]]}

        texts: list[str] = []
        if fmt == "messages":
            column = fields["messages"]
            if column not in examples:
                raise KeyError(f"messages 格式缺少字段 '{column}'，请检查 data.fields.messages")
            for raw in examples[column]:
                messages = _normalize_messages(raw)
                if system_prompt:
                    messages = [{"role": "system", "content": system_prompt}] + messages
                texts.append(render(messages))
            return {"text": texts}

        if fmt == "alpaca":
            column = fields["instruction"]
            if column not in examples:
                raise KeyError(f"alpaca 格式缺少字段 '{column}'，请检查 data.fields.instruction")
            total = len(examples[column])
            input_col = examples.get(fields["input"]) or [""] * total
            output_col = examples.get(fields["output"]) or [""] * total
            total = min(total, len(input_col), len(output_col))
            for i in range(total):
                row = {
                    fields["instruction"]: examples[column][i],
                    fields["input"]: input_col[i],
                    fields["output"]: output_col[i],
                }
                texts.append(render(alpaca_to_messages(row)))
            return {"text": texts}

        raise ValueError(f"不支持的数据格式: {fmt}（可选 alpaca / messages / text）")

    return formatting_func


def format_preview(tokenizer, data_cfg: dict[str, Any], examples: list[dict[str, Any]]) -> list[str]:
    """离线预览（dry-run 用）：把若干条原始样本渲染成训练文本。"""
    formatter = build_text_formatter(tokenizer, data_cfg)
    fmt = str(data_cfg.get("format") or "alpaca").lower()
    fields = deep_merge(DEFAULT_CONFIG["data"]["fields"], data_cfg.get("fields") or {})

    if fmt == "text":
        batch = {fields["text"]: [ex.get(fields["text"]) for ex in examples]}
    elif fmt == "messages":
        batch = {fields["messages"]: [ex.get(fields["messages"]) for ex in examples]}
    else:
        batch = {
            fields["instruction"]: [ex.get(fields["instruction"]) for ex in examples],
            fields["input"]: [ex.get(fields["input"]) for ex in examples],
            fields["output"]: [ex.get(fields["output"]) for ex in examples],
        }
    return formatter(batch)["text"]


def print_header(title: str) -> None:
    print("=" * 60)
    print(title)
    print("=" * 60)

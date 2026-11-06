# -*- coding: utf-8 -*-
"""
通用 LoRA 监督微调（SFT）脚本 —— 由本项目 train_qwen.py 抽象而来。

用法示例（在技能目录下执行，或使用绝对路径）：
    python scripts/finetune.py --config assets/finetune_config.gpu.json
    python scripts/finetune.py --config assets/finetune_config.cpu.json --dry-run
    python scripts/finetune.py --model-path /path/Qwen2.5-1.5B-Instruct \
        --data-path ./dataset/train.json --output-dir ./output/lora_adapter

与 train_qwen.py 一致的关键点：
- peft.LoraConfig + get_peft_model 插入 LoRA 层并冻结原始参数
- trl.SFTConfig / SFTTrainer（trl 1.x：max_length + processing_class）
- tokenizer.pad_token = eos_token，避免 padding 报错
- CPU 自动 fp32 + use_cpu=True；GPU 自动 bf16（不支持则回退 fp16）

相对原脚本的增强（便于复用）：
- 所有路径/超参来自 JSON 配置 + 命令行覆盖，不写死本机路径
- LoRA target_modules 按模型结构自动推断
- --dry-run 只校验配置/数据/格式化结果，不加载模型
- 训练结束保存最终 adapter、run_config.json、training_summary.json
"""

from __future__ import annotations

import argparse
import inspect
import math
import os
import time
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

from _common import (
    build_text_formatter,
    dtype_to_string,
    format_preview,
    get_device,
    load_config,
    model_load_kwargs,
    print_header,
    resolve_precision,
    resolve_target_modules,
    save_json,
    select_subset,
    setup_utf8,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="通用 LoRA SFT 微调脚本")
    parser.add_argument("--config", default=None, help="JSON 配置文件路径（可省略，使用默认值）")
    parser.add_argument("--model-path", default=None, help="基座模型路径或 HuggingFace 仓库名")
    parser.add_argument("--data-path", default=None, help="训练数据（json / jsonl）")
    parser.add_argument("--eval-data-path", default=None, help="验证数据（可选）")
    parser.add_argument("--output-dir", default=None, help="adapter 与日志输出目录")
    parser.add_argument("--max-samples", type=int, default=None, help="只用前 N 条训练数据（0=全量）")
    parser.add_argument("--epochs", type=float, default=None, help="训练轮数")
    parser.add_argument("--max-steps", type=int, default=None, help="最大训练步数（>0 时覆盖 epochs）")
    parser.add_argument("--learning-rate", type=float, default=None, help="学习率")
    parser.add_argument("--lora-r", type=int, default=None, help="LoRA 秩 r")
    parser.add_argument("--max-length", type=int, default=None, help="最大序列长度")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--dry-run", action="store_true", help="只校验配置与数据，不加载模型、不训练")
    parser.add_argument(
        "--resume-from-checkpoint",
        nargs="?",
        const=True,
        default=None,
        help="断点续训：不带值=自动找最新 checkpoint，也可指定路径",
    )
    return parser.parse_args(argv)


def apply_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    """命令行参数覆盖配置（命令行优先级最高）。"""
    if args.model_path:
        cfg["model"]["path"] = args.model_path
    if args.data_path:
        cfg["data"]["path"] = args.data_path
    if args.eval_data_path:
        cfg["data"]["eval_path"] = args.eval_data_path
    if args.output_dir:
        cfg["output"]["dir"] = args.output_dir
    if args.max_samples is not None:
        cfg["data"]["max_samples"] = args.max_samples
    if args.epochs is not None:
        cfg["training"]["num_train_epochs"] = args.epochs
    if args.max_steps is not None:
        cfg["training"]["max_steps"] = args.max_steps
    if args.learning_rate is not None:
        cfg["training"]["learning_rate"] = args.learning_rate
    if args.lora_r is not None:
        cfg["lora"]["r"] = args.lora_r
        cfg["lora"]["alpha"] = max(cfg["lora"].get("alpha") or 0, args.lora_r * 2)
    if args.max_length is not None:
        cfg["training"]["max_length"] = args.max_length
    if args.seed is not None:
        cfg["training"]["seed"] = args.seed
        cfg["data"]["shuffle_seed"] = args.seed


def _pick_split(dataset_dict, preferred: str | None) -> str:
    if preferred and preferred in dataset_dict:
        return preferred
    if "train" in dataset_dict:
        return "train"
    return list(dataset_dict.keys())[0]


def load_split(
    data_path: str,
    preferred_split: str | None,
    max_samples: int,
    seed: int | None,
    desc: str,
):
    """按 split 加载数据，并按 max_samples 截取子集。"""
    dataset_dict = load_dataset("json", data_files=str(data_path))
    split = _pick_split(dataset_dict, preferred_split)
    if preferred_split and preferred_split != split:
        print(f"⚠ 未找到 split '{preferred_split}'，改用 '{split}'")
    return select_subset(dataset_dict[split], max_samples, seed, desc)


def format_dataset(dataset, tokenizer, data_cfg: dict[str, Any], num_proc: int = 1):
    """把原始样本转换为 chat template 文本列（text）。"""
    formatter = build_text_formatter(tokenizer, data_cfg)
    before = len(dataset)
    formatted = dataset.map(
        formatter,
        batched=True,
        num_proc=max(1, int(num_proc)),
        remove_columns=dataset.column_names,
        desc="格式化训练文本",
    )
    formatted = formatted.filter(
        lambda ex: isinstance(ex.get("text"), str) and len(ex["text"].strip()) > 0,
        desc="过滤空文本",
    )
    if len(formatted) == 0:
        raise ValueError("格式化后没有有效样本，请检查 data.format / data.fields 配置")
    print(f"✅ 文本格式化完成: {before} 条 → {len(formatted)} 条")
    return formatted


def build_training_args(cfg: dict[str, Any], flags: dict[str, bool], output_dir: str):
    """按当前 trl 版本组装 SFTConfig，自动兼容 max_length / max_seq_length 等参数名。"""
    from trl import SFTConfig

    fields = set(inspect.signature(SFTConfig.__init__).parameters)

    raw = dict(cfg["training"])
    raw["output_dir"] = output_dir
    raw.update(flags)  # bf16 / fp16 / use_cpu 以设备推导结果为准
    raw.pop("dtype", None)

    if raw.get("dataloader_pin_memory") is None:
        raw["dataloader_pin_memory"] = not flags["use_cpu"]
    if raw.get("max_steps") in (-1, None, 0):
        raw.pop("max_steps", None)

    if "max_length" not in fields and "max_seq_length" in fields:
        raw["max_seq_length"] = raw.pop("max_length", None)

    accepted, dropped = {}, []
    for key, value in raw.items():
        if key in fields:
            accepted[key] = value
        else:
            dropped.append(key)
    if dropped:
        print(f"⚠ 当前 trl 版本不支持这些参数，已忽略: {', '.join(sorted(dropped))}")

    # None 交给 SFTConfig 默认值处理
    return SFTConfig(**{k: v for k, v in accepted.items() if v is not None})


def estimate_steps(n_samples: int, training_cfg: dict[str, Any]) -> int:
    max_steps = int(training_cfg.get("max_steps") or -1)
    if max_steps > 0:
        return max_steps
    batch = max(1, int(training_cfg.get("per_device_train_batch_size", 1)))
    accum = max(1, int(training_cfg.get("gradient_accumulation_steps", 1)))
    epochs = float(training_cfg.get("num_train_epochs", 1))
    return max(1, math.ceil(n_samples / (batch * accum) * epochs))


def print_plan(
    cfg: dict[str, Any],
    device: str,
    dtype: torch.dtype,
    flags: dict[str, bool],
    target_modules: Any,
    n_train: int,
    n_eval: int | None,
) -> None:
    training = cfg["training"]
    batch = int(training.get("per_device_train_batch_size", 1))
    accum = int(training.get("gradient_accumulation_steps", 1))
    print_header("训练配置")
    print(f"模型路径      : {cfg['model']['path']}")
    print(f"数据路径      : {cfg['data']['path']}（格式={cfg['data']['format']}）")
    print(f"输出目录      : {cfg['output']['dir']}")
    print(f"设备/精度     : {device} / {dtype_to_string(dtype)}（fp16={flags['fp16']}, bf16={flags['bf16']}）")
    print(f"LoRA          : r={cfg['lora']['r']}, alpha={cfg['lora']['alpha']}, "
          f"dropout={cfg['lora']['dropout']}")
    print(f"LoRA 目标层   : {target_modules}")
    print(f"batch 设置    : per_device={batch}, grad_accum={accum}, 等效 batch={batch * accum}")
    print(f"轮数/步数     : epochs={training.get('num_train_epochs')}, "
          f"max_steps={training.get('max_steps')}, 预计约 {estimate_steps(n_train, training)} 步")
    print(f"最大序列长度  : {training.get('max_length')}")
    print(f"样本数        : train={n_train}" + (f", eval={n_eval}" if n_eval else ""))


def dry_run(cfg: dict[str, Any], device: str, dtype: torch.dtype, flags: dict[str, bool]) -> None:
    """只校验：模型路径、数据路径、字段映射、chat template 渲染结果。"""
    print_header("DRY RUN：校验配置与数据（不加载模型、不训练）")

    model_path = str(cfg["model"]["path"])
    data_path = str(cfg["data"]["path"])

    if os.path.isdir(model_path):
        print(f"✅ 本地模型目录存在: {model_path}")
        config_file = os.path.join(model_path, "config.json")
        if os.path.exists(config_file):
            try:
                import json

                with open(config_file, "r", encoding="utf-8") as f:
                    model_config = json.load(f)
                print(f"   model_type={model_config.get('model_type')} "
                      f"architectures={model_config.get('architectures')}")
            except Exception:
                pass
    else:
        print(f"ℹ 模型不是本地目录（{model_path}），实际训练时将由 transformers 联网/缓存加载")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"训练数据不存在: {data_path}")
    size_mb = os.path.getsize(data_path) / 1024 / 1024
    print(f"✅ 训练数据存在: {data_path}（{size_mb:.1f} MB）")
    if size_mb > 200:
        print("⚠ 数据文件较大，dry-run 需要完整读取一次，可能需要几十秒")

    tokenizer = None
    if os.path.isdir(model_path):
        from transformers import AutoTokenizer

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=cfg["model"].get("trust_remote_code", True)
            )
            has_template = bool(getattr(tokenizer, "chat_template", None))
            print(f"✅ tokenizer 加载成功（chat_template: {'有' if has_template else '无'}）")
        except Exception as exc:
            print(f"⚠ tokenizer 加载失败（{exc}），预览将退化为纯文本拼接")

    max_samples = int(cfg["data"].get("max_samples") or 0)
    train_ds = load_split(
        data_path,
        cfg["data"].get("split"),
        max_samples,
        cfg["data"].get("shuffle_seed"),
        "训练集",
    )

    preview = format_preview(
        tokenizer, cfg["data"], [train_ds[i] for i in range(min(2, len(train_ds)))]
    )
    print("\n--- 渲染后的训练文本（前 2 条，截断展示）---")
    for idx, text in enumerate(preview, 1):
        snippet = text if len(text) <= 600 else text[:600] + " ..."
        print(f"[样本 {idx}]\n{snippet}\n")

    target_modules = resolve_target_modules(cfg["lora"]["target_modules"], model_path)
    print_plan(cfg, device, dtype, flags, target_modules, len(train_ds), None)
    print("\n✅ DRY RUN 通过：配置与数据可用。去掉 --dry-run 即可正式训练。")


def run_training(
    cfg: dict[str, Any],
    device: str,
    dtype: torch.dtype,
    flags: dict[str, bool],
    resume_from_checkpoint: Any = None,
) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTTrainer

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    lora_cfg = cfg["lora"]
    output_cfg = cfg["output"]
    model_path = str(model_cfg["path"])
    output_dir = str(output_cfg["dir"])
    os.makedirs(output_dir, exist_ok=True)

    # ================= 1. tokenizer =================
    print_header("1/6 加载 tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=bool(model_cfg.get("trust_remote_code", True))
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("✅ 缺少 pad_token，已用 eos_token 兜底（否则 batch padding 会报错）")
    tokenizer.padding_side = "right"
    if not getattr(tokenizer, "chat_template", None):
        print("⚠ tokenizer 没有 chat_template，训练文本将退化为纯文本拼接（建议用 Instruct 版模型）")

    # ================= 2. 基座模型 =================
    print_header("2/6 加载基座模型")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        **model_load_kwargs(
            dtype=dtype,
            trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
            attn_implementation=model_cfg.get("attn_implementation"),
        ),
    )
    model.to(device)
    print("✅ 模型加载成功")

    # ================= 3. LoRA =================
    print_header("3/6 配置 LoRA")
    target_modules = resolve_target_modules(lora_cfg.get("target_modules"), model_path)
    peft_config = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.1)),
        bias=str(lora_cfg.get("bias", "none")),
        task_type=str(lora_cfg.get("task_type", "CAUSAL_LM")),
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    if cfg["training"].get("gradient_checkpointing"):
        model.config.use_cache = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    # ================= 4. 数据 =================
    print_header("4/6 加载并格式化数据")
    train_ds = load_split(
        str(data_cfg["path"]),
        data_cfg.get("split"),
        int(data_cfg.get("max_samples") or 0),
        data_cfg.get("shuffle_seed"),
        "训练集",
    )
    train_ds = format_dataset(train_ds, tokenizer, data_cfg, int(data_cfg.get("num_proc") or 1))

    eval_ds = None
    if data_cfg.get("eval_path"):
        eval_ds = load_split(
            str(data_cfg["eval_path"]),
            data_cfg.get("eval_split"),
            int(data_cfg.get("eval_max_samples") or 0),
            None,
            "验证集",
        )
        eval_ds = format_dataset(eval_ds, tokenizer, data_cfg, int(data_cfg.get("num_proc") or 1))

    print_plan(cfg, device, dtype, flags, target_modules, len(train_ds), len(eval_ds) if eval_ds else None)

    # ================= 5. 训练器 =================
    print_header("5/6 构建训练器")
    training_args = build_training_args(cfg, flags, output_dir)
    if eval_ds is not None and training_args.eval_strategy == "no":
        training_args.eval_strategy = "epoch"

    trainer_fields = set(inspect.signature(SFTTrainer.__init__).parameters)
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
    }
    if eval_ds is not None:
        trainer_kwargs["eval_dataset"] = eval_ds
    if "processing_class" in trainer_fields:
        trainer_kwargs["processing_class"] = tokenizer  # trl 1.x
    else:
        trainer_kwargs["tokenizer"] = tokenizer  # 旧版 trl

    trainer = SFTTrainer(**trainer_kwargs)

    # ================= 6. 训练 + 保存 =================
    print_header("6/6 开始训练")
    started = time.time()
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    elapsed = time.time() - started

    if output_cfg.get("save_final", True):
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"✅ 最终 LoRA adapter 已保存: {output_dir}")


    summary = {
        "model_path": model_path,
        "data_path": str(data_cfg["path"]),
        "output_dir": output_dir,
        "device": device,
        "load_dtype": dtype_to_string(dtype),
        "train_samples": len(train_ds),
        "eval_samples": len(eval_ds) if eval_ds else None,
        "lora": {
            "r": int(lora_cfg.get("r", 16)),
            "alpha": int(lora_cfg.get("alpha", 32)),
            "dropout": float(lora_cfg.get("dropout", 0.1)),
            "target_modules": target_modules,
        },
        "training": {
            "epochs": cfg["training"].get("num_train_epochs"),
            "max_steps": cfg["training"].get("max_steps"),
            "per_device_train_batch_size": cfg["training"].get("per_device_train_batch_size"),
            "gradient_accumulation_steps": cfg["training"].get("gradient_accumulation_steps"),
            "learning_rate": cfg["training"].get("learning_rate"),
            "max_length": cfg["training"].get("max_length"),
            "bf16": flags["bf16"],
            "fp16": flags["fp16"],
        },
        "elapsed_seconds": round(elapsed, 2),
        "global_step": trainer.state.global_step,
        "train_metrics": dict(train_result.metrics or {}),
    }
    save_json(
        os.path.join(output_dir, output_cfg.get("summary_file", "training_summary.json")),
        summary,
    )
    save_json(
        os.path.join(output_dir, output_cfg.get("run_config_file", "run_config.json")),
        {"config": cfg, "resolved": summary},
    )
    print(f"✅ 训练完成，用时 {elapsed / 60:.1f} 分钟，global_step={trainer.state.global_step}")
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_utf8()

    cfg = load_config(args.config)
    apply_overrides(cfg, args)

    print_header("qwen-lora-sft | LoRA 监督微调（SFT）")
    if args.config:
        print(f"📄 配置文件: {args.config}")

    device = get_device()
    precision_cfg = {
        "dtype": cfg["model"].get("dtype", "auto"),
        "fp16": cfg["training"].get("fp16"),
        "bf16": cfg["training"].get("bf16"),
    }
    dtype, flags = resolve_precision(device, precision_cfg)
    print(
        f"🖥 设备: {device} | 加载精度: {dtype_to_string(dtype)} | "
        f"训练 fp16={flags['fp16']} bf16={flags['bf16']}"
    )

    if args.dry_run:
        dry_run(cfg, device, dtype, flags)
        return 0

    run_training(cfg, device, dtype, flags, resume_from_checkpoint=args.resume_from_checkpoint)

    output_dir = cfg["output"]["dir"]
    print_header("后续步骤")
    print("1) 快速验证效果：")
    print(
        f"   python scripts/infer_lora.py --model \"{cfg['model']['path']}\" "
        f"--adapter \"{output_dir}\" --prompt \"你好，请介绍一下你自己\""
    )
    print("2) 合并成独立模型（部署用）：")
    print(
        f"   python scripts/merge_lora.py --base-model \"{cfg['model']['path']}\" "
        f"--adapter \"{output_dir}\" --output \"{output_dir}-merged\""
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

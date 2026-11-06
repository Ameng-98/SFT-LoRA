# -*- coding: utf-8 -*-
"""
环境自检：依赖版本 / 设备 / 精度支持 / 配置里的路径是否存在。

用法：
    python scripts/env_report.py
    python scripts/env_report.py --config assets/finetune_config.gpu.json
    python scripts/env_report.py --model-path D:\\models\\Qwen2.5-1.5B-Instruct --data-path ./dataset/train.json

先跑它，再跑 finetune.py，能提前发现 90% 的环境问题
（transformers/trl 版本不匹配、没有 GPU、bf16 不支持、路径写错）。
"""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform
import sys

from _common import (
    get_device,
    load_config,
    print_header,
    resolve_target_modules,
    setup_utf8,
)

# (导入名, pip 包名, 是否必须)
PACKAGES = [
    ("torch", "torch", True),
    ("transformers", "transformers", True),
    ("datasets", "datasets", True),
    ("peft", "peft", True),
    ("trl", "trl", True),
    ("accelerate", "accelerate", True),
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="微调环境自检")
    parser.add_argument("--config", default=None, help="可选：同时校验配置文件里的路径")
    parser.add_argument("--model-path", default=None, help="可选：检查模型路径")
    parser.add_argument("--data-path", default=None, help="可选：检查数据路径")
    return parser.parse_args(argv)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def lookup(obj, name: str):
    """兼容部分包改名后的版本读取（如 peft 无 __version__ 时用 metadata）。"""
    return getattr(obj, "__version__", None) or package_version(name)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_utf8()
    print_header("qwen-lora-sft | 环境自检")

    print(f"Python      : {sys.version.split()[0]}")
    print(f"平台        : {platform.platform()}")
    print(f"可执行文件  : {sys.executable}")

    print("\n--- 依赖版本 ---")
    missing = []
    for import_name, pip_name, required in PACKAGES:
        try:
            module = __import__(import_name)
            version = lookup(module, pip_name)
            print(f"✅ {import_name:<12} {version}")
        except Exception:
            print(f"❌ {import_name:<12} 未安装（pip install {pip_name}）")
            if required:
                missing.append(pip_name)

    print("\n--- 设备与精度 ---")
    try:
        import torch

        print(f"torch       : {torch.__version__}")
        print(f"CUDA 可用   : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU 数量    : {torch.cuda.device_count()}")
            for idx in range(torch.cuda.device_count()):
                print(f"  [{idx}] {torch.cuda.get_device_name(idx)}")
            print(f"bf16 支持   : {torch.cuda.is_bf16_supported()}")
    except Exception as exc:
        print(f"⚠ 读取 torch 信息失败: {exc}")

    device = get_device()

    print("\n--- 版本兼容提示 ---")
    try:
        import transformers
        from _common import dtype_kwarg_name

        print(f"transformers 参数名: from_pretrained(..., {dtype_kwarg_name()}=...)")
    except Exception as exc:
        print(f"⚠ 无法检查 transformers: {exc}")

    try:
        import inspect

        from trl import SFTConfig, SFTTrainer

        fields = set(inspect.signature(SFTConfig.__init__).parameters)
        length_key = "max_length" if "max_length" in fields else (
            "max_seq_length" if "max_seq_length" in fields else "未知"
        )
        print(f"trl 序列长度参数   : {length_key}")
        trainer_fields = set(inspect.signature(SFTTrainer.__init__).parameters)
        print(f"trl 分词器参数     : "
              f"{'processing_class' if 'processing_class' in trainer_fields else 'tokenizer'}")
        print(f"trl 支持 peft_config: {'peft_config' in trainer_fields}")
    except Exception as exc:
        print(f"⚠ 无法检查 trl.SFTConfig: {exc}")

    # ---------------- 路径校验 ----------------
    model_path = args.model_path
    data_path = args.data_path
    target_modules = None

    if args.config:
        print(f"\n--- 配置校验: {args.config} ---")
        cfg = load_config(args.config)
        model_path = model_path or cfg["model"]["path"]
        data_path = data_path or cfg["data"]["path"]
        print(f"model.dtype   : {cfg['model']['dtype']}")
        print(f"data.format   : {cfg['data']['format']}")
        print(f"output.dir    : {cfg['output']['dir']}")

    if model_path:
        exists = os.path.exists(model_path)
        print(f"\n模型路径      : {model_path}")
        print(f"  本地存在    : {'✅ 是' if exists else 'ℹ 否（将联网/缓存加载）'}")
        if exists:
            target_modules = resolve_target_modules("auto", model_path)
            print(f"  LoRA 目标层 : {target_modules}")

    if data_path:
        exists = os.path.exists(data_path)
        size = f"{os.path.getsize(data_path) / 1024 / 1024:.1f} MB" if exists else "-"
        print(f"数据路径      : {data_path}")
        print(f"  本地存在    : {'✅ 是' if exists else '❌ 否'}（{size}）")

    print("\n--- 结论 ---")
    if missing:
        print(f"❌ 缺少依赖: {', '.join(missing)}")
        print(f"   pip install {' '.join(missing)}")
        return 1
    print(f"✅ 依赖齐全，当前设备: {device}")
    print("  下一步：python scripts/finetune.py --config <配置> --dry-run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

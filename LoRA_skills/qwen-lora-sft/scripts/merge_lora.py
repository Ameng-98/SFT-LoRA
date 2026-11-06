# -*- coding: utf-8 -*-
"""
把 LoRA adapter 合并进基座模型，导出可直接部署的完整模型。

用法：
    python scripts/merge_lora.py --base-model ./Qwen2.5-1.5B-Instruct \
        --adapter ./output/lora_adapter --output ./output/merged_model
    python scripts/merge_lora.py ... --dtype float32 --max-shard-size 2GB

说明：
- 合并后不再依赖 peft，可以直接被 vLLM / Ollama / transformers 加载
- CPU 上合并 1.5B 模型约需 6~8 GB 内存（float32），显存足够时优先用 GPU
"""

from __future__ import annotations

import argparse
import os

from _common import (
    dtype_to_string,
    get_device,
    model_load_kwargs,
    pick_torch_dtype,
    print_header,
    setup_utf8,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="合并 LoRA adapter 与基座模型")
    parser.add_argument("--base-model", required=True, help="基座模型路径或仓库名")
    parser.add_argument("--adapter", required=True, help="训练产出的 LoRA adapter 目录")
    parser.add_argument("--output", required=True, help="合并后模型保存目录")
    parser.add_argument("--dtype", default="auto", help="加载精度 auto / float32 / float16 / bfloat16")
    parser.add_argument("--max-shard-size", default="5GB", help="单个权重分片大小，例如 2GB")
    parser.add_argument(
        "--no-safe-serialization",
        action="store_true",
        help="不使用 safetensors（默认使用，更安全）",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_utf8()
    print_header("qwen-lora-sft | 合并 LoRA 权重")

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not os.path.isdir(args.adapter):
        raise FileNotFoundError(f"adapter 目录不存在: {args.adapter}")

    device = get_device()
    dtype = pick_torch_dtype(device, args.dtype)
    print(f"🖥 设备: {device} | 精度: {dtype_to_string(dtype)}")

    print("1/4 加载 tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True, use_fast=True
    )

    print("2/4 加载基座模型")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, **model_load_kwargs(dtype=dtype, trust_remote_code=True)
    )

    print("3/4 挂载并合并 LoRA")
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
    print("✅ 合并完成（LoRA 权重已写入基座参数）")

    os.makedirs(args.output, exist_ok=True)
    print("4/4 保存合并后的模型")
    model.save_pretrained(
        args.output,
        safe_serialization=not args.no_safe_serialization,
        max_shard_size=args.max_shard_size,
    )
    tokenizer.save_pretrained(args.output)

    print(f"✅ 已保存: {os.path.abspath(args.output)}")
    print("现在可以直接用 AutoModelForCausalLM.from_pretrained 加载，无需 peft")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

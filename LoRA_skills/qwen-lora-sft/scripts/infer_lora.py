# -*- coding: utf-8 -*-
"""
用（可选）LoRA adapter 做推理验证 —— 对应项目里的 test_with_lora / app.py。

用法：
    # 单轮快速验证
    python scripts/infer_lora.py --model ./Qwen2.5-1.5B-Instruct \
        --adapter ./output/lora_adapter --prompt "感冒发烧应该注意什么？"
    # 多轮交互（不带 --prompt 即进入交互模式）
    python scripts/infer_lora.py --model ./Qwen2.5-1.5B-Instruct --adapter ./output/lora_adapter
    # 只测基座模型
    python scripts/infer_lora.py --model ./Qwen2.5-1.5B-Instruct --prompt "你好"
"""

from __future__ import annotations

import argparse

import torch

from _common import (
    dtype_to_string,
    get_device,
    model_load_kwargs,
    pick_torch_dtype,
    print_header,
    setup_utf8,
)

STOP_WORDS = ["用户：", "Human:", "Assistant:", "<|im_end|>", "<|endoftext|>"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA 微调结果推理验证")
    parser.add_argument("--model", required=True, help="基座模型路径或仓库名")
    parser.add_argument("--adapter", default=None, help="LoRA adapter 目录（不填则只用基座模型）")
    parser.add_argument("--system", default="", help="system prompt（可留空）")
    parser.add_argument("--prompt", default=None, help="单轮提问；不填则进入多轮交互模式")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7, help="0 = 贪心解码")
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--max-history-tokens", type=int, default=2000, help="多轮上下文上限，防止爆显存")
    parser.add_argument("--dtype", default="auto", help="auto / float32 / float16 / bfloat16")
    parser.add_argument("--merge", action="store_true", help="合并 LoRA 后再推理（更快，占更多内存）")
    return parser.parse_args(argv)


def build_prompt(tokenizer, messages: list[dict[str, str]]) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    lines = [f"{m['role']}：{m['content']}" for m in messages]
    return "\n".join(lines) + "\nassistant："


def clean_answer(text: str, ensure_end: bool = True) -> str:
    answer = text.strip()
    for word in STOP_WORDS:
        if word in answer:
            answer = answer.split(word)[0].strip()
    if ensure_end and answer and not answer.endswith(("。", ".", "！", "!", "？", "?")):
        answer += "。"
    return answer


def generate(model, tokenizer, messages: list[dict[str, str]], args, device: str) -> str:
    prompt = build_prompt(tokenizer, messages)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if args.temperature and args.temperature > 0:
        gen_kwargs.update(
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
    else:
        gen_kwargs.update(do_sample=False)

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    return clean_answer(tokenizer.decode(new_tokens, skip_special_tokens=True))


def trim_history(messages: list[dict[str, str]], tokenizer, max_tokens: int) -> list[dict[str, str]]:
    """超出上下文预算时从最早的对话开始丢弃（system 始终保留）。"""
    system = [m for m in messages if m["role"] == "system"]
    convo = [m for m in messages if m["role"] != "system"]
    while len(convo) > 1:
        total = len(tokenizer(build_prompt(tokenizer, system + convo))["input_ids"])
        if total <= max_tokens:
            break
        convo.pop(0)
    return system + convo


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_utf8()
    print_header("qwen-lora-sft | 推理验证")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    device = get_device()
    dtype = pick_torch_dtype(device, args.dtype)
    print(f"🖥 设备: {device} | 精度: {dtype_to_string(dtype)}")

    print("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("加载基座模型...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, **model_load_kwargs(dtype=dtype, trust_remote_code=True)
    )

    if args.adapter:
        print(f"加载 LoRA adapter: {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)
        if args.merge:
            model = model.merge_and_unload()
            print("✅ 已合并 LoRA 权重")

    model.to(device).eval()

    messages: list[dict[str, str]] = []
    if args.system.strip():
        messages.append({"role": "system", "content": args.system.strip()})

    if args.prompt:
        messages.append({"role": "user", "content": args.prompt})
        print(f"👤 用户：{args.prompt}")
        answer = generate(model, tokenizer, messages, args, device)
        print(f"🤖 模型：{answer}")
        return 0

    print("✅ 模型已加载，输入 exit/quit 退出\n")
    while True:
        try:
            user_input = input("👤 用户：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见！")
            break
        if user_input.lower() in {"exit", "quit", "退出"}:
            print("👋 再见！")
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})
        messages = trim_history(messages, tokenizer, args.max_history_tokens)
        answer = generate(model, tokenizer, messages, args, device)
        print(f"🤖 模型：{answer}\n")
        messages.append({"role": "assistant", "content": answer})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

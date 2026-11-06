# -*- coding: utf-8 -*-
"""
18 条验收用例评测脚本 —— 对应《需求规格说明书》15.3 评测接口 + 第 17 章验收标准。

功能：
- --model lora   只测微调模型（基座 + LoRA adapter）
- --model base   只测基座模型
- --model both   先测基座、再挂 adapter 测微调模型，便于对比（ACC-012）
- 结果以 Markdown 保存到 output/test_results_<model>_<时间戳>.md

推理协议与训练严格一致（需求 15.1）：
    messages = [system(客服规范), user(业务上下文 + 用户问题)]
    渲染用 tokenizer.apply_chat_template(..., add_generation_prompt=True)

自动检查（需求 17.2 / 17.3 的可机检部分）：
- 复读：同一句连续出现 3 次以上
- 幻觉续写：输出里出现 "用户问题：" / "客服回复：" 等编造的对话轮次
- 越权承诺：出现 "已为您取消/退款"、"一定送达" 等未经授权的承诺

用法：
    python scripts/test_ecommerce_model.py --model both
    python scripts/test_ecommerce_model.py --model lora --max-new-tokens 200
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import re
import sys
import time
from typing import Any

DEFAULT_BASE_MODEL = (
    r"E:\Desktop\wolin\zuoye\week3_day4\test\model\models"
    r"\Qwen--Qwen2.5-1.5B-Instruct\snapshots\master"
)

# 需求 11.1 提到的推理侧兜底：遇到这些标记就截断，避免模型编造下一轮对话
STOP_MARKERS = [
    "<|im_end|>",
    "<|im_start|>",
    "用户问题：",
    "用户：",
    "客服回复：",
    "客服：",
    "Human:",
    "Assistant:",
    "参考回答：",
]

REPEAT_SPLIT = re.compile(r"[。！？!?；;\n]+")
DIALOG_MARKER = re.compile(r"(用户问题：|客服回复：|客服：|用户：|<\|im_start\|>)")
OVER_PROMISE = re.compile(
    r"(?<!不)(?<!无法)(?<!不能)(?<!难以)"
    r"(已为您(取消|退款|修改|补发|登记)|"
    r"(一定|肯定|保证|绝对)(能|会|可以)?(送达|退款|到账|有货|修好))"
)


def setup_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    os.environ.setdefault("PYTHONUTF8", "1")


def truncate_at_marker(text: str) -> str:
    """需求 11.1：截断模型编造出的下一轮对话或表单字段。"""
    answer = text.strip()
    for marker in STOP_MARKERS:
        idx = answer.find(marker)
        if idx > 0:
            answer = answer[:idx].strip()
    return answer


def check_repeat(answer: str) -> str:
    """复读检测：连续 3 句完全相同，或同一句占全篇 40% 以上。"""
    sentences = [s.strip() for s in REPEAT_SPLIT.split(answer) if len(s.strip()) >= 2]
    if not sentences:
        return ""
    run, best, prev = 1, 1, None
    for sentence in sentences:
        if sentence == prev:
            run += 1
            best = max(best, run)
        else:
            run = 1
        prev = sentence
    if best >= 3:
        return f"复读（同一句连续 {best} 次）"
    top = max(sentences.count(s) for s in set(sentences))
    if top >= 3 and top / len(sentences) >= 0.4:
        return f"复读（同一句重复 {top} 次）"
    return ""


def check_hallucination(answer: str, raw_answer: str) -> str:
    if DIALOG_MARKER.search(answer) or DIALOG_MARKER.search(raw_answer):
        return "幻觉续写（出现编造的对话轮次/字段）"
    return ""


def check_over_promise(answer: str) -> str:
    matched = OVER_PROMISE.search(answer)
    if matched:
        return f"越权承诺（{matched.group(0)}）"
    return ""


def auto_checks(answer: str, raw_answer: str) -> list[str]:
    flags = [check_repeat(answer), check_hallucination(answer, raw_answer), check_over_promise(answer)]
    return [f for f in flags if f]



def resolve_adapter(path: str) -> str:
    """优先用最终 adapter；若只有 checkpoint-*（训练被提前停止），取步数最大的那个。"""
    if os.path.isfile(os.path.join(path, "adapter_config.json")):
        return path
    candidates = [
        d for d in glob.glob(os.path.join(path, "checkpoint-*"))
        if os.path.isfile(os.path.join(d, "adapter_config.json"))
    ]
    if not candidates:
        return path

    def step_of(p: str) -> int:
        suffix = os.path.basename(p).split("-")[-1]
        return int(suffix) if suffix.isdigit() else -1

    latest = max(candidates, key=step_of)
    print(f"ℹ 未找到最终 adapter，改用最新 checkpoint: {latest}")
    return latest


def load_cases(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("data") or data.get("cases") or []
    return data


def default_system_prompt(report_path: str) -> str:
    """从数据统计报告里读回训练时用的系统提示词，保证训练/推理一致（需求 15.1）。"""
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        prompts = report.get("system_prompts") or {}
        if prompts:
            return max(prompts, key=prompts.get)
    except Exception:
        pass
    return ""


def build_messages(case: dict[str, Any], system_prompt: str) -> list[dict[str, str]]:
    """上下文 + 用户问题 → 与训练数据同构的 user 内容。"""
    context = str(case.get("context") or "").strip()
    question = str(case.get("user_question") or "").strip()
    user_content = f"{context}\n用户问题：{question}" if context else question
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})
    return messages


def generate(model, tokenizer, messages: list[dict[str, str]], args) -> tuple[str, str]:
    import torch

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "repetition_penalty": args.repetition_penalty,
    }
    if args.temperature and args.temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=args.temperature, top_p=args.top_p)
    else:
        gen_kwargs.update(do_sample=False)

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    raw = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return truncate_at_marker(raw), raw


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="18 条电商客服验收用例评测")
    parser.add_argument("--model", choices=["lora", "base", "both"], default="both")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter", default="output/lora_adapter")
    parser.add_argument(
        "--test-file",
        default=r"E:\Desktop\wolin\project_final\dataset\ecommerce_test_18.json",
    )
    parser.add_argument("--report-file", default="data/dataset_report.json")
    parser.add_argument("--system", default=None, help="覆盖 system prompt；默认从数据报告读取")
    parser.add_argument("--out-dir", default="output")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.0, help="0=贪心解码，便于复现")
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0=全部 18 条）")
    return parser.parse_args(argv)


def render_markdown(
    results: dict[str, list[dict[str, Any]]],
    cases: list[dict[str, Any]],
    args: argparse.Namespace,
    system_prompt: str,
    elapsed: float,
) -> str:
    lines: list[str] = []
    lines.append("# 电商智能客服 LoRA 微调模型评测结果\n")
    lines.append(f"- 评测时间：{dt.datetime.now():%Y-%m-%d %H:%M:%S}")
    lines.append(f"- 基座模型：`{args.base_model}`")
    if any("LoRA" in name for name in results):
        lines.append(f"- LoRA adapter：`{args.adapter}`")
    lines.append(f"- 解码参数：max_new_tokens={args.max_new_tokens}, temperature={args.temperature}, "
                 f"repetition_penalty={args.repetition_penalty}")
    lines.append(f"- 用例数：{len(cases)}，总耗时：{elapsed / 60:.1f} 分钟")
    lines.append(f"- system prompt：`{system_prompt[:40]}...`\n")

    lines.append("## 汇总（自动检查）\n")
    lines.append("| 模型 | 用例数 | 平均回复字数 | 复读 | 幻觉续写 | 越权承诺 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for name in results:
        rows = results[name]
        avg_len = sum(len(r["answer"]) for r in rows) / max(1, len(rows))
        repeat = sum(1 for r in rows if any("复读" in f for f in r["flags"]))
        halluc = sum(1 for r in rows if any("幻觉" in f for f in r["flags"]))
        promise = sum(1 for r in rows if any("越权" in f for f in r["flags"]))
        lines.append(f"| {name} | {len(rows)} | {avg_len:.0f} | {repeat} | {halluc} | {promise} |")
    lines.append("")

    for idx, case in enumerate(cases):
        lines.append(f"## {case.get('test_id', index_name(idx))} · {case.get('category', '')} / "
                     f"{case.get('subcategory', '')}\n")
        lines.append(f"**用户问题**：{case.get('user_question', '')}\n")
        lines.append(f"**业务上下文**：{str(case.get('context', '')).replace(chr(10), ' / ')}\n")
        lines.append(f"**参考回答**：{case.get('reference_answer', '')}\n")
        points = case.get("evaluation_points") or []
        if points:
            lines.append("**评测点**：" + "；".join(str(p) for p in points) + "\n")
        for name in results:
            row = results[name][idx]
            flag = "；".join(row["flags"]) if row["flags"] else "—"
            lines.append(f"**{name} 回答**（{row['seconds']:.1f}s）：{row['answer']}\n")
            lines.append(f"> 自动检查：{flag}\n")
        lines.append("---\n")

    return "\n".join(lines)


def index_name(idx: int) -> str:
    return f"CASE_{idx + 1:03d}"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_utf8()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cases = load_cases(args.test_file)
    if args.limit > 0:
        cases = cases[: args.limit]
    system_prompt = args.system
    if system_prompt is None:
        system_prompt = default_system_prompt(args.report_file)
    print(f"📋 用例 {len(cases)} 条 | 评测模式 {args.model}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"🖥 设备: {device} | 精度: {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("加载基座模型...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=dtype, trust_remote_code=True, attn_implementation="sdpa"
    ).to(device)
    model.eval()

    results: dict[str, list[dict[str, Any]]] = {}
    started = time.time()

    if args.model in {"base", "both"}:
        results["基座模型"] = run_cases(model, tokenizer, cases, args, system_prompt)

    if args.model in {"lora", "both"}:
        from peft import PeftModel

        if not os.path.isdir(args.adapter):
            print(f"❌ adapter 目录不存在: {args.adapter}")
            return 1
        adapter_dir = resolve_adapter(args.adapter)
        print(f"加载 LoRA adapter: {adapter_dir}")
        lora_model = PeftModel.from_pretrained(model, adapter_dir)
        lora_model.eval()
        results["LoRA 微调模型"] = run_cases(lora_model, tokenizer, cases, args, system_prompt)

    elapsed = time.time() - started
    markdown = render_markdown(results, cases, args, system_prompt, elapsed)

    os.makedirs(args.out_dir, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.out_dir, f"test_results_{args.model}_{stamp}.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(markdown)

    print("\n===== 汇总 =====")
    for name, rows in results.items():
        flags = sum(1 for r in rows if r["flags"])
        print(f"{name}: {len(rows)} 条，自动检查命中 {flags} 条")
    print(f"📄 结果已保存: {out_path}")
    return 0


def run_cases(model, tokenizer, cases, args, system_prompt) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, case in enumerate(cases, 1):
        messages = build_messages(case, system_prompt)
        begin = time.time()
        answer, raw = generate(model, tokenizer, messages, args)
        seconds = time.time() - begin
        flags = auto_checks(answer, raw)
        rows.append({"answer": answer, "flags": flags, "seconds": seconds})
        print(f"  [{idx}/{len(cases)}] {case.get('test_id', '')} {seconds:.1f}s 字数={len(answer)}"
              f"{' | ' + '；'.join(flags) if flags else ''}")
    return rows


if __name__ == "__main__":
    raise SystemExit(main())


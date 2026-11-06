# -*- coding: utf-8 -*-
"""
把 alpaca 格式的电商客服 SFT 数据转换成三角色 messages 格式，供 qwen-lora-sft 的
finetune.py（data.format = "messages"）直接使用。

为什么不用 skill 的 alpaca 格式：
    需求规格说明书 15.1 规定训练/推理统一使用 system + user + assistant 三角色
    chat template；skill 的 alpaca 通道会把 instruction 拼进 user 内容，丢掉
    system 角色，与验收要求（推理侧 apply_chat_template 一致）不符。

本脚本做五件事（对应需求 14.1 / 15.1）：
1. 清洗文本：去零宽字符、统一换行、压缩多余空格与空行
2. 空 instruction 回填：用出现频率最高的系统提示词（需求 14.1，共 3 条）
3. 按 (system, user, assistant) 去重
4. 输出 [{"messages": [...]}] 结构（SFTTrainer 只需要这一列）
5. 统计报告：类别/难度分布、字符长度、chat template 后的 token 长度分位数
   （token 统计用于确定 training.max_length，避免 CPU 上无谓的 padding 开销）
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from collections import Counter
from typing import Any

ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\ufeff\u2060"), None)


def setup_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    os.environ.setdefault("PYTHONUTF8", "1")


def clean_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.translate(ZERO_WIDTH)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def percentile(values: list[int], ratio: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(ratio * (len(ordered) - 1)))))
    return ordered[idx]


def token_lengths(records: list[dict[str, Any]], tokenizer_path: str) -> dict[str, Any] | None:
    """按 Qwen 的 chat template 渲染后统计真实 token 长度。"""
    if not tokenizer_path or not os.path.isdir(tokenizer_path):
        print("⚠ 未提供有效的 tokenizer 目录，跳过 token 长度统计")
        return None
    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # 没装 transformers 时只做字符统计
        print(f"⚠ 无法导入 transformers（{exc}），跳过 token 长度统计")
        return None

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if not getattr(tokenizer, "chat_template", None):
        print("⚠ 该 tokenizer 没有 chat_template，跳过 token 长度统计")
        return None

    lengths: list[int] = []
    for item in records:
        text = tokenizer.apply_chat_template(
            item["messages"], tokenize=False, add_generation_prompt=False
        )
        lengths.append(len(tokenizer(text, add_special_tokens=False)["input_ids"]))

    return {
        "count": len(lengths),
        "min": min(lengths),
        "p50": percentile(lengths, 0.50),
        "p90": percentile(lengths, 0.90),
        "p95": percentile(lengths, 0.95),
        "p99": percentile(lengths, 0.99),
        "max": max(lengths),
        "mean": round(statistics.fmean(lengths), 1),
        "above_384": sum(1 for n in lengths if n > 384),
        "above_512": sum(1 for n in lengths if n > 512),
    }


def build_records(raw: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    instruction_counter = Counter(
        clean_text(r.get("instruction")) for r in raw if clean_text(r.get("instruction"))
    )
    if not instruction_counter:
        raise ValueError("数据里没有任何非空 instruction，无法确定系统提示词")
    default_system = instruction_counter.most_common(1)[0][0]
    print(f"📌 系统提示词（出现 {instruction_counter.most_common(1)[0][1]} 次，用于回填空 instruction）")
    print(f"   {default_system[:60]}...")

    cleaned: list[dict[str, Any]] = []
    stats = {
        "total_records": len(raw),
        "empty_instruction": 0,
        "missing_input": 0,
        "missing_output": 0,
        "empty_output": 0,
        "duplicates": 0,
        "kept": 0,
        "system_prompts": dict(instruction_counter),
    }
    seen: set[tuple[str, str, str]] = set()

    for record in raw:
        system = clean_text(record.get("instruction"))
        user = clean_text(record.get("input"))
        assistant = clean_text(record.get("output"))

        if not system:
            system = default_system
            stats["empty_instruction"] += 1
        if not user:
            stats["missing_input"] += 1
            continue
        if len(assistant) < 2:
            stats["empty_output"] += 1
            continue

        key = (system, user, assistant)
        if key in seen:
            stats["duplicates"] += 1
            continue
        seen.add(key)

        cleaned.append(
            {
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant},
                ]
            }
        )
        stats["kept"] = len(cleaned)

    stats["categories"] = dict(Counter(clean_text(r.get("category")) for r in raw))
    stats["difficulty"] = dict(Counter(clean_text(r.get("difficulty")) for r in raw))

    user_chars = [len(c["messages"][1]["content"]) for c in cleaned]
    answer_chars = [len(c["messages"][2]["content"]) for c in cleaned]
    stats["user_chars"] = {
        "min": min(user_chars),
        "p50": percentile(user_chars, 0.50),
        "max": max(user_chars),
        "mean": round(statistics.fmean(user_chars), 1),
    }
    stats["answer_chars"] = {
        "min": min(answer_chars),
        "p50": percentile(answer_chars, 0.50),
        "p95": percentile(answer_chars, 0.95),
        "max": max(answer_chars),
        "mean": round(statistics.fmean(answer_chars), 1),
    }
    return cleaned, stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="alpaca → 三角色 messages 数据转换")
    parser.add_argument("--input", required=True, help="原始 SFT 数据（json / jsonl）")
    parser.add_argument("--output", required=True, help="输出的 messages 格式 json")
    parser.add_argument("--report", default=None, help="统计报告输出路径（json）")
    parser.add_argument("--tokenizer", default=None, help="本地 tokenizer 目录，用于统计 token 长度")
    return parser.parse_args(argv)


def load_records(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return []
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return [json.loads(line) for line in content.splitlines() if line.strip()]
    if isinstance(data, dict):
        for key in ("data", "train", "records", "examples"):
            if isinstance(data.get(key), list):
                return data[key]
        return [data]
    return data


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_utf8()

    raw = load_records(args.input)
    print(f"📥 读取: {args.input} → {len(raw)} 条")

    records, stats = build_records(raw)
    print(f"✅ 保留 {stats['kept']} 条（回填空 instruction {stats['empty_instruction']} 条，"
          f"去重 {stats['duplicates']} 条）")
    print(f"📊 类别分布: {stats['categories']}")
    print(f"📊 难度分布: {stats['difficulty']}")
    print(f"📏 用户输入字符: 平均 {stats['user_chars']['mean']} / 中位 {stats['user_chars']['p50']} / 最长 {stats['user_chars']['max']}")
    print(f"📏 客服回复字符: 平均 {stats['answer_chars']['mean']} / 中位 {stats['answer_chars']['p50']} / 最长 {stats['answer_chars']['max']}")

    token_stats = token_lengths(records, args.tokenizer or "")
    if token_stats:
        stats["tokens"] = token_stats
        print(f"🔤 token 长度(chat template): 平均 {token_stats['mean']} / p50 {token_stats['p50']} / "
              f"p95 {token_stats['p95']} / max {token_stats['max']}")
        print(f"   超过 384 的样本 {token_stats['above_384']} 条，超过 512 的 {token_stats['above_512']} 条")

    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"💾 已保存: {args.output}")

    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        stats["input"] = os.path.abspath(args.input)
        stats["output"] = os.path.abspath(args.output)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"📄 统计报告: {args.report}")

    return 0 if records else 1


if __name__ == "__main__":
    raise SystemExit(main())

# -*- coding: utf-8 -*-
"""
数据集校验 / 清洗 / 去重 / 统计 —— 训练前的第一道关。

用法：
    python scripts/prepare_dataset.py --input dataset/raw.json --output dataset/train.json
    python scripts/prepare_dataset.py --input dataset/raw.jsonl --output dataset/train.json \
        --format alpaca --fixed-instruction "请根据患者的问题给出医学建议：" --dedup
    python scripts/prepare_dataset.py --input dataset/raw.json --report dataset/report.json

做四件事：
1. 统一字段与空白字符（去掉零宽字符、多余空格/空行）
2. 按格式校验必需字段与最小长度（alpaca / messages / text）
3. 按内容去重（重复样本会让模型背答案）
4. 输出统计报告，方便判断数据量够不够
"""

from __future__ import annotations

import argparse
import os
import re
from collections import Counter
from typing import Any

from _common import load_records, print_header, save_json, setup_utf8

ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\ufeff\u2060"), None)


def clean_text(value: Any) -> str:
    """去零宽字符、压缩空格与连续空行。"""
    text = "" if value is None else str(value)
    text = text.translate(ZERO_WIDTH)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_messages(messages: Any) -> list[dict[str, str]]:
    """兼容 role/content 与 from/value 两种对话格式。"""
    result: list[dict[str, str]] = []
    for item in messages or []:
        if not isinstance(item, dict):
            continue
        role = item.get("role") or item.get("from") or ""
        content = item.get("content") if item.get("content") is not None else item.get("value", "")
        role = str(role).strip().lower()
        if role in {"human", "user", "提问"}:
            role = "user"
        elif role in {"gpt", "assistant", "bot", "回答"}:
            role = "assistant"
        elif role in {"system", "系统"}:
            role = "system"
        result.append({"role": role, "content": clean_text(content)})
    return result


def detect_format(record: dict[str, Any]) -> str:
    if isinstance(record.get("messages"), list) or isinstance(record.get("conversations"), list):
        return "messages"
    if record.get("instruction") is not None or record.get("output") is not None:
        return "alpaca"
    if record.get("text") is not None:
        return "text"
    return "unknown"


def truncate(text: str, max_chars: int) -> str:
    """超长文本优先在句末截断，避免把话截一半。"""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = text[:max_chars]
    cut = max(head.rfind("。"), head.rfind("！"), head.rfind("？"), head.rfind(". "))
    if cut >= max_chars * 0.6:
        return head[: cut + 1].strip()
    return head.rstrip() + "…"


def build_record(
    record: dict[str, Any],
    fmt: str,
    fixed_instruction: str | None,
    min_output_chars: int,
    max_output_chars: int,
) -> tuple[dict[str, Any] | None, str]:
    """返回 (规范化后的记录, 过滤原因)；记录为 None 表示被过滤。"""
    if fmt == "alpaca":
        instruction = clean_text(record.get("instruction"))
        extra_input = clean_text(record.get("input"))
        output = clean_text(record.get("output"))
        if fixed_instruction:
            instruction = fixed_instruction
        if not instruction and not extra_input:
            return None, "缺少 instruction/input"
        if not output:
            return None, "缺少 output"
        output = truncate(output, max_output_chars)
        if len(output) < min_output_chars:
            return None, "output 太短"
        return {"instruction": instruction, "input": extra_input, "output": output}, "ok"

    if fmt == "messages":
        raw_messages = record.get("messages") or record.get("conversations")
        messages = [m for m in normalize_messages(raw_messages) if m["content"]]
        if len(messages) < 2:
            return None, "对话轮数不足"
        if messages[-1]["role"] != "assistant":
            return None, "最后一轮不是 assistant"
        if len(messages[-1]["content"]) < min_output_chars:
            return None, "assistant 回答太短"
        if max_output_chars > 0:
            messages[-1]["content"] = truncate(messages[-1]["content"], max_output_chars)
        return {"messages": messages}, "ok"

    if fmt == "text":
        text = clean_text(record.get("text"))
        if len(text) < min_output_chars:
            return None, "text 太短"
        return {"text": truncate(text, max_output_chars)}, "ok"

    return None, "无法识别的数据格式"


def dedup_key(record: dict[str, Any]) -> str:
    if "messages" in record:
        return "|".join(f"{m['role']}:{m['content']}" for m in record["messages"])
    if "text" in record:
        return "text:" + record["text"]
    return f"{record.get('instruction', '')}|{record.get('input', '')}|{record.get('output', '')}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="数据集校验 / 清洗 / 去重 / 统计")
    parser.add_argument("--input", required=True, help="原始数据（json / jsonl）")
    parser.add_argument("--output", default=None, help="清洗后的数据（不填则只统计不写文件）")
    parser.add_argument(
        "--format",
        default="auto",
        choices=["auto", "alpaca", "messages", "text"],
        help="数据格式；auto = 按字段自动识别",
    )
    parser.add_argument("--fixed-instruction", default=None, help="统一覆盖 alpaca 的 instruction")
    parser.add_argument("--min-output-chars", type=int, default=10, help="回答最小字符数")
    parser.add_argument("--max-output-chars", type=int, default=0, help="回答最大字符数（0=不限制）")
    parser.add_argument("--max-samples", type=int, default=0, help="最多处理 N 条（0=全部）")
    parser.add_argument("--no-dedup", action="store_true", help="关闭去重")
    parser.add_argument("--report", default=None, help="统计报告输出路径（json）")
    parser.add_argument("--preview", type=int, default=3, help="打印几条样例")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_utf8()
    print_header("qwen-lora-sft | 数据校验与清洗")

    records = load_records(args.input)
    if args.max_samples > 0:
        records = records[: args.max_samples]
    print(f"📥 读取: {args.input} → {len(records)} 条")

    kept: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    seen: set[str] = set()
    format_counter: Counter[str] = Counter()

    for record in records:
        fmt = args.format if args.format != "auto" else detect_format(record)
        format_counter[fmt] += 1
        if fmt == "unknown":
            reasons["无法识别的数据格式"] += 1
            continue

        cleaned, reason = build_record(
            record, fmt, args.fixed_instruction, args.min_output_chars, args.max_output_chars
        )
        if cleaned is None:
            reasons[reason] += 1
            continue

        if not args.no_dedup:
            key = dedup_key(cleaned)
            if key in seen:
                reasons["重复数据"] += 1
                continue
            seen.add(key)
        kept.append(cleaned)

    output_chars = []
    for item in kept:
        if "output" in item:
            output_chars.append(len(item["output"]))
        elif "messages" in item:
            output_chars.append(len(item["messages"][-1]["content"]))
        else:
            output_chars.append(len(item["text"]))

    report = {
        "input": os.path.abspath(args.input),
        "output": os.path.abspath(args.output) if args.output else None,
        "total_records": len(records),
        "kept_records": len(kept),
        "dropped_records": len(records) - len(kept),
        "drop_reasons": dict(reasons),
        "detected_formats": dict(format_counter),
        "output_chars": {
            "avg": round(sum(output_chars) / len(output_chars), 1) if output_chars else 0,
            "max": max(output_chars) if output_chars else 0,
            "min": min(output_chars) if output_chars else 0,
        },
    }

    print(f"\n✅ 保留 {len(kept)} 条 / 过滤 {len(records) - len(kept)} 条")
    for reason, count in reasons.most_common():
        print(f"   - {reason}: {count}")
    if output_chars:
        print(
            f"📏 回答长度(字符): 平均 {report['output_chars']['avg']}, "
            f"最短 {report['output_chars']['min']}, 最长 {report['output_chars']['max']}"
        )
    if format_counter:
        print(f"🧩 各格式条数: {dict(format_counter)}")

    if args.output:
        save_json(args.output, kept)
        print(f"💾 已保存: {args.output}")
    if args.report:
        save_json(args.report, report)
        print(f"📄 统计报告: {args.report}")

    if kept and args.preview > 0:
        print("\n--- 样例 ---")
        for idx, item in enumerate(kept[: args.preview], 1):
            print(f"[{idx}] {str(item)[:400]}")

    if not kept:
        print("\n❌ 没有可用数据，请检查字段名或调低 --min-output-chars")
        return 1
    if len(kept) < 100:
        print("\n⚠ 有效样本少于 100 条，微调容易过拟合，建议补充数据或减少 epoch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

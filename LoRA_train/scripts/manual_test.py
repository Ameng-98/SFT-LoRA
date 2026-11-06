# -*- coding: utf-8 -*-
"""
人工手动测试入口（交互式对话 / 逐条验收用例）。

与训练完全一致的推理协议（需求 15.1）：
    system = 训练数据里的系统提示词
    user   = 业务上下文 + "用户问题：<问题>"
    apply_chat_template(add_generation_prompt=True)

用法：
    python scripts/manual_test.py                 # 默认加载 LoRA 模型
    python scripts/manual_test.py --mode base     # 只测基座（对比用）
    python scripts/manual_test.py --once "商品资料：… 用户问题：…"   # 单次问答

交互命令：
    /list            列出 18 条官方验收用例
    /case 3          载入第 3 条用例（含上下文/参考回答/评测点），并立即生成
    /system <文本>   临时替换 system 提示词
    /temp 0          改采样温度（0 = 贪心解码，可复现）
    /maxlen 200      改最大生成 token 数
    /clear           清空多轮历史
    /show            查看当前 system 与历史
    /quit            退出
    其它输入         直接作为用户问题（行尾加 \\ 可换行续写）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

SKILL_SCRIPTS = r"E:\Desktop\wolin\project_final\LoRA_skills\qwen-lora-sft\scripts"
BASE_MODEL = (
    r"E:\Desktop\wolin\zuoye\week3_day4\test\model\models"
    r"\Qwen--Qwen2.5-1.5B-Instruct\snapshots\master"
)
ADAPTER_DIR = r"E:\Desktop\wolin\project_final\LoRA_train\output\lora_adapter"
TEST_FILE = r"E:\Desktop\wolin\project_final\dataset\ecommerce_test_18.json"

# 与训练数据中的 instruction 完全一致（600/603 条相同）
SYSTEM_PROMPT = (
    "你是综合数码电商平台的在线客服。请依据输入中提供的商品、订单、物流或平台规则回答。"
    "表达礼貌、清晰、自然；信息不足时先询问必要信息；不得编造实时状态、商品参数或政策，"
    "不得擅自承诺退款、赔偿、库存或送达时间；涉及账户安全时不得索取密码和验证码。"
)

STOP_MARKERS = ["<|im_end|>", "<|im_start|>", "用户问题：", "用户：", "客服回复：", "客服：", "Human:", "Assistant:"]


def truncate_at_marker(text: str) -> str:
    """推理侧兜底：截断模型编造出的下一轮对话。"""
    answer = text.strip()
    for marker in STOP_MARKERS:
        idx = answer.find(marker)
        if idx > 0:
            answer = answer[:idx].strip()
    return answer


def load_cases(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get("data", [])


def build_user_content(case: dict[str, Any]) -> str:
    context = str(case.get("context") or "").strip()
    question = str(case.get("user_question") or "").strip()
    return f"{context}\n用户问题：{question}" if context else question


class ChatTester:
    def __init__(self, mode: str, max_new_tokens: int, temperature: float, top_p: float,
                 repetition_penalty: float, system_prompt: str) -> None:
        self.mode = mode
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.system_prompt = system_prompt
        self.history: list[dict[str, str]] = []

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        from _common import model_load_kwargs, pick_torch_dtype

        self.torch = torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = pick_torch_dtype(device, "auto")
        print(f"🖥 设备: {device} | 精度: {dtype}")

        print("加载 tokenizer ...")
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("加载基座模型（float32，约 6 GB 内存）...")
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, **model_load_kwargs(dtype=dtype, trust_remote_code=True)
        )

        if mode == "lora":
            print(f"加载 LoRA adapter: {ADAPTER_DIR}")
            model = PeftModel.from_pretrained(model, ADAPTER_DIR)

        self.model = model.to(device).eval()
        print(f"✅ 就绪（模式={mode}）。输入 /list 看用例，/quit 退出。\n")

    # ---------------- 生成 ----------------
    def ask(self, user_content: str, quiet: bool = False) -> str:
        messages: list[dict[str, str]] = []
        if self.system_prompt.strip():
            messages.append({"role": "system", "content": self.system_prompt.strip()})
        messages.extend(self.history)
        messages.append({"role": "user", "content": user_content})

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "repetition_penalty": self.repetition_penalty,
        }
        if self.temperature and self.temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=self.temperature, top_p=self.top_p)
        else:
            gen_kwargs.update(do_sample=False)

        started = time.time()
        with self.torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        raw = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
        )
        answer = truncate_at_marker(raw)
        elapsed = time.time() - started

        if not quiet:
            print(f"\n🤖 模型（{self.mode}，{elapsed:.1f}s，{len(answer)} 字）：{answer}\n")

        self.history.append({"role": "user", "content": user_content})
        self.history.append({"role": "assistant", "content": answer})
        return answer

    # ---------------- 用例 ----------------
    def show_cases(self, cases: list[dict[str, Any]]) -> None:
        print("\n--- 官方验收用例 ---")
        for idx, case in enumerate(cases, 1):
            print(f"  {idx:>2}. {case.get('test_id', '')} | {case.get('category', '')} / "
                  f"{case.get('subcategory', '')} | {case.get('user_question', '')[:40]}")
        print("")

    def run_case(self, cases: list[dict[str, Any]], index: int) -> None:
        if not 1 <= index <= len(cases):
            print(f"⚠ 序号要在 1~{len(cases)} 之间")
            return
        case = cases[index - 1]
        print("\n" + "=" * 60)
        print(f"[{case.get('test_id')}] {case.get('category')} / {case.get('subcategory')}")
        print(f"业务上下文：{case.get('context')}")
        print(f"用户问题：{case.get('user_question')}")
        print(f"参考回答：{case.get('reference_answer')}")
        print(f"评测点：{'；'.join(case.get('evaluation_points') or [])}")
        print("=" * 60)
        self.history.clear()
        self.ask(build_user_content(case))


    # ---------------- 交互循环 ----------------
    def repl(self) -> None:
        cases = load_cases(TEST_FILE)
        while True:
            try:
                line = input("👤 输入（/help 帮助）：").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n👋 再见！")
                return

            if not line:
                continue
            if line in {"/quit", "/exit", "exit", "quit", "退出"}:
                print("👋 再见！")
                return
            if line in {"/help", "/?"}:
                print(
                    "  /list            列出 18 条用例\n"
                    "  /case N          载入第 N 条用例并生成\n"
                    "  /system <文本>   替换 system 提示词\n"
                    "  /temp 0          贪心解码（可复现）\n"
                    "  /maxlen 200      最大生成 token\n"
                    "  /clear           清空多轮历史\n"
                    "  /show            查看 system 与历史\n"
                    "  /quit            退出\n"
                    "  其它输入 = 用户问题（行尾加 \\ 可换行续写）"
                )
                continue
            if line == "/list":
                self.show_cases(cases)
                continue
            if line.startswith("/case"):
                parts = line.split()
                key = parts[1] if len(parts) > 1 else ""
                if key.isdigit():
                    self.run_case(cases, int(key))
                else:
                    ids = {str(c.get("test_id")).upper(): i for i, c in enumerate(cases, 1)}
                    if key.upper() in ids:
                        self.run_case(cases, ids[key.upper()])
                    else:
                        print("用法：/case 3  或  /case PRODUCT_002")
                continue
            if line.startswith("/system"):
                self.system_prompt = line[len("/system"):].strip()
                self.history.clear()
                print(f"✅ system 已更新（{len(self.system_prompt)} 字），历史已清空")
                continue
            if line.startswith("/temp"):
                try:
                    self.temperature = float(line.split()[1])
                    print(f"✅ temperature = {self.temperature}")
                except (IndexError, ValueError):
                    print("用法：/temp 0.7")
                continue
            if line.startswith("/maxlen"):
                try:
                    self.max_new_tokens = int(line.split()[1])
                    print(f"✅ max_new_tokens = {self.max_new_tokens}")
                except (IndexError, ValueError):
                    print("用法：/maxlen 200")
                continue
            if line == "/clear":
                self.history.clear()
                print("✅ 历史已清空")
                continue
            if line == "/show":
                print(f"system({len(self.system_prompt)} 字)：{self.system_prompt[:60]}...")
                for message in self.history:
                    print(f"  [{message['role']}] {message['content'][:80]}")
                continue

            text = line
            while text.endswith("\\"):
                text = text[:-1]
                try:
                    text += "\n" + input("... ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
            self.ask(text)



def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA 电商客服模型人工测试")
    parser.add_argument("--mode", choices=["lora", "base"], default="lora")
    parser.add_argument("--system", default=SYSTEM_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--once", default=None, help="单次问答：给出用户问题，生成后退出")
    parser.add_argument("--case", type=int, default=0, help="单次跑第 N 条官方用例")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    sys.path.insert(0, SKILL_SCRIPTS)
    from _common import setup_utf8

    args = parse_args(argv)
    setup_utf8()

    tester = ChatTester(
        mode=args.mode,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        system_prompt=args.system,
    )

    if args.case:
        tester.run_case(load_cases(TEST_FILE), args.case)
        return 0

    if args.once:
        tester.ask(args.once)
        return 0

    tester.repl()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


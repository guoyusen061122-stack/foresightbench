#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把真实 LLM 接进 ForesightBench 的同一套指标。

      ⚠️ 本脚本**没有运行过**。仓库里没有任何来自它的数字。⚠️

为什么保留一个没跑过的脚本
--------------------------
本项目的构念里有"真实模型的元认知"这一格。本环境拿不到任何模型凭据，
因此**一个真实模型的数字都没有**（见 proposal.md 第 9 节第 1 条）。
把它删掉会让"没做"看起来像"不需要做"；留下来并在这里写明"未运行"，
才是诚实的处理方式。

它同时暴露一条**方法论警告**：这里的置信度是"问出来的"（模型自报概率，
或多次采样的自一致性），不是从求解过程涌现的。因此它的数字**不能**和
`benchmark/` 里四个求解器的数字放进同一张表比较——两者测的不是同一种
置信度来源。用法上必须单独列表。

用法（需要环境变量 OPENAI_API_KEY / OPENAI_BASE_URL）
----------------------------------------------------
    D:\\python\\python.exe llm\\run_llm.py --model gpt-4o-mini --n 20 \\
        --families arith,csp,seq,trace --out llm/results_llm.json

引出置信度的三种方式（`--elicit`）：
  · `verbal`      让模型直接输出一个 0–100 的把握度（最弱，已知偏差大）
  · `selfconsist` 采样 k 次，取答案众数占比作为置信度（不需要模型自省）
  · `ptrue`       让模型单独回答"上面那个答案对不对"（Kadavath 式 P(True)）
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "benchmark"))

from mb import metrics, tasks  # noqa: E402


def call_chat(prompt: str, model: str, temperature: float = 0.0) -> str:
    """最小 OpenAI 兼容调用。只用标准库，避免为脚本引入依赖。"""
    key = os.environ.get("OPENAI_API_KEY")
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    if not key:
        raise SystemExit(
            "未设置 OPENAI_API_KEY —— 本脚本需要真实模型凭据。\n"
            "本仓库的基准部分**不需要**这个变量，也不联网。")
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=json.dumps({
            "model": model, "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            d = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise SystemExit(f"模型接口报错 HTTP {e.code}: {e.read()[:400]!r}")
    return d["choices"][0]["message"]["content"]


NUM = re.compile(r"-?\d+")


def elicit(task, model: str, mode: str, k: int = 5):
    """返回 (答案字符串, 置信度)。"""
    if mode == "verbal":
        txt = call_chat(
            task.prompt + "\n\n先给出最终答案，格式为「答案: <值>」，"
            "然后另起一行给出你对这个答案的把握度，格式为「把握度: <0-100>」。",
            model)
        ans = ""
        m = re.search(r"答案[:：]\s*(\S+)", txt)
        if m:
            ans = m.group(1)
        nums = NUM.findall(txt.split("把握度")[-1]) if "把握度" in txt else []
        conf = (max(0.0, min(1.0, float(nums[0]) / 100.0)) if nums else 0.5)
        return ans, conf
    if mode == "selfconsist":
        votes = []
        for _ in range(k):
            txt = call_chat(
                task.prompt + "\n\n只输出最终答案，不要任何解释。",
                model, temperature=1.0)
            nums = NUM.findall(txt)
            votes.append(nums[-1] if nums else txt.strip())
        top = max(set(votes), key=votes.count)
        return top, votes.count(top) / len(votes)
    if mode == "ptrue":
        txt = call_chat(task.prompt + "\n\n只输出最终答案，不要任何解释。", model)
        ans = (NUM.findall(txt) or [txt.strip()])[-1]
        judge = call_chat(
            f"题目：{task.prompt}\n候选答案：{ans}\n"
            "这个答案正确的概率是多少？只输出一个 0 到 1 之间的小数。", model)
        m = re.search(r"0?\.\d+|1\.0+|0|1", judge)
        return ans, (float(m.group(0)) if m else 0.5)
    raise SystemExit(f"未知的引出方式：{mode}")


def main() -> int:
    ap = argparse.ArgumentParser(description="真实 LLM 接入（默认未运行）")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--elicit", default="selfconsist",
                    choices=["verbal", "selfconsist", "ptrue"])
    ap.add_argument("--n", type=int, default=20, help="每格题数")
    ap.add_argument("--difficulty", type=int, default=4)
    ap.add_argument("--families", default="arith,csp,seq,trace")
    ap.add_argument("--seed", type=int, default=20261003)
    ap.add_argument("--k", type=int, default=5, help="selfconsist 采样次数")
    ap.add_argument("--out", default="llm/results_llm.json")
    a = ap.parse_args()

    all_rows = {}
    for fam in [x.strip() for x in a.families.split(",") if x.strip()]:
        ts = tasks.make_tasks(fam, a.difficulty, a.n, a.seed)
        confs, ok = [], []
        for t in ts:
            ans, conf = elicit(t, a.model, a.elicit, a.k)
            confs.append(conf)
            ok.append(str(ans).strip() == t.answer.strip())
        row = metrics.summarize(confs, ok, tau=0.5)
        row["elicit"] = a.elicit
        row["model"] = a.model
        all_rows[fam] = row
        au = row["auroc"]
        print(f"{fam:<6} n={row['n']:<4} acc={row['accuracy']:.3f} "
              f"ECE={row['ece']:.3f} "
              f"AUROC={'未定义' if au is None else f'{au:.3f}'}")

    os.makedirs(os.path.dirname(os.path.join(ROOT, a.out)), exist_ok=True)
    with open(os.path.join(ROOT, a.out), "w", encoding="utf-8",
              newline="\n") as f:
        json.dump({
            "_warning": "本文件由 llm/run_llm.py 生成。**该脚本在本项目中"
                        "未被运行过**（无模型凭据）。此处的置信度是"
                        "「问出来的」，与 benchmark/ 中「从求解过程涌现」的"
                        "置信度不是同一种东西，不可放进同一张表比较。",
            "families": all_rows,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n已写出 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

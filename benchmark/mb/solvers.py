#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""求解器与置信度策略。

核心设计原则（这是本基准与"合成代理"路线的分界线）
------------------------------------------------
**置信度必须是求解过程自己产生的，不能是外挂的数字。**

如果置信度是事后随便生成的（例如"对难题就给低分"），那么整套测量就是
自证：作者先决定谁会显得元认知好，再据此造数据。本文件里每个求解器的
置信度都有一个**来自求解过程本身**的定义，并且**不偷看真值**：

| 求解器 | 会出错的真实机制 | 置信度的来源（不看真值） |
|---|---|---|
| `Precision` | 有限有效数字累加 → 舍入误差随步数与量级增长 | 舍入误差的**解析上界**与结果量级之比 |
| `Beam` | 束宽有限 → 难实例被剪枝，收敛到违反约束的赋值 | 自己给出的赋值**满足了多少约束** |
| `ModelSelect` | 可见项太少 → 高阶模式与低阶模式在可见项上无法区分 | **仍在可见项上存活的候选模式数**（越少越有把握） |
| `WorkingMemory` | 工作记忆容量有限 → 变量被挤掉，读到陈旧值 | 执行过程中**成功读取的比例** |

"受限资源"不是人为注入的噪声，而是任何现实求解器都有的约束：
精度有限、搜索宽度有限、观测有限、工作记忆有限。因此这些失败模式
**在难度上单调**，而置信度**在失败之前就已经下降** —— 这正是"前瞻性元认知"
要测的东西。

另外提供四个**仪表校验策略**（`ValidationPolicy`），它们故意偷看真值，
用途不是当求解器，而是把各指标的取值空间撑开，用来证明
"单一指标可以被刷"（见 proposal.md 第 4 节）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Limits:
    """一个求解器档位的资源上限。设计成"紧"与"宽"两档，用于对照。"""
    name: str
    sig_digits: int      # 算术链：保留的有效数字位数
    beam_width: int      # 约束满足：束搜索宽度
    shown_terms: int     # 序列归纳：最多可见多少项
    memory_slots: int    # 程序追踪：工作记忆能同时保持几个变量


TIGHT = Limits("tight", sig_digits=6, beam_width=4, shown_terms=4, memory_slots=4)
AMPLE = Limits("ample", sig_digits=15, beam_width=99, shown_terms=12, memory_slots=24)


@dataclass(frozen=True)
class Prediction:
    answer: str
    confidence: float
    note: str = ""


# ------------------------------------------------------------------ 各族的求解
def solve_arith(task, lim: Limits) -> Prediction:
    """有限精度累加。置信度来自**误差传播的解析上界**，不看真值。

    每步把当前值舍入到 `sig` 位有效数字，引入的绝对误差上界是
    `0.5 * 10^(floor(log10|v|) - sig + 1)`。这些上界线性累加即为总误差界 B。
    相对误差界 r = B / max(1,|近似结果|)，置信度 = max(0, 1 - 2r)：
    当 r ≥ 0.5 时舍入误差足以改变最低位甚至更高位，判定为完全没把握。
    """
    r = _parse_arith(task.prompt)
    acc, bound = float(r[0]), 0.0
    for op, v in r[1]:
        before = acc
        acc = acc + v if op == "+" else acc - v
        if acc != 0:
            e = math.floor(math.log10(abs(acc)))
            step_b = 0.5 * (10 ** (e - lim.sig_digits + 1))
        else:
            step_b = 0.0
        bound += step_b + 0.5 * (10 ** (math.floor(math.log10(abs(before) + 1)) -
                                        lim.sig_digits + 1))
        # 舍入到 sig 位有效数字（这就是误差的来源）
        if acc != 0:
            e = math.floor(math.log10(abs(acc)))
            q = 10 ** (lim.sig_digits - 1 - e)
            acc = round(acc * q) / q
    rel = bound / max(1.0, abs(acc))
    conf = max(0.0, min(1.0, 1.0 - 2.0 * rel))
    return Prediction(str(int(round(acc))), conf,
                      f"误差界 {bound:.3g}，相对 {rel:.2%}")


def _parse_arith(prompt: str):
    body = prompt.split("：", 1)[1]
    toks = body.split()
    start = int(toks[0])
    ops = [(toks[i], int(toks[i + 1])) for i in range(1, len(toks), 2)]
    return start, ops


def solve_csp(task, lim: Limits) -> Prediction:
    """宽度受限的束搜索。置信度 = 自己的赋值满足了多少约束（不看真值）。"""
    n = task.meta["n"]
    cons = task.meta["constraints"]
    # 束搜索：按变量顺序赋值，保留累计满足约束数最高的 beam_width 条部分解
    beam = [([], set())]
    for _v in range(n):
        cand = []
        for assign, used in beam:
            for val in range(1, n + 1):
                if val in used:
                    continue
                cand.append((assign + [val], used | {val}))
        def score(st):
            assign, _u = st
            sc = 0
            for a, b, s in cons:
                if a < len(assign) and b < len(assign):
                    sc += 1 if ((assign[a] < assign[b]) if s == "<"
                                else (assign[a] > assign[b])) else 0
            return sc
        cand.sort(key=score, reverse=True)
        beam = cand[:lim.beam_width]
    best = max(beam, key=score)
    assign = best[0]
    sat = score(best)
    conf = sat / max(1, len(cons))
    return Prediction(",".join(str(x) for x in assign), conf,
                      f"满足 {sat}/{len(cons)} 条约束，束宽 {lim.beam_width}")


def _candidates(seq: list[int]):
    """候选模式族：每个返回下一项预测（只在可见项上拟合，不看真值）。"""
    out = {}
    n = len(seq)
    if n >= 2:                                   # 等差
        d0 = seq[1] - seq[0]
        if all(seq[i + 1] - seq[i] == d0 for i in range(n - 1)):
            out["arithmetic"] = seq[-1] + d0
    if n >= 2 and all(x != 0 for x in seq[:-1]):  # 等比
        r0 = seq[1] / seq[0] if seq[0] else None
        if r0 and all(abs(seq[i + 1] - seq[i] * r0) < 1e-9 for i in range(n - 1)):
            out["geometric"] = int(round(seq[-1] * r0))
    if n >= 3:                                   # 二阶差
        d1 = [seq[i + 1] - seq[i] for i in range(n - 1)]
        d2 = [d1[i + 1] - d1[i] for i in range(n - 2)]
        if len(set(d2)) == 1:
            out["quadratic"] = seq[-1] + d1[-1] + d2[-1]
    if n >= 4:                                   # 交替加减
        alt = [seq[i + 1] - seq[i] for i in range(n - 1)]
        if len(set(alt[::2])) == 1 and len(set(alt[1::2])) == 1:
            step = alt[0] if (n - 2) % 2 == 0 else alt[1]
            out["alternating"] = seq[-1] + step
    return out


def solve_seq(task, lim: Limits) -> Prediction:
    """有限观测下的模型选择。置信度 = 可见项上仍存活的候选模式数的倒数。"""
    full = task.meta["terms"]
    shown = full[:min(lim.shown_terms, task.meta["shown"])]
    cands = _candidates(shown)
    if not cands:
        return Prediction(str(shown[-1]), 0.2, "无候选模式存活，输出占位")
    # Occam：按固定优先级取最简模式
    for name in ("arithmetic", "geometric", "quadratic", "alternating"):
        if name in cands:
            conf = 1.0 / len(cands)
            return Prediction(str(cands[name]), conf,
                              f"存活 {len(cands)} 个候选 {sorted(cands)}，取 {name}")
    name = sorted(cands)[0]
    return Prediction(str(cands[name]), 1.0 / len(cands), f"取 {name}")


def solve_trace(task, lim: Limits) -> Prediction:
    """受限工作记忆执行。置信度 = 执行过程中成功读取的比例（不看真值）。"""
    import re
    slots: dict[str, int] = {}
    order: list[str] = []
    ok = tot = 0
    for line in task.meta["program"]:
        m = re.match(r"^(\w+)\s*=\s*(.+)$", line)
        if not m:
            continue
        tgt, rhs = m.group(1), m.group(2)
        toks = rhs.split()
        if len(toks) == 1:
            val = int(toks[0])
        else:
            a, op, b = toks[0], toks[1], toks[2]
            av, bv = slots.get(a, 0), slots.get(b, 0)
            tot += 2
            ok += (1 if a in slots else 0) + (1 if b in slots else 0)
            val = av + bv if op == "+" else av - bv
        if tgt not in slots and len(slots) >= lim.memory_slots:
            ev = order.pop(0)                     # LRU 挤掉：读它就等于读到 0
            slots.pop(ev, None)
        if tgt in order:
            order.remove(tgt)
        order.append(tgt)
        slots[tgt] = val
    conf = (ok / tot) if tot else 1.0
    q = task.meta["query"]
    if q not in slots:
        conf = min(conf, 0.1)
    return Prediction(str(slots.get(q, 0)), max(0.0, min(1.0, conf)),
                      f"读取成功率 {ok}/{tot}，记忆槽 {lim.memory_slots}")


SOLVERS = {"arith": solve_arith, "csp": solve_csp,
           "seq": solve_seq, "trace": solve_trace}


def run(task, lim: Limits) -> Prediction:
    return SOLVERS[task.family](task, lim)


# ------------------------------------------------------ 仪表校验策略
# 这些**故意使用真值**。它们不是求解器，而是用来界定各指标的取值空间，
# 从而证明"任何单一指标都可以被刷"。必须在报告里与真实求解器分开标注。
class ValidationPolicy:
    """基类：给定任务与"真实求解器的预测"和"它是否正确"，产出一个预测。"""

    name = "?"
    desc = ""

    def __call__(self, task, base: Prediction, correct: bool) -> Prediction:
        raise NotImplementedError


class ConstantSure(ValidationPolicy):
    """恒报高置信。校准极差，但分辨会随基础求解器准确率变化。"""
    name = "constant_one"
    desc = "恒报置信 1.0（答案仍用基础求解器）"

    def __call__(self, task, base, correct):
        return Prediction(base.answer, 1.0, "恒报 1.0")


class ConstantHalf(ValidationPolicy):
    """恒报 0.5。**校准近乎完美（ECE≈0）却毫无分辨力** —— 这正是
    参考实现里被评审点出的可刷漏洞，本基准把它作为一个显式的校验项。"""
    name = "constant_half"
    desc = "恒报置信 0.5（校准完美、分辨为零）"

    def __call__(self, task, base, correct):
        return Prediction(base.answer, 0.5, "恒报 0.5")


class OracleConf(ValidationPolicy):
    name = "oracle_conf"
    desc = "完美前瞻：动手前就知道自己会对不对（指标上界）"

    def __call__(self, task, base, correct):
        return Prediction(base.answer, 1.0 if correct else 0.0, "完美前瞻")


class InvertedConf(ValidationPolicy):
    name = "inverted_conf"
    desc = "反向前瞻：把把握度用反（应给出 AUROC < 0.5）"

    def __call__(self, task, base, correct):
        return Prediction(base.answer, 0.0 if correct else 1.0, "反向前瞻")


VALIDATION_POLICIES = [OracleConf(), ConstantSure(), ConstantHalf(), InvertedConf()]


def as_dict() -> dict:
    return {
        "limits": {"tight": TIGHT.__dict__, "ample": AMPLE.__dict__},
        "solvers": sorted(SOLVERS),
        "validation_policies": [{"name": p.name, "desc": p.desc}
                                for p in VALIDATION_POLICIES],
    }

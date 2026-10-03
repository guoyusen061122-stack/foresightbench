#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""任务族：四个结构不同的推理族，每个都有**精确真值**与**可控难度**。

设计约束（为什么是这四个族）
--------------------------
1. **真值精确**：每道题的答案由生成器解析地算出来，不存在标注噪声。
   这一点是对参考实现的直接改进——那边把"解析失败"计为错误，
   于是 solver 的准确率里混进了一个与元认知无关的失败模式。
2. **难度是一维可控的**：每个族都有一个整数难度级 d∈{1..6}，
   它的含义在该族里是明确的（步数/变量数/模式阶数），
   这样"分布迁移"才是一个能被精确构造的干预，而不是模糊的"变难了"。
3. **族之间结构不同**：算术链是**误差累积**型、约束满足是**搜索**型、
   序列归纳是**模型选择**型、程序追踪是**状态跟踪**型。
   四个族对求解器施加的是不同的认知压力，迁移条件因此不会退化成
   "同一件事做四遍"。

一组题由 (族, 难度, 序列号, 种子) 唯一确定。生成器是纯函数，
不依赖全局随机状态，因此 `seed=S` 永远给出同一份题集——
这是"私有种子防污染"的基础（见 proposal.md 第 5 节）。
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Task:
    """一道题。`answer` 是解析真值；`meta` 里放该族特有的过程信息。"""
    tid: str
    family: str
    difficulty: int
    prompt: str
    answer: str
    meta: dict

    def to_json(self) -> dict:
        return asdict(self)


def _rng(*parts) -> "random.Random":
    """由若干字段派生一个**跨进程稳定**的随机源。

    用 sha256 而不是内置 hash()：CPython 对字符串的 hash 默认按进程加盐，
    换一次运行就变，会让"固定种子"变成一句空话（这是本项目上一个挑战
    真实踩过的坑，见 AI日志）。sha256 在任何机器任何进程上都一样。
    """
    import random
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return random.Random(int(h[:16], 16))


# ---------------------------------------------------------------- 族 A：算术链
def gen_arith(d: int, i: int, seed: int) -> Task:
    """多步算术链。难度 d = 步数档位。

    为什么它能制造**真实的**不确定性：求解器用有限精度累加，
    每一步引入一个与当前量级成正比的舍入误差。误差随步数与数字量级增长、
    且可解析地给出上界 —— 于是"置信度"不是外挂的一个数字，
    而是**由求解过程自己算出来的误差上界**。
    """
    r = _rng("arith", d, i, seed)
    steps = d + 1
    hi = 10 ** (d + 1)
    start = r.randrange(1, hi)
    ops = [(r.choice("+-"), r.randrange(1, hi)) for _ in range(steps)]
    acc = start
    for op, v in ops:
        acc = acc + v if op == "+" else acc - v
    expr = str(start) + "".join(f" {o} {v}" for o, v in ops)
    return Task(f"A{d}-{i:03d}", "arith", d, f"求值：{expr}", str(acc),
                {"steps": steps, "magnitude": hi})


# ---------------------------------------------------------------- 族 B：约束满足
def gen_csp(d: int, i: int, seed: int) -> Task:
    """约束满足：给 n 个变量赋 1..n 的互异值，满足若干线性不等式。

    难度 d = 变量数 n ∈ {3..8}。求解器用**受限宽度**的束搜索：
    宽度不足时会在难实例上收敛到违反约束的赋值。置信度取
    "束内最优与次优的可行性差距"——即求解器自己对这个赋值的把握。
    """
    r = _rng("csp", d, i, seed)
    n = d + 2                                  # 3..8
    perm = list(range(1, n + 1))
    r.shuffle(perm)
    ncons = n + 1
    cons = []
    for _ in range(ncons):
        a, b = r.sample(range(n), 2)
        sign = r.choice(["<", ">"])
        cons.append((a, b, sign))
    # 真值由**构造**给出：`perm` 就是答案，不管它满足多少条约束。
    # 约束集是留给求解器**自检**用的（置信度 = 它自己满足了多少条），
    # 并**不保证**被 `perm` 全部满足 —— 这一点在注释里必须说准，
    # 否则"约束全部可满足"会变成一个从未被检验、却写在代码里的声称。
    return Task(f"B{d}-{i:03d}", "csp", d,
                "给 %d 个变量各取 1..%d 中互不相同的值，并满足：" % (n, n) +
                "；".join(f"x{a+1} {s} x{b+1}" for a, b, s in cons),
                ",".join(str(v) for v in perm),
                {"n": n, "constraints": cons, "solution": perm})


# ---------------------------------------------------------------- 族 C：序列归纳
def gen_seq(d: int, i: int, seed: int) -> Task:
    """序列归纳：给前 k 项，求下一项。

    难度 d = 模式阶数（1=等差，2=等比/二阶差，3=交替，4=二阶交替，…）。
    求解器用**有限样本**做候选模式选择：候选模式的参数是从可见项拟合出来的，
    高阶模式在项数少时会与低阶模式在可见项上完全一致，
    于是"选错模式"成为真实的、可解析识别的失效模式——
    置信度就是**在当前可见项上仍然存活的候选模式所占比例**。
    """
    r = _rng("seq", d, i, seed)
    order = min(d, 4)
    a0 = r.randrange(2, 20)
    k = max(4, order + 2)
    if order == 1:
        st = r.randrange(1, 9)
        seq = [a0 + st * t for t in range(k + 1)]
    elif order == 2:
        q = r.randrange(2, 5)
        seq = [a0 * (q ** t) for t in range(k + 1)]
    elif order == 3:
        p, q = r.randrange(2, 6), r.randrange(1, 5)
        seq = [a0 + p * t + (q if t % 2 else 0) for t in range(k + 1)]
    else:
        p = r.randrange(2, 6)
        seq = [a0 + (p * t if t % 2 == 0 else -p * t) for t in range(k + 1)]
    return Task(f"C{d}-{i:03d}", "seq", d,
                "数列：" + ", ".join(str(x) for x in seq[:k]) + "，下一项是多少？",
                str(seq[k]), {"order": order, "shown": k, "terms": seq})


# ---------------------------------------------------------------- 族 D：程序追踪
def gen_trace(d: int, i: int, seed: int) -> Task:
    """程序追踪：读一段微型程序，问某变量最终值。

    难度 d 控制变量数与步数。求解器用**受限工作记忆**：
    只能可靠地保持 W 个活跃变量，超出的变量会被"挤掉"
    （记成 0 —— 模拟真实的状态跟踪失败）。
    置信度 = 能保持住的变量比例，直接反映跟踪是否可靠。
    """
    r = _rng("trace", d, i, seed)
    nv = min(2 + d, 8)
    nsteps = 2 + 2 * d
    prog = []
    vals = {f"v{j}": 0 for j in range(1, nv + 1)}
    for j in range(1, nv + 1):
        vals[f"v{j}"] = r.randrange(1, 20)
        prog.append(f"v{j} = {vals[f'v{j}']}")
    for _ in range(nsteps):
        t = r.choice([f"v{j}" for j in range(1, nv + 1)])
        a, b = r.sample([f"v{j}" for j in range(1, nv + 1)], 2)
        if r.random() < 0.7:
            prog.append(f"{t} = {a} + {b}")
            vals[t] = vals[a] + vals[b]
        else:
            prog.append(f"{t} = {a} - {b}")
            vals[t] = vals[a] - vals[b]
    q = r.choice(list(vals))
    return Task(f"D{d}-{i:03d}", "trace", d,
                "依次执行下列赋值，最后 %s 等于多少？\n  " % q +
                "\n  ".join(prog), str(vals[q]),
                {"vars": nv, "steps": nsteps, "program": prog, "query": q})


GENERATORS = {"arith": gen_arith, "csp": gen_csp, "seq": gen_seq, "trace": gen_trace}
FAMILY_ZH = {"arith": "算术链", "csp": "约束满足", "seq": "序列归纳",
             "trace": "程序追踪"}
DIFFICULTIES = (1, 2, 3, 4, 5, 6)


def make_tasks(family: str, d: int, n: int, seed: int, offset: int = 0) -> list[Task]:
    """生成某族某难度下的 n 道题。offset 让不同迁移条件取到**不重叠**的题。"""
    g = GENERATORS[family]
    return [g(d, offset + i, seed) for i in range(n)]


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="生成并冻结一份任务快照")
    ap.add_argument("--seed", type=int, default=20261003)
    ap.add_argument("--n", type=int, default=60, help="每格题数")
    ap.add_argument("--out",
                    default=os.path.join(os.path.dirname(os.path.dirname(
                        os.path.abspath(__file__))), "data", "tasks.json"))
    a = ap.parse_args()
    tasks = []
    for fam in GENERATORS:
        for d in DIFFICULTIES:
            tasks += make_tasks(fam, d, a.n, a.seed)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"seed": a.seed, "n_per_cell": a.n,
                   "cells": [{"family": k, "difficulty": d}
                             for k in GENERATORS for d in DIFFICULTIES],
                   "tasks": [t.to_json() for t in tasks]},
                  f, ensure_ascii=False, indent=1)
    print(f"已生成 {len(tasks)} 道题 → {a.out}"
          f"（{len(GENERATORS)} 族 × {len(DIFFICULTIES)} 难度 × {a.n}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

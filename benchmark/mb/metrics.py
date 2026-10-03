#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""指标层：把"前瞻性元认知"拆成四个**可分别报告**的量。

为什么不能只报一个数
--------------------
评测里最常见的一种可被"刷"的设计是：只报校准（ECE / Brier）。
一个**恒报 0.5** 的预测器在二分类上 ECE≈0、看起来校准完美，
但它对"这次会不会成"没有任何信息量。反过来，一个**恒报 1.0** 的预测器
分辨力并不差（若基础准确率不低），校准却极差。

因此本基准把构念拆成四个量，并规定**必须同时报告**：

1. `ECE` / `reliability`：**校准** —— 置信度与实际正确率的偏差。
2. `AUROC`：**分辨** —— 置信度能否把"将要对"与"将要错"分开。
   **对校准不敏感**，因此不能被"整体压低/抬高置信度"刷掉。
   全部置信度相同（完全无信息）时按惯例取 0.5。
3. `resolution`：**分辨率** —— Brier 分解中"置信度把样本分到不同正确率的
   组里"贡献的那一项。它与 reliability、uncertainty 一起精确分解 Brier。
4. `utility`：**行为控制** —— 在**非对称代价**下，按置信度决定
   "作答 / 弃权"所能拿到的净收益，并对照完美前瞻（oracle）与永不弃权，
   给出**效用捕获率**。这一项才真正回答"元认知有没有用"。

统计口径
--------
所有比例都附 **Wilson 95% 区间**；AUROC 与 Brier 附**百分位自助区间**
（固定随机种子，可复现）。样本量对区分度的影响在第 6 节的功效扫描里单独报告。
"""
from __future__ import annotations

import math
import random

BINS = 10


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 区间。小样本下正态近似会给出大于 1 的上界，Wilson 不会。"""
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - h) / d), min(1.0, (c + h) / d)


def ece(confs: list[float], correct: list[bool], bins: int = BINS) -> float:
    """期望校准误差：等宽分箱下 |平均置信 − 实际正确率| 的加权和。"""
    n = len(confs)
    if n == 0:
        return 0.0
    tot = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confs)
               if (lo <= c < hi) or (b == bins - 1 and c == 1.0)]
        if not idx:
            continue
        mc = sum(confs[i] for i in idx) / len(idx)
        ma = sum(1 for i in idx if correct[i]) / len(idx)
        tot += len(idx) / n * abs(mc - ma)
    return tot


def _groups(confs: list[float], bins: int):
    """等宽分箱（最后一箱含 1.0）。返回 [(下标列表), ...]。"""
    out = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        out.append([i for i, c in enumerate(confs)
                    if (lo <= c < hi) or (b == bins - 1 and c == 1.0)])
    return out


def brier_decompose(confs: list[float], correct: list[bool],
                    bins: int = BINS) -> dict:
    """Murphy 分解：Brier = reliability − resolution + uncertainty。

    三项都必须报：只报 Brier 会把"校准差"与"分辨差"混在一个数里。

    **这里有一个必须说清楚的细节。** 教科书上的三项恒等式
    `Brier = REL − RES + UNC` 只在"同一组内置信度完全相等"时才精确成立。
    一旦用**等宽分箱**把取值不同的置信度塞进同一箱，就多出一项组内离散项

        δ = (1/N) Σ_k [ Σ_{i∈k}(p_i − p̄_k)² + 2 Σ_{i∈k}(p_i − p̄_k)(ȳ_k − y_i) ]

    于是精确关系是 `Brier = REL_bin − RES_bin + UNC + δ`。

    很多论文只报 10 箱分解、对 δ≠0 只字不提。本实现两种口径都报：

    · `reliability_exact` / `resolution_exact`：按**置信度取值**分组
      （组内置信度恒等 → 恒等式精确成立，误差 < 1e-12），用于对账自检；
    · `reliability` / `resolution`：10 等宽分箱（文献口径，便于和 ECE 对照），
      并显式返回组内离散项 `within_bin_spread` 以及实测残差 `bin_residual`。

    两者之差必须恰等于 `within_bin_spread` —— 这条恒等式本身就是自检：
    如果分解写错了，`bin_residual` 与 `within_bin_spread` 不会相等。
    """
    n = len(confs)
    if n == 0:
        return {"brier": 0.0, "reliability": 0.0, "resolution": 0.0,
                "uncertainty": 0.0, "brier_from_parts": 0.0,
                "reliability_exact": 0.0, "resolution_exact": 0.0,
                "within_bin_spread": 0.0, "bin_residual": 0.0,
                "base_rate": 0.0, "n_groups": 0}
    base = sum(1 for c in correct if c) / n
    brier = sum((confs[i] - (1.0 if correct[i] else 0.0)) ** 2
                for i in range(n)) / n

    # (1) 精确分组分解：组 = 一个置信度取值。
    groups: dict[float, list[int]] = {}
    for i, c in enumerate(confs):
        groups.setdefault(c, []).append(i)
    rel_e = res_e = 0.0
    for p, idx in groups.items():
        w = len(idx) / n
        ma = sum(1 for i in idx if correct[i]) / len(idx)
        rel_e += w * (p - ma) ** 2          # 组内 p 恒等 → 无组内离散项
        res_e += w * (ma - base) ** 2

    # (2) 等宽分箱分解 + 显式组内离散项。
    rel_b = res_b = within = 0.0
    for idx in _groups(confs, bins):
        if not idx:
            continue
        w = len(idx) / n
        mc = sum(confs[i] for i in idx) / len(idx)
        ma = sum(1 for i in idx if correct[i]) / len(idx)
        rel_b += w * (mc - ma) ** 2
        res_b += w * (ma - base) ** 2
        a = sum((confs[i] - mc) ** 2 for i in idx)
        c = 2.0 * sum((confs[i] - mc) * (ma - (1.0 if correct[i] else 0.0))
                      for i in idx)
        within += (a + c) / n
    unc = base * (1 - base)
    return {"brier": brier,
            "reliability": rel_b, "resolution": res_b, "uncertainty": unc,
            "bin_residual": brier - (rel_b - res_b + unc),
            "within_bin_spread": within,
            "reliability_exact": rel_e, "resolution_exact": res_e,
            "brier_from_parts": rel_e - res_e + unc,
            "base_rate": base, "n_groups": len(groups)}


def auroc(confs: list[float], correct: list[bool]):
    """AUROC（Mann–Whitney 形式，**正确处理并列**）。

    返回 `None` 表示**未定义**：当样本里只有一类（全对或全错）时，
    "把对与错分开"这件事没有定义。
    这里**不**按惯例回落到 0.5 —— 那样会让"求解器全错"这种退化情形
    看起来像"零信息但有效"，是个会误导读数的地方。
    全部置信度相同（真正零信息）时才回落到 0.5，因为那是有定义的：
    并列 → 平均秩 → AUROC 恰好 0.5。

    并列处理很关键：一个恒报同一置信度的预测器等于零信息，
    AUROC 必须回落到 0.5，而不是被并列规则算成 1.0 或 0.0。
    """
    pos = [c for c, ok in zip(confs, correct) if ok]
    neg = [c for c, ok in zip(confs, correct) if not ok]
    if not pos or not neg:
        return None                     # 单类：未定义，如实返回 None
    allv = sorted(confs)
    rank_of = {}
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        rank_of[allv[i]] = (i + j) / 2 + 1
        i = j + 1
    r_sum = sum(rank_of[c] for c in pos)
    n1, n0 = len(pos), len(neg)
    return (r_sum - n1 * (n1 + 1) / 2) / (n1 * n0)


def risk_coverage(confs: list[float], correct: list[bool]) -> list[tuple]:
    """风险–覆盖曲线：按置信度从高到低作答，横轴覆盖比例、纵轴错误率。"""
    idx = sorted(range(len(confs)), key=lambda i: -confs[i])
    out, err = [], 0
    for k, i in enumerate(idx, 1):
        err += 0 if correct[i] else 1
        out.append((k / len(idx), err / k))
    return out


def aurc(confs: list[float], correct: list[bool]) -> float:
    """风险–覆盖曲线下的面积（越小越好）。"""
    rc = risk_coverage(confs, correct)
    if not rc:
        return 0.0
    return sum(r for _c, r in rc) / len(rc)


def utility(confs: list[float], correct: list[bool], tau: float = 0.5,
            cost_wrong: float = 1.0, cost_abstain: float = 0.35) -> dict:
    """非对称代价下的"作答/弃权"净收益，并给出**效用捕获率**。

    规则：置信度 ≥ τ 就作答，否则弃权。
      · 作答且错：代价 cost_wrong
      · 作答且对：代价 0
      · 弃权：代价 cost_abstain（固定，与对错无关）
    归一化后与两个极端对照：
      · 永不弃权（U_never）与全部弃权（U_all）
      · **完美前瞻**（U_oracle：只在对时作答、错时弃权）
    效用捕获率 = (U − U_never) / (U_oracle − U_never)（截断到 [0,1]）。
    这一项回答"这份元认知到底值多少钱"，是四个量里唯一带行为后果的。
    """
    n = len(confs)
    if n == 0:
        return {"utility": 0.0, "capture": 0.0, "tau": tau}
    def u(decision):
        c = 0.0
        for i in range(n):
            if decision(i) == "answer":
                c += 0.0 if correct[i] else cost_wrong
            else:
                c += cost_abstain
        return 1.0 - c / (n * max(cost_wrong, 1e-9))
    u_never = u(lambda i: "answer")
    u_all = u(lambda i: "abstain")
    u_oracle = u(lambda i: "answer" if correct[i] else "abstain")
    u_here = u(lambda i: "answer" if confs[i] >= tau else "abstain")
    span = u_oracle - u_never
    cap = (u_here - u_never) / span if abs(span) > 1e-9 else 0.0
    # **改进版捕获率**：基准不用"永不弃权"，而用 {永不弃权, 全部弃权} 里更好的那个。
    #
    # 为什么必须改：实测发现 constant_half（恒报 0.5、零信息）在低准确率条件下
    # 拿到 0.905 的"捕获率"，反而高于真正有分辨力的求解器（0.627）。
    # 原因不是它有用，而是当准确率远低于 50% 时"干脆全部弃权"本身就是好策略，
    # 而"以永不弃权为基准"把这份便宜算成了元认知的功劳。
    # 改基准之后，任何恒定置信度策略的捕获率都被压到 0 —— 因为它无法区分
    # 该答与不该答的题。这才是这一项想测的东西。
    base = max(u_never, u_all)
    span2 = u_oracle - base
    cap2 = (u_here - base) / span2 if abs(span2) > 1e-9 else 0.0
    return {"utility": u_here, "capture": max(0.0, min(1.0, cap)),
            "capture_vs_best_constant": max(0.0, min(1.0, cap2)),
            "u_never": u_never, "u_all": u_all, "u_oracle": u_oracle, "tau": tau}


def bootstrap_ci(fn, a: list, b: list, reps: int = 2000,
                 seed: int = 20261003) -> tuple[float, float]:
    """百分位自助区间。固定种子 → 可复现。"""
    n = len(a)
    if n < 8:
        return 0.0, 1.0
    r = random.Random(seed)
    vals = []
    for _ in range(reps):
        idx = [r.randrange(n) for _ in range(n)]
        vals.append(fn([a[i] for i in idx], [b[i] for i in idx]))
    vals.sort()
    return vals[int(0.025 * reps)], vals[int(0.975 * reps) - 1]


def summarize(confs: list[float], correct: list[bool],
              tau: float = 0.5) -> dict:
    """把一组 (置信度, 是否答对) 汇总成四个量 + 区间。

    注意 `auroc`（以及它的自助区间）在**单类样本**下为 `None` —— 那是
    "分辨力未定义"，不是"分辨力等于 0.5"。报告时必须区分这两者。
    """
    n = len(confs)
    k = sum(1 for c in correct if c)
    lo, hi = wilson(k, n)
    bd = brier_decompose(confs, correct)
    a = auroc(confs, correct)
    if a is None:
        a_ci = None
    else:
        a_ci = list(bootstrap_ci(lambda c, t: auroc(c, t) or 0.5, confs, correct))
    return {
        "n": n,
        "accuracy": k / n if n else 0.0,
        "accuracy_ci": [lo, hi],
        "ece": ece(confs, correct),
        "auroc": a,
        "auroc_ci": a_ci,
        "auroc_defined": a is not None,
        "brier": bd["brier"],
        "brier_ci": list(bootstrap_ci(
            lambda c, t: brier_decompose(c, t)["brier"], confs, correct)),
        "reliability": bd["reliability"],
        "resolution": bd["resolution"],
        "reliability_exact": bd["reliability_exact"],
        "resolution_exact": bd["resolution_exact"],
        "within_bin_spread": bd["within_bin_spread"],
        "bin_residual": bd["bin_residual"],
        "n_groups": bd["n_groups"],
        "uncertainty": bd["uncertainty"],
        "brier_from_parts": bd["brier_from_parts"],
        "aurc": aurc(confs, correct),
        "utility": utility(confs, correct, tau),
    }

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 results.json 变成人可读的报告（markdown 表）。

报告口径上的一条硬规矩：**校准与分辨必须并排出现**。
本项目的主要方法结论就是"任何单一指标都能被刷"，因此报告格式本身
必须强制把这两项放在一起，否则读表的人会重犯这个错。
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

POLICY_ZH = {
    "oracle_conf": "完美前瞻（上界）",
    "constant_one": "恒报 1.0",
    "constant_half": "恒报 0.5",
    "inverted_conf": "反向前瞻（下界）",
}


def f(x, nd=3):
    return "未定义" if x is None else f"{x:.{nd}f}"


def pct(x):
    return "—" if x is None else f"{x*100:.1f}%"


def main_with_dir(res_dir: str) -> None:
    """按给定结果目录生成 REPORT.md（供 run.py 在同一次运行里调用）。"""
    _build(res_dir)


def main() -> int:
    # 默认目录锚在**本文件**上，不锚在当前工作目录（见 runner.BENCH_DIR 的说明）。
    default_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    res_dir = sys.argv[1] if len(sys.argv) > 1 else default_dir
    _build(res_dir)
    return 0


def _build(res_dir: str) -> None:
    d = json.load(open(os.path.join(res_dir, "results.json"), encoding="utf-8"))
    L = []
    L.append("# ForesightBench 结果报告")
    L.append("")
    L.append(f"- 随机种子：`{d['seed']}`；每格题数：`{d['n_per_cell']}`")
    L.append(f"- 校准集冻结的阈值 τ*："
             + "、".join(f"`{k}` = {v:.2f}" for k, v in d["tau_frozen"].items()))
    L.append("- **迁移后不重新校准**：τ* 只在校准集上选一次，全部迁移条件共用。")
    L.append("")

    # ---- 表 1：只看真实求解器（不含校验策略）----
    L.append("## 表 1 · 真实求解器在六个迁移条件下的四个量")
    L.append("")
    L.append("| 条件 | 求解器 | n | 准确率 | 校准 ECE↓ | 分辨 AUROC↑ | "
             "分辨率↑ | Brier↓ | 效用捕获↑ |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for cname, block in d["conditions"].items():
        for k, v in block["per_solver"].items():
            if "::" in k:
                continue
            L.append(f"| {cname} | `{k}` | {v['n']} | {f(v['accuracy'])} | "
                     f"{f(v['ece'])} | {f(v['auroc'])} | "
                     f"{f(v['resolution'], 4)} | {f(v['brier'])} | "
                     f"{f(v['utility']['capture_vs_best_constant'])} |")
    L.append("")
    L.append("> 上表的「分辨率」是 **10 等宽分箱**口径（与 ECE 同箱，便于对照）；"
             "按置信度取值精确分组的版本见第 4 节，两者差值有明确定义、不是误差。"
             "「效用捕获」用的是改进版基准 `capture_vs_best_constant`"
             "（详见第 5 节说明），它把「干脆全部弃权」这份便宜排除在外。")
    L.append("")

    # ---- 表 2：仪表校验策略（证明单一指标可被刷）----
    L.append("## 表 2 · 仪表校验：单一指标可被刷，组合不可")
    L.append("")
    L.append("这些策略**故意使用真值**，不是求解器。它们的作用是把各指标的"
             "取值空间撑开，看哪个指标会被「什么都不做」的策略骗过。")
    L.append("")
    L.append("| 策略 | 准确率 | 校准 ECE↓ | 分辨 AUROC↑ | 分辨率↑ | 结论 |")
    L.append("|---|---:|---:|---:|---:|---|")
    # 汇总每个策略在**六个条件**上的 ECE 取值范围。
    # 只报一个条件会低估这件事的严重性：恒报 0.5 在基准准确率恰为 50%
    # 的那些条件里 ECE 是**精确的 0**，也就是说它在相当一部分条件下
    # 会拿到全场最好的校准分，而它的 AUROC 恒为 0.5（零信息）。
    agg = {}
    for cname, block in d["conditions"].items():
        for k, v in block["per_solver"].items():
            if "::" not in k:
                continue
            pol = k.split("::", 1)[1]
            a = agg.setdefault(pol, {"ece": [], "auroc": [], "res": [],
                                     "acc": []})
            a["ece"].append(v["ece"])
            if v["auroc"] is not None:
                a["auroc"].append(v["auroc"])
            a["res"].append(v["resolution"])
            a["acc"].append(v["accuracy"])
    for pol, a in agg.items():
        verdict = {
            "constant_half": "**在某些条件下 ECE 恰为 0（全场最好），"
                             "而 AUROC 恒 0.5、分辨率恒 0** —— 只看校准必被刷",
            "constant_one": "校准极差、分辨为零 —— 两样都不行",
            "oracle_conf": "上界：校准与分辨同时完美",
            "inverted_conf": "下界：分辨比随机还差（AUROC=0）",
        }.get(pol, "")
        ece_lo, ece_hi = min(a["ece"]), max(a["ece"])
        L.append(
            f"| {POLICY_ZH.get(pol, pol)} | {f(min(a['acc']))}–{f(max(a['acc']))} | "
            f"{f(ece_lo)}–{f(ece_hi)} | "
            f"{f(min(a['auroc']))}–{f(max(a['auroc']))} | "
            f"{f(min(a['res']), 4)}–{f(max(a['res']), 4)} | {verdict} |")
    L.append("")
    L.append("> 表 2 的每一行都是**同一批答案**配不同的置信度报法，"
             "因此准确率相同、只有四个量在变。这样才能看清："
             "**校准与分辨是两个独立的维度，只报其中一个一定会被策略刷掉。**")
    L.append("")

    # ---- 表 3：功效扫描 ----
    pp = os.path.join(res_dir, "power_scan.json")
    if os.path.exists(pp):
        rows = json.load(open(pp, encoding="utf-8"))
        L.append("## 表 3 · 样本量功效扫描：要多大 n 才能区分两档求解器")
        L.append("")
        L.append("| 每格题数 | 总题数 | tight AUROC | ample AUROC | Δ | "
                 "95% 区间分离 |")
        L.append("|---:|---:|---:|---:|---:|---|")
        for r in rows:
            L.append(f"| {r['n_per_cell']} | {r['total']} | "
                     f"{f(r['tight']['auroc'])} | {f(r['ample']['auroc'])} | "
                     f"{f(r['delta_auroc'])} | "
                     f"{'是' if r['ci_disjoint'] else '**否**'} |")
        L.append("")
        first = next((r["n_per_cell"] for r in rows if r["ci_disjoint"]), None)
        if first:
            L.append(f"> **结论：每格 ≥{first} 题**时该对比的置信区间才分离。"
                     f"本基准默认取 60，留有余量。")
            L.append("")

    # ---- 表 4：Brier 分解对账 ----
    L.append("## 表 4 · Brier 分解对账（自检）")
    L.append("")
    L.append("教科书恒等式 `Brier = 可靠性 − 分辨率 + 不确定性` **只在「同一组内"
             "置信度完全相等」时精确成立**。用等宽分箱把取值不同的置信度塞进"
             "同一箱，会多出一项组内离散项 δ，精确关系是 "
             "`Brier = REL − RES + UNC + δ`。")
    L.append("")
    L.append("所以下表报**两种口径**：")
    L.append("")
    L.append("- `精确重算`：按置信度**取值**分组（组内恒等 → 无 δ），"
             "必须与 Brier 逐位相符；")
    L.append("- `10箱残差 δ`：等宽分箱口径下的残差，必须**恰等于**独立按定义算出的"
             "组内离散项。这两者相等就是分解写对了的证据 —— 不是把误差藏起来。")
    L.append("")
    L.append("| 条件 | 求解器 | Brier | 精确重算 | 差 | 分组数 | "
             "10箱残差 δ | 组内离散项 | 一致 |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    worst = 0.0
    worst2 = 0.0
    for cname, block in d["conditions"].items():
        for k, v in block["per_solver"].items():
            if "::" in k:
                continue
            diff = abs(v["brier"] - v["brier_from_parts"])
            gap = abs(v["bin_residual"] - v["within_bin_spread"])
            worst = max(worst, diff)
            worst2 = max(worst2, gap)
            L.append(f"| {cname} | `{k}` | {f(v['brier'], 6)} | "
                     f"{f(v['brier_from_parts'], 6)} | {diff:.2e} | "
                     f"{v['n_groups']} | {v['bin_residual']:+.2e} | "
                     f"{v['within_bin_spread']:+.2e} | {gap:.2e} |")
    L.append("")
    L.append(f"> 精确恒等式最大误差 **{worst:.2e}**"
             + ("（通过）" if worst < 1e-9 else "（**未通过，分解有误**）")
             + f"；两口径组内离散项一致性最大偏差 **{worst2:.2e}**"
             + ("（通过）" if worst2 < 1e-9 else "（**未通过**）"))
    L.append("")

    # ---- 附：效用捕获率的基准问题（实测证据）----
    L.append("## 附 · 效用捕获率为什么要换基准（实测证据）")
    L.append("")
    L.append("原始定义 `捕获率 = (U − U_never) / (U_oracle − U_never)` 以"
             "「永不弃权」为基准。实测发现这个基准**会把便宜算成能力**："
             "当准确率远低于 50% 时，「干脆全部弃权」本身就是好策略，"
             "于是恒报同一置信度的零信息策略也能拿到高分。")
    L.append("")
    L.append("| 策略 | 原捕获率范围 | 改基准后范围 |")
    L.append("|---|---:|---:|")
    agg2 = {}
    for _cname, block in d["conditions"].items():
        for k, v in block["per_solver"].items():
            if "::" not in k:
                continue
            pol = k.split("::", 1)[1]
            a = agg2.setdefault(pol, {"c": [], "c2": []})
            a["c"].append(v["utility"]["capture"])
            a["c2"].append(v["utility"]["capture_vs_best_constant"])
    for pol, a in agg2.items():
        L.append(f"| {POLICY_ZH.get(pol, pol)} | "
                 f"{f(min(a['c']))}–{f(max(a['c']))} | "
                 f"{f(min(a['c2']))}–{f(max(a['c2']))} |")
    L.append("")
    L.append("> 改基准后，**任何恒定置信度策略的捕获率都被压到 0** —— "
             "因为它无法区分「该答」与「不该答」的题。这才是这一项想测的东西。")
    L.append("")

    out = os.path.join(res_dir, "REPORT.md")
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"已写出 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

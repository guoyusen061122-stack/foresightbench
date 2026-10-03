#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""编排：构造分布迁移条件，跑求解器，汇总四个量。

迁移条件的设计原则
------------------
**校准集与测试集严格分离，且迁移后不重新校准。**
"前瞻性元认知"的定义就是：动手之前就对自己会不会成有个判断。
如果允许在看到测试集之后再调阈值，那测的就不是前瞻性，而是**事后拟合**。
因此：

1. 阈值 τ* 只在**校准集**上按"效用最大"选一次，然后**冻结**，用于全部迁移条件；
2. 迁移条件之间、以及与校准集之间，题目实例**不重叠**（靠 offset 保证）。

六个条件覆盖三类迁移：
  · 难度迁移：更易 / 更难 / 双峰 / 极难
  · 同分布对照：S1 同分布（用来确认测量本身没有漂移）
  · 域迁移：S6 在 {算术链, 约束满足} 上校准，测 {序列归纳, 程序追踪}
"""
from __future__ import annotations

import json
import os

from . import metrics, solvers
from .solvers import AMPLE, TIGHT, Prediction
from .tasks import FAMILY_ZH, make_tasks

# (条件名, 说明, 参与族, 难度集合)
# 结果目录一律相对**本文件**解析，不相对当前工作目录。
# 为什么：如果写成 "results"，从仓库根目录跑就会写到 ./results，
# 而测试与报告读的是 benchmark/results —— 于是"跑过了但报告没更新"，
# 这正是 AI日志 案例 J 那一类静默不一致。路径必须锚在代码位置上。
BENCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONDITIONS = [
    ("S1_same", "同分布（校准集本身，作为对照）", ("arith", "csp", "seq", "trace"),
     (2, 3, 4)),
    ("S2_easier", "更易：难度整体下移一档", ("arith", "csp", "seq", "trace"),
     (1, 2, 3)),
    ("S3_harder", "更难：难度整体上移一档", ("arith", "csp", "seq", "trace"),
     (4, 5, 6)),
    ("S4_bimodal", "双峰：极易与极难混合（检验校准是否非单调退化）",
     ("arith", "csp", "seq", "trace"), (1, 2, 6)),
    ("S5_hardest", "极难：只取最高难度", ("arith", "csp", "seq", "trace"), (6,)),
    ("S6_crossfam", "域迁移：在算术链/约束满足上校准，测序列归纳/程序追踪",
     ("seq", "trace"), (3, 4)),
]
CALIB = CONDITIONS[0]

# 每一档资源限制对应一个"求解器档位"；校验策略复用 ample 的答案，
# 只改**置信度的报法**，从而把"能力"与"自我评估"分离。
REAL_SOLVERS = [("tight", TIGHT), ("ample", AMPLE)]


def _build(families, diffs, n, seed, offset):
    tasks = []
    for fam in families:
        for d in diffs:
            tasks += make_tasks(fam, d, n, seed, offset)
    return tasks


def _run_solver(tasks, lim):
    """返回 (预测列表, 是否答对列表)。真值只在这里用于判定对错，
    **不进入求解过程**（求解器拿不到 task.answer）。"""
    preds, ok = [], []
    for t in tasks:
        p = solvers.run(t, lim)
        preds.append(p)
        ok.append(p.answer.strip() == t.answer.strip())
    return preds, ok


def _conf_of(preds):
    return [max(0.0, min(1.0, p.confidence)) for p in preds]


def pick_tau(preds, ok, grid=None) -> float:
    """在**校准集**上选让效用最大的阈值 τ*。选完就冻结。"""
    grid = grid or [round(0.05 * i, 2) for i in range(1, 20)]
    confs = _conf_of(preds)
    best, best_u = 0.5, -9e9
    for tau in grid:
        u = metrics.utility(confs, ok, tau)["utility"]
        if u > best_u:
            best, best_u = tau, u
    return best


def evaluate(conditions=None, n_per_cell=60, seed=20261003, verbose=True):
    conditions = conditions or CONDITIONS
    out = {"seed": seed, "n_per_cell": n_per_cell,
           "conditions": {}, "tau_frozen": {}, "solver_meta": solvers.as_dict()}

    # ---------- 校准集（S1），实例偏移 0 ----------
    cal_tasks = _build(CALIB[2], CALIB[3], n_per_cell, seed, 0)
    cal_by_solver = {}
    for sname, lim in REAL_SOLVERS:
        preds, ok = _run_solver(cal_tasks, lim)
        tau = pick_tau(preds, ok)
        cal_by_solver[sname] = (preds, ok, tau)
        out["tau_frozen"][sname] = tau
        if verbose:
            m = metrics.summarize(_conf_of(preds), ok, tau)
            print(f"[校准集] {sname:<6} n={m['n']:<5} acc={m['accuracy']:.3f} "
                  f"ECE={m['ece']:.3f} AUROC={m['auroc']:.3f} τ*={tau:.2f}")

    # ---------- 各迁移条件 ----------
    for ci, (cname, cdesc, fams, diffs) in enumerate(conditions):
        offset = 10000 * (ci + 1)                 # 与校准集、与其他条件都不重叠
        tasks = _build(fams, diffs, n_per_cell, seed, offset)
        block = {"desc": cdesc, "families": list(fams), "difficulties": list(diffs),
                 "n_tasks": len(tasks), "per_solver": {}}
        for sname, lim in REAL_SOLVERS:
            preds, ok = _run_solver(tasks, lim)
            confs = _conf_of(preds)
            tau = out["tau_frozen"][sname]        # **冻结**，不在测试集上重选
            block["per_solver"][sname] = metrics.summarize(confs, ok, tau)
            # 按族拆分：单一汇总会掩盖"某个族系统性失效"这种问题
            # （参考实现的评审意见里就点了这一条）。
            per_fam = {}
            for fam in sorted({t.family for t in tasks}):
                sel = [i for i, t in enumerate(tasks) if t.family == fam]
                if sel:
                    per_fam[fam] = metrics.summarize(
                        [confs[i] for i in sel], [ok[i] for i in sel], tau)
            block["per_solver"][sname]["per_family"] = per_fam
            # 仪表校验策略：复用同一批答案，只改置信度报法。
            # **键必须带求解器前缀**：否则第二轮（ample）会把第一轮（tight）的
            # 校验策略结果覆盖掉，报告里出现的其实是 ample 的数字——
            # 这正是本轮实测撞到的 bug（见 AI日志 案例 K）。
            for pol in solvers.VALIDATION_POLICIES:
                pc, pok = [], []
                for t, p, k in zip(tasks, preds, ok):
                    q = pol(t, p, k)
                    pc.append(q.confidence)
                    pok.append(q.answer.strip() == t.answer.strip())
                block["per_solver"][f"{sname}::{pol.name}"] = \
                    metrics.summarize(pc, pok, tau)
        out["conditions"][cname] = block
        if verbose:
            print(f"\n[{cname}] {cdesc}  n={len(tasks)}")
            for k, v in block["per_solver"].items():
                f = lambda x: "未定义" if x is None else f"{x:.3f}"  # noqa: E731
                print(f"   {k:<24} acc={v['accuracy']:.3f} ECE={v['ece']:.3f} "
                      f"AUROC={f(v['auroc'])} res={v['resolution']:.4f} "
                      f"效用捕获={v['utility']['capture']:.3f}")
    return out


def power_scan(ns=(20, 40, 60, 120, 240), seed=20261003, verbose=True):
    """样本量功效扫描：区分"紧"与"宽"两档需要多大的 n。

    参考实现的评审意见里点名"Meta-Q 差距偏小，需扩大每格样本以提升统计功效"。
    与其只把 n 调大，不如**把需要多大的 n 报出来**——这样读者知道
    结论在什么样本量下才站得住，也让后续复现者能自行判断。
    """
    rows = []
    for n in ns:
        tasks = _build(CONDITIONS[1][2], CONDITIONS[1][3], n, seed, 777)
        got = {}
        for sname, lim in REAL_SOLVERS:
            preds, ok = _run_solver(tasks, lim)
            confs = _conf_of(preds)
            a = metrics.auroc(confs, ok)
            if a is None:                      # 单类：分辨力未定义
                got[sname] = {"auroc": None, "ci": None}
            else:
                lo, hi = metrics.bootstrap_ci(
                    lambda c, t: metrics.auroc(c, t) or 0.5, confs, ok)
                got[sname] = {"auroc": a, "ci": [lo, hi]}
        if got["ample"]["auroc"] is None or got["tight"]["auroc"] is None:
            d, sep = None, False
        else:
            d = got["ample"]["auroc"] - got["tight"]["auroc"]
            sep = (got["ample"]["ci"][0] > got["tight"]["ci"][1] or
                   got["tight"]["ci"][0] > got["ample"]["ci"][1])
        rows.append({"n_per_cell": n, "total": len(tasks), **got,
                     "delta_auroc": d, "ci_disjoint": bool(sep)})
        if verbose:
            f = lambda x: "未定义" if x is None else f"{x:.3f}"  # noqa: E731
            print(f"n/格={n:<4} 总={len(tasks):<6} "
                  f"tight AUROC={f(got['tight']['auroc'])} "
                  f"ample AUROC={f(got['ample']['auroc'])} "
                  f"Δ={f(d)} 区间分离={sep}")
    return rows


def main() -> int:
    import argparse
    from . import report as report_mod
    ap = argparse.ArgumentParser(description="ForesightBench 运行器")
    ap.add_argument("--n", type=int, default=60, help="每格题数")
    ap.add_argument("--seed", type=int, default=20261003)
    ap.add_argument("--out", default=os.path.join(BENCH_DIR, "results"))
    ap.add_argument("--power", action="store_true", help="只跑样本量功效扫描")
    ap.add_argument("--all", action="store_true",
                    help="功效扫描 + 全量评测 + 生成报告（复现用这一条）")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    def do_power():
        rows = power_scan(seed=a.seed, verbose=not a.quiet)
        with open(os.path.join(a.out, "power_scan.json"), "w",
                  encoding="utf-8", newline="\n") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print(f"已写出 {a.out}/power_scan.json")

    if a.power:
        do_power()
        return 0

    if a.all:
        do_power()

    res = evaluate(n_per_cell=a.n, seed=a.seed, verbose=not a.quiet)
    with open(os.path.join(a.out, "results.json"), "w",
              encoding="utf-8", newline="\n") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"已写出 {a.out}/results.json")
    # 报告必须由同一次运行产出：手写的数字迟早会和 JSON 不一致。
    report_mod.main_with_dir(a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

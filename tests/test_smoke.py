#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""冒烟测试：把"报告里声称的东西"逐条变成可执行的断言。

    D:\\python\\python.exe tests\\test_smoke.py

设计原则：**每条测试都对应报告里的一句声称**。测试失败就意味着报告在说谎。
不引入 pytest —— 零依赖是本项目的硬约束（见 proposal.md 第 8 节）。
"""
from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "benchmark"))

from mb import metrics, solvers, tasks                        # noqa: E402
from mb import runner                                        # noqa: E402

PY = sys.executable
N_OK = N_FAIL = 0
FAILS = []


def check(cond: bool, label: str, detail: str = "") -> None:
    global N_OK, N_FAIL
    if cond:
        N_OK += 1
        print(f"  [OK]   {label}")
    else:
        N_FAIL += 1
        FAILS.append(label)
        print(f"  [FAIL] {label}  {detail}")


def close(a, b, tol=1e-12) -> bool:
    return abs(a - b) <= tol


def section(t: str) -> None:
    print(f"\n== {t} ==")


# --------------------------------------------------------------- 指标层
def test_wilson():
    section("Wilson 区间（声称：小样本下不会给出 >1 的上界）")
    for k, n in [(0, 1), (1, 1), (0, 10), (10, 10), (3, 20), (0, 0)]:
        lo, hi = metrics.wilson(k, n)
        check(0.0 <= lo <= hi <= 1.0, f"wilson({k},{n}) ⊂ [0,1]",
              f"得到 ({lo},{hi})")


def test_ece_gaming():
    section("ECE 可被零信息策略刷到 0（proposal.md 第 1.2 节的命题）")
    # 基础正确率恰为 50% → 恒报 0.5 的 ECE 必须精确为 0
    correct = [True, False] * 50
    confs = [0.5] * 100
    e = metrics.ece(confs, correct)
    check(close(e, 0.0), "基础正确率 50% 时 恒报0.5 的 ECE == 0", f"ECE={e}")
    a = metrics.auroc(confs, correct)
    check(close(a, 0.5), "同一策略 AUROC == 0.5（零信息，并列取平均秩）",
          f"AUROC={a}")
    bd = metrics.brier_decompose(confs, correct)
    check(close(bd["resolution"], 0.0), "同一策略 分辨率 == 0", 
          f"RES={bd['resolution']}")
    # 恒报 0.5 但基础正确率不是 50% → ECE 应等于 |0.5 - 基础率|
    correct2 = [True] * 80 + [False] * 20
    e2 = metrics.ece([0.5] * 100, correct2)
    check(close(e2, 0.30, 1e-12), "ECE(恒报0.5) == |0.5 − 基础正确率|", f"得到 {e2}")


def test_auroc_semantics():
    section("AUROC 语义（声称：单类返回 None，并列返回 0.5）")
    check(metrics.auroc([0.9, 0.8], [True, True]) is None,
          "单类样本（全对）→ None（未定义，不是 0.5）")
    check(metrics.auroc([0.2, 0.1], [False, False]) is None,
          "单类样本（全错）→ None")
    check(close(metrics.auroc([0.9, 0.1], [True, False]), 1.0),
          "完美排序 → AUROC == 1.0")
    check(close(metrics.auroc([0.1, 0.9], [True, False]), 0.0),
          "反向排序 → AUROC == 0.0")
    check(close(metrics.auroc([0.5, 0.5, 0.5], [True, False, True]), 0.5),
          "全并列 → AUROC == 0.5")
    # 单调变换不变性 —— 这是"分辨"与"校准"正交的核心
    confs = [0.1, 0.4, 0.6, 0.9, 0.2, 0.7]
    cor = [False, False, True, True, False, True]
    a1 = metrics.auroc(confs, cor)
    a2 = metrics.auroc([c ** 2 for c in confs], cor)
    a3 = metrics.auroc([min(1.0, c + 0.05) for c in confs], cor)
    check(close(a1, a2, 1e-12) and close(a1, a3, 1e-12),
          "AUROC 对置信度的单调变换不变（校准与分辨正交）",
          f"{a1} vs {a2} vs {a3}")


def test_brier_identity():
    section("Brier 分解（声称：精确口径逐位相符，10箱残差恰等于组内离散项）")
    import random
    r = random.Random(7)
    for trial in range(20):
        n = r.randrange(5, 200)
        confs = [r.random() for _ in range(n)]
        cor = [r.random() < 0.5 for _ in range(n)]
        bd = metrics.brier_decompose(confs, cor)
        if not close(bd["brier"], bd["brier_from_parts"], 1e-12):
            break
        if not close(bd["bin_residual"], bd["within_bin_spread"], 1e-12):
            break
    else:
        check(True, "20 组随机数据：精确恒等式 + 组内离散项一致性 全部通过")
        return
    check(False, "Brier 分解自检（随机数据）",
          f"brier={bd['brier']} parts={bd['brier_from_parts']} "
          f"δ={bd['bin_residual']} within={bd['within_bin_spread']}")


def test_utility_rebase():
    section("效用捕获率（声称：改基准后恒定置信度策略被压到 0）")
    correct = [True, False, True, False, True, False, True, False]
    for c in (0.0, 0.25, 0.5, 0.75, 1.0):
        u = metrics.utility([c] * len(correct), correct, tau=0.5)
        check(close(u["capture_vs_best_constant"], 0.0),
              f"恒报 {c} → capture* == 0", f"得到 {u['capture_vs_best_constant']}")
    # 完美前瞻必须恰好拿满
    confs = [1.0 if k else 0.0 for k in correct]
    check(close(metrics.utility(confs, correct, tau=0.5)
                ["capture_vs_best_constant"], 1.0),
          "完美前瞻 → capture* == 1.0")


def test_narrow_edges():
    section("边界情形")
    s = metrics.summarize([], [])
    check(s["n"] == 0 and s["auroc"] is None, "空输入不崩且 AUROC 未定义")
    s1 = metrics.summarize([0.7], [True])
    check(s1["auroc"] is None, "n=1 时 AUROC 未定义（单类）")
    check(close(metrics.ece([], []), 0.0), "空输入 ECE == 0")


# --------------------------------------------------------------- 任务层
def test_tasks():
    section("任务族（声称：4 族 × 6 难度，真值精确，实例不重叠）")
    for fam in tasks.GENERATORS:
        for d in tasks.DIFFICULTIES:
            ts = tasks.make_tasks(fam, d, 5, 20261003)
            check(len(ts) == 5 and all(t.family == fam and t.difficulty == d
                                       for t in ts),
                  f"{fam} d={d} 生成 5 道且族/难度正确")
    # 真值可被独立重算（不依赖生成器内部状态）
    t = tasks.gen_arith(3, 0, 1)
    body = t.prompt.split("：", 1)[1].split()
    acc = int(body[0])
    for i in range(1, len(body), 2):
        acc = acc + int(body[i + 1]) if body[i] == "+" else acc - int(body[i + 1])
    check(str(acc) == t.answer, "算术链真值可由题面独立重算", f"{acc} vs {t.answer}")
    t2 = tasks.gen_csp(3, 0, 1)
    perm = [int(x) for x in t2.answer.split(",")]
    check(sorted(perm) == list(range(1, t2.meta["n"] + 1)),
          "约束满足的答案是一个合法排列")
    t3 = tasks.gen_trace(3, 0, 1)
    check(t3.answer.lstrip("-").isdigit(), "程序追踪的真值是整数")
    t4 = tasks.gen_seq(2, 0, 1)
    check(t4.answer.lstrip("-").isdigit(), "序列归纳的真值是整数")
    # 不同 offset 必须给出不同实例
    a = tasks.make_tasks("arith", 2, 4, 20261003, 0)
    b = tasks.make_tasks("arith", 2, 4, 20261003, 10000)
    check(not ({x.prompt for x in a} & {x.prompt for x in b}),
          "不同迁移条件（offset 不同）题面不重叠")


def test_determinism():
    section("可复现性（声称：sha256 派生种子，跨进程稳定）")
    a = [t.prompt for t in tasks.make_tasks("trace", 4, 6, 20261003, 0)]
    b = [t.prompt for t in tasks.make_tasks("trace", 4, 6, 20261003, 0)]
    check(a == b, "同进程内重复生成完全一致")
    # 换进程 + 强制不同的 PYTHONHASHSEED：如果用了内置 hash()，这里必然不一致
    code = (
        "import sys;sys.path.insert(0, r'%s');"
        "from mb import tasks;"
        "print('|'.join(t.prompt for t in tasks.make_tasks('trace',4,6,20261003,0)))"
        % os.path.join(ROOT, "benchmark")
    )
    outs = []
    for hs in ("0", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=hs, PYTHONIOENCODING="utf-8")
        # 不用管道捕获输出：本环境的安全策略禁止管道，
        # 因此把子进程的 stdout 直接重定向到一个临时文件再读回来。
        tmp = os.path.join(ROOT, f"_det_probe_{hs}.txt")
        with open(tmp, "w", encoding="utf-8") as fh:
            subprocess.run([PY, "-c", code], stdout=fh,
                           stderr=subprocess.STDOUT, env=env, check=False)
        with open(tmp, encoding="utf-8") as fh:
            outs.append(fh.read().strip())
        os.remove(tmp)
    check(outs[0] == outs[1] != "" and outs[0].split("|") == a,
          "跨进程 + 不同 PYTHONHASHSEED 仍完全一致（未使用内置 hash()）",
          f"长度 {[len(x) for x in outs]}")
    r1 = tasks._rng("x", 1, 2).random()
    r2 = tasks._rng("x", 1, 2).random()
    check(r1 == r2, "_rng 同参数同结果")
    check(r1 != tasks._rng("x", 1, 3).random(), "_rng 不同参数不同结果")


# --------------------------------------------------------------- 求解器层
def test_solvers_confidence_from_process():
    section("求解器（声称：置信度来自过程量，且不偷看真值）")
    for fam, fn in solvers.SOLVERS.items():
        confs = []
        for d in (1, 6):
            for t in tasks.make_tasks(fam, d, 12, 20261003):
                p = fn(t, solvers.TIGHT)
                check(0.0 <= p.confidence <= 1.0,
                      f"{fam} d={d} 置信度 ⊂ [0,1]", f"{p.confidence}")
                confs.append(p.confidence)
        check(len(set(round(c, 9) for c in confs)) > 1,
              f"{fam} 的置信度不是常数（确实随过程变化）")
    # 置信度必须在**不看真值**的前提下算得出来：把 task.answer 换成垃圾，
    # 置信度与答案都不能变（否则说明置信度依赖真值）。
    import dataclasses
    for fam, fn in solvers.SOLVERS.items():
        t = tasks.make_tasks(fam, 3, 1, 20261003)[0]
        p1 = fn(t, solvers.TIGHT)
        t_bad = dataclasses.replace(t, answer="<<不存在的答案>>")
        p2 = fn(t_bad, solvers.TIGHT)
        check(p1.answer == p2.answer and close(p1.confidence, p2.confidence),
              f"{fam} 求解器不读取真值（换掉 answer 结果不变）")
    # 资源越紧，置信度整体越低（单调性 —— "前瞻性"的机制证据）
    for fam, fn in solvers.SOLVERS.items():
        ts = tasks.make_tasks(fam, 5, 20, 20261003)
        mt = sum(fn(t, solvers.TIGHT).confidence for t in ts) / len(ts)
        ma = sum(fn(t, solvers.AMPLE).confidence for t in ts) / len(ts)
        check(mt <= ma + 1e-9,
              f"{fam}：TIGHT 的平均置信度 ≤ AMPLE（资源紧 → 更不自信）",
              f"{mt:.4f} vs {ma:.4f}")


def test_validation_policies():
    section("仪表校验策略（声称：改真值不影响策略输出，只改置信度报法）")
    t = tasks.make_tasks("arith", 2, 1, 20261003)[0]
    base = solvers.solve_arith(t, solvers.TIGHT)
    for pol in solvers.VALIDATION_POLICIES:
        for k in (True, False):
            q = pol(t, base, k)
            check(q.answer == base.answer,
                  f"{pol.name} 不改变答案（correct={k}）")
    check(close(solvers.ConstantHalf()(t, base, True).confidence, 0.5),
          "constant_half 恒为 0.5")
    check(solvers.OracleConf()(t, base, False).confidence == 0.0,
          "oracle_conf 在错误时给 0.0")


# --------------------------------------------------------------- 编排层
def test_conditions_no_overlap():
    section("迁移条件（声称：条件之间实例不重叠，τ* 冻结）")
    got = {}
    for ci, (cname, _d, fams, diffs) in enumerate(runner.CONDITIONS):
        off = 10000 * (ci + 1)
        got[cname] = {t.tid for f in fams for dd in diffs
                      for t in tasks.make_tasks(f, dd, 4, 20261003, off)}
    names = list(got)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            inter = got[names[i]] & got[names[j]]
            check(not inter, f"{names[i]} ∩ {names[j]} == ∅", f"{len(inter)} 条重叠")
    res = runner.evaluate(n_per_cell=8, seed=20261003, verbose=False)
    check(set(res["tau_frozen"]) == {"tight", "ample"},
          "τ* 两档各有一个冻结值")
    # 迁移条件里所有求解器行（含校验策略）用的都是同一个冻结 τ*
    taus = {k.split("::")[0]: v["utility"]["tau"]
            for _c, b in res["conditions"].items()
            for k, v in b["per_solver"].items()}
    check(taus.get("tight") == res["tau_frozen"]["tight"]
          and taus.get("ample") == res["tau_frozen"]["ample"],
          "所有条件（含校验策略）共用冻结的 τ*，未在测试集上重选")
    # 键必须带求解器前缀（否则 ample 会覆盖 tight 的校验策略结果）
    for _c, b in res["conditions"].items():
        keys = [k for k in b["per_solver"] if "::" in k]
        check(len(keys) == 2 * len(solvers.VALIDATION_POLICIES),
              "校验策略结果按求解器分键（未被覆盖）", f"{len(keys)} 个键")
        break


def test_report_consistency():
    section("报告与 JSON 一致（声称：报告由同一次运行产出）")
    import json
    p = os.path.join(ROOT, "benchmark", "results", "results.json")
    if not os.path.exists(p):
        check(True, "results.json 不存在，跳过（先跑 run.py --all）")
        return
    d = json.load(open(p, encoding="utf-8"))
    rows = [v for b in d["conditions"].values()
            for k, v in b["per_solver"].items() if "::" not in k]
    check(len(rows) == 12, "表 1 恰好 12 行（6 条件 × 2 档）", f"{len(rows)}")
    worst = max(abs(v["brier"] - v["brier_from_parts"]) for v in rows)
    check(worst < 1e-9, "Brier 精确恒等式在真实结果上通过", f"最大差 {worst:.2e}")
    worst2 = max(abs(v["bin_residual"] - v["within_bin_spread"]) for v in rows)
    check(worst2 < 1e-9, "组内离散项两口径一致", f"最大差 {worst2:.2e}")
    txt = open(os.path.join(ROOT, "benchmark", "results", "REPORT.md"),
               encoding="utf-8").read()
    check("未通过" not in txt, "REPORT.md 里没有出现任何「未通过」")
    check("未定义" in txt or all(v["auroc_defined"] for v in rows),
          "报告如实处理了 AUROC 未定义的情形")


def main() -> int:
    print("ForesightBench 冒烟测试")
    test_wilson()
    test_ece_gaming()
    test_auroc_semantics()
    test_brier_identity()
    test_utility_rebase()
    test_narrow_edges()
    test_tasks()
    test_determinism()
    test_solvers_confidence_from_process()
    test_validation_policies()
    test_conditions_no_overlap()
    test_report_consistency()
    print(f"\n{'='*56}\n通过 {N_OK} 项，失败 {N_FAIL} 项")
    if FAILS:
        print("失败清单：")
        for x in FAILS:
            print("  -", x)
    return 1 if N_FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())

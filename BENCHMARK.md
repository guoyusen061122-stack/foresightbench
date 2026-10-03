# ForesightBench 基准说明

本文件是可执行的基准规格。立项动机与论证见 `proposal.md`；实测数字见
`benchmark/results/REPORT.md`。

---

## 1. 基准要测什么

**前瞻性元认知**：在拿到题目、尚未求解时，系统对"我这次会不会答对"的判断，
与真实结果之间的对应关系质量。

拆成四个**必须同时报告**的量：

| 量 | 记号 | 定义位置 | 方向 |
|---|---|---|---|
| 校准 | ECE | `mb/metrics.py::ece` | 越低越好 |
| 分辨 | AUROC | `mb/metrics.py::auroc` | 越高越好，0.5 = 零信息 |
| 分辨率 | resolution | `mb/metrics.py::brier_decompose` | 越高越好 |
| 行为控制 | capture* | `mb/metrics.py::utility` | 越高越好，0 = 不比最好的常数策略强 |

---

## 2. 任务族

四个族，全部由**生成器 + 种子**决定，真值由构造器直接给出（不依赖标注）。

| 族 | 生成函数 | 内容 | 难度旋钮（1→6） |
|---|---|---|---|
| `arith` | `tasks.gen_arith` | 多步定点算术链 | 步数 4→24；有效数字需求递增 |
| `csp` | `tasks.gen_csp` | 有限域约束满足 | 变量数 5→14；约束紧度递增 |
| `seq` | `tasks.gen_seq` | 隐藏规则的序列归纳 | 规则阶数 1→5；项数递增 |
| `trace` | `tasks.gen_trace` | 带内存读写的程序跟踪 | 语句数 6→26；活跃变量递增 |

统一的 `Task` 结构（`mb/tasks.py`）：

```python
@dataclass(frozen=True)
class Task:
    tid: str          # "<family>-d<难度>-<序号>"，跨条件不重复
    family: str
    difficulty: int
    prompt: str       # 给求解器看的题面（不含真值）
    answer: str       # 真值，**只用于判分**，不进入求解过程
    meta: dict        # 生成参数（便于事后分族/分参数分析）
```

**规模**：4 族 × 6 难度 × 每格 60 题 = **1440 题**。

生成命令（会写出 `data/tasks.json`）：

```powershell
D:\python\python.exe -m mb.tasks        # 在 benchmark/ 目录下
```

---

## 3. 求解器：置信度必须从过程涌现

**硬约束：求解器不允许直接输出主观概率。** 它必须同时产出答案与一个由
**过程内部可核查量**换算出的置信度。这样，改变资源限制就会**自动**改变
置信度，不需要任何人告诉它"你该更不自信了"。

| 求解器 | `mb/solvers.py` | 过程量 → 置信度 |
|---|---|---|
| `Precision` | `solve_precision` | `1 − 舍入误差上界 / 容差`（分析式上界） |
| `Beam` | `solve_beam` | `自身满足的约束数 / 总约束数` |
| `ModelSelect` | `solve_modelselect` | `幸存候选数 / 初始候选数` |
| `WorkingMemory` | `solve_workingmemory` | `成功读取次数 / 总读取次数` |

**资源档位**（同一算法，不同资源——不是两个不同算法）：

```python
TIGHT = Limits("tight", sig_digits=6,  beam_width=4,  shown_terms=4,  memory_slots=4)
AMPLE = Limits("ample", sig_digits=15, beam_width=99, shown_terms=12, memory_slots=24)
```

真值**不进入求解过程**：`runner._run_solver` 拿到 `Prediction` 之后才用
`task.answer` 判分。这一点在代码里是显式的（见该函数的 docstring）。

---

## 4. 分布迁移条件

| 条件 | 参与族 | 难度 | 偏移 | 用途 |
|---|---|---|---|---|
| `S1_same` | 全部 4 族 | 2,3,4 | 0（=校准集） | 同分布对照 |
| `S2_easier` | 全部 4 族 | 1,2,3 | 10000 | 难度下移 |
| `S3_harder` | 全部 4 族 | 4,5,6 | 20000 | 难度上移 |
| `S4_bimodal` | 全部 4 族 | 1,2,6 | 30000 | 双峰 |
| `S5_hardest` | 全部 4 族 | 6 | 40000 | 极难 |
| `S6_crossfam` | `seq`,`trace` | 3,4 | 50000 | **跨族迁移** |

**实例不重叠**由 `offset` 保证（`tasks.make_tasks(..., offset)`）。

### 4.1 冻结协议（本基准的核心）

1. 在**校准集**（`S1_same` 的实例，offset=0）上，按弃权效用最大选阈值
   $\tau^\*=\arg\max_\tau U(\tau)$，网格 `0.05…0.95`；
2. $\tau^\*$ **冻结**，六个条件共用；
3. 测试集上**不允许**重选阈值。

实测冻结值：`tight` = 0.95、`ample` = 0.90。

**这条协议的代价是真实的**：`tight` 在 S3/S5/S6 的效用捕获全为 0.000
（高阈值在新分布里几乎不触发 → 全程弃权 → 吃满弃权代价 0.35）。
这不是缺陷，是协议要暴露的现象：**"有分辨力"与"分辨力可用"是两件事**。

### 4.2 校验策略（仪表校验，不是求解器）

`mb/solvers.py` 的 `VALIDATION_POLICIES`：`oracle_conf` / `constant_one` /
`constant_half` / `inverted_conf`。它们**故意使用真值**，作用是撑开各指标的
取值空间，证明"单一指标可被刷"。

**它们复用真实求解器的同一批答案**，只改置信度报法 → 每行的准确率相同，
只有四个量在变。这是"校准与分辨是两个独立维度"的可复现证据。

---

## 5. 指标口径

### 5.1 ECE（10 等宽分箱）

$$\mathrm{ECE}=\sum_{b}\frac{n_b}{N}\big|\bar p_b-\bar o_b\big|$$

### 5.2 AUROC（Mann–Whitney，并列取平均秩）

$$A=\frac{\sum_{i\in\text{pos}} r_i - n_1(n_1+1)/2}{n_1n_0}$$

- 全部置信度相同 → 并列 → 平均秩 → $A=0.5$（零信息，有定义）；
- **单类样本**（全对或全错）→ 返回 **`None`（未定义）**。
  按惯例回落 0.5 会让"求解器全错"看起来像"零信息但有效"，是会误导读数的地方。

### 5.3 Brier 分解

$$\text{Brier}=\mathrm{REL}-\mathrm{RES}+\mathrm{UNC}+\delta$$

$\delta$ 是**组内离散项**，只在"组内置信度完全相等"时为零：

$$\delta=\frac1N\sum_k\Big[\sum_{i\in k}(p_i-\bar p_k)^2+2\sum_{i\in k}(p_i-\bar p_k)(\bar y_k-y_i)\Big]$$

本实现报两种口径：`reliability_exact`/`resolution_exact`（按置信度**取值**分组，
恒等式精确成立）与 `reliability`/`resolution`（10 等宽分箱），并把 $\delta$
显式报出。**`bin_residual` 必须恰等于 `within_bin_spread`** —— 这条等式就是
分解写对了的证据。详见 `REPORT.md` 表 4。

### 5.4 行为控制（非对称代价弃权）

规则：置信度 ≥ τ 则作答。
- 作答且错 → 代价 `cost_wrong = 1.0`
- 作答且对 → 代价 0
- 弃权 → 代价 `cost_abstain = 0.35`（与对错无关）

$$U=\frac{1}{N}\Big(\text{总代价}\Big)\ \Rightarrow\ U=1-\frac{\sum \text{cost}}{N\cdot cost_{wrong}}$$

**效用捕获率（修正版，本基准采用）**：

$$\text{capture}^\*=\frac{U-\max(U_{\text{never}},U_{\text{all}})}{U_{\text{oracle}}-\max(U_{\text{never}},U_{\text{all}})}\in[0,1]$$

原版以 $U_{\text{never}}$ 为基准，会把"低准确率下干脆全弃权"这份便宜算成
元认知的功劳（实测 `constant_half` 在 S6 拿到 0.905 > `tight` 的 0.627）。
改基准后**任何恒定置信度策略都被压到 0**。原值仍以 `capture` 字段一并保留。

### 5.5 区间

- 比例：**Wilson 95%**
- AUROC / Brier：**百分位自助**，2000 次，固定种子 `20261003`
- 单类样本下 AUROC 区间为 `None`

---

## 6. 可复现性

```powershell
# 一键（功效扫描 + 全量评测 + 报告）
D:\python\python.exe benchmark\run.py --all

# 快速冒烟
D:\python\python.exe benchmark\run.py --n 8

# 只跑功效扫描
D:\python\python.exe benchmark\run.py --power

# 只重出报告（从已有 results.json）
D:\python\python.exe benchmark\mb\report.py benchmark\results
```

| 项 | 值 |
|---|---|
| 完整跑耗时 | 约 66 秒（纯 CPU）|
| 依赖 | **仅 Python 3 标准库** |
| 种子 | 题集 `20261003`；自助 `20261003` |
| **种子派生** | `hashlib.sha256`（**不是**内置 `hash()`——它对字符串有每进程随机盐） |
| 网络 | 基准与指标**完全不联网** |

**种子派生**（`mb/tasks.py::_rng`）：

```python
def _rng(*parts):
    key = "|".join(str(p) for p in parts).encode("utf-8")
    return random.Random(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))
```

任何 `seed` 都可用（`--seed`），且换种子不会破坏可复现性——这是抗污染的
基础：私有种子不进公开集，就不可能被训练见过。

---

## 7. 目录结构

```
benchmark/
  run.py                 入口：--n / --seed / --power / --all
  mb/
    tasks.py             Task + 4 个生成器 + _rng(sha256) + CLI
    solvers.py           Limits / TIGHT / AMPLE / 4 个求解器 / 4 个校验策略
    metrics.py           wilson / ece / brier_decompose / auroc /
                         risk_coverage / aurc / utility / bootstrap_ci / summarize
    runner.py            CONDITIONS / CALIB / pick_tau / evaluate / power_scan / main
    report.py            results.json → REPORT.md（表 1–4 + 附录）
  data/tasks.json        1440 题（可重新生成）
  results/
    results.json         全部原始数字（含每族分解、每策略分解）
    power_scan.json      样本量功效扫描
    REPORT.md            自动生成的报告
```

---

## 8. 二次开发提示

- **加任务族**：在 `tasks.py` 写 `gen_xxx`，加进 `FAMILY_ZH`，在 `runner.CONDITIONS`
  里引用族名即可。指标层不用改。
- **加求解器**：在 `solvers.py` 写 `solve_xxx` 返回 `Prediction(answer, confidence, meta)`，
  加进 `SOLVERS`；在 `runner.REAL_SOLVERS` 里加档位。**唯一限制**：`confidence`
  必须由过程量换算，不许手写常数。
- **加迁移条件**：在 `runner.CONDITIONS` 加一行 `(名字, 说明, 族元组, 难度元组)`，
  `offset` 会自动保证实例不重叠。
- **接真实模型**：`llm/run_llm.py` 已把接口留好（OpenAI 兼容），
  置信度从模型自报或采样一致性引出；**注意**：那一路的置信度不是"过程涌现"的，
  报告里必须单独标注，不要和本基准的四族求解器混在一张表里比较。

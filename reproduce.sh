#!/usr/bin/env sh
# ForesightBench 一键复现（Linux / macOS / WSL）
#
#   sh reproduce.sh
#
# 做三件事：核验参考文献 → 跑冒烟测试 → 跑基准并生成报告。
# 基准部分不联网；只有第 1 步需要网络。
set -e

export PYTHONIOENCODING=utf-8
PY="${FORESIGHT_PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || PY=python
echo "使用 Python: $PY"

echo
echo "[1/3] 核验参考文献 → references.bib"
"$PY" pipeline/verify_refs.py --apply || {
  echo "  参考文献核验需要网络，本次失败不影响基准结果。"
  echo "  已有的 references.bib 保持不变。"
}

echo
echo "[2/3] 冒烟测试（每条断言对应报告里的一句声称）"
"$PY" tests/test_smoke.py

echo
echo "[3/3] 基准：功效扫描 + 全量评测 + 报告（约 70 秒）"
"$PY" benchmark/run.py --all

echo
echo "完成。报告在 benchmark/results/REPORT.md"
head -n 30 benchmark/results/REPORT.md

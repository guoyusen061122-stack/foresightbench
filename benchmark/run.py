#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ForesightBench 一条命令入口。

    python run.py                 # 跑全量（默认每格 60 题）
    python run.py --n 20          # 快速冒烟
    python run.py --power         # 样本量功效扫描
    python run.py --tables        # 出 LaTeX 表
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mb import runner  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(runner.main())

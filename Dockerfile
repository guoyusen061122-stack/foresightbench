# ForesightBench —— 一键复现环境
#
# 基准与指标**只用 Python 标准库**，因此镜像里没有任何第三方依赖。
# 唯一联网的部分是参考文献核验（pipeline/verify_refs.py），
# 它需要网络；跑基准不需要。
#
#   docker build -t foresightbench .
#   docker run --rm foresightbench              # 冒烟测试 + 全量 + 报告
#   docker run --rm foresightbench python benchmark/run.py --n 8

FROM python:3.11-slim

WORKDIR /app
COPY . /app

# 只用标准库，所以这里没有 pip install。
# 用 python -m compileall 做一次语法自检，让镜像构建阶段就能发现语法错误。
RUN python -m compileall -q benchmark/mb benchmark/run.py \
        pipeline/verify_refs.py llm/run_llm.py tests/test_smoke.py

ENV PYTHONIOENCODING=utf-8

# 默认：跑测试（证明结构完好）→ 全量评测 + 功效扫描 + 报告。
# 只跑基准（跳过测试）用：
#   docker run --rm foresightbench python benchmark/run.py --all
CMD ["sh", "-c", "python tests/test_smoke.py && python benchmark/run.py --all"]

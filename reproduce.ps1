# ForesightBench 一键复现（Windows / PowerShell）
#
#   powershell -ExecutionPolicy Bypass -File .\reproduce.ps1
#
# 做三件事：核验参考文献 → 跑冒烟测试 → 跑基准并生成报告。
# 基准部分不联网；只有第 1 步需要网络。

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"

# 找 Python：优先用显式指定的，其次 PATH 上的 python
$py = $env:FORESIGHT_PYTHON
if (-not $py) {
    $cand = @("D:\python\python.exe", "python")
    foreach ($c in $cand) {
        if (Get-Command $c -ErrorAction SilentlyContinue) { $py = $c; break }
    }
}
if (-not $py) { throw "找不到 Python，请设置环境变量 FORESIGHT_PYTHON" }
Write-Host "使用 Python: $py" -ForegroundColor Cyan

Write-Host "`n[1/3] 核验参考文献 → references.bib" -ForegroundColor Cyan
try {
    & $py pipeline\verify_refs.py --apply
} catch {
    Write-Host "  参考文献核验需要网络，本次失败不影响基准结果。" -ForegroundColor Yellow
    Write-Host "  已有的 references.bib 保持不变。" -ForegroundColor Yellow
}

Write-Host "`n[2/3] 冒烟测试（每条断言对应报告里的一句声称）" -ForegroundColor Cyan
& $py tests\test_smoke.py
if ($LASTEXITCODE -ne 0) { throw "冒烟测试未通过，请先修好再继续" }

Write-Host "`n[3/3] 基准：功效扫描 + 全量评测 + 报告（约 70 秒）" -ForegroundColor Cyan
& $py benchmark\run.py --all

Write-Host "`n完成。报告在 benchmark\results\REPORT.md" -ForegroundColor Green
Get-Content benchmark\results\REPORT.md -TotalCount 30

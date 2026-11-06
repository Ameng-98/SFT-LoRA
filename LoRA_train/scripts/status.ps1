# 训练/评测进度速查：步骤、loss、ETA、产物状态
# 用法： powershell -ExecutionPolicy Bypass -File .\scripts\status.ps1

Set-Location "E:\Desktop\wolin\project_final\LoRA_train"

Write-Host "=== 进程 ===" -ForegroundColor Cyan
$train = Get-Process -Id 6260 -ErrorAction SilentlyContinue
if ($train) {
    Write-Host ("训练进程 PID 6260 运行中，已用 CPU {0:N0}s，内存 {1:N0} MB" -f $train.CPU, ($train.WorkingSet64 / 1MB))
} else {
    Write-Host "训练进程 PID 6260 已结束"
}

Write-Host "`n=== 进度 ===" -ForegroundColor Cyan
if (Test-Path "logs\train.err.log") {
    (Get-Content "logs\train.err.log" -Raw -Encoding utf8) -split "`n" |
        Where-Object { $_ -match "/152" } | Select-Object -Last 1
}
if (Test-Path "logs\train.log") {
    Get-Content "logs\train.log" -Encoding utf8 | Where-Object { $_ -match "loss" } | Select-Object -Last 3
}

Write-Host "`n=== 产物 ===" -ForegroundColor Cyan
Get-ChildItem "output" -ErrorAction SilentlyContinue | Select-Object Name, LastWriteTime | Format-Table -AutoSize

Write-Host "`n=== 日志尾部 ===" -ForegroundColor Cyan
Get-Content "logs\train.log" -Encoding utf8 -Tail 5
if (Test-Path "logs\eval_watch.log") {
    Write-Host "`n=== 自动评测 ===" -ForegroundColor Cyan
    Get-Content "logs\eval_watch.log" -Encoding utf8 -Tail 3
}

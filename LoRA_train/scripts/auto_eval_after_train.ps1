# 训练进程结束后自动执行 18 条验收用例评测（基座 vs LoRA），结果写入 output\test_results_both_*.md
# 用法： powershell -ExecutionPolicy Bypass -File .\scripts\auto_eval_after_train.ps1 -TrainPid 6260

param(
    [Parameter(Mandatory = $true)]
    [int]$TrainPid,
    [string]$Mode = "both"
)

$root = "E:\Desktop\wolin\project_final\LoRA_train"
$py = "C:\Users\16096\.conda\envs\langchain\python.exe"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Set-Location $root

# 等待训练进程退出（训练完成后 PID 会消失）
while (Get-Process -Id $TrainPid -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 20
}

$stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
"训练进程已结束 $stamp" | Out-File -Encoding utf8 "logs\eval_watch.log" -Append

if (-not (Test-Path "output\lora_adapter")) {
    "❌ 未找到 output\lora_adapter 目录，跳过评测" | Out-File -Encoding utf8 "logs\eval_watch.log" -Append
    exit 1
}

"开始评测（$Mode）" | Out-File -Encoding utf8 "logs\eval_watch.log" -Append
& $py -u "scripts\test_ecommerce_model.py" --model $Mode --adapter "output\lora_adapter" --max-new-tokens 160 *> "logs\eval.log"
"评测结束，退出码 $LASTEXITCODE，时间 $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" |
    Out-File -Encoding utf8 "logs\eval_watch.log" -Append

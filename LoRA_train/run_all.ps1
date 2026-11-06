# 一键：数据转换 → 环境自检 → dry-run → 正式训练 → 18 条评测
# 用法： powershell -ExecutionPolicy Bypass -File .\run_all.ps1
# 可选： -SkipTrain 只跑评测； -Mode lora|base|both 指定评测对象

param(
    [switch]$SkipTrain,
    [string]$Mode = "both"
)

$ErrorActionPreference = "Stop"
$py = "C:\Users\16096\.conda\envs\langchain\python.exe"
$skill = "E:\Desktop\wolin\project_final\LoRA_skills\qwen-lora-sft"
$model = "E:\Desktop\wolin\zuoye\week3_day4\test\model\models\Qwen--Qwen2.5-1.5B-Instruct\snapshots\master"
$root = "E:\Desktop\wolin\project_final\LoRA_train"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Set-Location $root

if (-not $SkipTrain) {
    Write-Host "== 0/4 生成三角色训练数据 ==" -ForegroundColor Cyan
    & $py scripts\build_sft_dataset.py `
        --input "E:\Desktop\wolin\project_final\dataset\ecommerce_customer_service_sft_600.json" `
        --output "data\train_messages.json" --report "data\dataset_report.json" --tokenizer $model

    Write-Host "== 1/4 环境自检 ==" -ForegroundColor Cyan
    & $py "$skill\scripts\env_report.py" --config config\finetune_cpu.json

    Write-Host "== 2/4 dry-run ==" -ForegroundColor Cyan
    & $py "$skill\scripts\finetune.py" --config config\finetune_cpu.json --dry-run

    Write-Host "== 3/4 正式训练 ==" -ForegroundColor Cyan
    & $py "$skill\scripts\finetune.py" --config config\finetune_cpu.json
}

Write-Host "== 4/4 18 条验收用例评测（$Mode）==" -ForegroundColor Cyan
& $py scripts\test_ecommerce_model.py --model $Mode --adapter output\lora_adapter

Write-Host "全部完成，结果见 output\" -ForegroundColor Green

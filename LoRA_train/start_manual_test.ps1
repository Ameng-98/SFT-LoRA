# 人工测试入口：交互式对话（默认 LoRA 模型）
# 用法：
#   .\start_manual_test.ps1                 # LoRA 交互对话
#   .\start_manual_test.ps1 -Mode base      # 基座交互对话（对比用）
#   .\start_manual_test.ps1 -Case 3         # 直接跑第 3 条官方用例
#   .\start_manual_test.ps1 -Mode base -Case 3

param(
    [ValidateSet("lora", "base")]
    [string]$Mode = "lora",
    [int]$Case = 0
)

$py = "C:\Users\16096\.conda\envs\langchain\python.exe"
$root = "E:\Desktop\wolin\project_final\LoRA_train"

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
chcp 65001 > $null

Set-Location $root

Write-Host "==============================================" -ForegroundColor Cyan
Write-Host " 电商客服模型人工测试   模式 = $Mode" -ForegroundColor Cyan
if ($Mode -eq "lora") {
    Write-Host " adapter = output\lora_adapter" -ForegroundColor DarkGray
} else {
    Write-Host " 只加载基座模型（未挂 LoRA）" -ForegroundColor DarkGray
}
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host " 首次加载约 10 秒，单条回复约 10~40 秒（CPU）" -ForegroundColor DarkGray
Write-Host " 输入 /help 看命令，/quit 退出" -ForegroundColor DarkGray
Write-Host ""

$pyArgs = @("scripts\manual_test.py", "--mode", $Mode)
if ($Case -gt 0) {
    $pyArgs += @("--case", "$Case")
}

& $py @pyArgs

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n模型启动失败，退出码 $LASTEXITCODE" -ForegroundColor Red
    Write-Host "常见原因：Python 路径不对、adapter 缺失、内存不足" -ForegroundColor Yellow
}

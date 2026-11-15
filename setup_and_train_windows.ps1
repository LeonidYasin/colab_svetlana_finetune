# PowerShell скрипт автоматической подготовки и запуска обучения Светланы на RTX 3060
# Запуск в PowerShell: powershell -ExecutionPolicy Bypass -File setup_and_train_windows.ps1

$ErrorActionPreference = "Stop"

Write-Host "===========================================================" -ForegroundColor Cyan
Write-Host "  🎙️ Подготовка и запуск дообучения Светланы на RTX 3060" -ForegroundColor Cyan
Write-Host "===========================================================" -ForegroundColor Cyan

# 1. Проверка наличия Git
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "⚠️ Git не найден. Устанавливаем через winget..." -ForegroundColor Yellow
    winget install --id Git.Git -e --source winget
}

# 2. Клонирование GPT-SoVITS (если еще не склонирован)
if (-not (Test-Path "GPT-SoVITS")) {
    Write-Host "📦 Клонирование репозитория GPT-SoVITS..." -ForegroundColor Green
    git clone --depth=1 https://github.com/RVC-Boss/GPT-SoVITS.git
}

Set-Location "GPT-SoVITS"

# 3. Копирование скриптов обучения в папку GPT-SoVITS
Copy-Item -Path "..\train_svetlana_windows.py" -Destination "." -Force -ErrorAction SilentlyContinue
Copy-Item -Path "..\train_svetlana.bat" -Destination "." -Force -ErrorAction SilentlyContinue
Copy-Item -Path "..\raw_data.json" -Destination "." -Force -ErrorAction SilentlyContinue

# 4. Проверка Python и PyTorch с CUDA
$PythonExe = "python"
if (Test-Path "runtime\python.exe") {
    $PythonExe = (Resolve-Path "runtime\python.exe").Path
}

Write-Host "🐍 Проверка Python ($PythonExe)..." -ForegroundColor Green
& $PythonExe -c "import torch; print('CUDA доступен:', torch.cuda.is_available(), '| Устройство:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# 5. Запуск обучения
Write-Host "🚀 Запуск скрипта дообучения на RTX 3060..." -ForegroundColor Green
& $PythonExe train_svetlana_windows.py

Write-Host "===========================================================" -ForegroundColor Cyan
Write-Host "🎉 Обучение успешно завершено!" -ForegroundColor Cyan
Write-Host "===========================================================" -ForegroundColor Cyan

# Автоматическая настройка воркера GPT-SoVITS на Windows
$ServerUrl = "http://{{SERVER_IP}}:{{PORT}}"
$WorkDir = "$HOME\GPT-SoVITS-Svetlana"

Write-Host "=========================================================" -ForegroundColor Cyan
Write-Host "  🎙️ Подготовка окружения Светланы на RTX 3060" -ForegroundColor Cyan
Write-Host "=========================================================" -ForegroundColor Cyan

# 1. Создание рабочей папки
if (-not (Test-Path $WorkDir)) {
    New-Item -ItemType Directory -Path $WorkDir -Force | Out-Null
}
Set-Location $WorkDir
Write-Host "📁 Рабочая директория: $WorkDir" -ForegroundColor Green

# 2. Определение и настройка Python
# Для PyTorch CUDA на Windows требуется Python 3.10 или 3.11
$PythonExe = $null

# Проверяем Windows Launcher (py)
if (Get-Command py -ErrorAction SilentlyContinue) {
    $p11 = & py -3.11 -c "import sys; print(sys.executable)" 2>$null
    if ($p11) {
        $PythonExe = $p11.Trim()
        Write-Host "🐍 Найден Python 3.11: $PythonExe" -ForegroundColor Green
    } else {
        $p10 = & py -3.10 -c "import sys; print(sys.executable)" 2>$null
        if ($p10) {
            $PythonExe = $p10.Trim()
            Write-Host "🐍 Найден Python 3.10: $PythonExe" -ForegroundColor Green
        }
    }
}

if (-not $PythonExe -and (Test-Path "C:\Python310\python.exe")) {
    $PythonExe = "C:\Python310\python.exe"
} elseif (-not $PythonExe -and (Test-Path "C:\Python311\python.exe")) {
    $PythonExe = "C:\Python311\python.exe"
}

if (-not $PythonExe -and (Get-Command python -ErrorAction SilentlyContinue)) {
    $ver = & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
    if ($ver -and [float]$ver -ge 3.13) {
        Write-Host "⚠️ Обнаружен Python $ver. Для CUDA на RTX 3060 устанавливаем Python 3.10..." -ForegroundColor Yellow
        winget install --id Python.Python.3.10 -e --source winget --accept-package-agreements --accept-source-agreements
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
        if (Get-Command py -ErrorAction SilentlyContinue) {
            $p = & py -3.10 -c "import sys; print(sys.executable)" 2>$null
            if ($p) { $PythonExe = $p.Trim() } else { $PythonExe = "python" }
        } else {
            $PythonExe = "python"
        }
    } else {
        $PythonExe = "python"
        Write-Host "🐍 Используется системный Python ($ver)!" -ForegroundColor Green
    }
}

if (-not $PythonExe) {
    Write-Host "⚠️ Python не найден. Устанавливаем Python 3.10 через winget..." -ForegroundColor Yellow
    winget install --id Python.Python.3.10 -e --source winget --accept-package-agreements --accept-source-agreements
    $PythonExe = "python"
}

# 2.2 Короткое виртуальное окружение venv (защита от WinError 206 / лимита 260 символов Windows)
$VenvDir = "$WorkDir\venv"
if (-not (Test-Path "$VenvDir\Scripts\python.exe")) {
    Write-Host "📦 Создание окружения venv (защита от длинных путей)..." -ForegroundColor Cyan
    & $PythonExe -m venv $VenvDir
}
if (Test-Path "$VenvDir\Scripts\python.exe") {
    $PythonExe = (Resolve-Path "$VenvDir\Scripts\python.exe").Path
    Write-Host "🐍 Используется изолированный Python (venv): $PythonExe" -ForegroundColor Green
}

# 2.3 Настройка PyTorch CUDA и зависимостей GPT-SoVITS
Write-Host "⚙️ Настройка зависимостей GPT-SoVITS (transformers, librosa, soundfile)..." -ForegroundColor Cyan
& $PythonExe -m pip install "numpy<2" -q
$cudaCheck = & $PythonExe -c "import torch; print(torch.cuda.is_available())" 2>$null
if ($cudaCheck -ne "True") {
    Write-Host "📦 Установка PyTorch с поддержкой CUDA для вашей RTX 3060..." -ForegroundColor Yellow
    & $PythonExe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    & $PythonExe -m pip install "numpy<2" -q
}
Write-Host "📦 Установка модулей (transformers==4.43.3, pandas==2.1.4, x-transformers)..." -ForegroundColor Cyan
& $PythonExe -m pip install -q "numpy<2" "pandas==2.1.4" "transformers==4.43.3" librosa scipy tensorboard pytorch-lightning tqdm einops rotary_embedding_torch soundfile pyyaml psutil numba x-transformers
Write-Host "✅ PyTorch CUDA и все библиотеки готовы к работе!" -ForegroundColor Green

# 3. Клонирование GPT-SoVITS если директория не содержит репозитория
if (-not (Test-Path "GPT_SoVITS")) {
    Write-Host "📦 Клонирование репозитория GPT-SoVITS..." -ForegroundColor Green
    git clone --depth=1 https://github.com/RVC-Boss/GPT-SoVITS.git .
}

# 4. Проверка базовых предобученных весов v2
$s2g = "GPT_SoVITS\pretrained_models\gsv-v2final-pretrained\s2G2333k.pth"
if (-not (Test-Path $s2g)) {
    Write-Host "📥 Базовые модели v2 не найдены. Загрузка предобученных моделей с HuggingFace..." -ForegroundColor Yellow
    Invoke-WebRequest -Uri "https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/pretrained_models.zip" -OutFile "pretrained_models.zip"
    Write-Host "📦 Распаковка архива моделей (пожалуйста, подождите 1-3 мин)..." -ForegroundColor Yellow
    if (Get-Command tar.exe -ErrorAction SilentlyContinue) {
        tar.exe -xf "pretrained_models.zip" -C "GPT_SoVITS"
    } else {
        Expand-Archive -Path "pretrained_models.zip" -DestinationPath "GPT_SoVITS" -Force
    }
    Remove-Item "pretrained_models.zip" -Force -ErrorAction SilentlyContinue
    Write-Host "✅ Базовые модели успешно загружены и распакованы!" -ForegroundColor Green
} else {
    Write-Host "✅ Базовые предобученные модели GPT-SoVITS v2 в наличии!" -ForegroundColor Green
}

# 5. Скачиваем воркер и датасет с сервера
Write-Host "📥 Загрузка файлов воркера с сервера $ServerUrl ..." -ForegroundColor Green
Invoke-WebRequest -Uri "$ServerUrl/worker.py" -OutFile "worker.py"
Invoke-WebRequest -Uri "$ServerUrl/raw_data.json" -OutFile "raw_data.json"

# 6. Запуск воркера
Write-Host ""
Write-Host "=========================================================" -ForegroundColor Green
Write-Host "  ✅ Окружение настроено! Запуск фонового воркера..." -ForegroundColor Green
Write-Host "  🌐 Панель управления доступна по адресу: $ServerUrl" -ForegroundColor Cyan
Write-Host "=========================================================" -ForegroundColor Green
Start-Process "$ServerUrl" -ErrorAction SilentlyContinue
& $PythonExe worker.py

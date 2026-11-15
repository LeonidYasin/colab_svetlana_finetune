@echo off
chcp 65001 > nul
title Веб-панель дообучения Светланы на RTX 3060

if exist runtime\python.exe (
    set PYTHON_EXEC=runtime\python.exe
) else (
    if exist runtime\bin\python.exe (
        set PYTHON_EXEC=runtime\bin\python.exe
    ) else (
        set PYTHON_EXEC=python
    )
)

echo =======================================================
echo   🌐 Запуск веб-панели дообучения Светланы на RTX 3060
echo =======================================================
echo Интерпретатор: %PYTHON_EXEC%
echo Открытие браузера по адресу http://localhost:7860 ...

%PYTHON_EXEC% svetlana_trainer_web.py
pause

@echo off
chcp 65001 > nul
title Дообучение голоса Светланы на RTX 3060

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
echo   🎙️ Запуск дообучения голоса Светланы на RTX 3060
echo =======================================================
echo Интерпретатор: %PYTHON_EXEC%

%PYTHON_EXEC% train_svetlana_windows.py

echo.
echo =======================================================
echo   🎉 Дообучение завершено! Нажмите любую клавишу...
echo =======================================================
pause

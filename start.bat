@echo off
chcp 65001 >nul
title Vecrafter - 矢量艺术字工坊
echo ========================================
echo   Vecrafter - 矢量艺术字工坊
echo   正在启动服务...
echo ========================================
echo.

:: 检测 Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未检测到 Python，请安装 Python 3.10+
    pause
    exit /b 1
)

:: 检测 ComfyUI（仅提示，不阻塞）
echo [检测] 检查 ComfyUI 连接...
python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8188', timeout=2)" >nul 2>&1
if %errorlevel% equ 0 (
    echo [检测] ComfyUI ✅ 本地 127.0.0.1:8188
) else (
    python -c "import urllib.request; urllib.request.urlopen('http://10.195.155.46:8188', timeout=2)" >nul 2>&1
    if %errorlevel% equ 0 (
        echo [检测] ComfyUI ✅ 远程 10.195.155.46:8188
    ) else (
        echo [警告] ComfyUI ❌ 未连接到 ComfyUI（生成功能不可用）
    )
)

echo.
echo [启动] FastAPI 后端...
start "Vecrafter Backend" /B python back_end/main.py
timeout /t 3 /nobreak >nul

echo [启动] Streamlit 前端...
python -m streamlit run front_end/Vecrafter.py --server.port 8501

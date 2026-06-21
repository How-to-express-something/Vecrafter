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

:: 绕过系统代理（ComfyUI 在本地/LAN，不应走代理）
set "NO_PROXY=127.0.0.1,localhost,::1,%NO_PROXY%"

:: 自动启动 ComfyUI（如果设置了 COMFYUI_HOME）
if defined COMFYUI_HOME (
    echo [启动] 自动拉起 ComfyUI（路径: %COMFYUI_HOME%）...
    start "ComfyUI" /B python "%COMFYUI_HOME%\main.py" >nul 2>&1
    echo [等待] 等待 ComfyUI 就绪（最长 120s）...
    for /l %%i in (1,1,120) do (
        python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8188', timeout=1)" >nul 2>&1
        if not errorlevel 1 goto :comfyui_ready
        timeout /t 1 /nobreak >nul
    )
    echo [警告] ComfyUI 启动超时（120s），请手动检查
    goto :comfyui_done
    :comfyui_ready
    echo [启动] ComfyUI ✅ 已就绪
    :comfyui_done
    echo.
)

:: 检测 ComfyUI（仅提示，不阻塞）
echo [检测] 检查 ComfyUI 连接...
if not defined COMFYUI_URL set "COMFYUI_URL=http://127.0.0.1:8188"
python -c "import urllib.request; urllib.request.urlopen('%COMFYUI_URL%', timeout=2)" >nul 2>&1
if %errorlevel% equ 0 (
    echo [检测] ComfyUI ✅ %COMFYUI_URL%
) else (
    echo [警告] ComfyUI ❌ 无法连接 %COMFYUI_URL%
    echo        可通过 set COMFYUI_URL=http://你的IP:8188 修改地址
)

echo.
echo [启动] FastAPI 后端...
start "Vecrafter Backend" /B python back_end/main.py
timeout /t 3 /nobreak >nul

echo [启动] Streamlit 前端...
python -m streamlit run front_end/Vecrafter.py --server.port 8501

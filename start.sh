#!/usr/bin/env bash
# Vecrafter 启动脚本 (Linux/macOS + WSL + Git Bash)
set -e

echo "========================================"
echo "  Vecrafter - 矢量艺术字工坊"
echo "  正在启动服务..."
echo "========================================"
echo ""

# ---- 自动检测 Python 命令（python3 或 python） ----
PYTHON=""
if command -v python3 &>/dev/null; then
    PYTHON="python3"
elif command -v python &>/dev/null; then
    PYTHON="python"
else
    echo "[错误] 未检测到 Python（python3 / python 均不可用）"
    exit 1
fi
echo "[检测] Python: $($PYTHON --version 2>&1)"

# ---- 绕过系统代理（ComfyUI 在本地/LAN，不应走代理） ----
# 确保内联 Python URL 检测不受 HTTP_PROXY 干扰
export NO_PROXY="127.0.0.1,localhost,::1,${NO_PROXY}"

# ---- 自动启动 ComfyUI（如果设置了 COMFYUI_HOME） ----
COMFYUI_PID=""
if [ -n "${COMFYUI_HOME}" ]; then
    echo "[启动] 自动拉起 ComfyUI（路径: ${COMFYUI_HOME}）..."
    $PYTHON "${COMFYUI_HOME}/main.py" &
    COMFYUI_PID=$!
    echo "[等待] 等待 ComfyUI 就绪（最长 120s）..."
    for i in $(seq 1 120); do
        if $PYTHON -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8188', timeout=1)" 2>/dev/null; then
            echo "[启动] ComfyUI ✅ 已就绪"
            break
        fi
        if [ "$i" -eq 120 ]; then
            echo "[警告] ComfyUI 启动超时（120s），请手动检查"
        fi
        sleep 1
    done
    echo ""
fi

# ---- 检测 ComfyUI 连通性 ----
echo "[检测] 检查 ComfyUI..."
COMFYUI_URL="${COMFYUI_URL:-http://127.0.0.1:8188}"
if $PYTHON -c "import urllib.request; urllib.request.urlopen('${COMFYUI_URL}', timeout=2)" 2>/dev/null; then
    echo "[检测] ComfyUI ✅ ${COMFYUI_URL}"
else
    echo "[警告] ComfyUI ❌ 无法连接 ${COMFYUI_URL}"
    echo "       可通过 export COMFYUI_URL=http://你的IP:8188 修改地址"
fi

echo ""
echo "[启动] FastAPI 后端..."
$PYTHON back_end/main.py &
BACKEND_PID=$!
echo "  PID: $BACKEND_PID"
sleep 3

echo ""
echo "[启动] Streamlit 前端..."
$PYTHON -m streamlit run front_end/Vecrafter.py --server.port 8501

# ---- 清理 ----
kill $BACKEND_PID 2>/dev/null || true
if [ -n "${COMFYUI_PID}" ]; then
    kill $COMFYUI_PID 2>/dev/null || true
    echo "[清理] 已停止 ComfyUI (PID: $COMFYUI_PID)"
fi
echo "服务已停止"

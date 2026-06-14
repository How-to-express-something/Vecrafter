#!/usr/bin/env bash
# Vecrafter 启动脚本 (Linux/macOS + WSL)
set -e

echo "========================================"
echo "  Vecrafter - 矢量艺术字工坊"
echo "  正在启动服务..."
echo "========================================"
echo ""

# 检测 Python
if ! command -v python3 &>/dev/null; then
    echo "[错误] 未检测到 Python3"
    exit 1
fi

# 检测 ComfyUI
echo "[检测] 检查 ComfyUI..."
if python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8188', timeout=2)" 2>/dev/null; then
    echo "[检测] ComfyUI ✅ 本地 127.0.0.1:8188"
elif python3 -c "import urllib.request; urllib.request.urlopen('http://10.195.155.46:8188', timeout=2)" 2>/dev/null; then
    echo "[检测] ComfyUI ✅ 远程 10.195.155.46:8188"
else
    echo "[警告] ComfyUI ❌ 未连接"
fi

echo ""
echo "[启动] FastAPI 后端..."
python3 back_end/main.py &
BACKEND_PID=$!
echo "  PID: $BACKEND_PID"
sleep 3

echo ""
echo "[启动] Streamlit 前端..."
streamlit run front_end/Vecrafter.py --server.port 8501

# 清理
kill $BACKEND_PID 2>/dev/null
echo "服务已停止"

# ComfyUI 连接问题修复

## 根因

| 问题 | 说明 |
|---|---|
| **硬编码 IP** | `back_end/main.py` 写死了旧内网 IP `10.195.155.46:8188` |
| **系统代理拦截** | `HTTP_PROXY=http://127.0.0.1:7890`，Python `requests` 走代理访问 `127.0.0.1` 返回 502 |

两个问题叠加 → ComfyUI 始终不可达。

## 修复内容

| 文件 | 改动 |
|---|---|
| `back_end/main.py` | `ComfyUIWrapper` 默认 `127.0.0.1:8188`，支持 `COMFYUI_URL` 环境变量；创建 `Session(trust_env=False)` 绕过代理 |
| `back_end/comfyui_wrapper.py` | 同上 |
| `cli.py` | 全局 `_req_session(trust_env=False)` 绕过代理；`check-env` 检测代理并告警 |
| `start.bat` / `start.sh` | 移除硬编码 IP，改用 `COMFYUI_URL` 环境变量 |

## 使用

```powershell
# 默认本地 ComfyUI（无需任何设置）
python cli.py check-env          # 检查环境
python back_end/main.py          # 启动后端
python cli.py generate --text "青山集" --style "国风书法" --seed 42

# 如果 ComfyUI 在其他机器
$env:COMFYUI_URL = "http://192.168.1.100:8188"
```

## 如果仍有问题

`check-env` 会检测系统代理并提示。若代理干扰本地连接：

```powershell
# 临时方案：设置 NO_PROXY
$env:NO_PROXY = "127.0.0.1,localhost"
```

## 验证

```powershell
python cli.py check-env
# 预期输出：
#   配置地址: http://127.0.0.1:8188
#   连通性: [OK] 可达
```

---

## 补充修复（2026-06-21）：启动脚本增强

### 发现的问题

| # | 问题 | 影响 |
|---|------|------|
| 1 | `start.sh` 硬编码 `python3`，Windows Git Bash 上只有 `python` | 脚本直接报「未检测到 Python3」退出，后端/前端均不启动 |
| 2 | `start.sh` / `start.bat` 内联 Python URL 检测未绕过代理 | 有系统代理时 `127.0.0.1:8188` 检测失败（502），误报 ComfyUI 不可达 |
| 3 | `start.sh` 用裸 `streamlit` 命令，Windows 上不在 PATH 中 | 前端无法启动 |

### 修复内容

| 文件 | 改动 |
|------|------|
| `start.sh` | 自动检测 `python3` → 回退 `python`；设置 `NO_PROXY=127.0.0.1,localhost` 绕过代理；`streamlit` → `$PYTHON -m streamlit`；`kill` 加 `|| true` 容错 |
| `start.bat` | 新增 `set "NO_PROXY=127.0.0.1,localhost,::1,%NO_PROXY%"` 绕过代理 |

### 排查流程

如果前端仍无法生成图片，按顺序检查：

```
1. ComfyUI 是否在运行？
   → 浏览器打开 http://127.0.0.1:8188，应看到 ComfyUI WebUI

2. 是否有系统代理拦截？
   → echo %HTTP_PROXY%  (Windows) / echo $HTTP_PROXY  (Linux)
   → 如果有输出，检查是否误将 127.0.0.1 走了代理

3. 后端日志
   → 查看 logs/backend.log，搜索 ERROR 或 ComfyUI 相关错误

4. 用 CLI 直接测试
   → python cli.py check-env
   → python cli.py generate --text "测试" --seed 42
```

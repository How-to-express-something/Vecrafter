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

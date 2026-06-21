# Vecrafter 架构与启动流程

## 整体架构

```
┌──────────────┐     HTTP(8000)     ┌──────────────┐   HTTP+WS(8188)   ┌──────────────┐
│  前端/CLI     │ ─────────────────→│  Vecrafter   │ ────────────────→│   ComfyUI    │
│              │                   │   Backend    │                   │  (Stable     │
│  Streamlit   │ ←── JSON/base64 ──│  (FastAPI)   │ ←── PNG ─────────│  Diffusion)  │
│  CLI (cli.py)│                   │              │                   │              │
└──────────────┘                   └──────────────┘                   └──────────────┘
   用户界面                          业务逻辑层                          AI 推理引擎
   :8501                             :8000                              :8188
```

## ComfyUI 通信方式

ComfyUI 是一个独立的 Stable Diffusion 推理服务，提供 **HTTP + WebSocket** API：

| 协议 | 用途 | 示例 |
|---|---|---|
| **HTTP POST** | 提交生成任务 | `POST /prompt` → 传入工作流 JSON |
| **WebSocket** | 实时监听进度 | 连接 `/ws?clientId=xxx`，接收 `executing`/`execution_error` 事件 |
| **HTTP GET** | 拉取结果 | `GET /history/{prompt_id}` → 获取生成完成的图片列表 |
| **HTTP GET** | 下载图片 | `GET /view?filename=xxx` → 下载 PNG |

### 一次生成的完整流程

```
Vecrafter Backend                         ComfyUI
      │                                      │
      │── POST /prompt (工作流JSON)──────────→│  ① 提交任务，获得 prompt_id
      │                                      │
      │── WS /ws?clientId=xxx ──────────────→│  ② 建立 WebSocket 监听
      │   ←── {"type":"executing","node":"4"}│     实时回调当前执行节点
      │   ←── {"type":"executing","node":null}│    node=null → 生成完成
      │                                      │
      │── GET /history/{prompt_id} ─────────→│  ③ 获取输出信息（文件名列表）
      │                                      │
      │── GET /view?filename=xxx ───────────→│  ④ 逐张下载生成的 PNG
      │                                      │
```

关键代码在 `back_end/main.py` 的 `ComfyUIWrapper` 类中（`queue_prompt` → `wait_for_prompt` → `get_output_images`）。

---

## 当前启动流程（需要手动 2 步）

```powershell
# 步骤 1：用户手动启动 ComfyUI
#   → 进入 ComfyUI 安装目录，运行 python main.py
#   → 或运行 ComfyUI 的 run_nvidia_gpu.bat

# 步骤 2：启动 Vecrafter
.\start.bat
# 或
python back_end/main.py          # 后端 :8000
python -m streamlit run front_end/Vecrafter.py  # 前端 :8501
```

`start.bat` 仅**检测** ComfyUI 是否在线，不会自动启动它。

---

## 为什么不能自动启动 ComfyUI？

| 障碍 | 说明 |
|---|---|
| **路径未知** | ComfyUI 安装位置因人而异（可能放 D 盘、C 盘、便携版等） |
| **依赖环境** | ComfyUI 需要独立的 Python 环境 + PyTorch + 自定义节点 |
| **模型文件大** | GGUF 模型动辄 5-10GB，必须提前下载好 |
| **启动慢** | 加载模型需要 30 秒 ~ 数分钟 |

---

## 可行的改进方向

### 方案 A：环境变量指定 ComfyUI 路径（推荐，改动最小）

新增 `COMFYUI_HOME` 环境变量，Vecrafter 启动时自动拉起 ComfyUI：

```powershell
# 用户一次性配置
$env:COMFYUI_HOME = "C:\ComfyUI_windows_portable"

# 之后 Vecrafter 直接一条命令启动
.\start.bat  # 自动启动 ComfyUI → 启动后端 → 启动前端
```

代码改动：
```python
# 在 start.bat / main.py 启动时
comfyui_home = os.environ.get("COMFYUI_HOME")
if comfyui_home:
    subprocess.Popen([sys.executable, f"{comfyui_home}/main.py"])
    wait_for_comfyui_ready("http://127.0.0.1:8188", timeout=120)
```

### 方案 B：内置轻量推理（大改动，不推荐）

把 Stable Diffusion 推理直接嵌入 Vecrafter 后端（省去 ComfyUI 进程），但需要捆绑 PyTorch + 模型，安装包会膨胀到 10GB+，维护成本极高。

### 方案 C：Docker Compose 一键部署

```yaml
# docker-compose.yml
services:
  comfyui:
    image: comfyui/comfyui  # 需要自制镜像
    ports: ["8188:8188"]
  vecrafter:
    build: .
    ports: ["8000:8000", "8501:8501"]
    environment:
      - COMFYUI_URL=http://comfyui:8188
```

用户只需 `docker compose up`，但需要用户装了 Docker。

---

## 当前状态总结

| 场景 | 状态 |
|---|---|
| 本地开发（ComfyUI 已安装） | ✅ `python cli.py check-env` → `python back_end/main.py` |
| 换一台新电脑 | ⚠️ 需先装 ComfyUI + 模型，再装 Vecrafter |
| 给别人部署 | ⚠️ 需要部署文档列出 ComfyUI 依赖 |
| 代理环境 | ✅ 已自动绕过，无需额外配置 |

**核心结论**：ComfyUI 是一个独立的推理引擎，必须单独安装和启动。我们的代码已经做了最大化兼容（绕过代理 + 可配置地址），但无法跳过"用户需要装 ComfyUI"这一步。如果想简化，推荐**方案 A**（`COMFYUI_HOME` 自动拉起），改造成本最低。

# Vecrafter — 矢量艺术字生成应用

基于开源文生图模型（ComfyUI + z-image）的矢量艺术字生成系统。输入文字和风格描述，自动完成 **AI 生成 → 预处理 → 矢量化 → 结果打包** 全流程，输出可直接交付生产的 SVG 矢量文件。

---

## 快速开始

### 安装

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 检测环境（确认一切就绪）
python cli.py check-env
```

### 启动（任选一种）

**方式一：交互式 CLI（推荐）**
```bash
vecrafter        # Windows: 双击 vecrafter.bat
                  # 进入交互命令行

vecrafter> check-env          # 检测环境
vecrafter> generate 青山集    # 生成艺术字
vecrafter> status             # 查看后端
vecrafter> exit               # 退出
```

**方式二：一键启动（Web 界面）**
```bash
start.bat                     # Windows
./start.sh                    # Linux/macOS
```
浏览器打开 http://localhost:8501

**方式三：手动分开启动**
```bash
# 终端 1：启动后端
python back_end/main.py

# 终端 2：启动前端
streamlit run front_end/Vecrafter.py --server.port 8501
```

---

## CLI 命令参考

| 命令 | 用途 | 示例 |
|------|------|------|
| `generate <text>` | 生成艺术字 | `generate 青山集 --style 国风书法` |
| `vectorize <path>` | 矢量化图片 | `vectorize input.png` |
| `batch <file>` | 批量生成 | `batch prompts.csv` |
| `check-env` | 环境检测 | `check-env` |
| `status` | 后端状态 | `status` |

单次命令模式：
```bash
python cli.py generate --text "青山集" --style "国风书法" --seed 42
python cli.py vectorize --input result.png --output result.svg
python cli.py batch --file prompts.csv --output ./batch_output
```

### 批量文件格式

**prompts.csv**
```csv
text,style,seed
青山集,国风书法,42
Hello,促销卡通,123
2024,海洋浪漫,
```

**prompts.json**
```json
[{"text": "青山集", "style": "国风书法", "seed": 42}]
```

**prompts.txt**
```
青山集
Hello World
2024
```

---

## 系统架构

```
┌──────────┐     HTTP     ┌──────────┐    WS/HTTP    ┌──────────┐
│ Streamlit │ ──────────→ │ FastAPI  │ ────────────→ │ ComfyUI  │
│ (:8501)   │ ←────────── │ (:8000)  │ ←──────────── │ (:8188)  │
└──────────┘              └────┬─────┘               └──────────┘
                               │
                    ┌──────────┼──────────┐
                    ▼          ▼          ▼
            Prompt       Image       Vector
            Preprocessor Preprocessor Converter
```

### 核心模块

| 模块 | 文件 | 功能 |
|------|------|------|
| **后端 API** | `back_end/main.py` | FastAPI 服务，编排生成/预处理/矢量化全流程 |
| **ComfyUI 封装** | `back_end/comfyui_wrapper.py` | WebSocket+HTTP 双通道通信，工作流提交与结果轮询 |
| **提示词预处理** | `back_end/prompt_preprocessor.py` | 文本类型感知包装（中文/英文/数字/混合 7 类），CFG 推荐，负面词扩展 |
| **图像预处理** | `back_end/image_preprocessor.py` | 背景分离、主体裁剪、边缘去噪、颜色量化、抗锯齿 |
| **矢量转换** | `back_end/vector_converter.py` | OpenCV 轮廓追踪 + 全局 K-Means 调色板 + RETR_TREE 层级渲染 |
| **前端** | `front_end/Vecrafter.py` | Streamlit UI，三种操作模式，历史管理，风格配置 |
| **CLI** | `cli.py` | 交互式命令行 + 单次命令双模式 |
| **测试** | `scripts/test_runner.py` | 162 条自动化测试套件 |

---

## 输出文件

每次生成在 `output/YYYY-MM-DD/HHMMSS_seed_textslug/` 下：

```
├── original.png        # 原始生成图（ComfyUI 输出）
├── preview.png         # 预览图
├── transparent.png     # 透明背景 PNG（自动去背景）
├── result.svg          # SVG 矢量图（可编辑）
├── metadata.json       # 完整元数据（含模型版本、工作流版本）
└── run.log             # 本次运行日志
```

六件套全部自动打包，日志记录了每步的参数和耗时。

### 矢量化质量指标

| 指标 | 数据 |
|------|------|
| 1024×1024 单图耗时 | **0.07–0.15s**（赛题要求 ≤10s） |
| 端到端生成+矢量化 | **~52s**（赛题要求 ≤90s） |
| 测试集通过率 | **160/162 (98.8%)** |
| SVG 包含 | xmlns + viewBox + 贝塞尔 C 曲线 + 闭合 Z 路径 |
| 孔洞渲染 | evenodd 填充，O/P 等字母内部正确镂空 |
| 抗桥接 | erode+per-label dilate，字符独立不粘连 |

---

## 环境检测

```bash
python cli.py check-env
```

自动检查：Python 版本、9 个核心依赖包、ComfyUI 连通性（本地/远程）、GPU 可用性、配置文件完整性。未安装的依赖会明确提示。

### ComfyUI 工作流

默认配置文件：`config/stableOutput_26_6_7.json`（推荐）

工作流节点：UnetLoader → CLIPLoader → LoRA Stack → ModelPatchLoader → CharAutoStyle → ControlNet → KSampler → VAE Decode → Rembg → SaveImage

**模型/节点依赖：**
- [z-image Turbo GGUF](https://huggingface.co/)（Unet + CLIP + VAE）
- [Z-Image-Turbo-Fun-Controlnet-Union](https://huggingface.co/)（ControlNet）
- [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager)（插件管理）
- rgthree's ComfyUI Nodes（LoRA Loader Stack）

---

## 项目结构

```
Vecrafter/
├── back_end/
│   ├── main.py                    # FastAPI 服务入口
│   ├── comfyui_wrapper.py         # ComfyUI API 封装
│   ├── prompt_preprocessor.py     # 提示词预处理
│   ├── image_preprocessor.py      # 图像预处理
│   └── vector_converter.py        # 矢量转换引擎
├── front_end/
│   └── Vecrafter.py               # Streamlit 前端
├── config/
│   ├── stableOutput_26_6_7.json   # ComfyUI 工作流（推荐）
│   ├── updated26_6_7.json         # ComfyUI 工作流（新版）
│   └── CFG_test.json              # ComfyUI 工作流（旧版）
├── scripts/
│   └── test_runner.py             # 162 条自动化测试
├── docs/
│   ├── 测试用例.xlsx              # 110 条标准测试集
│   └── 4.测试报告.docx            # 系统测试报告
├── cli.py                         # 交互式 + 单次 CLI
├── vecrafter.bat                  # Windows 快捷入口
├── start.bat                      # Windows 一键启动
├── start.sh                       # Linux/macOS 启动
├── requirements.txt               # Python 依赖清单
└── README.md                      # 本文件
```

---

## 交付物清单

| 交付项 | 状态 | 说明 |
|--------|------|------|
| 应用源代码 | ✅ | `back_end/` + `front_end/` + `cli.py` |
| CLI 命令行 | ✅ | `cli.py` + `vecrafter.bat` |
| Web 界面 | ✅ | `front_end/Vecrafter.py`（Streamlit） |
| ComfyUI 工作流 | ✅ | `config/stableOutput_26_6_7.json` |
| 启动脚本 | ✅ | `start.bat`（Win）/ `start.sh`（Linux） |
| 环境检测 | ✅ | `cli.py check-env` |
| 安装说明 | ✅ | 本文档 |
| 测试数据 | ✅ | `docs/测试用例.xlsx`（110 条）|
| 验收脚本 | ✅ | `scripts/test_runner.py`（162 条）|
| 输出样例 | ✅ | `output/2026-06-07/` 含完整六件套 |
| 运行日志 | ✅ | `logs/backend.log` + `logs/frontend.log` |

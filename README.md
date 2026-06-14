# Vecrafter - 矢量艺术字生成应用

基于开源文生图模型的矢量艺术字生成系统，支持：

- **生成艺术字** — 输入文字和风格，ComfyUI 生成艺术字图像
- **矢量化（SVG 转换）** — 将艺术字 PNG 转为可编辑 SVG 矢量图
- **批量处理** — CSV/JSON/TXT 提示词清单批量生成
- **结果管理** — 透明 PNG、SVG、元数据、运行日志

---

## 系统架构

```
前端 (Streamlit :8501)  →  后端 (FastAPI :8000)  →  ComfyUI (:8188)
                              │
                              ├─ 提示词预处理 (prompt_preprocessor.py)
                              ├─ 图像预处理 (image_preprocessor.py)
                              └─ 矢量转换 (vector_converter.py)
```

---

## 快速开始

### 1. 环境要求

| 依赖 | 版本要求 |
|------|----------|
| Python | >= 3.10 |
| ComfyUI | 本地或远程 (:8188) |
| NVIDIA GPU | 可选（CPU 也可运行但生成慢） |

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 启动服务

**Windows:**
```bash
start.bat
```

**Linux/macOS/WSL:**
```bash
chmod +x start.sh
./start.sh
```

或手动启动：

```bash
# 1. 启动后端
python back_end/main.py

# 2. 另开终端，启动前端
streamlit run front_end/Vecrafter.py --server.port 8501
```

### 4. 访问

打开浏览器访问: http://localhost:8501

---

## CLI 命令行使用

```bash
# 环境检测
python cli.py check-env

# 单条生成
python cli.py generate --text "青山集" --style "国风书法" --output ./output

# 矢量化已有图片
python cli.py vectorize --input input.png --output result.svg

# 批量生成（从 CSV/JSON/TXT）
python cli.py batch --file prompts.csv --output ./batch_output
```

### 批量文件格式

**prompts.csv:**
```csv
text,style,seed
青山集,国风书法,42
Hello World,促销卡通,123
2024,海洋浪漫,
```

**prompts.json:**
```json
[
  {"text": "青山集", "style": "国风书法", "seed": 42},
  {"text": "Hello"}
]
```

**prompts.txt:**
```
青山集
Hello World
2024
```

---

## 输出文件结构

每次生成在 `output/` 下自动建立目录：
```
output/
└── YYYY-MM-DD/
    └── HHMMSS_seed_textslug/
        ├── original.png        # 原始生成图
        ├── preview.png         # 预览图
        ├── transparent.png     # 透明背景 PNG
        ├── result.svg          # SVG 矢量图
        ├── metadata.json       # 完整元数据(含模型版本)
        └── run.log             # 运行日志
```

---

## 项目结构

```
├── back_end/                   # 后端
│   ├── main.py                 # FastAPI 服务入口
│   ├── comfyui_wrapper.py      # ComfyUI API 封装
│   ├── prompt_preprocessor.py  # 提示词预处理
│   ├── image_preprocessor.py   # 图像预处理
│   └── vector_converter.py     # 矢量转换引擎
│
├── front_end/
│   └── Vecrafter.py            # Streamlit 前端
│
├── config/
│   ├── CFG_test.json           # ComfyUI 工作流(旧)
│   ├── updated26_6_7.json      # ComfyUI 工作流(新版1)
│   └── stableOutput_26_6_7.json# ComfyUI 工作流(新版2,推荐)
│
├── scripts/
│   └── test_runner.py          # 自动化测试脚本
│
├── cli.py                      # CLI 命令行入口
├── start.bat                   # Windows 启动脚本
├── start.sh                    # Linux 启动脚本
├── requirements.txt            # Python 依赖
└── README.md                   # 本文件
```

---

## 环境检测

```bash
python cli.py check-env
```

检查项：Python 版本、核心依赖包、ComfyUI 连通性、GPU 可用性。

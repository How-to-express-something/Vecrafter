# app.py
# 运行: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import zipfile
import time
import json
import base64
import re
import requests
import logging
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Dict, Any, List

# ======================= 页面配置 =======================
st.set_page_config(
    page_title="Vecrafter | 矢量艺术字工坊",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================= 自定义样式 =======================
st.markdown("""
<style>
    .main {
        background: linear-gradient(145deg, #f0f4f9 0%, #e9eff5 100%);
    }
    .big-input textarea {
        font-size: 1.2rem !important;
        padding: 1rem !important;
        border-radius: 28px !important;
        border: 1px solid #cde0ea;
        background: white;
    }
    .right-sidebar {
        background: rgba(255,255,255,0.7);
        backdrop-filter: blur(4px);
        border-radius: 28px;
        padding: 1rem;
        height: 100%;
        overflow-y: auto;
    }
    .result-card {
        background: white;
        border-radius: 20px;
        padding: 0.8rem;
        margin-bottom: 1rem;
        border-left: 5px solid #1e6d7e;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .sidebar-btn {
        width: 100%;
        margin-bottom: 0.5rem;
        border-radius: 40px;
    }
    .art-title {
        text-align: center;
        margin-bottom: 1.2rem;
    }
    .input-card {
        background: white;
        border-radius: 28px;
        padding: 1.5rem 1.5rem 1rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        margin: 1rem 0 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ======================= 艺术字标题 SVG（仅 Vecrafter，无副标题） =======================
def render_art_title():
    """生成 Vecrafter 艺术字 SVG（无副标题）"""
    svg_title = '''
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 60" width="100%" height="70">
        <defs>
            <linearGradient id="titleGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stop-color="#1c5f6e"/>
                <stop offset="50%" stop-color="#32849e"/>
                <stop offset="100%" stop-color="#5ba0b5"/>
            </linearGradient>
            <filter id="shadow" x="-5%" y="-5%" width="110%" height="110%">
                <feDropShadow dx="2" dy="2" stdDeviation="2" flood-color="#1c5f6e" flood-opacity="0.3"/>
            </filter>
        </defs>
        <text x="250" y="42" font-family="'Segoe UI', 'Inter', 'Poppins', sans-serif" font-size="40" 
              font-weight="800" fill="url(#titleGrad)" text-anchor="middle" filter="url(#shadow)"
              letter-spacing="4">
            Vecrafter
        </text>
    </svg>
    '''
    return svg_title


BACKEND_URL = "http://127.0.0.1:8000"

# ======================= 前端持久化日志 =======================
_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

_frontend_logger = logging.getLogger("vecrafter.frontend")
_frontend_logger.setLevel(logging.DEBUG)
_fh = RotatingFileHandler(
    _LOG_DIR / "frontend.log",
    maxBytes=5 * 1024 * 1024,
    backupCount=3,
    encoding="utf-8",
)
_fh.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))
_frontend_logger.addHandler(_fh)


def _make_slug(text: str, max_len: int = 20) -> str:
    """将文本转为文件系统安全的 ASCII slug"""
    safe = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    slug = re.sub(r'\s+', '_', safe.strip())
    return slug[:max_len] if slug else "art"


def _make_download_name(meta: dict, idx: int) -> str:
    """根据元数据生成有意义的下载文件名"""
    if meta:
        seed = meta.get("seed", idx)
        text = meta.get("text", "art")
        slug = _make_slug(text)
        return f"vecrafter_{slug}_{seed}.png"
    return f"art_{idx}.png"


# ======================= 后端 API =======================
class BackendAPI:
    @staticmethod
    def generate_art(text: str, style_prompt: str, negative_prompt: str,
                     seed: int, resolution: str, vector_params: Dict) -> Dict:
        try:
            w, h = resolution.split("x")
            payload = {
                "text": text,
                "style_prompt": style_prompt,
                "negative_prompt": negative_prompt,
                "seed": int(seed),
                "width": int(w),
                "height": int(h),
                "steps": 8,
                "cfg": 1.0,
                "sampler_name": "res_multistep",
                "scheduler": "simple",
            }
            resp = requests.post(
                f"{BACKEND_URL}/generate",
                json=payload,
                timeout=900
            )
            if resp.status_code != 200:
                add_log(f"后端返回错误: {resp.status_code}", level="ERROR")
                return {"success": False, "error_msg": f"Backend error: {resp.text}"}

            data = resp.json()
            images_b64 = data.get("images", [])
            if not images_b64:
                add_log("后端未返回图片", level="ERROR")
                return {"success": False, "error_msg": "No images returned"}

            png_bytes = base64.b64decode(images_b64[0])

            # 使用后端返回的真实元数据 + 输出文件路径
            metadata = data.get("metadata", {})
            for key in ("preview_path", "metadata_path", "original_path",
                        "transparent_path", "svg_path"):
                if data.get(key):
                    metadata[key] = data[key]

            return {"success": True, "png_bytes": png_bytes, "metadata": metadata}
        except requests.exceptions.ConnectionError:
            add_log("无法连接后端服务", level="ERROR")
            return {"success": False, "error_msg": "无法连接后端服务，请确保后端已启动 (python back_end/main.py)"}
        except Exception as e:
            add_log(f"生成异常: {e}", level="ERROR")
            return {"success": False, "error_msg": str(e)}
    
    @staticmethod
    def preprocess_image(image_bytes: bytes, params: Dict) -> Dict:
        """调用后端 /preprocess 端点进行图像预处理"""
        try:
            payload = {
                "image_b64": base64.b64encode(image_bytes).decode(),
                "aspect_ratio": params.get("aspect_ratio", "1:1"),
                "target_width": params.get("target_width", 1024),
                "target_height": params.get("target_height", 1024),
                "resize_mode": params.get("resize_mode", "fit"),
                "remove_background": params.get("remove_background", True),
                "edge_denoise": params.get("edge_denoise", True),
                "subject_crop": params.get("subject_crop", True),
                "crop_padding": params.get("crop_padding", 16),
                "color_quantize": params.get("color_quantize", False),
                "quantize_colors": params.get("quantize_colors", 256),
                "anti_alias": params.get("anti_alias", True),
                "output_format": params.get("output_format", "png_rgba"),
            }
            resp = requests.post(
                f"{BACKEND_URL}/preprocess",
                json=payload,
                timeout=600,
            )
            if resp.status_code != 200:
                add_log(f"预处理失败: {resp.status_code}", level="ERROR")
                return {"success": False, "error_msg": f"Preprocess error: {resp.text}"}
            data = resp.json()
            png_bytes = base64.b64decode(data["image_b64"]) if data.get("image_b64") else None
            return {
                "success": True,
                "png_bytes": png_bytes,
                "original_size": data.get("original_size"),
                "output_size": data.get("output_size"),
                "bbox": data.get("bbox"),
            }
        except requests.exceptions.ConnectionError:
            add_log("无法连接后端服务", level="ERROR")
            return {"success": False, "error_msg": "无法连接后端服务"}
        except Exception as e:
            add_log(f"预处理异常: {e}", level="ERROR")
            return {"success": False, "error_msg": str(e)}

    @staticmethod
    def vectorize_image(image_bytes: bytes, params: Dict) -> Dict:
        """调用后端 /vectorize 端点进行艺术字矢量化"""
        try:
            payload = {
                "image_b64": base64.b64encode(image_bytes).decode(),
                "color_clusters": params.get("color_k", 8),
                "smooth_threshold": params.get("smooth", 1.2),
                "min_region_area": params.get("min_region_area", 16),
                "path_precision": params.get("path_precision", 0.5),
                "preserve_gradient": params.get("preserve_gradient", True),
                "preserve_shadow": params.get("preserve_shadow", True),
                "embed_preview": params.get("embed_preview", True),
                "output_preview_png": params.get("output_preview_png", False),
            }
            resp = requests.post(
                f"{BACKEND_URL}/vectorize",
                json=payload,
                timeout=600,
            )
            if resp.status_code != 200:
                add_log(f"矢量化失败: {resp.status_code}", level="ERROR")
                return {"success": False, "error_msg": f"Vectorize error: {resp.text}"}
            data = resp.json()
            result = {
                "success": True,
                "svg_str": data.get("svg_string"),
                "total_paths": data.get("total_paths", 0),
                "color_layer_count": data.get("color_layer_count", 0),
                "region_type_counts": data.get("region_type_counts"),
                "warnings": data.get("warnings"),
            }
            if data.get("preview_b64"):
                result["preview_bytes"] = base64.b64decode(data["preview_b64"])
            return result
        except requests.exceptions.ConnectionError:
            add_log("无法连接后端服务", level="ERROR")
            return {"success": False, "error_msg": "无法连接后端服务"}
        except Exception as e:
            add_log(f"矢量化异常: {e}", level="ERROR")
            return {"success": False, "error_msg": str(e)}

# ======================= 历史记录渲染 =======================

def _render_history_item(item: dict, idx: int, location: str = "main"):
    """渲染单条历史记录条目（主区域 / 侧边栏共用）

    Args:
        item: 历史记录条目
        idx: 在 history 列表中的索引（全局索引，非位置索引）
        location: "main"（右侧主区域）或 "sidebar"（侧边栏），用于生成唯一 key
    """
    prefix = f"{location}_{idx}"
    in_sidebar = location == "sidebar"
    with st.container():
        st.markdown(f"**{item['type']}** · {item['title'][:30]}")
        st.caption(f"🕒 {item['time']}")
        data = item['data']
        if "metadata" in data:
            meta = data["metadata"]
            with st.expander("📋 参数详情", expanded=False):
                st.caption(f"种子: {meta.get('seed', 'N/A')}")
                gen_time = meta.get('generation_time_seconds')
                if gen_time is not None:
                    st.caption(f"生成耗时: {gen_time}s")
                st.caption(f"采样: {meta.get('sampler_name', 'N/A')} / {meta.get('scheduler', 'N/A')}")
                st.caption(f"步数: {meta.get('steps', 'N/A')}  CFG: {meta.get('cfg', 'N/A')}")
                if meta.get('prompt_id'):
                    st.caption(f"Prompt ID: {meta['prompt_id']}")
        if data.get("png_bytes"):
            st.image(data["png_bytes"], width=120 if in_sidebar else 180)
            meta = data.get("metadata", {})
            filename = _make_download_name(meta, idx)
            st.download_button("⬇️ PNG", data=data["png_bytes"],
                              file_name=filename, mime="image/png",
                              key=f"png_{prefix}")
            # 快捷键：直接从历史记录生成矢量图
            if "svg_str" not in data:
                if st.button("🔄 生成矢量图", key=f"vec_{prefix}",
                             use_container_width=True):
                    with st.spinner("🔄 正在矢量化..."):
                        vec_result = BackendAPI.vectorize_image(
                            data["png_bytes"],
                            {"color_k": 8, "smooth": 1.2},
                        )
                    if vec_result["success"]:
                        data["svg_str"] = vec_result["svg_str"]
                        add_log(f"历史记录 #{idx} 矢量化完成")
                        st.toast("✅ 矢量图已生成")
                        if location != "dialog":
                            st.rerun()
                    else:
                        st.error(
                            f"矢量化失败: {vec_result.get('error_msg', '未知错误')}"
                        )
        elif data.get("preview_path"):
            meta = data.get("metadata", {})
            direct_url = f"{BACKEND_URL}/results/file?path={data['preview_path']}"
            if location == "dialog":
                # 弹窗内直接用 URL 加载，不用按钮
                st.image(direct_url, width=180)
            else:
                if st.button("📷 加载预览", key=f"load_{prefix}"):
                    try:
                        resp = requests.get(direct_url, timeout=30)
                        if resp.status_code == 200:
                            data["png_bytes"] = resp.content
                            st.rerun()
                    except Exception:
                        st.error("加载失败")
        if "svg_str" in data:
            st.components.v1.html(data["svg_str"], height=100)
            st.download_button("⬇️ SVG", data=data["svg_str"],
                              file_name=f"vector_{idx}.svg", mime="image/svg+xml",
                              key=f"svg_{prefix}")
        # 额外文件下载链接（来自后端返回的文件路径）
        meta = data.get("metadata", {})
        extra_items = [
            ("transparent_path", "🖼️ 透明 PNG"),
            ("svg_path", "📐 SVG"),
            ("original_path", "📦 原始图"),
            ("log_path", "📋 运行日志"),
            ("metadata_path", "📊 元数据"),
        ]
        link_cols = st.columns(len(extra_items))
        for ci, (key, label) in enumerate(extra_items):
            fpath = meta.get(key, "")
            if fpath:
                url = f"{BACKEND_URL}/results/file?path={fpath}"
                with link_cols[ci]:
                    st.markdown(f'<a href="{url}" target="_blank">🔗 {label}</a>',
                                unsafe_allow_html=True)
        st.markdown("---")


# ======================= 模态弹窗 =======================

@st.dialog("📋 全部历史记录", width="large")
def show_all_history_dialog():
    """模态弹窗：浏览全部历史记录"""
    if not st.session_state.history:
        st.info("暂无历史记录")
    else:
        for idx, item in enumerate(st.session_state.history):
            _render_history_item(item, idx, location="dialog")


@st.dialog("📄 系统日志", width="large")
def show_logs_dialog():
    """模态弹窗：查看系统日志"""
    log_text = "\n".join(st.session_state.logs)
    st.text_area("", log_text, height=400, disabled=True, label_visibility="collapsed")
    if st.button("关闭", key="close_logs_btn"):
        st.rerun()


# ======================= 会话状态管理 =======================
def init_session():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "logs" not in st.session_state:
        st.session_state.logs = ["✨ 系统就绪"]
    if "trigger_generate" not in st.session_state:
        st.session_state.trigger_generate = False
    if "trigger_vectorize" not in st.session_state:
        st.session_state.trigger_vectorize = False
    if "style_preset" not in st.session_state:
        st.session_state.style_preset = ""
    if "show_custom" not in st.session_state:
        st.session_state.show_custom = False
    if "_history_loaded" not in st.session_state:
        _load_history_from_backend()
        st.session_state._history_loaded = True
    # ---- 批量模式状态 ----
    if "batch_items" not in st.session_state:
        st.session_state.batch_items = []
    if "batch_results" not in st.session_state:
        st.session_state.batch_results = []
    if "batch_running" not in st.session_state:
        st.session_state.batch_running = False
    if "batch_default_style" not in st.session_state:
        st.session_state.batch_default_style = ""
    if "batch_default_seed" not in st.session_state:
        st.session_state.batch_default_seed = 42
    if "batch_default_resolution" not in st.session_state:
        st.session_state.batch_default_resolution = "1024x1024"
    # ---- 矢量化历史选择 ----
    if "vector_history_idx" not in st.session_state:
        st.session_state.vector_history_idx = None


def _load_history_from_backend():
    """启动时从后端加载 output/ 中的历史结果"""
    try:
        resp = requests.get(f"{BACKEND_URL}/results?limit=20", timeout=10)
        if resp.status_code != 200:
            return
        results = resp.json()
        for meta in results:
            st.session_state.history.append({
                "type": "🎨 生成",
                "title": meta.get("text", "untitled"),
                "data": {
                    "png_bytes": None,
                    "metadata": meta,
                    "preview_path": meta.get("preview_path"),
                },
                "time": meta.get("timestamp_utc", ""),
            })
        if results:
            add_log(f"从历史记录加载了 {len(results)} 条结果")
    except Exception:
        pass

def add_log(msg, level="INFO"):
    log_line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    st.session_state.logs.append(log_line)
    if len(st.session_state.logs) > 30:
        st.session_state.logs = st.session_state.logs[-30:]

    # 同步写入持久化日志文件
    log_func = getattr(_frontend_logger, level.lower(), _frontend_logger.info)
    log_func(msg)

def add_to_history(item_type, title, data):
    st.session_state.history.insert(0, {
        "type": item_type,
        "title": title,
        "data": data,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    if len(st.session_state.history) > 20:
        st.session_state.history = st.session_state.history[:20]

# ---------------- 批量模式辅助函数 ----------------

STYLE_PRESETS = [
    "默认艺术风格",
    "国风书法，墨色渐变，金色描边，梅花装饰，透明背景",
    "海洋浪漫，蓝青渐变，海浪贝壳装饰，透明背景",
    "促销卡通，粉紫配色，弧形横幅，描边醒目，数字突出",
]


def parse_batch_file(content: bytes, filename: str) -> List[Dict[str, str]]:
    """解析上传的 CSV / JSON / TXT 文件为 [{text, style?, seed?}, ...]

    Args:
        content: 文件二进制内容
        filename: 文件名（用于判断后缀）

    Returns:
        提示词字典列表（text 必填，style/seed 可选）
    """
    ext = Path(filename).suffix.lower()
    items: List[Dict[str, str]] = []

    if ext == ".csv":
        import io as _io
        df = pd.read_csv(_io.BytesIO(content))
        col_map: Dict[str, str] = {c.strip().lower(): c for c in df.columns}
        text_col = col_map.get("text")
        if text_col is None:
            raise ValueError("CSV 文件中未找到 'text' 列，可用列: " + ", ".join(df.columns))
        style_col = col_map.get("style")
        seed_col = col_map.get("seed")
        for _, row in df.iterrows():
            text = str(row.get(text_col, "")).strip()
            if not text:
                continue
            item: Dict[str, str] = {"text": text}
            if style_col and pd.notna(row.get(style_col)):
                item["style"] = str(row[style_col]).strip()
            if seed_col and pd.notna(row.get(seed_col)):
                try:
                    item["seed"] = str(int(row[seed_col]))
                except (ValueError, TypeError):
                    pass
            items.append(item)

    elif ext == ".json":
        data = json.loads(content.decode("utf-8"))
        if not isinstance(data, list):
            raise ValueError("JSON 文件应为数组格式")
        for entry in data:
            if not isinstance(entry, dict):
                continue
            text = str(entry.get("text", "")).strip()
            if not text:
                continue
            item: Dict[str, str] = {"text": text}
            if entry.get("style"):
                item["style"] = str(entry["style"]).strip()
            if entry.get("seed") is not None:
                item["seed"] = str(int(entry["seed"]))
            items.append(item)

    elif ext == ".txt":
        lines = content.decode("utf-8").splitlines()
        for line in lines:
            line = line.strip()
            if line and not line.startswith("#"):
                items.append({"text": line})
    else:
        raise ValueError(f"不支持的文件格式: {ext}（支持 .csv / .json / .txt）")

    return items


# ======================= 批量模式渲染 =======================


def _render_batch_mode():
    """渲染批量生成模式的完整 UI（上传 → 运行 → 汇总）"""
    # ── Phase 1: 上传与配置 ──
    if not st.session_state.batch_running and not st.session_state.batch_results:
        st.markdown("### 📋 批量生成艺术字")
        st.caption("上传包含多条提示词的清单文件，批量生成艺术字并输出汇总表")

        # 文件上传 + 手动粘贴
        uploaded = st.file_uploader(
            "上传提示词清单",
            type=["csv", "json", "txt"],
            key="batch_uploader",
            help="支持 CSV (text/style/seed 列) / JSON (对象数组) / TXT (每行一条)",
        )
        paste_text = st.text_area(
            "或手动输入（每行一条文字）",
            height=100,
            key="batch_paste",
            placeholder="青山集\n爱情海\nFreedom\n晚风",
        )

        # 默认参数
        with st.expander("⚙️ 默认参数（文件中缺失时使用）", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                batch_style = st.selectbox(
                    "默认风格", STYLE_PRESETS, index=0,
                    key="batch_def_style",
                )
                batch_seed = st.number_input(
                    "默认种子", value=42, step=1, key="batch_def_seed",
                )
            with col2:
                batch_res = st.selectbox(
                    "分辨率", ["1024x1024", "1024x768", "768x1024"],
                    key="batch_def_res",
                )
            batch_neg = st.text_input(
                "负面提示词", value="模糊，杂乱背景，错误文字",
                key="batch_def_neg",
            )

        # 解析
        items: List[Dict] = []
        parse_error: str = ""
        if uploaded:
            try:
                items = parse_batch_file(uploaded.getvalue(), uploaded.name)
            except Exception as e:
                parse_error = str(e)
        elif paste_text.strip():
            items = [
                {"text": line.strip()}
                for line in paste_text.strip().splitlines()
                if line.strip()
            ]

        if parse_error:
            st.error(f"📄 文件解析失败: {parse_error}")
        elif items:
            for item in items:
                item.setdefault("style", batch_style)
                item.setdefault("seed", str(batch_seed))
            st.session_state.batch_items = items

            st.markdown("#### 📝 待生成列表（可编辑）")
            df = pd.DataFrame(items)
            df["style_display"] = df["style"].apply(
                lambda s: s[:30] + "…" if len(s) > 30 else s
            )

            edited = st.data_editor(
                df[["text", "style_display", "seed"]],
                column_config={
                    "text": st.column_config.TextColumn(
                        "文字", width="large", required=True,
                        help="要生成艺术字的文字",
                    ),
                    "style_display": st.column_config.Column(
                        "风格", width="medium",
                        help="风格描述",
                    ),
                    "seed": st.column_config.NumberColumn(
                        "种子", min_value=0, max_value=2 ** 32 - 1,
                    ),
                },
                num_rows="dynamic",
                use_container_width=True,
                key="batch_editor",
            )

            # 同步编辑内容
            try:
                synced = []
                for i, row in edited.iterrows():
                    t = str(row.get("text", "")).strip()
                    if not t:
                        continue
                    orig = items[i] if i < len(items) else {}
                    synced.append({
                        "text": t,
                        "style": orig.get("style", batch_style),
                        "seed": str(int(row.get("seed", batch_seed))),
                        "resolution": batch_res,
                        "negative": batch_neg,
                    })
                st.session_state.batch_items = synced
            except Exception:
                pass

            st.info(f"共 {len(st.session_state.batch_items)} 条待生成")

            col_a, col_b = st.columns([1, 5])
            with col_a:
                if st.button("🚀 开始批量生成", type="primary",
                             use_container_width=True):
                    st.session_state.batch_running = True
                    st.session_state.batch_results = []
                    st.rerun()
            with col_b:
                if st.button("🗑️ 清空", use_container_width=True):
                    st.session_state.batch_items = []
                    st.session_state.batch_results = []
                    st.rerun()
        else:
            st.info("请上传文件或输入提示词")
            st.session_state.batch_items = []

    # ── Phase 2: 批量运行中 ──
    elif st.session_state.batch_running:
        _render_batch_running()

    # ── Phase 3: 结果汇总 ──
    elif st.session_state.batch_results:
        _render_batch_summary()


def _render_batch_running():
    """渲染批量运行进度"""
    items = st.session_state.batch_items
    total = len(items)
    results = st.session_state.batch_results
    completed = len(results)

    status = st.status(
        label=f"⏳ 批量生成中 … {completed}/{total}",
        state="running",
        expanded=True,
    )
    progress = st.progress(0, text="准备中…")

    # 预创建 item 级别的占位
    placeholders = []
    for i in range(total):
        placeholders.append(status.empty())

    # 先渲染已完成的结果
    for idx, r in enumerate(results):
        progress.progress((idx + 1) / total, text=f"已完成 {idx + 1}/{total}")
        with placeholders[idx]:
            if r["status"] == "success":
                st.success(f"✅ #{idx + 1} {r['text']}  ({r['time']}s)")
            else:
                st.error(f"❌ #{idx + 1} {r['text']} — {r.get('error_msg', '失败')}")

    # 逐一生成剩余项
    batch_start = time.time()

    for i in range(completed, total):
        item = items[i]
        progress.progress(
            i / total,
            text=f"⏳ 正在生成 {i + 1}/{total}: {item['text']}",
        )
        with placeholders[i]:
            st.info(f"⏳ #{i + 1} {item['text']} (生成中…)")

        item_start = time.time()
        res = BackendAPI.generate_art(
            text=item["text"],
            style_prompt=item.get("style", ""),
            negative_prompt=item.get("negative", "模糊，杂乱背景，错误文字"),
            seed=int(item.get("seed", 42)),
            resolution=item.get("resolution", "1024x1024"),
            vector_params={"color_clusters": 6, "smooth": 1.2},
        )
        item_time = round(time.time() - item_start, 1)

        if res["success"]:
            row = {
                "text": item["text"],
                "style": item.get("style", ""),
                "seed": item.get("seed", ""),
                "status": "success",
                "time": item_time,
                "png_bytes": res["png_bytes"],
                "metadata": res.get("metadata", {}),
            }
            add_to_history("🎨 批量", item["text"], res)
            with placeholders[i]:
                st.success(f"✅ #{i + 1} {item['text']}  ({item_time}s)")
        else:
            row = {
                "text": item["text"],
                "style": item.get("style", ""),
                "seed": item.get("seed", ""),
                "status": "failed",
                "time": item_time,
                "error_msg": res.get("error_msg", "未知错误"),
            }
            add_log(
                f"批量 #{i + 1} 失败: {item['text']} — {res.get('error_msg')}",
                level="ERROR",
            )
            with placeholders[i]:
                st.error(f"❌ #{i + 1} {item['text']} — {res.get('error_msg', '失败')}")

        st.session_state.batch_results.append(row)
        status.label = f"⏳ 批量生成中 … {i + 1}/{total}"

    # 完成
    total_time = round(time.time() - batch_start, 1)
    progress.progress(1.0, text="✅ 全部完成!")
    status.label = f"✅ 批量完成: {total} 条, 耗时 {total_time}s"
    status.state = "complete"
    st.session_state.batch_running = False
    time.sleep(1.5)
    st.rerun()


def _render_batch_summary():
    """渲染批量结果汇总表"""
    results = st.session_state.batch_results
    total = len(results)
    success = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "failed")
    total_time = sum(r.get("time", 0) for r in results)

    st.markdown("### ✅ 批量生成完成")
    col_s, col_f, col_t = st.columns(3)
    col_s.metric("✅ 成功", success, f"{success}/{total}")
    col_f.metric("❌ 失败", failed)
    col_t.metric("⏱️ 总耗时", f"{total_time:.0f}s")

    st.markdown("---")

    # 逐行渲染结果
    for idx, row in enumerate(results):
        cols = st.columns([0.5, 1.8, 1.5, 0.8, 0.8, 0.8, 1.2, 1.2])
        cols[0].write(f"**#{idx + 1}**")
        cols[1].write(row["text"])
        cols[2].caption(row.get("style", "")[:24])
        cols[3].write(str(row.get("seed", "")))

        if row["status"] == "success":
            cols[4].success("✅")
            cols[5].write(f"{row['time']}s")
            cols[6].image(row["png_bytes"], width=80)
            meta = row.get("metadata", {})
            fname = _make_download_name(meta, idx)
            cols[7].download_button(
                "⬇️",
                data=row["png_bytes"],
                file_name=fname,
                mime="image/png",
                key=f"batch_dl_{idx}",
            )
        else:
            cols[4].error("❌")
            cols[5].write(f"{row['time']}s")
            cols[6].write("—")
            cols[7].caption(row.get("error_msg", "")[:16])
        st.markdown("---")

    # 底部操作区
    st.markdown("#### 💾 操作")
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        if st.button("🔄 新批次", type="primary", use_container_width=True):
            st.session_state.batch_items = []
            st.session_state.batch_results = []
            st.session_state.batch_running = False
            st.rerun()

    with col_b:
        if success > 0:
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for i, r in enumerate(results):
                    if r["status"] == "success":
                        meta = r.get("metadata", {})
                        fn = _make_download_name(meta, i)
                        zf.writestr(fn, r["png_bytes"])
            st.download_button(
                "📥 下载全部 (ZIP)",
                data=zip_buf.getvalue(),
                file_name="vecrafter_batch.zip",
                mime="application/zip",
                use_container_width=True,
            )

    with col_c:
        if st.button("📋 复制汇总报告", use_container_width=True):
            lines = ["text,style,seed,status,time_s"]
            for r in results:
                lines.append(
                    f"{r['text']},{r.get('style', '')},{r.get('seed', '')},"
                    f"{r['status']},{r.get('time', '')}"
                )
            report = "\n".join(lines)
            st.toast("✅ 汇总报告已生成（可在日志中查看）")
            add_log(f"批量汇总:\n{report}")


def main():
    init_session()
    
    # ----- 侧边栏（仅功能按钮） -----
    with st.sidebar:
        st.markdown("## 🧰 功能")
        st.markdown("---")

        if st.button("🗂️ 全部历史", use_container_width=True):
            show_all_history_dialog()
        if st.button("📄 系统日志", use_container_width=True):
            show_logs_dialog()
        if st.button("🗑️ 清空历史", use_container_width=True):
            st.session_state.history = []
            add_log("历史记录已清空")
            st.rerun()
        if st.button("📋 复制 SVG", use_container_width=True):
            if st.session_state.history:
                latest = st.session_state.history[0]
                if "svg_str" in latest.get("data", {}):
                    st.toast("✅ 已复制 SVG 到剪贴板（模拟）")
                    add_log("复制 SVG 内容")
                else:
                    st.warning("暂无可用 SVG")
            else:
                st.warning("无历史记录")

        st.markdown("---")
    
    # ----- 主区域：两列布局（左侧输入区 + 右侧历史）-----
    col_main, col_right = st.columns([2, 1], gap="large")
    
    with col_main:
        # 艺术字标题（居中，无副标题）
        st.markdown('<div class="art-title">', unsafe_allow_html=True)
        st.components.v1.html(render_art_title(), height=70)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 模式切换（选项卡）
        mode = st.segmented_control(
            "操作模式",
            ["🎨 生成艺术字", "🖼️ 图片矢量化", "📋 批量生成"],
            default="🎨 生成艺术字",
            label_visibility="collapsed",
        )

        # ---- 生成艺术字模式 ----
        if mode == "🎨 生成艺术字":
            # 输入卡片（输入框 + 启动按钮）
            st.markdown('<div class="input-card">', unsafe_allow_html=True)
            text_input = st.text_area(
                "输入文字 (支持中英文/数字)",
                value="青山集",
                height=120,
                key="big_text",
                label_visibility="collapsed",
                placeholder="例如：七里香、爱情海、咖啡 Latte 2.0 ..."
            )
            st.caption("💡 支持2-8个汉字、英文单词或促销数字")

            if st.button("✨ 立即生成", type="primary", use_container_width=True):
                st.session_state.trigger_generate = True
            st.markdown('</div>', unsafe_allow_html=True)

            # 提示词配置：小按钮区域
            st.markdown("### 🎨 风格配置")
            col_btns = st.columns(4)
            with col_btns[0]:
                if st.button("🌸 国风书法", use_container_width=True):
                    st.session_state.style_preset = "国风书法，墨色渐变，金色描边，梅花装饰，透明背景"
                    st.session_state.show_custom = False
            with col_btns[1]:
                if st.button("🌊 海洋浪漫", use_container_width=True):
                    st.session_state.style_preset = "海洋浪漫，蓝青渐变，海浪贝壳装饰，透明背景"
                    st.session_state.show_custom = False
            with col_btns[2]:
                if st.button("🎈 促销卡通", use_container_width=True):
                    st.session_state.style_preset = "促销卡通，粉紫配色，弧形横幅，描边醒目，数字突出"
                    st.session_state.show_custom = False
            with col_btns[3]:
                if st.button("✨ 自定义", use_container_width=True):
                    st.session_state.show_custom = True

            if st.session_state.style_preset and not st.session_state.show_custom:
                st.info(f"当前风格: {st.session_state.style_preset}")
            if st.session_state.show_custom:
                style_prompt = st.text_area("自定义风格提示词", value=st.session_state.style_preset, height=80)
                st.session_state.style_preset = style_prompt

            # 高级参数折叠
            with st.expander("⚙️ 高级参数 (种子/分辨率/矢量化)"):
                col1, col2 = st.columns(2)
                with col1:
                    seed = st.number_input("随机种子", value=42, step=1)
                    resolution = st.selectbox("分辨率", ["1024x1024", "1024x768", "768x1024"])
                with col2:
                    color_clusters = st.slider("颜色聚类", 2, 12, 6)
                    smooth = st.slider("平滑阈值", 0.5, 3.0, 1.2)
                negative = st.text_input("负面提示词", value="模糊，杂乱背景，错误文字")

        # ---- 图片矢量化模式 ----
        elif mode == "🖼️ 图片矢量化":
            st.markdown("### 🖼️ 图片矢量化")
            st.caption("从历史记录选择一张图片，或上传新图片进行矢量化")
            st.markdown('<div class="input-card">', unsafe_allow_html=True)

            # 状态占位 —— 放在用户可见位置
            status_placeholder = st.empty()

            # 从历史记录选择
            if st.session_state.history:
                hist_options = []
                hist_index_map = []
                for i, h in enumerate(st.session_state.history):
                    d = h.get("data", {})
                    if d.get("png_bytes") or d.get("preview_path"):
                        label = f"{h['title'][:20]} · {h['time'][:10]}"
                        hist_options.append(label)
                        hist_index_map.append(i)

                if hist_options:
                    selected_label = st.selectbox(
                        "从历史记录选择图片",
                        hist_options,
                        key="vec_hist_select",
                        label_visibility="collapsed",
                        placeholder="选择一条历史记录…",
                    )
                    if selected_label:
                        idx = hist_options.index(selected_label)
                        st.session_state.vector_history_idx = hist_index_map[idx]
                        sel = st.session_state.history[hist_index_map[idx]]["data"]
                        # 显示选中图片
                        if sel.get("png_bytes"):
                            st.image(sel["png_bytes"], width=120)
                        elif sel.get("preview_path"):
                            st.image(
                                f"{BACKEND_URL}/results/file?path={sel['preview_path']}",
                                width=120,
                            )

            # 或从文件上传
            st.markdown("— 或 —")
            uploaded_file = st.file_uploader("上传新图片", type=["png", "jpg", "jpeg"],
                                              key="vec_uploader")
            if uploaded_file:
                st.session_state.vector_file = uploaded_file
                st.session_state.vector_history_idx = None
                st.image(uploaded_file, width=120)

            # 矢量化高级参数
            with st.expander("⚙️ 矢量化参数", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.slider("颜色聚类", 2, 12, 6, key="vec_clusters")
                with col2:
                    st.slider("平滑阈值", 0.5, 3.0, 1.2, key="vec_smooth")

            # 启动按钮 + 就地处理结果
            if st.button("🔄 开始矢量化", type="primary", use_container_width=True):
                st.session_state.trigger_vectorize = True

            if st.session_state.get("trigger_vectorize", False):
                st.session_state.trigger_vectorize = False

                # 获取图像数据：从历史选择（含懒加载）或文件上传
                img_bytes = None
                source_name = ""
                hist_idx = st.session_state.get("vector_history_idx")

                if hist_idx is not None and hist_idx < len(st.session_state.history):
                    hist_item = st.session_state.history[hist_idx]
                    hist_data = hist_item["data"]
                    if hist_data.get("png_bytes"):
                        img_bytes = hist_data["png_bytes"]
                    elif hist_data.get("preview_path"):
                        try:
                            resp = requests.get(
                                f"{BACKEND_URL}/results/file",
                                params={"path": hist_data["preview_path"]},
                                timeout=30,
                            )
                            if resp.status_code == 200:
                                img_bytes = resp.content
                                hist_data["png_bytes"] = img_bytes  # 缓存
                        except Exception:
                            pass
                    source_name = hist_item["title"]

                if img_bytes is None:
                    vf = st.session_state.get("vector_file")
                    if vf is not None:
                        img_bytes = vf.getvalue()
                        source_name = vf.name

                if img_bytes is None:
                    msg = "请从上方选择一条历史记录，或上传图片"
                    add_log(f"矢量化跳过: {msg}", level="WARNING")
                    status_placeholder.warning(f"⚠️ {msg}")
                else:
                    status_placeholder.info("🔄 正在矢量化，请稍候...")
                    params = {
                        "color_k": st.session_state.get("vec_clusters", 6),
                        "smooth": st.session_state.get("vec_smooth", 1.2),
                    }
                    result = BackendAPI.vectorize_image(img_bytes, params)
                    if result["success"]:
                        add_to_history("🖼️ 矢量化", source_name, result)
                        add_log(f"矢量化完成: {source_name}")
                        status_placeholder.success("✅ 矢量化完成，结果已保存")
                        st.rerun()
                    else:
                        err = result.get("error_msg", "未知错误")
                        add_log(f"矢量化失败: {source_name} — {err}", level="ERROR")
                        status_placeholder.error(f"❌ 矢量化失败: {err}")

            st.markdown('</div>', unsafe_allow_html=True)

        # ---- 批量模式 ----
        else:
            _render_batch_mode()
    
    with col_right:
        st.markdown("## 🕐 最近结果（3条）")
        st.markdown('<div class="right-sidebar">', unsafe_allow_html=True)
        if not st.session_state.history:
            st.info("暂无结果，请左侧生成后显示")
        else:
            for idx, item in enumerate(st.session_state.history[:3]):
                _render_history_item(item, idx)
            if len(st.session_state.history) > 3:
                st.info(f"📋 还有 {len(st.session_state.history) - 3} 条历史记录，"
                        "点击侧边栏「🗂️ 全部历史」查看")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ----- 生成逻辑触发 -----
    if mode == "🎨 生成艺术字" and st.session_state.get("trigger_generate", False):
        st.session_state.trigger_generate = False
        text = text_input.strip()
        if not text:
            add_log("输入文字为空，跳过生成", level="WARNING")
            st.warning("请输入文字内容")
        else:
            style = st.session_state.style_preset if st.session_state.style_preset else "默认艺术风格"
            vector_params = {"color_clusters": color_clusters, "smooth": smooth}
            with st.spinner("🖌️ 正在生成艺术字并矢量化..."):
                result = BackendAPI.generate_art(
                    text, style, negative, int(seed), resolution, vector_params
                )
            if result["success"]:
                add_to_history("🎨 生成", text, result)
                add_log(f"生成成功: {text}")
                st.success("✅ 生成完成，结果已显示在右侧历史中")
                st.rerun()
            else:
                add_log(f"生成失败: {result.get('error_msg')}", level="ERROR")
                st.error(f"生成失败: {result.get('error_msg')}")
    


if __name__ == "__main__":
    main()
# app.py
# 运行: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import time
import json
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional, List
import re

# ======================= 页面配置 =======================
st.set_page_config(
    page_title="矢量艺术字工坊 | ArtForge",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================= 自定义样式 =======================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(145deg, #f0f4f9 0%, #e9eff5 100%);
    }
    .main-header {
        text-align: center;
        padding: 1rem 0 0.5rem 0;
        background: rgba(255,255,255,0.6);
        border-radius: 24px;
        margin-bottom: 1.5rem;
    }
    .main-header h1 {
        font-size: 2.5rem;
        background: linear-gradient(135deg, #1c5f6e, #32849e);
        -webkit-background-clip: text;
        color: transparent;
        font-weight: 700;
    }
    .status-badge {
        background-color: #1e6d7e20;
        padding: 0.2rem 0.8rem;
        border-radius: 40px;
        font-size: 0.8rem;
        font-weight: 500;
        color: #1e6d7e;
    }
    .result-item {
        background: white;
        border-radius: 12px;
        padding: 0.5rem;
        margin-bottom: 0.5rem;
        border-left: 4px solid #1e6d7e;
        cursor: pointer;
    }
    .result-item:hover {
        background: #f0f7fa;
    }
</style>
""", unsafe_allow_html=True)

# ======================= 1. 前后端通信接口抽象层 =======================
# 本层定义了与后端（ComfyUI + 矢量化服务）通信的标准接口。
# 实际部署时，只需替换此模块中的函数实现（例如改用 requests 调用真实 API），
# 前端（Streamlit UI）无需改动。

class BackendAPI:
    """前后端通信接口类，封装所有与后端服务的交互。"""
    
    @staticmethod
    def generate_art(text: str, style_prompt: str, negative_prompt: str,
                     seed: int, resolution: str, vector_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        提交艺术字生成任务（同步模拟）。
        实际应异步提交并轮询结果，此处为简化演示，直接返回结果。
        
        返回格式:
        {
            "success": bool,
            "png_bytes": bytes,
            "svg_str": str,
            "metadata": dict,
            "error_msg": str (if failed)
        }
        """
        # 模拟后端处理耗时
        time.sleep(1.2)
        # 模拟生成过程（实际应调用 ComfyUI API 或本地模型）
        try:
            # 创建透明背景图像
            size = 512
            img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("simhei.ttf", 48)
            except:
                font = ImageFont.load_default()
            draw.text((size//2 - 60, size//2 - 30), text[:4], fill=(60, 120, 80, 255), font=font)
            draw.ellipse((size-80, 20, size-20, 80), fill=(200, 180, 100, 180))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            png_bytes = buf.getvalue()
            
            # 构造 SVG
            svg_content = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 400" width="100%" height="100%">
            <rect width="600" height="400" fill="transparent"/>
            <g id="main-text" fill="#2C6B4A" stroke="#D4AF37" stroke-width="2">
                <text x="80" y="200" font-family="'Noto Serif SC', 'KaiTi'" font-size="56" fill="#2C6B4A" stroke="#D4AF37">{text}</text>
            </g>
            <g id="decoration" fill="#7FB07F" opacity="0.8">
                <circle cx="500" cy="80" r="18"/>
                <path d="M520,110 Q540,140 510,150 Q490,140 520,110Z"/>
            </g>
            </svg>'''
            metadata = {
                "text": text,
                "style": style_prompt[:100],
                "negative": negative_prompt[:100],
                "seed": seed,
                "resolution": resolution,
                "vector_params": vector_params,
                "timestamp": datetime.now().isoformat()
            }
            return {"success": True, "png_bytes": png_bytes, "svg_str": svg_content, "metadata": metadata}
        except Exception as e:
            return {"success": False, "error_msg": str(e)}
    
    @staticmethod
    def vectorize_image(image_bytes: bytes, params: Dict[str, Any]) -> Dict[str, Any]:
        """将位图转换为 SVG 矢量图。"""
        time.sleep(0.8)
        svg_content = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 400" width="100%" height="100%">
        <rect width="500" height="400" fill="transparent"/>
        <g id="vector-result" fill="#8B5A2B" stroke="#D2B48C" stroke-width="2">
            <path d="M100,100 L200,80 L300,130 L250,220 L150,200 Z"/>
            <circle cx="350" cy="200" r="45"/>
            <text x="120" y="320" font-family="'KaiTi'" font-size="36" fill="#5a3a1a">矢量提取模拟</text>
        </g>
        </svg>'''
        metadata = {"params": params, "timestamp": datetime.now().isoformat()}
        return {"success": True, "svg_str": svg_content, "metadata": metadata}
    
    @staticmethod
    def batch_generate(items: List[Dict], base_seed: int, resolution: str, vector_params: Dict) -> List[Dict]:
        """批量生成，返回每个任务的结果列表。"""
        results = []
        for idx, item in enumerate(items):
            text = item.get("text", "")
            style = item.get("style", "")
            seed = base_seed + idx
            res = BackendAPI.generate_art(text, style, "", seed, resolution, vector_params)
            results.append({
                "text": text,
                "success": res["success"],
                "png_bytes": res.get("png_bytes"),
                "svg_str": res.get("svg_str"),
                "metadata": res.get("metadata"),
                "error": res.get("error_msg")
            })
        return results


# ======================= 2. 会话状态管理 =======================
def init_session_state():
    """初始化 session_state 中的数据结构。"""
    if "generation_history" not in st.session_state:
        st.session_state.generation_history = []  # 存储历史记录项，每个项包含 type, title, result_data
    if "batch_results_cache" not in st.session_state:
        st.session_state.batch_results_cache = None
    if "logs" not in st.session_state:
        st.session_state.logs = ["✨ 系统就绪 | 等待任务"]

def add_log(message: str):
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append(f"[{timestamp}] {message}")
    if len(st.session_state.logs) > 100:
        st.session_state.logs = st.session_state.logs[-100:]

def add_to_history(item_type: str, title: str, result_data: Dict):
    """将生成结果添加到侧边栏历史记录。"""
    st.session_state.generation_history.insert(0, {
        "type": item_type,  # "single", "batch_item", "vectorize"
        "title": title,
        "data": result_data,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    # 保留最近50条
    if len(st.session_state.generation_history) > 50:
        st.session_state.generation_history = st.session_state.generation_history[:50]


# ======================= 3. 主界面 =======================
def main():
    init_session_state()
    
    # 标题区域
    st.markdown("""
    <div class="main-header">
        <h1>🎨 ArtForge · 矢量艺术字工坊</h1>
        <p>基于开源文生图模型 (ComfyUI) + Python 智能矢量化 | 透明背景 · 可编辑SVG · 批量生产</p>
        <div style="display: flex; justify-content: center; gap: 1rem; margin-top: 0.5rem;">
            <span class="status-badge">🏠 本地部署</span>
            <span class="status-badge">🎨 多风格控制</span>
            <span class="status-badge">✍️ 路径可编辑</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 侧边栏：展示所有结果和日志
    with st.sidebar:
        st.header("📁 结果历史记录")
        # 清空历史按钮
        if st.button("🗑️ 清空所有历史"):
            st.session_state.generation_history = []
            add_log("已清空历史记录")
            st.rerun()
        st.markdown("---")
        
        # 显示历史记录列表
        if not st.session_state.generation_history:
            st.info("暂无结果，请在主区域生成后显示")
        else:
            for idx, item in enumerate(st.session_state.generation_history):
                # 每个结果项显示为可展开的折叠块
                with st.expander(f"{item['type'].upper()} - {item['title'][:30]}", expanded=False):
                    # 根据类型展示不同内容
                    if item['type'] == 'single':
                        data = item['data']
                        if data.get("success"):
                            st.image(data["png_bytes"], caption="透明PNG", width=200)
                            st.components.v1.html(data["svg_str"], height=200)
                            st.download_button("💾 下载PNG", data=data["png_bytes"], file_name=f"art_{item['title']}.png", mime="image/png", key=f"dl_png_{idx}")
                            st.download_button("📄 下载SVG", data=data["svg_str"], file_name=f"vector_{item['title']}.svg", mime="image/svg+xml", key=f"dl_svg_{idx}")
                            with st.expander("📋 元数据"):
                                st.json(data.get("metadata", {}))
                        else:
                            st.error(f"生成失败: {data.get('error_msg')}")
                    elif item['type'] == 'batch_item':
                        data = item['data']
                        st.write(f"**文字**: {data.get('text')}")
                        if data.get("success"):
                            st.image(data["png_bytes"], width=150)
                            st.download_button("下载PNG", data=data["png_bytes"], file_name=f"batch_{data['text']}.png", mime="image/png", key=f"batch_png_{idx}")
                            st.download_button("下载SVG", data=data["svg_str"], file_name=f"batch_{data['text']}.svg", mime="image/svg+xml", key=f"batch_svg_{idx}")
                        else:
                            st.error("失败")
                    elif item['type'] == 'vectorize':
                        data = item['data']
                        if data.get("success"):
                            st.components.v1.html(data["svg_str"], height=200)
                            st.download_button("⬇️ 下载 SVG", data=data["svg_str"], file_name="vectorized.svg", mime="image/svg+xml", key=f"vec_{idx}")
                            with st.expander("元数据"):
                                st.json(data.get("metadata", {}))
                        else:
                            st.error("矢量化失败")
                    st.caption(f"时间: {item['timestamp']}")
        
        st.markdown("---")
        st.subheader("📋 系统日志")
        log_text = "\n".join(st.session_state.logs[-20:])
        st.text_area("", log_text, height=300, disabled=True, label_visibility="collapsed")
    
    # 主区域：模式选择与参数输入
    tab1, tab2, tab3 = st.tabs(["✨ 单条生成", "📚 批量生成", "🖼️ 图片矢量化"])
    
    with tab1:
        st.subheader("📝 提示词配置")
        col1, col2 = st.columns(2)
        with col1:
            text_content = st.text_input("文字内容", value="青山集")
            style_prompt = st.text_area("风格提示词", height=100, value="青绿色水墨质感，金色细描边，搭配山形与竹叶装饰，透明背景")
            negative_prompt = st.text_input("负面提示词", value="模糊，杂乱背景，错误文字")
        with col2:
            resolution = st.selectbox("分辨率", ["1024x1024", "1024x768", "768x1024"])
            seed = st.number_input("随机种子", value=42, step=1)
            with st.expander("🎛️ 矢量化参数"):
                color_clusters = st.slider("颜色聚类数量", 2, 12, 6)
                smooth_thresh = st.slider("平滑阈值", 0.5, 3.0, 1.2)
                min_area = st.number_input("最小区域过滤(px)", value=50)
                keep_gradient = st.checkbox("保留渐变/阴影")
        
        if st.button("🚀 生成艺术字 + 矢量转换", type="primary", use_container_width=True):
            vector_params = {
                "color_clusters": color_clusters,
                "smooth_thresh": smooth_thresh,
                "min_area": min_area,
                "keep_gradient": keep_gradient
            }
            with st.spinner("正在调用后端生成..."):
                result = BackendAPI.generate_art(
                    text_content, style_prompt, negative_prompt,
                    int(seed), resolution, vector_params
                )
            if result["success"]:
                add_log(f"单条生成成功: {text_content}")
                # 添加到侧边栏历史
                add_to_history("single", text_content, result)
                st.success("生成完成！结果已保存到侧边栏「结果历史记录」中。")
            else:
                st.error(f"生成失败: {result.get('error_msg')}")
                add_log(f"单条生成失败: {result.get('error_msg')}")
    
    with tab2:
        st.subheader("📦 批量提示词清单")
        upload_type = st.radio("输入方式", ["文本区域编辑", "上传 CSV/JSON"], horizontal=True)
        batch_items = []
        if upload_type == "文本区域编辑":
            batch_text = st.text_area(
                "每行格式: 文字 | 风格提示词",
                height=200,
                value="七里香 | 清新国风、墨绿色金边、植物叶片装饰\n爱情海 | 浪漫海洋、蓝青色配色、爱心海浪\n红豆抹茶 30% | 促销卡通描边、弧形横幅、醒目数字"
            )
            if st.button("解析清单"):
                lines = batch_text.strip().split('\n')
                for line in lines:
                    if '|' in line:
                        parts = line.split('|')
                        batch_items.append({"text": parts[0].strip(), "style": parts[1].strip()})
                if batch_items:
                    st.success(f"解析到 {len(batch_items)} 条任务")
                    st.session_state.batch_items = batch_items
                else:
                    st.error("请使用 文字 | 风格 格式")
        else:
            uploaded_file = st.file_uploader("上传 CSV/JSON", type=["csv", "json"])
            if uploaded_file:
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                    if 'text' in df.columns and 'style' in df.columns:
                        batch_items = df[['text', 'style']].to_dict('records')
                    else:
                        st.error("CSV需包含 text 和 style 列")
                else:
                    data = json.load(uploaded_file)
                    if isinstance(data, list):
                        batch_items = data
                    else:
                        st.error("JSON应为对象数组")
                if batch_items:
                    st.session_state.batch_items = batch_items
                    st.write("预览前5条:", batch_items[:5])
        
        if "batch_items" in st.session_state and st.session_state.batch_items:
            col_seed, col_res = st.columns(2)
            with col_seed:
                batch_seed = st.number_input("起始种子", value=1001, step=1)
            with col_res:
                batch_res = st.selectbox("分辨率", ["1024x1024", "1024x768"])
            if st.button("▶️ 开始批量生成", type="primary", use_container_width=True):
                vector_params = {"color_clusters": 6, "smooth_thresh": 1.2, "min_area": 50, "keep_gradient": True}
                progress_bar = st.progress(0)
                results = []
                total = len(st.session_state.batch_items)
                for idx, item in enumerate(st.session_state.batch_items):
                    text = item.get("text", "")
                    style = item.get("style", "")
                    res = BackendAPI.generate_art(text, style, "", int(batch_seed)+idx, batch_res, vector_params)
                    if res["success"]:
                        add_to_history("batch_item", text, {
                            "text": text,
                            "success": True,
                            "png_bytes": res["png_bytes"],
                            "svg_str": res["svg_str"],
                            "metadata": res["metadata"]
                        })
                        results.append({"text": text, "success": True})
                    else:
                        results.append({"text": text, "success": False, "error": res.get("error_msg")})
                    progress_bar.progress((idx+1)/total)
                st.success(f"批量生成完成，成功 {sum(1 for r in results if r['success'])} / {total}")
                add_log(f"批量生成完成，共 {total} 条任务")
    
    with tab3:
        st.subheader("🖼️ 上传已有艺术字图片，转换为 SVG 矢量")
        uploaded_image = st.file_uploader("选择 PNG 或 JPG 图片", type=["png", "jpg", "jpeg"])
        with st.expander("矢量化参数"):
            vec_k = st.slider("颜色聚类数量", 2, 10, 5)
            vec_smooth = st.slider("平滑度", 0.5, 3.0, 1.5)
            vec_min = st.number_input("最小区域过滤", value=30)
        if uploaded_image is not None and st.button("🔁 执行矢量化", type="primary"):
            img_bytes = uploaded_image.getvalue()
            params = {"color_k": vec_k, "smooth": vec_smooth, "min_area": vec_min}
            with st.spinner("正在处理..."):
                result = BackendAPI.vectorize_image(img_bytes, params)
            if result["success"]:
                add_to_history("vectorize", uploaded_image.name, result)
                st.success("矢量化完成！结果已保存到侧边栏。")
            else:
                st.error(f"矢量化失败: {result.get('error_msg')}")
                add_log(f"矢量化失败: {result.get('error_msg')}")
    
    # 页脚
    st.markdown("---")
    st.caption("© ArtForge 矢量艺术字工坊 | 前后端分离设计 | 所有结果均在侧边栏历史中")


if __name__ == "__main__":
    main()

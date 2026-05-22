# app.py
# 运行: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import time
import json
from datetime import datetime
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

# ======================= 后端模拟接口 =======================
class BackendAPI:
    @staticmethod
    def generate_art(text: str, style_prompt: str, negative_prompt: str,
                     seed: int, resolution: str, vector_params: Dict) -> Dict:
        time.sleep(1.2)
        try:
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
            
            svg_content = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 400" width="100%" height="100%">
            <rect width="600" height="400" fill="transparent"/>
            <g id="main-text" fill="#2C6B4A" stroke="#D4AF37" stroke-width="2">
                <text x="80" y="200" font-family="'Noto Serif SC', 'KaiTi'" font-size="56" fill="#2C6B4A">{text}</text>
            </g>
            <g id="decoration" fill="#7FB07F" opacity="0.8">
                <circle cx="500" cy="80" r="18"/>
                <path d="M520,110 Q540,140 510,150 Q490,140 520,110Z"/>
            </g>
            </svg>'''
            metadata = {
                "text": text, "style": style_prompt[:80], "seed": seed,
                "resolution": resolution, "timestamp": datetime.now().isoformat()
            }
            return {"success": True, "png_bytes": png_bytes, "svg_str": svg_content, "metadata": metadata}
        except Exception as e:
            return {"success": False, "error_msg": str(e)}
    
    @staticmethod
    def vectorize_image(image_bytes: bytes, params: Dict) -> Dict:
        time.sleep(0.8)
        svg_content = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 400" width="100%" height="100%">
        <rect width="500" height="400" fill="transparent"/>
        <g id="vector-result" fill="#8B5A2B" stroke="#D2B48C" stroke-width="2">
            <path d="M100,100 L200,80 L300,130 L250,220 L150,200 Z"/>
            <circle cx="350" cy="200" r="45"/>
            <text x="120" y="320" font-family="'KaiTi'" font-size="36" fill="#5a3a1a">矢量提取模拟</text>
        </g>
        </svg>'''
        return {"success": True, "svg_str": svg_content, "metadata": {"params": params}}

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

def add_log(msg):
    st.session_state.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
    if len(st.session_state.logs) > 30:
        st.session_state.logs = st.session_state.logs[-30:]

def add_to_history(item_type, title, data):
    st.session_state.history.insert(0, {
        "type": item_type,
        "title": title,
        "data": data,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    if len(st.session_state.history) > 20:
        st.session_state.history = st.session_state.history[:20]

# ======================= 主界面 =======================
def main():
    init_session()
    
    # ----- 左侧边栏：仅功能按钮 -----
    with st.sidebar:
        st.markdown("## 🧰 功能面板")
        st.markdown("---")
        
        mode = st.radio(
            "操作模式",
            ["🎨 生成艺术字", "🖼️ 图片矢量化"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        if mode == "🎨 生成艺术字":
            if st.button("✨ 立即生成", type="primary", use_container_width=True):
                st.session_state.trigger_generate = True
        else:
            if st.button("🔄 开始矢量化", type="primary", use_container_width=True):
                st.session_state.trigger_vectorize = True
        
        st.markdown("---")
        if st.button("🗑️ 清空历史", use_container_width=True):
            st.session_state.history = []
            add_log("历史记录已清空")
            st.rerun()
        
        if st.button("📋 复制最新 SVG", use_container_width=True):
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
        with st.expander("📄 系统日志"):
            st.text_area("", "\n".join(st.session_state.logs), height=200, disabled=True, label_visibility="collapsed")
    
    # ----- 主区域：两列布局（左侧输入区 + 右侧历史）-----
    col_main, col_right = st.columns([2, 1], gap="large")
    
    with col_main:
        # 艺术字标题（居中，无副标题）
        st.markdown('<div class="art-title">', unsafe_allow_html=True)
        st.components.v1.html(render_art_title(), height=70)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 大输入框
        text_input = st.text_area(
            "输入文字 (支持中英文/数字)",
            value="青山集",
            height=120,
            key="big_text",
            label_visibility="collapsed",
            placeholder="例如：七里香、爱情海、咖啡 Latte 2.0 ..."
        )
        st.caption("💡 支持2-8个汉字、英文单词或促销数字")
        
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
        
        # 显示当前风格
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
        
        # 矢量化模式下的文件上传
        if mode == "🖼️ 图片矢量化":
            uploaded_file = st.file_uploader("上传艺术字图片 (PNG/JPG)", type=["png", "jpg", "jpeg"])
            st.session_state.vector_file = uploaded_file
    
    with col_right:
        st.markdown("## 📜 结果历史")
        st.markdown('<div class="right-sidebar">', unsafe_allow_html=True)
        if not st.session_state.history:
            st.info("暂无结果，请左侧生成后显示")
        else:
            for idx, item in enumerate(st.session_state.history):
                with st.container():
                    st.markdown(f"**{item['type']}** · {item['title'][:20]}")
                    st.caption(f"🕒 {item['time']}")
                    data = item['data']
                    if "png_bytes" in data:
                        st.image(data["png_bytes"], width=120)
                    if "svg_str" in data:
                        st.components.v1.html(data["svg_str"], height=100)
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.download_button("⬇️ PNG", data=data["png_bytes"], file_name=f"art_{idx}.png", mime="image/png", key=f"png_{idx}")
                        with col_b:
                            st.download_button("⬇️ SVG", data=data["svg_str"], file_name=f"vector_{idx}.svg", mime="image/svg+xml", key=f"svg_{idx}")
                    st.markdown("---")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ----- 生成逻辑触发 -----
    if mode == "🎨 生成艺术字" and st.session_state.get("trigger_generate", False):
        st.session_state.trigger_generate = False
        text = text_input.strip()
        if not text:
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
                st.error(f"生成失败: {result.get('error_msg')}")
    
    if mode == "🖼️ 图片矢量化" and st.session_state.get("trigger_vectorize", False):
        st.session_state.trigger_vectorize = False
        if "vector_file" not in st.session_state or st.session_state.vector_file is None:
            st.warning("请先上传图片")
        else:
            file = st.session_state.vector_file
            img_bytes = file.getvalue()
            params = {"color_k": color_clusters, "smooth": smooth}
            with st.spinner("🔄 正在矢量化..."):
                result = BackendAPI.vectorize_image(img_bytes, params)
            if result["success"]:
                add_to_history("🖼️ 矢量化", file.name, result)
                add_log(f"矢量化完成: {file.name}")
                st.success("✅ 矢量化完成，结果已保存")
                st.rerun()
            else:
                st.error("矢量化失败")

if __name__ == "__main__":
    main()
"""

知乎导来看看咋用的
https://zhuanlan.zhihu.com/p/1991517711923164045

"""





import streamlit as st
import time
import pandas as pd

# 设置页面基本信息
st.set_page_config(page_title="Streamlit 组件博物馆", layout="wide", page_icon="🏛️")

st.title("🏛️ Streamlit 常用组件速查表")
st.markdown("这是一个教学 Demo，展示了后端开发最常用的 Streamlit 组件。")

# 使用 Tabs 把不同的知识点分开，避免页面太长
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📝 文本展示", 
    "🎛️ 交互组件", 
    "📐 布局排版", 
    "🚦 状态反馈", 
    "🤖 聊天界面"
])

# ==================================================
# Tab 1: 文本与数据展示 (Output)
# ==================================================
with tab1:
    st.header("1. 像 Print 一样输出")
    
    with st.echo():
        # st.write 是最万能的，扔进去什么都可以
        st.write("这是普通文本")
        st.write('////')
        st.write({"key": "value", "list": [1, 2, 3]}) # 字典会被格式化
    
    st.divider() # 分割线
    
    with st.echo():
        # Markdown 语法支持
        st.markdown("这是 **加粗**, *斜体*, 和 [链接](https://streamlit.io)")
        
        # 专门用来显示代码块
        code_example = """
        def hello():
            print("Hello World")
        """
        st.code(code_example, language="python")
        
    st.divider()
    
    with st.echo():
        # 类似于后端接口返回的 JSON
        data = {
            "user_id": 101,
            "roles": ["admin", "editor"],
            "meta": {"created_at": "2023-01-01"}
        }
        st.json(data)

# ==================================================
# Tab 2: 交互组件 (Input)
# ==================================================
with tab2:
    st.header("2. 获取用户输入")
    st.info("试着操作一下下面的组件，看下方显示的 value 变化！")
    
    col_demo_1, col_demo_2 = st.columns(2)
    
    with col_demo_1:
        with st.echo():
            # 文本输入
            name = st.text_input("请输入你的名字", value="Python 大佬")
            st.write(f"当前变量 name = {name}")
            
        with st.echo():
            # 数字滑块
            age = st.slider("选择年龄", 0, 100, 25)
            st.write(f"当前变量 age = {age}")

    with col_demo_2:
        with st.echo():
            # 下拉选择
            role = st.selectbox(
                "选择你的角色", 
                ["Backend", "Frontend", "Fullstack"]
            )
            st.write(f"当前变量 role = {role}")
            
        with st.echo():
            # 开关 (Checkbox)
            is_debug = st.checkbox("开启调试模式")
            if is_debug:
                st.write("🔴 调试模式已激活")

    st.divider()
    with st.echo():
        # 按钮 (最常用)
        if st.button("点击我发送请求"):
            st.write("🚀 按钮被点击了！(Button 返回了 True)")

# ==================================================
# Tab 3: 布局排版 (Layout)
# ==================================================
with tab3:
    st.header("3. 页面布局")
    
    st.subheader("侧边栏 (Sidebar)")
    st.write("👈 看左边！侧边栏通常用于放全局配置。")
    with st.sidebar:
        st.header("侧边栏配置区")
        st.write("这里是 sidebar")
        st.radio("模型选择", ["GPT-3.5", "GPT-4", "Claude"])
        
    st.subheader("列布局 (Columns)")
    with st.echo():
        # 把一行分成 3 列
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.write("我是左边列")
            st.button("按钮 A")
        with c2:
            st.write("我是中间列")
            st.image("https://streamlit.io/images/brand/streamlit-mark-color.png", width=50)
        with c3:
            st.write("我是右边列")
            st.metric("服务器负载", "85%", "12%")

    st.subheader("折叠面板 (Expander)")
    with st.echo():
        # 像手风琴一样折叠，适合放长文本或日志
        with st.expander("点击查看详细 Prompt"):
            st.write("你是一个专业的 Python 助手...")
            st.write("请帮我写代码...")

# ==================================================
# Tab 4: 状态反馈 (Status)
# ==================================================
with tab4:
    st.header("4. 提示与反馈")
    
    with st.echo():
        st.success("操作成功 (Success)")
        st.info("这是一条提示信息 (Info)")
        st.warning("注意，配置可能不兼容 (Warning)")
        st.error("连接数据库失败 (Error)")
    
    st.divider()
    
    st.write("点击下面按钮模拟耗时操作：")
    if st.button("开始模拟计算"):
        with st.echo():
            # 这是一个转圈圈 Loading
            with st.spinner('AI 正在疯狂思考中...'):
                time.sleep(2) # 假装在干活
            st.toast("计算完成！", icon="🎉") # 右下角弹出

# ==================================================
# Tab 5: 聊天界面 (Chat - Agent 必备)
# ==================================================
with tab5:
    st.header("5. 聊天机器人界面")
    st.write("这是专门为 LLM 设计的组件。")
    
    with st.echo():
        # 1. 显示一条 AI 的消息
        with st.chat_message("assistant"):
            st.write("你好，我是你的 AI 助手，有什么可以帮你的吗？")
            
        # 2. 显示一条用户的消息
        with st.chat_message("user"):
            st.write("我想学习 Streamlit！")
            
        # 3. 再显示一条 AI 的消息
        with st.chat_message("assistant"):
            st.write("没问题，Streamlit 非常简单！")
            st.bar_chart([1, 2, 3, 4]) # 甚至可以在对话里画图
            
    st.divider()
    st.write("👇 这是一个输入框，但他还没接通逻辑（下一课讲）：")
    st.chat_input("在这里输入你的问题...")

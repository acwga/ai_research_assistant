"""
AI研究助手 - Streamlit前端界面
"""
import streamlit as st
from src.agent.stream_executor import StreamingResearchExecutor
from datetime import datetime

# 页面配置
st.set_page_config(page_title="AI 研究助手", layout="wide")
st.title("🔬 AI 研究助手")
st.markdown("输入研究问题，助手将自动规划任务、执行搜索并生成结构化报告。")

# 初始化会话状态
if "history" not in st.session_state:
    st.session_state["history"] = []    # 存储历史研究：{"query":, "report":, "steps":, "time":}

if "selected_report" not in st.session_state:
    st.session_state["selected_report"] = None

# 侧边栏：历史记录
with st.sidebar:
    st.header("📚 研究历史")
    if st.session_state["history"]:
        for i, item in enumerate(st.session_state["history"]):
            if st.button(f"{item['time']} - {item['query'][:30]}...", key=f"hist_{i}"):
                st.session_state["selected_report"] = item
                st.rerun()

    else:
        st.info("暂无历史记录")

    st.markdown("---")
    if st.button("清空历史"):
        st.session_state["history"] = []
        st.session_state["selected_report"] = None
        st.rerun()

# 主界面
with st.form("research_form"):
    user_query = st.text_input("研究问题", placeholder="例如：Transformer 和 BERT 在文本分类任务中的对比")
    submitted = st.form_submit_button("开始研究")

if submitted and user_query:
    # 清空之前显示的结果
    st.session_state["selected_report"] = None
    main_placeholder = st.empty()

    with main_placeholder.container():
        st.subheader("📋 规划步骤")
        step_container = st.empty()  # 用于显示步骤列表
        result_placeholders = []      # 用于存放每个步骤的结果占位符
        progress_bar = st.progress(0, text="准备执行...")

        # 创建执行器并开始流式执行
        executor = StreamingResearchExecutor()
        steps_displayed = False
        step_index = 0
        final_report = None

        for output in executor.execute(user_query):
            if output["type"] == "steps":
                # 显示步骤列表
                steps = output["steps"]
                with step_container.container():
                    for i, step in enumerate(steps, 1):
                        st.write(f"{i}. {step}")
                # 为每个步骤创建结果占位符
                result_placeholders = [st.empty() for _ in steps]
                steps_displayed = True

            elif output["type"] == "step_result":
                idx = output["index"] - 1
                step = output["step"]
                result = output["result"]
                # 更新对应步骤的结果占位符
                with result_placeholders[idx]:
                    st.markdown(f"**步骤 {idx+1} 结果:**")
                    st.write(result)
                # 更新进度
                progress_bar.progress(
                    (idx + 1) / len(result_placeholders),
                    text=f"执行步骤 {idx+1}/{len(result_placeholders)}"
                )

            elif output["type"] == "report":
                final_report = output["report"]
        
        # 执行完成，显示最终报告
        if final_report:
            st.subheader("📝 最终研究报告")
            st.markdown(final_report)

            # 保存到历史
            st.session_state["history"].append({
                "query": user_query,
                "report": final_report,
                "steps": steps if steps_displayed else [],
                "time": datetime.now().strftime("%Y-%m-%d %H:%M")
            })
            st.success("研究完成！")

# 显示选中的历史报告
if st.session_state["selected_report"]:
    item = st.session_state["selected_report"]
    st.subheader(f"📄 历史研究：{item['query']}")
    st.caption(item['time'])
    st.markdown(item['report'])
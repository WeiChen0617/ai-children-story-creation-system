import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from src.prompt_builder import PromptBuilder
from src.story_generator import StoryGenerator
from src.readability import ReadabilityAnalyzer
from src.feedback import FeedbackCollector
import uuid

st.set_page_config(page_title="AI儿童故事创作系统", layout="wide")
st.title("AI增强儿童故事创作系统")
st.markdown("""
> 面向5-8岁儿童，集成Prompt工程、可读性分析、教育性控制与用户反馈。
""")

# --- 侧边栏：参数输入 ---
st.sidebar.header("故事参数设置")
age = st.sidebar.slider("目标年龄", 5, 8, 6)
character = st.sidebar.text_input("主角（如小狐狸、小熊等）", "小狐狸")
theme = st.sidebar.text_input("教育主题（如合作、诚实、环保等）", "合作")
prompt_style = st.sidebar.selectbox("Prompt风格", ["模板式", "结构式", "问句式"])
word_limit = st.sidebar.number_input("字数上限", min_value=50, max_value=500, value=300, step=10)
model_choice = st.sidebar.selectbox("生成模型", ["gpt-4o", "claude-3", "gemini-1.5-pro"], index=0)
if model_choice == "gpt-4o":
    openai_model = "gpt-4o"
    backend = "openai"
elif model_choice == "claude-3":
    openai_model = None
    backend = "claude"
elif model_choice == "gemini-1.5-pro":
    openai_model = None
    backend = "gemini"
else:
    openai_model = None
    backend = None

# --- Prompt 构造 ---
prompt_builder = PromptBuilder(age=age)
if prompt_style == "模板式":
    prompt = prompt_builder.build_template_prompt(character, theme, word_limit)
elif prompt_style == "结构式":
    prompt = prompt_builder.build_structured_prompt(character, theme)
else:
    prompt = prompt_builder.build_question_prompt(theme)

st.subheader("生成的Prompt")
st.code(prompt, language="text")

# --- 故事生成 ---
story = ""
if st.button("生成故事"):
    with st.spinner("故事生成中，请稍候..."):
        if backend == "openai":
            generator = StoryGenerator(model="openai", openai_model=openai_model)
        else:
            generator = StoryGenerator(model=backend)
        story = generator.generate_story(prompt)
    st.success("故事生成完毕！")
    st.info(f"当前使用的模型：{model_choice}")

if story:
    st.subheader("AI生成的故事")
    st.write(story)

    # --- 可读性分析 ---
    analyzer = ReadabilityAnalyzer()
    readability = analyzer.analyze(story)
    st.subheader("可读性分析结果")
    st.json(readability)

    # --- 用户反馈 ---
    st.subheader("故事反馈")
    with st.form("feedback_form"):
        rating = st.slider("请为本故事打分（1-5分）", 1, 5, 5)
        comment = st.text_area("您的评价或建议", "")
        submitted = st.form_submit_button("提交反馈")
        if submitted:
            feedback_collector = FeedbackCollector()
            story_id = str(uuid.uuid4())
            path = feedback_collector.collect_feedback(
                story_id=story_id,
                rating=rating,
                comment=comment,
                extra={"theme": theme, "character": character, "age": age, "model": model_choice}
            )
            st.success(f"反馈已保存！(文件: {path})")

st.markdown("---")
st.caption("© 2024 AI增强儿童故事创作系统 | Powered by Streamlit & OpenAI API")

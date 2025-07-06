import streamlit as st
from dotenv import load_dotenv
load_dotenv()

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.core.prompt_builder import PromptBuilder
from src.core.story_generator import StoryGenerator
from src.core.readability import ReadabilityAnalyzer
from src.core.feedback import FeedbackCollector
import uuid

# 页面配置
st.set_page_config(page_title="故事生成 - AI儿童故事创作系统", layout="wide")

# 语言设置
if 'lang' not in st.session_state:
    st.session_state['lang'] = 'en'

def T(zh, en):
    return zh if st.session_state['lang']=='zh' else en

# 页面标题
title_col, lang_col = st.columns([8, 1])
with title_col:
    st.markdown(
        f"<h1 style='font-size:2.3em;margin-bottom:0.2em;'>{T('AI故事生成','AI Story Generation')}</h1>",
        unsafe_allow_html=True
    )
with lang_col:
    lang_map = {"中文": "zh", "English": "en"}
    lang_display = st.selectbox(
        "", 
        options=list(lang_map.keys()),
        index=1 if st.session_state.get('lang', 'en') == 'en' else 0
    )
    st.session_state['lang'] = lang_map[lang_display]

# 返回主页按钮
if st.button(T("返回主页", "Back to Home")):
    st.switch_page("streamlit_app.py")

col1, col2, col3 = st.columns([1, 2, 1.2], gap="large")

with col1:
    st.markdown(f"### {T('故事参数设置', 'Story Parameters')}")
    
    age = st.slider(T('目标年龄','Target Age'), 5, 8, 6)
    character = st.text_input(
        T('主角（如小狐狸、小熊等）','Main Character (e.g. Little Fox, Little Bear, etc.)'), 
        T('小狐狸','Little Fox')
    )
    theme = st.text_input(
        T('教育主题（如合作、诚实、环保等）','Educational Theme (e.g. Cooperation, Honesty, Environmental Protection, etc.)'), 
        T('合作','Cooperation')
    )
    prompt_style = st.selectbox(
        T('Prompt风格','Prompt Style'), 
        [T('模板式','Template'), T('结构式','Structured'), T('问句式','Question')]
    )
    word_limit = st.number_input(
        T('字数上限','Word Limit'), 
        min_value=50, max_value=500, value=300, step=10
    )
    model_choice = st.selectbox(
        T('生成模型','Model'), 
        [T('gpt-4o','gpt-4o')], # T('claude-3','claude-3'), T('gemini-1.5-pro','gemini-1.5-pro')], 
        index=0, 
        help=T('选择不同大模型进行故事生成','Choose different LLMs to generate stories')
    )
    
    # 模型配置
    if model_choice == "gpt-4o":
        openai_model = "gpt-4o"
        backend = "openai"
    # elif model_choice == "claude-3":
    #     openai_model = None
    #     backend = "claude"
    # elif model_choice == "gemini-1.5-pro":
    #     openai_model = None
    #     backend = "gemini"
    else:
        openai_model = "gpt-4o"  # 默认使用GPT-4o
        backend = "openai"
    
    st.markdown("---")
    st.markdown("<div style='height:40px;'></div>", unsafe_allow_html=True)

with col2:
    st.markdown(f"### {T('Prompt预览与故事生成', 'Prompt Preview & Story Generation')}")
    
    # Prompt生成
    prompt_builder = PromptBuilder(age=age, lang=st.session_state['lang'])
    if prompt_style == T('模板式','Template'):
        prompt = prompt_builder.build_template_prompt(character, theme, word_limit)
    elif prompt_style == T('结构式','Structured'):
        prompt = prompt_builder.build_structured_prompt(character, theme)
    else:
        prompt = prompt_builder.build_question_prompt(theme)
    
    # Prompt展示
    st.subheader(T('生成的Prompt','Generated Prompt'))
    with st.expander(T('查看完整Prompt', 'View Full Prompt'), expanded=False):
        st.markdown(f"{prompt}", unsafe_allow_html=True)
    
    # 故事生成
    story = ""
    btn_label = T('再生成故事','Regenerate Story') if "story" in st.session_state and st.session_state["story"] else T('生成故事','Generate Story')
    story_generated = False
    
    if st.button(btn_label, help=T('点击生成AI故事','Click to generate AI story'), use_container_width=True):
        with st.spinner(T('故事生成中，请稍候...','Generating story, please wait...')):
            try:
                if backend == "openai":
                    generator = StoryGenerator(model="openai", openai_model=openai_model)
                else:
                    generator = StoryGenerator(model=backend)
                story = generator.generate_story(prompt)
                st.session_state["story"] = story
                story_generated = True
                st.success(T('故事生成成功！', 'Story generated successfully!'))
            except Exception as e:
                st.error(T(f'故事生成失败：{str(e)}', f'Story generation failed: {str(e)}'))
    
    # 故事展示
    if "story" in st.session_state and st.session_state["story"]:
        st.markdown(f"<h3 style='font-size:1.15em;margin-bottom:0.5em;'>{T('AI生成的故事','AI Generated Story')}</h3>", unsafe_allow_html=True)
        
        # 故事内容展示
        story_container = st.container()
        with story_container:
            indent_style = "text-indent:2em;" if st.session_state['lang'] == 'zh' else "text-indent:0;"
            for para in st.session_state["story"].split("\n"):
                if para.strip():
                    st.markdown(
                        f"<p style='{indent_style}font-size:1.08em;line-height:1.7;margin-bottom:10px;color:#333;'>{para.strip()}</p>",
                        unsafe_allow_html=True
                    )
        
        # 故事操作按钮
        col_download, col_save = st.columns(2)
        with col_download:
            st.download_button(
                label=T('下载故事', 'Download Story'),
                data=st.session_state["story"],
                file_name=f"story_{character}_{theme}.txt",
                mime="text/plain"
            )
        with col_save:
            if st.button(T('保存到数据库', 'Save to Database')):
                try:
                    from src.core.data_manager import DataManager
                    data_manager = DataManager()
                    story_id = str(uuid.uuid4())
                    story_data = {
                        "story_id": story_id,
                        "content": st.session_state["story"],
                        "parameters": {
                            "age": age,
                            "character": character,
                            "theme": theme,
                            "prompt_style": prompt_style,
                            "word_limit": word_limit,
                            "model": model_choice
                        },
                        "prompt": prompt
                    }
                    data_manager.save_story(story_data)
                    st.success(T(f'故事已保存！ID: {story_id}', f'Story saved! ID: {story_id}'))
                except Exception as e:
                    st.error(T(f'保存失败：{str(e)}', f'Save failed: {str(e)}'))
    
    st.markdown("<div style='height:40px;'></div>", unsafe_allow_html=True)

with col3:
    story = st.session_state.get("story", "")

    # 可读性分析区块
    st.markdown(f"### {T('可读性分析','Readability Analysis')}")
    if story:
        try:
            analyzer = ReadabilityAnalyzer()
            readability = analyzer.analyze(story)
            
            # 主要指标展示
            met1, met2 = st.columns(2)
            with met1:
                st.metric("Flesch Reading Ease", f"{readability['Flesch Reading Ease']:.1f}")
                st.metric(T('推荐年龄段','Recommended Age Range'), readability["Recommended Age Range"])
            with met2:
                st.metric(T('句子数','Sentence Count'), readability["Sentence Count"])
                st.metric(T('词数','Word Count'), readability["Word Count"])
            
            # 详细分析
            with st.expander(T('详细可读性分析','Detailed Readability Analysis')):
                st.json(readability)
                
            # 可读性建议
            flesch_score = readability['Flesch Reading Ease']
                
        except Exception as e:
            st.error(T(f'可读性分析失败：{str(e)}', f'Readability analysis failed: {str(e)}'))
    else:
        met1, met2 = st.columns(2)
        with met1:
            st.metric("Flesch Reading Ease", "--")
            st.metric(T('推荐年龄段','Recommended Age Range'), "--")
        with met2:
            st.metric(T('句子数','Sentence Count'), "--")
            st.metric(T('词数','Word Count'), "--")
        st.caption(T('生成故事后可查看可读性分析','You can view readability analysis after generating a story'))
    
    st.markdown("<div style='height:40px;'></div>", unsafe_allow_html=True)

# 页脚
st.markdown("---")
st.markdown(
    f"<div style='text-align: center; color: #666; margin-top: 1rem;'>{T('提示：尝试不同的参数组合来生成更有趣的故事！', 'Tip: Try different parameter combinations to generate more interesting stories!')}</div>",
    unsafe_allow_html=True
)
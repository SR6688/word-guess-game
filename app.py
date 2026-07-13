import streamlit as st
import spacy
import random
import os

# --- 1. 配置与模型加载 ---
@st.cache_resource
def load_model():
    # 注意：在 Streamlit Cloud，模型下载需要时间，这行代码在第一次部署时会较慢
    return spacy.load("zh_core_web_md")

nlp = load_model()

# --- 2. 词库读取 (从预处理好的文件读取) ---
@st.cache_data
def get_word_list():
    if os.path.exists("filtered_words.txt"):
        with open("filtered_words.txt", "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    else:
        # 兜底方案：如果文件丢失，给一个基础词库
        return ["苹果", "香蕉", "电脑", "手机", "人工智能", "编程"]

# --- 3. 游戏逻辑 ---
def reset_game():
    words = get_word_list()
    st.session_state.target_word = random.choice(words)
    st.session_state.guesses = []
    st.session_state.game_won = False

# 初始化
if 'target_word' not in st.session_state:
    reset_game()

# --- 4. 界面布局 ---
st.title("词语猜猜看游戏")

st.sidebar.title("游戏控制")
if st.sidebar.button("显示答案"):
    st.sidebar.warning(f"当前目标词是: {st.session_state.target_word}")

if st.sidebar.button("开始新一局"):
    reset_game()
    st.rerun()

# --- 5. 猜测逻辑 ---
user_input = st.text_input("请输入你猜的词：")

if st.button("提交猜测"):
    if user_input:
        # 检查词语在模型中是否有向量 (兼容性检查)
        guess_token = nlp(user_input)
        
        if guess_token.vector_norm == 0:
            st.error(f"抱歉，我不认识 '{user_input}' 这个词，请换一个试试！")
        elif user_input == st.session_state.target_word:
            st.session_state.game_won = True
            st.success(f"恭喜！猜对了！答案就是：{st.session_state.target_word}")
        else:
            target_token = nlp(st.session_state.target_word)
            sim = target_token.similarity(guess_token)
            st.session_state.guesses.append((user_input, sim))
            # 排序并取前5
            st.session_state.guesses.sort(key=lambda x: x[1], reverse=True)
            st.session_state.guesses = st.session_state.guesses[:5]

# --- 6. 展示记录 ---
if st.session_state.guesses:
    st.write("--- 历史猜测记录 (Top 5) ---")
    for word, sim in st.session_state.guesses:
        st.write(f"词语: **{word}** | 相似度: {sim:.4f}")

if st.session_state.get('game_won', False):
    if st.button("再来一局"):
        reset_game()
        st.rerun()
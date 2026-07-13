import streamlit as st
import spacy
from spacy.lang.zh import Chinese
from spacy.vectors import Vectors
import os
import random

@st.cache_resource
def load_model():
    try:
        # 1. 创建一个空的中文语言对象，不加载任何导致崩溃的管道组件
        nlp = Chinese()
        
        # 2. 手动指定模型路径 (通常在 site-packages/zh_core_web_md 下)
        # 如果你不知道绝对路径，让 spacy 报错并从日志中读取该路径
        model_path = "/home/adminuser/venv/lib/python3.14/site-packages/zh_core_web_md/zh_core_web_md-3.8.0"
        
        # 3. 直接加载向量表，跳过所有模型配置检查
        # 这绕过了对 pkuseg 的所有依赖
        vec_path = os.path.join(model_path, "vectors")
        nlp.vocab.vectors.from_disk(vec_path)
        
        return nlp
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        # 如果上方路径不匹配，查看日志中显示的路径并修正
        return None

nlp = load_model()

# --- 2. 词库读取 ---
@st.cache_data
def get_word_list():
    if os.path.exists("filtered_words.txt"):
        with open("filtered_words.txt", "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    else:
        # 兜底词库
        return ["苹果", "香蕉", "电脑", "手机", "人工智能", "编程", "宇宙", "大海"]

# --- 3. 游戏逻辑 ---
def reset_game():
    words = get_word_list()
    st.session_state.target_word = random.choice(words)
    st.session_state.guesses = []
    st.session_state.game_won = False

# 初始化 Session State
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
    if nlp is None:
        st.error("模型未正确加载，无法进行计算。")
    elif user_input:
        guess_token = nlp(user_input)
        
        # 检查是否为停用词或词库外词汇
        if guess_token.vector_norm == 0:
            st.error(f"抱歉，我不认识 '{user_input}' 这个词，请换一个试试！")
        elif user_input == st.session_state.target_word:
            st.session_state.game_won = True
            st.success(f"恭喜！猜对了！答案就是：{st.session_state.target_word}")
        else:
            target_token = nlp(st.session_state.target_word)
            sim = target_token.similarity(guess_token)
            st.session_state.guesses.append((user_input, sim))
            # 排序：按相似度从高到低
            st.session_state.guesses.sort(key=lambda x: x[1], reverse=True)
            st.session_state.guesses = st.session_state.guesses[:5]

# --- 6. 展示记录 ---
if st.session_state.guesses:
    st.write("--- 历史猜测记录 (Top 5) ---")
    for word, sim in st.session_state.guesses:
        st.write(f"词语: **{word}** | 相似度: {sim:.4f}")

if st.session_state.game_won:
    if st.button("再来一局"):
        reset_game()
        st.rerun()
import os
from typing import Dict, List, TypedDict
from langgraph.graph import StateGraph, END
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import numpy as np
import markdown
import re
import html
import unicodedata


# 配置 Gemini API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# 加載 embedding 模型
embedder = SentenceTransformer('all-MiniLM-L6-v2')


# 定義狀態
class State(TypedDict):
    question: str
    retrieved_docs: List[str]
    answer: str


def sanitize_input(text: str) -> str:
    """清理輸入"""
    if not isinstance(text, str):
        return ""
    text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'Cs')
    text = re.sub(r'<[^>]*>', '', text)
    text = html.escape(text)
    text = re.sub(r'[^\w\s\u4e00-\u9fa5]', '', text)
    text = ' '.join(text.split())
    cleaned = text.encode('utf-8', errors='replace').decode('utf-8')
    return cleaned


# 讀取並解析 markdown 文件
def load_markdown_with_tags(file_path: str) -> Dict[str, str]:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    html = markdown.markdown(content)
    lines = content.split('\n')
    docs = {}
    current_tag = None
    current_content = []

    for line in lines:
        if line.startswith('# '):
            if current_tag and current_content:
                docs[current_tag] = '\n'.join(current_content).strip()
            current_tag = line[2:].strip()
            current_content = []
        elif current_tag:
            current_content.append(line)

    if current_tag and current_content:
        docs[current_tag] = '\n'.join(current_content).strip()

    return docs


# 計算文本的 embedding
def get_embedding(text: str) -> np.ndarray:
    return embedder.encode(text)


# 使用 LLM 檢查是否是列出 API 的問題
def is_list_api_question(question: str) -> bool:
    prompt = f"""
    判斷以下問題是否在要求列出文件中的所有 API 名稱或統計 API 數量，或文件所有內容。
    問題: {question}
    如果是，返回 "yes"；如果不是，返回 "no"。
    只返回 "yes" 或 "no"，不要有多餘文字。
    """
    response = model.generate_content(prompt)
    return response.text.strip().lower() == "yes"


# 使用 LLM 檢查是否是詢問文件概述的問題
def is_file_summary_question(question: str) -> bool:
    prompt = f"""
    判斷以下問題是否在詢問文件的整體內容、目的或用途。
    問題: {question}
    如果是，返回 "yes"；如果不是，返回 "no"。
    只返回 "yes" 或 "no"，不要有多餘文字。
    """
    response = model.generate_content(prompt)
    return response.text.strip().lower() == "yes"


# 檢查問題中是否提到某個 API 名稱
def extract_api_from_question(question: str, api_list: List[str]) -> List[str]:
    prompt = f"""
    根據以下問題和 API 列表，判斷問題中提到了哪些 API，找出最相關的就好。
    問題: {question}
    API 列表: {api_list}
    請返回問題中提到的 API 名稱列表，以逗號分隔。如果沒有提到任何 API，返回 "None"。
    """
    response = model.generate_content(prompt)
    api_names = response.text.strip()

    if api_names.lower() == "none":
        return []
    else:
        return [name.strip() for name in api_names.split(",")]


# 檢索相關內容
def retrieve(state: State) -> State:
    question = sanitize_input(state['question'])
    docs = load_markdown_with_tags("KEYPO功能手冊文件.md")
    api_list = list(docs.keys())

    # 如果是要求列出所有 API，直接返回完整列表
    if is_list_api_question(question):
        state["retrieved_docs"] = [
            f"文件中的所有 API ({len(api_list)} 個):\n" + "\n".join(api_list)]
        return state

    # 如果是詢問文件概述，提供所有 API 的簡要內容
    if is_file_summary_question(question):
        summary_docs = [
            f"API: {tag}\n內容簡述: {docs[tag][:100]}..." for tag in api_list]  # 取前100字作為簡述
        state["retrieved_docs"] = ["\n\n".join(summary_docs)]
        return state

    # 檢查問題是否提到某個 API
    exact_apis = extract_api_from_question(question, api_list)
    if exact_apis:
        retrieved_docs = []
        for api in exact_apis:
            if api in docs:  # 確保 API 名稱在 docs 字典中存在
                retrieved_docs.append(f"API: {api}\n內容: {docs[api]}")
        state["retrieved_docs"] = retrieved_docs
        return state

    # 否則進行 RAG 檢索
    question_emb = get_embedding(question)
    similarities = {}
    for tag, content in docs.items():
        content_emb = get_embedding(content)
        similarity = np.dot(question_emb, content_emb) / \
            (np.linalg.norm(question_emb) * np.linalg.norm(content_emb))
        similarities[tag] = similarity

    # 選取相似度最高的前 3 個
    top_docs = sorted(similarities.items(),
                      key=lambda x: x[1], reverse=True)[:3]
    retrieved = [f"API: {tag}\n內容: {docs[tag]}" for tag, _ in top_docs]

    # 反向檢索：查詢內容後輸出 API 名稱
    content_similarities = {}
    for tag, content in docs.items():
        content_emb = get_embedding(content)
        content_similarity = np.dot(question_emb, content_emb) / \
            (np.linalg.norm(question_emb) * np.linalg.norm(content_emb))
        content_similarities[content] = content_similarity

    # 選取內容相似度最高的前 3 個
    top_contents = sorted(content_similarities.items(),
                          key=lambda x: x[1], reverse=True)[:3]
    retrieved_by_content = []
    for content, _ in top_contents:
        for tag, doc_content in docs.items():
            if doc_content == content:
                retrieved_by_content.append(f"內容: {content}\nAPI: {tag}")
                break

    # 關鍵字檢索增強
    keyword_matches = []
    for tag, content in docs.items():
        # 檢查標題和內容是否包含查詢中的所有詞彙
        question_words = question.split()
        if all(word in (tag + " " + content) for word in question_words):
            keyword_matches.append(f"API: {tag}\n內容: {docs[tag]}")

    # 合併檢索結果，優先使用關鍵字匹配
    if keyword_matches:
        state["retrieved_docs"] = keyword_matches
    else:
        state["retrieved_docs"] = retrieved + retrieved_by_content if retrieved or retrieved_by_content else [
            "知識庫中無相關內容"]
    return state


# 生成回答
def generate(state: State) -> State:
    question = sanitize_input(state['question'])
    retrieved_docs = "\n\n".join(state["retrieved_docs"])

    # 檢查問題是否與檢索到的內容相關
    relevance_check_prompt = f"""
    判斷以下問題是否與提供的文件內容有直接關聯，文件內容如下，如果和文件無關，請回答 "no"。
    ### 這裡是問題 ###
    {question}
    ### 文件內容 ###
    {retrieved_docs}
    ### 注意事項和回答守則 ###
    1. 如果問題中包含「讚」，請回答"no"。
    2. 如果問題與文件內容有直接關聯，回答 "yes"；否則回答 "no"。
    3. 只需要回答 "yes" 或 "no"，不要有多餘文字。
    4. 如果詢問文章的相關資訊，回答yes。
    """
    relevance_check_response = model.generate_content(relevance_check_prompt)
    if relevance_check_response.text.strip().lower() != "yes":
        state["answer"] = "對不起，這個問題與文件內容無關。"
        return state

    # 如果問題相關，則生成回答
    prompt = f"""
    根據以下文件內容回答問題，如果文件中無相關內容，請明確說「無法回答」。
    如果問題是關於文件的整體內容或用途，請提供一個全面的概述，描述文件的主題和主要功能。
    切記！只能用繁體中文回答。
    問題: {question}
    文件內容:
    {retrieved_docs}
    """
    response = model.generate_content(prompt)
    state["answer"] = response.text.strip()
    return state


# 構建 LangGraph 流程
workflow = StateGraph(State)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)
workflow.set_entry_point("retrieve")
app = workflow.compile()


# 列出所有 API
def list_apis():
    docs = load_markdown_with_tags("KEYPO功能手冊文件.md")
    return list(docs.keys())


if __name__ == "__main__":
    print("歡迎使用文件檢索聊天機器人！輸入 'exit' 可退出。")
    while True:
        question = input("請輸入問題：")
        if question.lower() == 'exit':
            print("感謝使用，再見！")
            break
        result = app.invoke(
            {"question": question, "retrieved_docs": [], "answer": ""})
        print(result["answer"])

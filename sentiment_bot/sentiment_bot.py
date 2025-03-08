import requests
from bs4 import BeautifulSoup
from datetime import datetime
import google.generativeai as genai
import urllib.parse
import os
from typing import TypedDict, Annotated, List, Dict
from langgraph.graph import StateGraph, END

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# 定義狀態結構


class SentimentState(TypedDict):
    question: str
    is_related: bool
    keywords: str
    articles: List[Dict]
    analyses: List[Dict]
    response: str


# 初始化 Gemini 模型
llm = genai.GenerativeModel('gemini-1.5-flash')


# 節點 1：檢查是否與輿情分析相關
def check_sentiment_related(state: SentimentState) -> SentimentState:
    prompt = f"""請判斷以下問題是否與輿情分析相關（例如涉及新聞、媒體、情緒、報導等主題）。
    如果相關，返回 "是"；如果無關，返回 "否"。
    問題：{state['question']}
    """
    try:
        response = llm.generate_content(prompt)
        result = response.text.strip()
        state['is_related'] = (result == "是")
    except Exception as e:
        print(f"判斷輿情相關性失敗: {e}")
        state['is_related'] = False
    return state


# 節點 2：提取新聞關鍵字
def extract_keywords(state: SentimentState) -> SentimentState:
    if not state['is_related']:
        return state
    prompt = f"""請分析以下問題，提取出使用者想要查找的新聞相關關鍵字。
    返回單一關鍵字或短語（繁體中文），避免包含標點符號或無關詞彙。
    若無法確定，返回問題中最核心的詞彙。
    問題：{state['question']}
    """
    try:
        response = llm.generate_content(prompt)
        state['keywords'] = response.text.strip()
    except Exception as e:
        print(f"關鍵字提取失敗: {e}")
        state['keywords'] = state['question'].split(
        )[-1] if state['question'] else ""
    return state


# 節點 3：抓取新聞
def fetch_news(state: SentimentState) -> SentimentState:
    if not state['is_related'] or not state['keywords']:
        return state

    query = state['keywords'].encode('utf-8', errors='replace').decode('utf-8')
    query = urllib.parse.quote_plus(
        f"{query} site:*.tw | site:*.com -inurl:(login | signup)")
    url = f"https://news.google.com/rss/search?q={query}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"新聞請求失敗: {e}")
        state['articles'] = []
        return state

    soup = BeautifulSoup(response.content, 'xml')
    items = soup.find_all('item')[:3]  # 限制為 3 篇

    articles = []
    for item in items:
        try:
            title = item.find('title').text
            link = item.find('link').text
            pub_date = item.find('pubDate').text if item.find(
                'pubDate') else '未知時間'
            description = item.find('description').text if item.find(
                'description') else None

            article_response = requests.get(link, headers=headers, timeout=10)
            article_soup = BeautifulSoup(article_response.text, 'html.parser')

            content_elements = (
                article_soup.select('p') or
                article_soup.select('div[class*=content]') or
                article_soup.select('article p')
            )
            content = ' '.join(elem.text.strip()
                               for elem in content_elements[:3] if elem.text.strip())

            if not content and description:
                content = BeautifulSoup(
                    description, 'html.parser').text.strip()

            content = content or "無法獲取內文（可能是動態加載或網站限制）"

            articles.append({
                'title': title,
                'pub_time': pub_date,
                'content': content,
                'link': link
            })
        except Exception as e:
            print(f"解析單篇新聞失敗: {e}")
            continue

    state['articles'] = articles
    return state


# 節點 4：分析新聞內容
def analyze_content(state: SentimentState) -> SentimentState:
    if not state['is_related'] or not state['articles']:
        return state

    analyses = []
    for article in state['articles']:
        # 情緒分析
        sentiment_prompt = f"""請分析以下中文文本的情緒，並返回 "正向"、"負向" 或 "中性"。
        文本：{article['content']}
        """
        try:
            sentiment_response = llm.generate_content(sentiment_prompt)
            sentiment = sentiment_response.text.strip()
            if sentiment not in ['正向', '負向', '中性']:
                sentiment = '中性'
        except Exception as e:
            print(f"情緒分析失敗: {e}")
            sentiment = '中性'

        # NER
        ner_prompt = f"""請對以下中文文本進行命名實體識別（NER），並以以下格式返回結果：
                        - 組織 (ORG)
                        - 人物 (PERSON)
                        - 地點 (LOC)
                        - 日期 (DATE)
                        - 時間 (TIME)
                        - 貨幣 (MONEY)
                        - 數字 (NUM)
                        - 事件 (EVENT)
                        只返回識別出的實體，每行一個，格式為 "標籤: 實體"。若無實體，則返回 "無"。
                        {article['content']}
                    """
        try:
            ner_response = llm.generate_content(ner_prompt)
            ner_text = ner_response.text.strip()
            entities = ner_text.split('\n')
            entities = [e.strip() for e in entities if e.strip() and ':' in e]
            if not entities or entities == ["無"]:
                entities = []
            for i, entity in enumerate(entities):
                label, text = entity.split(': ', 1)
                if text in {'PChome Online', 'Yahoo新聞', '自由時報'} and label == 'PERSON':
                    entities[i] = f"ORG: {text}"
        except Exception as e:
            print(f"NER 生成失敗: {e}")
            entities = []

        # 摘要
        summary_prompt = f"請將以下新聞內文總結成100字以內的摘要（繁體中文）：\n{article['content']}"
        try:
            summary_response = llm.generate_content(summary_prompt)
            summary = summary_response.text.strip()
        except Exception as e:
            print(f"生成摘要失敗: {e}")
            summary = "無法生成摘要"

        analyses.append({
            'sentiment': sentiment,
            'entities': entities,
            'summary': summary,
            'title': article['title'],
            'pub_time': article['pub_time'],
            'link': article['link']
        })

    state['analyses'] = analyses
    return state

# 節點 5：格式化回應


def format_response(state: SentimentState) -> SentimentState:
    if not state['is_related']:
        state['response'] = "抱歉，我只能回答與輿情分析相關的問題。"
        return state

    if not state['articles'] or not state['analyses']:
        state['response'] = f"抱歉，無法找到與「{state['keywords']}」相關的新聞，請嘗試更具體的關鍵詞。"
        return state

    response = f"分析時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    for i, analysis in enumerate(state['analyses'], 1):
        response += f"新聞 {i}:\n"
        response += f"標題：{analysis['title']}\n"
        response += f"發佈時間：{analysis['pub_time']}\n"
        response += f"摘要：{analysis['summary']}\n"
        response += f"情緒分析：{analysis['sentiment']}\n"
        response += f"命名實體：{', '.join(analysis['entities']) if analysis['entities'] else '無'}\n"
        response += f"來源：{analysis['link']}\n\n"

    state['response'] = response.strip()
    return state


# 構建工作流
workflow = StateGraph(SentimentState)

# 添加節點
workflow.add_node("check_sentiment_related", check_sentiment_related)
workflow.add_node("extract_keywords", extract_keywords)
workflow.add_node("fetch_news", fetch_news)
workflow.add_node("analyze_content", analyze_content)
workflow.add_node("format_response", format_response)

# 設置入口點
workflow.set_entry_point("check_sentiment_related")

# 添加邊
workflow.add_edge("check_sentiment_related", "extract_keywords")
workflow.add_edge("extract_keywords", "fetch_news")
workflow.add_edge("fetch_news", "analyze_content")
workflow.add_edge("analyze_content", "format_response")
workflow.add_edge("format_response", END)

# 編譯工作流
app = workflow.compile()


def process_query(question: str) -> str:
    initial_state = {
        "question": question,
        "is_related": False,
        "keywords": "",
        "articles": [],
        "analyses": [],
        "response": ""
    }
    result = app.invoke(initial_state)
    return result["response"]


if __name__ == "__main__":
    print("歡迎使用輿情分析機器人！輸入 'exit' 可退出。")
    while True:
        question = input("請輸入問題：")
        if question.lower() == 'exit':
            print("感謝使用，再見！")
            break
        result = process_query(question)
        print(result)

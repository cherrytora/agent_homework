import requests
from bs4 import BeautifulSoup
from langchain.prompts import PromptTemplate
from datetime import datetime
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import re
import google.generativeai as genai
import urllib.parse
import spacy  # 新增spaCy
import os

# 初始化NLTK資源（僅用於情緒分析）
nltk.download('vader_lexicon')
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


class SentimentBot:
    def __init__(self):
        genai.configure(api_key=GOOGLE_API_KEY)
        self.llm = genai.GenerativeModel('gemini-2.0-flash')

        # 加載spaCy的英文和中文模型
        self.nlp_en = spacy.load("en_core_web_sm")
        self.nlp_zh = spacy.load("zh_core_web_lg")

        self.template = """你是一個專門處理輿情分析的聊天機器人，只能回答與輿情分析相關的問題。
        如果問題與輿情分析無關，請回應：「抱歉，我只能回答與輿情分析相關的問題。」
        當回答輿情分析問題時，請提供結構化的分析結果，包括：
        - 新聞標題
        - 發佈時間
        - 內文摘要
        - 情緒分析結果
        - 命名實體識別（NER）結果
        
        使用者問題：{input}
        回應："""

        self.prompt = PromptTemplate(
            input_variables=["input"],
            template=self.template
        )

        self.sid = SentimentIntensityAnalyzer()

    def is_sentiment_related(self, question):
        keywords = ['輿情', '新聞', '分析', '情緒', '媒體', '報導', '評論', '趨勢']
        return any(keyword in question for keyword in keywords)

    def fetch_news(self, query, num_articles=3):
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
            return []

        soup = BeautifulSoup(response.content, 'xml')
        items = soup.find_all('item')[:num_articles]

        articles = []
        for item in items:
            try:
                title = item.find('title').text
                link = item.find('link').text
                pub_date = item.find('pubDate').text if item.find(
                    'pubDate') else '未知時間'
                description = item.find('description').text if item.find(
                    'description') else None

                article_response = requests.get(
                    link, headers=headers, timeout=10)
                article_soup = BeautifulSoup(
                    article_response.text, 'html.parser')

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

        return articles

    def analyze_content(self, content, title=None):
        # 情緒分析
        sentiment_scores = self.sid.polarity_scores(content)
        sentiment = '正向' if sentiment_scores['compound'] > 0.05 else \
                    '負向' if sentiment_scores['compound'] < -0.05 else '中性'

        # 使用中文模型進行NER
        doc_zh = self.nlp_zh(content)

        # 提取實體
        entities = []
        for ent in doc_zh.ents:
            entities.append(f"{ent.label_}: {ent.text}")

        # 後處理：修正常見錯誤並增強識別
        cleaned_entities = []
        seen_entities = set()  # 避免重複

        for entity in entities:
            label, text = entity.split(': ', 1)
            # 過濾過長或不合理的實體（例如標題）
            if len(text) > 15 or ' ' in text.strip() or text == title:
                continue
            cleaned_entities.append(f"{label}: {text}")
            seen_entities.add(text)

        # 自動識別職位（基於詞性和上下文）
        for token in doc_zh:
            if token.text not in seen_entities:
                # 職位：名詞 + '長'、'總'、'監' 等
                if token.pos_ == 'NOUN' and any(token.text.endswith(suffix) for suffix in ['長', '總', '監', '理']):
                    cleaned_entities.append(f"ROLE: {token.text}")
                    seen_entities.add(token.text)
                # 事件關鍵詞：動詞或負面名詞（簡化版）
                elif token.pos_ in ('VERB', 'NOUN') and token.text in content and len(token.text) >= 2:
                    # 這裡僅簡單示範，可根據詞頻或語義進一步過濾
                    cleaned_entities.append(f"KEYWORD: {token.text}")
                    seen_entities.add(token.text)

            # 使用Gemini生成摘要
            summary_prompt = f"請將以下新聞內文總結成100字以內的摘要（繁體中文）：\n{content}"
            try:
                summary_response = self.llm.generate_content(summary_prompt)
                summary = summary_response.text.strip()
            except Exception as e:
                print(f"生成摘要失敗: {e}")
                summary = "無法生成摘要"

            return {
                'sentiment': sentiment,
                'entities': cleaned_entities,
                'summary': summary
            }

    def process_query(self, question):
        if not self.is_sentiment_related(question):
            return "抱歉，我只能回答與輿情分析相關的問題。"

        query = re.sub(r'[？！。]', '', question)
        articles = self.fetch_news(query)

        if not articles:
            return "抱歉，無法找到相關新聞，請嘗試更具體的關鍵詞。"

        response = f"分析時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        for i, article in enumerate(articles, 1):
            analysis = self.analyze_content(article['content'])
            response += f"新聞 {i}:\n"
            response += f"標題：{article['title']}\n"
            response += f"發佈時間：{article['pub_time']}\n"
            response += f"摘要：{analysis['summary']}\n"
            response += f"情緒分析：{analysis['sentiment']}\n"
            response += f"命名實體：{', '.join(analysis['entities']) if analysis['entities'] else '無'}\n"
            response += f"來源：{article['link']}\n\n"

        return response


if __name__ == "__main__":
    bot = SentimentBot()
    print("歡迎使用輿情分析機器人！輸入 'exit' 可退出。")
    while True:
        question = input("請輸入問題：")
        if question.lower() == 'exit':
            print("感謝使用，再見！")
            break
        result = bot.process_query(question)
        print(result)

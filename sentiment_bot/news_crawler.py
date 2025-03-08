import requests
from bs4 import BeautifulSoup
from datetime import datetime


class NewsCrawler:
    def __init__(self):
        self.base_url = "https://news.google.com/search"

    def fetch_news(self, query, limit=5):
        params = {"q": query, "hl": "zh-TW", "gl": "TW"}
        response = requests.get(self.base_url, params=params)
        soup = BeautifulSoup(response.text, "html.parser")
        articles = []

        for item in soup.select("article")[:limit]:
            title = item.select_one(
                "h3").text if item.select_one("h3") else "N/A"
            link = "https://news.google.com" + item.select_one("a")["href"][1:]
            time = item.select_one("time")
            pub_time = datetime.strptime(
                time["datetime"], "%Y-%m-%dT%H:%M:%SZ") if time else None
            articles.append(
                {"title": title, "link": link, "pub_time": pub_time})
        return articles

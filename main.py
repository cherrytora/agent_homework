from sentiment_bot.sentiment_bot import app as sentiment_app, process_query
from rag_bot.rag_bot import app as rag_app, list_apis
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "rag_bot"))
sys.path.append(os.path.join(os.path.dirname(__file__), "sentiment_bot"))


def run_rag_bot():
    print("進入 RAG Bot 模式！輸入 'exit' 返回主選單，輸入 'list apis' 列出所有 API。")
    while True:
        question = input("請輸入問題：")
        if question.lower() == 'exit':
            print("離開 RAG Bot 模式。")
            break
        elif question.lower() == 'list apis':
            apis = list_apis()
            print(f"文件中的所有 API ({len(apis)} 個):\n" + "\n".join(apis))
        else:
            result = rag_app.invoke(
                {"question": question, "retrieved_docs": [], "answer": ""})
            print(result["answer"])


def run_sentiment_bot():
    print("進入 Sentiment Bot 模式！輸入 'exit' 返回主選單。")
    while True:
        question = input("請輸入問題：")
        if question.lower() == 'exit':
            print("離開 Sentiment Bot 模式。")
            break
        process_query(question)


def main():
    print("歡迎使用 Bot 切換系統！")
    print("可用指令：'rag' 進入 RAG Bot，'sentiment' 進入 Sentiment Bot，'exit' 退出程式。")

    while True:
        choice = input("請選擇模式 (rag/sentiment/exit)：").lower()
        if choice == 'rag':
            run_rag_bot()
        elif choice == 'sentiment':
            run_sentiment_bot()
        elif choice == 'exit':
            print("感謝使用，再見！")
            break
        else:
            print("無效指令，請輸入 'rag'、'sentiment' 或 'exit'。")


if __name__ == "__main__":
    main()

# 使用說明
### 建立環境 
```
docker build -t bot-system .
docker run -it -e GOOGLE_API_KEY={YOUR API KEY} bot-system
```
起始畫面如下
![image](img/起始畫面.png)
### 起始指令
輸入`rag`進入文件檢索聊天機器人，輸入`sentiment`進入輿情分析聊天機器人，輸入`exit`離開程式
#### 文件檢索聊天機器人
1. 輸入問題
2. 輸入`list apis`會列出文件中所有API
3. 輸入`exit`返回主選單
- 起始畫面如下
![image](img/文件檢索起始畫面.png)
#### 輿情分析聊天機器人
1. 輸入問題
2. 輸入`exit`返回主選單
- 起始畫面如下
![image](img/輿情分析起始畫面.png)


# code structure
```
root/
├── img/
│   └── Sentiment_flow.svg
└── rag_bot/
    ├── KEYPO功能手冊文件.md
    └── rag_bot.py
└── sentiment_bot/
    └── sentiment_bot.py
├── .env
├── .gitignore
├── dockerfile
├── README.md
└── requirements.txt
```

## 輿情分析機器人
### 篩選了幾種
NER和情感分析Model選取，嘗試了以下幾種，最後綜合表現由`Gemini本人`勝出  
1. spaCy（zh-core-web-sm, zh-core-web-md, zh-core-web-lg, zh-core-web-trf）
2. Transformers(Bert) - huggingface
3. Flair
4. nltk（沒有支援中文）
5. Gemini本人

### 流程說明
<img src="img/Sentiment_flow.svg" alt="輿情分析folw" width="30%">

1. **A[Start]**：工作流開始，接收使用者輸入的問題。
2. **B[check_sentiment_related]**：檢查問題是否與輿情分析相關。
   - 若「是」，進入 **C[extract_keywords]**。
   - 若「否」，直接跳到 **F[format_response]**，返回「抱歉」訊息。
3. **C[extract_keywords]**：提取新聞關鍵字。
4. **D[fetch_news]**：根據關鍵字抓取新聞。
5. **E[analyze_content]**：分析新聞內容（情緒、NER、摘要）。
6. **F[format_response]**：格式化最終回應。
7. **G[End]**：工作流結束，輸出結果。

## Rag 機器人
### 文件怎麼切
把每個大標下的內容切開，順便把大標當成tag。搜尋的時候tag和內容對照搜過一遍。



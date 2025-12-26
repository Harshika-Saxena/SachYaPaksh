from newspaper import Article
from newsapi import NewsApiClient
import json
from datetime import datetime

KEY = "32de97bd379c4b5e9a20c80bc3da3a2a"
newsapi = NewsApiClient(api_key=KEY)

TOPICS = [ "politics", "gender equality", "religion", "crime", "terrorism", "LGBTQ", "feminism"]

def get_articles(save_to_json=True):
    articles = {}

    for topic in TOPICS:
        print(f"Fetching articles for topic: {topic}")
        response = newsapi.get_everything(q=topic, language='en', sort_by='relevancy', page_size=5)

        for entry in response['articles']:
            url = entry['url']
            try:
                article = Article(url)
                article.download()
                article.parse()
                
                articles[url] = {"source": entry['source']['name'],"topic": topic,"title": article.title,"content": article.text}

                print(f"Added article: {article.title}")

            except Exception as e:
                print(f"Error processing {url}: {e}")

    if save_to_json:
        filename = "articles.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(articles, f, ensure_ascii=False, indent=4)
        print(f"Articles saved to {filename}")

    return articles


if __name__ == "__main__":
    get_articles()

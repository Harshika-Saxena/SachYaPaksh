import json
from nltk.tokenize import sent_tokenize
import nltk
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, pipeline


nltk.download('punkt')

model_name = "d4data/bias-detection-model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSequenceClassification.from_pretrained(model_name)

classifier = pipeline('text-classification', model=model, tokenizer=tokenizer)


def is_biased(text):
    if not text.strip():
        return False
    result = classifier(text)[0] 
    label = result['label']

    return label == "Biased"


with open("articles.json", "r", encoding="utf-8") as f:
    articles = json.load(f)

biased_articles = {}
biased_articles_count = 0



for url, article in articles.items():
    content = article.get("content", "")
    if not content.strip():
        continue

    sentences = sent_tokenize(content)
    biased_count = sum(1 for s in sentences if is_biased(s))

    if biased_count > 11:
        biased_articles_count += 1
        biased_articles[url] = {
            "title": article.get("title", ""),
            "source": article.get("source", ""),
            "topic": article.get("topic", ""),
            "biased_sentences_count": biased_count,
            "content": content
        }


with open("biased_articles_filtered.json", "w", encoding="utf-8") as f:
    json.dump(biased_articles, f, ensure_ascii=False, indent=4)

print(f"\n{biased_articles_count} articles saved to biased_articles_filtered.json")

from flask import Flask, render_template, request, jsonify
import json
import os
import nltk
import pytesseract
from PIL import Image
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, pipeline


nltk.download('punkt')

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def load_biased_articles():
    try:
        with open("biased_articles_filtered.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


model_name = "d4data/bias-detection-model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

def predict_bias(text):
    if not text.strip():
        return "Neutral", 0.0
    result = classifier(text)[0] 
    label = "Biased" if result["label"] == "Biased" else "Neutral"
    return label, round(result["score"] * 100, 2)

@app.route("/")
def home():
    articles = load_biased_articles()
    return render_template("index.html", articles=articles)

@app.route("/analyze", methods=["GET", "POST"])
def analyze():
    analysis_results = []
    user_text = ""
    if request.method == "POST":
        user_text = request.form.get("user_text", "").strip()
        if user_text:
            sentences = sent_tokenize(user_text)
            for sentence in sentences:
                label, prob = predict_bias(sentence)
                analysis_results.append({
                    "sentence": sentence,
                    "label": label,
                    "prob": prob,
                })
    return render_template("analyze.html", analysis_results=analysis_results, user_text=user_text)

@app.route("/docucheck", methods=["GET", "POST"])
def docucheck():
    results = []
    error = None
    if request.method == "POST":
        file = request.files.get("file")
        if not file:
            error = "No file uploaded."
        else:
            try:
                filepath = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(filepath)

                extracted_text = pytesseract.image_to_string(Image.open(filepath))

                txt_path = os.path.splitext(filepath)[0] + ".txt"
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(extracted_text)

                sentences = sent_tokenize(extracted_text)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if sentence:
                        label, prob = predict_bias(sentence)
                        results.append({
                            "line": sentence,
                            "label": label,
                            "prob": prob
                        })

            except Exception as e:
                error = f"Error processing the file: {str(e)}"

    return render_template("docucheck.html", results=results, error=error)

@app.route("/api/articles")
def api_articles():
    return jsonify(load_biased_articles())


if __name__ == "__main__":
    app.run(debug=True)

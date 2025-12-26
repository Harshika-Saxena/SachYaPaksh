# SachYaPaksh
### *Reclaiming the truth, one click at a time.*

SachYaPaksh is an AI-powered media analysis tool designed to detect and flag bias in real-time. By combining Natural Language Processing (NLP) and a fine-tuned **DistilBERT** model, the platform helps users distinguish between objective reporting and biased influence.

---

## 🛠️ Technical Implementation
* **Backend:** Developed in **Python** to handle real-time data ingestion.
* **Data Pipeline:** Custom logic scrapes live news via APIs and structures raw data into **JSON format** for seamless model ingestion.
* **AI Model:** Utilizes a **Transformer-based architecture** with a fine-tuned **DistilBERT** model for sequence classification.
* **Bias Detection:** Calculates a "Bias Confidence Score" based on narrative portrayal and sentiment analysis rather than simple keyword matching.



## ⚡ Core Modules
* **FairFeed:** A real-time news scanner that identifies and flags skewed headlines and narrative slants in live articles.
* **DocuCheck:** An inference engine that acts as a "mirror" for content creators, performing an instant pass on input text to identify harmful generalizations.
* **TrueView:** A visual analysis tool using **OCR** to extract text from posters and ads to find hidden representation gaps that the human eye often overlooks.

## 🔮 Roadmap & Vision
We are currently scaling SachYaPaksh toward real-time video analysis and browser extensions. Our goal is to make unbiased media analysis accessible to everyone, ensuring that the stories we consume are balanced and transparent.

**With SachYaPaksh, we are reclaiming the truth, one click at a time.**

---
*Developed by Harshika Saxena and Yajur Sharma.*

from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained("d4data/bias-detection-model")
model = TFAutoModelForSequenceClassification.from_pretrained('d4data/bias-detection-model')
classifier = pipeline('text-classification', model=model, tokenizer=tokenizer)
result = classifier("black boy shot dead because he was black")[0]
label = "Biased" if result["label"] == "Biased" else 'Neutral'
if label=='Biased':
    print('hello')
print(result)

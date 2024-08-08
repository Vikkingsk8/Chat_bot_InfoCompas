from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Загрузка модели и токенизатора
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased')

def classify_question(question):
    inputs = tokenizer(question, return_tensors='pt')
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    return predictions.item()

# Пример использования
question = "Как мне оформить заявку на получение кредита?"
category = classify_question(question)
print(f"Вопрос относится к категории: {category}")
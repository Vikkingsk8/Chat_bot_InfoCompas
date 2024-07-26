from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import logging

app = Flask(__name__)

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Конфигурация
DATA_PATH = os.getenv('DATA_PATH', r'C:\Users\vikto\.vscode\project_1\main_project\Flask_project\data\ответы.xlsx')
THRESHOLD = float(os.getenv('THRESHOLD', 0.3))

# Загрузка данных и кэширование
def load_data():
    data = pd.read_excel(DATA_PATH)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(data['Текст вопроса'])
    return data, vectorizer, tfidf_matrix

data, vectorizer, tfidf_matrix = load_data()

def get_answer(user_question):
    # Обработка специальных вопросов
    greetings = ['привет', 'здравствуй', 'здравствуйте']
    how_are_you = ['как дела?', 'как ты?', 'как поживаешь?']
    capabilities = ['что ты умеешь?', 'что можешь?', 'твои возможности']
    name = ['как тебя зовут?', 'твое имя?']

    if user_question.lower() in greetings:
        return "Привет! Я здесь, чтобы помочь. Задайте ваш вопрос."
    elif user_question.lower() in how_are_you:
        return "У меня все хорошо, спасибо! А у вас?"
    elif user_question.lower() in capabilities:
        return "Я могу ответить на ваши вопросы и помочь найти информацию."
    elif user_question.lower() in name:
        return "Меня зовут ИнфоКомпас. Я здесь, чтобы помочь вам."

    user_vector = vectorizer.transform([user_question])
    similarity = cosine_similarity(user_vector, tfidf_matrix).flatten()
    most_similar_index = similarity.argmax()
    if similarity[most_similar_index] > THRESHOLD:
        answer = data['Текст ответа'][most_similar_index]
        correction = data['Корректировки'][most_similar_index]
        if pd.notna(correction) and correction.strip():
            return correction
        else:
            return answer
    else:
        return """Пока я не знаю ответ на этот вопрос.
        Пожалуйста обратитесь к инструкции: https://www.cbr.ru/Content/Document/File/85699/instruction.pdf ,
        также вы можете посмотреть видеоинструкции по ссылке: https://www.cbr.ru/lk_uio/video_instructions/,
        и не забывайте есть контакты ЕСПП внизу страницы"""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_question = request.json.get('question')
    if user_question:
        answer = get_answer(user_question)
        return jsonify({'answer': answer})
    else:
        return jsonify({'error': 'No question provided'}), 400

if __name__ == '__main__':
    app.run(debug=True)
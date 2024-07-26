from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import logging
import random
import functools
import pickle

app = Flask(__name__)

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Конфигурация
DATA_PATH = os.getenv('DATA_PATH', r'C:\Users\vikto\.vscode\project_1\main_project\Flask_project\data\ответы.xlsx')
THRESHOLD = float(os.getenv('THRESHOLD', 0.3))
CACHE_PATH = os.getenv('CACHE_PATH', r'C:\Users\vikto\.vscode\project_1\main_project\Flask_project\data\cache.pkl')

# Кэширование данных и векторизатора
def load_data():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, 'rb') as f:
            data, vectorizer, tfidf_matrix = pickle.load(f)
    else:
        data = pd.read_excel(DATA_PATH)
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(data['Текст вопроса'])
        with open(CACHE_PATH, 'wb') as f:
            pickle.dump((data, vectorizer, tfidf_matrix), f)
    return data, vectorizer, tfidf_matrix

data, vectorizer, tfidf_matrix = load_data()

def get_answer(user_question, previous_answers=None):
    # Обработка специальных вопросов
    greetings = ['привет', 'здравствуй', 'здравствуйте']
    how_are_you = ['как дела?', 'как ты?', 'как поживаешь?']
    capabilities = ['что ты умеешь?', 'что можешь?', 'твои возможности']
    name = ['как тебя зовут?', 'твое имя?']

    if any(greeting in user_question.lower() for greeting in greetings):
        return random.choice(["Привет! Я здесь, чтобы помочь. Задайте ваш вопрос.", "Здравствуйте! Чем могу помочь?"]), False
    elif any(question in user_question.lower() for question in how_are_you):
        return random.choice(["У меня все хорошо, спасибо! А у вас?", "Отлично, спасибо за интерес! Как у вас?"]), False
    elif any(capability in user_question.lower() for capability in capabilities):
        return random.choice(["Я могу ответить на ваши вопросы и помочь найти информацию.", "Я здесь, чтобы отвечать на ваши вопросы и предоставлять информацию."]), False
    elif any(n in user_question.lower() for n in name):
        return random.choice(["Меня зовут ИнфоКомпас. Я здесь, чтобы помочь вам.", "Вы можете называть меня ИнфоКомпас. Чем могу помочь?"]), False

    user_vector = vectorizer.transform([user_question])
    similarity = cosine_similarity(user_vector, tfidf_matrix).flatten()
    most_similar_index = similarity.argmax()
    if similarity[most_similar_index] > THRESHOLD:
        answer = data['Текст ответа'][most_similar_index]
        correction = data['Корректировки'][most_similar_index]
        if pd.notna(correction) and correction.strip():
            answer = correction
        if previous_answers and answer in previous_answers:
            similarity[most_similar_index] = 0
            most_similar_index = similarity.argmax()
            answer = data['Текст ответа'][most_similar_index]
            correction = data['Корректировки'][most_similar_index]
            if pd.notna(correction) and correction.strip():
                answer = correction
        return answer, True
    else:
        return random.choice([
            "Пока я не знаю ответ на этот вопрос. Пожалуйста, обратитесь к инструкции: <a href='https://www.cbr.ru/Content/Document/File/85699/instruction.pdf' target='_blank'>инструкция</a>, также вы можете посмотреть видеоинструкции по ссылке: <a href='https://www.cbr.ru/lk_uio/video_instructions/' target='_blank'>видеоинструкции</a>, и не забывайте есть контакты ЕСПП внизу страницы",
            "Извините, но я не могу ответить на этот вопрос. Пожалуйста, проверьте инструкции (<a href='https://www.cbr.ru/Content/Document/File/85699/instruction.pdf' target='_blank'>инструкция</a>, <a href='https://www.cbr.ru/lk_uio/video_instructions/' target='_blank'>видеоинструкции</a>) или свяжитесь с ЕСПП для получения помощи.",
            "Этот вопрос выходит за рамки моих знаний. Пожалуйста, обратитесь к дополнительным ресурсам (<a href='https://www.cbr.ru/Content/Document/File/85699/instruction.pdf' target='_blank'>инструкция</a>, <a href='https://www.cbr.ru/lk_uio/video_instructions/' target='_blank'>видеоинструкции</a>) или свяжитесь с поддержкой."
        ]), False

@app.route('/')
def index():
    return render_template('index.html', initial_message="Привет! Я здесь, чтобы помочь. Задайте ваш вопрос.")

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_question = request.json.get('question')
        previous_answers = request.json.get('previous_answers', [])
        if user_question:
            answer, feedback = get_answer(user_question, previous_answers)
            return jsonify({'answer': answer, 'feedback': feedback})
        else:
            return jsonify({'error': 'No question provided'}), 400
    except Exception as e:
        logging.error(f"Error in /chat: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    try:
        feedback_data = request.json
        user_question = feedback_data.get('question')
        feedback = feedback_data.get('feedback')
        previous_answers = feedback_data.get('previous_answers', [])

        if feedback == 'more':
            if len(previous_answers) >= 3:
                return jsonify({'answer': "Извините, но я исчерпал все попытки найти другой ответ. Пожалуйста, попробуйте сформулировать вопрос по-другому.", 'feedback': False})
            new_answer, feedback = get_answer(user_question, previous_answers)
            return jsonify({'answer': new_answer, 'feedback': feedback})
        else:
            return jsonify({'error': 'Invalid feedback'}), 400
    except Exception as e:
        logging.error(f"Error in /feedback: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500

if __name__ == '__main__':
    app.run(debug=True)
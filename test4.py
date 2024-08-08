from flask import Flask, request, jsonify, render_template, send_file, session
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import logging
import fitz  # PyMuPDF
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import pickle
import concurrent.futures
import time
import sqlite3
from apscheduler.schedulers.background import BackgroundScheduler
import atexit

nltk.download('stopwords')
nltk.download('punkt')

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Необходимо для использования сессий

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Конфигурация
class Config:
    PDF_PATH = r'C:\Users\vikto\.vscode\project_1\main_project\InfoCompas\data\instruction.pdf'
    EXCEL_PATH = r'C:\Users\vikto\.vscode\project_1\main_project\InfoCompas\data\ответы.xlsx'
    LINKS_PATH = r'C:\Users\vikto\.vscode\project_1\main_project\InfoCompas\data\links.xlsx'
    THRESHOLD_PDF = 0.1  # Порог сходства для PDF
    THRESHOLD_EXCEL = 0.3  # Порог сходства для Excel
    EXCLUDE_PAGES = 9  # Количество страниц оглавления, которые нужно исключить
    CACHE_PATH = r'C:\Users\vikto\.vscode\project_1\main_project\InfoCompas\data\cache.pkl'
    DB_PATH = r'C:\Users\vikto\.vscode\project_1\main_project\InfoCompas\data\feedback.db'  # Путь к базе данных

# Загрузка данных из Excel
def load_excel_data(excel_path):
    try:
        df = pd.read_excel(excel_path)
        return df
    except Exception as e:
        logging.error(f"Error loading Excel data: {e}")
        return None

# Загрузка данных из файла links.xlsx
def load_links_data(links_path):
    try:
        df = pd.read_excel(links_path)
        expanded_rows = []
        for _, row in df.iterrows():
            questions = str(row['Вопрос']).split('?')
            for question in questions:
                if question.strip():
                    new_row = row.copy()
                    new_row['Вопрос'] = question.strip()
                    expanded_rows.append(new_row)
        expanded_df = pd.DataFrame(expanded_rows)
        return expanded_df
    except Exception as e:
        logging.error(f"Error loading links data: {e}")
        return None

def extract_text_and_pages_from_pdf(pdf_path):
    try:
        pdf_document = fitz.open(pdf_path)
        text_and_pages = []
        for page_num in range(len(pdf_document)):
            if Config.EXCLUDE_PAGES <= page_num < 450 or page_num > 638:
                page = pdf_document.load_page(page_num)
                text = page.get_text()
                text_and_pages.append((text, page_num + 1))
        return text_and_pages
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {e}")
        return []

def split_text_into_paragraphs(text_and_pages):
    paragraphs = [text for text, page in text_and_pages]
    pages = [page for text, page in text_and_pages]
    return paragraphs, pages

def create_tfidf_matrix(paragraphs):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(paragraphs)
    return vectorizer, tfidf_matrix

def find_best_answer(user_question, vectorizer, tfidf_matrix, paragraphs, pages, threshold=Config.THRESHOLD_PDF):
    try:
        user_vector = vectorizer.transform([user_question])
        similarity = cosine_similarity(user_vector, tfidf_matrix).flatten()
        most_similar_index = similarity.argmax()
        if similarity[most_similar_index] > threshold:
            answer = paragraphs[most_similar_index]
            page = pages[most_similar_index]
            return answer, True, page
        else:
            return "UNKNOWN_QUESTION_RESPONSE", False, None
    except Exception as e:
        logging.error(f"Error finding best answer: {e}")
        return "Извините, произошла ошибка при поиске ответа.", False, None

def preprocess_answer(answer):
    answer = re.sub(r'Рисунок \d+ -.*\n?', '', answer)
    answer = re.sub(r'\s*РУКОВОДСТВО ПОЛЬЗОВАТЕЛЯ \(ЕПВВ\)\s*\d+\s*', '', answer)
    return answer

def format_answer(answer):
    answer = preprocess_answer(answer)
    answer = re.sub(r'', '', answer)
    paragraphs = re.split(r'\n\s*\n', answer)
    formatted_answer = "<div>"
    for paragraph in paragraphs:
        if paragraph.strip():
            formatted_answer += f"<p>{paragraph.strip()}</p>"
    formatted_answer += "</div>"
    return formatted_answer

# Инициализация стеммера для русского языка
stemmer = SnowballStemmer('russian')

def preprocess_text(text):
    text = text.lower()
    # Удаление или замена символов, таких как дефисы
    text = re.sub(r'[^a-zA-Zа-яА-Я0-9\s]', ' ', text)
    tokens = word_tokenize(text, language='russian')
    stop_words = set(stopwords.words('russian'))
    filtered_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words and len(word) > 1]
    processed_text = ' '.join(filtered_tokens)
    return processed_text if processed_text else text

def get_file_modification_time(file_path):
    return os.path.getmtime(file_path)

def load_cache():
    if os.path.exists(Config.CACHE_PATH):
        try:
            with open(Config.CACHE_PATH, 'rb') as f:
                cache = pickle.load(f)
                if 'mod_times' in cache:
                    current_mod_times = {
                        'pdf': get_file_modification_time(Config.PDF_PATH),
                        'excel': get_file_modification_time(Config.EXCEL_PATH),
                        'links': get_file_modification_time(Config.LINKS_PATH)
                    }
                    if cache['mod_times'] == current_mod_times:
                        return cache
                else:
                    return cache
        except Exception as e:
            logging.error(f"Error loading cache: {e}")
    return None

def save_cache(cache):
    try:
        cache['mod_times'] = {
            'pdf': get_file_modification_time(Config.PDF_PATH),
            'excel': get_file_modification_time(Config.EXCEL_PATH),
            'links': get_file_modification_time(Config.LINKS_PATH)
        }
        with open(Config.CACHE_PATH, 'wb') as f:
            pickle.dump(cache, f)
    except Exception as e:
        logging.error(f"Error saving cache: {e}")

# Параллельная загрузка данных
def load_data_parallel():
    with concurrent.futures.ThreadPoolExecutor() as executor:
        pdf_future = executor.submit(extract_text_and_pages_from_pdf, Config.PDF_PATH)
        excel_future = executor.submit(load_excel_data, Config.EXCEL_PATH)
        links_future = executor.submit(load_links_data, Config.LINKS_PATH)
        
        pdf_text_and_pages = pdf_future.result()
        excel_data = excel_future.result()
        links_data = links_future.result()
        
        if not all([pdf_text_and_pages, excel_data is not None and not excel_data.empty, links_data is not None and not links_data.empty]):
            return None
        
        pdf_paragraphs, pdf_pages = split_text_into_paragraphs(pdf_text_and_pages)
        pdf_vectorizer, pdf_tfidf_matrix = create_tfidf_matrix(pdf_paragraphs[Config.EXCLUDE_PAGES:])
        
        return {
            'pdf_paragraphs': pdf_paragraphs,
            'pdf_pages': pdf_pages,
            'pdf_vectorizer': pdf_vectorizer,
            'pdf_tfidf_matrix': pdf_tfidf_matrix,
            'excel_data': excel_data,
            'links_data': links_data
        }

# Загрузка кэша или создание нового
cache = load_cache()
if cache is None or 'excel_data' not in cache or 'links_data' not in cache:
    cache = load_data_parallel()
    if cache:
        save_cache(cache)
    else:
        logging.error("Failed to load data. Exiting.")
        exit(1)

pdf_paragraphs = cache['pdf_paragraphs']
pdf_pages = cache['pdf_pages']
pdf_vectorizer = cache['pdf_vectorizer']
pdf_tfidf_matrix = cache['pdf_tfidf_matrix']
excel_data = cache['excel_data']
links_data = cache['links_data']

# Повторяющаяся строка
UNKNOWN_QUESTION_RESPONSE = "Попробуйте перефразировать свой вопрос или обратитесь к <a href='https://www.cbr.ru/Content/Document/File/85699/instruction.pdf' target='_blank'>инструкции</a>, также вы можете посмотреть <a href='https://www.cbr.ru/lk_uio/video_instructions/' target='_blank'>видеоинструкции</a>"

# Разговорные фразы
greetings = ["привет", "здравствуй", "добрый день", "доброе утро", "добрый вечер"]
how_are_you = ["как дела", "как ты", "как поживаешь", "как жизнь"]
what_can_you_do = ["что ты умеешь", "что ты можешь", "какие у тебя возможности"]
what_is_your_name = ["как тебя зовут", "ты кто", "кто ты"]

def check_like_count():
    try:
        conn = sqlite3.connect(Config.DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM feedback WHERE feedback_type = 'like'")
        like_count = cursor.fetchone()[0]
        conn.close()
        return like_count
    except Exception as e:
        logging.error(f"Error checking like count: {e}")
        return 0

# Функция для автоматического переобучения модели
def auto_retrain_model():
    like_count = check_like_count()
    if like_count >= 10:  # Порог количества лайков для переобучения
        retrain_model()

# Функция для переобучения модели
def retrain_model():
    try:
        # Загрузка данных обратной связи из базы данных
        conn = sqlite3.connect(Config.DB_PATH)
        feedback_df = pd.read_sql_query("SELECT * FROM feedback", conn)
        conn.close()

        if feedback_df.empty:
            logging.info("No feedback data available for retraining.")
            return

        # Обработка данных обратной связи
        feedback_df['question'] = feedback_df['question'].apply(preprocess_text)
        feedback_df['answer'] = feedback_df['answer'].apply(preprocess_text)

        # Создание TF-IDF матрицы на основе данных обратной связи
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(feedback_df['question'])

        # Сохранение новой модели и векторизатора
        with open(Config.CACHE_PATH, 'wb') as f:
            pickle.dump({
                'vectorizer': vectorizer,
                'tfidf_matrix': tfidf_matrix,
                'feedback_data': feedback_df
            }, f)

        logging.info("Model retrained successfully.")
    except Exception as e:
        logging.error(f"Error retraining model: {e}")

# Настройка планировщика задач
scheduler = BackgroundScheduler()
scheduler.add_job(func=auto_retrain_model, trigger="interval", minutes=30)  # Проверка каждые 30 минут
scheduler.start()

# Зарегистрируйте функцию для остановки планировщика при выходе
atexit.register(lambda: scheduler.shutdown())

@app.route('/')
def index():
    return render_template('index.html', initial_message="Привет! Я ИнфоКомпас, ваш виртуальный помощник. Чем могу помочь?")

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_question = request.json.get('question')
        previous_answers = request.json.get('previous_answers', [])
        
        if 'last_link_question' in session and session['last_link_question'] == user_question:
            # Обработка вопроса, связанного с последней нажатой ссылкой
            answer = f"Вы нажали на ссылку с вопросом: {user_question}"
            return jsonify({'answer': answer, 'feedback': False})
        
        if len(user_question.strip()) < 2:
            return jsonify({'answer': "Пожалуйста, задайте более конкретный вопрос.", 'feedback': False})
        
        user_question_lower = user_question.lower()
        if any(greeting in user_question_lower for greeting in greetings):
            return jsonify({'answer': "Привет! Я ИнфоКомпас, ваш виртуальный помощник. Чем могу помочь?", 'feedback': False})
        elif any(question in user_question_lower for question in how_are_you):
            return jsonify({'answer': "У меня все хорошо, спасибо! А у вас?", 'feedback': False})
        elif any(question in user_question_lower for question in what_can_you_do):
            return jsonify({'answer': "Я могу помочь вам найти информацию в инструкции и ответить на ваши вопросы.", 'feedback': False})
        elif any(question in user_question_lower for question in what_is_your_name):
            return jsonify({'answer': "Меня зовут ИнфоКомпас. Я ваш виртуальный помощник.", 'feedback': False})
        
        user_question = preprocess_text(user_question)
        
        # Векторизация вопросов из Excel
        excel_questions = excel_data['Текст вопроса'].apply(preprocess_text).tolist()
        excel_vectorizer = TfidfVectorizer()
        excel_tfidf_matrix = excel_vectorizer.fit_transform(excel_questions)
        
        user_vector = excel_vectorizer.transform([user_question])
        similarity = cosine_similarity(user_vector, excel_tfidf_matrix).flatten()
        most_similar_index = similarity.argmax()
        
        if similarity[most_similar_index] > Config.THRESHOLD_EXCEL:
            answer = excel_data.iloc[most_similar_index]['Текст ответа']
            links = find_relevant_links(user_question)
            formatted_links = [{'question': row['Вопрос'], 'url': row['Ссылка']} for _, row in links.iterrows()]
            return jsonify({'answer': answer, 'feedback': True, 'links': formatted_links})
        
        # Поиск ответа в PDF
        answer, feedback, pdf_page = find_best_answer(user_question, pdf_vectorizer, pdf_tfidf_matrix, pdf_paragraphs[Config.EXCLUDE_PAGES:], pdf_pages[Config.EXCLUDE_PAGES:], threshold=Config.THRESHOLD_PDF)
        if feedback:
            formatted_answer = format_answer(answer)
            links = find_relevant_links(user_question)
            formatted_links = [{'question': row['Вопрос'], 'url': row['Ссылка']} for _, row in links.iterrows()]
            response = {'answer': formatted_answer, 'feedback': feedback, 'links': formatted_links}
            if pdf_page is not None:
                response['pdf_page'] = pdf_page
            return jsonify(response)
        else:
            return jsonify({'answer': UNKNOWN_QUESTION_RESPONSE, 'feedback': False})
    except Exception as e:
        logging.error(f"Error in /chat: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500
    
@app.route('/download_pdf')
def download_pdf():
    return send_file(Config.PDF_PATH, as_attachment=False)

@app.route('/like', methods=['POST'])
def like():
    try:
        feedback = request.json.get('feedback')
        # Сохранение данных обратной связи
        save_feedback(feedback, 'like')
        return jsonify({'status': 'success'})
    except Exception as e:
        logging.error(f"Error in /like: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500

@app.route('/dislike', methods=['POST'])
def dislike():
    try:
        feedback = request.json.get('feedback')
        # Сохранение данных обратной связи
        save_feedback(feedback, 'dislike')
        return jsonify({'status': 'success'})
    except Exception as e:
        logging.error(f"Error in /dislike: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500

def save_feedback(feedback, feedback_type):
    try:
        if not os.path.exists(os.path.dirname(Config.DB_PATH)):
            os.makedirs(os.path.dirname(Config.DB_PATH))
        conn = sqlite3.connect(Config.DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS feedback (id INTEGER PRIMARY KEY, question TEXT, answer TEXT, feedback_type TEXT)''')
        cursor.execute('''INSERT INTO feedback (question, answer, feedback_type) VALUES (?, ?, ?)''', (feedback['question'], feedback['answer'], feedback_type))
        conn.commit()
        conn.close()
    except Exception as e:
        logging.error(f"Error saving feedback: {e}")

def find_relevant_links(user_question, threshold=0.3):
    links_questions = links_data['Вопрос'].apply(preprocess_text).tolist()
    links_vectorizer = TfidfVectorizer()
    links_tfidf_matrix = links_vectorizer.fit_transform(links_questions)
    
    user_vector = links_vectorizer.transform([user_question])
    similarity = cosine_similarity(user_vector, links_tfidf_matrix).flatten()
    
    # Фильтрация по порогу сходства
    relevant_indices = [i for i, sim in enumerate(similarity) if sim > threshold]
    
    return links_data.iloc[relevant_indices]

if __name__ == '__main__':
    app.run(debug=True)
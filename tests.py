import unittest
from app import app, load_excel_data, extract_text_and_pages_from_pdf, preprocess_text, excel_data, pdf_vectorizer, pdf_tfidf_matrix, pdf_paragraphs, pdf_pages, EXCLUDE_PAGES, find_best_answer, format_answer, UNKNOWN_QUESTION_RESPONSE, THRESHOLD_PDF
import os

class TestApp(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_index_route(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)

    def test_load_excel_data(self):
        data_path = os.path.join(os.path.dirname(__file__), 'data', 'ответы.xlsx')
        data = load_excel_data(data_path)
        self.assertIsNotNone(data)
        self.assertGreater(len(data), 0)

    def test_extract_text_and_pages_from_pdf(self):
        pdf_path = os.path.join(os.path.dirname(__file__), 'data', 'instruction.pdf')
        text_and_pages = extract_text_and_pages_from_pdf(pdf_path)
        self.assertIsNotNone(text_and_pages)
        self.assertGreater(len(text_and_pages), 0)

    def test_preprocess_text(self):
        text = "Привет, как дела?"
        processed_text = preprocess_text(text)
        self.assertEqual(processed_text, "привет дела")

    def test_preprocess_text_empty(self):
        text = ""
        processed_text = preprocess_text(text)
        self.assertEqual(processed_text, "")

    def test_preprocess_text_with_extra_chars(self):
        text = "Привет, @#$% как дела?"
        processed_text = preprocess_text(text)
        self.assertEqual(processed_text, "привет дела")

    def test_find_answer_in_excel(self):
        question = "УКЭП"
        answer = excel_data[excel_data['Текст вопроса'].str.contains(question, case=False, na=False)]
        self.assertFalse(answer.empty)

    def test_find_answer_in_pdf(self):
        question = "регистрация"
        answer, feedback, pdf_page = find_best_answer(question, pdf_vectorizer, pdf_tfidf_matrix, pdf_paragraphs[EXCLUDE_PAGES:], pdf_pages[EXCLUDE_PAGES:], threshold=THRESHOLD_PDF)
        self.assertTrue(feedback)

    def test_chat_endpoint(self):
        response = self.app.post('/chat', json={'question': 'привет'})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('answer', data)
        self.assertIn('feedback', data)

    def test_chat_endpoint_invalid_question(self):
        response = self.app.post('/chat', json={'question': ''})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['answer'], "Пожалуйста, задайте более конкретный вопрос.")

    def test_chat_endpoint_unknown_question(self):
        response = self.app.post('/chat', json={'question': 'что такое квантовая физика?'})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['answer'], UNKNOWN_QUESTION_RESPONSE)

    def test_format_answer(self):
        answer = "Привет, как дела?\n\nЯ ИнфоКомпас, ваш виртуальный помощник."
        formatted_answer = format_answer(answer)
        expected_answer = "<div><p>Привет, как дела?</p><p>Я ИнфоКомпас, ваш виртуальный помощник.</p></div>"
        self.assertEqual(formatted_answer, expected_answer)

    def test_load_excel_data_nonexistent_file(self):
        with self.assertRaises(FileNotFoundError):
            load_excel_data('data/nonexistent.xlsx')

    def test_load_excel_data_invalid_file(self):
        with self.assertRaises(Exception):
            load_excel_data('data/invalid_file.txt')

    def test_download_pdf_endpoint(self):
        response = self.app.get('/download_pdf')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, 'application/pdf')

if __name__ == '__main__':
    unittest.main()
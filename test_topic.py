import unittest
from topic import Topic

class TestTopic(unittest.TestCase):

    def setUp(self):
        self.lang_code = "en"
        self.text = "Football is a family of team sports that involve, to varying degrees, kicking a ball to score a goal."
    
    def tearDown(self):
        pass

    def test_predict_topic_for_invalid_language_model(self):
        text = """ La Ligue des champions de l'UEFA (UEFA Champions League), parfois abrégée en C1 et anciennement 
                  dénommée Coupe des clubs champions européens (de sa création en 1955 jusqu'en 1992), est une compétition 
                  annuelle de football organisée par l'Union des associations européennes de football (UEFA) et regroupant 
                  les meilleurs clubs du continent européen1. """
        
        t = Topic(lang_code="fr")
        pred = t.predict_topic(text)
        self.assertTrue(isinstance(pred, str))
        self.assertEqual(pred, 'language not supported')

    def test_predict_topic_for_invalid_model_version(self):
        t = Topic(lang_code=self.lang_code, version="1.3")
        pred = t.predict_topic(self.text)
        self.assertTrue(isinstance(pred, str))
        self.assertEqual(pred, 'model not found')

    def test_predict_topic_for_invalid_method_name(self):
        t = Topic(lang_code=self.lang_code, method="METHOD_NAME")
        pred = t.predict_topic(self.text)
        self.assertTrue(isinstance(pred, str))
        self.assertEqual(pred, 'method not exist')

    def test_predict_topic_on_empty_text(self):
        text = ""
        t = Topic(lang_code=self.lang_code)
        pred = t.predict_topic(text)
        self.assertTrue(isinstance(pred, str))
        self.assertEqual(pred, 'required textual content')

    def test_predict_topic_when_doc_length_less_than_min_words(self):
        min_words = 20
        t = Topic(lang_code=self.lang_code, min_words=min_words)
        pred = t.predict_topic(self.text)
        self.assertTrue(isinstance(pred, str))
        self.assertEqual("required at least {} words for prediction".format(min_words), pred)

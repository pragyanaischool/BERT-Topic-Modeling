import unittest
from preprocessing import Preprocessing
from nltk import word_tokenize

class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        self.lang_code = "en"
        self.text = "Apple is looking at buying U.K. startup for $1 billion"        
        self.token_list = ["result", "football_player", "goal", "goal_goal", 
                                 "cup_league", "game", "cup", "result_result"]

    def tearDown(self):
        pass

    def test_remove_stopwords(self):
        p = Preprocessing(lang_code=self.lang_code, min_doc_len=10)
        actual = p.remove_stopwords(self.text)
        self.assertTrue(isinstance(actual, str))
        self.assertTrue('is' or 'at' or 'for' not in actual)

    def test_filter_entities(self):
        p = Preprocessing(lang_code=self.lang_code, filter_entity_tags=['ORG'])
        actual = p.filter_entities(self.text)
        self.assertTrue('Apple' not in actual)
    
    def test_keep_valid_doc(self):
        p = Preprocessing(lang_code=self.lang_code, min_doc_len=50)
        actual = p.keep_valid_doc(self.text)
        self.assertTrue(isinstance(actual, str))
        self.assertEqual(actual, '')

    def test_extract_unigrams(self):
        expected = ["result", "goal", "game", "cup"]
        p = Preprocessing(lang_code=self.lang_code)
        unigrams = p.extract_unigrams(self.token_list)
        self.assertTrue(isinstance(unigrams, list))
        self.assertEqual(unigrams, expected)

    def test_extract_bigrams(self):
        expected = ["football_player", "goal_goal", "cup_league", "result_result"]
        p = Preprocessing(lang_code='en')
        bigrams = p.extract_bigrams(self.token_list)
        self.assertTrue(isinstance(bigrams, list))
        self.assertEqual(bigrams, expected)

    def test_keep_valid_tokens(self):
        expected = ["result", "goal", "game", "cup", "football_player", "cup_league"]
        p = Preprocessing(lang_code=self.lang_code)
        valid_tokens = p.keep_valid_tokens(self.token_list)
        self.assertTrue('goal_goal' and 'result_result' not in valid_tokens)
        self.assertTrue(isinstance(valid_tokens, list))
        self.assertEqual(valid_tokens, expected)
    
    def test_clean_tokens(self):
        expected = ["apple", "looking", "buying", "startup", "for", "billion"]
        p = Preprocessing(lang_code=self.lang_code)
        actual = p.clean_tokens(self.text)
        self.assertTrue(isinstance(actual, list))
        self.assertEqual(actual, expected)

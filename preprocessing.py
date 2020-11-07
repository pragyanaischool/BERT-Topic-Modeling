from gensim.utils import simple_preprocess
from gensim.models.phrases import Phraser
from gensim.models import Phrases
from collections import Counter
from nltk.corpus import stopwords 
from nltk import word_tokenize
import contextlib
import fasttext
import spacy
import re
import os

from shared.utils import read_txt_as_list
from functools import lru_cache
from tqdm import tqdm

fasttext.FastText.eprint = print

PRETRAINED_MODEL_PATH = 'fasttext/model/lid.176.ftz'
stopwords_path = os.path.dirname(os.path.abspath("__file__")) + "/stopwords"

class Preprocessing(object):
    """ Class for pre-processing textual content

    :param lang_code: language text
    :param min_lang_conf: min language confidence score
    :param valid_pos_tags: list for keeping valid part-of-speech tags
    :param filter_entity_tags: list for filtering entities from textual content
    :param min_doc_len: min number of words required in document for preprocessing
    :param min_token_len: min number of characters in a token
    :param deacc: boolean flag to clean punctuation marks
    """
    def __init__(self, lang_code, min_lang_conf=0.30, valid_pos_tags=["NOUN", "VERB"], 
                 filter_entity_tags=["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", 
                                     "WORK_OF_ART", "LANGUAGE","DATE", "TIME", "ORDINAL", "CARDINAL"],
                min_doc_len=10, min_token_len=3, deacc=True):

        self.lang_code = lang_code
        self.min_lang_conf = min_lang_conf
        self.valid_pos_tags = valid_pos_tags
        self.filter_entity_tags = filter_entity_tags
        self.deacc = deacc
        self.min_token_len = min_token_len      # min token characters
        self.min_doc_len = min_doc_len          # min words per document
        
        self.stopwords = []
        self.spacy_nlp = None
        self.valid_langs = ["en"]

        self.fasttext_model = self.load_fasttext_model(PRETRAINED_MODEL_PATH)
        
        if lang_code in self.valid_langs:
            if lang_code == "en":
                self.spacy_nlp = self.load_spacy_model(lang_code)
                self.stopwords = stopwords.words('english')
                nouns = read_txt_as_list(stopwords_path + "/" + lang_code + "/nouns.txt")
                verbs = read_txt_as_list(stopwords_path + "/" + lang_code + "/verbs.txt")
                adjectives = read_txt_as_list(stopwords_path + "/" + lang_code + "/adjectives.txt")
                adverbs = read_txt_as_list(stopwords_path + "/" + lang_code + "/adverbs.txt")
                self.stopwords = self.stopwords + nouns + verbs + adjectives + adverbs
                self.stopwords = Counter(self.stopwords)
  
    @lru_cache(maxsize=256)
    def load_spacy_model(self, lang_code):
        """ Load spacy model 
        
        :param lang_code: language model
        :return: Spacy model
        """
        model = None
        if lang_code == "en":
            model = spacy.load(lang_code + "_core_web_sm")
        return model

    @lru_cache(maxsize=256)
    def load_fasttext_model(self, filepath):
        """ Load FastText model 
        
        :param filepath: 
        :return: FastText model
        """
        model = None
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            model = fasttext.load_model(filepath)
        return model

    def lemmatize(self, text):
        """ Lemmatize words in text and keep valid pos tags 
        
        :param text: text 
        :return: valid part-of-speech and lemmatized words in text
        """
        text = self.spacy_nlp(text)
        lemma_tokens = [token.lemma_ for token in text if token.pos_ in self.valid_pos_tags]
        text = " ".join(lemma_tokens)
        return text

    def remove_stopwords(self, text):
        """ Remove stopwords from text 
        
        :param text: text
        :return: filtered stopwords from text
        """
        tokens = word_tokenize(text)
        filtered_tokens = [token for token in tokens if token.lower() not in self.stopwords]
        text = " ".join(filtered_tokens)
        return text

    def filter_entities(self, text):
        """ Filter entities from index position using Spacy 
        
        :param text: text
        :return: filtered entities from text
        """
        text = self.spacy_nlp(text)
        filtered_tokens = [token.text for token in text if token.ent_type_ not in self.filter_entity_tags]
        text = " ".join(filtered_tokens)
        return text

    def clean_tokens(self, text):
        """ Remove digits, special characters and tokens with less than 3 characters.
        
        :param text: text
        :return: cleaned list of tokens
        """
        tokens = simple_preprocess(text, deacc=self.deacc, min_len=self.min_token_len)
        return tokens

    def keep_valid_doc(self, text):
        """ Keep document with minimum number of words 
        
        :param text: text
        :return: text
        """
        text = self.spacy_nlp(text)
        if len(text) <= self.min_doc_len:
            return ""
        return str(text)
            
    def make_bigrams(self, clean_token_lists, min_count=20, threshold=20, delimiter=b'_'):
        """ Generate bigrams from a list of normalized documents. 
        
        :param clean_token_lists: list of document tokens
        :return: list of document tokens (unigrams + bigrams) 
        """
        bigram = Phrases(clean_token_lists, min_count=min_count, threshold=threshold, delimiter=delimiter)
        bigram_model = Phraser(bigram)
        norm_bigrams = [bigram_model[doc] for doc in clean_token_lists]
        return norm_bigrams

    def extract_unigrams(self, token_list):
        """ Extract unigrams from list of tokens 
        
        :param token_list: list of tokens (unigrams + bigrams)
        :return: list of unigrams
        """
        unigrams = []
        for token in token_list:
            unigram = re.match(r"[a-zA-Z']+$", token)
            if unigram:
                unigram = unigram.group(0)
                unigrams.append(unigram)
        return unigrams

    def extract_bigrams(self, token_list):
        """ Extract bigrams from list of tokens 
        
        :param token_list: list of tokens (unigrams + bigrams)
        :return: list of bigrams
        """
        bigrams = []
        for token in token_list:
            bigram = re.match(r"[a-zA-Z]+_[a-zA-Z]+$", token)
            if bigram:
                bigram = bigram.group(0)
                bigrams.append(bigram)
        return bigrams

    def keep_valid_tokens(self, token_list):
        """ Filter repeating words in bigrams & keep valid unigram, bigram tokens 
        
        :param token_list: list of tokens (unigrams + bigrams)
        :return: valid list of unigrams + bigrams
        """
        unigrams = self.extract_unigrams(token_list)
        bigrams = self.extract_bigrams(token_list)
        valid_bigrams = []
        for bigram in bigrams:
            first_word = bigram.split("_")[0]
            second_word = bigram.split("_")[1]
            if first_word != second_word:
                valid_bigrams.append(bigram)
        valid_tokens = unigrams + valid_bigrams
        return valid_tokens

    def predict_language(self, text):
        """ Predict language code, confidence for a given text
        
        :param text: text to predict
        :return: dictionary
        """
        pred = self.fasttext_model.predict(text)    # ('__label__en',), array([0.13628894]))
        lang_code = pred[0][0][9:]                  # 'en'
        confidence = pred[1][0]                     #  0.13628894
        data = {
            "lang_code"  : lang_code,
            "confidence" : float("{0:.4f}".format(confidence))
        }
        return data

    def keep_lang_tokens(self, token_list):
        """ Keep valid language tokens 
        
        :param token_list: list of tokens
        :return: valid language tokens
        """
        lang_tokens = []
        for token in token_list:
            pred = self.predict_language(token)
            if isinstance(pred, dict):
                lang_code = pred.get('lang_code')
                confidence = pred.get('confidence')
                if lang_code == self.lang_code:
                    if confidence >= self.min_lang_conf:
                        lang_tokens.append(token)
    
        return lang_tokens

    def text_preprocessing(self, text):
        """ Generate list of clean & valid language tokens 
        
        :param text: text
        :return: list of tokens
        """
        lang_tokens = []
        text = self.keep_valid_doc(text)
        if text:
            text = self.filter_entities(text)
            text = self.lemmatize(text)
            text = self.remove_stopwords(text)
            clean_tokens = self.clean_tokens(text)
            lang_tokens = self.keep_lang_tokens(clean_tokens)
        return lang_tokens

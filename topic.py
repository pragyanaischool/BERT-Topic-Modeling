from sentence_transformers import SentenceTransformer
from shared.utils import load_from_pickle
from preprocessing import Preprocessing
from functools import lru_cache
from shared.utils import isFile
from utility import Utility
import pandas as pd 
import numpy as np
import hdbscan
import logging
import re
import os

abs_path = os.path.dirname(os.path.abspath("__file__")) + "/output"

class Topic(object):
    """ Class for predicting topic in new documents
    :param lang_code: language text
    :param method: topic method
    :param version: model version number
    :param k: number of topics
    :param clean_text: boolean flag for cleaning text
    :param num_words: num words per topic 
    :param min_words: min number of words for prediction
    :param min_conf_score: minimum confidence threshold
    :param pre_trained_name: pre-trained model name
    """
    def __init__(self, lang_code, method="BERT", version="1.1", k=134, clean_text=False, num_words=10,
                 pre_trained_name="distilbert-base-nli-mean-tokens", min_words=10, min_conf_score=0.10):
          
        self.lang_code = lang_code
        self.method = method
        self.version = version
        self.k = k                      
        self.clean_text = clean_text
        self.num_words = num_words     
        self.min_words = min_words
        self.min_conf_score = min_conf_score
        self.pre_trained_name = pre_trained_name
        subdir = "{}_k_{}_{}_{}".format(self.lang_code, str(self.k), self.method, self.version)

        self.valid_langs = ["en"]
        if lang_code in self.valid_langs:
            self.p = Preprocessing(lang_code=lang_code)
            if method == "BERT":
                self.model = SentenceTransformer(self.pre_trained_name)
                self.cluster_file = abs_path + "/models/" + subdir + "/cluster.pkl"
                self.umap_file = abs_path + "/models/" + subdir + "/umap.pkl"
                self.topics_file = abs_path + "/models/" + subdir + "/topics.pkl"

                if isFile(self.cluster_file) and isFile(self.umap_file) and isFile(self.topics_file):
                    self.cluster_model = self.load_cluster_model(self.cluster_file)
                    self.umap_model = self.load_umap_model(self.umap_file)
                    self.topics = self.load_topics(self.topics_file)
            
    @lru_cache(maxsize=128)
    def load_cluster_model(self, filepath):
        """ Load HDBSCAN cluster model from filepath
        :param filepath: 
        :return: cluster model
        """
        model = load_from_pickle(filepath)
        return model

    @lru_cache(maxsize=128)
    def load_topics(self, filepath):
        """ Load topics from filepath 
        
        :param filepath: 
        :return: dictionary (topic:top_n_words)
        """
        topics = load_from_pickle(filepath)
        return topics

    @lru_cache(maxsize=128)
    def load_umap_model(self, filepath):
        """ Load UMAP model from filepath 
        
        :param filepath: 
        :return: umap model
        """
        topics = load_from_pickle(filepath)
        return topics
    
    def text_preprocessing_pipeline(self, text):
        """ Text normalization pipeline 
        
        :param text: 
        :return: list of tokens
        """ 
        token_list = self.p.text_preprocessing(text)
        token_list = self.p.make_bigrams([token_list])
        return token_list

    def predict_topic(self, text):
        """ Predict topic for new documents 
        
        :param text: text to predict
        :return: python dictionary
        """
        try:
            prediction = dict()
    
            if text:
                if Utility.get_doc_length(text) > self.min_words:
                    if self.lang_code in self.valid_langs:
                        if self.clean_text:
                            text = Utility.clean_text(text)
                            
                        # generate token list from given text
                        token_lists = self.text_preprocessing_pipeline(text)
        
                        # convert list of tokens to sentence
                        sentences = [Utility.to_sentence(token_list) for token_list in token_lists]
                        
                        if self.method == "BERT":
                            # check if model and dictionary files exist
                            if isFile(self.cluster_file) and isFile(self.umap_file) and isFile(self.topics_file):
                                # generate sentence embeddings
                                embeddings = self.model.encode(sentences)
                                # reduce dimensionality of embeddings
                                umap_embeddings = self.umap_model.transform(embeddings)
                            
                                # predict cluster label
                                cluster_pred = hdbscan.approximate_predict(self.cluster_model, umap_embeddings)
                                # (array([1]), array([0.4937])) 
                               
                                # get cluster label, confidence
                                label = cluster_pred[0][0]          # 1
                                confidence = cluster_pred[1][0]     # 0.4937
                                
                                if float(confidence) <= self.min_conf_score:
                                    return "unknown topic, confidence below threshold"
                                
                                topic_terms = sorted(self.topics[label], key=lambda x: x[1], reverse=True)[:self.num_words]
                                topic_terms = [{ "term": term, "weight": float("{0:.4f}".format(weight))} for term, weight in topic_terms]
                        
                                prediction["topic_id"] = label
                                prediction['confidence'] = float("{0:.4f}".format(confidence))
                                prediction["topic_terms"] = topic_terms
                                prediction["message"] = "successful"
                            else:
                                return "model not found"
                        else:
                            return "method not exist"  
                    else:
                        return "language not supported"
                else:
                    return 'required at least {} words for prediction'.format(self.min_words)
            else:
                return "required textual content"
            return prediction
        except Exception:
            logging.error("exception occured", exc_info=True)  
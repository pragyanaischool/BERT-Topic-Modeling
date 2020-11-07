import gensim
from gensim.corpora import Dictionary
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from utility import Utility
import pandas as pd
import numpy as np
import logging
import hdbscan
import umap

class TM(object):
    """ Class for generating topic model
    """
    def __init__(self, method="BERT", random_state=42, pre_trained_name="bert-base-nli-mean-tokens",
                n_neighbors=15, n_components=5, min_dist=0.0, umap_metric='cosine', min_cluster_size=30, cluster_metric='euclidean', 
                cluster_selection_method='eom', prediction_data=True
                ):
        
        self.method = method
        self.random_state = random_state
        self.pre_trained_name = pre_trained_name
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.min_dist = min_dist
        self.umap_metric = umap_metric
        self.min_cluster_size = min_cluster_size
        self.cluster_metric = cluster_metric
        self.cluster_selection_method = cluster_selection_method
        self.prediction_data = prediction_data
    
    def get_label_docs_df(self, sentences, labels):
        """ Generate dataframe consisting cluster label and its associated documents.

        :param sentences: sentences
        :param labels: cluster labels
        """
        df = pd.DataFrame()
        df['doc'] = sentences
        df['label'] = labels
        df['doc_id'] = range(len(df))
        
        # group documents per cluster label into a single string
        label_docs_df = df.groupby(['label'], as_index=False).agg({'doc': ' '.join})
        return label_docs_df

    def c_tf_idf(self, docs, m):
        """ Calculate a class-based TF-IDF where m is total number of documents.

        :param docs: cluster documents
        :param m: number of documents in corpus
        """
        cv = CountVectorizer().fit(docs)
        t = cv.transform(docs).toarray()
        w = t.sum(axis=1)
        tf = np.divide(t.T, w)
        sum_t = t.sum(axis=0)
        idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
        tf_idf = np.multiply(tf, idf)
        return tf_idf, cv

    def fit(self, token_lists):
        """ Generate topic model, dictionary, corpus from token lists 
        
        :param token_lists: list of document tokens
        """
        try:
            
            # create Gensim dictionary & corpus for validation
            dictionary = Dictionary(token_lists)
            corpus = [dictionary.doc2bow(text) for text in token_lists]

            if self.method == "BERT":
                model = SentenceTransformer(self.pre_trained_name)
            
                # convert list of document tokens to list of sentences
                sentences = [Utility.to_sentence(token_list) for token_list in token_lists]
                
                # generate BERT sentence embeddings
                embeddings = model.encode(sentences, show_progress_bar=True)
                
                # reduce dimensionality of all embeddings using umap model
                umap_model = umap.UMAP(
                    n_neighbors=self.n_neighbors, n_components=self.n_components,
                    min_dist=self.min_dist, metric=self.umap_metric,random_state=self.random_state
                ).fit(embeddings)
                umap_embeddings = umap_model.transform(embeddings)

                # cluster documents using HDBSCAN
                cluster_model = hdbscan.HDBSCAN(
                    min_cluster_size=self.min_cluster_size, metric=self.cluster_metric,
                    cluster_selection_method=self.cluster_selection_method, 
                    prediction_data=self.prediction_data
                ).fit(umap_embeddings)
                
                # get cluster labels
                labels = cluster_model.labels_
                
                # generate label_docs dataframe
                label_docs_df = self.get_label_docs_df(sentences, labels)
               
                # calculate word importance per topic
                tf_idf, cv = self.c_tf_idf(label_docs_df.doc.values, m=len(sentences))

                self.k = len(np.unique(labels))
                self.labels = labels
                self.dictionary = dictionary
                self.corpus = corpus
                self.sentences = sentences
                self.token_lists = token_lists
                self.cluster_model = cluster_model
                self.umap_model = umap_model
                self.embeddings = embeddings
                self.umap_embeddings = umap_embeddings
                self.cv = cv
                self.tf_idf = tf_idf
                self.feature_names = cv.get_feature_names()

            else:
                raise Exception('method not exist')
        except Exception:
            logging.error("exception occured", exc_info=True)   

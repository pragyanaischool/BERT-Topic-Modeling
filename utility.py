import re

class Utility(object):
    @staticmethod
    def clean_text(text):
        """ Generate a clean textual content"""
        text = Utility.lowercase(text)
        text = Utility.remove_newlines(text)
        text = Utility.remove_tabs(text)
        text = Utility.remove_digits(text)
        text = Utility.replace_special_chars(text)
        text = Utility.remove_multiple_whitespaces(text)
        text = Utility.remove_trailing_whitespaces(text)
        return text
    
    @staticmethod
    def lowercase(text):
        """ Lowercase text """
        return text.lower()
    
    @staticmethod
    def remove_newlines(text):
        """ Remove newlines characters from text """
        return text.replace("\n", "")
    
    @staticmethod
    def remove_tabs(text):
        """ Remove tab characters from text """
        return text.replace("\t", "")
    
    @staticmethod
    def remove_digits(text):
        """ Remove digits from text """
        return ''.join([word for word in text if not word.isdigit()])

    @staticmethod
    def replace_special_chars(text):
        """ Remove special characters from text """
        return re.sub(r'\W+', ' ', text)

    @staticmethod
    def remove_multiple_whitespaces(text):
        """ Remove whitespaces from text """
        return re.sub(r'\s+', ' ', text, flags=re.I)

    @staticmethod
    def remove_trailing_whitespaces(text):
        """ Remove whitespaces from beggining and ending text """
        return text.strip()

    @staticmethod
    def get_doc_length(text):
        """ Determine number of words in a document."""
        doc_length = len(re.findall(r'\w+', text))
        return doc_length

    @staticmethod
    def to_sentence(token_list):
        """ Convert list of tokens to sentence
    
        :param token_list: tokens
        :return: text
        """ 
        sentence = " ".join(token_list)
        return sentence
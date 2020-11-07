from preprocessing import Preprocessing
from shared.utils import load_from_txt
from shared.utils import dump_to_pickle
from shared.utils import make_dirs
from tqdm import tqdm
import glob
import sys
import os

def process_chunk_files(input_path, output_path, min_doc_len=100):
    """ Process chunk files from command line 
    
    :param input_path: chunk input path files
    :param output_path: save preprocessed chunk files to output_path
    :param min_doc_len: minimum number of words required in a document for processing
    """
    
    # get input args lang_code, chunk_no
    lang_code = sys.argv[1]
    chunk_no = sys.argv[2]

    input_path = input_path + "/chunk_" + str(chunk_no)
    output_path = output_path + "/chunk_" + str(chunk_no)
    make_dirs(output_path)
    
    p = Preprocessing(lang_code=lang_code, min_doc_len=min_doc_len)
    token_lists = []
    
    # generate a list of txt file numbers in sequential order
    seq_file_nums = []
    for file in tqdm(glob.iglob(input_path + "/*.txt")):
        num_file = file.split("/")[-1]
        num_file = num_file.split(".")[0]
        seq_file_nums.append(int(num_file))
    seq_file_nums = sorted(seq_file_nums)
    
    # loop and process files in sequential order
    for num in tqdm(seq_file_nums):
        filepath = input_path + "/" + str(num) + ".txt"
        text = load_from_txt(filepath)
        tokens = p.text_preprocessing(text)
        token_lists.append(tokens)
        
    # filter empty lists 
    token_lists = [token_lst for token_lst in token_lists if token_lst != [] ]
    # generate bigrams
    token_lists = p.make_bigrams(token_lists)
    # filter repeating words in bigrams
    valid_tokens = [p.keep_valid_tokens(token_list) for token_list in tqdm(token_lists)]
    # dump to pickle files
    dump_to_pickle(valid_tokens, output_path + "/token_lists.pkl")

if __name__ == "__main__":
    input_path = os.path.dirname(os.path.abspath("__file__")) + "/chunks"
    output_path = os.path.dirname(os.path.abspath("__file__")) + "/chunks_prep"
    process_chunk_files(input_path, output_path)

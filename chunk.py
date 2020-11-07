from shared.utils import divide_chunks
from shared.utils import make_dirs
from shared.utils import dump_to_txt
from tqdm import tqdm
import os

def generate_chunk_files(docs, output_path, chunk_size=100):
    """ Generate chunk files from list of documents 
    
    :param output_path: path to save chunk txt files
    :param chunk_size: total number of documents in a chunk
    """
    
    # divide docs into chunks
    chunks = divide_chunks(docs, chunk_size)
   
    output_path += "/chunks"
    # create chunks directory
    make_dirs(output_path)

    # loop through chunks and save txt files
    chunk_no = 0
    for chunk in tqdm(chunks):
        chunk_no += 1
        chunk_dir = output_path + "/chunk_" + str(chunk_no)
        make_dirs(chunk_dir)
        num_file = 0
        for doc in chunk:
            num_file += 1
            dump_to_txt(doc, chunk_dir+ "/" + str(num_file) + ".txt")
